"""
Vision Navigation Training — SAC + CnnPolicy + VecFrameStack (n_stack=4).

Frame stacking concatenates n_stack consecutive frames along the channel dim:
  (3, H, W) x 4 frames → (12, H, W)   (handled transparently by NatureCNN)

With img_wh=(48,32):
  Single frame obs: (3, 32, 48)
  Stacked obs:      (12, 32, 48)  — SB3 VecFrameStack produces this automatically.

Why frame stacking?
  - Temporal motion information: velocity/direction of the sphere encoded in pixel diff.
  - NatureCNN processes (12, H, W) identically to (3, H, W) — just more input channels.
  - No architecture changes needed.

SAC config (same as train_vision_nav.py):
  learning_rate=1e-4
  buffer_size=100k
  batch_size=64
  learning_starts=5k
  train_freq=4

Replay buffer saved every 200k steps and at end — HARD REQUIREMENT for warm-starts.

Usage:
    python train_vision_nav_stacked.py
    python train_vision_nav_stacked.py --timesteps 2000000 --device cuda
"""

import argparse
import os
import time
import numpy as np
from collections import deque

import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs.vision_nav_aviary import VisionNavAviary


# ── callbacks ─────────────────────────────────────────────────────────────────

class VisionNavCallback(BaseCallback):
    """Logs success_rate, mean_dist, eval_reward. Reports fps every check_freq steps."""

    def __init__(self, window=200, check_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.window      = window
        self.check_freq  = check_freq
        self._successes  = deque(maxlen=window)
        self._ep_dists   = deque(maxlen=window)
        self.last_eval_reward = -np.inf
        self._t0         = None
        self._step0      = 0

    def _on_training_start(self):
        self._t0    = time.time()
        self._step0 = self.model.num_timesteps

    def notify_eval_reward(self, r):
        self.last_eval_reward = r

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._successes.append(float(info.get("success", 0.0)))
                if "dist_to_target" in info:
                    self._ep_dists.append(info["dist_to_target"])

        if self.n_calls % self.check_freq != 0:
            return True

        step = self.model.num_timesteps

        elapsed = time.time() - self._t0 if self._t0 else 1
        fps = (step - self._step0) / elapsed if elapsed > 0 else 0

        success_rate = float(np.mean(self._successes)) if self._successes else float("nan")
        mean_dist    = float(np.mean(self._ep_dists))  if self._ep_dists  else float("nan")

        try:
            ent_coef = float(th.exp(self.model.log_ent_coef.detach()))
        except Exception:
            ent_coef = float("nan")

        self.logger.record("vis/success_rate",    success_rate)
        self.logger.record("vis/mean_final_dist", mean_dist)
        self.logger.record("vis/eval_reward",     self.last_eval_reward)
        self.logger.record("vis/ent_coef",        ent_coef)
        self.logger.record("vis/fps",             fps)

        moving = "MOVING" if (not np.isnan(mean_dist) and mean_dist < 1.5) else "no movement yet"
        if self.verbose:
            print(f"[STACKED] step={step:,} | fps={fps:.1f} | eval={self.last_eval_reward:.2f} | "
                  f"suc={success_rate:.3f} | dist={mean_dist:.3f}m | ent={ent_coef:.4f} | {moving}")

        return True


class ReplayBufferSaveCallback(BaseCallback):
    """Saves replay buffer every save_freq steps."""

    def __init__(self, save_dir, save_freq=200_000, verbose=1):
        super().__init__(verbose)
        self.save_dir  = save_dir
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_dir, "replay_buffer")
            self.model.save_replay_buffer(path)
            if self.verbose:
                print(f"[STACKED] Replay buffer saved -> {path}")
        return True


class EvalVisionCallback(EvalCallback):
    def __init__(self, vis_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vis_cb = vis_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.vis_cb.notify_eval_reward(self.last_mean_reward)
        return result


# ── env factory ───────────────────────────────────────────────────────────────

def make_stacked_env(target_range: float, img_wh: tuple, n_stack: int = 4):
    """
    Returns a VecFrameStack-wrapped DummyVecEnv.

    Stack pipeline:
      VisionNavAviary (raw obs: (3, H, W))
      → Monitor
      → DummyVecEnv  (adds batch dim: (1, 3, H, W))
      → VecFrameStack (stacks channels: (1, 3*n_stack, H, W))
    """
    def _make():
        env = VisionNavAviary(target_range=target_range, img_wh=img_wh)
        return Monitor(env)

    venv = DummyVecEnv([_make])
    return VecFrameStack(venv, n_stack=n_stack)


# ── training ──────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int   = 2_000_000,
    save_dir:        str   = "./models_vision_stacked",
    log_dir:         str   = "./logs_vision_stacked",
    target_range:    float = 1.0,
    img_wh:          tuple = (48, 32),   # (width, height)
    n_stack:         int   = 4,
    learning_rate:   float = 1e-4,
    buffer_size:     int   = 100_000,
    batch_size:      int   = 64,
    learning_starts: int   = 5_000,
    train_freq:      int   = 4,
    device:          str   = "auto",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"

    img_w, img_h = img_wh
    stacked_channels = 3 * n_stack

    print(f"\n[INFO] Vision Nav Stacked Training — SAC CnnPolicy")
    print(f"[INFO] device={device}  img_wh={img_wh}  n_stack={n_stack}")
    print(f"[INFO] raw obs: (3,{img_h},{img_w})  stacked obs: ({stacked_channels},{img_h},{img_w})")
    print(f"[INFO] target_range={target_range}m  total_steps={total_timesteps:,}")
    print(f"[INFO] lr={learning_rate}  buffer={buffer_size:,}  batch={batch_size}")
    print(f"[INFO] learning_starts={learning_starts}  train_freq={train_freq}")
    print(f"[INFO] save_dir={save_dir}\n")

    # Training env
    train_env = make_stacked_env(target_range=target_range, img_wh=img_wh, n_stack=n_stack)

    # Eval env — separate instance, same stack config
    eval_env = make_stacked_env(target_range=target_range, img_wh=img_wh, n_stack=n_stack)

    model = SAC(
        "CnnPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=1,
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
    )

    vis_cb = VisionNavCallback(window=200, check_freq=10_000, verbose=1)

    callbacks = CallbackList([
        vis_cb,
        ReplayBufferSaveCallback(save_dir=save_dir, save_freq=200_000, verbose=1),
        EvalVisionCallback(
            vis_cb=vis_cb,
            eval_env=eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=20_000,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=100_000,
            save_path=save_dir,
            name_prefix="vis_stacked",
        ),
    ])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final = os.path.join(save_dir, "vis_stacked_final")
    model.save(final)
    model.save_replay_buffer(os.path.join(save_dir, "replay_buffer"))
    print(f"\n[INFO] Final model   -> {final}.zip")
    print(f"[INFO] Replay buffer -> {save_dir}/replay_buffer.pkl")

    train_env.close()
    eval_env.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Frame-stacked vision nav training (SAC CnnPolicy)")
    p.add_argument("--timesteps",   type=int,   default=2_000_000)
    p.add_argument("--save-dir",                default="./models_vision_stacked")
    p.add_argument("--log-dir",                 default="./logs_vision_stacked")
    p.add_argument("--device",                  default="auto")
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device,
    )
