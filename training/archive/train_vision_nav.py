"""
Vision Navigation Training — SAC + CnnPolicy from scratch.

Privileged teacher design:
  Obs:    (3, H, W) RGB pixels — no GPS, no position
  Reward: ground-truth PyBullet distance to colored sphere
  Policy: NatureCNN feature extractor + SAC actor/critic

Phase 1 (0–2M steps): target_range=1.0m
  If success_rate > 15% at 1M steps, consider expanding to 2.0m manually.

SAC config tuned for CNN:
  learning_rate=1e-4  (lower than MLP 3e-4 — stable CNN gradients)
  buffer_size=100k    (images are memory-heavy vs scalars)
  batch_size=64       (smaller batch for CNN)
  learning_starts=5k  (fill buffer before first update)
  train_freq=4        (update every 4 env steps, not every step)

Replay buffer saved every 200k steps and at end — HARD REQUIREMENT.
Load replay buffer on any future fine-tune to prevent catastrophic forgetting.

Usage:
    python train_vision_nav.py
    python train_vision_nav.py --timesteps 2000000 --img-w 48 --img-h 32  # faster
"""

import argparse
import os
import time
import numpy as np
from collections import deque

import torch as th
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.vision_nav_aviary import VisionNavAviary


class SmallCNN(BaseFeaturesExtractor):
    """Compact CNN for images smaller than 64x48 (NatureCNN requires >= 64x48).

    For (3, 32, 48) input:
      Conv(k=4,s=2) → (32, 15, 23)
      Conv(k=3,s=2) → (64,  7, 11)
      Conv(k=3,s=1) → (64,  5,  9)
      Flatten → 2880 → Linear(256)
    """
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_ch = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32,   64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64,   64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flat = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(obs))


class VisionNavCallback(BaseCallback):
    """Logs success_rate, mean_dist, eval_reward. Reports fps every 10k steps."""

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

        # fps
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
            print(f"[VIS] step={step:,} | fps={fps:.1f} | eval={self.last_eval_reward:.2f} | "
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
                print(f"[VIS] Replay buffer saved -> {path}")
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


def train(
    total_timesteps: int   = 2_000_000,
    save_dir:        str   = "./models_vision_nav",
    log_dir:         str   = "./logs_vision_nav",
    target_range:    float = 1.0,
    img_w:           int   = 64,
    img_h:           int   = 48,
    learning_rate:   float = 1e-4,
    buffer_size:     int   = 100_000,
    batch_size:      int   = 64,
    learning_starts: int   = 5_000,
    train_freq:      int   = 4,
    device:          str   = "auto",   # "auto"=cuda if available, else cpu
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    img_wh = (img_w, img_h)

    train_env = Monitor(VisionNavAviary(target_range=target_range, img_wh=img_wh))
    eval_env  = Monitor(VisionNavAviary(target_range=target_range, img_wh=img_wh))

    import torch as _torch
    if device == "auto":
        device = "cuda" if _torch.cuda.is_available() else "cpu"

    print(f"\n[INFO] Vision Nav Training — SAC CnnPolicy from scratch")
    print(f"[INFO] device={device}  img_wh={img_wh}  obs_shape=(3,{img_h},{img_w})")
    print(f"[INFO] target_range={target_range}m  total_steps={total_timesteps:,}")
    print(f"[INFO] lr={learning_rate}  buffer={buffer_size:,}  batch={batch_size}")
    print(f"[INFO] learning_starts={learning_starts}  train_freq={train_freq}")
    print(f"[INFO] save_dir={save_dir}\n")

    # NatureCNN needs H>=36, W>=36 to survive 3 conv layers.
    # Use SmallCNN for images smaller than 64x48.
    need_small_cnn = (img_h < 36 or img_w < 36)
    policy_kwargs = (
        dict(features_extractor_class=SmallCNN,
             features_extractor_kwargs=dict(features_dim=256))
        if need_small_cnn else {}
    )
    if need_small_cnn:
        print(f"[INFO] Using SmallCNN extractor (NatureCNN too large for {img_h}x{img_w})")

    model = SAC(
        "CnnPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=1,
        policy_kwargs=policy_kwargs or None,
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
            name_prefix="vis_nav",
        ),
    ])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final = os.path.join(save_dir, "vis_nav_final")
    model.save(final)
    model.save_replay_buffer(os.path.join(save_dir, "replay_buffer"))
    print(f"\n[INFO] Final model   -> {final}.zip")
    print(f"[INFO] Replay buffer -> {save_dir}/replay_buffer.pkl")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps",        type=int,   default=2_000_000)
    p.add_argument("--save-dir",                     default="./models_vision_nav")
    p.add_argument("--log-dir",                      default="./logs_vision_nav")
    p.add_argument("--target-range",     type=float, default=1.0)
    p.add_argument("--img-w",            type=int,   default=64)
    p.add_argument("--img-h",            type=int,   default=48)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--buffer-size",      type=int,   default=100_000)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--learning-starts",  type=int,   default=5_000)
    p.add_argument("--train-freq",       type=int,   default=4)
    p.add_argument("--device",                       default="auto")
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        target_range=args.target_range,
        img_w=args.img_w,
        img_h=args.img_h,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        device=args.device,
    )
