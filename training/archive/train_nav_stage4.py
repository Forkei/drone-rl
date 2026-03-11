"""
Stage 4 resume: 2.0m range from stage3 checkpoint.

Fixes from Option D:
1. eval_env fixed at target_range=2.0m (not 0.3m)
2. ent_coef=0.001 from step 0 (stage4 entropy level, not 0.01)
3. lr=3e-5 (10x reduction to tame clip_fraction at 2.0m range)

Resume from: models_nav_entropy/nav_ent_stage3_1.2m.zip
Policy std at resume: ~0.34 (already annealed from Option D)

Diagnostic thresholds:
  clip_fraction < 0.4 at 100k -> lr=3e-5 is sufficient, keep training
  clip_fraction > 0.5 at 100k -> 2.0m range needs different approach (SAC?)
  std < 0.2 by 2M          -> precision settling should work
  std > 0.4 at 1M          -> may need more entropy annealing

Usage:
    python train_nav_stage4.py [--timesteps 2000000]
"""

import argparse
import os
import numpy as np
from collections import deque

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from envs.nav_aviary import NavAviary


CLIP_WARN_100K = 0.4   # above this at 100k -> diagnostic flag
STD_TARGET     = 0.2   # target std by 2M steps
SUCCESS_TARGET = 0.60  # stage4 threshold


# -- callback ------------------------------------------------------------------

class Stage4Callback(BaseCallback):
    """
    Monitors policy_std, clip_fraction, success_rate for stage4 (2.0m range).

    TensorBoard metrics:
        s4/clip_fraction, s4/clip_fraction_25k
        s4/policy_std
        s4/eval_reward
        s4/success_rate, s4/mean_final_dist
    """

    def __init__(self, save_dir, window=100, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.save_dir   = save_dir
        self.window     = window
        self.check_freq = check_freq
        self._successes = deque(maxlen=window)
        self._ep_dists  = deque(maxlen=window)
        self._clip_hist = deque(maxlen=15)   # ~25k steps rolling
        self.last_eval_reward   = -np.inf
        self._diag_reported_100k = False

    def notify_eval_reward(self, r):
        self.last_eval_reward = r

    def _policy_std(self):
        return float(th.exp(self.model.policy.log_std).mean().detach())

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._successes.append(float(info.get("success", 0.0)))
                if "dist_to_target" in info:
                    self._ep_dists.append(info["dist_to_target"])

        clip_frac = self.model.logger.name_to_value.get("train/clip_fraction", float("nan"))
        if not np.isnan(clip_frac):
            self._clip_hist.append(clip_frac)

        # -- diagnostic at 100k --
        step = self.model.num_timesteps
        if not self._diag_reported_100k and step >= 100_000:
            self._diag_reported_100k = True
            rolling = float(np.nanmean(self._clip_hist)) if self._clip_hist else float("nan")
            std = self._policy_std()
            print(f"\n{'='*60}")
            print(f"[S4] DIAGNOSTIC at step {step:,}")
            print(f"  clip_fraction (25k rolling) : {rolling:.4f}")
            print(f"  policy_std                  : {std:.4f}")
            print(f"  eval_reward                 : {self.last_eval_reward:.3f}")
            if not np.isnan(rolling):
                if rolling < CLIP_WARN_100K:
                    print(f"  VERDICT: clip={rolling:.3f} < {CLIP_WARN_100K} -> lr=3e-5 OK, continue")
                else:
                    print(f"  VERDICT: clip={rolling:.3f} >= {CLIP_WARN_100K} -> CAUTION: 2.0m may need rethink")
            print(f"{'='*60}\n")

        if self.n_calls % self.check_freq != 0 or len(self._successes) < 10:
            return True

        # -- regular metrics --
        success_rate = float(np.mean(self._successes))
        mean_dist    = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")
        rolling_clip = float(np.nanmean(self._clip_hist)) if self._clip_hist else float("nan")
        std          = self._policy_std()

        self.logger.record("s4/clip_fraction",    clip_frac)
        self.logger.record("s4/clip_fraction_25k", rolling_clip)
        self.logger.record("s4/policy_std",        std)
        self.logger.record("s4/eval_reward",       self.last_eval_reward)
        self.logger.record("s4/success_rate",      success_rate)
        self.logger.record("s4/mean_final_dist",   mean_dist)

        clip_warn = ""
        if not np.isnan(rolling_clip) and rolling_clip > 0.4:
            clip_warn = f" [CLIP {rolling_clip:.2f}]"
        std_warn = ""
        if step >= 1_000_000 and std > 0.4:
            std_warn = f" [STD HIGH {std:.3f}]"

        if self.verbose:
            print(f"[S4] step={step:,} | eval={self.last_eval_reward:.2f} | "
                  f"suc={success_rate:.2f}/{SUCCESS_TARGET:.2f} | "
                  f"dist={mean_dist:.3f}m | "
                  f"std={std:.3f} | clip={clip_frac:.3f}"
                  f"{clip_warn}{std_warn}")

        return True


class EvalS4Callback(EvalCallback):
    def __init__(self, s4_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s4_cb = s4_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.s4_cb.notify_eval_reward(self.last_mean_reward)
        return result


# -- training ------------------------------------------------------------------

def train(
    total_timesteps: int = 2_000_000,
    n_envs:          int = 8,
    save_dir:        str = "./models_nav_stage4",
    log_dir:         str = "./logs_nav_stage4",
    resume_path:     str = "./models_nav_entropy/nav_ent_stage3_1.2m",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    target_range = 2.0

    train_env = make_vec_env(
        lambda: Monitor(NavAviary(target_range=target_range)), n_envs=n_envs
    )
    eval_env = Monitor(NavAviary(target_range=target_range))  # fixed at 2.0m -- bug fix

    if not os.path.isfile(resume_path + ".zip"):
        raise FileNotFoundError(f"Not found: {resume_path}.zip")

    print(f"[INFO] Loading {resume_path}.zip")
    model = PPO.load(
        resume_path,
        env=train_env,
        tensorboard_log=log_dir,
        verbose=1,
        device="cpu",
        custom_objects={
            "learning_rate": 3e-5,
            "lr_schedule":   lambda _: 3e-5,
            "ent_coef":      0.001,
            "clip_range":    0.2,
            "clip_range_vf": 0.2,
        },
    )

    # Belt-and-suspenders: override lr_schedule directly to guarantee constant lr
    model.learning_rate = 3e-5
    model.lr_schedule   = get_schedule_fn(3e-5)
    model.ent_coef      = 0.001

    std0 = float(th.exp(model.policy.log_std).mean().detach())
    print(f"[INFO] Loaded. std={std0:.4f}  lr=3e-5  ent_coef=0.001  clip_range=0.2")
    print(f"[INFO] eval_env: target_range=2.0m (fixed -- eval bug corrected)")
    print(f"[INFO] {total_timesteps:,} steps on 2.0m range")
    print(f"[INFO] Diagnostic checkpoint at 100k: clip<{CLIP_WARN_100K} -> continue, else rethink\n")

    s4_cb = Stage4Callback(
        save_dir=save_dir,
        window=100,
        check_freq=2048,
        verbose=1,
    )
    callbacks = CallbackList([
        s4_cb,
        EvalS4Callback(
            s4_cb=s4_cb,
            eval_env=eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=max(20_000 // n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(100_000 // n_envs, 1),
            save_path=save_dir,
            name_prefix="ppo_s4",
        ),
    ])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final = os.path.join(save_dir, "ppo_stage4_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs",    type=int, default=8)
    p.add_argument("--save-dir",  default="./models_nav_stage4")
    p.add_argument("--log-dir",   default="./logs_nav_stage4")
    p.add_argument("--resume",    default="./models_nav_entropy/nav_ent_stage3_1.2m")
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_path=args.resume,
    )
