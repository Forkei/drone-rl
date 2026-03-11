"""
Nav v2 — Settling Fix Training.

Changes from v1:
  - warm_zone_bonus: 0.5 -> 1.5
  - warm_zone_thresh: 0.15m -> 0.12m
  - velocity penalty: -0.5 * |vel| when dist < 0.3m

Resumed from best_model (navigation skill preserved, just fixing settling).

Usage:
    python train_nav_v2.py [--resume ./models_nav/best_model] [--timesteps 500000]
"""

import argparse
import os
import numpy as np
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from envs.nav_aviary import NavAviary


# ── Same curriculum stages as v1 ──────────────────────────────────────────────

CURRICULUM_STAGES = [
    {"target_range": 0.3, "success_threshold": 0.20, "eval_reward_threshold": 3.0,  "label": "stage1_0.3m"},
    {"target_range": 0.6, "success_threshold": 0.35, "eval_reward_threshold": 4.0,  "label": "stage2_0.6m"},
    {"target_range": 1.2, "success_threshold": 0.50, "eval_reward_threshold": 5.0,  "label": "stage3_1.2m"},
    {"target_range": 2.0, "success_threshold": 0.60, "eval_reward_threshold": 6.0,  "label": "stage4_2.0m"},
]


class EvalCurriculumCallback(EvalCallback):
    """EvalCallback that forwards eval mean reward to NavCallback for curriculum gating."""

    def __init__(self, nav_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nav_cb = nav_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.nav_cb.notify_eval_reward(self.last_mean_reward)
        return result


class NavCallback(BaseCallback):
    def __init__(self, train_envs, save_dir, window=100, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.train_envs    = train_envs
        self.save_dir      = save_dir
        self.window        = window
        self.check_freq    = check_freq
        self.stage_idx     = len(CURRICULUM_STAGES) - 1  # start at final stage (2.0m)
        self.last_eval_reward = -np.inf
        self._successes = deque(maxlen=window)
        self._ep_dists  = deque(maxlen=window)

    @property
    def current_stage(self):
        return CURRICULUM_STAGES[self.stage_idx]

    def notify_eval_reward(self, mean_reward):
        self.last_eval_reward = mean_reward

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._successes.append(float(info.get("success", 0.0)))
                if "dist_to_target" in info:
                    self._ep_dists.append(info["dist_to_target"])

        if self.n_calls % self.check_freq == 0 and len(self._successes) >= 10:
            success_rate = float(np.mean(self._successes))
            mean_dist    = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")
            self.logger.record("nav/success_rate",    success_rate)
            self.logger.record("nav/mean_final_dist", mean_dist)
            self.logger.record("nav/curriculum_stage", self.stage_idx)
            self.logger.record("nav/last_eval_reward", self.last_eval_reward)
            if self.verbose:
                stage = self.current_stage
                print(f"\n[Nav] {stage['label']} | "
                      f"train_success={success_rate:.2f} | "
                      f"eval_reward={self.last_eval_reward:.2f} | "
                      f"mean_dist={mean_dist:.3f}m")
        return True


def make_nav_env(target_range=2.0, gui=False):
    def _init():
        env = NavAviary(
            target_range=target_range,
            warm_zone_bonus=1.5,    # v2: increased from 0.5
            warm_zone_thresh=0.12,  # v2: tightened from 0.15
            gui=gui,
        )
        return Monitor(env)
    return _init


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 8,
    save_dir: str = "./models_nav_v2",
    log_dir:  str = "./logs_nav_v2",
    resume_path: str = "./models_nav/best_model",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    train_env = make_vec_env(make_nav_env(target_range=2.0), n_envs=n_envs)
    eval_env  = make_nav_env(target_range=2.0)()

    if resume_path and os.path.isfile(resume_path + ".zip"):
        print(f"[INFO] Resuming from {resume_path}.zip")
        model = PPO.load(resume_path, env=train_env,
                         tensorboard_log=log_dir, verbose=1)
    else:
        raise FileNotFoundError(
            f"Expected resume checkpoint at {resume_path}.zip. "
            "Pass --resume or train from scratch with train_nav.py first."
        )

    nav_cb = NavCallback(
        train_envs=train_env,
        save_dir=save_dir,
        window=100,
        check_freq=2048,
        verbose=1,
    )
    callbacks = CallbackList([
        nav_cb,
        EvalCurriculumCallback(
            nav_cb=nav_cb,
            eval_env=eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=max(20_000 // n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=save_dir,
            name_prefix="ppo_nav_v2",
        ),
    ])

    print(f"\n[INFO] Nav v2 training — {total_timesteps:,} steps, {n_envs} envs")
    print(f"[INFO] Reward changes: warm_zone_bonus=1.5 (was 0.5), "
          f"warm_zone_thresh=0.12m (was 0.15m), vel_penalty when dist<0.3m")
    print(f"[INFO] Starting at stage4 2.0m (resuming from best_model)\n")

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks, progress_bar=True)

    final = os.path.join(save_dir, "ppo_nav_v2_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs",    type=int, default=8)
    p.add_argument("--save-dir",  default="./models_nav_v2")
    p.add_argument("--log-dir",   default="./logs_nav_v2")
    p.add_argument("--resume",    default="./models_nav/best_model",
                   help="Path to nav model checkpoint to resume (without .zip)")
    args = p.parse_args()

    train(args.timesteps, n_envs=args.n_envs,
          save_dir=args.save_dir, log_dir=args.log_dir,
          resume_path=args.resume)
