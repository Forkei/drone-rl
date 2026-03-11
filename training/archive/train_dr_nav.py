"""
DR Navigation Training.

Trains on DRNavAviary (wind + mass + drag randomization) but evaluates on
clean NavAviary (no perturbation). This separates training robustness from
eval quality — if DR is helping, clean eval reward should stay near v1 peak.

DR settings (mild start):
  - Mass:  ±10%
  - Drag:  ±15%
  - Wind:  max 0.003N per axis, smooth random walk (0.9 decay)

Key metric: clean eval_reward >= 8.0 → DR is helping generalization
            clean eval_reward <  5.0 → DR too aggressive, destabilizing

Usage:
    python train_dr_nav.py [--resume ./models_nav/best_model] [--timesteps 2000000]
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
from envs.dr_nav_aviary import DRNavAviary


# ── Same curriculum stages — but we start at the last stage (2.0m) on resume ─

CURRICULUM_STAGES = [
    {"target_range": 0.3, "success_threshold": 0.20, "eval_reward_threshold": 3.0,  "label": "stage1_0.3m"},
    {"target_range": 0.6, "success_threshold": 0.35, "eval_reward_threshold": 4.0,  "label": "stage2_0.6m"},
    {"target_range": 1.2, "success_threshold": 0.50, "eval_reward_threshold": 5.0,  "label": "stage3_1.2m"},
    {"target_range": 2.0, "success_threshold": 0.60, "eval_reward_threshold": 6.0,  "label": "stage4_2.0m"},
]


class EvalCurriculumCallback(EvalCallback):
    """EvalCallback that forwards eval mean reward to DRNavCallback."""

    def __init__(self, nav_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nav_cb = nav_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.nav_cb.notify_eval_reward(self.last_mean_reward)
        return result


class DRNavCallback(BaseCallback):
    """
    Tracks success_rate + last eval reward.
    Stage index starts at final stage (2.0m) because we're resuming.
    Logs DR health metric: 'dr/clean_eval_reward' should stay >= 8.0.
    """

    def __init__(self, train_envs, save_dir, start_stage=3,
                 window=100, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.train_envs    = train_envs
        self.save_dir      = save_dir
        self.window        = window
        self.check_freq    = check_freq
        self.stage_idx     = start_stage  # start at final stage for resume
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

            self.logger.record("dr/success_rate",    success_rate)
            self.logger.record("dr/mean_final_dist", mean_dist)
            self.logger.record("dr/curriculum_stage", self.stage_idx)
            self.logger.record("dr/clean_eval_reward", self.last_eval_reward)

            stage = self.current_stage
            s_thr = stage["success_threshold"]
            e_thr = stage["eval_reward_threshold"]

            # DR health warning
            dr_status = ""
            if self.last_eval_reward != -np.inf:
                if self.last_eval_reward >= 8.0:
                    dr_status = " [DR OK: clean eval good]"
                elif self.last_eval_reward >= 5.0:
                    dr_status = " [DR WARN: degraded]"
                else:
                    dr_status = " [DR CRITICAL: too aggressive!]"

            if self.verbose:
                print(f"\n[DR] {stage['label']} | "
                      f"train_success={success_rate:.2f}/{s_thr:.2f} | "
                      f"clean_eval={self.last_eval_reward:.2f}/{e_thr:.2f} | "
                      f"mean_dist={mean_dist:.3f}m{dr_status}")

        return True


# ── env factories ─────────────────────────────────────────────────────────────

def make_dr_env(target_range=2.0, gui=False,
                mass_dr=0.10, drag_dr=0.15, wind_max=0.003):
    """Training env: DR applied."""
    def _init():
        env = DRNavAviary(
            target_range=target_range,
            mass_dr=mass_dr,
            drag_dr=drag_dr,
            wind_max_force=wind_max,
            gui=gui,
        )
        return Monitor(env)
    return _init


def make_clean_eval_env(target_range=2.0):
    """Eval env: clean physics (no DR) — measures true policy quality."""
    return NavAviary(target_range=target_range)


# ── training ──────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    save_dir: str = "./models_dr_nav",
    log_dir:  str = "./logs_dr_nav",
    resume_path: str = "./models_nav/best_model",
    mass_dr: float = 0.10,
    drag_dr: float = 0.15,
    wind_max: float = 0.003,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    train_env = make_vec_env(
        make_dr_env(target_range=2.0, mass_dr=mass_dr,
                    drag_dr=drag_dr, wind_max=wind_max),
        n_envs=n_envs
    )
    # Eval on CLEAN physics — no DR perturbation during evaluation
    eval_env = make_clean_eval_env(target_range=2.0)

    if resume_path and os.path.isfile(resume_path + ".zip"):
        print(f"[INFO] Resuming from {resume_path}.zip")
        model = PPO.load(resume_path, env=train_env,
                         tensorboard_log=log_dir, verbose=1,
                         device="cpu")
    else:
        raise FileNotFoundError(
            f"Expected resume checkpoint at {resume_path}.zip. "
            "Run train_nav.py first to get the base navigation policy."
        )

    dr_cb = DRNavCallback(
        train_envs=train_env,
        save_dir=save_dir,
        start_stage=3,       # resume at stage4 = 2.0m
        window=100,
        check_freq=2048,
        verbose=1,
    )
    callbacks = CallbackList([
        dr_cb,
        EvalCurriculumCallback(
            nav_cb=dr_cb,
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
            name_prefix="ppo_dr_nav",
        ),
    ])

    print(f"\n[INFO] DR Nav training — {total_timesteps:,} steps, {n_envs} envs")
    print(f"[INFO] DR: mass±{mass_dr*100:.0f}%, drag±{drag_dr*100:.0f}%, "
          f"wind_max={wind_max:.4f}N smooth-walk")
    print(f"[INFO] Train env: DRNavAviary | Eval env: NavAviary (CLEAN — no perturbation)")
    print(f"[INFO] DR health: clean_eval >= 8.0 = good, < 5.0 = too aggressive\n")

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks, progress_bar=True,
                reset_num_timesteps=False)

    final = os.path.join(save_dir, "ppo_dr_nav_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs",    type=int, default=8)
    p.add_argument("--save-dir",  default="./models_dr_nav")
    p.add_argument("--log-dir",   default="./logs_dr_nav")
    p.add_argument("--resume",    default="./models_nav/best_model")
    p.add_argument("--mass-dr",   type=float, default=0.10)
    p.add_argument("--drag-dr",   type=float, default=0.15)
    p.add_argument("--wind-max",  type=float, default=0.003)
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_path=args.resume,
        mass_dr=args.mass_dr,
        drag_dr=args.drag_dr,
        wind_max=args.wind_max,
    )
