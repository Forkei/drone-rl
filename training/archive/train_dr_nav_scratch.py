"""
DR Navigation Training — From Scratch.

Trains DRNavAviary from a fresh policy (full curriculum 0.3m→0.6m→1.2m→2.0m).
Evaluates on clean NavAviary (no perturbation) to measure true policy quality.

Key differences from train_dr_nav.py:
  - No resume — fresh policy init avoids value function mismatch
  - clip_range=0.15 (was 0.1) — allows slightly larger updates, better for
    varied DR state distribution; also makes clip_fraction a meaningful signal
  - clip_fraction logged as first-class TensorBoard metric
  - DR health alert: clip_fraction > 0.3 sustained = early warning for future resumes

DR settings (mild):
  - Mass:  ±10%
  - Drag:  ±15%
  - Wind:  max 0.003N, smooth random walk (0.9 decay)

Usage:
    python train_dr_nav_scratch.py [--timesteps 2000000]
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


CURRICULUM_STAGES = [
    {"target_range": 0.3, "success_threshold": 0.20, "eval_reward_threshold": 3.0,  "label": "stage1_0.3m"},
    {"target_range": 0.6, "success_threshold": 0.35, "eval_reward_threshold": 4.0,  "label": "stage2_0.6m"},
    {"target_range": 1.2, "success_threshold": 0.50, "eval_reward_threshold": 5.0,  "label": "stage3_1.2m"},
    {"target_range": 2.0, "success_threshold": 0.60, "eval_reward_threshold": 6.0,  "label": "stage4_2.0m"},
]


class EvalCurriculumCallback(EvalCallback):
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
    Curriculum + DR health monitoring.

    First-class metrics logged to TensorBoard:
      dr/clip_fraction     — copied from PPO trainer each check
      dr/clean_eval_reward — eval on clean physics
      dr/success_rate      — training rollout success rate
      dr/curriculum_stage  — current stage index

    Early warning: clip_fraction > 0.3 sustained = value function mismatch.
    For reference: normal PPO clip_fraction = 0.05-0.15.
    """

    def __init__(self, train_envs, save_dir,
                 window=100, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.train_envs       = train_envs
        self.save_dir         = save_dir
        self.window           = window
        self.check_freq       = check_freq
        self.stage_idx        = 0
        self.last_eval_reward = -np.inf
        self._successes       = deque(maxlen=window)
        self._ep_dists        = deque(maxlen=window)

        # Clip fraction tracking for early warning
        self._clip_fraction_history = deque(maxlen=25)  # ~50k steps at check_freq=2048

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

            # Grab clip_fraction from PPO's logger (written after each update)
            clip_frac = self.model.logger.name_to_value.get("train/clip_fraction", float("nan"))
            self._clip_fraction_history.append(clip_frac)
            sustained_clip = float(np.nanmean(self._clip_fraction_history))

            self.logger.record("dr/success_rate",       success_rate)
            self.logger.record("dr/mean_final_dist",    mean_dist)
            self.logger.record("dr/curriculum_stage",   self.stage_idx)
            self.logger.record("dr/clean_eval_reward",  self.last_eval_reward)
            self.logger.record("dr/clip_fraction",      clip_frac)
            self.logger.record("dr/clip_fraction_50k",  sustained_clip)

            stage  = self.current_stage
            s_thr  = stage["success_threshold"]
            e_thr  = stage["eval_reward_threshold"]

            # DR health
            dr_health = ""
            if self.last_eval_reward != -np.inf:
                if self.last_eval_reward >= 8.0:
                    dr_health = " [DR OK]"
                elif self.last_eval_reward >= 5.0:
                    dr_health = " [DR WARN: degraded]"
                else:
                    dr_health = " [DR CRITICAL]"

            # Clip fraction early warning
            clip_warn = ""
            if not np.isnan(sustained_clip) and sustained_clip > 0.3:
                clip_warn = f" [CLIP WARN: {sustained_clip:.2f}>0.3 — value mismatch risk]"

            if self.verbose:
                print(f"\n[DR] {stage['label']} | "
                      f"success={success_rate:.2f}/{s_thr:.2f} | "
                      f"eval={self.last_eval_reward:.2f}/{e_thr:.2f} | "
                      f"dist={mean_dist:.3f}m | "
                      f"clip={clip_frac:.3f}{dr_health}{clip_warn}")

            if self.stage_idx < len(CURRICULUM_STAGES) - 1:
                if success_rate >= s_thr or self.last_eval_reward >= e_thr:
                    trigger = "train_success" if success_rate >= s_thr else "eval_reward"
                    self._advance_stage(trigger)

        return True

    def _advance_stage(self, trigger=""):
        label = self.current_stage["label"]
        ckpt  = os.path.join(self.save_dir, f"dr_nav_{label}.zip")
        self.model.save(ckpt)
        print(f"\n[Curriculum] {label} -> COMPLETE (trigger: {trigger}) | ckpt -> {ckpt}")

        self.stage_idx += 1
        new_range = self.current_stage["target_range"]
        self.train_envs.env_method("_set_target_range", new_range)
        print(f"[Curriculum] -> {self.current_stage['label']}  (radius {new_range}m)\n")

        self._successes.clear()
        self._ep_dists.clear()
        self.last_eval_reward = -np.inf


def make_dr_env(target_range=0.3, gui=False,
                mass_dr=0.10, drag_dr=0.15, wind_max=0.003):
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


def make_clean_eval_env(target_range=0.3):
    """Eval on clean physics — no perturbation."""
    return Monitor(NavAviary(target_range=target_range))


def train(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    save_dir: str = "./models_dr_nav_scratch",
    log_dir:  str = "./logs_dr_nav_scratch",
    mass_dr:  float = 0.10,
    drag_dr:  float = 0.15,
    wind_max: float = 0.003,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    NavAviary._set_target_range = lambda self, r: setattr(self, "target_range", r)
    DRNavAviary._set_target_range = lambda self, r: setattr(self, "target_range", r)

    initial_range = CURRICULUM_STAGES[0]["target_range"]
    train_env = make_vec_env(
        make_dr_env(target_range=initial_range,
                    mass_dr=mass_dr, drag_dr=drag_dr, wind_max=wind_max),
        n_envs=n_envs
    )
    eval_env = make_clean_eval_env(target_range=initial_range)

    model = PPO(
        "MlpPolicy", train_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,        # wider than v1's 0.1 — DR state distribution is more varied
        clip_range_vf=0.15,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cpu",
        seed=42,
    )

    nav_cb = DRNavCallback(
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
            name_prefix="ppo_dr_nav",
        ),
    ])

    print(f"\n[INFO] DR Nav scratch training — {total_timesteps:,} steps, {n_envs} envs")
    print(f"[INFO] DR: mass±{mass_dr*100:.0f}%, drag±{drag_dr*100:.0f}%, wind_max={wind_max:.4f}N")
    print(f"[INFO] clip_range=0.15 | Train: DRNavAviary | Eval: NavAviary (CLEAN)")
    print(f"[INFO] Clip fraction early warning: sustained >0.3 = value mismatch risk")
    print(f"[INFO] Curriculum: {[s['label'] for s in CURRICULUM_STAGES]}\n")

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks, progress_bar=True)

    final = os.path.join(save_dir, "ppo_dr_nav_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs",    type=int, default=8)
    p.add_argument("--save-dir",  default="./models_dr_nav_scratch")
    p.add_argument("--log-dir",   default="./logs_dr_nav_scratch")
    p.add_argument("--mass-dr",   type=float, default=0.10)
    p.add_argument("--drag-dr",   type=float, default=0.15)
    p.add_argument("--wind-max",  type=float, default=0.003)
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        mass_dr=args.mass_dr,
        drag_dr=args.drag_dr,
        wind_max=args.wind_max,
    )
