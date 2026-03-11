"""
Option D: Entropy-annealed curriculum training from scratch.

Root cause of all fine-tuning failures: v1 best_model has std=1.15 (high entropy),
ent_coef=0.01 prevents entropy reduction, clip_range=0.1 x n_epochs=10 creates
a feedback loop that keeps the policy diffuse. No fine-tuning can escape this.

Solution: train from scratch with explicit entropy decay:
    0-500k:   ent_coef=0.05  (high entropy for exploration)
    500k-1M:  ent_coef=0.01  (moderate)
    1M-2M:    ent_coef=0.001 (force convergence)
    clip_range=0.2            (wider clip, allows meaningful updates early)

Key metric to watch: policy_std (exp(log_std).mean())
    Target: drops from ~1.0 early -> ~0.1-0.3 by 2M steps
    Warning: if std > 0.5 at 1M steps, increase annealing rate

Full curriculum: 0.3m -> 0.6m -> 1.2m -> 2.0m (same as v1)
Clean NavAviary throughout (no DR).

Usage:
    python train_nav_entropy.py [--timesteps 2000000]
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

from envs.nav_aviary import NavAviary


CURRICULUM_STAGES = [
    {"target_range": 0.3, "success_threshold": 0.20, "eval_reward_threshold": 3.0,  "label": "stage1_0.3m"},
    {"target_range": 0.6, "success_threshold": 0.35, "eval_reward_threshold": 4.0,  "label": "stage2_0.6m"},
    {"target_range": 1.2, "success_threshold": 0.50, "eval_reward_threshold": 5.0,  "label": "stage3_1.2m"},
    {"target_range": 2.0, "success_threshold": 0.60, "eval_reward_threshold": 6.0,  "label": "stage4_2.0m"},
]

# Entropy annealing schedule: (progress_remaining_threshold, ent_coef)
# progress_remaining: 1.0 at start, 0.0 at end
ENT_SCHEDULE = [
    (1.00, 0.05),   # 0-500k   (progress 1.0-0.75 for 2M total)
    (0.75, 0.01),   # 500k-1M  (progress 0.75-0.5)
    (0.50, 0.001),  # 1M-2M    (progress 0.5-0.0)
]

STD_WARN_THRESHOLD = 0.5   # if std > this at 1M steps, warn


# ── callbacks ─────────────────────────────────────────────────────────────────

class EvalNavCallback(EvalCallback):
    def __init__(self, nav_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nav_cb = nav_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.nav_cb.notify_eval_reward(self.last_mean_reward)
        return result


class EntropyNavCallback(BaseCallback):
    """
    Curriculum + entropy annealing + policy std monitoring.

    Entropy schedule is applied based on progress_remaining from model.
    Curriculum advances on success_rate OR eval_reward threshold.
    Policy std (exp(log_std)) logged every check_freq steps.

    TensorBoard metrics:
        nav/clip_fraction, nav/clip_fraction_25k
        nav/policy_std            -- exp(log_std).mean()
        nav/ent_coef              -- current entropy coefficient
        nav/eval_reward           -- clean eval
        nav/curriculum_stage
        nav/success_rate, nav/mean_final_dist
    """

    def __init__(self, train_envs, save_dir, total_timesteps,
                 window=100, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.train_envs       = train_envs
        self.save_dir         = save_dir
        self.total_timesteps  = total_timesteps
        self.window           = window
        self.check_freq       = check_freq
        self.stage_idx        = 0
        self.last_eval_reward = -np.inf
        self._successes       = deque(maxlen=window)
        self._ep_dists        = deque(maxlen=window)
        self._clip_hist       = deque(maxlen=15)   # ~25k steps rolling
        self._warned_std      = False
        self._last_ent_coef   = ENT_SCHEDULE[0][1]

    @property
    def current_stage(self):
        return CURRICULUM_STAGES[self.stage_idx]

    def notify_eval_reward(self, r):
        self.last_eval_reward = r

    def _policy_std(self):
        return float(th.exp(self.model.policy.log_std).mean().detach())

    def _update_ent_coef(self):
        """Apply entropy annealing based on training progress."""
        progress = 1.0 - self.model.num_timesteps / self.total_timesteps
        new_coef = ENT_SCHEDULE[-1][1]  # default to final value
        for threshold, coef in ENT_SCHEDULE:
            if progress >= threshold:
                new_coef = coef
                break
        if new_coef != self._last_ent_coef:
            old = self._last_ent_coef
            self.model.ent_coef = new_coef
            self._last_ent_coef = new_coef
            step = self.model.num_timesteps
            print(f"\n[EntropyAnneal] step={step:,} | ent_coef {old} -> {new_coef}\n")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._successes.append(float(info.get("success", 0.0)))
                if "dist_to_target" in info:
                    self._ep_dists.append(info["dist_to_target"])

        self._update_ent_coef()

        clip_frac = self.model.logger.name_to_value.get("train/clip_fraction", float("nan"))
        if not np.isnan(clip_frac):
            self._clip_hist.append(clip_frac)

        if self.n_calls % self.check_freq != 0 or len(self._successes) < 10:
            return True

        # ── metrics ──────────────────────────────────────────────────────
        success_rate  = float(np.mean(self._successes))
        mean_dist     = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")
        rolling_clip  = float(np.nanmean(self._clip_hist)) if self._clip_hist else float("nan")
        std           = self._policy_std()
        step          = self.model.num_timesteps

        self.logger.record("nav/clip_fraction",    clip_frac)
        self.logger.record("nav/clip_fraction_25k", rolling_clip)
        self.logger.record("nav/policy_std",        std)
        self.logger.record("nav/ent_coef",          self._last_ent_coef)
        self.logger.record("nav/eval_reward",       self.last_eval_reward)
        self.logger.record("nav/curriculum_stage",  self.stage_idx)
        self.logger.record("nav/success_rate",      success_rate)
        self.logger.record("nav/mean_final_dist",   mean_dist)

        stage = self.current_stage
        s_thr = stage["success_threshold"]
        e_thr = stage["eval_reward_threshold"]

        # std warning at 1M steps
        if step >= 1_000_000 and not self._warned_std and std > STD_WARN_THRESHOLD:
            print(f"\n[STD WARN] At 1M steps, policy_std={std:.3f} > {STD_WARN_THRESHOLD}.")
            print(f"  Entropy annealing may be too slow. Consider increasing rate.\n")
            self._warned_std = True

        clip_warn = ""
        if not np.isnan(rolling_clip) and rolling_clip > 0.3:
            clip_warn = f" [CLIP {rolling_clip:.2f}]"
        std_note = f" std={std:.3f}"

        if self.verbose:
            print(f"[D] {stage['label']} | step={step:,} | "
                  f"eval={self.last_eval_reward:.2f}/{e_thr:.2f} | "
                  f"suc={success_rate:.2f}/{s_thr:.2f} | "
                  f"dist={mean_dist:.3f}m | "
                  f"ent={self._last_ent_coef:.4f} |"
                  f"{std_note}{clip_warn}")

        # ── curriculum advance ────────────────────────────────────────────
        if self.stage_idx < len(CURRICULUM_STAGES) - 1:
            if success_rate >= s_thr or self.last_eval_reward >= e_thr:
                trigger = "train_success" if success_rate >= s_thr else "eval_reward"
                self._advance_stage(trigger)

        return True

    def _advance_stage(self, trigger=""):
        label = self.current_stage["label"]
        ckpt  = os.path.join(self.save_dir, f"nav_ent_{label}.zip")
        self.model.save(ckpt)
        print(f"\n[Curriculum] {label} -> COMPLETE (trigger: {trigger}) | ckpt -> {ckpt}")

        self.stage_idx += 1
        new_range = self.current_stage["target_range"]
        self.train_envs.env_method("_set_target_range", new_range)
        print(f"[Curriculum] -> {self.current_stage['label']}  (radius {new_range}m)\n")

        self._successes.clear()
        self._ep_dists.clear()
        self.last_eval_reward = -np.inf


# ── training ──────────────────────────────────────────────────────────────────

def make_env(target_range=0.3):
    def _init():
        return Monitor(NavAviary(target_range=target_range))
    return _init


def train(
    total_timesteps: int = 2_000_000,
    n_envs:          int = 8,
    save_dir:        str = "./models_nav_entropy",
    log_dir:         str = "./logs_nav_entropy",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    NavAviary._set_target_range = lambda self, r: setattr(self, "target_range", r)

    initial_range = CURRICULUM_STAGES[0]["target_range"]
    train_env = make_vec_env(make_env(initial_range), n_envs=n_envs)
    eval_env  = Monitor(NavAviary(target_range=initial_range))

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
        clip_range=0.2,         # wider than v1's 0.1 — allows meaningful early updates
        clip_range_vf=0.2,
        ent_coef=ENT_SCHEDULE[0][1],   # start at 0.05, annealed by callback
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cpu",
        seed=42,
    )

    nav_cb = EntropyNavCallback(
        train_envs=train_env,
        save_dir=save_dir,
        total_timesteps=total_timesteps,
        window=100,
        check_freq=2048,
        verbose=1,
    )
    callbacks = CallbackList([
        nav_cb,
        EvalNavCallback(
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
            save_freq=max(100_000 // n_envs, 1),
            save_path=save_dir,
            name_prefix="ppo_nav_ent",
        ),
    ])

    print(f"\n[INFO] Option D: Entropy-annealed curriculum from scratch")
    print(f"[INFO] {total_timesteps:,} steps, {n_envs} envs, clip_range=0.2")
    print(f"[INFO] Entropy schedule: 0.05 (0-500k) -> 0.01 (500k-1M) -> 0.001 (1M-2M)")
    print(f"[INFO] Watch: nav/policy_std — target drop from ~1.0 to <0.3 by 2M\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final = os.path.join(save_dir, "ppo_nav_entropy_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs",    type=int, default=8)
    p.add_argument("--save-dir",  default="./models_nav_entropy")
    p.add_argument("--log-dir",   default="./logs_nav_entropy")
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
