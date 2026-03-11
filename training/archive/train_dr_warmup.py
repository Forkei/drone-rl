"""
DR Warmup Training — Curriculum DR introduced gradually from v1 best_model.

Resumes v1 best_model (clean, 2.0m stage) then ramps DR over 300k steps:

  Local step    wind        mass      drag
  0             0           ±0%       ±0%     (clean — let V confirm landscape)
  25k           0.0005N     ±3%       ±5%
  75k           0.001N      ±6%       ±10%
  150k          0.002N      ±8%       ±12%
  300k          0.003N      ±10%      ±15%    (full DR)
  300k+         hold full DR for remainder

Adaptive hold: if clip_fraction > 0.4 at any transition, hold at that
level for an extra 50k steps before advancing to the next.

Eval on CLEAN NavAviary throughout — measures true policy quality.
DR health: clean_eval >= 8.0 = good, < 5.0 = too aggressive.

Mid-run 9-case checkpoint saved at local step 750k for comparison.

Usage:
    python train_dr_warmup.py [--resume ./models_nav/best_model]
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


# ── DR ramp schedule (local steps since resume) ───────────────────────────────

RAMP_SCHEDULE = [
    # (local_step_start, wind_max, mass_dr, drag_dr, label)
    (0,      0.0,    0.00, 0.00, "L0-clean"),
    (25000,  0.0005, 0.03, 0.05, "L1-mild"),
    (75000,  0.001,  0.06, 0.10, "L2-moderate"),
    (150000, 0.002,  0.08, 0.12, "L3-strong"),
    (300000, 0.003,  0.10, 0.15, "L4-full"),
]

MID_RUN_CHECKPOINT_STEP = 750_000   # save named checkpoint at this local step


# ── Warmup callback ───────────────────────────────────────────────────────────

class DRWarmupCallback(BaseCallback):
    """
    Manages the DR ramp schedule and monitors training health.

    Adaptive hold: if clip_fraction > 0.4 when a ramp step is due,
    hold the current DR level for 50k extra steps before advancing.

    Logs to TensorBoard:
      dr/wind_max, dr/mass_dr, dr/drag_dr  — current DR levels
      dr/ramp_level                         — 0-4 index
      dr/clean_eval_reward                  — from EvalCallback
      dr/clip_fraction                      — from PPO trainer
      dr/clip_fraction_50k                  — rolling 50k avg (25 checks)
      dr/success_rate, dr/mean_final_dist
    """

    def __init__(self, train_envs, save_dir,
                 window=100, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.train_envs       = train_envs
        self.save_dir         = save_dir
        self.window           = window
        self.check_freq       = check_freq
        self.last_eval_reward = -np.inf
        self._successes       = deque(maxlen=window)
        self._ep_dists        = deque(maxlen=window)
        self._clip_history    = deque(maxlen=25)

        # Ramp state
        self._ramp_idx        = 0      # current schedule index
        self._hold_until      = 0      # local step — don't advance before this
        self._start_timesteps = None   # set in _on_training_start
        self._mid_saved       = False  # mid-run checkpoint flag

        # Current DR params (initialised to level 0 = clean)
        self._current = RAMP_SCHEDULE[0]

    def _on_training_start(self):
        self._start_timesteps = self.model.num_timesteps
        self._apply_ramp(0, force=True)

    def notify_eval_reward(self, mean_reward: float):
        self.last_eval_reward = mean_reward

    @property
    def _local_steps(self) -> int:
        return self.model.num_timesteps - self._start_timesteps

    def _apply_ramp(self, idx: int, force: bool = False):
        level = RAMP_SCHEDULE[idx]
        if not force and idx == self._ramp_idx:
            return
        self._ramp_idx = idx
        self._current  = level
        wind, mass, drag, label = level[1], level[2], level[3], level[4]
        self.train_envs.env_method("_set_dr_params", wind, mass, drag)
        if self.verbose:
            print(f"\n[DR Ramp] -> {label} "
                  f"(wind={wind:.4f}N, mass±{mass*100:.0f}%, drag±{drag*100:.0f}%) "
                  f"@ local step {self._local_steps:,}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._successes.append(float(info.get("success", 0.0)))
                if "dist_to_target" in info:
                    self._ep_dists.append(info["dist_to_target"])

        local = self._local_steps

        # ── Mid-run checkpoint ──────────────────────────────────────────────
        if not self._mid_saved and local >= MID_RUN_CHECKPOINT_STEP:
            ckpt = os.path.join(self.save_dir, "mid_750k_model")
            self.model.save(ckpt)
            print(f"\n[DR Warmup] Mid-run checkpoint saved -> {ckpt}.zip "
                  f"(local step {local:,})\n")
            self._mid_saved = True

        if self.n_calls % self.check_freq != 0 or len(self._successes) < 10:
            return True

        # ── Ramp advancement ────────────────────────────────────────────────
        next_idx = self._ramp_idx + 1
        if next_idx < len(RAMP_SCHEDULE) and local >= self._hold_until:
            next_level = RAMP_SCHEDULE[next_idx]
            if local >= next_level[0]:
                # Check clip_fraction before advancing
                clip_frac = self.model.logger.name_to_value.get(
                    "train/clip_fraction", 0.0)
                if not np.isnan(clip_frac) and clip_frac > 0.4:
                    self._hold_until = local + 50_000
                    print(f"\n[DR Ramp] HOLD at {self._current[4]} — "
                          f"clip_fraction={clip_frac:.3f}>0.4, "
                          f"holding 50k more steps\n")
                else:
                    self._apply_ramp(next_idx)

        # ── Metrics ─────────────────────────────────────────────────────────
        success_rate = float(np.mean(self._successes))
        mean_dist    = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")
        clip_frac    = self.model.logger.name_to_value.get(
            "train/clip_fraction", float("nan"))
        self._clip_history.append(clip_frac)
        sustained    = float(np.nanmean(self._clip_history))

        self.logger.record("dr/wind_max",          self._current[1])
        self.logger.record("dr/mass_dr",           self._current[2])
        self.logger.record("dr/drag_dr",           self._current[3])
        self.logger.record("dr/ramp_level",        self._ramp_idx)
        self.logger.record("dr/success_rate",      success_rate)
        self.logger.record("dr/mean_final_dist",   mean_dist)
        self.logger.record("dr/clean_eval_reward", self.last_eval_reward)
        self.logger.record("dr/clip_fraction",     clip_frac)
        self.logger.record("dr/clip_fraction_50k", sustained)

        # DR health
        dr_health = ""
        if self.last_eval_reward != -np.inf:
            if self.last_eval_reward >= 8.0:
                dr_health = " [DR OK]"
            elif self.last_eval_reward >= 5.0:
                dr_health = " [DR WARN]"
            else:
                dr_health = " [DR CRITICAL]"

        clip_warn = ""
        if not np.isnan(sustained) and sustained > 0.3:
            clip_warn = f" [CLIP {sustained:.2f}]"

        if self.verbose:
            print(f"[DR] {self._current[4]} | "
                  f"local={local:,} | "
                  f"eval={self.last_eval_reward:.2f} | "
                  f"dist={mean_dist:.3f}m | "
                  f"suc={success_rate:.2f} | "
                  f"clip={clip_frac:.3f}"
                  f"{dr_health}{clip_warn}")

        return True


class EvalWarmupCallback(EvalCallback):
    """Forwards eval reward to DRWarmupCallback."""
    def __init__(self, warmup_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_cb = warmup_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.warmup_cb.notify_eval_reward(self.last_mean_reward)
        return result


# ── env factories ─────────────────────────────────────────────────────────────

def make_dr_env(target_range=2.0):
    """Training env: DRNavAviary starting clean (DR applied via ramp callback)."""
    def _init():
        env = DRNavAviary(
            target_range=target_range,
            wind_max_force=0.0,  # starts clean; warmup callback updates this
            mass_dr=0.0,
            drag_dr=0.0,
        )
        return Monitor(env)
    return _init


# ── training ──────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int = 1_500_000,
    n_envs: int = 8,
    save_dir: str = "./models_dr_warmup",
    log_dir:  str = "./logs_dr_warmup",
    resume_path: str = "./models_nav/best_model",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    train_env = make_vec_env(make_dr_env(target_range=2.0), n_envs=n_envs)
    eval_env  = Monitor(NavAviary(target_range=2.0))   # CLEAN eval throughout

    if not os.path.isfile(resume_path + ".zip"):
        raise FileNotFoundError(f"v1 best_model not found at {resume_path}.zip")

    print(f"[INFO] Resuming from {resume_path}.zip")
    model = PPO.load(resume_path, env=train_env,
                     tensorboard_log=log_dir, verbose=1,
                     device="cpu")

    warmup_cb = DRWarmupCallback(
        train_envs=train_env,
        save_dir=save_dir,
        window=100,
        check_freq=2048,
        verbose=1,
    )
    callbacks = CallbackList([
        warmup_cb,
        EvalWarmupCallback(
            warmup_cb=warmup_cb,
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
            name_prefix="ppo_dr_warmup",
        ),
    ])

    print(f"\n[INFO] DR Warmup training — {total_timesteps:,} steps from v1 best_model")
    print(f"[INFO] Ramp: clean 25k -> L1 mild -> L2 moderate -> L3 strong -> L4 full at 300k")
    print(f"[INFO] Adaptive hold: clip_fraction > 0.4 at transition -> hold 50k extra")
    print(f"[INFO] Eval: clean NavAviary | Mid checkpoint at local step 750k\n")

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False)

    final = os.path.join(save_dir, "ppo_dr_warmup_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_500_000)
    p.add_argument("--n-envs",    type=int, default=8)
    p.add_argument("--save-dir",  default="./models_dr_warmup")
    p.add_argument("--log-dir",   default="./logs_dr_warmup")
    p.add_argument("--resume",    default="./models_nav/best_model")
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_path=args.resume,
    )
