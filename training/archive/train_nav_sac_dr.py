"""
SAC + Domain Randomization — resume from best_model, train with mild DR.

Loads models_nav_sac/best_model.zip (8/9), wraps train env with DRNavAviary.
Eval env uses CLEAN NavAviary so we can benchmark against clean physics.

DR config (mild):
    wind_max_force = 0.002 N
    mass_dr        = 0.08 (±8%)
    drag_dr        = 0.10 (±10%)

Run 1M steps. Benchmark on CLEAN NavAviary when done.
If 8/9 holds → DR works, safe to raise intensity.

Usage:
    python train_nav_sac_dr.py
"""

import argparse
import os
import numpy as np
from collections import deque

import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from envs.nav_aviary import NavAviary
from envs.dr_nav_aviary import DRNavAviary


class SACDRCallback(BaseCallback):
    def __init__(self, window=500, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.window     = window
        self.check_freq = check_freq
        self._successes = deque(maxlen=window)
        self._ep_dists  = deque(maxlen=window)
        self.last_eval_reward = -np.inf

    def notify_eval_reward(self, r):
        self.last_eval_reward = r

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._successes.append(float(info.get("success", 0.0)))
                if "dist_to_target" in info:
                    self._ep_dists.append(info["dist_to_target"])

        if self.n_calls % self.check_freq != 0 or len(self._successes) < 20:
            return True

        step = self.model.num_timesteps
        success_rate = float(np.mean(self._successes))
        mean_dist    = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")

        try:
            ent_coef = float(th.exp(self.model.log_ent_coef.detach()))
        except Exception:
            ent_coef = float("nan")

        self.logger.record("sacdr/success_rate",    success_rate)
        self.logger.record("sacdr/mean_final_dist", mean_dist)
        self.logger.record("sacdr/eval_reward",     self.last_eval_reward)
        self.logger.record("sacdr/ent_coef",        ent_coef)

        if self.verbose:
            print(f"[SACDR] step={step:,} | eval={self.last_eval_reward:.2f} | "
                  f"suc={success_rate:.2f} | dist={mean_dist:.3f}m | ent={ent_coef:.4f}")

        return True


class EvalDRCallback(EvalCallback):
    def __init__(self, dr_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dr_cb = dr_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.dr_cb.notify_eval_reward(self.last_mean_reward)
        return result


def train(
    total_timesteps: int = 1_000_000,
    resume_path:     str = "./models_nav_sac/best_model",
    save_dir:        str = "./models_nav_sac_dr",
    log_dir:         str = "./logs_nav_sac_dr",
    target_range:    float = 2.0,
    wind_max_force:  float = 0.002,
    mass_dr:         float = 0.08,
    drag_dr:         float = 0.10,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # Train with DR, eval on clean NavAviary
    train_env = Monitor(DRNavAviary(
        target_range=target_range,
        wind_max_force=wind_max_force,
        mass_dr=mass_dr,
        drag_dr=drag_dr,
    ))
    eval_env = Monitor(NavAviary(target_range=target_range))

    print(f"\n[INFO] SAC + DR — Resume from {resume_path}.zip")
    print(f"[INFO] DR: wind={wind_max_force}N, mass±{mass_dr*100:.0f}%, drag±{drag_dr*100:.0f}%")
    print(f"[INFO] Eval env: CLEAN NavAviary (no DR)")
    print(f"[INFO] target_range={target_range}m  total_timesteps={total_timesteps:,}\n")

    model = SAC.load(
        resume_path,
        env=train_env,
        device="cpu",
        verbose=1,
        tensorboard_log=log_dir,
    )
    model.learning_starts = 1000

    dr_cb = SACDRCallback(window=500, check_freq=5000, verbose=1)
    callbacks = CallbackList([
        dr_cb,
        EvalDRCallback(
            dr_cb=dr_cb,
            eval_env=eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=20_000,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=200_000,
            save_path=save_dir,
            name_prefix="sac_dr",
        ),
    ])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final = os.path.join(save_dir, "sac_dr_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps",      type=int,   default=1_000_000)
    p.add_argument("--resume-path",               default="./models_nav_sac/best_model")
    p.add_argument("--save-dir",                  default="./models_nav_sac_dr")
    p.add_argument("--log-dir",                   default="./logs_nav_sac_dr")
    p.add_argument("--target-range",   type=float, default=2.0)
    p.add_argument("--wind",           type=float, default=0.002)
    p.add_argument("--mass-dr",        type=float, default=0.08)
    p.add_argument("--drag-dr",        type=float, default=0.10)
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        resume_path=args.resume_path,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        target_range=args.target_range,
        wind_max_force=args.wind,
        mass_dr=args.mass_dr,
        drag_dr=args.drag_dr,
    )
