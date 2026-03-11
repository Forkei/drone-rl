"""
SAC Altitude Fine-tune — targeted vertical curriculum from best_model (8/9).

Loads models_nav_sac/best_model.zip.
Trains on AltitudeNavAviary (targets always directly above drone).
Eval env is also AltitudeNavAviary with a fixed 0.8m-above target.
No reward function changes — distribution shift only.

Usage:
    python train_nav_sac_altitude.py
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

from envs.altitude_nav_aviary import AltitudeNavAviary
from envs.nav_aviary import NavAviary


class SACaltCallback(BaseCallback):
    def __init__(self, window=200, check_freq=5000, verbose=1):
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

        if self.n_calls % self.check_freq != 0 or len(self._successes) < 10:
            return True

        step = self.model.num_timesteps
        success_rate = float(np.mean(self._successes))
        mean_dist    = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")

        try:
            ent_coef = float(th.exp(self.model.log_ent_coef.detach()))
        except Exception:
            ent_coef = float("nan")

        self.logger.record("alt/success_rate",    success_rate)
        self.logger.record("alt/mean_final_dist", mean_dist)
        self.logger.record("alt/eval_reward",     self.last_eval_reward)
        self.logger.record("alt/ent_coef",        ent_coef)

        if self.verbose:
            print(f"[ALT] step={step:,} | eval={self.last_eval_reward:.2f} | "
                  f"suc={success_rate:.2f} | dist={mean_dist:.3f}m | ent={ent_coef:.4f}")
        return True


class EvalAltCallback(EvalCallback):
    def __init__(self, alt_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alt_cb = alt_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.alt_cb.notify_eval_reward(self.last_mean_reward)
        return result


def train(
    total_timesteps: int = 200_000,
    resume_path:     str = "./models_nav_sac/best_model",
    save_dir:        str = "./models_nav_sac_altitude",
    log_dir:         str = "./logs_nav_sac_altitude",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    train_env = Monitor(AltitudeNavAviary(target_range=2.0))

    # Eval env: fixed target 0.8m above start (mirrors high_target benchmark case)
    eval_env  = Monitor(AltitudeNavAviary(target_range=2.0))

    print(f"\n[INFO] SAC Altitude fine-tune — Resume from {resume_path}.zip")
    print(f"[INFO] Train: AltitudeNavAviary (z_target = drone_z + uniform(0.5, 1.5))")
    print(f"[INFO] Eval:  AltitudeNavAviary")
    print(f"[INFO] total_timesteps={total_timesteps:,}  save_dir={save_dir}\n")

    model = SAC.load(
        resume_path,
        env=train_env,
        device="cpu",
        verbose=1,
        tensorboard_log=log_dir,
    )
    model.learning_starts = 1000

    alt_cb = SACaltCallback(window=200, check_freq=5000, verbose=1)
    callbacks = CallbackList([
        alt_cb,
        EvalAltCallback(
            alt_cb=alt_cb,
            eval_env=eval_env,
            best_model_save_path=save_dir,
            log_path=log_dir,
            eval_freq=10_000,       # more frequent — only 200k total
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=save_dir,
            name_prefix="sac_alt",
        ),
    ])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final = os.path.join(save_dir, "sac_altitude_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps",    type=int, default=200_000)
    p.add_argument("--resume-path",            default="./models_nav_sac/best_model")
    p.add_argument("--save-dir",               default="./models_nav_sac_altitude")
    p.add_argument("--log-dir",                default="./logs_nav_sac_altitude")
    args = p.parse_args()
    train(
        total_timesteps=args.timesteps,
        resume_path=args.resume_path,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
