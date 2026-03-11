"""
SAC Navigation Training v2 — resume from best_model with altitude shaping.

Loads models_nav_sac/best_model.zip (8/9, trained to 1M steps).
Adds altitude_bonus_w=0.3 to NavAviary reward to address high_target failure.
Runs 2M more steps.

Usage:
    python train_nav_sac_v2.py
    python train_nav_sac_v2.py --timesteps 2000000 --save-dir ./models_nav_sac_v2
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


class SACv2Callback(BaseCallback):
    def __init__(self, save_dir, window=500, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.save_dir   = save_dir
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

        step = self.model.num_timesteps

        if self.n_calls % self.check_freq != 0 or len(self._successes) < 20:
            return True

        success_rate = float(np.mean(self._successes))
        mean_dist    = float(np.mean(self._ep_dists)) if self._ep_dists else float("nan")

        try:
            ent_coef = float(th.exp(self.model.log_ent_coef.detach()))
        except Exception:
            ent_coef = float("nan")

        self.logger.record("sac2/success_rate",    success_rate)
        self.logger.record("sac2/mean_final_dist", mean_dist)
        self.logger.record("sac2/eval_reward",     self.last_eval_reward)
        self.logger.record("sac2/ent_coef",        ent_coef)

        if self.verbose:
            print(f"[SAC2] step={step:,} | eval={self.last_eval_reward:.2f} | "
                  f"suc={success_rate:.2f} | dist={mean_dist:.3f}m | ent={ent_coef:.4f}")

        return True


class EvalSACv2Callback(EvalCallback):
    def __init__(self, sac_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sac_cb = sac_cb

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward != -np.inf:
            self.sac_cb.notify_eval_reward(self.last_mean_reward)
        return result


def train(
    total_timesteps: int = 2_000_000,
    resume_path:     str = "./models_nav_sac/best_model",
    save_dir:        str = "./models_nav_sac_v2",
    log_dir:         str = "./logs_nav_sac_v2",
    target_range:    float = 2.0,
    altitude_bonus_w: float = 0.3,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    train_env = Monitor(NavAviary(
        target_range=target_range,
        altitude_bonus_w=altitude_bonus_w,
    ))
    eval_env = Monitor(NavAviary(
        target_range=target_range,
        altitude_bonus_w=altitude_bonus_w,
    ))

    print(f"\n[INFO] SAC v2 — Resume from {resume_path}.zip")
    print(f"[INFO] altitude_bonus_w={altitude_bonus_w} (altitude shaping active)")
    print(f"[INFO] target_range={target_range}m  total_timesteps={total_timesteps:,}")
    print(f"[INFO] save_dir={save_dir}  log_dir={log_dir}\n")

    model = SAC.load(
        resume_path,
        env=train_env,
        device="cpu",
        verbose=1,
        tensorboard_log=log_dir,
    )
    # Replay buffer is not saved — rebuild from scratch but keep weights
    # Use small learning_starts so we don't wait long before training
    model.learning_starts = 1000

    sac_cb = SACv2Callback(save_dir=save_dir, window=500, check_freq=5000, verbose=1)
    callbacks = CallbackList([
        sac_cb,
        EvalSACv2Callback(
            sac_cb=sac_cb,
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
            name_prefix="sac_v2",
        ),
    ])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    final = os.path.join(save_dir, "sac_v2_final")
    model.save(final)
    print(f"\n[INFO] Final model -> {final}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps",         type=int,   default=2_000_000)
    p.add_argument("--resume-path",                   default="./models_nav_sac/best_model")
    p.add_argument("--save-dir",                      default="./models_nav_sac_v2")
    p.add_argument("--log-dir",                       default="./logs_nav_sac_v2")
    p.add_argument("--target-range",      type=float, default=2.0)
    p.add_argument("--altitude-bonus-w",  type=float, default=0.3)
    args = p.parse_args()

    train(
        total_timesteps=args.timesteps,
        resume_path=args.resume_path,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        target_range=args.target_range,
        altitude_bonus_w=args.altitude_bonus_w,
    )
