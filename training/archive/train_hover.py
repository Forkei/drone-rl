"""
PPO Hover Training Script
Trains a quadrotor to hover at a fixed target position using gym-pybullet-drones + SB3.

Usage:
    python train_hover.py [--timesteps 500000] [--eval-freq 20000] [--save-dir ./models]
"""

import argparse
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def make_env(gui: bool = False, rank: int = 0):
    """Factory for a single HoverAviary instance."""
    def _init():
        env = HoverAviary(
            obs=ObservationType.KIN,   # kinematic obs, shape (1, 72) incl. neighbor info
            act=ActionType.RPM,        # normalized RPM commands [-1, 1]
            gui=gui,
        )
        env = Monitor(env)
        return env
    return _init


def train(
    total_timesteps: int = 500_000,
    eval_freq: int = 20_000,
    n_envs: int = 4,
    save_dir: str = "./models",
    log_dir: str = "./logs",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- Parallel training envs ---
    train_env = make_vec_env(make_env(), n_envs=n_envs)

    # --- Single eval env (no GUI during training) ---
    eval_env = make_env(gui=False)()

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq // n_envs, 1),
        save_path=save_dir,
        name_prefix="ppo_hover",
        verbose=0,
    )

    # --- PPO model ---
    # Policy net [256, 256] works well for drone hover
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,           # small entropy bonus keeps exploration alive
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=42,
    )

    print(f"\n[INFO] Training PPO for {total_timesteps:,} timesteps on {n_envs} envs...")
    print(f"[INFO] Models -> {save_dir}  |  Logs -> {log_dir}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, "ppo_hover_final")
    model.save(final_path)
    print(f"\n[INFO] Final model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="./models")
    parser.add_argument("--log-dir", type=str, default="./logs")
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
