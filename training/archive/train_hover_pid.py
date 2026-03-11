"""
PID Hover Training — ActionType.PID (waypoint → inner PID loop → RPMs).
Policy outputs a 3D target position [-1,1]^3; built-in PID handles attitude.
Converges much faster and flies smoother than raw RPM.

Usage:
    python train_hover_pid.py [--timesteps 150000]
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def make_env(gui: bool = False):
    def _init():
        env = HoverAviary(obs=ObservationType.KIN, act=ActionType.PID, gui=gui)
        return Monitor(env)
    return _init


def train(total_timesteps=150_000, eval_freq=10_000, n_envs=4,
          save_dir="./models_pid", log_dir="./logs_pid"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_env = make_vec_env(make_env(), n_envs=n_envs)
    eval_env  = make_env()()

    callbacks = [
        EvalCallback(eval_env,
                     best_model_save_path=save_dir,
                     log_path=log_dir,
                     eval_freq=max(eval_freq // n_envs, 1),
                     n_eval_episodes=5,
                     deterministic=True,
                     verbose=1),
        CheckpointCallback(save_freq=max(eval_freq // n_envs, 1),
                           save_path=save_dir,
                           name_prefix="ppo_hover_pid"),
    ]

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
        clip_range=0.2,
        ent_coef=0.005,          # less entropy needed — PID action space is smooth
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cpu",            # MLP PPO runs faster on CPU (bottleneck = sim, not net)
        seed=42,
    )

    print(f"\n[INFO] PID Hover — {total_timesteps:,} steps on {n_envs} envs")
    print(f"[INFO] Obs shape: (1, 57)  |  Act shape: (1, 3) = target [x, y, z]\n")

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks, progress_bar=True)

    model.save(os.path.join(save_dir, "ppo_hover_pid_final"))
    print(f"\n[INFO] Saved to {save_dir}/ppo_hover_pid_final.zip")
    train_env.close(); eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--n-envs",    type=int, default=4)
    p.add_argument("--save-dir",  default="./models_pid")
    p.add_argument("--log-dir",   default="./logs_pid")
    args = p.parse_args()
    train(args.timesteps, n_envs=args.n_envs,
          save_dir=args.save_dir, log_dir=args.log_dir)
