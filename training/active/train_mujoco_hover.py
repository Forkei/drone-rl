"""
MuJoCo HoverEnv — SAC training + FPS benchmark.

Usage:
    python train_mujoco_hover.py                  # 100k steps
    python train_mujoco_hover.py --timesteps 50000

Reports:
  - Steps/sec (FPS) vs PyBullet baseline (~40 fps)
  - Final success rate on 20-case eval
  - Camera obs test (drone-mounted RGB frame)
"""

import argparse
import time
import os
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from envs.mujoco.hover_env import HoverEnv


def fps_benchmark(n_steps=5000):
    """Pure env throughput — no policy, random actions."""
    env = HoverEnv()
    obs, _ = env.reset()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    elapsed = time.perf_counter() - t0
    env.close()
    fps = n_steps / elapsed
    print(f"\n[FPS benchmark] {n_steps} steps in {elapsed:.2f}s → {fps:.1f} steps/sec")
    print(f"  PyBullet baseline: ~40 fps")
    print(f"  Speedup:           {fps/40:.1f}×")
    return fps


def eval_policy(model, n_episodes=20):
    """Quick success-rate evaluation."""
    env = HoverEnv()
    successes = 0
    dists = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        if info["success"]:
            successes += 1
        dists.append(info["dist_to_target"])
    env.close()
    return successes / n_episodes, float(np.mean(dists))


def camera_obs_test():
    """Grab a drone-cam frame and save it."""
    env = HoverEnv()
    obs, _ = env.reset()
    frame = env.get_drone_cam_frame(height=32, width=48)
    env.close()

    os.makedirs("./benchmark_results", exist_ok=True)
    out = "./benchmark_results/mujoco_drone_cam.png"
    try:
        from PIL import Image
        Image.fromarray(frame, "RGB").save(out)
        print(f"  [cam] Frame shape: {frame.shape}  saved → {out}")
    except ImportError:
        print(f"  [cam] Frame shape: {frame.shape}  (PIL not available — not saved)")
    return frame


def train(total_timesteps=100_000, save_dir="./models_mujoco", log_dir="./logs_mujoco"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("MuJoCo HoverEnv — SAC Training")
    print("=" * 60)

    # Sanity-check the env
    print("\n[1/4] Checking env...")
    check_env(HoverEnv(), warn=True)
    print("  Env check passed.")

    # FPS benchmark
    print("\n[2/4] FPS benchmark (5k random steps)...")
    fps = fps_benchmark(5000)

    # Train
    print(f"\n[3/4] SAC training for {total_timesteps:,} steps...")
    train_env = Monitor(HoverEnv())
    eval_env  = Monitor(HoverEnv())

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        learning_starts=5_000,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    t_start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=True)
    t_total = time.perf_counter() - t_start

    train_fps = total_timesteps / t_total
    print(f"\n  Training done: {t_total:.1f}s  →  {train_fps:.1f} steps/sec (with SAC updates)")

    model.save(os.path.join(save_dir, "hover_final"))
    model.save_replay_buffer(os.path.join(save_dir, "replay_buffer"))

    # Eval
    print("\n[4/4] Final evaluation (20 episodes)...")
    success_rate, mean_dist = eval_policy(model, n_episodes=20)
    print(f"  Success rate: {success_rate*100:.1f}%")
    print(f"  Mean final dist: {mean_dist:.3f}m")

    # Camera obs test
    print("\n[camera] Testing drone-mounted camera obs...")
    camera_obs_test()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Env FPS (random policy):    {fps:.0f} steps/sec")
    print(f"  Training FPS (SAC):         {train_fps:.0f} steps/sec")
    print(f"  PyBullet baseline:          ~40 steps/sec")
    print(f"  Speedup (env only):         {fps/40:.1f}×")
    print(f"  Speedup (training):         {train_fps/40:.1f}×")
    print(f"  Success rate @ {total_timesteps//1000}k steps: {success_rate*100:.1f}%")
    print(f"  Mean final dist:            {mean_dist:.3f}m")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--save-dir",  default="./models_mujoco")
    p.add_argument("--log-dir",   default="./logs_mujoco")
    args = p.parse_args()
    train(args.timesteps, args.save_dir, args.log_dir)
