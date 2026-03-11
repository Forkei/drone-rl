"""
Visualize trained nav policy in MuJoCo viewer.

Usage:
  python evaluation/visualize_nav_mujoco.py                 # random targets, 2.0m range
  python evaluation/visualize_nav_mujoco.py --range 0.5     # short-range targets
  python evaluation/visualize_nav_mujoco.py --episodes 5    # run 5 episodes
  python evaluation/visualize_nav_mujoco.py --model models_nav_mujoco/nav_stage2_1.2m

Controls (MuJoCo viewer):
  Mouse drag  — rotate
  Scroll      — zoom
  Space       — pause
  Esc         — quit
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from envs.mujoco.nav_env import NavEnv


def run_episode(env, model, viewer, pause_on_success=True):
    obs, _ = env.reset()
    total_reward = 0.0
    step = 0
    success = False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Sync viewer with physics
        viewer.sync()

        # Slow down to real-time (50Hz control = 0.02s per step, N_SUBSTEPS=10 × 0.002s)
        time.sleep(0.02)

        if info.get("success"):
            success = True
            print(f"  SUCCESS in {step} steps | reward={total_reward:.2f} | "
                  f"dist={info['dist_to_target']:.3f}m")
            if pause_on_success:
                time.sleep(1.0)

        if terminated or truncated:
            if not success:
                status = "CRASHED" if info.get("crashed") else "TIMEOUT"
                print(f"  {status} at step {step} | reward={total_reward:.2f} | "
                      f"dist={info['dist_to_target']:.3f}m")
            break

    return success, total_reward, step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="models_nav_mujoco/best_model",
                        help="Path to model zip (without .zip)")
    parser.add_argument("--range",    type=float, default=2.0,
                        help="Target spawn range in meters")
    parser.add_argument("--episodes", type=int,   default=20,
                        help="Number of episodes to run")
    args = parser.parse_args()

    model_path = args.model
    print(f"Loading model: {model_path}")
    model = SAC.load(model_path, device="cpu")

    env = NavEnv(target_range=args.range, render_mode=None)

    successes = 0
    rewards = []

    print(f"\nRunning {args.episodes} episodes, target_range={args.range}m")
    print("MuJoCo viewer window will open. Controls: drag=rotate, scroll=zoom, Esc=quit\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Set a good default camera angle
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45

        for ep in range(args.episodes):
            print(f"Episode {ep+1}/{args.episodes}:")
            ok, r, steps = run_episode(env, model, viewer)
            successes += ok
            rewards.append(r)

            if not viewer.is_running():
                print("Viewer closed — stopping.")
                break

            time.sleep(0.5)  # brief pause between episodes

    env.close()

    print(f"\n{'='*50}")
    print(f"Results over {len(rewards)} episodes:")
    print(f"  Success rate: {successes}/{len(rewards)} ({100*successes/len(rewards):.0f}%)")
    print(f"  Mean reward:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Best reward:  {max(rewards):.2f}")


if __name__ == "__main__":
    main()
