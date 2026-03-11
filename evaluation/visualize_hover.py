"""
Visualize a trained hover policy in PyBullet GUI.

Usage:
    python visualize_hover.py [--model ./models/best_model] [--episodes 3] [--record]
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def run_episode(model, env, render_delay: float = 1 / 48):
    """Run one episode and return trajectory data."""
    obs, _ = env.reset()
    positions, rewards = [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(float(reward))
        # KIN obs layout for single drone: obs shape (1, 72)
        # First 3 values: [x, y, z] position
        positions.append(obs[0, :3].copy())
        time.sleep(render_delay)

    return np.array(positions), np.array(rewards)


def plot_trajectory(positions_list: list, save_path: str = "hover_trajectory.png"):
    """Plot 3D trajectory and per-step reward for each episode."""
    fig = plt.figure(figsize=(14, 5))

    # --- 3D trajectory ---
    ax3d = fig.add_subplot(121, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, len(positions_list)))
    for i, ((pos, _rew), col) in enumerate(zip(positions_list, colors)):
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=col, label=f"Ep {i+1}", lw=1.5)
        ax3d.scatter(*pos[0], color=col, marker="o", s=40, zorder=5)
        ax3d.scatter(*pos[-1], color=col, marker="x", s=60, zorder=5)

    # Target hover point (HoverAviary default = [0, 0, 1])
    ax3d.scatter(0, 0, 1, color="red", marker="*", s=200, zorder=6, label="Target")
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("Drone Trajectory (3D)")
    ax3d.legend(fontsize=8)

    # --- Reward over time ---
    ax2d = fig.add_subplot(122)
    for i, (_, rew) in enumerate(positions_list):
        ax2d.plot(rew, label=f"Ep {i+1}", lw=1.5)
    ax2d.set_xlabel("Step"); ax2d.set_ylabel("Reward")
    ax2d.set_title("Step Reward per Episode")
    ax2d.legend(fontsize=8); ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Trajectory plot saved -> {save_path}")
    plt.show()


def main(model_path: str, n_episodes: int = 3, record: bool = False):
    if not os.path.isfile(model_path + ".zip"):
        # Try best_model fallback
        fallback = os.path.join(os.path.dirname(model_path), "best_model")
        if os.path.isfile(fallback + ".zip"):
            print(f"[WARN] {model_path}.zip not found, using {fallback}.zip")
            model_path = fallback
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}.zip\n"
                "Run train_hover.py first, or pass --model <path>."
            )

    print(f"[INFO] Loading model from {model_path}.zip")
    model = PPO.load(model_path)

    env = HoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=True,           # opens PyBullet window
        record=record,
    )

    episode_data = []
    for ep in range(n_episodes):
        print(f"\n[INFO] Episode {ep + 1}/{n_episodes}")
        pos, rew = run_episode(model, env)
        episode_data.append((pos, rew))
        dist = float(np.linalg.norm(pos[-1] - np.array([0.0, 0.0, 1.0])))
        print(f"  Steps: {len(rew)} | Total reward: {rew.sum():.2f} | "
              f"Final dist to target: {dist:.4f} m")

    env.close()

    # Rebuild list for plotting (pos array, reward array)
    plot_trajectory(episode_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/best_model",
                        help="Path to saved model (without .zip)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--record", action="store_true",
                        help="Save video via PyBullet recorder")
    args = parser.parse_args()

    main(model_path=args.model, n_episodes=args.episodes, record=args.record)
