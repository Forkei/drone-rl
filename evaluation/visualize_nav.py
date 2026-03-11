"""
Navigation Policy Visualizer

Runs a trained nav policy across structured test cases and plots full 3D trajectories.
Reveals overshooting, oscillation, altitude bias, and heading asymmetry.

Usage:
    # GUI + plots for all test cases
    python visualize_nav.py --model ./models_nav/best_model

    # Headless (no GUI window), just plots
    python visualize_nav.py --model ./models_nav/best_model --no-gui

    # Single specific test case
    python visualize_nav.py --model ./models_nav/best_model --case low_target
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from stable_baselines3 import PPO
from envs.nav_aviary import NavAviary


# ── Test cases ────────────────────────────────────────────────────────────────
# Each case defines where to spawn the drone and where the target is.
# drone_pos: initial [x, y, z]
# target:    goal [x, y, z]

TEST_CASES = {
    # Approach from each axis
    "approach_+X":  {"drone_pos": [1.0, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "approach_-X":  {"drone_pos": [-1.0, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "approach_+Y":  {"drone_pos": [0.0, 1.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "approach_-Y":  {"drone_pos": [0.0, -1.0, 1.0], "target": [0.0, 0.0, 1.0]},
    # Altitude extremes
    "low_target":   {"drone_pos": [0.0, 0.0, 1.0], "target": [0.0, 0.0, 0.3]},
    "high_target":  {"drone_pos": [0.0, 0.0, 1.0], "target": [0.0, 0.0, 1.8]},
    # Distance
    "near_0.3m":    {"drone_pos": [0.3, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "far_1.8m":     {"drone_pos": [1.8, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    # Diagonal
    "diagonal":     {"drone_pos": [0.7, 0.7, 1.5], "target": [0.0, 0.0, 0.5]},
}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_test_case(model, case_cfg: dict, gui: bool = False,
                  render_delay: float = 1 / 30) -> dict:
    """Run one test case and return trajectory data."""
    drone_pos = np.array(case_cfg["drone_pos"])
    target    = np.array(case_cfg["target"])

    env = NavAviary(
        target_range=0.5,
        home_pos=drone_pos,   # spawn drone at specified position
        gui=gui,
        record=False,
    )
    # Override target directly
    env.TARGET_POS = target.copy()

    obs, _ = env.reset()
    # Force drone to start at specified position
    env.TARGET_POS = target.copy()   # reset may resample — override again

    positions, rewards, dists = [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = env._getDroneStateVector(0)
        positions.append(state[0:3].copy())
        rewards.append(float(reward))
        dists.append(info["dist_to_target"])

        if gui:
            time.sleep(render_delay)

    env.close()

    positions = np.array(positions)
    return {
        "positions": positions,
        "rewards":   np.array(rewards),
        "dists":     np.array(dists),
        "target":    target,
        "drone_start": drone_pos,
        "success":   info.get("success", False),
        "final_dist": dists[-1] if dists else float("inf"),
        "n_steps":   len(rewards),
        "total_reward": float(np.sum(rewards)),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, out: str = "nav_trajectories.png"):
    n_cases = len(results)
    cols = 3
    rows = (n_cases + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    fig.suptitle("Navigation Policy — Trajectory Analysis", fontsize=14, fontweight="bold")

    for idx, (name, data) in enumerate(results.items()):
        pos    = data["positions"]
        target = data["target"]
        start  = data["drone_start"]
        dists  = data["dists"]
        success = data["success"]

        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")

        # Trajectory line coloured by time (early=blue, late=red)
        n = len(pos)
        colors = plt.cm.coolwarm(np.linspace(0, 1, n))
        for i in range(n - 1):
            ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2],
                    color=colors[i], lw=1.5, alpha=0.8)

        # Markers
        ax.scatter(*pos[0],   color="blue",  s=60,  marker="o", zorder=5, label="Start")
        ax.scatter(*pos[-1],  color="red",   s=60,  marker="x", zorder=5, label="End")
        ax.scatter(*target,   color="gold",  s=120, marker="*", zorder=6, label="Target")

        # Status
        status = "SUCCESS" if success else f"dist={data['final_dist']:.2f}m"
        color  = "green" if success else "red"
        ax.set_title(f"{name}\n{status} | {data['n_steps']} steps",
                     fontsize=9, color=color, fontweight="bold" if success else "normal")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(fontsize=6, loc="upper left")

        # Make axes equal-ish around the action range
        all_pts = np.vstack([pos, target.reshape(1, 3), start.reshape(1, 3)])
        centre  = all_pts.mean(axis=0)
        span    = max(np.ptp(all_pts, axis=0).max(), 0.5) * 0.6
        ax.set_xlim(centre[0] - span, centre[0] + span)
        ax.set_ylim(centre[1] - span, centre[1] + span)
        ax.set_zlim(max(0, centre[2] - span), centre[2] + span)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[INFO] Trajectory plot saved -> {out}")
    plt.show()


def plot_distance_curves(results: dict, out: str = "nav_dist_curves.png"):
    """Distance-to-target over time for each test case."""
    n_cases = len(results)
    cols = 3
    rows = (n_cases + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    fig.suptitle("Distance to Target Over Time", fontsize=13, fontweight="bold")
    axes = np.array(axes).flatten()

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        dists = data["dists"]
        steps = np.arange(len(dists))
        ax.plot(steps, dists, lw=1.5, color="steelblue")
        ax.axhline(0.1,  color="green",  ls="--", lw=1, label="Goal (0.1m)")
        ax.axhline(0.15, color="orange", ls=":",  lw=1, label="Warm zone (0.15m)")
        ax.fill_between(steps, 0, dists, alpha=0.1, color="steelblue")
        success = data["success"]
        title_color = "green" if success else "gray"
        ax.set_title(name, fontsize=8, color=title_color)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Dist (m)", fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[INFO] Distance curves saved -> {out}")
    plt.show()


def print_summary(results: dict):
    print(f"\n{'='*60}")
    print(f"  Navigation Policy — Test Summary")
    print(f"{'='*60}")
    print(f"  {'Case':<16} {'Steps':>6} {'Final dist':>11} {'Reward':>8} {'Success':>8}")
    print(f"  {'-'*56}")
    for name, data in results.items():
        status = "YES" if data["success"] else "no"
        print(f"  {name:<16} {data['n_steps']:>6} {data['final_dist']:>10.3f}m "
              f"{data['total_reward']:>8.2f} {status:>8}")
    successes = sum(1 for d in results.values() if d["success"])
    print(f"  {'-'*56}")
    print(f"  Success rate: {successes}/{len(results)} cases")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_path: str, gui: bool = True,
         cases: list = None, out_dir: str = "."):
    # Find model
    if not os.path.isfile(model_path + ".zip"):
        fallback = os.path.join(os.path.dirname(model_path), "best_model")
        if os.path.isfile(fallback + ".zip"):
            print(f"[WARN] {model_path}.zip not found, using {fallback}.zip")
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model not found: {model_path}.zip")

    print(f"[INFO] Loading model: {model_path}.zip")
    model = PPO.load(model_path)

    run_cases = {k: v for k, v in TEST_CASES.items()
                 if cases is None or k in cases}

    print(f"[INFO] Running {len(run_cases)} test cases "
          f"({'with' if gui else 'without'} GUI)\n")

    results = {}
    for name, cfg in run_cases.items():
        print(f"  Running: {name} ...", end=" ", flush=True)
        data = run_test_case(model, cfg, gui=gui)
        results[name] = data
        status = "SUCCESS" if data["success"] else f"dist={data['final_dist']:.2f}m"
        print(status)

    print_summary(results)

    os.makedirs(out_dir, exist_ok=True)
    plot_results(results,
                 out=os.path.join(out_dir, "nav_trajectories.png"))
    plot_distance_curves(results,
                         out=os.path.join(out_dir, "nav_dist_curves.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="./models_nav/best_model",
                   help="Path to model (without .zip)")
    p.add_argument("--no-gui",  action="store_true",
                   help="Skip PyBullet GUI (headless — faster)")
    p.add_argument("--case",    nargs="+", default=None,
                   help=f"Run specific cases. Options: {list(TEST_CASES.keys())}")
    p.add_argument("--out-dir", default=".",
                   help="Directory for output plots")
    args = p.parse_args()

    main(
        model_path=args.model,
        gui=not args.no_gui,
        cases=args.case,
        out_dir=args.out_dir,
    )
