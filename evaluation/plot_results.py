"""
Post-training result plotter.
Reads SB3 monitor CSVs and EvalCallback evaluations, produces a summary figure.

Usage:
    python plot_results.py [--log-dir ./logs] [--out results.png]
"""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── helpers ────────────────────────────────────────────────────────────────────

def read_monitor_csvs(log_dir: str):
    paths = glob.glob(os.path.join(log_dir, "**", "monitor.csv"), recursive=True)
    rewards, lengths = [], []
    for p in paths:
        with open(p) as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    rewards.append(float(parts[0]))
                    lengths.append(int(parts[1]))
                except ValueError:
                    pass
    return np.array(rewards), np.array(lengths)


def read_eval_results(log_dir: str):
    """Read evaluations.npz written by EvalCallback."""
    npz_path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.isfile(npz_path):
        return None
    data = np.load(npz_path)
    return data  # keys: timesteps, results, ep_lengths


def smooth(arr: np.ndarray, window: int = 50) -> np.ndarray:
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ── main ───────────────────────────────────────────────────────────────────────

def plot(log_dir: str, out: str = "results.png"):
    rewards, lengths = read_monitor_csvs(log_dir)
    eval_data = read_eval_results(log_dir)

    if len(rewards) == 0 and eval_data is None:
        print(f"[WARN] No data found in {log_dir}. Train first.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PPO Hover Training Results", fontsize=14, fontweight="bold")

    # ── 1. Training reward curve ────────────────────────────────────────────
    ax = axes[0]
    if len(rewards) > 0:
        ax.plot(rewards, alpha=0.25, color="steelblue", lw=0.8, label="Episode reward")
        if len(rewards) >= 50:
            s = smooth(rewards)
            ax.plot(range(len(rewards) - len(s), len(rewards)), s,
                    color="steelblue", lw=2, label="Smoothed (50-ep)")
        ax.set_title("Training Episode Reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── 2. Eval mean reward ─────────────────────────────────────────────────
    ax = axes[1]
    if eval_data is not None:
        ts = eval_data["timesteps"]
        means = eval_data["results"].mean(axis=1)
        stds = eval_data["results"].std(axis=1)
        ax.plot(ts, means, color="darkorange", lw=2, marker="o", ms=4, label="Eval mean")
        ax.fill_between(ts, means - stds, means + stds, alpha=0.25, color="darkorange")
        ax.set_title("Eval Mean Reward vs Timestep")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Mean Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    else:
        ax.text(0.5, 0.5, "No eval data\n(evaluations.npz not found)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Eval Mean Reward vs Timestep")

    # ── 3. Episode length distribution ─────────────────────────────────────
    ax = axes[2]
    if len(lengths) > 0:
        ax.hist(lengths, bins=40, color="mediumseagreen", edgecolor="white", lw=0.5)
        ax.axvline(np.mean(lengths), color="darkgreen", lw=2,
                   linestyle="--", label=f"Mean: {np.mean(lengths):.0f}")
        ax.set_title("Episode Length Distribution")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[INFO] Results plot saved -> {out}")
    plt.show()

    # ── Summary stats ───────────────────────────────────────────────────────
    if len(rewards) > 0:
        print(f"\n{'='*50}")
        print(f"  Training Summary")
        print(f"{'='*50}")
        print(f"  Total episodes   : {len(rewards):,}")
        print(f"  Best episode     : {rewards.max():.2f}")
        print(f"  Last-100 mean    : {rewards[-100:].mean():.2f}")
        print(f"  Last-100 std     : {rewards[-100:].std():.2f}")
        print(f"  Avg ep length    : {lengths.mean():.0f} steps")
    if eval_data is not None:
        best_idx = means.argmax()
        print(f"  Best eval reward : {means[best_idx]:.2f} ± {stds[best_idx]:.2f}"
              f"  @ {ts[best_idx]:,} steps")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--out", default="results.png")
    args = parser.parse_args()
    plot(args.log_dir, args.out)
