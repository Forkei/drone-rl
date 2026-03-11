"""
Live training monitor — reads SB3 monitor CSV and prints a rolling reward summary.

Usage:
    python monitor_training.py [--log-dir ./logs] [--interval 10]
"""

import argparse
import os
import time
import glob
import numpy as np


def read_monitor_csv(path: str):
    """Parse SB3/gymnasium Monitor CSV → (episode_rewards, episode_lengths)."""
    rewards, lengths = [], []
    with open(path, "r") as f:
        lines = f.readlines()
    # Skip first 2 comment lines
    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                rewards.append(float(parts[0]))
                lengths.append(int(parts[1]))
            except ValueError:
                continue
    return np.array(rewards), np.array(lengths)


def find_monitor_files(log_dir: str):
    return glob.glob(os.path.join(log_dir, "**", "monitor.csv"), recursive=True)


def print_stats(rewards: np.ndarray, label: str = ""):
    if len(rewards) == 0:
        print(f"  {label}  No episodes yet.")
        return
    window = rewards[-50:]
    print(
        f"  {label}  Episodes: {len(rewards):5d} | "
        f"Last-50 mean: {window.mean():8.2f} | "
        f"Last-50 max: {window.max():8.2f} | "
        f"Best ever: {rewards.max():8.2f}"
    )


def monitor(log_dir: str, interval: int = 10):
    print(f"[Monitor] Watching {log_dir} every {interval}s  (Ctrl-C to stop)\n")
    while True:
        files = find_monitor_files(log_dir)
        if not files:
            print("  Waiting for monitor files...")
        else:
            all_rewards = []
            for f in files:
                r, _ = read_monitor_csv(f)
                all_rewards.extend(r.tolist())
            all_rewards = np.array(all_rewards)
            print_stats(all_rewards, label=time.strftime("%H:%M:%S"))
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    monitor(args.log_dir, args.interval)
