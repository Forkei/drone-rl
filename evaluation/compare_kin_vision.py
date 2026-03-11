"""
compare_kin_vision.py — Side-by-side benchmark: KIN policy vs Vision policy.

Runs both models on the same 9 test cases (same positions, same targets).
Produces:
  - Printed comparison table
  - 3D trajectory comparison plots (KIN blue, Vision orange, side by side)
  - JSON results file

"Here's the drone with GPS — here's the same drone using only its camera."

Usage:
    python compare_kin_vision.py --no-gui
    python compare_kin_vision.py --no-gui --out-dir ./results/comparison/
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from envs.nav_aviary import NavAviary
from envs.vision_nav_aviary import VisionNavAviary


TEST_CASES = {
    "approach_+X": {"drone_pos": [1.0,  0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "approach_-X": {"drone_pos": [-1.0, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "approach_+Y": {"drone_pos": [0.0,  1.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "approach_-Y": {"drone_pos": [0.0, -1.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "low_target":  {"drone_pos": [0.0,  0.0, 1.0], "target": [0.0, 0.0, 0.3]},
    "high_target": {"drone_pos": [0.0,  0.0, 1.0], "target": [0.0, 0.0, 1.8]},
    "near_0.3m":   {"drone_pos": [0.3,  0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "far_1.8m":    {"drone_pos": [1.8,  0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "diagonal":    {"drone_pos": [0.7,  0.7, 1.5], "target": [0.0, 0.0, 0.5]},
}


def run_kin_case(model, case_cfg, gui=False):
    drone_pos = np.array(case_cfg["drone_pos"])
    target    = np.array(case_cfg["target"])
    env = NavAviary(target_range=0.5, home_pos=drone_pos, gui=gui, record=False)
    env.TARGET_POS = target.copy()
    obs, _ = env.reset()
    env.TARGET_POS = target.copy()

    positions, dists = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = env._getDroneStateVector(0)
        positions.append(state[0:3].copy())
        dists.append(info["dist_to_target"])
    env.close()

    return {
        "positions": np.array(positions),
        "dists":     dists,
        "target":    target,
        "start":     drone_pos,
        "success":   info.get("success", False),
        "final_dist": dists[-1] if dists else 999,
    }


def run_vision_case(model, case_cfg, gui=False):
    drone_pos = np.array(case_cfg["drone_pos"])
    target    = np.array(case_cfg["target"])
    env = VisionNavAviary(target_range=0.5, home_pos=drone_pos, gui=gui)
    env.TARGET_POS = target.copy()
    obs, _ = env.reset()
    env.TARGET_POS = target.copy()

    positions, dists = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = env._getDroneStateVector(0)
        positions.append(state[0:3].copy())
        dists.append(info["dist_to_target"])
    env.close()

    return {
        "positions": np.array(positions),
        "dists":     dists,
        "target":    target,
        "start":     drone_pos,
        "success":   info.get("success", False),
        "final_dist": dists[-1] if dists else 999,
    }


def print_comparison_table(kin_results, vis_results):
    cases = list(TEST_CASES.keys())
    print(f"\n{'='*62}")
    print(f"  KIN (GPS) vs Vision (Camera-only) — 9-case comparison")
    print(f"{'='*62}")
    print(f"  {'Case':<16}  {'KIN':>10}  {'VISION':>10}  {'Better':>8}")
    print(f"  {'-'*56}")

    kin_ok = vis_ok = 0
    for case in cases:
        k = kin_results.get(case, {})
        v = vis_results.get(case, {})
        k_str = "YES" if k.get("success") else f"{k.get('final_dist', 999):.3f}m"
        v_str = "YES" if v.get("success") else f"{v.get('final_dist', 999):.3f}m"

        if k.get("success"): kin_ok += 1
        if v.get("success"): vis_ok += 1

        if k.get("success") and v.get("success"):
            better = "BOTH"
        elif k.get("success"):
            better = "KIN"
        elif v.get("success"):
            better = "VISION"
        else:
            kd = k.get("final_dist", 999)
            vd = v.get("final_dist", 999)
            better = f"KIN Δ{vd-kd:+.2f}" if kd < vd else f"VIS Δ{kd-vd:+.2f}"

        print(f"  {case:<16}  {k_str:>10}  {v_str:>10}  {better:>8}")

    print(f"  {'-'*56}")
    print(f"  {'Success rate':<16}  {f'{kin_ok}/9':>10}  {f'{vis_ok}/9':>10}")
    print(f"{'='*62}\n")


def plot_comparison(kin_results, vis_results, out_path):
    cases = list(TEST_CASES.keys())
    n = len(cases)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(7 * cols, 5 * rows))
    fig.suptitle("KIN (blue) vs Vision (orange) — Same Cases, Same Targets",
                 fontsize=13, fontweight="bold")

    for idx, case in enumerate(cases):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        target = np.array(TEST_CASES[case]["target"])
        start  = np.array(TEST_CASES[case]["drone_pos"])

        for data, color, label in [
            (kin_results.get(case), "steelblue",  "KIN"),
            (vis_results.get(case), "darkorange",  "Vision"),
        ]:
            if data is None:
                continue
            pos = data["positions"]
            if len(pos) > 1:
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                        color=color, lw=1.5, alpha=0.8, label=label)
            ax.scatter(*pos[-1], color=color, s=50, marker="x")

        ax.scatter(*start,  color="blue", s=60,  marker="o", label="Start", zorder=5)
        ax.scatter(*target, color="gold", s=120, marker="*", label="Target", zorder=5)

        kin_ok  = kin_results.get(case, {}).get("success", False)
        vis_ok  = vis_results.get(case, {}).get("success", False)
        kin_str = "✓" if kin_ok  else f"{kin_results.get(case,{}).get('final_dist',999):.2f}m"
        vis_str = "✓" if vis_ok  else f"{vis_results.get(case,{}).get('final_dist',999):.2f}m"
        ax.set_title(f"{case}\nKIN:{kin_str}  VIS:{vis_str}", fontsize=8)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(fontsize=5, loc="upper left")

        all_pts = [target.reshape(1, 3), start.reshape(1, 3)]
        for d in [kin_results.get(case), vis_results.get(case)]:
            if d and len(d["positions"]) > 0:
                all_pts.append(d["positions"])
        all_pts = np.vstack(all_pts)
        centre = all_pts.mean(axis=0)
        span   = max(np.ptp(all_pts, axis=0).max(), 0.5) * 0.65
        ax.set_xlim(centre[0]-span, centre[0]+span)
        ax.set_ylim(centre[1]-span, centre[1]+span)
        ax.set_zlim(max(0, centre[2]-span), centre[2]+span)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Comparison trajectories -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="KIN vs Vision comparison benchmark")
    p.add_argument("--kin-model",    default="./models_nav_sac/best_model",
                   help="Path to KIN model (default: golden 8/9 SAC)")
    p.add_argument("--vision-model", default="./models_vision_nav/best_model",
                   help="Path to vision model")
    p.add_argument("--no-gui",  action="store_true")
    p.add_argument("--out-dir", default="./benchmark_results/kin_vs_vision/")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gui = not args.no_gui

    print(f"\nLoading KIN model:    {args.kin_model}")
    kin_model = SAC.load(args.kin_model, device="cpu")

    print(f"Loading Vision model: {args.vision_model}")
    vis_model = SAC.load(args.vision_model, device="cpu")

    kin_results, vis_results = {}, {}

    print("\nRunning KIN cases...")
    for name, cfg in TEST_CASES.items():
        print(f"  {name} ...", end=" ", flush=True)
        kin_results[name] = run_kin_case(kin_model, cfg, gui=gui)
        print("YES" if kin_results[name]["success"] else f"{kin_results[name]['final_dist']:.3f}m")

    print("\nRunning Vision cases...")
    for name, cfg in TEST_CASES.items():
        print(f"  {name} ...", end=" ", flush=True)
        vis_results[name] = run_vision_case(vis_model, cfg, gui=gui)
        print("YES" if vis_results[name]["success"] else f"{vis_results[name]['final_dist']:.3f}m")

    print_comparison_table(kin_results, vis_results)

    plot_comparison(kin_results, vis_results,
                    os.path.join(args.out_dir, "kin_vs_vision_trajectories.png"))

    # Save JSON
    slim = {}
    for case in TEST_CASES:
        slim[case] = {
            "kin":    {"success": kin_results[case]["success"],
                       "final_dist": round(kin_results[case]["final_dist"], 4)},
            "vision": {"success": vis_results.get(case, {}).get("success", False),
                       "final_dist": round(vis_results.get(case, {}).get("final_dist", 999), 4)},
        }
    json_path = os.path.join(args.out_dir, "comparison.json")
    with open(json_path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"[json] Results -> {json_path}")


if __name__ == "__main__":
    main()
