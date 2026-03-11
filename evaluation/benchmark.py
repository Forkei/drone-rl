"""
benchmark.py -- 9-case nav policy benchmark with JSON output.

Runs the standard 9 test cases on one or more saved models and produces:
  - Printed comparison table
  - JSON results file  (--out-dir / results.json)
  - 3D trajectory plots (--out-dir / <model_name>_trajectories.png)
  - Distance curves     (--out-dir / <model_name>_dist_curves.png)

Supports PPO and SAC models (auto-detected).

Usage:
    # Single model, headless
    python benchmark.py --model ./models_nav/best_model --no-gui

    # Compare multiple models
    python benchmark.py --no-gui \\
        --model ./models_nav/best_model \\
                ./models_nav_entropy/ppo_nav_entropy_final \\
                ./models_nav_stage4/ppo_stage4_final

    # Custom output directory
    python benchmark.py --model ./models_nav/best_model --no-gui --out-dir ./results/
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from envs.nav_aviary import NavAviary


SUCCESS_THRESHOLD = 0.10   # metres


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
    "far_3.0m":    {"drone_pos": [3.0,  0.0, 1.0], "target": [0.0, 0.0, 1.0]},
}

# Standard 9-case subset (excludes far_3.0m)
STANDARD_9 = {k: v for k, v in TEST_CASES.items() if k != "far_3.0m"}


# -- model loading -------------------------------------------------------------

def load_model(path: str):
    """Load PPO or SAC model, auto-detect algorithm."""
    if not path.endswith(".zip"):
        path_zip = path + ".zip"
    else:
        path_zip = path
        path = path[:-4]

    if not os.path.isfile(path_zip):
        raise FileNotFoundError(f"Model not found: {path_zip}")

    for AlgoCls in (PPO, SAC):
        try:
            model = AlgoCls.load(path)
            algo = AlgoCls.__name__
            return model, algo
        except Exception:
            continue
    raise ValueError(f"Could not load {path_zip} as PPO or SAC")


# -- episode runner ------------------------------------------------------------

def run_case(model, case_cfg: dict, gui: bool = False) -> dict:
    drone_pos = np.array(case_cfg["drone_pos"], dtype=np.float64)
    target    = np.array(case_cfg["target"],    dtype=np.float64)

    env = NavAviary(
        target_range=0.5,
        home_pos=drone_pos,
        gui=gui,
        record=False,
    )
    env.TARGET_POS = target.copy()
    obs, _ = env.reset()
    env.TARGET_POS = target.copy()

    positions, rewards, dists = [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = env._getDroneStateVector(0)
        positions.append(state[0:3].copy())
        rewards.append(float(reward))
        dists.append(float(info["dist_to_target"]))

        if gui:
            time.sleep(1 / 30)

    env.close()

    positions = np.array(positions)
    min_dist  = float(np.min(dists)) if dists else float("inf")
    return {
        "positions":    positions.tolist(),
        "dists":        dists,
        "target":       target.tolist(),
        "drone_start":  drone_pos.tolist(),
        "success":      bool(info.get("success", False)),
        "final_dist":   dists[-1] if dists else float("inf"),
        "min_dist":     min_dist,
        "n_steps":      len(rewards),
        "total_reward": float(np.sum(rewards)),
    }


def run_benchmark(model, cases: dict = None, gui: bool = False) -> dict:
    cases = cases or STANDARD_9
    results = {}
    for name, cfg in cases.items():
        print(f"    {name} ...", end=" ", flush=True)
        data = run_case(model, cfg, gui=gui)
        results[name] = data
        status = "OK" if data["success"] else f"{data['final_dist']:.3f}m"
        print(status)
    return results


def run_robustness(model, n_runs: int, cases: dict = None, gui: bool = False) -> dict:
    """Run benchmark n_runs times and aggregate results."""
    cases = cases or STANDARD_9
    all_runs = []
    for i in range(n_runs):
        print(f"\n  --- Run {i+1}/{n_runs} ---")
        all_runs.append(run_benchmark(model, cases=cases, gui=gui))

    # Aggregate: success_rate per case, final_dist mean/std
    aggregated = {}
    for name in cases:
        successes = [r[name]["success"] for r in all_runs]
        dists     = [r[name]["final_dist"] for r in all_runs]
        aggregated[name] = {
            "success_rate": sum(successes) / n_runs,
            "successes":    successes,
            "mean_dist":    float(np.mean(dists)),
            "std_dist":     float(np.std(dists)),
            "final_dists":  dists,
        }
    return aggregated


# -- reporting -----------------------------------------------------------------

def print_table(model_results: dict):
    """model_results: {model_label: {case_name: result_dict}}"""
    # Infer case names from first model's results (handles extended/standard sets)
    first = next(iter(model_results.values()))
    case_names = list(first.keys())
    labels     = list(model_results.keys())

    col_w = 11
    header = f"  {'Case':<16}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    sep    = "  " + "-" * (16 + col_w * len(labels))

    print(f"\n{'='*60}")
    print("  9-Case Benchmark Summary")
    print(f"{'='*60}")
    print(header)
    print(sep)

    for case in case_names:
        row = f"  {case:<16}"
        for lb in labels:
            data = model_results[lb].get(case)
            if data is None:
                row += f"{'N/A':>{col_w}}"
            elif data["success"]:
                row += f"{'YES':>{col_w}}"
            else:
                row += f"{data['final_dist']:>{col_w-1}.3f}m"
        print(row)

    print(sep)
    success_row = f"  {'Success rate':<16}"
    for lb in labels:
        res = model_results[lb]
        n_ok = sum(1 for d in res.values() if d.get("success", False))
        success_row += f"{f'{n_ok}/{len(res)}':>{col_w}}"
    print(success_row)
    print(f"{'='*60}\n")


def plot_trajectories(results: dict, label: str, out_path: str):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig  = plt.figure(figsize=(6 * cols, 5 * rows))
    fig.suptitle(f"Trajectories: {label}", fontsize=13, fontweight="bold")

    for idx, (name, data) in enumerate(results.items()):
        pos    = np.array(data["positions"])
        target = np.array(data["target"])
        start  = np.array(data["drone_start"])

        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")

        if len(pos) > 1:
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(pos)))
            for i in range(len(pos) - 1):
                ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2],
                        color=colors[i], lw=1.5, alpha=0.8)

        ax.scatter(*pos[0],   color="blue", s=60,  marker="o", label="Start")
        ax.scatter(*pos[-1],  color="red",  s=60,  marker="x", label="End")
        ax.scatter(*target,   color="gold", s=120, marker="*", label="Target")

        ok    = data["success"]
        color = "green" if ok else "red"
        status = "SUCCESS" if ok else f"dist={data['final_dist']:.2f}m"
        ax.set_title(f"{name}\n{status}", fontsize=9, color=color)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(fontsize=6)

        all_pts = np.vstack([pos, target.reshape(1, 3), start.reshape(1, 3)])
        centre  = all_pts.mean(axis=0)
        span    = max(np.ptp(all_pts, axis=0).max(), 0.5) * 0.6
        ax.set_xlim(centre[0] - span, centre[0] + span)
        ax.set_ylim(centre[1] - span, centre[1] + span)
        ax.set_zlim(max(0, centre[2] - span), centre[2] + span)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Trajectories -> {out_path}")


def plot_dist_curves(results: dict, label: str, out_path: str):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    fig.suptitle(f"Distance Curves: {label}", fontsize=12)
    axes = np.array(axes).flatten()

    for idx, (name, data) in enumerate(results.items()):
        ax    = axes[idx]
        dists = data["dists"]
        ax.plot(dists, lw=1.5, color="steelblue")
        ax.axhline(SUCCESS_THRESHOLD, color="green",  ls="--", lw=1, label="0.10m goal")
        ax.axhline(0.15,              color="orange", ls=":",  lw=1, label="0.15m warm")
        ax.fill_between(range(len(dists)), 0, dists, alpha=0.1, color="steelblue")
        color = "green" if data["success"] else "gray"
        ax.set_title(name, fontsize=8, color=color)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Dist (m)", fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Dist curves  -> {out_path}")


# -- main ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="9-case nav benchmark")
    p.add_argument("--model",    nargs="+", required=True,
                   help="Path(s) to model(s) without .zip extension")
    p.add_argument("--label",    nargs="+", default=None,
                   help="Short label(s) for each model (default: basename)")
    p.add_argument("--no-gui",   action="store_true", help="Headless mode")
    p.add_argument("--no-plots", action="store_true", help="Skip trajectory plots")
    p.add_argument("--out-dir",  default="./benchmark_results",
                   help="Directory for JSON + plots")
    p.add_argument("--runs",     type=int, default=1,
                   help="Number of runs for robustness check (default: 1)")
    p.add_argument("--extended", action="store_true",
                   help="Include far_3.0m test case")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gui = not args.no_gui
    cases = TEST_CASES if args.extended else STANDARD_9

    labels = args.label or [os.path.basename(m) for m in args.model]
    if len(labels) != len(args.model):
        labels = [os.path.basename(m) for m in args.model]

    all_results = {}

    for model_path, label in zip(args.model, labels):
        print(f"\n[{label}] Loading model...")
        try:
            model, algo = load_model(model_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print(f"  Algorithm: {algo}")

        if args.runs > 1:
            print(f"  Robustness check: {args.runs} runs x {len(cases)} cases")
            rob = run_robustness(model, args.runs, cases=cases, gui=gui)

            # Print robustness table
            print(f"\n{'='*60}")
            print(f"  Robustness Report: {label} ({args.runs} runs)")
            print(f"{'='*60}")
            total_successes = 0
            for name, d in rob.items():
                rate = d["success_rate"]
                total_successes += rate
                bar  = "✓" * int(rate * args.runs) + "✗" * (args.runs - int(rate * args.runs))
                print(f"  {name:<16}  {rate*100:5.1f}%  [{bar}]  "
                      f"dist={d['mean_dist']:.3f}±{d['std_dist']:.3f}m")
            print(f"{'='*60}")
            print(f"  Mean success rate: {total_successes/len(rob)*100:.1f}%")
            print(f"{'='*60}\n")

            # Save robustness JSON
            rob_path = os.path.join(args.out_dir, f"{label.replace('/', '_')}_robustness.json")
            with open(rob_path, "w") as f:
                json.dump(rob, f, indent=2)
            print(f"[INFO] Robustness JSON -> {rob_path}")
            # Use last run results for table comparison
            all_results[label] = {
                name: {
                    "success":     d["success_rate"] >= 0.6,
                    "final_dist":  d["mean_dist"],
                    "min_dist":    d["mean_dist"],
                    "n_steps":     0,
                    "total_reward": 0.0,
                }
                for name, d in rob.items()
            }
        else:
            print(f"  Running {len(cases)} test cases:")
            results = run_benchmark(model, cases=cases, gui=gui)
            all_results[label] = results

            # Save plots
            if not args.no_plots:
                safe_label = label.replace("/", "_").replace("\\", "_")
                plot_trajectories(results, label,
                                  os.path.join(args.out_dir, f"{safe_label}_trajectories.png"))
                plot_dist_curves(results, label,
                                 os.path.join(args.out_dir, f"{safe_label}_dist_curves.png"))

    # Print comparison table
    if all_results:
        print_table(all_results)

    # Save JSON
    json_path = os.path.join(args.out_dir, "results.json")
    slim = {}
    for lb, res in all_results.items():
        slim[lb] = {
            case: {
                "success":     d["success"],
                "final_dist":  round(d["final_dist"],  4),
                "min_dist":    round(d["min_dist"],    4),
                "n_steps":     d["n_steps"],
                "total_reward": round(d["total_reward"], 3),
            }
            for case, d in res.items()
        }
    with open(json_path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"[INFO] Results JSON -> {json_path}")


if __name__ == "__main__":
    main()
