"""
9-case benchmark for MuJoCo NavEnv — mirrors PyBullet benchmark.py.

Usage:
  python evaluation/benchmark_mujoco.py
  python evaluation/benchmark_mujoco.py --model models_nav_mujoco/nav_final
  python evaluation/benchmark_mujoco.py --runs 5
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from stable_baselines3 import SAC
from envs.mujoco.nav_env import NavEnv

# ── 9 fixed cases ─────────────────────────────────────────────────────────────
CASES = {
    "approach_+X": {"start": [1.5,  0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "approach_-X": {"start": [-1.5, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "approach_+Y": {"start": [0.0,  1.5, 1.0], "target": [0.0, 0.0, 1.0]},
    "approach_-Y": {"start": [0.0, -1.5, 1.0], "target": [0.0, 0.0, 1.0]},
    "low_target":  {"start": [0.0,  0.0, 1.0], "target": [0.0, 0.0, 0.3]},
    "high_target": {"start": [0.0,  0.0, 1.0], "target": [0.0, 0.0, 1.8]},
    "near_0.3m":   {"start": [0.3,  0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "far_1.8m":    {"start": [1.8,  0.0, 1.0], "target": [0.0, 0.0, 1.0]},
    "diagonal":    {"start": [1.0,  1.0, 1.5], "target": [0.0, 0.0, 1.0]},
}


def set_state(env: NavEnv, start: list, target: list):
    """Force-set drone position and target, bypassing random reset."""
    mujoco.mj_resetData(env.model, env.data)
    start = np.array(start, dtype=float)
    target = np.array(target, dtype=float)

    env.data.qpos[:3] = start
    env.data.qpos[3:7] = [1, 0, 0, 0]   # upright
    env.data.qvel[:] = 0.0

    env._set_target(target)
    mujoco.mj_forward(env.model, env.data)

    env._prev_dist  = float(np.linalg.norm(target - start))
    env._step_count = 0
    env._consec_goal = 0

    return env._get_obs()


def run_case(env: NavEnv, model, start: list, target: list, n_runs: int = 3):
    results = []
    for _ in range(n_runs):
        obs = set_state(env, start, target)
        total_reward = 0.0
        steps = 0
        success = False
        crashed = False
        final_dist = float(np.linalg.norm(np.array(target) - np.array(start)))

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += r
            steps += 1
            final_dist = info["dist_to_target"]
            if info["success"]:
                success = True
            if info["crashed"]:
                crashed = True
            if terminated or truncated:
                break

        results.append({
            "dist":    round(final_dist, 4),
            "success": success,
            "crashed": crashed,
            "steps":   steps,
            "reward":  round(total_reward, 3),
        })
    return results


def summarise(results: list) -> dict:
    dists   = [r["dist"]    for r in results]
    rewards = [r["reward"]  for r in results]
    steps_l = [r["steps"]   for r in results]
    n_ok    = sum(r["success"] for r in results)
    return {
        "success": f"{n_ok}/{len(results)}",
        "dist":    f"{np.mean(dists):.4f} ± {np.std(dists):.4f}",
        "steps":   f"{np.mean(steps_l):.0f}",
        "reward":  f"{np.mean(rewards):.2f}",
        "pass":    n_ok == len(results),
    }


def benchmark(model_path: str, n_runs: int = 3):
    print(f"\nLoading: {model_path}")
    env   = NavEnv(target_range=2.0)
    model = SAC.load(model_path, device="cpu")

    print(f"\n{'Case':<15} {'Success':>9} {'Dist (m)':>16} {'Steps':>7} {'Reward':>8}")
    print("-" * 62)

    all_pass = []
    for name, case in CASES.items():
        results = run_case(env, model, case["start"], case["target"], n_runs)
        s = summarise(results)
        marker = "✓" if s["pass"] else "✗"
        print(f"{name:<15} {marker} {s['success']:>7}  {s['dist']:>16}  {s['steps']:>5}  {s['reward']:>8}")
        all_pass.append(s["pass"])

    total = sum(all_pass)
    print("-" * 62)
    print(f"{'TOTAL':<15}   {total}/9")

    env.close()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",  type=int, default=3)
    args = parser.parse_args()

    models = {
        "best_model": "models_nav_mujoco/best_model",
        "nav_final":  "models_nav_mujoco/nav_final",
    }

    scores = {}
    for label, path in models.items():
        if not os.path.exists(path + ".zip"):
            print(f"[SKIP] {path}.zip not found")
            continue
        print(f"\n{'='*62}")
        print(f"  {label}")
        print(f"{'='*62}")
        scores[label] = benchmark(path, n_runs=args.runs)

    print(f"\n{'='*62}")
    print("SUMMARY")
    for label, score in scores.items():
        print(f"  {label}: {score}/9")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
