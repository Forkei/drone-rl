"""
check_vision.py — Vision training monitor with milestone actions.

Milestones:
  10k:   First sign of life (mean_dist < 1.3m)
  100k:  Save sample obs frame (what is the CNN actually seeing?)
  500k:  3-case mini-benchmark (approach_+X, approach_-Y, far_1.0m)
  >1M + success>15%: flag to expand target_range
"""

import os, sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR   = './logs_vision_nav/SAC_3'
MODEL_DIR = './models_vision_nav'

# ── helpers ──────────────────────────────────────────────────────────────────

def latest(ea, tag):
    try:
        evs = ea.Scalars(tag)
        return evs[-1].value, evs[-1].step
    except Exception:
        return None, None


def save_obs_frame(step):
    """Instantiate env, reset, capture one frame, save as PNG."""
    try:
        import numpy as np
        from PIL import Image
        from envs.vision_nav_aviary import VisionNavAviary

        env = VisionNavAviary(target_range=1.0, img_wh=(48, 32), gui=False)
        obs, _ = env.reset()
        env.close()

        # obs is (3, H, W) channel-first uint8 — convert to (H, W, 3) for PIL
        frame = obs.transpose(1, 2, 0)
        out = f"./benchmark_results/vision_obs_step{step//1000}k.png"
        os.makedirs('./benchmark_results', exist_ok=True)
        Image.fromarray(frame, 'RGB').save(out)
        print(f"  [frame] Obs frame saved -> {out}")
        return out
    except Exception as e:
        print(f"  [frame] ERROR: {e}")
        return None


def run_mini_benchmark(model_path):
    """3-case mini-benchmark for early directional bias check."""
    try:
        from stable_baselines3 import SAC
        from envs.vision_nav_aviary import VisionNavAviary
        import numpy as np

        cases = {
            "approach_+X": {"drone_pos": [1.0, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
            "approach_-Y": {"drone_pos": [0.0,-1.0, 1.0], "target": [0.0, 0.0, 1.0]},
            "far_1.0m":    {"drone_pos": [1.0, 0.0, 1.0], "target": [0.0, 0.0, 1.0]},
        }

        model = SAC.load(model_path, device="cpu")
        results = {}
        for name, cfg in cases.items():
            drone_pos = np.array(cfg["drone_pos"])
            target    = np.array(cfg["target"])
            env = VisionNavAviary(target_range=0.5, img_wh=(48, 32), home_pos=drone_pos, gui=False)
            env.TARGET_POS = target.copy()
            obs, _ = env.reset()
            env.TARGET_POS = target.copy()

            done = False
            dists = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                dists.append(info["dist_to_target"])
                done = terminated or truncated
            env.close()

            results[name] = {
                "success":    info.get("success", False),
                "final_dist": dists[-1] if dists else 999,
                "min_dist":   min(dists) if dists else 999,
            }
            status = "YES" if results[name]["success"] else f"{results[name]['final_dist']:.3f}m"
            print(f"    {name}: {status}")

        return results
    except Exception as e:
        print(f"  [mini-bench] ERROR: {e}")
        return None


# ── state tracking (written to disk so it persists across cron calls) ────────

STATE_FILE = './logs_vision_nav/.monitor_state'

def load_state():
    state = {"frame_saved": False, "mini_bench_done": False,
             "milestone1": False, "milestone2": False, "milestone3": False}
    if os.path.exists(STATE_FILE):
        import json
        with open(STATE_FILE) as f:
            state.update(json.load(f))
    return state

def save_state(state):
    import json
    os.makedirs('./logs_vision_nav', exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


# ── main ─────────────────────────────────────────────────────────────────────

if not os.path.exists(LOG_DIR):
    print("Vision log not found"); sys.exit(0)

ea = EventAccumulator(LOG_DIR); ea.Reload()
avail = ea.Tags().get('scalars', [])

# Get step/fps
try:
    s = ea.Scalars('rollout/ep_rew_mean')
    first, last = s[0], s[-1]
    elapsed = last.wall_time - first.wall_time
    fps = (last.step - first.step) / elapsed if elapsed > 0 else 0
    step = last.step
    remaining = 2_000_000 - step
    eta_hr = remaining / fps / 3600 if fps > 0 else 999
    print(f"Vision: step={step:,}/2M  fps={fps:.1f}  ETA={eta_hr:.1f}h  rew={last.value:.3f}")
except Exception as e:
    print(f"No rollout data yet: {e}"); sys.exit(0)

mean_dist,    _ = latest(ea, 'vis/mean_final_dist')
success_rate, _ = latest(ea, 'vis/success_rate')
eval_reward,  _ = latest(ea, 'vis/eval_reward')
vis_fps,      _ = latest(ea, 'vis/fps')

if mean_dist    is not None: print(f"  vis/mean_final_dist: {mean_dist:.4f}m")
if success_rate is not None: print(f"  vis/success_rate:    {success_rate:.4f}")
if eval_reward  is not None: print(f"  vis/eval_reward:     {eval_reward:.4f}")
if vis_fps      is not None: print(f"  vis/fps:             {vis_fps:.1f}")

state = load_state()
actions_taken = []

# ── milestone 1: first sign of life ─────────────────────────────────────────
if mean_dist is not None and mean_dist < 1.3 and not state["milestone1"]:
    state["milestone1"] = True
    actions_taken.append(f"MILESTONE 1: mean_dist={mean_dist:.3f}m < 1.3m at step {step:,}")

# ── milestone 2: first successes ─────────────────────────────────────────────
if success_rate is not None and success_rate > 0.01 and not state["milestone2"]:
    state["milestone2"] = True
    actions_taken.append(f"MILESTONE 2: success_rate={success_rate:.3f} > 1% at step {step:,}")

# ── milestone 3: real progress ────────────────────────────────────────────────
if success_rate is not None and success_rate > 0.05 and not state["milestone3"]:
    state["milestone3"] = True
    actions_taken.append(f"MILESTONE 3: success_rate={success_rate:.3f} > 5% at step {step:,}")

# ── 100k: save obs frame ──────────────────────────────────────────────────────
if step >= 100_000 and not state["frame_saved"]:
    print(f"\n[100k] Saving sample obs frame...")
    frame_path = save_obs_frame(step)
    state["frame_saved"] = True
    if frame_path:
        actions_taken.append(f"OBS FRAME saved at step {step:,} -> {frame_path}")

# ── 500k: mini-benchmark ──────────────────────────────────────────────────────
if step >= 500_000 and not state["mini_bench_done"]:
    # Find best available model
    candidates = [
        f"{MODEL_DIR}/best_model.zip",
        f"{MODEL_DIR}/vis_nav_500000_steps.zip",
        f"{MODEL_DIR}/vis_nav_400000_steps.zip",
    ]
    model_path = next((p[:-4] for p in candidates if os.path.exists(p)), None)
    if model_path:
        print(f"\n[500k] Running 3-case mini-benchmark on {model_path}...")
        results = run_mini_benchmark(model_path)
        state["mini_bench_done"] = True
        if results:
            summary = ", ".join(
                f"{k}: {'YES' if v['success'] else str(round(v['final_dist'], 3)) + 'm'}"
                for k, v in results.items()
            )
            actions_taken.append(f"MINI-BENCH at step {step:,}: {summary}")
    else:
        print("[500k] No model checkpoint found yet for mini-benchmark")

save_state(state)

# ── write message if anything notable happened ────────────────────────────────
if actions_taken:
    import subprocess, datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    msg_path = f"./messages/message{ts}.md"
    os.makedirs('./messages', exist_ok=True)

    expand_flag = (step > 1_000_000 and success_rate is not None and success_rate > 0.15)

    lines = [
        f"# Vision Training Update — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Actions this check",
        *[f"- {a}" for a in actions_taken],
        "",
        "## Current metrics",
        f"| step | fps | ETA | mean_dist | success_rate | eval_reward |",
        f"|---|---|---|---|---|---|",
        f"| {step:,} | {fps:.1f} | {eta_hr:.1f}h | {f'{mean_dist:.4f}' if mean_dist else 'n/a'} | {f'{success_rate:.4f}' if success_rate else 'n/a'} | {f'{eval_reward:.4f}' if eval_reward and eval_reward != float('-inf') else 'n/a'} |",
        "",
        f"## Expand target_range to 2.0m?",
        "YES — step > 1M and success_rate > 15%" if expand_flag else f"No — step={step:,}, success_rate={f'{success_rate:.3f}' if success_rate else '0.000'}",
        "",
        "## Recommendation",
        "Let it run. No changes needed." if not expand_flag else
        "Consider relaunching with target_range=2.0m.",
    ]

    with open(msg_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n[msg] Written -> {msg_path}")
