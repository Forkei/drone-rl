"""
Upward-thrust diagnostic on models_nav_sac/best_model.zip.

Drone at [0,0,1.0], target at [0,0,1.8] (pure vertical, 0.8m above).
Runs 50 steps deterministically. Logs z, action_z, dist every step.

Answers:
  A) Trying to go up but PID not responding (z increases, then stalls)
  B) Not trying to go up at all (action_z small/negative throughout)
  C) Going up briefly then retreating (z peaks then drops)
"""

import numpy as np
from stable_baselines3 import SAC
from envs.nav_aviary import NavAviary

MODEL = "./models_nav_sac/best_model"
TARGET = np.array([0.0, 0.0, 1.8])
START  = np.array([0.0, 0.0, 1.0])

env = NavAviary(target_range=0.5, home_pos=START, gui=False)
env.TARGET_POS = TARGET.copy()
obs, _ = env.reset()
env.TARGET_POS = TARGET.copy()   # reset() re-samples, force it back

model = SAC.load(MODEL, device="cpu")

print(f"\nDiagnostic: drone at {START}, target at {TARGET}")
print(f"{'step':>4}  {'z':>6}  {'action_z':>9}  {'dist':>7}  {'vz':>7}")
print("-" * 42)

for step in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    state = env._getDroneStateVector(0)
    z      = state[2]
    vz     = state[8]   # linear velocity z
    act_z  = float(action[0, 2])   # PID z waypoint (in [-1,1])
    dist   = info["dist_to_target"]

    print(f"{step+1:>4}  {z:>6.3f}  {act_z:>+9.4f}  {dist:>7.4f}  {vz:>+7.4f}")

    if terminated or truncated:
        print(f"  [Episode ended at step {step+1}: {'success' if info['success'] else 'timeout/crash'}]")
        break

env.close()

# Summary
print("\nInterpretation:")
print("  action_z > 0 → PID commanded upward waypoint")
print("  action_z < 0 → PID commanded downward waypoint")
print("  If action_z consistently negative → policy is NOT trying to climb (case B)")
print("  If action_z positive but z plateaus → PID clamp or control issue (case A)")
print("  If z rises then falls → policy retreats (case C)")
