"""Quick smoke test for HoverEnv."""
import numpy as np
from envs.mujoco.hover_env import HoverEnv

env = HoverEnv()
obs, _ = env.reset()
print(f"obs shape: {obs.shape}  space: {env.observation_space}")
print(f"act space: {env.action_space}")
print(f"obs sample: {obs}")

total_r = 0
for i in range(50):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    total_r += r
    if term or trunc:
        obs, _ = env.reset()

print(f"50 steps OK, total_r={total_r:.3f}")
print(f"last info: {info}")

# Camera test
frame = env.get_drone_cam_frame()
print(f"drone cam frame shape: {frame.shape}  dtype={frame.dtype}  min={frame.min()} max={frame.max()}")
env.close()
print("PASS")
