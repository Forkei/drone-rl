"""Test drone camera with updated lighting."""
from envs.mujoco.hover_env import HoverEnv
from PIL import Image

env = HoverEnv()
obs, _ = env.reset()
frame = env.get_drone_cam_frame(height=32, width=48)
env.close()

print(f"shape={frame.shape} dtype={frame.dtype} min={frame.min()} max={frame.max()} mean={frame.mean():.1f}")
Image.fromarray(frame, 'RGB').save('./benchmark_results/mujoco_drone_cam_lit.png')
print("Saved -> ./benchmark_results/mujoco_drone_cam_lit.png")
