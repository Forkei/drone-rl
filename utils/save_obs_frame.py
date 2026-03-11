"""Diagnostic: is the orange sphere visible in the camera frame?

Correct geometry:
  - Drone spawns at home_pos=[-1,0,1] with default yaw=0 (facing +X)
  - After reset(), move sphere via PyBullet directly to [0,0,1] (1.0m ahead in +X)
  - Take one step to capture obs with sphere in field of view

Tests both radius=0.08m and radius=0.15m for comparison.
"""
import numpy as np
from PIL import Image
import os
import pybullet as p
from envs.vision_nav_aviary import VisionNavAviary

os.makedirs('./benchmark_results', exist_ok=True)


def orange_pixels(frame):
    r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    return int(np.sum((r > 180) & (g < 120) & (b < 120)))


def save_scaled(frame, path, scale=8):
    h, w = frame.shape[:2]
    Image.fromarray(frame, 'RGB').resize((w*scale, h*scale), Image.NEAREST).save(path)


def capture_frame(sphere_radius_m, label):
    import envs.vision_nav_aviary as vna
    original = vna._SPHERE_RADIUS
    vna._SPHERE_RADIUS = sphere_radius_m

    # Drone at [-1,0,1] facing +X (yaw=0), target_range doesn't matter
    env = VisionNavAviary(target_range=1.0, home_pos=np.array([-1.0, 0.0, 1.0]),
                          img_wh=(48, 32), gui=False)
    obs, _ = env.reset()

    # Place sphere at [0,0,1] — 1.0m directly ahead of drone along +X
    sphere_pos = [0.0, 0.0, 1.0]
    p.resetBasePositionAndOrientation(
        env._sphere_id, sphere_pos, [0,0,0,1], physicsClientId=env.CLIENT)
    env.TARGET_POS = np.array(sphere_pos)

    # Step once; obs is captured after sphere is in position
    obs, _, _, _, info = env.step(np.array([[0.0, 0.0, 1.0]]))
    env.close()

    vna._SPHERE_RADIUS = original

    frame = obs.transpose(1, 2, 0)
    n_or = orange_pixels(frame)
    path = f'./benchmark_results/obs_sphere_{label}_x8.png'
    save_scaled(frame, path)

    print(f"=== radius={sphere_radius_m}m at 1.0m distance ===")
    print(f"  Orange pixels: {n_or} / {frame.shape[0]*frame.shape[1]}")
    print(f"  Channel means: R={frame[:,:,0].mean():.1f}  G={frame[:,:,1].mean():.1f}  B={frame[:,:,2].mean():.1f}")
    print(f"  R max={frame[:,:,0].max()}  G min={frame[:,:,1].min()}  B min={frame[:,:,2].min()}")
    print(f"  Saved 8x -> {path}")
    print()
    return frame


frame_08 = capture_frame(0.08,  'r0.08')
frame_15 = capture_frame(0.15,  'r0.15')
frame_25 = capture_frame(0.25,  'r0.25')   # very large — sanity check

# Save a side-by-side comparison (3 panels stacked vertically, 8x scale)
panels = []
for f, lbl in [(frame_08,'r=0.08'), (frame_15,'r=0.15'), (frame_25,'r=0.25')]:
    h, w = f.shape[:2]
    img = np.array(Image.fromarray(f,'RGB').resize((w*8,h*8),Image.NEAREST))
    # Add label strip
    strip = np.full((20, w*8, 3), 40, dtype=np.uint8)
    panels.extend([img, strip])

comparison = np.vstack(panels[:-1])
Image.fromarray(comparison, 'RGB').save('./benchmark_results/obs_sphere_comparison.png')
print("Side-by-side saved -> ./benchmark_results/obs_sphere_comparison.png")
