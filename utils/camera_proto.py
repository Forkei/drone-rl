"""
camera_proto.py — Confirm gym-pybullet-drones RGB camera pipeline works.

Creates a HoverAviary with ObservationType.RGB, steps it once, saves the
raw RGBA frame to ./camera_proto_frame.png.  No training, no policy — just
confirms rendering is functional.

Camera details (from BaseAviary):
  - Resolution: 64×48 (W×H), RGBA uint8
  - FOV: 60°, forward-facing (drone body-X axis)
  - Capture rate: 24 fps (every 10 PyBullet steps at 240 Hz)

Also calls _getDroneImages() directly to confirm the util works standalone.

Usage:
    conda run -n drone-rl python camera_proto.py
"""

import os
import numpy as np
from PIL import Image

from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


def run():
    print("\n[camera_proto] Creating HoverAviary with ObservationType.RGB ...")
    # ctrl_freq=24 is required: IMG_CAPTURE_FREQ = PYB_FREQ/24 = 10 steps,
    # PYB_STEPS_PER_CTRL = PYB_FREQ/CTRL_FREQ = 240/24 = 10 → 10%10=0 ✓
    # Default ctrl_freq=30 gives PYB_STEPS_PER_CTRL=8 → 10%8≠0 (error)
    env = HoverAviary(
        obs=ObservationType.RGB,
        act=ActionType.RPM,
        gui=False,
        record=False,
        ctrl_freq=24,
    )

    print(f"[camera_proto] Observation space: {env.observation_space}")
    print(f"[camera_proto] IMG_RES (W×H): {env.IMG_RES}")          # [64, 48]
    print(f"[camera_proto] IMG_CAPTURE_FREQ: {env.IMG_CAPTURE_FREQ} steps")

    obs, info = env.reset()
    print(f"[camera_proto] obs shape after reset: {obs.shape}")    # (1, 48, 64, 4)
    print(f"[camera_proto] obs dtype: {obs.dtype}")

    # ── take enough steps so at least one camera frame is captured ──────────
    # IMG_CAPTURE_FREQ = PYB_FREQ / IMG_FRAME_PER_SEC = 240 / 24 = 10
    # We need step_counter % IMG_CAPTURE_FREQ == 0 INSIDE _computeObs
    # reset sets step_counter=0; the first call to _computeObs is at step 1
    # So step once — that triggers capture at the first PYB_STEPS_PER_CTRL boundary
    action = np.zeros((1, 4))          # zero RPM (drone falls, but we just want a frame)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"[camera_proto] obs shape after step: {obs.shape}")

    # ── get a frame directly via _getDroneImages() ─────────────────────────
    rgb, dep, seg = env._getDroneImages(0, segmentation=False)
    print(f"[camera_proto] _getDroneImages rgb shape: {rgb.shape}  dtype: {rgb.dtype}")
    print(f"[camera_proto] rgb min={rgb.min()} max={rgb.max()}")

    # ── save the RGBA frame ────────────────────────────────────────────────
    out_path = "./camera_proto_frame.png"
    img = Image.fromarray(rgb.astype(np.uint8), "RGBA")
    img.save(out_path)
    print(f"[camera_proto] Frame saved -> {os.path.abspath(out_path)}")

    # ── also save the observation frame (first drone) ────────────────────────
    # NOTE: SB3 returns obs as float32 despite obs_space dtype=uint8
    obs_frame = obs[0].astype(np.uint8)  # (48, 64, 4)
    img2 = Image.fromarray(obs_frame, "RGBA")
    img2.save("./camera_proto_obs.png")
    print(f"[camera_proto] Obs frame saved -> {os.path.abspath('./camera_proto_obs.png')}")

    # ── report obs-space stats ─────────────────────────────────────────────
    print("\n[camera_proto] Summary:")
    print(f"  Camera resolution : {env.IMG_RES[0]}×{env.IMG_RES[1]} px (W×H)")
    print(f"  Channels          : 4 (RGBA)")
    print(f"  Obs tensor shape  : {obs.shape}  (NUM_DRONES, H, W, C)")
    print(f"  Capture freq      : {env.IMG_FRAME_PER_SEC} fps")
    print(f"  Forward-facing    : body-X axis, 60° FOV")
    print(f"  Landmarks present : block, cube_small, duck, teddy (±1m XY)")

    env.close()
    print("\n[camera_proto] Pipeline confirmed OK.\n")


if __name__ == "__main__":
    run()
