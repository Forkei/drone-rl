"""
MuJoCo VisionNavEnv — camera-only obs for vision navigation.

Inherits NavEnv (same reward, same curriculum support).
Observation: drone_cam RGB frame (3, H, W) uint8, channel-first.
No KIN state in obs — policy must navigate by sight.

Target: red sphere (rgba="1 0 0 0.8", radius=0.2m) in quadrotor.xml.
Camera: pos=(0,0,0), euler=(0,15,0), fovy=90° — 15° down tilt, wide FOV.

Reward additions vs NavEnv:
  +red_pixel_bonus: up to +1.0 when red pixels fill frame
    → bootstraps visual attention before distance reward kicks in

Usage:
  env = VisionNavEnv(target_range=0.3)
  obs, _ = env.reset()   # obs.shape == (3, 32, 48)

For training (DummyVecEnv — OpenGL context conflicts SubprocVecEnv):
  from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
  env = DummyVecEnv([lambda: VisionNavEnv(target_range=0.3)])
  env = VecFrameStack(env, n_stack=4)   # obs: (12, 32, 48)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from envs.mujoco.nav_env import NavEnv

IMG_H = 32
IMG_W = 48


class VisionNavEnv(NavEnv):
    """Navigation env with camera-only observation (no privileged KIN state)."""

    def __init__(
        self,
        target_range: float = 0.3,
        render_mode:  str   = None,
        img_h:        int   = IMG_H,
        img_w:        int   = IMG_W,
    ):
        self._img_h = img_h
        self._img_w = img_w
        super().__init__(target_range=target_range, render_mode=render_mode)

        # Override observation space: (3, H, W) uint8, channel-first (SB3 CNN convention)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, img_h, img_w),
            dtype=np.uint8,
        )

        # Camera renderer (lazy-init to avoid OpenGL context issues at import time)
        self._vis_renderer = None
        self._last_frame   = None   # cache for red pixel bonus

    # ── camera ───────────────────────────────────────────────────────────────

    def _get_cam_frame(self) -> np.ndarray:
        """Returns (3, H, W) uint8 channel-first frame from drone_cam."""
        if self._vis_renderer is None:
            self._vis_renderer = mujoco.Renderer(
                self.model, height=self._img_h, width=self._img_w
            )
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "drone_cam"
        )
        self._vis_renderer.update_scene(self.data, camera=cam_id)
        frame = self._vis_renderer.render()          # (H, W, 3) uint8
        self._last_frame = frame
        return np.transpose(frame, (2, 0, 1))        # → (3, H, W)

    def _red_pixel_bonus(self, frame: np.ndarray) -> float:
        """Bonus proportional to red pixels in frame. Caps at 1.0.
        frame: (H, W, 3) uint8 channel-last.
        """
        red_mask = (
            (frame[:, :, 0] > 150) &
            (frame[:, :, 1] < 80)  &
            (frame[:, :, 2] < 80)
        )
        red_fraction = red_mask.sum() / red_mask.size
        return float(min(red_fraction * 5.0, 1.0))

    # ── gymnasium API overrides ───────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        obs = self._get_cam_frame()
        return obs, info

    def step(self, action: np.ndarray):
        _, reward, terminated, truncated, info = super().step(action)
        obs = self._get_cam_frame()
        # Add red pixel bonus using the freshly rendered frame (channel-last)
        red_mask = (
            (self._last_frame[:, :, 0] > 150) &
            (self._last_frame[:, :, 1] < 80)  &
            (self._last_frame[:, :, 2] < 80)
        )
        red_fraction = float(red_mask.sum() / red_mask.size)
        bonus = float(min(red_fraction * 5.0, 1.0))
        reward += bonus
        info["red_fraction"] = red_fraction
        info["red_bonus"]    = round(bonus, 4)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._vis_renderer is not None:
            self._vis_renderer.close()
            self._vis_renderer = None
        super().close()
