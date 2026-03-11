"""
MuJoCo VisionNavEnv — camera-only obs for vision navigation.

Inherits NavEnv (same reward, same curriculum support).
Observation: drone_cam RGB frame (3, H, W) uint8, channel-first.
No KIN state in obs — policy must navigate by sight.

Target is red sphere (rgba="1 0 0 0.5") in quadrotor.xml.
Camera: forward-looking, tilted -20° down, fovy=70°, pos=(0.04, 0, 0.01).

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
        return np.transpose(frame, (2, 0, 1))        # → (3, H, W)

    # ── gymnasium API overrides ───────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        obs = self._get_cam_frame()
        return obs, info

    def step(self, action: np.ndarray):
        _, reward, terminated, truncated, info = super().step(action)
        obs = self._get_cam_frame()
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._vis_renderer is not None:
            self._vis_renderer.close()
            self._vis_renderer = None
        super().close()
