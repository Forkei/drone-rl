"""
MovingTargetAviary — extends VisionNavAviary with a moving sphere target.

The sphere moves in the XY plane each step at a constant speed and slowly-
rotating direction, clamped to a safe volume.

Reward structure (follow-me behavior):
  The goal is to maintain ~1.5m distance from the moving sphere AND keep it
  centered in frame. Neither pure proximity nor pure centering alone produces
  good follow-me behavior.

  dist_error  = abs(dist_to_target - ideal_dist)   # 0 = perfect following distance
  reward = (prev_dist_error - dist_error) * dist_w  # potential-based distance maintenance
         + centering_bonus                           # 0.3 when centered, 0 at max_angle
         - time_w                                    # time pressure

  centering_bonus = max(0, 1 - angular_offset / max_angle) * center_w
    where angular_offset = angle between camera forward and drone-to-sphere vector

Design:
  - Inherits observation, action, and env infrastructure from VisionNavAviary.
  - target_speed: uniform random [SPEED_MIN, SPEED_MAX] m/s per episode.
  - target_direction: random unit XY vector, rotates slowly each step
    (±DIR_JITTER random perturbation per step keeps motion smooth but varied).
  - Bounds: XY ∈ [-XY_BOUND, XY_BOUND], Z fixed at sampled height.
"""

import numpy as np
import pybullet as p

from envs.vision_nav_aviary import VisionNavAviary


class MovingTargetAviary(VisionNavAviary):
    """VisionNavAviary with a moving sphere and distance-maintenance reward."""

    # Speed range (m/s) for the moving sphere
    SPEED_MIN = 0.1
    SPEED_MAX = 0.5

    # Direction jitter per ctrl step (radians, random ±)
    DIR_JITTER = np.deg2rad(5.0)

    # Spatial bounds for sphere clamping
    XY_BOUND = 2.0
    Z_MIN    = 0.2
    Z_MAX    = 2.5

    def __init__(
        self,
        ideal_dist:   float = 1.5,    # desired following distance (metres)
        dist_w:       float = 2.0,    # weight for potential-based distance maintenance
        center_w:     float = 0.3,    # weight for centering bonus
        max_angle:    float = 0.5,    # radians — centering bonus goes to 0 at this angle
        time_w:       float = 0.01,   # time penalty per step
        **kwargs,
    ):
        self.ideal_dist = ideal_dist
        self.dist_w     = dist_w
        self.center_w   = center_w
        self.max_angle  = max_angle
        self.time_w     = time_w

        # Episode-level motion state (initialised in reset)
        self._target_speed     = 0.0
        self._target_dir_angle = 0.0
        self._prev_dist_error  = None

        super().__init__(**kwargs)

        self._dt = 1.0 / self.CTRL_FREQ

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._target_speed     = float(np.random.uniform(self.SPEED_MIN, self.SPEED_MAX))
        self._target_dir_angle = float(np.random.uniform(0.0, 2 * np.pi))
        self._prev_dist_error  = None
        return super().reset(seed=seed, options=options)

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Replace parent reward with distance-maintenance reward + centering bonus
        # (parent reward is proximity-based — not what we want here)
        reward = self._computeFollowReward() + self._center_of_frame_bonus()

        # Move sphere after the physics step
        self._move_sphere()

        # Update info with current distance
        state = self._getDroneStateVector(0)
        dist  = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        info["dist_to_target"]  = dist
        info["dist_error"]      = abs(dist - self.ideal_dist)

        return obs, reward, terminated, truncated, info

    # ── reward ────────────────────────────────────────────────────────────────

    def _computeFollowReward(self) -> float:
        """Potential-based distance maintenance reward minus time penalty."""
        state = self._getDroneStateVector(0)
        dist  = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        dist_error = abs(dist - self.ideal_dist)

        prev = self._prev_dist_error if self._prev_dist_error is not None else dist_error
        reward = (prev - dist_error) * self.dist_w - self.time_w
        self._prev_dist_error = dist_error
        self._prev_dist = dist  # keep parent compat

        return float(reward)

    def _center_of_frame_bonus(self) -> float:
        """
        Returns center_w * max(0, 1 - angular_offset / max_angle).

        angular_offset: angle (radians) between camera forward axis and the
        drone-to-sphere vector.  Camera forward = drone body-X axis (yaw-rotated).
        """
        state     = self._getDroneStateVector(0)
        drone_pos = state[0:3]
        yaw       = float(state[9])   # index 9 = yaw in getDroneStateVector

        cam_forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        rel  = self.TARGET_POS - drone_pos
        dist = np.linalg.norm(rel)
        if dist < 1e-6:
            return 0.0

        cos_a = float(np.clip(np.dot(cam_forward, rel / dist), -1.0, 1.0))
        angular_offset = np.arccos(cos_a)

        bonus = self.center_w * max(0.0, 1.0 - angular_offset / self.max_angle)
        return float(bonus)

    # ── sphere motion ─────────────────────────────────────────────────────────

    def _move_sphere(self):
        """Advance sphere position by one ctrl step and update PyBullet body."""
        self._target_dir_angle += float(np.random.uniform(-self.DIR_JITTER,
                                                           self.DIR_JITTER))
        dx = np.cos(self._target_dir_angle) * self._target_speed * self._dt
        dy = np.sin(self._target_dir_angle) * self._target_speed * self._dt

        new_pos = self.TARGET_POS.copy()
        new_pos[0] += dx
        new_pos[1] += dy

        # Clamp and bounce
        hit_x = abs(new_pos[0] + dx) >= self.XY_BOUND
        hit_y = abs(new_pos[1] + dy) >= self.XY_BOUND
        new_pos[0] = float(np.clip(new_pos[0], -self.XY_BOUND, self.XY_BOUND))
        new_pos[1] = float(np.clip(new_pos[1], -self.XY_BOUND, self.XY_BOUND))
        new_pos[2] = float(np.clip(new_pos[2], self.Z_MIN, self.Z_MAX))

        if hit_x or hit_y:
            self._target_dir_angle += np.pi  # reverse direction on wall hit

        self.TARGET_POS = new_pos

        if self._sphere_id is not None:
            p.resetBasePositionAndOrientation(
                self._sphere_id,
                self.TARGET_POS.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )
