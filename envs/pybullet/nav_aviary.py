"""
NavAviary — single-drone navigation environment.

The drone must fly to a randomly sampled target position each episode.
Built as a clean, parameterized foundation for future extensions
(obstacle avoidance, person tracking, multi-agent, etc.).

Observation (per step):
    [x, y, z,                   # position          (3)
     roll, pitch, yaw,          # euler angles       (3)
     vx, vy, vz,                # linear velocity    (3)
     wx, wy, wz,                # angular velocity   (3)
     rel_tx, rel_ty, rel_tz,    # target - pos       (3)  ← key for generalization
     ...action_buffer...]       # last N actions     (3*N)
    Total: 15 + 3*ACTION_BUFFER_SIZE = 15 + 45 = 60 dims

Action:
    ActionType.PID — 3D waypoint target in [-1, 1]^3
    Inner PID loop converts to RPMs automatically.

Reward (potential-based + sparse bonus):
    per step:   (prev_dist - curr_dist) * potential_w   # positive when approaching
                - time_w                                 # time pressure
    on success: +goal_bonus  (dist < goal_thresh for success_steps consecutive steps)
    Note: potential-based reward gives gradient even on short episodes,
    unlike -dist which only signals absolute position.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel, Physics


class NavAviary(HoverAviary):
    """Single-drone navigation to a randomly sampled target."""

    def __init__(
        self,
        # --- curriculum / task ---
        target_range: float = 0.5,      # radius of sphere around home [0,0,1] to sample targets
        home_pos: np.ndarray = None,    # centre of the target sphere
        # --- reward weights ---
        potential_w: float = 2.0,       # scale for potential-based reward (prev_dist - curr_dist)
        time_w: float = 0.01,           # time penalty per step
        goal_bonus: float = 10.0,       # sparse bonus on reaching goal
        warm_zone_bonus: float = 0.5,   # per-step bonus when dist < warm_zone_thresh
        warm_zone_thresh: float = 0.15, # metres — "warm zone" around goal
        crash_penalty: float = 1.0,     # penalty on crash/out-of-bounds
        goal_thresh: float = 0.1,       # metres — "reached goal"
        success_steps: int = 3,         # consecutive steps within goal_thresh for success
        altitude_bonus_w: float = 0.0,  # extra Z-shaping when XY is already aligned
        altitude_xy_thresh: float = 0.2,# XY distance threshold to activate altitude bonus
        # --- env config ---
        episode_len_sec: float = 10.0,
        gui: bool = False,
        record: bool = False,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
    ):
        self.target_range = target_range
        self.home_pos = np.array([0.0, 0.0, 1.0]) if home_pos is None else np.array(home_pos)

        # reward config
        self.potential_w      = potential_w
        self.time_w           = time_w
        self.goal_bonus       = goal_bonus
        self.warm_zone_bonus  = warm_zone_bonus
        self.warm_zone_thresh = warm_zone_thresh
        self.crash_penalty    = crash_penalty
        self.goal_thresh  = goal_thresh
        self.success_steps = success_steps
        self.altitude_bonus_w    = altitude_bonus_w
        self.altitude_xy_thresh  = altitude_xy_thresh

        # internal state
        self._consecutive_success = 0
        self._episode_success = False
        self._prev_dist = None
        self._prev_z_error = None

        # Start drone at home_pos (already in the target zone), not at ground level.
        # This removes the need to learn takeoff before learning navigation.
        init_xyz = self.home_pos.reshape(1, 3)

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=init_xyz,
            obs=ObservationType.KIN,
            act=ActionType.PID,
            gui=gui,
            record=record,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
        )
        # Override episode length after super().__init__
        self.EPISODE_LEN_SEC = episode_len_sec

    # ── target sampling ───────────────────────────────────────────────────────

    def _sample_target(self) -> np.ndarray:
        """Uniform sample inside a sphere of radius target_range around home_pos."""
        # Rejection-sample inside sphere
        while True:
            delta = np.random.uniform(-self.target_range, self.target_range, 3)
            if np.linalg.norm(delta) <= self.target_range:
                t = self.home_pos + delta
                t[2] = max(t[2], 0.2)   # don't go underground
                return t

    # ── gymnasium overrides ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self.TARGET_POS = self._sample_target()
        self._consecutive_success = 0
        self._episode_success = False
        self._prev_dist = None
        self._prev_z_error = None
        obs, info = super().reset(seed=seed, options=options)
        # Initialise prev_dist from the reset state
        state = self._getDroneStateVector(0)
        rel_target = self.TARGET_POS - state[0:3]
        self._prev_dist = float(np.linalg.norm(rel_target))
        self._prev_z_error = float(abs(rel_target[2]))
        return obs, info

    def step(self, action):
        """Clamp z of PID waypoint target to prevent commanding underground flight."""
        action = np.array(action, dtype=np.float32)
        action[:, 2] = np.clip(action[:, 2], 0.15, 2.0)
        return super().step(action)

    def _observationSpace(self):
        """Extend base KIN obs with 3D relative target vector."""
        base_space = super()._observationSpace()
        lo = base_space.low
        hi = base_space.high
        # Append [-inf, inf] for [rel_tx, rel_ty, rel_tz]
        rel_lo = np.full((1, 3), -np.inf, dtype=np.float32)
        rel_hi = np.full((1, 3),  np.inf, dtype=np.float32)
        return spaces.Box(
            low=np.hstack([lo, rel_lo]),
            high=np.hstack([hi, rel_hi]),
            dtype=np.float32,
        )

    def _computeObs(self):
        """Base KIN obs + relative target vector appended."""
        base_obs = super()._computeObs()           # shape (1, 57)
        state = self._getDroneStateVector(0)
        rel_target = (self.TARGET_POS - state[0:3]).astype(np.float32)
        rel_target = rel_target.reshape(1, 3)
        return np.hstack([base_obs, rel_target])   # shape (1, 60)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        rel_target = self.TARGET_POS - state[0:3]
        dist  = float(np.linalg.norm(rel_target))

        # Potential-based: reward progress toward target
        prev = self._prev_dist if self._prev_dist is not None else dist
        reward = (prev - dist) * self.potential_w - self.time_w
        self._prev_dist = dist

        # Altitude shaping: extra Z signal when XY is already aligned
        if self.altitude_bonus_w > 0.0:
            z_error = float(abs(rel_target[2]))
            xy_dist = float(np.linalg.norm(rel_target[:2]))
            if xy_dist < self.altitude_xy_thresh and self._prev_z_error is not None:
                reward += self.altitude_bonus_w * (self._prev_z_error - z_error)
            self._prev_z_error = z_error

        # Warm zone: per-step bonus for staying close (prevents oscillation without succeeding)
        if dist < self.warm_zone_thresh:
            reward += self.warm_zone_bonus

        # Sparse success bonus
        if dist < self.goal_thresh:
            self._consecutive_success += 1
            if self._consecutive_success >= self.success_steps:
                reward += self.goal_bonus
                self._episode_success = True
        else:
            self._consecutive_success = 0

        return float(reward)

    def _computeTerminated(self):
        # End episode on sustained success
        return self._episode_success

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        # Out of bounds or excessive tilt
        if (abs(state[0]) > 2.5 or abs(state[1]) > 2.5 or
                state[2] > 3.0 or state[2] < 0.05 or
                abs(state[7]) > 0.8 or abs(state[8]) > 0.8):  # 0.8 rad ≈ 46° — less strict
            return True
        # Timeout
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        return {
            "dist_to_target": dist,
            "success": self._episode_success,
            "target_pos": self.TARGET_POS.copy(),
            "target_range": self.target_range,
        }
