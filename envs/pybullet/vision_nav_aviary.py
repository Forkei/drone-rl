"""
VisionNavAviary — single-drone vision-based navigation.

The drone must fly to a colored sphere target using only its onboard camera.
No GPS / position data in the observation — pure pixels.

Design: "Privileged teacher" approach (standard in sim-to-real literature).
  - Observation: RGB camera frames, channel-first (3, H, W) uint8
  - Reward:      ground-truth PyBullet distance to sphere (privileged signal)
  - Policy:      must learn to associate pixel patterns with proximity

Alpha channel dropped — SB3 NatureCNN handles 1/3/4 channels but RGB is
standard and avoids confusion. img_wh controls resolution (default 64×48,
can drop to (48,32) for speed: ~2x fps, slight visual fidelity loss).

Observation:
    (3, H, W) RGB uint8, channel-first — SB3 CnnPolicy compatible

Action:
    ActionType.PID — 3D waypoint in [-1, 1]^3 (inner PID → RPMs)

Reward (potential-based + sparse bonus, ground-truth dist):
    per step:   (prev_dist - curr_dist) * potential_w - time_w
    warm zone:  +warm_zone_bonus when dist < warm_zone_thresh
    on success: +goal_bonus (dist < goal_thresh for success_steps steps)

Technical note:
    ctrl_freq=24 is REQUIRED when obs=RGB.
    PyBullet: IMG_CAPTURE_FREQ = PYB_FREQ/24 = 10 steps/frame.
    PYB_STEPS_PER_CTRL = PYB_FREQ/CTRL_FREQ must divide 10 evenly.
    ctrl_freq=24 → PYB_STEPS_PER_CTRL=10 → 10%10=0 ✓
    ctrl_freq=30 → PYB_STEPS_PER_CTRL=8  → 10%8≠0  ✗ (error)
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel, Physics


# Target sphere appearance
_SPHERE_RADIUS  = 0.15          # metres — 0.15 gives ~43px at 1m in 32×48; 0.08 gave only 10px
_SPHERE_COLOR   = [1.0, 0.2, 0.0, 1.0]   # bright orange-red RGBA


class VisionNavAviary(HoverAviary):
    """
    Camera-only navigation to a visible colored sphere.

    The policy sees pixels; reward is computed from ground-truth distance.
    No position or relative-target vector in the observation.
    """

    def __init__(
        self,
        # --- task ---
        target_range:    float = 0.5,       # sphere sampled within this radius of home
        home_pos:        np.ndarray = None, # drone start + sphere sampling centre
        # --- reward ---
        potential_w:     float = 2.0,
        time_w:          float = 0.01,
        goal_bonus:      float = 10.0,
        warm_zone_bonus: float = 0.5,
        warm_zone_thresh: float = 0.15,
        crash_penalty:   float = 1.0,
        goal_thresh:     float = 0.1,
        success_steps:   int = 3,
        # --- env config ---
        episode_len_sec: float = 10.0,
        img_wh:          tuple = (64, 48),  # camera (width, height); use (48,32) for speed
        gui:             bool = False,
        record:          bool = False,
        drone_model:     DroneModel = DroneModel.CF2X,
        physics:         Physics = Physics.PYB,
        pyb_freq:        int = 240,
        ctrl_freq:       int = 24,   # MUST be 24 (or 48) for RGB obs — see module docstring
    ):
        self._img_wh = img_wh   # store before super().__init__ which sets IMG_RES
        self.target_range     = target_range
        self.home_pos         = np.array([0.0, 0.0, 1.0]) if home_pos is None else np.array(home_pos)

        # reward config
        self.potential_w      = potential_w
        self.time_w           = time_w
        self.goal_bonus       = goal_bonus
        self.warm_zone_bonus  = warm_zone_bonus
        self.warm_zone_thresh = warm_zone_thresh
        self.crash_penalty    = crash_penalty
        self.goal_thresh      = goal_thresh
        self.success_steps    = success_steps

        # episode state
        self._consecutive_success = 0
        self._episode_success     = False
        self._prev_dist           = None
        self.TARGET_POS           = self.home_pos.copy()
        self._sphere_id           = None

        init_xyz = self.home_pos.reshape(1, 3)

        super().__init__(
            drone_model=drone_model,
            initial_xyzs=init_xyz,
            obs=ObservationType.RGB,
            act=ActionType.PID,
            gui=gui,
            record=record,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
        )
        self.EPISODE_LEN_SEC = episode_len_sec
        # Override IMG_RES after super().__init__ — buffers re-allocated below
        if img_wh != (64, 48):
            self.IMG_RES = np.array([img_wh[0], img_wh[1]])
            h, w = img_wh[1], img_wh[0]
            self.rgb = np.zeros((1, h, w, 4))
            self.dep = np.ones((1, h, w))
            self.seg = np.zeros((1, h, w))

    # ── target sampling ───────────────────────────────────────────────────────

    def _sample_target(self) -> np.ndarray:
        """Uniform sample inside sphere of radius target_range around home_pos."""
        while True:
            delta = np.random.uniform(-self.target_range, self.target_range, 3)
            if np.linalg.norm(delta) <= self.target_range:
                t = self.home_pos + delta
                t[2] = max(t[2], 0.2)
                return t

    # ── scene setup ───────────────────────────────────────────────────────────

    def _addObstacles(self):
        """Add landmarks (for camera texture variety) + colored target sphere."""
        super()._addObstacles()   # adds 4 landmark objects (block, cube, duck, teddy)

        # Create the target sphere (visual-only — no mass, no collision)
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=_SPHERE_RADIUS,
            rgbaColor=_SPHERE_COLOR,
            physicsClientId=self.CLIENT,
        )
        self._sphere_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,   # no collision
            baseVisualShapeIndex=vis_id,
            basePosition=self.TARGET_POS.tolist(),
            physicsClientId=self.CLIENT,
        )

    # ── observation space ────────────────────────────────────────────────────

    def _observationSpace(self):
        """Override to return 3D channel-first RGB obs required by SB3 CnnPolicy.

        BaseRLAviary returns (NUM_DRONES, H, W, 4) which is 4D — SB3 rejects it.
        We return (3, H, W) RGB channel-first (alpha dropped) for single drone.
        SB3's NatureCNN accepts: len(shape)==3, dtype=uint8, bounds=[0,255].

        Uses self._img_wh (set before super().__init__) so the declared obs space
        matches the actual output when a non-default img_wh is requested.
        """
        # _img_wh is stored before super().__init__ so it is available here.
        # Fall back to IMG_RES only if _img_wh is not yet set (should not happen).
        if hasattr(self, "_img_wh"):
            w, h = int(self._img_wh[0]), int(self._img_wh[1])
        else:
            h, w = int(self.IMG_RES[1]), int(self.IMG_RES[0])
        return spaces.Box(low=0, high=255, shape=(3, h, w), dtype=np.uint8)

    def _computeObs(self):
        """Capture camera frame, return (3, H, W) RGB channel-first uint8."""
        rgba, _, _ = self._getDroneImages(0, segmentation=False)
        # rgba: (H, W, 4) int64 → drop alpha → uint8 → (3, H, W)
        return rgba[:, :, :3].astype(np.uint8).transpose(2, 0, 1)

    # ── gymnasium overrides ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self.TARGET_POS = self._sample_target()
        self._consecutive_success = 0
        self._episode_success     = False
        self._prev_dist           = None

        obs, info = super().reset(seed=seed, options=options)

        # Move sphere to new target position
        if self._sphere_id is not None:
            p.resetBasePositionAndOrientation(
                self._sphere_id,
                self.TARGET_POS.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT,
            )

        # Initialise prev_dist from reset state (use PyBullet state, not obs)
        state = self._getDroneStateVector(0)
        self._prev_dist = float(np.linalg.norm(self.TARGET_POS - state[0:3]))

        return obs, info

    def step(self, action):
        """Clamp z of PID waypoint to prevent underground commands."""
        action = np.array(action, dtype=np.float32)
        action[:, 2] = np.clip(action[:, 2], 0.15, 2.0)
        return super().step(action)

    def _computeReward(self):
        state     = self._getDroneStateVector(0)
        rel       = self.TARGET_POS - state[0:3]
        dist      = float(np.linalg.norm(rel))

        # Potential-based: reward for closing distance
        prev   = self._prev_dist if self._prev_dist is not None else dist
        reward = (prev - dist) * self.potential_w - self.time_w
        self._prev_dist = dist

        # Warm zone
        if dist < self.warm_zone_thresh:
            reward += self.warm_zone_bonus

        # Success
        if dist < self.goal_thresh:
            self._consecutive_success += 1
            if self._consecutive_success >= self.success_steps:
                reward += self.goal_bonus
                self._episode_success = True
        else:
            self._consecutive_success = 0

        return float(reward)

    def _computeTerminated(self):
        return self._episode_success

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 2.5 or abs(state[1]) > 2.5 or
                state[2] > 3.0 or state[2] < 0.05 or
                abs(state[7]) > 0.8 or abs(state[8]) > 0.8):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist  = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        return {
            "dist_to_target": dist,
            "success":        self._episode_success,
            "target_pos":     self.TARGET_POS.copy(),
            "target_range":   self.target_range,
        }
