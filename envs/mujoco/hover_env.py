"""
MuJoCo HoverEnv — Gymnasium-compatible quadrotor navigation environment.

Mirrors NavAviary interface:
  Obs (16-dim): pos(3) + vel(3) + quat(4) + angvel(3) + rel_target(3)
  Act ( 4-dim): per-motor thrust, normalized [-1, 1]
  Reward: potential-based (like NavAviary) + time penalty + goal bonus

Physics: MuJoCo 3.x, timestep=0.002s, n_substeps=10 → 50 Hz control
Episode: 10s max → 500 steps
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

_XML_PATH = os.path.join(os.path.dirname(__file__), "quadrotor.xml")

# Crazyflie constants (match PyBullet NavAviary)
MASS        = 0.027          # kg
GRAVITY     = 9.81           # m/s²
MAX_THRUST  = 0.149          # N per motor (T/W 2.25)
HOVER_THRUST = MASS * GRAVITY / 4  # N per motor at hover ≈ 0.0662 N

# Normalised action → thrust
#   action ∈ [-1, 1] → thrust = clip((action+1)/2, 0, 1) * MAX_THRUST
# At action=0 → thrust = MAX_THRUST/2 = 0.0745 N (slightly above hover)
# At action≈-0.11 → thrust = HOVER_THRUST (perfect hover)

CTRL_FREQ   = 50             # Hz (1 / (timestep * n_substeps)) = 1/(0.002*10)
N_SUBSTEPS  = 10
MAX_STEPS   = 500            # 10s episodes
SUCCESS_DIST = 0.10          # m — goal radius (matches NavAviary)

OBS_DIM  = 16
ACT_DIM  = 4


class HoverEnv(gym.Env):
    """Single-drone hover/navigation env backed by MuJoCo."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": CTRL_FREQ}

    def __init__(
        self,
        target_range: float = 1.0,   # m — max spawn distance
        render_mode:  str   = None,
    ):
        super().__init__()
        self.target_range = target_range
        self.render_mode  = render_mode

        self.model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.data  = mujoco.MjData(self.model)

        self.model.opt.timestep = 0.002
        self._drone_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self._target_mid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")

        # Sensor slice indices
        # Sensors declared in XML: pos(3), quat(4), vel(3), angvel(3) → 13 total
        self._pos_adr    = self.model.sensor_adr[0]   # 3
        self._quat_adr   = self.model.sensor_adr[1]   # 4
        self._vel_adr    = self.model.sensor_adr[2]   # 3
        self._angvel_adr = self.model.sensor_adr[3]   # 3

        # Obs/act spaces
        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(
            low=-np.ones(ACT_DIM, dtype=np.float32),
            high=np.ones(ACT_DIM, dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count  = 0
        self._target_pos  = np.array([0.0, 0.0, 1.0])
        self._prev_dist   = None

        # Renderer (lazy init)
        self._renderer = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        d = self.data
        pos    = d.sensordata[self._pos_adr    : self._pos_adr    + 3].copy()
        quat   = d.sensordata[self._quat_adr   : self._quat_adr   + 4].copy()
        vel    = d.sensordata[self._vel_adr    : self._vel_adr    + 3].copy()
        angvel = d.sensordata[self._angvel_adr : self._angvel_adr + 3].copy()
        rel    = self._target_pos - pos
        return np.concatenate([pos, vel, quat, angvel, rel]).astype(np.float32)

    def _set_target(self, pos: np.ndarray):
        self._target_pos = pos.copy()
        self.data.mocap_pos[0] = pos

    def _apply_action(self, action: np.ndarray):
        """action=0 → hover thrust, action=±1 → ±0.02N around hover."""
        thrust = np.clip(HOVER_THRUST + action * 0.02, 0.0, 0.1)
        self.data.ctrl[:] = thrust

    # ── gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomise drone spawn position
        drone_home = np.array([
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(0.5, 1.5),
        ])
        # Set freejoint qpos: pos(3) + quat(4)
        self.data.qpos[:3] = drone_home
        self.data.qpos[3:7] = [1, 0, 0, 0]   # upright
        self.data.qvel[:] = 0.0

        # Randomise target position within target_range
        direction = self.np_random.uniform(-1, 1, size=3)
        direction /= np.linalg.norm(direction) + 1e-8
        dist = self.np_random.uniform(0.2, self.target_range)
        target = drone_home + direction * dist
        target[2] = np.clip(target[2], 0.3, 2.5)   # keep target above floor
        self._set_target(target)

        mujoco.mj_forward(self.model, self.data)

        self._prev_dist  = np.linalg.norm(self._target_pos - drone_home)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        pos    = obs[:3]
        angvel = obs[10:13]

        dist     = float(np.linalg.norm(self._target_pos - pos))
        success  = bool(dist < SUCCESS_DIST)
        crashed  = bool(pos[2] < 0.05)   # hit the floor

        # Potential-based reward + angular velocity stability penalty
        reward  = (self._prev_dist - dist)              # approach = positive
        reward -= 0.001                                  # time penalty
        reward -= 0.1 * float(np.linalg.norm(angvel))   # anti-tumble
        if success:
            reward += 2.0
        if crashed:
            reward -= 1.0

        self._prev_dist  = dist
        self._step_count += 1

        terminated = success or crashed
        truncated  = self._step_count >= MAX_STEPS

        info = {"dist_to_target": dist, "success": success, "crashed": crashed}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)   # -1 = free camera
        return self._renderer.render()

    def get_drone_cam_frame(self, height=32, width=48):
        """RGB frame from drone-mounted camera (channel-last, uint8)."""
        if not hasattr(self, "_cam_renderer") or self._cam_renderer is None:
            self._cam_renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "drone_cam")
        self._cam_renderer.update_scene(self.data, camera=cam_id)
        return self._cam_renderer.render()   # (H, W, 3) uint8

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if hasattr(self, "_cam_renderer") and self._cam_renderer is not None:
            self._cam_renderer.close()
            self._cam_renderer = None
