"""
MuJoCo NavEnv — quadrotor navigation with curriculum.

Inherits HoverEnv, overrides reward with PyBullet NavAviary weights:
  potential_w  = 2.0   (prev_dist - curr_dist) * potential_w
  time_w       = 0.01  per-step penalty
  goal_bonus   = 10.0  awarded when dist < goal_thresh for 3 consecutive steps
  goal_thresh  = 0.10  m
  crash_penalty= 1.0

Curriculum stages: 0.3m → 0.6m → 1.2m → 2.0m
Call set_target_range(r) to advance stage (used by EvalCurriculumCallback).

Obs (16-dim): pos(3) + vel(3) + quat(4) + angvel(3) + rel_target(3)
Act ( 4-dim): per-motor thrust [-1, 1] → hover±0.02N
"""

import numpy as np
import mujoco
from envs.mujoco.hover_env import HoverEnv, N_SUBSTEPS, HOVER_THRUST

# Nav reward weights (match PyBullet NavAviary)
POTENTIAL_W   = 2.0
TIME_W        = 0.01
GOAL_BONUS    = 10.0
GOAL_THRESH   = 0.10   # m — 3 consecutive steps inside = success
CRASH_PENALTY = 1.0
ANGVEL_W      = 0.05   # small stability penalty (no PID cushion in MuJoCo)
MAX_STEPS_NAV = 750    # 15s — more time for approach than hover


class NavEnv(HoverEnv):
    """Navigation env with curriculum-aware target sampling."""

    def __init__(
        self,
        target_range:  float = 0.3,
        render_mode:   str   = None,
        episode_len:   int   = MAX_STEPS_NAV,
    ):
        super().__init__(target_range=target_range, render_mode=render_mode)
        self.episode_len   = episode_len
        self._consec_goal  = 0   # consecutive steps inside goal_thresh

    # ── curriculum support ────────────────────────────────────────────────

    def set_target_range(self, r: float):
        """Called by EvalCurriculumCallback to advance the curriculum stage."""
        self.target_range = float(r)

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        import mujoco as _mj
        self._consec_goal = 0
        # Call grandparent (gym.Env) to handle seed
        super(HoverEnv, self).reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        drone_home = np.array([
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(0.5, 1.5),
        ])
        self.data.qpos[:3] = drone_home
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qvel[:] = 0.0

        # min_dist scales with target_range so early curriculum isn't trivial
        min_dist = max(0.1, self.target_range * 0.25)
        direction = self.np_random.uniform(-1, 1, size=3)
        direction /= np.linalg.norm(direction) + 1e-8
        dist = self.np_random.uniform(min_dist, self.target_range)
        target = drone_home + direction * dist
        target[2] = np.clip(target[2], 0.3, 2.5)
        self._set_target(target)

        mujoco.mj_forward(self.model, self.data)
        self._prev_dist  = np.linalg.norm(self._target_pos - drone_home)
        self._step_count = 0
        return self._get_obs(), {}

    # ── step: nav reward weights + 3-step goal ────────────────────────────

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        obs    = self._get_obs()
        pos    = obs[:3]
        angvel = obs[10:13]

        dist    = float(np.linalg.norm(self._target_pos - pos))
        crashed = bool(pos[2] < 0.05)

        # 3-consecutive goal tracking
        if dist < GOAL_THRESH:
            self._consec_goal += 1
        else:
            self._consec_goal = 0
        success = self._consec_goal >= 3

        # Nav reward (PyBullet NavAviary weights)
        reward  = POTENTIAL_W * (self._prev_dist - dist)
        reward -= TIME_W
        reward -= ANGVEL_W * float(np.linalg.norm(angvel))
        if success:
            reward += GOAL_BONUS
        if crashed:
            reward -= CRASH_PENALTY

        self._prev_dist   = dist
        self._step_count += 1

        terminated = success or crashed
        truncated  = self._step_count >= self.episode_len

        info = {"dist_to_target": dist, "success": success, "crashed": crashed}
        return obs, reward, terminated, truncated, info
