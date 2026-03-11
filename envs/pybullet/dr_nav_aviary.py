"""
DRNavAviary — NavAviary with domain randomization.

Per-episode randomization (applied at reset):
    - Mass:     nominal * Uniform(1-mass_dr, 1+mass_dr)
    - Drag:     nominal * Uniform(1-drag_dr, 1+drag_dr)  (scales self.DRAG_COEFF)

Per-step randomization (wind disturbance):
    - Wind model: smooth random walk (exponential moving average)
        wind = wind * 0.9 + U(-wind_max, wind_max, 3) * 0.1
    - Applied via _physics() override so force acts on every physics substep.

Observation space and reward are identical to NavAviary (60 dims).

DR parameters:
    Start mild: mass_dr=0.10, drag_dr=0.15, wind_max_force=0.003
    Raise to:   mass_dr=0.15, drag_dr=0.20, wind_max_force=0.005 after confirming
                policy survives the mild regime.
"""

import numpy as np
import pybullet as p

from envs.nav_aviary import NavAviary


class DRNavAviary(NavAviary):
    """NavAviary with per-episode mass/drag and per-step wind domain randomization."""

    def __init__(
        self,
        mass_dr: float = 0.10,         # mass scale half-range
        drag_dr: float = 0.15,         # drag scale half-range
        wind_max_force: float = 0.003, # max wind force per axis [N]
        **kwargs,
    ):
        self.mass_dr = mass_dr
        self.drag_dr = drag_dr
        self.wind_max_force = wind_max_force

        super().__init__(**kwargs)

        # Nominal values captured after super().__init__ reads the URDF
        self._nominal_mass = float(self.M)
        self._nominal_drag = self.DRAG_COEFF.copy()

        # Wind state: persists within an episode, reset each episode
        self._wind_force = np.zeros(3, dtype=np.float64)

    def _set_dr_params(self, wind_max: float, mass_dr: float, drag_dr: float):
        """Called via env_method() to update DR intensity mid-training."""
        self.wind_max_force = wind_max
        self.mass_dr        = mass_dr
        self.drag_dr        = drag_dr

    # ── reset: mass + drag DR ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Mass randomization via PyBullet dynamics API
        mass_scale = np.random.uniform(1.0 - self.mass_dr, 1.0 + self.mass_dr)
        new_mass = self._nominal_mass * mass_scale
        p.changeDynamics(
            self.DRONE_IDS[0], -1,
            mass=new_mass,
            physicsClientId=self.CLIENT,
        )
        self.M = new_mass
        self.GRAVITY = self.G * self.M

        # Drag randomization (analytical model — scale coefficient array)
        drag_scale = np.random.uniform(1.0 - self.drag_dr, 1.0 + self.drag_dr)
        self.DRAG_COEFF = self._nominal_drag * drag_scale

        # Reset wind to zero at episode start
        self._wind_force = np.zeros(3, dtype=np.float64)

        return obs, info

    # ── step: advance wind random walk once per control step ─────────────────

    def step(self, action):
        # Update wind: smooth random walk (0.9 decay, 0.1 injection)
        self._wind_force = (
            self._wind_force * 0.9
            + np.random.uniform(-self.wind_max_force, self.wind_max_force, 3) * 0.1
        )
        self._wind_force = np.clip(
            self._wind_force, -self.wind_max_force, self.wind_max_force
        )
        return super().step(action)

    # ── _physics: apply wind on every physics substep ────────────────────────

    def _physics(self, rpms, nth_drone):
        super()._physics(rpms, nth_drone)
        # Only apply to drone 0 (single-drone env); avoids double-apply if multi-drone
        if nth_drone == 0:
            p.applyExternalForce(
                self.DRONE_IDS[0],
                -1,
                forceObj=self._wind_force.tolist(),
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )
