"""
AltitudeNavAviary — NavAviary with pure-vertical target sampling.

Inherits NavAviary exactly. Only _sample_target() is overridden:
  z_target = drone_z + uniform(0.5, 1.5)   (always above the drone)
  xy stays at drone's current XY

No reward function changes. Distribution shift only.
Used to targeted-train the high_target failure case.
"""

import numpy as np
from envs.nav_aviary import NavAviary


class AltitudeNavAviary(NavAviary):
    """NavAviary with targets always directly above the drone."""

    def _sample_target(self) -> np.ndarray:
        """Target is always directly above current drone position."""
        # Use current drone state if available, else home_pos
        try:
            state = self._getDroneStateVector(0)
            drone_xyz = state[0:3]
        except Exception:
            drone_xyz = self.home_pos.copy()

        z_offset = np.random.uniform(0.5, 1.5)
        target = np.array([drone_xyz[0], drone_xyz[1], drone_xyz[2] + z_offset])
        target[2] = max(target[2], 0.2)   # safety floor
        return target
