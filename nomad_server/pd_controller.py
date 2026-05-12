"""PD controller for converting NoMaD waypoints to navarena displacement actions.

Ported from deployment/src/pd_controller.py with ROS dependencies removed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-8


class PDController:
    """Convert a 2D waypoint (dx, dy) in robot frame to a navarena action dict.

    The controller computes linear/angular velocities using a proportional
    scheme and then multiplies by ``dt`` to produce per-step displacements.

    Args:
        max_v: Maximum forward velocity (m/s).
        max_w: Maximum angular velocity (rad/s).
        dt: Simulation time-step (seconds).
    """

    def __init__(self, max_v: float = 0.5, max_w: float = 1.0, dt: float = 1.0) -> None:
        self.max_v = max_v
        self.max_w = max_w
        self.dt = dt
        logger.debug(
            "PDController: max_v=%.3f, max_w=%.3f, dt=%.3f",
            max_v, max_w, dt,
        )

    def control(self, waypoint: np.ndarray) -> dict[str, float]:
        """Compute a navarena action from a 2D waypoint.

        Args:
            waypoint: Array of shape (2,) with (dx, dy) in robot body frame.

        Returns:
            Action dict ``{"x": ..., "y": 0.0, "yaw": ...}``.
        """
        dx, dy = float(waypoint[0]), float(waypoint[1])

        if abs(dx) < EPS and abs(dy) < EPS:
            logger.debug("PD: waypoint ~zero (dx=%.6f, dy=%.6f), returning zero action", dx, dy)
            return {"x": 0.0, "y": 0.0, "yaw": 0.0}

        if abs(dx) < EPS:
            v = 0.0
            w = np.sign(dy) * np.pi / (2 * self.dt)
        else:
            v = dx / self.dt
            w = np.arctan(dy / dx) / self.dt

        v_raw, w_raw = v, w
        v = float(np.clip(v, 0, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))

        action = {
            "x": v * self.dt,
            "y": 0.0,
            "yaw": w * self.dt,
        }
        logger.debug(
            "PD: wp=(%.4f, %.4f) -> v=%.4f(raw %.4f) w=%.4f(raw %.4f) -> action %s",
            dx, dy, v, v_raw, w, w_raw, action,
        )
        return action
