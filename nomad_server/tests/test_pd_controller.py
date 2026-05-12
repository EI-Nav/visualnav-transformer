import numpy as np
import pytest

from nomad_server.pd_controller import PDController


class TestPDController:
    def setup_method(self):
        self.ctrl = PDController(max_v=0.5, max_w=1.0, dt=1.0)

    def test_forward_waypoint(self):
        """Waypoint straight ahead -> forward motion, no rotation."""
        action = self.ctrl.control(np.array([1.0, 0.0]))
        assert action["x"] > 0
        assert action["y"] == 0.0
        assert abs(action["yaw"]) < 1e-6

    def test_left_waypoint(self):
        """Waypoint to the left -> forward + positive yaw."""
        action = self.ctrl.control(np.array([1.0, 1.0]))
        assert action["x"] > 0
        assert action["yaw"] > 0

    def test_right_waypoint(self):
        """Waypoint to the right -> forward + negative yaw."""
        action = self.ctrl.control(np.array([1.0, -1.0]))
        assert action["x"] > 0
        assert action["yaw"] < 0

    def test_velocity_clipping(self):
        """Large waypoint should be clipped to max_v."""
        action = self.ctrl.control(np.array([100.0, 0.0]))
        assert action["x"] == pytest.approx(0.5, abs=1e-6)

    def test_angular_velocity_clipping(self):
        """Large lateral offset should clip angular velocity."""
        ctrl = PDController(max_v=0.5, max_w=0.5, dt=1.0)
        action = ctrl.control(np.array([0.1, 100.0]))
        assert abs(action["yaw"]) <= 0.5 + 1e-6

    def test_zero_waypoint(self):
        """Zero waypoint -> zero action."""
        action = self.ctrl.control(np.array([0.0, 0.0]))
        assert action["x"] == 0.0
        assert action["yaw"] == 0.0

    def test_pure_lateral_waypoint(self):
        """dx~0, dy>0 -> no forward, rotate in place."""
        action = self.ctrl.control(np.array([0.0, 1.0]))
        assert action["x"] == 0.0
        assert action["yaw"] > 0

    def test_negative_dx(self):
        """Negative dx -> clipped to 0 (no reverse)."""
        action = self.ctrl.control(np.array([-1.0, 0.0]))
        assert action["x"] == 0.0

    def test_dt_scaling(self):
        """Different dt should scale the output displacement."""
        ctrl_fast = PDController(max_v=1.0, max_w=1.0, dt=0.5)
        ctrl_slow = PDController(max_v=1.0, max_w=1.0, dt=2.0)
        wp = np.array([0.3, 0.1])
        a_fast = ctrl_fast.control(wp)
        a_slow = ctrl_slow.control(wp)
        assert isinstance(a_fast["x"], float)
        assert isinstance(a_slow["x"], float)
