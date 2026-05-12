import numpy as np
import pytest
from unittest.mock import MagicMock

from nomad_server.nomad_server import NoMaDServer, ZERO_ACTION
from nomad_server.pd_controller import PDController


class FakeSessionContext:
    def __init__(self, session_id="s1", episode_id="e1", slot_id=None, step=0):
        self.session_id = session_id
        self.episode_id = episode_id
        self.slot_id = slot_id
        self.step = step
        self.is_first = step == 0
        self.task = {}


class TestNoMaDServer:
    def _make_server(self):
        inference = MagicMock()
        inference.reset = MagicMock()
        inference.infer = MagicMock(return_value=(np.array([0.5, 0.1]), 2.5))
        inference.batch_infer = MagicMock(return_value={
            "slot0": (np.array([0.5, 0.1]), 2.5),
            "slot1": (np.array([0.3, -0.2]), 3.0),
        })
        controller = PDController(max_v=0.5, max_w=1.0, dt=1.0)
        server = NoMaDServer(inference=inference, controller=controller)
        return server, inference

    @pytest.mark.asyncio
    async def test_on_episode_start_resets_context(self):
        server, inference = self._make_server()
        ctx = FakeSessionContext()
        await server.on_episode_start({"task": "imagenav"}, ctx)
        inference.reset.assert_called_once_with("s1")

    @pytest.mark.asyncio
    async def test_predict_returns_action_dict(self):
        server, inference = self._make_server()
        ctx = FakeSessionContext()
        obs = {
            "rgb": {"camera": np.zeros((100, 100, 3), dtype=np.uint8)},
            "goal_image": np.zeros((100, 100, 3), dtype=np.uint8),
        }
        action = await server.predict(obs, ctx)
        assert "waypoints" in action
        waypoints = action["waypoints"]
        assert isinstance(waypoints, list) and len(waypoints) == 1
        assert all(k in waypoints[0] for k in ("x", "y", "yaw"))
        inference.infer.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_returns_zero_when_not_ready(self):
        server, inference = self._make_server()
        inference.infer.return_value = (None, None)
        ctx = FakeSessionContext()
        obs = {
            "rgb": {"camera": np.zeros((100, 100, 3), dtype=np.uint8)},
            "goal_image": np.zeros((100, 100, 3), dtype=np.uint8),
        }
        action = await server.predict(obs, ctx)
        assert action == ZERO_ACTION

    @pytest.mark.asyncio
    async def test_batch_predict(self):
        server, inference = self._make_server()
        observations = {
            "slot0": {
                "rgb": {"cam": np.zeros((50, 50, 3), dtype=np.uint8)},
                "goal_image": np.zeros((50, 50, 3), dtype=np.uint8),
            },
            "slot1": {
                "rgb": {"cam": np.zeros((50, 50, 3), dtype=np.uint8)},
                "goal_image": np.zeros((50, 50, 3), dtype=np.uint8),
            },
        }
        contexts = {
            "slot0": FakeSessionContext(slot_id="slot0"),
            "slot1": FakeSessionContext(slot_id="slot1"),
        }
        actions = await server.batch_predict(observations, contexts)
        assert "slot0" in actions
        assert "slot1" in actions
        assert "waypoints" in actions["slot0"]
        assert isinstance(actions["slot0"]["waypoints"], list)
        assert all(k in actions["slot0"]["waypoints"][0] for k in ("x", "y", "yaw"))
