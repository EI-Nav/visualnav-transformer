"""NoMaD navigation model server for NavArena evaluation.

Wraps NoMaDInference and PDController into a NavigationModelServer
that communicates with navarena-bench over WebSocket.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from navarena_server import NavigationModelServer, SessionContext

from nomad_server.nomad_inference import NoMaDInference
from nomad_server.pd_controller import PDController

logger = logging.getLogger(__name__)

_ZERO_PRIMITIVE: dict[str, float] = {"x": 0.0, "y": 0.0, "yaw": 0.0}
ZERO_ACTION: dict[str, Any] = {"waypoints": [_ZERO_PRIMITIVE]}


def _wrap_action(primitive: dict[str, float]) -> dict[str, Any]:
    """Wrap a primitive {x, y, yaw} into waypoint action format."""
    return {"waypoints": [primitive]}


class NoMaDServer(NavigationModelServer):
    """NavArena-compatible server that runs NoMaD inference.

    Args:
        inference: Configured NoMaDInference instance.
        controller: Configured PDController instance.
    """

    def __init__(self, inference: NoMaDInference, controller: PDController) -> None:
        self._inference = inference
        self._controller = controller
        self._goal_images: dict[str, Any] = {}

    def _slot_key(self, ctx: SessionContext) -> str:
        return ctx.slot_id if ctx.slot_id is not None else ctx.session_id

    async def on_episode_start(
        self, task_info: dict[str, Any], ctx: SessionContext
    ) -> None:
        key = self._slot_key(ctx)
        self._inference.reset(key)
        self._goal_images.pop(key, None)
        logger.info("Episode started for slot %s (episode %s)", key, ctx.episode_id)

    async def on_episode_end(
        self, result: dict[str, Any], ctx: SessionContext
    ) -> None:
        key = self._slot_key(ctx)
        self._goal_images.pop(key, None)
        logger.info(
            "Episode ended for slot %s: success=%s, steps=%d",
            key,
            result.get("success"),
            ctx.step,
        )

    async def predict(
        self, observation: dict[str, Any], ctx: SessionContext
    ) -> dict[str, Any]:
        key = self._slot_key(ctx)
        rgb_dict = observation.get("rgb", {})
        if not rgb_dict:
            logger.warning("No RGB data in observation for slot %s", key)
            return ZERO_ACTION
        obs_image = next(iter(rgb_dict.values()))

        goal_image = observation.get("goal_image")
        if goal_image is not None:
            self._goal_images[key] = goal_image
        else:
            goal_image = self._goal_images.get(key)

        if goal_image is None:
            logger.warning("No goal image available for slot %s", key)
            return ZERO_ACTION

        try:
            waypoint, distance = self._inference.infer(obs_image, goal_image, key)
        except Exception:
            logger.exception("Inference failed for slot %s", key)
            return ZERO_ACTION

        if waypoint is None:
            logger.debug(
                "slot %s step %d: context not yet full, returning zero action",
                key, ctx.step,
            )
            return ZERO_ACTION

        logger.debug(
            "slot=%s wp=(%.3f, %.3f) dist=%.3f",
            key, waypoint[0], waypoint[1], distance,
        )
        return _wrap_action(self._controller.control(waypoint))

    async def batch_predict(
        self,
        observations: dict[str, dict[str, Any]],
        contexts: dict[str, SessionContext],
    ) -> dict[str, dict[str, Any]]:
        obs_images: dict[str, Any] = {}
        goal_imgs: dict[str, Any] = {}
        slots_no_rgb: list[str] = []

        for slot_id, obs in observations.items():
            rgb_dict = obs.get("rgb", {})
            if not rgb_dict:
                slots_no_rgb.append(slot_id)
                continue
            obs_images[slot_id] = next(iter(rgb_dict.values()))

            goal_image = obs.get("goal_image")
            if goal_image is not None:
                self._goal_images[slot_id] = goal_image
            goal_imgs[slot_id] = self._goal_images.get(slot_id)

        if slots_no_rgb:
            logger.warning(
                "batch_predict: %d slot(s) missing RGB data: %s",
                len(slots_no_rgb), slots_no_rgb,
            )

        slots_without_goal = [s for s, g in goal_imgs.items() if g is None]
        if slots_without_goal:
            logger.warning(
                "batch_predict: %d slot(s) missing goal image: %s",
                len(slots_without_goal), slots_without_goal,
            )
        for s in slots_without_goal:
            obs_images.pop(s, None)
            goal_imgs.pop(s, None)

        logger.debug(
            "batch_predict: %d total, %d valid (after filtering rgb/goal)",
            len(observations), len(obs_images),
        )

        if not obs_images:
            logger.debug("batch_predict: no valid slots, returning all zero actions")
            return {s: ZERO_ACTION for s in observations}

        try:
            batch_results = self._inference.batch_infer(obs_images, goal_imgs)
        except Exception:
            logger.exception("Batch inference failed")
            return {s: ZERO_ACTION for s in observations}

        actions: dict[str, dict[str, Any]] = {}
        for slot_id in observations:
            result = batch_results.get(slot_id)
            if result is None or result[0] is None:
                logger.debug(
                    "batch_predict: slot %s -> zero action (no waypoint)",
                    slot_id,
                )
                actions[slot_id] = ZERO_ACTION
            else:
                actions[slot_id] = _wrap_action(self._controller.control(result[0]))
                logger.debug(
                    "batch_predict: slot %s -> action %s",
                    slot_id, actions[slot_id],
                )
        return actions
