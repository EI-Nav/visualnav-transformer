"""GNM model inference wrapper for NavArena (ImageNav)."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image as PILImage

from nomad_server.image_utils import transform_images

logger = logging.getLogger(__name__)


def _torch_load_checkpoint(path: str, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_state_dict_into_model(model: torch.nn.Module, checkpoint: Any) -> None:
    """Load weights like ``train_eval_loop.load_model`` for non-nomad checkpoints."""
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        loaded_model = checkpoint["model"]
        if isinstance(loaded_model, dict):
            state_dict = {
                k.removeprefix("module."): v for k, v in loaded_model.items()
            }
        else:
            try:
                state_dict = loaded_model.module.state_dict()
            except AttributeError:
                state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict, strict=False)
    elif isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        state_dict = {
            k.removeprefix("module."): v for k, v in checkpoint.items()
        }
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)


class GNMInference:
    """GNM forward pass with temporal context queues per slot."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        data_config_path: str,
        device: str = "cuda:0",
        waypoint_index: int = 2,
        dataset_name: str | None = None,
        center_crop: bool = True,
    ) -> None:
        self._device = torch.device(device)

        with open(config_path) as f:
            self._config: dict[str, Any] = yaml.safe_load(f)

        if self._config.get("model_type") != "gnm":
            raise ValueError(
                f"gnm_server expects model_type 'gnm', got {self._config.get('model_type')!r}"
            )

        with open(data_config_path) as f:
            all_data_config: dict[str, Any] = yaml.safe_load(f)

        ds_keys = list(self._config.get("datasets", {}).keys())
        if not ds_keys:
            raise ValueError("Training config has no 'datasets' entry")

        self._dataset_name = dataset_name if dataset_name is not None else ds_keys[0]
        if self._dataset_name not in self._config["datasets"]:
            raise ValueError(
                f"dataset_name {self._dataset_name!r} not in training config datasets"
            )
        if self._dataset_name not in all_data_config:
            raise ValueError(
                f"dataset_name {self._dataset_name!r} not in data config {data_config_path}"
            )

        ds_train_cfg = self._config["datasets"][self._dataset_name]
        self._waypoint_spacing: int = int(ds_train_cfg.get("waypoint_spacing", 1))
        metric_wp = all_data_config[self._dataset_name].get("metric_waypoint_spacing")
        if metric_wp is None:
            raise ValueError(
                f"metric_waypoint_spacing missing for dataset {self._dataset_name!r}"
            )
        self._metric_scale: float = float(metric_wp) * float(self._waypoint_spacing)

        self._normalize: bool = bool(self._config.get("normalize", True))
        self._context_size: int = int(self._config["context_size"])
        self._image_size: list[int] = list(self._config["image_size"])
        self._waypoint_index = int(waypoint_index)
        self._center_crop = center_crop

        self._model = self._build_and_load(checkpoint_path)
        self._model.eval()

        self._context_queues: dict[str, deque[PILImage.Image]] = {}

        logger.info(
            "GNMInference: dataset=%s metric_scale=%.4f normalize=%s context_size=%d "
            "image_size=%s waypoint_index=%d device=%s center_crop=%s",
            self._dataset_name,
            self._metric_scale,
            self._normalize,
            self._context_size,
            self._image_size,
            self._waypoint_index,
            self._device,
            self._center_crop,
        )

    def _build_and_load(self, checkpoint_path: str) -> torch.nn.Module:
        from vint_train.models.gnm.gnm import GNM

        cfg = self._config
        model = GNM(
            cfg["context_size"],
            cfg["len_traj_pred"],
            cfg["learn_angle"],
            cfg["obs_encoding_size"],
            cfg["goal_encoding_size"],
        )

        ckpt = _torch_load_checkpoint(checkpoint_path, self._device)
        _load_state_dict_into_model(model, ckpt)
        model.to(self._device)
        return model

    def _init_slot(self, slot_id: str) -> None:
        self._context_queues[slot_id] = deque(maxlen=self._context_size + 1)

    def reset(self, slot_id: str) -> None:
        self._context_queues[slot_id] = deque(maxlen=self._context_size + 1)
        logger.debug("Reset context queue for slot %s", slot_id)

    def _push_obs(self, slot_id: str, pil_img: PILImage.Image) -> None:
        if slot_id not in self._context_queues:
            self._init_slot(slot_id)
        self._context_queues[slot_id].append(pil_img)

    def _is_ready(self, slot_id: str) -> bool:
        return len(self._context_queues.get(slot_id, [])) > self._context_size

    def _obs_goal_tensors(
        self,
        context: list[PILImage.Image],
        goal_image: PILImage.Image | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_image = transform_images(
            context, self._image_size, center_crop=self._center_crop
        ).to(self._device)

        if isinstance(goal_image, np.ndarray):
            goal_image = PILImage.fromarray(goal_image)
        goal_image_t = transform_images(
            [goal_image], self._image_size, center_crop=self._center_crop
        ).to(self._device)
        return obs_image, goal_image_t

    @torch.no_grad()
    def infer(
        self,
        obs_image: np.ndarray | PILImage.Image,
        goal_image: np.ndarray | PILImage.Image,
        slot_id: str,
    ) -> tuple[np.ndarray | None, float | None]:
        if isinstance(obs_image, np.ndarray):
            obs_image = PILImage.fromarray(obs_image)
        self._push_obs(slot_id, obs_image)

        if not self._is_ready(slot_id):
            return None, None

        if isinstance(goal_image, np.ndarray):
            goal_image = PILImage.fromarray(goal_image)

        t0 = time.monotonic()
        context = list(self._context_queues[slot_id])
        obs_b, goal_b = self._obs_goal_tensors(context, goal_image)

        dist_pred, action_pred = self._model(obs_b, goal_b)
        dist_val = float(dist_pred.flatten().cpu().numpy()[0])

        wp = action_pred[0, self._waypoint_index, :2].cpu().numpy().astype(np.float64)
        if self._normalize:
            wp = wp * self._metric_scale

        logger.debug(
            "slot %s: wp=(%.4f, %.4f) dist=%.4f infer_time=%.3fs",
            slot_id, wp[0], wp[1], dist_val, time.monotonic() - t0,
        )
        return wp, dist_val

    @torch.no_grad()
    def batch_infer(
        self,
        obs_images: dict[str, np.ndarray | PILImage.Image],
        goal_images: dict[str, np.ndarray | PILImage.Image],
    ) -> dict[str, tuple[np.ndarray | None, float | None]]:
        results: dict[str, tuple[np.ndarray | None, float | None]] = {}
        ready_slots: list[str] = []

        for slot_id, obs in obs_images.items():
            if isinstance(obs, np.ndarray):
                obs = PILImage.fromarray(obs)
            self._push_obs(slot_id, obs)
            if self._is_ready(slot_id):
                ready_slots.append(slot_id)
            else:
                results[slot_id] = (None, None)

        if not ready_slots:
            return results

        obs_batch_list: list[torch.Tensor] = []
        goal_batch_list: list[torch.Tensor] = []

        for slot_id in ready_slots:
            context = list(self._context_queues[slot_id])
            g_img = goal_images[slot_id]
            obs_b, goal_b = self._obs_goal_tensors(context, g_img)
            obs_batch_list.append(obs_b)
            goal_batch_list.append(goal_b)

        obs_batch = torch.cat(obs_batch_list, dim=0)
        goal_batch = torch.cat(goal_batch_list, dim=0)

        dist_pred, action_pred = self._model(obs_batch, goal_batch)
        dist_np = dist_pred.flatten().cpu().numpy()

        for i, slot_id in enumerate(ready_slots):
            wp = action_pred[i, self._waypoint_index, :2].cpu().numpy().astype(
                np.float64
            )
            if self._normalize:
                wp = wp * self._metric_scale
            results[slot_id] = (wp, float(dist_np[i]))

        return results
