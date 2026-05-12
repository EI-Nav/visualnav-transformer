"""NoMaD model inference wrapper for NavArena evaluation.

Handles model loading, context queue management, DDPM denoising, and
action extraction for ImageNav tasks.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from nomad_server.image_utils import transform_images

logger = logging.getLogger(__name__)


class NoMaDInference:
    """Wraps the NoMaD model for single-step or batched inference.

    Args:
        config_path: Path to model YAML config (e.g. ``train/config/nomad.yaml``).
        checkpoint_path: Path to model checkpoint (``.pth``).
        data_config_path: Path to data config with ``action_stats``
            (e.g. ``train/vint_train/data/data_config.yaml``).
        device: Torch device string.
        num_samples: Number of trajectories sampled during DDPM denoising.
        waypoint_index: Index into the predicted trajectory to select waypoint.
        normalize: Whether to apply velocity-based scaling to waypoints.
        max_v: Max forward velocity for normalization scaling (m/s).
        frame_rate: Frame rate for normalization scaling (Hz).
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        data_config_path: str,
        device: str = "cuda:0",
        num_samples: int = 8,
        waypoint_index: int = 2,
        normalize: bool | None = None,
        max_v: float = 0.2,
        frame_rate: float = 4.0,
    ) -> None:
        self._device = torch.device(device)

        with open(config_path) as f:
            self._config: dict[str, Any] = yaml.safe_load(f)

        with open(data_config_path) as f:
            data_config = yaml.safe_load(f)
        self._action_stats = {
            k: np.array(v) for k, v in data_config["action_stats"].items()
        }

        self._context_size: int = self._config["context_size"]
        self._image_size: list[int] = self._config["image_size"]
        self._len_traj_pred: int = self._config["len_traj_pred"]
        self._num_diffusion_iters: int = self._config["num_diffusion_iters"]
        self._num_samples = num_samples
        self._waypoint_index = waypoint_index

        if normalize is None:
            self._normalize: bool = self._config.get("normalize", False)
        else:
            self._normalize = normalize
        self._max_v = max_v
        self._frame_rate = frame_rate

        self._model = self._load_model(checkpoint_path)
        self._model.eval()

        self._noise_scheduler = DDPMScheduler(
            num_train_timesteps=self._num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self._context_queues: dict[str, deque[PILImage.Image]] = {}

        logger.info(
            "NoMaDInference ready: context_size=%d, image_size=%s, "
            "len_traj_pred=%d, num_diffusion_iters=%d, num_samples=%d, "
            "waypoint_index=%d, normalize=%s, device=%s",
            self._context_size, self._image_size, self._len_traj_pred,
            self._num_diffusion_iters, self._num_samples, self._waypoint_index,
            self._normalize, self._device,
        )
        logger.info(
            "action_stats: min=%s, max=%s",
            self._action_stats["min"], self._action_stats["max"],
        )

    # -- context queue management --

    def _init_slot(self, slot_id: str) -> None:
        self._context_queues[slot_id] = deque(maxlen=self._context_size + 1)

    def reset(self, slot_id: str) -> None:
        """Clear the context queue for *slot_id*."""
        self._context_queues[slot_id] = deque(maxlen=self._context_size + 1)
        logger.debug("Reset context queue for slot %s", slot_id)

    def _push_obs(self, slot_id: str, pil_img: PILImage.Image) -> None:
        if slot_id not in self._context_queues:
            self._init_slot(slot_id)
        self._context_queues[slot_id].append(pil_img)
        q = self._context_queues[slot_id]
        logger.debug(
            "slot %s: context queue %d/%d (need >%d to be ready)",
            slot_id, len(q), q.maxlen, self._context_size,
        )

    def _is_ready(self, slot_id: str) -> bool:
        return len(self._context_queues.get(slot_id, [])) > self._context_size

    # -- model loading --

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
        from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

        cfg = self._config
        logger.info(
            "Building model: vision_encoder=%s, encoding_size=%d",
            cfg.get("vision_encoder"), cfg["encoding_size"],
        )
        if cfg.get("vision_encoder") == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=cfg["encoding_size"],
                context_size=cfg["context_size"],
                mha_num_attention_heads=cfg["mha_num_attention_heads"],
                mha_num_attention_layers=cfg["mha_num_attention_layers"],
                mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Unsupported vision encoder: {cfg.get('vision_encoder')}")

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=cfg["encoding_size"],
            down_dims=cfg["down_dims"],
            cond_predict_scale=cfg["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=cfg["encoding_size"])

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        num_params = sum(p.numel() for p in model.parameters())
        logger.info("Model built: %.2fM parameters", num_params / 1e6)

        logger.info("Loading checkpoint: %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            logger.warning(
                "Checkpoint missing %d keys: %s",
                len(load_result.missing_keys), load_result.missing_keys,
            )
        if load_result.unexpected_keys:
            logger.warning(
                "Checkpoint has %d unexpected keys: %s",
                len(load_result.unexpected_keys), load_result.unexpected_keys,
            )
        if not load_result.missing_keys and not load_result.unexpected_keys:
            logger.info("Checkpoint loaded cleanly (no missing/unexpected keys)")

        model.to(self._device)
        logger.info("Model moved to %s", self._device)
        return model

    # -- inference --

    @torch.no_grad()
    def infer(
        self,
        obs_image: np.ndarray | PILImage.Image,
        goal_image: np.ndarray | PILImage.Image,
        slot_id: str,
    ) -> tuple[np.ndarray | None, float | None]:
        """Run a single-step inference for one slot.

        Returns ``(waypoint, distance)`` or ``(None, None)`` if context is
        not yet full.
        """
        if isinstance(obs_image, np.ndarray):
            obs_image = PILImage.fromarray(obs_image)
        self._push_obs(slot_id, obs_image)

        if not self._is_ready(slot_id):
            q = self._context_queues.get(slot_id, [])
            logger.debug(
                "slot %s: context not ready (%d/%d), returning (None, None)",
                slot_id, len(q), self._context_size + 1,
            )
            return None, None

        if isinstance(goal_image, np.ndarray):
            goal_image = PILImage.fromarray(goal_image)

        t0 = time.monotonic()

        context = list(self._context_queues[slot_id])
        obs_tensor = transform_images(context, self._image_size, center_crop=False)
        obs_tensor = torch.split(obs_tensor, 3, dim=1)
        obs_tensor = torch.cat(obs_tensor, dim=1).to(self._device)

        goal_tensor = transform_images(
            [goal_image], self._image_size, center_crop=False
        ).to(self._device)

        logger.debug(
            "slot %s: obs_tensor %s, goal_tensor %s",
            slot_id, list(obs_tensor.shape), list(goal_tensor.shape),
        )

        mask = torch.zeros(1, dtype=torch.long, device=self._device)

        obsgoal_cond = self._model(
            "vision_encoder",
            obs_img=obs_tensor,
            goal_img=goal_tensor,
            input_goal_mask=mask,
        )

        distance = self._model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dist_val = float(distance.flatten().cpu().numpy()[0])

        obs_cond = obsgoal_cond.repeat(self._num_samples, 1)

        naction = torch.randn(
            (self._num_samples, self._len_traj_pred, 2), device=self._device
        )
        self._noise_scheduler.set_timesteps(self._num_diffusion_iters)
        for k in self._noise_scheduler.timesteps:
            noise_pred = self._model(
                "noise_pred_net",
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )
            naction = self._noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        waypoints = self._get_action(naction)  # (num_samples, len_traj_pred, 2)
        chosen = waypoints[0].cpu().numpy()  # first sample
        wp = chosen[self._waypoint_index]  # (2,)

        elapsed = time.monotonic() - t0
        logger.debug(
            "slot %s: raw waypoint=(%.4f, %.4f), distance=%.4f, "
            "inference_time=%.3fs",
            slot_id, wp[0], wp[1], dist_val, elapsed,
        )

        if self._normalize:
            wp_before = wp.copy()
            wp = wp * (self._max_v / self._frame_rate)
            logger.debug(
                "slot %s: normalized waypoint (%.4f, %.4f) -> (%.4f, %.4f) "
                "(max_v=%.3f, frame_rate=%.1f)",
                slot_id, wp_before[0], wp_before[1], wp[0], wp[1],
                self._max_v, self._frame_rate,
            )

        return wp, dist_val

    @torch.no_grad()
    def batch_infer(
        self,
        obs_images: dict[str, np.ndarray | PILImage.Image],
        goal_images: dict[str, np.ndarray | PILImage.Image],
    ) -> dict[str, tuple[np.ndarray | None, float | None]]:
        """Batched inference across multiple slots.

        Slots whose context queues are not yet full receive ``(None, None)``.
        Ready slots are batched together for a single vision encoder forward
        pass; DDPM denoising runs per-slot.
        """
        results: dict[str, tuple[np.ndarray | None, float | None]] = {}
        ready_slots: list[str] = []
        not_ready_slots: list[str] = []

        for slot_id, obs in obs_images.items():
            if isinstance(obs, np.ndarray):
                obs = PILImage.fromarray(obs)
            self._push_obs(slot_id, obs)
            if self._is_ready(slot_id):
                ready_slots.append(slot_id)
            else:
                not_ready_slots.append(slot_id)
                results[slot_id] = (None, None)

        logger.debug(
            "batch_infer: total=%d, ready=%d, not_ready=%d %s",
            len(obs_images), len(ready_slots), len(not_ready_slots),
            not_ready_slots if not_ready_slots else "",
        )

        if not ready_slots:
            logger.debug("batch_infer: no ready slots, returning all (None, None)")
            return results

        t0 = time.monotonic()

        obs_tensors = []
        goal_tensors = []
        for slot_id in ready_slots:
            context = list(self._context_queues[slot_id])
            obs_t = transform_images(context, self._image_size, center_crop=False)
            obs_t = torch.split(obs_t, 3, dim=1)
            obs_t = torch.cat(obs_t, dim=1)
            obs_tensors.append(obs_t)

            g_img = goal_images[slot_id]
            if isinstance(g_img, np.ndarray):
                g_img = PILImage.fromarray(g_img)
            goal_t = transform_images([g_img], self._image_size, center_crop=False)
            goal_tensors.append(goal_t)

        obs_batch = torch.cat(obs_tensors, dim=0).to(self._device)
        goal_batch = torch.cat(goal_tensors, dim=0).to(self._device)
        mask = torch.zeros(len(ready_slots), dtype=torch.long, device=self._device)

        logger.debug(
            "batch_infer: obs_batch %s, goal_batch %s",
            list(obs_batch.shape), list(goal_batch.shape),
        )

        obsgoal_cond = self._model(
            "vision_encoder",
            obs_img=obs_batch,
            goal_img=goal_batch,
            input_goal_mask=mask,
        )

        distances = self._model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dist_np = distances.flatten().cpu().numpy()

        for i, slot_id in enumerate(ready_slots):
            obs_cond = obsgoal_cond[i].unsqueeze(0).repeat(self._num_samples, 1)
            naction = torch.randn(
                (self._num_samples, self._len_traj_pred, 2), device=self._device
            )
            self._noise_scheduler.set_timesteps(self._num_diffusion_iters)
            for k in self._noise_scheduler.timesteps:
                noise_pred = self._model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond,
                )
                naction = self._noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

            waypoints = self._get_action(naction)
            chosen = waypoints[0].cpu().numpy()
            wp = chosen[self._waypoint_index]

            if self._normalize:
                wp = wp * (self._max_v / self._frame_rate)

            results[slot_id] = (wp, float(dist_np[i]))
            logger.debug(
                "batch_infer: slot %s -> wp=(%.4f, %.4f), dist=%.4f",
                slot_id, wp[0], wp[1], float(dist_np[i]),
            )

        elapsed = time.monotonic() - t0
        logger.debug(
            "batch_infer: %d slots processed in %.3fs",
            len(ready_slots), elapsed,
        )

        return results

    # -- action processing --

    def _get_action(self, diffusion_output: torch.Tensor) -> torch.Tensor:
        """Convert normalized diffusion deltas to cumulative waypoints."""
        ndeltas = diffusion_output.cpu().numpy()
        logger.debug(
            "_get_action: diffusion_output range [%.4f, %.4f]",
            ndeltas.min(), ndeltas.max(),
        )
        ndeltas = self._unnormalize(ndeltas, self._action_stats)
        logger.debug(
            "_get_action: after unnormalize range [%.4f, %.4f]",
            ndeltas.min(), ndeltas.max(),
        )
        actions = np.cumsum(ndeltas, axis=1)
        return torch.from_numpy(actions).float().to(diffusion_output.device)

    @staticmethod
    def _unnormalize(ndata: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
        ndata = (ndata + 1) / 2
        return ndata * (stats["max"] - stats["min"]) + stats["min"]
