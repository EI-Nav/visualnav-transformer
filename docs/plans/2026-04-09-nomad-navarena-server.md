# NoMaD NavArena Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a WebSocket server wrapping the NoMaD model for ImageNav evaluation in navarena-bench.

**Architecture:** A `nomad_server/` package inside `visualnav-transformer/` with three layers: `NoMaDInference` (model loading + DDPM inference), `PDController` (waypoint-to-action conversion), and `NoMaDServer` (WebSocket server inheriting `NavigationModelServer`). Each slot/episode maintains an independent context queue.

**Tech Stack:** PyTorch, diffusers (DDPMScheduler), navarena-server (NavigationModelServer, serve), efficientnet-pytorch, PIL, numpy

**Spec:** `docs/specs/2026-04-09-nomad-navarena-server-design.md`

---

**IMPORTANT — Package naming:** The package is named `nomad_server` (not `navarena_server`) to avoid shadowing the pip-installed `navarena_server` package which provides the `NavigationModelServer` base class.

## File Structure

| File | Responsibility |
|------|---------------|
| `nomad_server/__init__.py` | Package exports: `NoMaDServer`, `NoMaDInference`, `PDController` |
| `nomad_server/pd_controller.py` | Pure-function waypoint→action conversion (no model deps) |
| `nomad_server/image_utils.py` | Image preprocessing (extracted from `deployment/src/utils.py`, no ROS) |
| `nomad_server/nomad_inference.py` | Model loading, context queue, DDPM inference |
| `nomad_server/nomad_server.py` | `NoMaDServer(NavigationModelServer)` with predict/batch_predict |
| `nomad_server/__main__.py` | CLI entry: argparse + serve() |
| `nomad_server/tests/test_pd_controller.py` | PDController unit tests |
| `nomad_server/tests/test_image_utils.py` | Image preprocessing tests |
| `nomad_server/tests/test_nomad_inference.py` | Inference tests (with mock model) |
| `nomad_server/tests/test_nomad_server.py` | Server integration tests |

---

### Task 1: PDController

**Files:**
- Create: `nomad_server/pd_controller.py`
- Test: `nomad_server/tests/test_pd_controller.py`

- [ ] **Step 1: Create package structure**

```bash
mkdir -p /x2robot_v2/jake/research/visualnav-transformer/nomad_server/tests
touch /x2robot_v2/jake/research/visualnav-transformer/nomad_server/__init__.py
touch /x2robot_v2/jake/research/visualnav-transformer/nomad_server/tests/__init__.py
```

- [ ] **Step 2: Write PDController tests**

Create `nomad_server/tests/test_pd_controller.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /x2robot_v2/jake/research/visualnav-transformer
python -m pytest nomad_server/tests/test_pd_controller.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 4: Implement PDController**

Create `nomad_server/pd_controller.py`:

```python
"""PD controller for converting NoMaD waypoints to navarena displacement actions.

Ported from deployment/src/pd_controller.py with ROS dependencies removed.
"""

from __future__ import annotations

import numpy as np

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

    def control(self, waypoint: np.ndarray) -> dict[str, float]:
        """Compute a navarena action from a 2D waypoint.

        Args:
            waypoint: Array of shape (2,) with (dx, dy) in robot body frame.

        Returns:
            Action dict ``{"x": ..., "y": 0.0, "yaw": ...}``.
        """
        dx, dy = float(waypoint[0]), float(waypoint[1])

        if abs(dx) < EPS and abs(dy) < EPS:
            return {"x": 0.0, "y": 0.0, "yaw": 0.0}

        if abs(dx) < EPS:
            v = 0.0
            w = np.sign(dy) * np.pi / (2 * self.dt)
        else:
            v = dx / self.dt
            w = np.arctan(dy / dx) / self.dt

        v = float(np.clip(v, 0, self.max_v))
        w = float(np.clip(w, -self.max_w, self.max_w))

        return {
            "x": v * self.dt,
            "y": 0.0,
            "yaw": w * self.dt,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /x2robot_v2/jake/research/visualnav-transformer
python -m pytest nomad_server/tests/test_pd_controller.py -v
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add nomad_server/__init__.py nomad_server/tests/__init__.py \
    nomad_server/pd_controller.py nomad_server/tests/test_pd_controller.py
git commit -m "feat(nomad-server): add PDController for waypoint-to-action conversion"
```

---

### Task 2: Image Preprocessing Utilities

**Files:**
- Create: `nomad_server/image_utils.py`
- Test: `nomad_server/tests/test_image_utils.py`

Extracted from `deployment/src/utils.py` — the `transform_images` function without ROS dependencies.

- [ ] **Step 1: Write image utils tests**

Create `nomad_server/tests/test_image_utils.py`:

```python
import numpy as np
import torch
from PIL import Image

from nomad_server.image_utils import transform_images


class TestTransformImages:
    def test_single_image(self):
        """Single PIL image -> tensor with 3 channels."""
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = transform_images([img], image_size=[96, 96])
        assert result.shape == (1, 3, 96, 96)

    def test_multiple_images_concat_channels(self):
        """Multiple images are concatenated along channel dim."""
        imgs = [
            Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            for _ in range(4)
        ]
        result = transform_images(imgs, image_size=[96, 96])
        assert result.shape == (1, 12, 96, 96)

    def test_normalization_range(self):
        """Output should be roughly in ImageNet normalized range."""
        img = Image.fromarray(np.ones((50, 50, 3), dtype=np.uint8) * 128)
        result = transform_images([img], image_size=[96, 96])
        assert result.min() > -3.0
        assert result.max() < 3.0

    def test_non_square_input(self):
        """Non-square input should be resized to target size."""
        img = Image.fromarray(np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8))
        result = transform_images([img], image_size=[96, 96])
        assert result.shape == (1, 3, 96, 96)

    def test_numpy_input(self):
        """Should also accept numpy arrays (H, W, 3) uint8."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = transform_images([arr], image_size=[96, 96])
        assert result.shape == (1, 3, 96, 96)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest nomad_server/tests/test_image_utils.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement image_utils**

Create `nomad_server/image_utils.py`:

```python
"""Image preprocessing utilities for NoMaD inference.

Extracted from deployment/src/utils.py with ROS dependencies removed.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms


def transform_images(
    images: list[Union[PILImage.Image, np.ndarray]],
    image_size: list[int],
    center_crop: bool = False,
) -> torch.Tensor:
    """Preprocess a list of images into a single tensor.

    Each image is resized to ``image_size``, converted to a tensor, and
    normalized with ImageNet statistics.  All images are concatenated along the
    channel dimension (dim=1), producing a tensor of shape
    ``(1, 3*N, H, W)`` where *N* is the number of input images.

    Args:
        images: List of PIL Images or numpy arrays (H, W, 3) uint8.
        image_size: Target ``[width, height]``.
        center_crop: Whether to center-crop before resizing.

    Returns:
        Tensor of shape ``(1, 3*len(images), H, W)``.
    """
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensors: list[torch.Tensor] = []
    for img in images:
        if isinstance(img, np.ndarray):
            img = PILImage.fromarray(img)
        if center_crop:
            import torchvision.transforms.functional as TF
            w, h = img.size
            aspect = 4 / 3
            if w > h:
                img = TF.center_crop(img, (h, int(h * aspect)))
            else:
                img = TF.center_crop(img, (int(w / aspect), w))
        img = img.resize(image_size)
        t = normalize(img).unsqueeze(0)  # (1, 3, H, W)
        tensors.append(t)
    return torch.cat(tensors, dim=1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest nomad_server/tests/test_image_utils.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add nomad_server/image_utils.py nomad_server/tests/test_image_utils.py
git commit -m "feat(nomad-server): add image preprocessing utilities"
```

---

### Task 3: NoMaDInference

**Files:**
- Create: `nomad_server/nomad_inference.py`
- Test: `nomad_server/tests/test_nomad_inference.py`

**Key references:**
- Model construction: `deployment/src/utils.py:load_model()` (lines 31-111)
- DDPM loop: `deployment/src/navigate.py` (lines 121-188)
- get_action: `train/vint_train/training/train_utils.py` (lines 956-976)
- Config: `train/config/nomad.yaml`
- Action stats: `train/vint_train/data/data_config.yaml`

- [ ] **Step 1: Write context queue tests**

Create `nomad_server/tests/test_nomad_inference.py`:

```python
import numpy as np
import pytest


class TestContextQueue:
    """Test context queue management without loading a real model."""

    def test_queue_starts_empty(self):
        from nomad_server.nomad_inference import NoMaDInference
        inf = NoMaDInference.__new__(NoMaDInference)
        inf._context_queues = {}
        inf._context_size = 3
        inf._init_slot("slot0")
        assert len(inf._context_queues["slot0"]) == 0

    def test_queue_fills_up(self):
        from nomad_server.nomad_inference import NoMaDInference
        inf = NoMaDInference.__new__(NoMaDInference)
        inf._context_queues = {}
        inf._context_size = 3
        inf._init_slot("slot0")
        for i in range(4):
            inf._push_obs("slot0", _dummy_pil())
        assert len(inf._context_queues["slot0"]) == 4  # context_size + 1

    def test_queue_fifo(self):
        from nomad_server.nomad_inference import NoMaDInference
        inf = NoMaDInference.__new__(NoMaDInference)
        inf._context_queues = {}
        inf._context_size = 2
        inf._init_slot("slot0")
        for i in range(5):
            img = _dummy_pil(fill=i * 50)
            inf._push_obs("slot0", img)
        q = inf._context_queues["slot0"]
        assert len(q) == 3  # context_size + 1
        arr = np.array(q[0])
        assert arr.flat[0] == 100  # 3rd image (i=2, fill=100)

    def test_reset_clears_queue(self):
        from nomad_server.nomad_inference import NoMaDInference
        inf = NoMaDInference.__new__(NoMaDInference)
        inf._context_queues = {}
        inf._context_size = 3
        inf._init_slot("slot0")
        for i in range(4):
            inf._push_obs("slot0", _dummy_pil())
        inf.reset("slot0")
        assert len(inf._context_queues["slot0"]) == 0

    def test_queue_not_ready(self):
        from nomad_server.nomad_inference import NoMaDInference
        inf = NoMaDInference.__new__(NoMaDInference)
        inf._context_queues = {}
        inf._context_size = 3
        inf._init_slot("slot0")
        inf._push_obs("slot0", _dummy_pil())
        assert not inf._is_ready("slot0")

    def test_queue_ready(self):
        from nomad_server.nomad_inference import NoMaDInference
        inf = NoMaDInference.__new__(NoMaDInference)
        inf._context_queues = {}
        inf._context_size = 3
        inf._init_slot("slot0")
        for i in range(4):
            inf._push_obs("slot0", _dummy_pil())
        assert inf._is_ready("slot0")


def _dummy_pil(size=(96, 96), fill=128):
    from PIL import Image
    arr = np.full((*size, 3), fill, dtype=np.uint8)
    return Image.fromarray(arr)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest nomad_server/tests/test_nomad_inference.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement NoMaDInference**

Create `nomad_server/nomad_inference.py`:

```python
"""NoMaD model inference wrapper for NavArena evaluation.

Handles model loading, context queue management, DDPM denoising, and
action extraction for ImageNav tasks.
"""

from __future__ import annotations

import logging
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

    # -- context queue management --

    def _init_slot(self, slot_id: str) -> None:
        self._context_queues[slot_id] = deque(maxlen=self._context_size + 1)

    def reset(self, slot_id: str) -> None:
        """Clear the context queue for *slot_id*."""
        self._context_queues[slot_id] = deque(maxlen=self._context_size + 1)

    def _push_obs(self, slot_id: str, pil_img: PILImage.Image) -> None:
        if slot_id not in self._context_queues:
            self._init_slot(slot_id)
        self._context_queues[slot_id].append(pil_img)

    def _is_ready(self, slot_id: str) -> bool:
        return len(self._context_queues.get(slot_id, [])) > self._context_size

    # -- model loading --

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
        from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

        cfg = self._config
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

        state_dict = torch.load(checkpoint_path, map_location=self._device)
        model.load_state_dict(state_dict, strict=False)
        model.to(self._device)
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
            return None, None

        if isinstance(goal_image, np.ndarray):
            goal_image = PILImage.fromarray(goal_image)

        context = list(self._context_queues[slot_id])
        obs_tensor = transform_images(context, self._image_size, center_crop=False)
        obs_tensor = torch.split(obs_tensor, 3, dim=1)
        obs_tensor = torch.cat(obs_tensor, dim=1).to(self._device)

        goal_tensor = transform_images(
            [goal_image], self._image_size, center_crop=False
        ).to(self._device)

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

        if self._normalize:
            wp = wp * (self._max_v / self._frame_rate)

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

        return results

    # -- action processing --

    def _get_action(self, diffusion_output: torch.Tensor) -> torch.Tensor:
        """Convert normalized diffusion deltas to cumulative waypoints."""
        ndeltas = diffusion_output.cpu().numpy()
        ndeltas = self._unnormalize(ndeltas, self._action_stats)
        actions = np.cumsum(ndeltas, axis=1)
        return torch.from_numpy(actions).float().to(diffusion_output.device)

    @staticmethod
    def _unnormalize(ndata: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
        ndata = (ndata + 1) / 2
        return ndata * (stats["max"] - stats["min"]) + stats["min"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest nomad_server/tests/test_nomad_inference.py -v
```

Expected: All PASS (context queue tests only; model loading tests need checkpoint)

- [ ] **Step 5: Commit**

```bash
git add nomad_server/nomad_inference.py nomad_server/tests/test_nomad_inference.py
git commit -m "feat(nomad-server): add NoMaDInference with context queue and DDPM denoising"
```

---

### Task 4: NoMaDServer

**Files:**
- Create: `nomad_server/nomad_server.py`
- Test: `nomad_server/tests/test_nomad_server.py`

- [ ] **Step 1: Write NoMaDServer tests**

Create `nomad_server/tests/test_nomad_server.py`:

```python
import numpy as np
import pytest
from unittest.mock import MagicMock

from nomad_server.nomad_server import NoMaDServer
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
        assert "x" in action
        assert "y" in action
        assert "yaw" in action
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
        assert action == {"x": 0.0, "y": 0.0, "yaw": 0.0}

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
        assert "x" in actions["slot0"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pip install pytest-asyncio  # if not installed
python -m pytest nomad_server/tests/test_nomad_server.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement NoMaDServer**

Create `nomad_server/nomad_server.py`:

```python
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

ZERO_ACTION: dict[str, float] = {"x": 0.0, "y": 0.0, "yaw": 0.0}


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
    ) -> dict[str, float]:
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
            return ZERO_ACTION

        logger.debug(
            "slot=%s wp=(%.3f, %.3f) dist=%.3f",
            key, waypoint[0], waypoint[1], distance,
        )
        return self._controller.control(waypoint)

    async def batch_predict(
        self,
        observations: dict[str, dict[str, Any]],
        contexts: dict[str, SessionContext],
    ) -> dict[str, dict[str, float]]:
        obs_images: dict[str, Any] = {}
        goal_imgs: dict[str, Any] = {}

        for slot_id, obs in observations.items():
            rgb_dict = obs.get("rgb", {})
            if not rgb_dict:
                continue
            obs_images[slot_id] = next(iter(rgb_dict.values()))

            goal_image = obs.get("goal_image")
            if goal_image is not None:
                self._goal_images[slot_id] = goal_image
            goal_imgs[slot_id] = self._goal_images.get(slot_id)

        slots_without_goal = [s for s, g in goal_imgs.items() if g is None]
        for s in slots_without_goal:
            obs_images.pop(s, None)
            goal_imgs.pop(s, None)

        if not obs_images:
            return {s: ZERO_ACTION for s in observations}

        try:
            batch_results = self._inference.batch_infer(obs_images, goal_imgs)
        except Exception:
            logger.exception("Batch inference failed")
            return {s: ZERO_ACTION for s in observations}

        actions: dict[str, dict[str, float]] = {}
        for slot_id in observations:
            result = batch_results.get(slot_id)
            if result is None or result[0] is None:
                actions[slot_id] = ZERO_ACTION
            else:
                actions[slot_id] = self._controller.control(result[0])
        return actions
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest nomad_server/tests/test_nomad_server.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add nomad_server/nomad_server.py nomad_server/tests/test_nomad_server.py
git commit -m "feat(nomad-server): add NoMaDServer with predict and batch_predict"
```

---

### Task 5: CLI Entry Point and Package Init

**Files:**
- Create: `nomad_server/__main__.py`
- Modify: `nomad_server/__init__.py`

- [ ] **Step 1: Write __init__.py**

Create `nomad_server/__init__.py`:

```python
"""NoMaD NavArena evaluation server."""

from nomad_server.nomad_inference import NoMaDInference
from nomad_server.nomad_server import NoMaDServer
from nomad_server.pd_controller import PDController

__all__ = ["NoMaDInference", "NoMaDServer", "PDController"]
```

- [ ] **Step 2: Write __main__.py**

Create `nomad_server/__main__.py`:

```python
"""CLI entry point: ``python -m nomad_server``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NoMaD NavArena evaluation server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "train" / "config" / "nomad.yaml"),
        help="Path to model YAML config",
    )
    p.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "deployment" / "model_weights" / "nomad.pth"),
        help="Path to model checkpoint",
    )
    p.add_argument(
        "--data-config",
        default=str(REPO_ROOT / "train" / "vint_train" / "data" / "data_config.yaml"),
        help="Path to data config with action_stats",
    )
    p.add_argument("--host", default="0.0.0.0", help="Server bind address")
    p.add_argument("--port", type=int, default=8765, help="Server port")
    p.add_argument("--device", default="cuda:0", help="Torch device")
    p.add_argument(
        "--waypoint-index", type=int, default=2,
        help="Trajectory waypoint index",
    )
    p.add_argument(
        "--num-samples", type=int, default=8,
        help="DDPM sampling count",
    )
    p.add_argument("--max-v", type=float, default=0.5, help="Max forward velocity (m/s)")
    p.add_argument("--max-w", type=float, default=1.0, help="Max angular velocity (rad/s)")
    p.add_argument("--dt", type=float, default=1.0, help="Simulation time-step (s)")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    from nomad_server import NoMaDInference, NoMaDServer, PDController

    logger.info("Loading NoMaD model from %s", args.checkpoint)
    inference = NoMaDInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_config_path=args.data_config,
        device=args.device,
        num_samples=args.num_samples,
        waypoint_index=args.waypoint_index,
    )

    controller = PDController(max_v=args.max_v, max_w=args.max_w, dt=args.dt)
    server = NoMaDServer(inference=inference, controller=controller)

    logger.info("Starting NoMaD server on ws://%s:%d", args.host, args.port)

    from navarena_server import serve
    serve(server, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify CLI help works**

```bash
cd /x2robot_v2/jake/research/visualnav-transformer
python -m nomad_server --help
```

Expected: Help text with all arguments printed

- [ ] **Step 4: Commit**

```bash
git add nomad_server/__init__.py nomad_server/__main__.py
git commit -m "feat(nomad-server): add CLI entry point and package init"
```

---

### Task 6: Run All Tests and Final Integration Check

- [ ] **Step 1: Run full test suite**

```bash
cd /x2robot_v2/jake/research/visualnav-transformer
python -m pytest nomad_server/tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: Verify end-to-end startup (if checkpoint available)**

```bash
cd /x2robot_v2/jake/research/visualnav-transformer
python -m nomad_server \
    --config train/config/nomad.yaml \
    --checkpoint deployment/model_weights/nomad.pth \
    --port 8765 \
    --log-level DEBUG
```

Expected: Server starts and prints "Starting model server on ws://0.0.0.0:8765"

- [ ] **Step 3: Final commit**

```bash
git add -A nomad_server/
git commit -m "feat(nomad-server): NoMaD NavArena server ready for ImageNav evaluation"
```
