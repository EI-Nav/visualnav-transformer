"""Image preprocessing utilities for NoMaD inference.

Extracted from deployment/src/utils.py with ROS dependencies removed.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

logger = logging.getLogger(__name__)


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
    if logger.isEnabledFor(logging.DEBUG):
        sizes = []
        for img in images:
            if isinstance(img, PILImage.Image):
                sizes.append(img.size)
            elif isinstance(img, np.ndarray):
                sizes.append(img.shape[:2][::-1])
            else:
                sizes.append("?")
        logger.debug(
            "transform_images: n=%d, input_sizes=%s, target=%s, center_crop=%s",
            len(images), sizes, image_size, center_crop,
        )

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
    result = torch.cat(tensors, dim=1)
    logger.debug("transform_images: output tensor %s", list(result.shape))
    return result
