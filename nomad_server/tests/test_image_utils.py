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
