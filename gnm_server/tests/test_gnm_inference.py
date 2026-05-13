import numpy as np
import pytest


class TestGNMContextQueue:
    """Context queue behavior matches ViNT-style temporal windows."""

    def test_queue_starts_empty(self):
        from gnm_server.gnm_inference import GNMInference

        inf = GNMInference.__new__(GNMInference)
        inf._context_queues = {}
        inf._context_size = 3
        inf._init_slot("slot0")
        assert len(inf._context_queues["slot0"]) == 0

    def test_queue_ready(self):
        from gnm_server.gnm_inference import GNMInference

        inf = GNMInference.__new__(GNMInference)
        inf._context_queues = {}
        inf._context_size = 3
        inf._init_slot("slot0")
        for _ in range(4):
            inf._push_obs("slot0", _dummy_pil())
        assert inf._is_ready("slot0")

    def test_reset_clears_queue(self):
        from gnm_server.gnm_inference import GNMInference

        inf = GNMInference.__new__(GNMInference)
        inf._context_queues = {}
        inf._context_size = 3
        inf._init_slot("slot0")
        for _ in range(4):
            inf._push_obs("slot0", _dummy_pil())
        inf.reset("slot0")
        assert len(inf._context_queues["slot0"]) == 0


def _dummy_pil(size=(96, 96), fill=128):
    from PIL import Image

    arr = np.full((*size, 3), fill, dtype=np.uint8)
    return Image.fromarray(arr)
