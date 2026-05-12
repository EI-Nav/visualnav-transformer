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
