"""GNM NavArena evaluation server.

``GNMServer`` imports ``navarena_server``; keep package ``__init__`` lazy so
``python -m gnm_server --help`` works without NavArena installed.
"""

from __future__ import annotations

__all__ = ["GNMInference", "GNMServer"]


def __getattr__(name: str):
    if name == "GNMInference":
        from gnm_server.gnm_inference import GNMInference

        return GNMInference
    if name == "GNMServer":
        from gnm_server.gnm_server import GNMServer

        return GNMServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
