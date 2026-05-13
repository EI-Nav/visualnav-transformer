"""ViNT NavArena evaluation server.

``ViNTServer`` imports ``navarena_server``; keep package ``__init__`` lazy so
``python -m vint_server --help`` works without NavArena installed.
"""

from __future__ import annotations

__all__ = ["ViNTInference", "ViNTServer"]


def __getattr__(name: str):
    if name == "ViNTInference":
        from vint_server.vint_inference import ViNTInference

        return ViNTInference
    if name == "ViNTServer":
        from vint_server.vint_server import ViNTServer

        return ViNTServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
