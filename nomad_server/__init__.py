"""NoMaD NavArena evaluation server.

Heavy imports are lazy so ``nomad_server.image_utils`` can be used without
installing ``navarena_server`` (e.g. from ``vint_server``).
"""

from __future__ import annotations

__all__ = ["NoMaDInference", "NoMaDServer", "PDController"]


def __getattr__(name: str):
    if name == "NoMaDInference":
        from nomad_server.nomad_inference import NoMaDInference

        return NoMaDInference
    if name == "NoMaDServer":
        from nomad_server.nomad_server import NoMaDServer

        return NoMaDServer
    if name == "PDController":
        from nomad_server.pd_controller import PDController

        return PDController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
