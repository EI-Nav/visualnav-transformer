"""CLI entry point: ``python -m vint_server``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ViNT NavArena evaluation server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "train" / "config" / "vint.yaml"),
        help="Path to ViNT training YAML",
    )
    p.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "deployment" / "model_weights" / "vint.pth"),
        help="Path to ViNT checkpoint (.pth); default matches deployment/config/models.yaml",
    )
    p.add_argument(
        "--data-config",
        default=str(
            REPO_ROOT / "train" / "vint_train" / "data" / "data_config.yaml"
        ),
        help="Path to data_config.yaml (metric_waypoint_spacing per dataset)",
    )
    p.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset key for metric spacing (default: first key in training config)",
    )
    p.add_argument("--host", default="0.0.0.0", help="Server bind address")
    p.add_argument("--port", type=int, default=8766, help="Server port")
    p.add_argument("--device", default="cuda:0", help="Torch device")
    p.add_argument(
        "--waypoint-index",
        type=int,
        default=2,
        help="Index into predicted trajectory for the waypoint",
    )
    p.add_argument(
        "--no-center-crop",
        action="store_true",
        help="Disable 4:3 center crop before resize (training uses aspect crop)",
    )
    p.add_argument(
        "--max-v", type=float, default=0.5, help="Max forward velocity (m/s)"
    )
    p.add_argument(
        "--max-w", type=float, default=1.0, help="Max angular velocity (rad/s)"
    )
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

    import sys
    import torch

    logger.info(
        "Environment: Python %s, PyTorch %s, CUDA available=%s%s",
        sys.version.split()[0],
        torch.__version__,
        torch.cuda.is_available(),
        f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "",
    )
    logger.info(
        "Args: config=%s checkpoint=%s data_config=%s device=%s "
        "waypoint_index=%d max_v=%.3f max_w=%.3f dt=%.3f dataset_name=%s "
        "center_crop=%s log_level=%s",
        args.config,
        args.checkpoint,
        args.data_config,
        args.device,
        args.waypoint_index,
        args.max_v,
        args.max_w,
        args.dt,
        args.dataset_name,
        not args.no_center_crop,
        args.log_level,
    )

    from nomad_server.pd_controller import PDController
    from vint_server.vint_inference import ViNTInference
    from vint_server.vint_server import ViNTServer

    logger.info("Loading ViNT model from %s", args.checkpoint)
    inference = ViNTInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_config_path=args.data_config,
        device=args.device,
        waypoint_index=args.waypoint_index,
        dataset_name=args.dataset_name,
        center_crop=not args.no_center_crop,
    )

    controller = PDController(max_v=args.max_v, max_w=args.max_w, dt=args.dt)
    server = ViNTServer(inference=inference, controller=controller)

    logger.info("Starting ViNT server on ws://%s:%d", args.host, args.port)

    from navarena_server import serve

    serve(server, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
