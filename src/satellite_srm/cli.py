"""Command-line interface for Satellite-SRM."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from satellite_srm.config import load_config
from satellite_srm.device import select_device
from satellite_srm.logging_config import configure_logging


def _print_setup_banner() -> None:
    print("Satellite-SRM setup")
    print("- CUDA device preference: enabled when available")
    print("- Default config loaded from configs/default.yaml")
    print("- Logs written to logs/satellite_srm.log")


def cmd_setup(_args: argparse.Namespace) -> int:
    cfg = load_config()
    device = select_device(str(cfg.get("device", "cuda")).lower())
    configure_logging(str(cfg.get("log_level", "INFO")))
    _print_setup_banner()
    print(f"Detected runtime device: {device}")
    print("Project environment is ready for training/inference.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    print("Training entry is available. Run the training workflow from the project scripts or configure a dataset first.")
    print(f"Requested epochs: {args.epochs}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    print("Inference entry is available.")
    if args.input:
        print(f"Input: {args.input}")
    if args.output:
        print(f"Output: {args.output}")
    return 0


def cmd_validate(_args: argparse.Namespace) -> int:
    print("Validation workflow is ready to run once data is prepared.")
    return 0


def cmd_download_data(args: argparse.Namespace) -> int:
    from satellite_srm.acquisition.sentinel2 import Sentinel2Downloader
    print("Starting data download workflow...")
    downloader = Sentinel2Downloader()
    scene_id = getattr(args, "scene_id", "S2A_MSIL2A_20260515_agriculture")
    bbox = getattr(args, "bbox", [13.4, 52.5, 13.5, 52.6])
    downloader.acquire_scene(scene_id=scene_id, bbox=bbox)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="satellite-srm",
        description="Satellite-SRM: deep learning-based super-resolution mapping for multispectral satellite imagery.",
    )
    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser("setup", help="run environment setup and GPU detection")
    setup_parser.set_defaults(func=cmd_setup)

    train_parser = subparsers.add_parser("train", help="train the SRM model")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.set_defaults(func=cmd_train)

    infer_parser = subparsers.add_parser("infer", help="run full-scene inference")
    infer_parser.add_argument("--input", type=str, default="")
    infer_parser.add_argument("--output", type=str, default="")
    infer_parser.set_defaults(func=cmd_infer)

    validate_parser = subparsers.add_parser("validate", help="validate dataset or model outputs")
    validate_parser.set_defaults(func=cmd_validate)

    download_parser = subparsers.add_parser("download-data", help="download satellite data")
    download_parser.add_argument("--scene-id", dest="scene_id", type=str, default="S2A_MSIL2A_20260515_agriculture")
    download_parser.set_defaults(func=cmd_download_data)

    for name in [
        "prepare-data",
        "validate-data",
        "download-models",
        "app",
        "demo",
    ]:
        sub = subparsers.add_parser(name, help=f"{name} command")
        sub.set_defaults(func=lambda _a, _name=name: print(f"{_name} command is available but not fully wired in this environment."))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    if os.environ.get("DEVICE"):
        os.environ["DEVICE"] = os.environ["DEVICE"].lower()

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
