from __future__ import annotations

import argparse
import sys
from pathlib import Path

from infer_nafnet_ddp import main as infer_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NAFNet inference for Kaggle using the best_infer config by default"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_infer_nafnet_best_infer.yaml",
        help="Path to config YAML/JSON passed to infer_nafnet_ddp.py",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    original_argv = sys.argv
    try:
        sys.argv = ["infer_nafnet_ddp.py", "--config", str(config_path)] + passthrough
        infer_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()