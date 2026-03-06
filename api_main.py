"""Run MMis API server."""

from __future__ import annotations

import argparse
import os

import uvicorn

from config.settings import load_config, setup_logging
from utils.api_process_cleaner import clean_mmis_api_processes


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMis API server")
    parser.add_argument("--mmis-tag", default="mmis", help="MMis API process tag for cleanup")
    parser.add_argument(
        "--no-clean-tagged",
        action="store_true",
        help="Do not cleanup previously tagged MMis API processes before startup",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    cfg = load_config()
    setup_logging(cfg)
    if not bool(args.no_clean_tagged):
        clean_mmis_api_processes(
            tag=str(args.mmis_tag or "mmis"),
            root=os.getcwd(),
            exclude_pid=os.getpid(),
        )
    uvicorn.run("api.app:app", host=cfg.host, port=cfg.port, reload=False)
