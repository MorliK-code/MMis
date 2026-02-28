from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.paths import LOG_DIR
from config.settings import AppSettings, load_config


_IS_CONFIGURED = False


def setup_logging(settings: AppSettings | None = None, *, force: bool = False) -> None:
    global _IS_CONFIGURED
    if _IS_CONFIGURED and not force:
        return

    cfg = settings or load_config()
    log_level_name = str(os.getenv("MMIS_LOG_LEVEL", "DEBUG" if cfg.debug else "INFO")).strip().upper()
    level = getattr(logging, log_level_name, logging.INFO)

    log_dir = Path(str(os.getenv("MMIS_LOG_DIR", str(cfg.log_dir or LOG_DIR)))).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    if not root.handlers:
        root.addHandler(_console_handler(level))
        root.addHandler(_file_handler(log_dir / "app.log", level))

    # Optional category files.
    for channel, filename in (("llm", "llm.log"), ("memory", "memory.log"), ("tools", "tools.log"), ("ui", "ui.log")):
        logger = logging.getLogger(channel)
        logger.setLevel(level)
        if not _has_file_handler(logger, filename):
            logger.addHandler(_file_handler(log_dir / filename, level))
        logger.propagate = True

    logging.captureWarnings(True)
    _IS_CONFIGURED = True


def _console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    return handler


def _file_handler(path: Path, level: int) -> logging.Handler:
    handler = RotatingFileHandler(
        path,
        mode="a",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    return handler


def _has_file_handler(logger: logging.Logger, filename: str) -> bool:
    needle = str(filename).lower()
    for handler in logger.handlers:
        base = getattr(handler, "baseFilename", "")
        if base and str(base).lower().endswith(needle):
            return True
    return False


# Backward compatibility alias.
configure_logging = setup_logging
