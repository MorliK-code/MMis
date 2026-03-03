from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.paths import LOG_DIR
from config.settings import AppSettings, load_config


_IS_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_CHANNEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "llm": ("llm",),
    "memory": ("memory", "metadata"),
    "tools": ("modules", "tools"),
    "ui": ("ui", "api", "ui_console", "ui_pyside6"),
}


class _LoggerPrefixFilter(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self.prefixes = tuple(str(x or "").strip().lower() for x in prefixes if str(x or "").strip())

    def filter(self, record: logging.LogRecord) -> bool:
        name = str(getattr(record, "name", "") or "").strip().lower()
        if not name:
            return False
        for prefix in self.prefixes:
            if name == prefix or name.startswith(prefix + "."):
                return True
        return False


def setup_logging(settings: AppSettings | None = None, *, force: bool = False) -> None:
    global _IS_CONFIGURED
    if _IS_CONFIGURED and not force:
        return

    cfg = settings or load_config()
    log_level_name = str(cfg.log_level).strip().upper()
    level = getattr(logging, log_level_name, logging.INFO)
    max_bytes = cfg.log_max_bytes
    backup_count = cfg.log_backup_count

    log_dir = Path(str(cfg.log_dir or LOG_DIR)).expanduser().resolve()
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

    if not _has_stream_handler(root):
        root.addHandler(_console_handler(level))
    if not _has_file_handler(root, "app.log"):
        root.addHandler(_file_handler(log_dir / "app.log", level, max_bytes=max_bytes, backup_count=backup_count))

    # Category files are attached to root with name-prefix filters.
    for channel, prefixes in _CHANNEL_PREFIXES.items():
        filename = f"{channel}.log"
        if _has_file_handler(root, filename):
            continue
        handler = _file_handler(log_dir / filename, level, max_bytes=max_bytes, backup_count=backup_count)
        handler.addFilter(_LoggerPrefixFilter(prefixes))
        root.addHandler(handler)

    for channel, prefixes in _CHANNEL_PREFIXES.items():
        logging.getLogger(channel).setLevel(level)
        for prefix in prefixes:
            logging.getLogger(prefix).setLevel(level)

    logging.captureWarnings(True)
    _IS_CONFIGURED = True


def _console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    return handler


def _file_handler(path: Path, level: int, *, max_bytes: int, backup_count: int) -> logging.Handler:
    handler = RotatingFileHandler(
        path,
        mode="a",
        maxBytes=max(1024, int(max_bytes)),
        backupCount=max(1, int(backup_count)),
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    return handler


def _has_file_handler(logger: logging.Logger, filename: str) -> bool:
    needle = str(filename).lower()
    for handler in logger.handlers:
        base = getattr(handler, "baseFilename", "")
        if base and str(base).lower().endswith(needle):
            return True
    return False


def _has_stream_handler(logger: logging.Logger) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return True
    return False


# Backward compatibility alias.
configure_logging = setup_logging
