from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from config.paths import LOGS_DIR


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: str | Path | None = None
    use_colors: bool = True


_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return project logger (kept for backward compatibility)."""
    return logging.getLogger(name)


def configure_logging(config: LoggingConfig | None = None) -> None:
    """Configure root logging once with sane defaults."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = config or _config_from_env()
    level_name = str(cfg.level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = _build_formatter(use_colors=bool(cfg.use_colors))
    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    log_file = cfg.log_file or (LOGS_DIR / "app.log")
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(_build_plain_formatter())
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)
    logging.captureWarnings(True)

    for channel in ("llm", "memory", "tools", "ui"):
        logging.getLogger(channel).setLevel(level)

    _CONFIGURED = True


def log_json(logger: logging.Logger, event: str, **payload) -> None:
    row = {
        "event": str(event or ""),
        **payload,
    }
    logger.info(json.dumps(row, ensure_ascii=False, sort_keys=True))


def _config_from_env() -> LoggingConfig:
    level = str(os.getenv("MMIS_LOG_LEVEL", "INFO")).strip().upper() or "INFO"
    file_raw = str(os.getenv("MMIS_LOG_FILE", "")).strip()
    use_colors_raw = str(os.getenv("MMIS_LOG_COLORS", "1")).strip().lower()
    use_colors = use_colors_raw in {"1", "true", "yes", "on"}
    log_file: str | Path | None = Path(file_raw).expanduser() if file_raw else None
    return LoggingConfig(level=level, log_file=log_file, use_colors=use_colors)


def _build_formatter(*, use_colors: bool) -> logging.Formatter:
    if not use_colors:
        return _build_plain_formatter()
    if os.name == "nt":
        return _build_plain_formatter()
    return _ColorFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def _build_plain_formatter() -> logging.Formatter:
    return logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


class _ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[35m",  # magenta
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self.COLORS.get(record.levelname)
        if not color:
            return msg
        return f"{color}{msg}{self.RESET}"
