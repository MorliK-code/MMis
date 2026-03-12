from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from config.settings import LOGS_DIR, load_config, setup_logging


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: str | Path | None = None
    use_colors: bool = True


_CONFIGURED = False
_LOG_CONTEXT_KEYS = ("trace_id", "request_id", "turn_id", "conversation_id")
_HUMAN_LOG_LOCK = RLock()
_HUMAN_LOG_FILE = "human.log"


def get_logger(name: str) -> logging.Logger:
    """Return project logger (kept for backward compatibility)."""
    raw = str(name or "").strip()
    return logging.getLogger(_map_logger_name(raw))


def configure_logging(config: LoggingConfig | None = None) -> None:
    """Configure root logging once with sane defaults."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if config is None:
        setup_logging()
        _CONFIGURED = True
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


def extract_log_context(
    *candidates: Mapping[str, Any] | None,
    trace_id: Any = None,
    request_id: Any = None,
    turn_id: Any = None,
    conversation_id: Any = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        trace_value = candidate.get("trace_id")
        if trace_value is None:
            trace_value = candidate.get("trace")
        _set_context_value(out, "trace_id", trace_value)
        _set_context_value(out, "request_id", candidate.get("request_id"))
        _set_context_value(out, "turn_id", candidate.get("turn_id"))
        _set_context_value(out, "conversation_id", candidate.get("conversation_id"))
    _set_context_value(out, "trace_id", trace_id)
    _set_context_value(out, "request_id", request_id)
    _set_context_value(out, "turn_id", turn_id)
    _set_context_value(out, "conversation_id", conversation_id)
    return out


def merge_log_context(
    payload: Mapping[str, Any] | None = None,
    *,
    context: Mapping[str, Any] | None = None,
    trace_id: Any = None,
    request_id: Any = None,
    turn_id: Any = None,
    conversation_id: Any = None,
) -> dict[str, Any]:
    row = dict(payload or {})
    context_map = extract_log_context(
        row,
        context,
        trace_id=trace_id,
        request_id=request_id,
        turn_id=turn_id,
        conversation_id=conversation_id,
    )
    for key in _LOG_CONTEXT_KEYS:
        if key in row and _has_textual_value(row.get(key)):
            continue
        if key in context_map:
            row[key] = context_map[key]
    return row


def log_json(
    logger: logging.Logger,
    event: str,
    *,
    summary: str = "",
    context: Mapping[str, Any] | None = None,
    **payload,
) -> None:
    row = {
        "event": str(event or ""),
    }
    summary_text = str(summary or payload.pop("summary", "") or "").strip()
    if summary_text:
        row["summary"] = summary_text

    merged_payload = merge_log_context(payload, context=context)
    for key in _LOG_CONTEXT_KEYS:
        if key in merged_payload:
            row[key] = merged_payload.pop(key)
    row.update(merged_payload)
    logger.info(json.dumps(row, ensure_ascii=False))


def append_human_log(
    section: str,
    *,
    lines: list[str] | tuple[str, ...],
    context: Mapping[str, Any] | None = None,
    file_name: str = _HUMAN_LOG_FILE,
) -> str:
    path = human_log_path(file_name=file_name)
    context_map = extract_log_context(context)
    header = (
        f"[{_utc_now_iso()}] {str(section or 'ENTRY').strip().upper()} "
        f"{_human_context_suffix(context_map)}"
    ).rstrip()
    body = [str(line).rstrip() for line in list(lines or []) if str(line).strip()]
    block = "\n".join([header, *body, ""])
    with _HUMAN_LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(block + "\n")
    return str(path)


def human_log_path(*, file_name: str = _HUMAN_LOG_FILE) -> Path:
    active_dir = _active_log_dir()
    active_dir.mkdir(parents=True, exist_ok=True)
    return active_dir / str(file_name or _HUMAN_LOG_FILE)


def _config_from_env() -> LoggingConfig:
    cfg = load_config()
    level = str(cfg.log_level).strip().upper() or "INFO"
    use_colors = bool(cfg.log_colors)
    log_file: str | Path | None = Path(cfg.log_file).expanduser() if cfg.log_file else None
    return LoggingConfig(level=level, log_file=log_file, use_colors=use_colors)


def _active_log_dir() -> Path:
    root = logging.getLogger()
    for handler in list(root.handlers):
        base = str(getattr(handler, "baseFilename", "") or "").strip()
        if not base:
            continue
        try:
            path = Path(base).expanduser().resolve()
        except Exception:
            continue
        if path.name.lower() == "app.log":
            return path.parent
    cfg = load_config()
    return Path(str(cfg.log_dir or LOGS_DIR)).expanduser().resolve()


def _set_context_value(target: dict[str, Any], key: str, value: Any) -> None:
    if key in target and _has_textual_value(target.get(key)):
        return
    if not _has_textual_value(value):
        return
    target[key] = str(value).strip()


def _human_context_suffix(context: Mapping[str, Any] | None) -> str:
    row = extract_log_context(context)
    parts: list[str] = []
    for key, label in (
        ("trace_id", "trace"),
        ("request_id", "request"),
        ("turn_id", "turn"),
        ("conversation_id", "conversation"),
    ):
        value = str(row.get(key) or "").strip()
        if value:
            parts.append(f"{label}={value}")
    return " ".join(parts)


def _has_textual_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    return bool(str(value).strip())


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _map_logger_name(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "app"

    low = raw.lower()
    for prefix in ("llm", "memory", "tools", "ui"):
        if low == prefix or low.startswith(prefix + "."):
            return raw

    if low.startswith("modules."):
        return f"tools.{raw}"
    if low.startswith("metadata.") or low.startswith("memory."):
        return f"memory.{raw}"
    if low.startswith("api.") or low == "api" or low.startswith("ui.") or low in {"ui_console", "ui_pyside6"}:
        return f"ui.{raw}"
    return raw


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
