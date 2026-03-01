from __future__ import annotations

from datetime import datetime
from typing import Any


def now_local_iso(*, with_ms: bool = True) -> str:
    timespec = "milliseconds" if with_ms else "seconds"
    return datetime.now().astimezone().isoformat(timespec=timespec)


def now_local_ts(*, with_ms: bool = True) -> str:
    """Canonical human-readable timestamp for `ts` fields."""
    return now_local_iso(with_ms=with_ms)


def to_local_iso(value: Any, default: str = "") -> str:
    """Convert unix epoch / ISO / numeric string to local ISO string."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return str(default or "")
        try:
            # Keep already-valid ISO strings as-is.
            datetime.fromisoformat(text)
            return text
        except Exception:
            pass

    ts = parse_time_to_epoch(value, float("nan"))
    if ts != ts:  # NaN check
        return str(default or "")
    if float(ts) <= 0.0:
        return str(default or "")
    try:
        return datetime.fromtimestamp(float(ts)).astimezone().isoformat(timespec="milliseconds")
    except Exception:
        return str(default or "")


def parse_time_to_epoch(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return float(default)

    text = str(value).strip()
    if not text:
        return float(default)

    try:
        return float(text)
    except Exception:
        pass

    try:
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return float(default)
