from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import LOGS_DIR
from utils.datetime_local import now_local_ts


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


@dataclass(frozen=True)
class HistogramSummary:
    count: int
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    last: float
    total: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "min": float(self.min),
            "max": float(self.max),
            "avg": float(self.avg),
            "p50": float(self.p50),
            "p95": float(self.p95),
            "last": float(self.last),
            "total": float(self.total),
        }


class MetricsRegistry:
    """In-memory app metrics (counter/gauge/histogram)."""

    def __init__(self, *, sample_limit: int = 2048):
        self.sample_limit = max(64, int(sample_limit))
        self._lock = threading.RLock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}

    def inc(self, name: str, value: float = 1.0) -> float:
        key = _metric_name(name)
        amount = float(value)
        with self._lock:
            self._counters[key] = float(self._counters.get(key, 0.0)) + amount
            return float(self._counters[key])

    def set(self, name: str, value: float) -> float:
        key = _metric_name(name)
        with self._lock:
            self._gauges[key] = float(value)
            return float(self._gauges[key])

    def observe(self, name: str, value: float) -> None:
        key = _metric_name(name)
        sample = float(value)
        with self._lock:
            bucket = list(self._histograms.get(key) or [])
            bucket.append(sample)
            if len(bucket) > self.sample_limit:
                bucket = bucket[-self.sample_limit :]
            self._histograms[key] = bucket

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = {k: float(v) for k, v in self._counters.items()}
            gauges = {k: float(v) for k, v in self._gauges.items()}
            hist = {k: _summarize(v).to_dict() for k, v in self._histograms.items()}
        return {
            "ts": now_local_ts(),
            "counters": counters,
            "gauges": gauges,
            "histograms": hist,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters = {}
            self._gauges = {}
            self._histograms = {}

    def dump_jsonl(self, path: str | Path | None = None) -> Path:
        log_path = Path(path).expanduser() if path is not None else (LOGS_DIR / "metrics.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        row = self.snapshot()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return log_path


def _metric_name(name: str) -> str:
    key = str(name or "").strip().lower().replace(" ", "_")
    return key if key else "unnamed"


def _summarize(values: list[float]) -> HistogramSummary:
    rows = sorted(float(x) for x in list(values or []))
    if not rows:
        return HistogramSummary(count=0, min=0.0, max=0.0, avg=0.0, p50=0.0, p95=0.0, last=0.0, total=0.0)

    count = len(rows)
    total = float(sum(rows))
    avg = total / max(1, count)
    p50 = rows[min(count - 1, int((count - 1) * 0.50))]
    p95 = rows[min(count - 1, int((count - 1) * 0.95))]
    return HistogramSummary(
        count=count,
        min=float(rows[0]),
        max=float(rows[-1]),
        avg=float(avg),
        p50=float(p50),
        p95=float(p95),
        last=float(values[-1]),
        total=total,
    )


_DEFAULT = MetricsRegistry()


def inc(name: str, value: float = 1.0) -> float:
    return _DEFAULT.inc(name, value)


def set_gauge(name: str, value: float) -> float:
    return _DEFAULT.set(name, value)


def observe(name: str, value: float) -> None:
    _DEFAULT.observe(name, value)


def snapshot() -> dict[str, Any]:
    return _DEFAULT.snapshot()


def reset() -> None:
    _DEFAULT.reset()


def dump_jsonl(path: str | Path | None = None) -> Path:
    return _DEFAULT.dump_jsonl(path)
