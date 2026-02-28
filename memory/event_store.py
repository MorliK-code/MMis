from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config


def _now_ts() -> float:
    import time

    return float(time.time())


@dataclass(frozen=True)
class Event:
    event_id: str
    ts: float
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    trace_id: str = ""
    model: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ts": self.ts,
            "type": self.type,
            "payload": dict(self.payload or {}),
            "tags": list(self.tags or []),
            "trace_id": self.trace_id,
            "model": self.model,
            "latency_ms": float(self.latency_ms or 0.0),
        }


class EventStore:
    """Immutable event log persisted as JSONL."""

    def __init__(self, path: str | Path | None = None):
        cfg = load_config()
        default_path = cfg.memory_dir / "events.jsonl"
        self.path = Path(path).expanduser() if path is not None else default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._events: list[dict[str, Any]] = []
        self._index: dict[str, dict[str, Any]] = {}
        self._load()

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        row = dict(event or {})
        row["event_id"] = str(row.get("event_id") or f"evt-{uuid.uuid4().hex[:16]}")
        row["ts"] = float(row.get("ts") or _now_ts())
        row["type"] = str(row.get("type") or "system")
        row["payload"] = dict(row.get("payload") or {})
        row["tags"] = [str(x) for x in list(row.get("tags") or []) if str(x).strip()]
        row["trace_id"] = str(row.get("trace_id") or "")
        row["model"] = str(row.get("model") or "")
        row["latency_ms"] = float(row.get("latency_ms") or 0.0)

        with self._lock:
            self._events.append(row)
            self._index[row["event_id"]] = row
            self._append_jsonl(row)
        return dict(row)

    def get(self, event_id: str) -> dict[str, Any] | None:
        key = str(event_id or "").strip()
        if not key:
            return None
        with self._lock:
            row = self._index.get(key)
            return dict(row) if isinstance(row, dict) else None

    def range(self, time_from: float | None = None, time_to: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            tf = float(time_from) if time_from is not None else None
            tt = float(time_to) if time_to is not None else None
            for row in self._events:
                ts = float(row.get("ts") or 0.0)
                if tf is not None and ts < tf:
                    continue
                if tt is not None and ts > tt:
                    continue
                out.append(dict(row))
            if limit is not None:
                n = max(1, int(limit))
                out = out[-n:]
            return out

    def last(self, n: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            count = max(1, int(n))
            return [dict(x) for x in self._events[-count:]]

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.last(limit)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8-sig").splitlines()
        except Exception:
            return
        with self._lock:
            for raw in lines:
                line = str(raw or "").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                event_id = str(row.get("event_id") or "")
                if not event_id:
                    continue
                item = {
                    "event_id": event_id,
                    "ts": float(row.get("ts") or 0.0),
                    "type": str(row.get("type") or "system"),
                    "payload": dict(row.get("payload") or {}),
                    "tags": [str(x) for x in list(row.get("tags") or []) if str(x).strip()],
                    "trace_id": str(row.get("trace_id") or ""),
                    "model": str(row.get("model") or ""),
                    "latency_ms": float(row.get("latency_ms") or 0.0),
                }
                self._events.append(item)
                self._index[event_id] = item

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

