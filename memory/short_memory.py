from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config
from utils.datetime_local import now_local_ts, to_local_iso


class ShortMemory:
    """Short-lived rolling window of recent messages/events."""

    def __init__(
        self,
        limit: int = 60,
        *,
        summary_trigger: int = 50,
        path: str | Path | None = None,
        autosave: bool = True,
    ):
        cfg = load_config()
        default_path = cfg.memory_dir / "short_memory.json"
        self.path = Path(path).expanduser() if path is not None else default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.limit = max(10, int(limit))
        self.summary_trigger = max(10, int(summary_trigger))
        self.autosave = bool(autosave)
        self._lock = RLock()

        self._items: list[dict[str, Any]] = []
        self._rolling_summary: str = ""
        self.load()

    def append(self, item: dict[str, Any]) -> None:
        row = self._normalize_item(item)
        if not row.get("text"):
            return
        with self._lock:
            self._items.append(row)
            if len(self._items) > self.limit:
                overflow = self._items[:-self.limit]
                self._items = self._items[-self.limit :]
                self._update_summary(overflow)
            elif len(self._items) > self.summary_trigger:
                # Keep summary fresh even before overflow.
                self._update_summary(self._items[:-self.summary_trigger])
        self._autosave()

    def tail(self, n: int = 12) -> list[dict[str, Any]]:
        count = max(1, int(n))
        with self._lock:
            return [dict(x) for x in self._items[-count:]]

    def clear(self) -> None:
        with self._lock:
            self._items = []
            self._rolling_summary = ""
        self._autosave()

    def rolling_summary(self) -> str:
        with self._lock:
            return str(self._rolling_summary or "")

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        items = payload.get("items")
        summary = payload.get("rolling_summary")
        with self._lock:
            self._items = [self._normalize_item(x) for x in list(items or []) if isinstance(x, dict)]
            self._items = [x for x in self._items if x.get("text")]
            if len(self._items) > self.limit:
                self._items = self._items[-self.limit :]
            self._rolling_summary = str(summary or "")

    def save(self) -> None:
        with self._lock:
            payload = {
                "items": [dict(x) for x in self._items],
                "rolling_summary": str(self._rolling_summary or ""),
            }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _autosave(self) -> None:
        if self.autosave:
            self.save()

    def _update_summary(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        parts = []
        for row in items[-10:]:
            role = str(row.get("role") or "")
            text = str(row.get("text") or "")
            if not text:
                continue
            short = text if len(text) <= 80 else text[:77].rstrip() + "..."
            parts.append(f"{role}: {short}")
        if not parts:
            return
        block = " | ".join(parts)
        if self._rolling_summary:
            merged = f"{self._rolling_summary} || {block}"
        else:
            merged = block
        # Keep summary bounded.
        self._rolling_summary = merged[-1600:]

    @staticmethod
    def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        row = dict(item or {})
        text = str(row.get("text") or row.get("content") or "").strip()
        ts_value = to_local_iso(row.get("ts"), default="")
        if not ts_value:
            ts_value = now_local_ts()
        return {
            "id": str(row.get("id") or row.get("event_id") or ""),
            "role": str(row.get("role") or "user"),
            "type": str(row.get("type") or "message"),
            "text": text,
            "ts": ts_value,
            "lang": str(row.get("lang") or ""),
            "intent": str(row.get("intent") or ""),
            "emotion": str(row.get("emotion") or ""),
            "tags": [str(x) for x in list(row.get("tags") or []) if str(x).strip()],
            "has_code": bool(row.get("has_code", False)),
            "meta": dict(row.get("meta") or {}),
        }
