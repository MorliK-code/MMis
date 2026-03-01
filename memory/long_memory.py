from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config
from utils.datetime_local import now_local_iso, parse_time_to_epoch


@dataclass(frozen=True)
class MemoryDoc:
    id: str
    text: str
    created_at: float
    updated_at: str
    source: str = "chat"
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "created_at": float(self.created_at),
            "updated_at": str(self.updated_at),
            "source": self.source,
            "tags": list(self.tags or []),
            "importance": float(self.importance),
            "confidence": float(self.confidence),
            "meta": dict(self.meta or {}),
        }


class LongMemory:
    """Persistent textual long-term memory (docs only, no vectors)."""

    def __init__(self, path: str | Path | None = None):
        cfg = load_config()
        default_path = cfg.memory_dir / "long_memory_docs.json"
        self.path = Path(path).expanduser() if path is not None else default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = RLock()
        self._docs: dict[str, dict[str, Any]] = {}
        self.load()

    def add_doc(
        self,
        text: str,
        meta: dict[str, Any] | None = None,
        *,
        source: str = "chat",
        tags: list[str] | None = None,
        importance: float = 0.5,
        confidence: float = 0.5,
        doc_id: str | None = None,
    ) -> MemoryDoc:
        doc_text = str(text or "").strip()
        if not doc_text:
            raise ValueError("LongMemory.add_doc requires non-empty text")

        now = time.time()
        item = MemoryDoc(
            id=str(doc_id or f"doc-{uuid.uuid4().hex[:16]}"),
            text=doc_text,
            created_at=now,
            updated_at=now_local_iso(),
            source=str(source or "chat"),
            tags=[str(x) for x in list(tags or []) if str(x).strip()],
            importance=_clamp01(importance),
            confidence=_clamp01(confidence),
            meta=dict(meta or {}),
        )
        with self._lock:
            self._docs[item.id] = item.to_dict()
            self.save()
        return item

    def update_doc(self, doc_id: str, **changes) -> MemoryDoc | None:
        key = str(doc_id or "").strip()
        if not key:
            return None
        with self._lock:
            current = self._docs.get(key)
            if not isinstance(current, dict):
                return None
            row = dict(current)
            if "text" in changes and str(changes.get("text") or "").strip():
                row["text"] = str(changes.get("text")).strip()
            if "source" in changes and str(changes.get("source") or "").strip():
                row["source"] = str(changes.get("source")).strip()
            if "tags" in changes:
                row["tags"] = [str(x) for x in list(changes.get("tags") or []) if str(x).strip()]
            if "importance" in changes and changes.get("importance") is not None:
                row["importance"] = _clamp01(float(changes.get("importance")))
            if "confidence" in changes and changes.get("confidence") is not None:
                row["confidence"] = _clamp01(float(changes.get("confidence")))
            if "meta" in changes and isinstance(changes.get("meta"), dict):
                merged = dict(row.get("meta") or {})
                merged.update(dict(changes.get("meta") or {}))
                row["meta"] = merged
            row["updated_at"] = now_local_iso()
            self._docs[key] = row
            self.save()
            return _doc_from_dict(row)

    def delete_doc(self, doc_id: str) -> bool:
        key = str(doc_id or "").strip()
        if not key:
            return False
        with self._lock:
            if key not in self._docs:
                return False
            self._docs.pop(key, None)
            self.save()
            return True

    def get_doc(self, doc_id: str) -> MemoryDoc | None:
        key = str(doc_id or "").strip()
        if not key:
            return None
        with self._lock:
            row = self._docs.get(key)
            return _doc_from_dict(row) if isinstance(row, dict) else None

    def list_docs(
        self,
        *,
        limit: int | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> list[MemoryDoc]:
        with self._lock:
            rows = list(self._docs.values())
        rows.sort(key=lambda x: parse_time_to_epoch(x.get("updated_at"), 0.0), reverse=True)
        tag_filter = {str(x).strip().lower() for x in list(tags or []) if str(x).strip()}
        source_filter = str(source or "").strip().lower()
        out: list[MemoryDoc] = []
        for row in rows:
            if source_filter and str(row.get("source") or "").strip().lower() != source_filter:
                continue
            if tag_filter:
                row_tags = {str(x).strip().lower() for x in list(row.get("tags") or [])}
                if not tag_filter.issubset(row_tags):
                    continue
            doc = _doc_from_dict(row)
            if doc is not None:
                out.append(doc)
            if limit is not None and len(out) >= max(1, int(limit)):
                break
        return out

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except Exception:
            return
        rows = payload.get("docs") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return
        with self._lock:
            self._docs = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("id") or "").strip()
                if not key:
                    continue
                self._docs[key] = {
                    "id": key,
                    "text": str(row.get("text") or ""),
                    "created_at": float(row.get("created_at") or time.time()),
                    "updated_at": str(row.get("updated_at") or now_local_iso()),
                    "source": str(row.get("source") or "chat"),
                    "tags": [str(x) for x in list(row.get("tags") or []) if str(x).strip()],
                    "importance": _clamp01(float(row.get("importance") or 0.5)),
                    "confidence": _clamp01(float(row.get("confidence") or 0.5)),
                    "meta": dict(row.get("meta") or {}),
                }

    def save(self) -> None:
        with self._lock:
            rows = list(self._docs.values())
        payload = {"docs": rows}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _doc_from_dict(row: dict[str, Any] | None) -> MemoryDoc | None:
    if not isinstance(row, dict):
        return None
    text = str(row.get("text") or "").strip()
    if not text:
        return None
    return MemoryDoc(
        id=str(row.get("id") or ""),
        text=text,
        created_at=float(row.get("created_at") or time.time()),
        updated_at=str(row.get("updated_at") or now_local_iso()),
        source=str(row.get("source") or "chat"),
        tags=[str(x) for x in list(row.get("tags") or []) if str(x).strip()],
        importance=_clamp01(float(row.get("importance") or 0.5)),
        confidence=_clamp01(float(row.get("confidence") or 0.5)),
        meta=dict(row.get("meta") or {}),
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
