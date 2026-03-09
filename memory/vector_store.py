from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config
from utils.logger import get_logger, log_json


_TOKEN_RE = re.compile(r"[A-Za-z\u0400-\u04ff0-9_]+")
EMBED_VERSION = "blake2b64_v1"
LOGGER = get_logger(__name__)


class VectorStore:
    """Simple vector store abstraction (replaceable with Chroma/FAISS/Qdrant)."""

    def __init__(self, path: str | Path | None = None, *, dim: int = 128):
        cfg = load_config()
        default_path = cfg.memory_dir / "vector_store.json"
        self.path = Path(path).expanduser() if path is not None else default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.dim = max(32, int(dim))
        self.embed_version = EMBED_VERSION
        self._lock = RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self.load()

    def upsert(
        self,
        id: str,
        text: str = "",
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        key = str(id or "").strip()
        if not key:
            return
        vector = list(embedding or [])
        if not vector:
            vector = embed_text(str(text or ""), dim=self.dim)
        vector = _normalize_vector(vector, dim=self.dim)
        row = {
            "id": key,
            "text": str(text or ""),
            "embedding": vector,
            "metadata": dict(metadata or {}),
        }
        row["metadata"].setdefault("embed_version", self.embed_version)
        with self._lock:
            self._records[key] = row
            self.save()
        log_json(
            LOGGER,
            "vector_upsert",
            id=key,
            embed_version=self.embed_version,
            dim=self.dim,
            text_chars=len(str(text or "")),
        )

    def query(self, query_embedding: list[float], k: int = 5, filters: dict[str, Any] | None = None) -> list[tuple[str, float, dict[str, Any]]]:
        q = _normalize_vector(list(query_embedding or []), dim=self.dim)
        if not q:
            return []
        count = max(1, int(k))
        filt = dict(filters or {})
        with self._lock:
            rows = list(self._records.values())
        scored: list[tuple[str, float, dict[str, Any]]] = []
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            if not _matches_filters(metadata, filt):
                continue
            vec = _normalize_vector(list(row.get("embedding") or []), dim=self.dim)
            if not vec:
                continue
            cosine = _cosine(q, vec)
            score = (cosine + 1.0) / 2.0  # normalize to 0..1 where higher is better
            payload = dict(metadata)
            payload.setdefault("text", str(row.get("text") or ""))
            payload.setdefault("distance", 1.0 - score)
            payload.setdefault("embed_version", self.embed_version)
            scored.append((str(row.get("id") or ""), float(score), payload))
        scored.sort(key=lambda x: x[1], reverse=True)
        out = scored[:count]
        log_json(
            LOGGER,
            "vector_query",
            k=count,
            returned=len(out),
            filters=sorted([str(x) for x in filt.keys()]),
            embed_version=self.embed_version,
            dim=self.dim,
        )
        return out

    def query_text(self, query_text: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[tuple[str, float, dict[str, Any]]]:
        vector = embed_text(str(query_text or ""), dim=self.dim)
        return self.query(vector, k=k, filters=filters)

    def delete(self, id: str) -> bool:
        key = str(id or "").strip()
        if not key:
            return False
        with self._lock:
            if key not in self._records:
                return False
            self._records.pop(key, None)
            self.save()
            return True

    def purge(self) -> None:
        with self._lock:
            self._records = {}
            self.save()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except Exception:
            return
        if isinstance(payload, dict):
            loaded_embed_version = str(payload.get("embed_version") or "").strip()
            if loaded_embed_version:
                self.embed_version = loaded_embed_version
        rows = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return
        with self._lock:
            self._records = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("id") or "").strip()
                if not key:
                    continue
                record = {
                    "id": key,
                    "text": str(row.get("text") or ""),
                    "embedding": _normalize_vector(list(row.get("embedding") or []), dim=self.dim),
                    "metadata": dict(row.get("metadata") or {}),
                }
                record["metadata"].setdefault("embed_version", self.embed_version)
                self._records[key] = record

    def save(self) -> None:
        with self._lock:
            rows = list(self._records.values())
        payload = {"embed_version": self.embed_version, "records": rows}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def embed_text(text: str, *, dim: int = 128) -> list[float]:
    n = max(32, int(dim))
    vec = [0.0] * n
    tokens = _TOKEN_RE.findall(str(text or "").lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="little", signed=False) % n
        vec[idx] += 1.0
    return _normalize_vector(vec, dim=n)


def _normalize_vector(value: list[float], *, dim: int) -> list[float]:
    if not value:
        return [0.0] * int(dim)
    out = list(value[: int(dim)])
    if len(out) < int(dim):
        out.extend([0.0] * (int(dim) - len(out)))
    norm = math.sqrt(sum(x * x for x in out))
    if norm <= 1e-12:
        return [0.0] * int(dim)
    return [float(x / norm) for x in out]


def _cosine(a: list[float], b: list[float]) -> float:
    size = min(len(a), len(b))
    if size == 0:
        return 0.0
    return float(sum(a[i] * b[i] for i in range(size)))


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        if value is None:
            continue
        if key not in metadata:
            return False
        current = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            if current not in set(value):
                return False
            continue
        if current != value:
            return False
    return True

