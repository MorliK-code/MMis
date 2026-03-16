from __future__ import annotations

import gc
import json
import math
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from memory.embedding_provider import HashEmbeddingProvider
from memory.memory_models import (
    LexicalIndexBackend,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    VectorIndexBackend,
)
from memory.retrieval_projection import ensure_memory_views, record_search_text


EMBED_VERSION = "memory_v2"


def _record_search_text(record: MemoryRecord) -> str:
    return record_search_text(record.text, metadata=record.metadata)


def _with_memory_views(record: MemoryRecord) -> MemoryRecord:
    metadata = ensure_memory_views(record.text, metadata=record.metadata)
    if metadata == dict(record.metadata or {}):
        return record
    return MemoryRecord(
        id=record.id,
        text=record.text,
        memory_type=record.memory_type,
        level=record.level,
        scope=record.scope,
        namespace=record.namespace,
        metadata=metadata,
        embedding=(list(record.embedding) if isinstance(record.embedding, list) else record.embedding),
        importance=record.importance,
        confidence=record.confidence,
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=record.expires_at,
        status=record.status,
        version=record.version,
        parent_id=record.parent_id,
        chunk_index=record.chunk_index,
        source_event_id=record.source_event_id,
        embedding_model=record.embedding_model,
        embedding_fingerprint=record.embedding_fingerprint,
        embedding_version=record.embedding_version,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    if size <= 0:
        return 0.0
    dot = sum(float(a[i]) * float(b[i]) for i in range(size))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a[:size]))
    norm_b = math.sqrt(sum(float(x) * float(x) for x in b[:size]))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            return value
    return value


def _coerce_for_compare(value: Any) -> Any:
    raw = _enum_value(value)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        return raw.strip().lower()
    return raw


def _nested_lookup(payload: dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    parts = [x for x in str(dotted or "").split(".") if x]
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _matches_filters(record: MemoryRecord, filters: dict[str, Any]) -> bool:
    if not filters:
        return True
    meta = dict(record.metadata or {})
    record_fields = {
        "id",
        "namespace",
        "scope",
        "memory_type",
        "status",
        "level",
        "importance",
        "confidence",
        "created_at",
        "updated_at",
        "expires_at",
        "parent_id",
        "chunk_index",
        "version",
        "source_event_id",
        "embedding_model",
        "embedding_fingerprint",
        "embedding_version",
    }
    for key, value in dict(filters or {}).items():
        if value is None:
            continue
        if str(key).startswith("metadata."):
            current = _nested_lookup(meta, str(key)[9:])
        elif key in record_fields:
            current = getattr(record, key, None)
        else:
            current = meta.get(key)
        current_cmp = _coerce_for_compare(current)
        if isinstance(value, (list, tuple, set)):
            allowed = {_coerce_for_compare(x) for x in value}
            if current_cmp not in allowed:
                return False
            continue
        if current_cmp != _coerce_for_compare(value):
            return False
    return True


def _scope_allowed(scope: MemoryScope, scopes: list[MemoryScope]) -> bool:
    if not scopes:
        return True
    names = {str(x.value) for x in list(scopes or [])}
    return str(scope.value) in names


class LocalVectorBackend(VectorIndexBackend):
    def __init__(self, path: str | Path):
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: dict[str, MemoryRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8-sig") or "{}")
        except Exception:
            return
        rows = list(payload.get("records") or []) if isinstance(payload, dict) else []
        loaded: dict[str, MemoryRecord] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            record = MemoryRecord.from_dict(row)
            if record.id:
                loaded[record.id] = record
        with self._lock:
            self._records = loaded

    def _save(self) -> None:
        with self._lock:
            rows = [x.to_dict() for x in self._records.values()]
        self._path.write_text(json.dumps({"records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert(self, record: MemoryRecord) -> None:
        with self._lock:
            self._records[record.id] = record
        self._save()

    def batch_upsert(self, records: list[MemoryRecord]) -> None:
        with self._lock:
            for record in list(records or []):
                self._records[record.id] = record
        self._save()

    def delete(self, record_id: str) -> bool:
        key = str(record_id or "").strip()
        if not key:
            return False
        deleted = False
        with self._lock:
            if key in self._records:
                self._records.pop(key, None)
                deleted = True
        if deleted:
            self._save()
        return deleted

    def search(
        self,
        embedding: list[float],
        top_k: int,
        *,
        namespace: str,
        scopes: list[MemoryScope],
        include_stale: bool,
        metadata_filters: dict[str, Any],
    ) -> list[tuple[MemoryRecord, float]]:
        q = list(embedding or [])
        if not q:
            return []
        with self._lock:
            rows = list(self._records.values())
        scored: list[tuple[MemoryRecord, float]] = []
        for record in rows:
            if record.namespace != str(namespace):
                continue
            if record.scope == MemoryScope.PRIVATE_RUNTIME:
                continue
            if not _scope_allowed(record.scope, scopes):
                continue
            if not include_stale and record.status in {MemoryStatus.STALE, MemoryStatus.ARCHIVED, MemoryStatus.DELETED}:
                continue
            if not _matches_filters(record, metadata_filters):
                continue
            score = (_cosine(q, list(record.embedding or [])) + 1.0) / 2.0
            scored.append((record, float(score)))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return scored[: max(1, int(top_k))]

    def iter_records(self, *, namespace: str | None = None) -> list[MemoryRecord]:
        with self._lock:
            rows = list(self._records.values())
        if namespace is None:
            return rows
        ns = str(namespace)
        return [x for x in rows if x.namespace == ns]

    def reset(self) -> None:
        with self._lock:
            self._records = {}
        self._save()

    def close(self) -> None:
        self._save()


class ChromaVectorBackend(VectorIndexBackend):
    def __init__(self, *, root_dir: str | Path, collection_name: str = "memory_v2"):
        self._root = Path(root_dir).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._collection_name = str(collection_name or "memory_v2")
        try:
            import chromadb  # type: ignore
        except Exception as exc:
            raise RuntimeError("chromadb is required for Memory V2 vector index.") from exc
        try:
            self._client = chromadb.PersistentClient(path=str(self._root / "chroma_db"))
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
        except Exception as exc:
            raise RuntimeError("Failed to initialize ChromaDB persistent collection.") from exc

    @staticmethod
    def _to_chroma_metadata(record: MemoryRecord) -> dict[str, Any]:
        # Keep metadata primitive-only for Chroma compatibility.
        return {
            "__record_json": json.dumps(record.to_dict(), ensure_ascii=False),
            "namespace": str(record.namespace or ""),
            "scope": str(record.scope.value),
            "status": str(record.status.value),
        }

    @staticmethod
    def _record_from_metadata(value: Any) -> MemoryRecord | None:
        if not isinstance(value, dict):
            return None
        payload_raw = value.get("__record_json")
        if isinstance(payload_raw, str) and payload_raw.strip():
            try:
                payload = json.loads(payload_raw)
                if isinstance(payload, dict):
                    return _with_memory_views(MemoryRecord.from_dict(payload))
            except Exception:
                return None
        try:
            return _with_memory_views(MemoryRecord.from_dict(dict(value)))
        except Exception:
            return None

    def upsert(self, record: MemoryRecord) -> None:
        try:
            self._collection.upsert(
                ids=[record.id],
                documents=[record.text],
                embeddings=[list(record.embedding or [])],
                metadatas=[self._to_chroma_metadata(record)],
            )
        except Exception as exc:
            raise RuntimeError("ChromaDB upsert failed.") from exc

    def batch_upsert(self, records: list[MemoryRecord]) -> None:
        rows = [x for x in list(records or []) if isinstance(x, MemoryRecord)]
        if not rows:
            return
        try:
            self._collection.upsert(
                ids=[x.id for x in rows],
                documents=[x.text for x in rows],
                embeddings=[list(x.embedding or []) for x in rows],
                metadatas=[self._to_chroma_metadata(x) for x in rows],
            )
        except Exception as exc:
            raise RuntimeError("ChromaDB batch upsert failed.") from exc

    def delete(self, record_id: str) -> bool:
        key = str(record_id or "").strip()
        if not key:
            return False
        try:
            self._collection.delete(ids=[key])
        except Exception as exc:
            raise RuntimeError("ChromaDB delete failed.") from exc
        return True

    def search(
        self,
        embedding: list[float],
        top_k: int,
        *,
        namespace: str,
        scopes: list[MemoryScope],
        include_stale: bool,
        metadata_filters: dict[str, Any],
    ) -> list[tuple[MemoryRecord, float]]:
        try:
            out = self._collection.query(
                query_embeddings=[list(embedding or [])],
                n_results=max(1, int(top_k * (8 if metadata_filters else 4))),
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            raise RuntimeError("ChromaDB query failed.") from exc

        ids = list((out.get("ids") or [[]])[0] or [])
        dists = list((out.get("distances") or [[]])[0] or [])
        metas = list((out.get("metadatas") or [[]])[0] or [])
        scored: list[tuple[MemoryRecord, float]] = []
        for idx, rec_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            record = self._record_from_metadata(meta)
            if record is None:
                continue
            if record.id != str(rec_id):
                continue
            if record.namespace != str(namespace):
                continue
            if record.scope == MemoryScope.PRIVATE_RUNTIME:
                continue
            if not _scope_allowed(record.scope, scopes):
                continue
            if not include_stale and record.status in {MemoryStatus.STALE, MemoryStatus.ARCHIVED, MemoryStatus.DELETED}:
                continue
            if not _matches_filters(record, metadata_filters):
                continue
            distance = float(dists[idx]) if idx < len(dists) else 1.0
            score = max(0.0, min(1.0, 1.0 - distance))
            scored.append((record, score))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return scored[: max(1, int(top_k))]

    def iter_records(self, *, namespace: str | None = None) -> list[MemoryRecord]:
        try:
            out = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise RuntimeError("ChromaDB get failed.") from exc
        rows = list(out.get("metadatas") or [])
        selected: list[MemoryRecord] = []
        ns = str(namespace) if namespace is not None else ""
        for row in rows:
            record = self._record_from_metadata(row)
            if record is None:
                continue
            if ns and record.namespace != ns:
                continue
            selected.append(record)
        return selected

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
        except Exception as exc:
            raise RuntimeError("ChromaDB reset failed.") from exc

    def close(self) -> None:
        try:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            system = getattr(self._client, "_system", None)
            stop = getattr(system, "stop", None)
            if callable(stop):
                stop()
            clear_cache = getattr(self._client, "clear_system_cache", None)
            if callable(clear_cache):
                clear_cache()
        except Exception:
            pass
        self._collection = None
        self._client = None
        gc.collect()
        time.sleep(0.02)


class SQLiteFTSBackend(LexicalIndexBackend):
    def __init__(self, path: str | Path):
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._setup()

    def _setup(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lexical_records (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    status TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    parent_id TEXT,
                    chunk_index INTEGER,
                    version INTEGER NOT NULL,
                    embedding_model TEXT,
                    embedding_fingerprint TEXT,
                    embedding_version TEXT
                )
                """
            )
            columns = {
                str(row[1] or "").strip().lower()
                for row in self._conn.execute("PRAGMA table_info(lexical_records)").fetchall()
            }
            if "search_text" not in columns:
                self._conn.execute(
                    "ALTER TABLE lexical_records ADD COLUMN search_text TEXT NOT NULL DEFAULT ''"
                )
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5(id UNINDEXED, text)")
            self._conn.commit()

    def _to_record(self, row: sqlite3.Row | tuple[Any, ...]) -> MemoryRecord:
        if isinstance(row, tuple):
            data = {
                "id": row[0],
                "text": row[1],
                "search_text": row[2],
                "namespace": row[3],
                "scope": row[4],
                "status": row[5],
                "memory_type": row[6],
                "metadata_json": row[7],
                "importance": row[8],
                "confidence": row[9],
                "created_at": row[10],
                "updated_at": row[11],
                "expires_at": row[12],
                "parent_id": row[13],
                "chunk_index": row[14],
                "version": row[15],
                "embedding_model": row[16],
                "embedding_fingerprint": row[17],
                "embedding_version": row[18],
            }
        else:
            data = dict(row)
        metadata = json.loads(str(data.get("metadata_json") or "{}"))
        if str(data.get("search_text") or "").strip():
            metadata = ensure_memory_views(str(data.get("text") or ""), metadata=metadata)
            views = dict(metadata.get("memory_views") or {})
            views["search_text"] = str(data.get("search_text") or "").strip()
            metadata["memory_views"] = views
        return MemoryRecord.from_dict(
            {
                "id": str(data.get("id") or ""),
                "text": str(data.get("text") or ""),
                "namespace": str(data.get("namespace") or "default"),
                "scope": str(data.get("scope") or MemoryScope.CONVERSATION.value),
                "status": str(data.get("status") or MemoryStatus.ACTIVE.value),
                "memory_type": str(data.get("memory_type") or "message"),
                "metadata": metadata,
                "importance": float(data.get("importance") or 0.5),
                "confidence": float(data.get("confidence") or 0.5),
                "created_at": float(data.get("created_at") or 0.0),
                "updated_at": float(data.get("updated_at") or 0.0),
                "expires_at": data.get("expires_at"),
                "parent_id": data.get("parent_id"),
                "chunk_index": data.get("chunk_index"),
                "version": int(data.get("version") or 1),
                "embedding_model": str(data.get("embedding_model") or ""),
                "embedding_fingerprint": str(data.get("embedding_fingerprint") or ""),
                "embedding_version": str(data.get("embedding_version") or ""),
            }
        )

    def upsert(self, record: MemoryRecord) -> None:
        self.batch_upsert([record])

    def batch_upsert(self, records: list[MemoryRecord]) -> None:
        rows = [x for x in list(records or []) if isinstance(x, MemoryRecord)]
        if not rows:
            return
        with self._lock:
            for record in rows:
                metadata = ensure_memory_views(record.text, metadata=record.metadata)
                search_text = record_search_text(record.text, metadata=metadata)
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO lexical_records (
                        id, text, search_text, namespace, scope, status, memory_type, metadata_json,
                        importance, confidence, created_at, updated_at, expires_at,
                        parent_id, chunk_index, version, embedding_model,
                        embedding_fingerprint, embedding_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.text,
                        search_text,
                        record.namespace,
                        record.scope.value,
                        record.status.value,
                        record.memory_type.value,
                        json.dumps(metadata, ensure_ascii=False),
                        float(record.importance),
                        float(record.confidence),
                        float(record.created_at),
                        float(record.updated_at),
                        (float(record.expires_at) if record.expires_at is not None else None),
                        record.parent_id,
                        record.chunk_index,
                        int(record.version),
                        record.embedding_model,
                        record.embedding_fingerprint,
                        record.embedding_version,
                    ),
                )
                self._conn.execute("DELETE FROM lexical_fts WHERE id=?", (record.id,))
                self._conn.execute("INSERT INTO lexical_fts(id, text) VALUES (?, ?)", (record.id, search_text))
            self._conn.commit()

    def delete(self, record_id: str) -> bool:
        key = str(record_id or "").strip()
        if not key:
            return False
        with self._lock:
            cur = self._conn.execute("DELETE FROM lexical_records WHERE id=?", (key,))
            self._conn.execute("DELETE FROM lexical_fts WHERE id=?", (key,))
            self._conn.commit()
            return int(cur.rowcount or 0) > 0

    def _search_like(self, query: str, limit: int) -> list[tuple[MemoryRecord, float]]:
        tokens = [x.strip() for x in re.split(r"\s+", str(query or "").strip()) if x.strip()]
        if not tokens:
            return []
        where = " OR ".join(["search_text LIKE ?"] * len(tokens))
        params = tuple([f"%{x}%" for x in tokens] + [max(1, int(limit))])
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, text, search_text, namespace, scope, status, memory_type, metadata_json,
                       importance, confidence, created_at, updated_at, expires_at,
                       parent_id, chunk_index, version, embedding_model,
                       embedding_fingerprint, embedding_version
                FROM lexical_records
                WHERE {where}
                LIMIT ?
                """,
                params,
            ).fetchall()
        out: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            record = self._to_record(row)
            text_low = _record_search_text(record).lower()
            overlap = sum(1 for token in tokens if token.lower() in text_low)
            score = min(1.0, 0.2 + (0.8 * (float(overlap) / float(max(1, len(tokens))))))
            out.append((record, score))
        return out

    def search(
        self,
        query: str,
        top_k: int,
        *,
        namespace: str,
        scopes: list[MemoryScope],
        include_stale: bool,
        metadata_filters: dict[str, Any],
    ) -> list[tuple[MemoryRecord, float]]:
        q = str(query or "").strip()
        if not q:
            return []
        limit = max(1, int(top_k * 4))
        try:
            sql = """
                    SELECT r.id, r.text, r.search_text, r.namespace, r.scope, r.status, r.memory_type, r.metadata_json,
                           r.importance, r.confidence, r.created_at, r.updated_at, r.expires_at,
                           r.parent_id, r.chunk_index, r.version, r.embedding_model,
                           r.embedding_fingerprint, r.embedding_version,
                           bm25(lexical_fts) AS rank
                    FROM lexical_fts
                    JOIN lexical_records r ON r.id = lexical_fts.id
                    WHERE lexical_fts MATCH ?
                    LIMIT ?
                    """
            with self._lock:
                rows = self._conn.execute(sql, (q, limit)).fetchall()
            if not rows:
                tokens = [x.strip() for x in re.split(r"\s+", q) if x.strip()]
                if tokens:
                    relaxed = " OR ".join(tokens[:12])
                    with self._lock:
                        rows = self._conn.execute(sql, (relaxed, limit)).fetchall()
            scored: list[tuple[MemoryRecord, float]] = []
            for row in rows:
                record = self._to_record(row[:19])
                if record.namespace != str(namespace):
                    continue
                if record.scope == MemoryScope.PRIVATE_RUNTIME:
                    continue
                if not _scope_allowed(record.scope, scopes):
                    continue
                if not include_stale and record.status in {MemoryStatus.STALE, MemoryStatus.ARCHIVED, MemoryStatus.DELETED}:
                    continue
                if not _matches_filters(record, metadata_filters):
                    continue
                rank = float(row[19] or 0.0)
                lexical_score = 1.0 / (1.0 + max(0.0, rank))
                scored.append((record, lexical_score))
        except Exception:
            scored = self._search_like(q, limit)
        if not scored:
            scored = self._search_like(q, limit)

        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return scored[: max(1, int(top_k))]

    def iter_records(self, *, namespace: str | None = None) -> list[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, text, search_text, namespace, scope, status, memory_type, metadata_json,
                       importance, confidence, created_at, updated_at, expires_at,
                       parent_id, chunk_index, version, embedding_model,
                       embedding_fingerprint, embedding_version
                FROM lexical_records
                """
            ).fetchall()
        out = [self._to_record(row) for row in rows]
        if namespace is None:
            return out
        ns = str(namespace)
        return [x for x in out if x.namespace == ns]

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM lexical_records")
            self._conn.execute("DELETE FROM lexical_fts")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()


class VectorStore:
    """Memory V2 store facade: canonical records + vector index + lexical FTS index."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        embedding_provider,
        use_chroma: bool = True,
        collection_name: str = "memory_v2",
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider
        self.records_path = self.root_dir / "records.json"
        self.index_meta_path = self.root_dir / "index_meta.json"
        self._lock = threading.RLock()

        self._records: dict[str, MemoryRecord] = {}
        self._load_records()

        vector_root = self.root_dir / "vector"
        if not bool(use_chroma):
            raise ValueError("Memory V2 vector backend supports only ChromaDB. Set use_chroma=True.")
        self.vector_backend: VectorIndexBackend = ChromaVectorBackend(root_dir=vector_root, collection_name=collection_name)
        self.lexical_backend: LexicalIndexBackend = SQLiteFTSBackend(path=self.root_dir / "lexical_index.sqlite3")

        self.reindex_required = False
        self._load_index_meta_and_validate()

        # Ensure lexical index is populated from canonical store at startup.
        startup_rows = list(self._records.values())
        self.lexical_backend.batch_upsert(startup_rows)

    def _load_records(self) -> None:
        if not self.records_path.exists():
            return
        try:
            payload = json.loads(self.records_path.read_text(encoding="utf-8-sig") or "{}")
        except Exception:
            return
        rows = list(payload.get("records") or []) if isinstance(payload, dict) else []
        loaded: dict[str, MemoryRecord] = {}
        changed = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            record = _with_memory_views(MemoryRecord.from_dict(row))
            if record.id:
                loaded[record.id] = record
            changed = changed or dict(record.metadata or {}) != dict(row.get("metadata") or {})
        self._records = loaded
        if changed and loaded:
            self._save_records()

    def _save_records(self) -> None:
        rows = [x.to_dict() for x in self._records.values()]
        self.records_path.write_text(json.dumps({"records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index_meta_and_validate(self) -> None:
        current_model = str(getattr(self.embedding_provider, "model_name", "") or "")
        current_fingerprint = str(self.embedding_provider.model_fingerprint())
        current_version = str(getattr(self.embedding_provider, "embedding_version", EMBED_VERSION) or EMBED_VERSION)

        payload: dict[str, Any] = {}
        if self.index_meta_path.exists():
            try:
                payload = json.loads(self.index_meta_path.read_text(encoding="utf-8-sig") or "{}")
            except Exception:
                payload = {}

        old_fingerprint = str(payload.get("embedding_fingerprint") or "")
        old_model = str(payload.get("embedding_model") or "")
        if old_fingerprint and old_fingerprint != current_fingerprint:
            self.reindex_required = True
        self.reindex_required = bool(payload.get("reindex_required", False) or self.reindex_required)

        self._write_index_meta(
            {
                "embedding_model": current_model,
                "embedding_fingerprint": current_fingerprint,
                "embedding_version": current_version,
                "previous_embedding_model": old_model,
                "previous_embedding_fingerprint": old_fingerprint,
                "reindex_required": bool(self.reindex_required),
            }
        )

    def _write_index_meta(self, payload: dict[str, Any]) -> None:
        self.index_meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _to_canonical_record(record: MemoryRecord) -> MemoryRecord:
        return MemoryRecord(
            id=record.id,
            text=record.text,
            memory_type=record.memory_type,
            level=record.level,
            scope=record.scope,
            namespace=record.namespace,
            metadata=dict(record.metadata or {}),
            embedding=None,
            importance=record.importance,
            confidence=record.confidence,
            created_at=record.created_at,
            updated_at=record.updated_at,
            expires_at=record.expires_at,
            status=record.status,
            version=record.version,
            parent_id=record.parent_id,
            chunk_index=record.chunk_index,
            source_event_id=record.source_event_id,
            embedding_model=record.embedding_model,
            embedding_fingerprint=record.embedding_fingerprint,
            embedding_version=record.embedding_version,
        )

    def _prepare_record(self, record: MemoryRecord, *, force_embed: bool = False) -> MemoryRecord:
        model_name = str(getattr(self.embedding_provider, "model_name", "") or "")
        fingerprint = str(self.embedding_provider.model_fingerprint())
        version = str(getattr(self.embedding_provider, "embedding_version", EMBED_VERSION) or EMBED_VERSION)
        embedding = list(record.embedding or [])
        metadata = ensure_memory_views(record.text, metadata=record.metadata)
        search_text = record_search_text(record.text, metadata=metadata)
        if not embedding and record.scope != MemoryScope.PRIVATE_RUNTIME and (force_embed or not self.reindex_required):
            embedding = list(self.embedding_provider.embed(search_text))
        return MemoryRecord(
            id=record.id,
            text=record.text,
            memory_type=record.memory_type,
            level=record.level,
            scope=record.scope,
            namespace=record.namespace,
            metadata=metadata,
            embedding=embedding,
            importance=record.importance,
            confidence=record.confidence,
            created_at=record.created_at,
            updated_at=record.updated_at,
            expires_at=record.expires_at,
            status=record.status,
            version=record.version,
            parent_id=record.parent_id,
            chunk_index=record.chunk_index,
            source_event_id=record.source_event_id,
            embedding_model=model_name,
            embedding_fingerprint=fingerprint,
            embedding_version=version,
        )

    def upsert(self, record: MemoryRecord) -> None:
        prepared = self._prepare_record(record)
        canonical = self._to_canonical_record(prepared)
        with self._lock:
            self._records[canonical.id] = canonical
            self._save_records()
        if canonical.scope != MemoryScope.PRIVATE_RUNTIME:
            self.lexical_backend.upsert(canonical)
            if not self.reindex_required:
                self.vector_backend.upsert(prepared)

    def batch_upsert(self, records: list[MemoryRecord]) -> None:
        prepared_rows = [self._prepare_record(x) for x in list(records or []) if isinstance(x, MemoryRecord)]
        if not prepared_rows:
            return
        canonical_rows = [self._to_canonical_record(x) for x in prepared_rows]
        with self._lock:
            for row in canonical_rows:
                self._records[row.id] = row
            self._save_records()
        lexical_rows = [x for x in canonical_rows if x.scope != MemoryScope.PRIVATE_RUNTIME]
        if lexical_rows:
            self.lexical_backend.batch_upsert(lexical_rows)
            if not self.reindex_required:
                vector_rows = [x for x in prepared_rows if x.scope != MemoryScope.PRIVATE_RUNTIME]
                self.vector_backend.batch_upsert(vector_rows)

    def delete(self, record_id: str) -> bool:
        key = str(record_id or "").strip()
        if not key:
            return False
        deleted = False
        with self._lock:
            if key in self._records:
                self._records.pop(key, None)
                self._save_records()
                deleted = True
        self.lexical_backend.delete(key)
        self.vector_backend.delete(key)
        return deleted

    def semantic_search(
        self,
        *,
        query_text: str,
        top_k: int,
        namespace: str,
        scopes: list[MemoryScope],
        include_stale: bool = False,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        query = str(query_text or "").strip()
        if not query:
            return []
        if self.reindex_required:
            return []
        query_embedding = list(self.embedding_provider.embed(query))
        return self.vector_backend.search(
            embedding=query_embedding,
            top_k=max(1, int(top_k)),
            namespace=namespace,
            scopes=scopes,
            include_stale=include_stale,
            metadata_filters=dict(metadata_filters or {}),
        )

    def lexical_search(
        self,
        *,
        query_text: str,
        top_k: int,
        namespace: str,
        scopes: list[MemoryScope],
        include_stale: bool = False,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        query = str(query_text or "").strip()
        if not query:
            return []
        return self.lexical_backend.search(
            query=query,
            top_k=max(1, int(top_k)),
            namespace=namespace,
            scopes=scopes,
            include_stale=include_stale,
            metadata_filters=dict(metadata_filters or {}),
        )

    def search(
        self,
        *,
        query_text: str,
        top_k: int,
        namespace: str,
        scopes: list[MemoryScope],
        include_stale: bool = False,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = str(query_text or "").strip()
        if not query:
            return []
        filt = dict(metadata_filters or {})

        lexical_hits = self.lexical_search(
            query_text=query,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=include_stale,
            metadata_filters=filt,
        )

        vector_hits = self.semantic_search(
            query_text=query,
            top_k=max(1, int(top_k * 3)),
            namespace=namespace,
            scopes=scopes,
            include_stale=include_stale,
            metadata_filters=filt,
        )

        merged: dict[str, dict[str, Any]] = {}
        for record, score in lexical_hits:
            merged[record.id] = {
                "record": record,
                "semantic": 0.0,
                "lexical": float(score),
            }
        for record, score in vector_hits:
            row = merged.setdefault(
                record.id,
                {
                    "record": record,
                    "semantic": 0.0,
                    "lexical": 0.0,
                },
            )
            row["semantic"] = max(float(row.get("semantic") or 0.0), float(score))

        out: list[dict[str, Any]] = []
        for row in merged.values():
            record = row["record"]
            out.append(
                {
                    "record": record,
                    "semantic_score": float(row.get("semantic") or 0.0),
                    "lexical_score": float(row.get("lexical") or 0.0),
                }
            )
        out.sort(key=lambda x: max(float(x["semantic_score"]), float(x["lexical_score"])), reverse=True)
        return out[: max(1, int(top_k * 3))]

    def children_of(
        self,
        *,
        parent_id: str,
        namespace: str | None = None,
        include_stale: bool = False,
    ) -> list[MemoryRecord]:
        key = str(parent_id or "").strip()
        if not key:
            return []
        rows = self.iter_records(namespace=namespace)
        out: list[MemoryRecord] = []
        for row in rows:
            if str(row.parent_id or "") != key:
                continue
            if not include_stale and row.status in {MemoryStatus.STALE, MemoryStatus.ARCHIVED, MemoryStatus.DELETED}:
                continue
            out.append(row)
        out.sort(key=lambda x: (float(x.updated_at), int(x.version)), reverse=True)
        return out

    def iter_records(self, *, namespace: str | None = None) -> list[MemoryRecord]:
        with self._lock:
            rows = list(self._records.values())
        if namespace is None:
            return rows
        ns = str(namespace)
        return [x for x in rows if x.namespace == ns]

    def mark_reindex_required(self) -> None:
        self.reindex_required = True
        self._load_index_meta_and_validate()

    def rebuild_indexes(self, *, namespace: str | None = None, incremental: bool = False) -> dict[str, Any]:
        records = self.iter_records(namespace=namespace)
        if not incremental:
            self.lexical_backend.reset()
            self.vector_backend.reset()

        prepared = [self._prepare_record(x, force_embed=True) for x in records if x.scope != MemoryScope.PRIVATE_RUNTIME]
        canonical = [self._to_canonical_record(x) for x in prepared]
        self.lexical_backend.batch_upsert(canonical)
        self.vector_backend.batch_upsert(prepared)
        self.reindex_required = False
        self._write_index_meta(
            {
                "embedding_model": str(getattr(self.embedding_provider, "model_name", "") or ""),
                "embedding_fingerprint": str(self.embedding_provider.model_fingerprint()),
                "embedding_version": str(getattr(self.embedding_provider, "embedding_version", EMBED_VERSION) or EMBED_VERSION),
                "reindex_required": False,
            }
        )
        return {
            "records_indexed": len(prepared),
            "namespace": namespace or "*",
            "incremental": bool(incremental),
            "reindex_required": False,
        }

    def close(self) -> None:
        self.lexical_backend.close()
        self.vector_backend.close()
        if hasattr(self.embedding_provider, "close"):
            try:
                self.embedding_provider.close()
            except Exception:
                pass


# Backward-compatible utility used by legacy scripts.
def embed_text(text: str, *, dim: int = 128) -> list[float]:
    return HashEmbeddingProvider(dim=max(32, int(dim))).embed(str(text or ""))
