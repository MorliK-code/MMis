from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config
from llm.tokenizer import create_tokenizer
from memory.document_memory import ChunkingConfig, DocumentMemory
from memory.embedding_provider import build_embedding_provider
from memory.event_store import EventStore
from memory.fact_extractor import FactExtractor
from memory.memory_debug import BasicMemoryDebugger
from memory.memory_lifecycle import MemoryLifecycleManager
from memory.memory_models import (
    ContextBuildRequest,
    ContextBuildResult,
    DebugRequest,
    DocumentIngestRequest,
    DocumentIngestResult,
    FactRecordV2,
    IngestResult,
    MemoryEvent,
    MemoryLevel,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResult,
)
from memory.retrieval import HybridRetriever
from memory.reranker import HeuristicReranker
from memory.vector_store import VectorStore
from memory.context_builder import ContextBuilderV2
from utils.logger import get_logger, log_json


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class MemoryDebugSnapshot:
    payload: dict[str, Any]


class MemoryManager:
    """Memory V2 manager.

    Pure API (V2):
      - ingest_event
      - retrieve
      - build_context
      - ingest_document
      - debug_snapshot
      - reindex_embeddings
    """

    def __init__(self, *, root_dir: str | Path | None = None):
        self._cfg = load_config()
        base = Path(root_dir).expanduser().resolve() if root_dir is not None else Path(self._cfg.memory_dir).resolve()
        self._root = base / "memory_v2"
        self._root.mkdir(parents=True, exist_ok=True)
        self._state_path = self._root / "manager_state.json"

        self._lock = RLock()

        embedding_backend = str(getattr(self._cfg, "memory_embedding_backend", "sentence_transformers") or "sentence_transformers")
        embedding_model = str(getattr(self._cfg, "memory_embedding_model", "all-MiniLM-L6-v2") or "all-MiniLM-L6-v2")
        self._embedding_provider = build_embedding_provider(
            backend=embedding_backend,
            model_name=embedding_model,
            cache_path=self._root / "embedding_cache.sqlite3",
            dim=int(getattr(self._cfg, "memory_embedding_dim", 384) or 384),
        )

        memory_backend = str(getattr(self._cfg, "memory_backend", "chroma") or "chroma").strip().lower()
        self._store = VectorStore(
            root_dir=self._root,
            embedding_provider=self._embedding_provider,
            use_chroma=(memory_backend in {"chroma", "chromadb", "chroma_hybrid"}),
            collection_name="mmis_memory_v2",
        )

        self._event_store = EventStore(path=self._root / "events_v2.jsonl")
        self._fact_extractor = FactExtractor()
        self._lifecycle = MemoryLifecycleManager(
            stale_after_days=int(getattr(self._cfg, "memory_stale_after_days", 30) or 30),
            archive_after_days=int(getattr(self._cfg, "memory_archive_after_days", 90) or 90),
        )
        self._retriever = HybridRetriever(
            store=self._store,
            stale_after_days=int(getattr(self._cfg, "memory_stale_after_days", 30) or 30),
        )
        self._reranker = HeuristicReranker()
        self._context_builder = ContextBuilderV2(tokenizer=create_tokenizer())
        self._document_memory = DocumentMemory(
            store=self._store,
            chunking=ChunkingConfig(
                chunk_size=int(getattr(self._cfg, "memory_chunk_size", 1200) or 1200),
                chunk_overlap=int(getattr(self._cfg, "memory_chunk_overlap", 160) or 160),
            ),
        )
        self._debugger = BasicMemoryDebugger(store=self._store)

        self._working_records: list[MemoryRecord] = []
        self._session_summary: str = ""
        self._open_questions: list[str] = []
        self._current_decisions: list[str] = []
        self._active_preferences: list[str] = []
        self._private_runtime: dict[str, dict[str, Any]] = {}

        self._temporary_ttl_sec = int(getattr(self._cfg, "memory_temporary_ttl_sec", 3600) or 3600)
        self._private_runtime_ttl_sec = int(getattr(self._cfg, "memory_private_runtime_ttl_sec", 900) or 900)
        self._working_limit = int(getattr(self._cfg, "memory_working_limit", 120) or 120)
        self._retrieval_top_k = int(getattr(self._cfg, "memory_retrieval_top_k", 8) or 8)
        self._rerank_top_k = int(getattr(self._cfg, "memory_rerank_top_k", 8) or 8)

        self._load_state()
        self._cleanup_expired()

    def ingest_event(self, event: MemoryEvent) -> IngestResult:
        with self._lock:
            self._cleanup_expired()

            text = str(event.text or "").strip()
            if not text and event.scope != MemoryScope.PRIVATE_RUNTIME:
                return IngestResult(stored_ids=[])

            now_ts = float(event.ts or time.time())
            event_id = f"evt:{uuid.uuid4().hex[:16]}"
            metadata = dict(event.metadata or {})
            namespace = str(event.namespace or "default")
            scope = event.scope
            memory_type = event.memory_type

            level = self._initial_level(memory_type)
            importance = self._importance_score(text=text, metadata=metadata)
            confidence = self._confidence_score(metadata=metadata)
            expires_at = self._expires_at(scope=scope, metadata=metadata, now_ts=now_ts)

            record = MemoryRecord(
                id=f"{memory_type.value}:{event_id}",
                text=text,
                memory_type=memory_type,
                level=level,
                scope=scope,
                namespace=namespace,
                metadata=metadata,
                importance=importance,
                confidence=confidence,
                created_at=now_ts,
                updated_at=now_ts,
                expires_at=expires_at,
                status=MemoryStatus.ACTIVE,
                source_event_id=event_id,
            )

            stored_ids: list[str] = []
            promoted_ids: list[str] = []
            extracted_facts: list[FactRecordV2] = []

            if scope == MemoryScope.PRIVATE_RUNTIME:
                key = str(metadata.get("runtime_key") or record.id)
                self._private_runtime[key] = {
                    "value": metadata.get("runtime_value", text),
                    "namespace": namespace,
                    "expires_at": float(expires_at or now_ts + self._private_runtime_ttl_sec),
                    "updated_at": now_ts,
                }
                stored_ids.append(record.id)
            else:
                self._store.upsert(record)
                stored_ids.append(record.id)
                self._update_session_state_from_event(record)

            self._upsert_working_record(record)

            lifecycle_decision = self._lifecycle.decide(record, now_ts=now_ts)
            if lifecycle_decision.promote_to is not None and lifecycle_decision.promote_to != record.level:
                promoted = self._promote_record(record, target=lifecycle_decision.promote_to, now_ts=now_ts)
                self._store.upsert(promoted)
                promoted_ids.append(promoted.id)

            if memory_type in {MemoryType.MESSAGE, MemoryType.SUMMARY} and scope != MemoryScope.PRIVATE_RUNTIME:
                facts = self._fact_extractor.extract_v2(
                    text=text,
                    metadata={"event_id": event_id, **metadata},
                    speaker=str(event.role or "user"),
                    scope=scope,
                    mode=str(metadata.get("quality_profile") or "BALANCED"),
                )
                extracted_facts = self._write_fact_records(
                    facts=facts,
                    namespace=namespace,
                    now_ts=now_ts,
                    event_id=event_id,
                )

            self._event_store.append(
                {
                    "event_id": event_id,
                    "ts": now_ts,
                    "type": "memory_ingest_v2",
                    "payload": {
                        "role": str(event.role or ""),
                        "scope": scope.value,
                        "memory_type": memory_type.value,
                        "record_id": record.id,
                        "namespace": namespace,
                    },
                    "tags": [scope.value, memory_type.value],
                }
            )
            self._save_state()

        log_json(
            LOGGER,
            "memory_v2_ingest",
            namespace=namespace,
            scope=scope.value,
            memory_type=memory_type.value,
            stored=len(stored_ids),
            promoted=len(promoted_ids),
            facts=len(extracted_facts),
        )
        return IngestResult(
            stored_ids=stored_ids,
            promoted_ids=promoted_ids,
            extracted_facts=list(extracted_facts),
        )

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        with self._lock:
            self._cleanup_expired()
            scopes = list(query.scopes or self._default_retrieval_scopes())
            normalized_query = RetrievalQuery(
                query_text=str(query.query_text or ""),
                namespace=str(query.namespace or "default"),
                scopes=scopes,
                top_k=max(1, int(query.top_k or self._retrieval_top_k)),
                include_private_runtime=bool(query.include_private_runtime),
                include_stale=bool(query.include_stale),
                metadata_filters=dict(query.metadata_filters or {}),
            )

            retrieved = self._retriever.retrieve(normalized_query)
            reranked = self._reranker.rerank(
                query=normalized_query,
                candidates=list(retrieved.candidates or []),
                top_k=max(1, int(self._rerank_top_k)),
            )

            return RetrievalResult(
                query=normalized_query,
                candidates=reranked,
                reindex_required=bool(retrieved.reindex_required),
            )

    def build_context(self, request: ContextBuildRequest) -> ContextBuildResult:
        budget_total = int(request.context_budget_total or getattr(self._cfg, "memory_context_budget_total", 2200) or 2200)
        budget_memory = int(request.context_budget_memory or getattr(self._cfg, "memory_context_budget_memory", 700) or 700)
        budget_docs = int(request.context_budget_docs or getattr(self._cfg, "memory_context_budget_docs", 600) or 600)
        budget_tools = int(request.context_budget_tools or getattr(self._cfg, "memory_context_budget_tools", 220) or 220)
        budget_reserve = int(
            request.context_budget_response_reserve
            or getattr(self._cfg, "memory_context_budget_response_reserve", 260)
            or 260
        )

        retrieval_query = RetrievalQuery(
            query_text=str(request.user_message or "").strip(),
            namespace=str(request.namespace or "default"),
            scopes=list(request.scopes or self._default_retrieval_scopes()),
            top_k=max(1, int(request.top_k or self._retrieval_top_k)),
            include_private_runtime=False,
            include_stale=False,
            metadata_filters={},
        )
        retrieval_result = self.retrieve(retrieval_query)

        private_runtime = self._private_runtime_for_namespace(request.namespace)
        working = self._working_for_namespace(namespace=request.namespace, scopes=list(request.scopes or []))
        req = ContextBuildRequest(
            system_prompt=request.system_prompt,
            user_message=request.user_message,
            namespace=request.namespace,
            scopes=list(request.scopes or []),
            top_k=request.top_k,
            session_summary=request.session_summary or self._session_summary,
            working_memory=working,
            tool_state=dict(request.tool_state or {}),
            unresolved_items=list(request.unresolved_items or self._open_questions),
            context_budget_total=max(256, budget_total),
            context_budget_memory=max(64, budget_memory),
            context_budget_docs=max(64, budget_docs),
            context_budget_tools=max(32, budget_tools),
            context_budget_response_reserve=max(64, budget_reserve),
        )

        result = self._context_builder.build(
            request=req,
            retrieved=list(retrieval_result.candidates or []),
            private_runtime_state=private_runtime,
        )
        self._debugger.record_retrieval_trace(
            query=req.user_message,
            selected=list(result.selected or []),
            dropped=list(result.dropped or []),
        )
        return result

    def ingest_document(self, request: DocumentIngestRequest) -> DocumentIngestResult:
        return self._document_memory.ingest_document(request)

    def debug_snapshot(self, request: DebugRequest) -> dict[str, Any]:
        self._cleanup_expired()
        return self._debugger.snapshot(request)

    def reindex_embeddings(self, *, namespace: str | None = None, incremental: bool = False) -> dict[str, Any]:
        return self._store.rebuild_indexes(namespace=namespace, incremental=incremental)

    def close(self) -> None:
        with self._lock:
            # Runtime-only state should not survive session termination.
            self._private_runtime = {}
            self._save_state()
            self._store.close()

    def set_private_runtime_state(
        self,
        *,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl_sec: int | None = None,
    ) -> None:
        with self._lock:
            now_ts = float(time.time())
            ttl = max(30, int(ttl_sec or self._private_runtime_ttl_sec))
            self._private_runtime[str(key)] = {
                "value": value,
                "namespace": str(namespace or "default"),
                "expires_at": now_ts + ttl,
                "updated_at": now_ts,
            }
            self._save_state()

    def _write_fact_records(
        self,
        *,
        facts: list[FactRecordV2],
        namespace: str,
        now_ts: float,
        event_id: str,
    ) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        for fact in list(facts or []):
            canonical = str(fact.canonical_key or f"{fact.subject}.{fact.predicate}")
            existing = self._find_active_fact_by_canonical(namespace=namespace, canonical_key=canonical)
            record = MemoryRecord(
                id=f"fact:{uuid.uuid4().hex[:18]}",
                text=f"{fact.subject}.{fact.predicate}={fact.value}",
                memory_type=MemoryType.FACT,
                level=MemoryLevel.L3_SEMANTIC,
                scope=fact.scope,
                namespace=namespace,
                metadata={
                    "fact": fact.to_dict(),
                    "canonical_key": canonical,
                    "relation": fact.relation,
                },
                importance=float(fact.importance),
                confidence=float(fact.confidence),
                created_at=now_ts,
                updated_at=now_ts,
                expires_at=fact.valid_to,
                status=fact.status,
                source_event_id=event_id,
            )

            if existing is not None and str(existing.text) != str(record.text):
                decision = self._lifecycle.resolve_conflict(old=existing, new=record)
                if decision.superseded_record_id:
                    superseded = MemoryRecord(
                        id=existing.id,
                        text=existing.text,
                        memory_type=existing.memory_type,
                        level=existing.level,
                        scope=existing.scope,
                        namespace=existing.namespace,
                        metadata=dict(existing.metadata or {}),
                        embedding=list(existing.embedding or []) if isinstance(existing.embedding, list) else None,
                        importance=existing.importance,
                        confidence=existing.confidence,
                        created_at=existing.created_at,
                        updated_at=now_ts,
                        expires_at=existing.expires_at,
                        status=MemoryStatus.SUPERSEDED,
                        version=existing.version + 1,
                        parent_id=existing.parent_id,
                        chunk_index=existing.chunk_index,
                        source_event_id=existing.source_event_id,
                        embedding_model=existing.embedding_model,
                        embedding_fingerprint=existing.embedding_fingerprint,
                        embedding_version=existing.embedding_version,
                    )
                    self._store.upsert(superseded)
                if decision.keep_record_id != record.id:
                    continue

            self._store.upsert(record)
            out.append(fact)
        return out

    def _find_active_fact_by_canonical(self, *, namespace: str, canonical_key: str) -> MemoryRecord | None:
        rows = self._store.iter_records(namespace=namespace)
        key = str(canonical_key or "").strip().lower()
        if not key:
            return None
        for row in rows:
            if row.memory_type != MemoryType.FACT:
                continue
            if row.status != MemoryStatus.ACTIVE:
                continue
            existing = str(dict(row.metadata or {}).get("canonical_key") or "").strip().lower()
            if existing == key:
                return row
        return None

    def _promote_record(self, record: MemoryRecord, *, target: MemoryLevel, now_ts: float) -> MemoryRecord:
        return MemoryRecord(
            id=f"{record.id}:p:{target.value}",
            text=record.text,
            memory_type=record.memory_type,
            level=target,
            scope=record.scope,
            namespace=record.namespace,
            metadata={**dict(record.metadata or {}), "promoted_from": record.level.value},
            embedding=list(record.embedding or []) if isinstance(record.embedding, list) else None,
            importance=min(1.0, float(record.importance) + 0.04),
            confidence=record.confidence,
            created_at=record.created_at,
            updated_at=now_ts,
            expires_at=record.expires_at,
            status=record.status,
            version=record.version + 1,
            parent_id=record.id,
            chunk_index=record.chunk_index,
            source_event_id=record.source_event_id,
            embedding_model=record.embedding_model,
            embedding_fingerprint=record.embedding_fingerprint,
            embedding_version=record.embedding_version,
        )

    def _upsert_working_record(self, record: MemoryRecord) -> None:
        rows = [x for x in self._working_records if x.id != record.id]
        rows.append(record)
        rows.sort(key=lambda x: float(x.updated_at), reverse=True)
        self._working_records = rows[: max(20, int(self._working_limit))]

    def _working_for_namespace(self, *, namespace: str, scopes: list[MemoryScope]) -> list[MemoryRecord]:
        allowed = set(scopes or self._default_retrieval_scopes())
        out: list[MemoryRecord] = []
        for row in list(self._working_records or []):
            if row.namespace != str(namespace):
                continue
            if row.scope not in allowed:
                continue
            if row.scope == MemoryScope.PRIVATE_RUNTIME:
                continue
            if row.status in {MemoryStatus.DELETED, MemoryStatus.ARCHIVED}:
                continue
            out.append(row)
        out.sort(key=lambda x: (float(x.importance), float(x.updated_at)), reverse=True)
        return out[:40]

    def _private_runtime_for_namespace(self, namespace: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, row in dict(self._private_runtime or {}).items():
            if str(row.get("namespace") or "default") != str(namespace or "default"):
                continue
            out[str(key)] = row.get("value")
        return out

    def _cleanup_expired(self) -> None:
        now_ts = float(time.time())
        # Cleanup in-memory private runtime entries.
        self._private_runtime = {
            k: v
            for k, v in dict(self._private_runtime or {}).items()
            if float(v.get("expires_at") or 0.0) > now_ts
        }

        # Mark expired records from store as deleted.
        changed: list[MemoryRecord] = []
        for row in self._store.iter_records():
            if row.expires_at is None:
                continue
            if float(row.expires_at) > now_ts:
                continue
            if row.status == MemoryStatus.DELETED:
                continue
            changed.append(
                MemoryRecord(
                    id=row.id,
                    text=row.text,
                    memory_type=row.memory_type,
                    level=row.level,
                    scope=row.scope,
                    namespace=row.namespace,
                    metadata=dict(row.metadata or {}),
                    embedding=list(row.embedding or []) if isinstance(row.embedding, list) else None,
                    importance=row.importance,
                    confidence=row.confidence,
                    created_at=row.created_at,
                    updated_at=now_ts,
                    expires_at=row.expires_at,
                    status=MemoryStatus.DELETED,
                    version=row.version + 1,
                    parent_id=row.parent_id,
                    chunk_index=row.chunk_index,
                    source_event_id=row.source_event_id,
                    embedding_model=row.embedding_model,
                    embedding_fingerprint=row.embedding_fingerprint,
                    embedding_version=row.embedding_version,
                )
            )
        if changed:
            self._store.batch_upsert(changed)

    def _default_retrieval_scopes(self) -> list[MemoryScope]:
        return [
            MemoryScope.CONVERSATION,
            MemoryScope.SESSION,
            MemoryScope.PROJECT,
            MemoryScope.GLOBAL_USER,
            MemoryScope.CHARACTER,
            MemoryScope.TEMPORARY,
        ]

    def _initial_level(self, memory_type: MemoryType) -> MemoryLevel:
        if memory_type in {MemoryType.DOCUMENT, MemoryType.DOCUMENT_CHUNK}:
            return MemoryLevel.L4_DOCUMENT
        if memory_type == MemoryType.FACT:
            return MemoryLevel.L3_SEMANTIC
        if memory_type == MemoryType.SUMMARY:
            return MemoryLevel.L1_SESSION
        return MemoryLevel.L0_WORKING

    def _expires_at(self, *, scope: MemoryScope, metadata: dict[str, Any], now_ts: float) -> float | None:
        if scope == MemoryScope.TEMPORARY:
            ttl = int(metadata.get("ttl_sec") or self._temporary_ttl_sec)
            return float(now_ts + max(30, ttl))
        if scope == MemoryScope.PRIVATE_RUNTIME:
            ttl = int(metadata.get("ttl_sec") or self._private_runtime_ttl_sec)
            return float(now_ts + max(30, ttl))
        return None

    @staticmethod
    def _confidence_score(*, metadata: dict[str, Any]) -> float:
        try:
            value = float(metadata.get("confidence") or 0.65)
        except Exception:
            value = 0.65
        return max(0.0, min(1.0, value))

    @staticmethod
    def _importance_score(*, text: str, metadata: dict[str, Any]) -> float:
        explicit = metadata.get("importance")
        if explicit is not None:
            try:
                value = float(explicit)
                return max(0.0, min(1.0, value))
            except Exception:
                pass

        src = str(text or "").lower()
        score = 0.42
        if any(token in src for token in ("decide", "decision", "решили", "фикс", "issue", "error", "bug")):
            score += 0.24
        if any(token in src for token in ("remember", "важно", "save", "запомни")):
            score += 0.18
        if any(token in src for token in ("project", "архитект", "design", "release")):
            score += 0.10
        return max(0.0, min(1.0, score))

    def _update_session_state_from_event(self, record: MemoryRecord) -> None:
        text = str(record.text or "").strip()
        if not text:
            return

        if record.scope == MemoryScope.SESSION and record.memory_type == MemoryType.SUMMARY:
            self._session_summary = text[:2400]

        low = text.lower()
        if "?" in text and len(text) <= 220:
            self._open_questions = self._append_unique_tail(self._open_questions, text, limit=24)
        if any(token in low for token in ("decide", "decision", "решили", "приняли")):
            self._current_decisions = self._append_unique_tail(self._current_decisions, text, limit=32)
        if any(token in low for token in ("prefer", "like", "предпочитаю", "люблю")):
            self._active_preferences = self._append_unique_tail(self._active_preferences, text, limit=32)

    @staticmethod
    def _append_unique_tail(items: list[str], value: str, *, limit: int) -> list[str]:
        src = str(value or "").strip()
        if not src:
            return list(items or [])
        out: list[str] = []
        seen: set[str] = set()
        for row in list(items or []):
            item = str(row or "").strip()
            if not item:
                continue
            low = item.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(item)
        if src.lower() not in seen:
            out.append(src)
        return out[-max(1, int(limit)) :]

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8-sig") or "{}")
        except Exception:
            return
        self._session_summary = str(payload.get("session_summary") or "")
        self._open_questions = [str(x) for x in list(payload.get("open_questions") or []) if str(x).strip()]
        self._current_decisions = [str(x) for x in list(payload.get("current_decisions") or []) if str(x).strip()]
        self._active_preferences = [str(x) for x in list(payload.get("active_preferences") or []) if str(x).strip()]
        self._private_runtime = dict(payload.get("private_runtime") or {})
        self._working_records = []
        for row in list(payload.get("working_records") or []):
            if not isinstance(row, dict):
                continue
            try:
                self._working_records.append(MemoryRecord.from_dict(row))
            except Exception:
                continue

    def _save_state(self) -> None:
        payload = {
            "session_summary": self._session_summary,
            "open_questions": list(self._open_questions or []),
            "current_decisions": list(self._current_decisions or []),
            "active_preferences": list(self._active_preferences or []),
            "private_runtime": dict(self._private_runtime or {}),
            "working_records": [x.to_dict() for x in list(self._working_records or [])],
        }
        self._state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

