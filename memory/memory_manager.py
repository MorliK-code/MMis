from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config
from llm.tokenizer import create_tokenizer
from memory.document_memory import ChunkingConfig, DocumentMemory
from memory.embedding_provider import build_embedding_provider
from memory.event_store import EventStore
from memory.fact_extractor import FactExtractor
from memory.ingest_analyzer import IngestAnalysis, analyze_message_for_memory
from memory.long_memory import LongMemoryV2
from memory.memory_debug import BasicMemoryDebugger
from memory.memory_lifecycle import MemoryLifecycleManager
from memory.memory_models import (
    ContextCompressor,
    ContextBuildRequest,
    ContextBuildResult,
    DebugRequest,
    DocumentIngestRequest,
    DocumentIngestResult,
    EmbeddingProvider,
    FactRecordV2,
    IngestResult,
    MemoryDebugger,
    MemoryEvent,
    MemoryLifecycle,
    MemoryLevel,
    LifecycleDecision,
    MemoryRecord,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    MemoryType,
    Reranker,
    RetrievalQuery,
    RetrievalResult,
)
from memory.memory_policy import AssistantWriteDecision, MemoryPolicy
from memory.retrieval import HybridRetriever
from memory.reranker import HeuristicReranker
from memory.memory_scoring import (
    SalienceWeights,
    ScoreWeights,
    build_message_signal_breakdown,
    build_salience_score,
)
from memory.retrieval_projection import (
    build_memory_search_text,
    extract_query_entity_keys,
    extract_query_numeric_keys,
    merge_projection_keys,
)
from memory.vector_store import VectorStore
from memory.context_builder import ContextBuilderV2
from utils.logger import get_logger, log_json


LOGGER = get_logger(__name__)


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
        memory_backend = str(getattr(self._cfg, "memory_backend", "chroma") or "chroma").strip().lower()
        if memory_backend not in {"chroma", "chromadb"}:
            raise ValueError(
                "Unsupported memory.backend for Memory V2: "
                f"{memory_backend!r}. Use 'chroma' or 'chromadb'."
            )

        embedding_backend = str(getattr(self._cfg, "memory_embedding_backend", "") or "").strip()
        if not embedding_backend:
            embedding_backend = "sentence_transformers"
        embedding_model = str(getattr(self._cfg, "memory_embedding_model", "") or "").strip()
        if not embedding_model:
            embedding_model = "all-MiniLM-L6-v2"
        self._embedding_provider: EmbeddingProvider = build_embedding_provider(
            backend=embedding_backend,
            model_name=embedding_model,
            cache_path=self._root / "embedding_cache.sqlite3",
            dim=int(getattr(self._cfg, "memory_embedding_dim", 384) or 384),
            ollama_host=str(getattr(self._cfg, "ollama_base_url", "") or ""),
            ollama_timeout_sec=float(getattr(self._cfg, "ollama_timeout_sec", 120.0) or 120.0),
        )

        try:
            self._store = VectorStore(
                root_dir=self._root,
                embedding_provider=self._embedding_provider,
                use_chroma=True,
                collection_name="mmis_memory_v2",
            )
        except Exception:
            if hasattr(self._embedding_provider, "close"):
                try:
                    self._embedding_provider.close()
                except Exception:
                    pass
            raise

        self._event_store = EventStore(path=self._root / "events_v2.jsonl")
        self._fact_extractor = FactExtractor()
        self._policy = MemoryPolicy()
        self._lifecycle: MemoryLifecycle = MemoryLifecycleManager(
            stale_after_days=int(getattr(self._cfg, "memory_stale_after_days", 30) or 30),
            archive_after_days=int(getattr(self._cfg, "memory_archive_after_days", 90) or 90),
            promote_message_importance_threshold=float(
                getattr(self._cfg, "memory_promotion_message_importance_threshold", 0.55) or 0.55
            ),
            promote_message_confidence_threshold=float(
                getattr(self._cfg, "memory_promotion_message_confidence_threshold", 0.50) or 0.50
            ),
            promote_project_signal_boost=float(
                getattr(self._cfg, "memory_promotion_project_signal_boost", 0.12) or 0.12
            ),
            promote_fact_signal_boost=float(
                getattr(self._cfg, "memory_promotion_fact_signal_boost", 0.16) or 0.16
            ),
            promote_decision_signal_boost=float(
                getattr(self._cfg, "memory_promotion_decision_signal_boost", 0.12) or 0.12
            ),
            promote_smalltalk_penalty=float(
                getattr(self._cfg, "memory_promotion_smalltalk_penalty", 0.20) or 0.20
            ),
        )
        self._retriever = HybridRetriever(
            store=self._store,
            stale_after_days=int(getattr(self._cfg, "memory_stale_after_days", 30) or 30),
            weights=ScoreWeights(
                semantic_similarity=float(
                    getattr(self._cfg, "memory_retrieval_weight_semantic_similarity", 0.34) or 0.34
                ),
                lexical_score=float(getattr(self._cfg, "memory_retrieval_weight_lexical_score", 0.25) or 0.25),
                recency_score=float(getattr(self._cfg, "memory_retrieval_weight_recency_score", 0.10) or 0.10),
                importance_score=float(
                    getattr(self._cfg, "memory_retrieval_weight_importance_score", 0.09) or 0.09
                ),
                confidence_score=float(
                    getattr(self._cfg, "memory_retrieval_weight_confidence_score", 0.08) or 0.08
                ),
                entity_overlap_score=float(
                    getattr(self._cfg, "memory_retrieval_weight_entity_overlap_score", 0.07) or 0.07
                ),
                exact_match_boost=float(
                    getattr(self._cfg, "memory_retrieval_weight_exact_match_boost", 0.04) or 0.04
                ),
                scope_match_score=float(
                    getattr(self._cfg, "memory_retrieval_weight_scope_match_score", 0.03) or 0.03
                ),
            ),
        )
        self._reranker: Reranker = HeuristicReranker()
        self._context_builder = ContextBuilderV2(tokenizer=create_tokenizer())
        self._document_memory = DocumentMemory(
            store=self._store,
            chunking=ChunkingConfig(
                chunk_size=int(getattr(self._cfg, "memory_chunk_size", 1200) or 1200),
                chunk_overlap=int(getattr(self._cfg, "memory_chunk_overlap", 160) or 160),
            ),
        )
        self._long_memory = LongMemoryV2(store=self._store, document_memory=self._document_memory)
        self._debugger: MemoryDebugger = BasicMemoryDebugger(store=self._store)
        # Phase 1 extension point: real compressor can be injected in next phases.
        self._context_compressor: ContextCompressor | None = None

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
        self._importance_weights = {
            "base": float(getattr(self._cfg, "memory_importance_weight_base", 0.42) or 0.42),
            "decision": float(getattr(self._cfg, "memory_importance_weight_decision", 0.24) or 0.24),
            "remember": float(getattr(self._cfg, "memory_importance_weight_remember", 0.18) or 0.18),
            "project": float(getattr(self._cfg, "memory_importance_weight_project", 0.10) or 0.10),
        }
        self._salience_weights = SalienceWeights(
            novelty=float(getattr(self._cfg, "memory_salience_weight_novelty", 0.22) or 0.22),
            permanence=float(getattr(self._cfg, "memory_salience_weight_permanence", 0.20) or 0.20),
            repetition=float(getattr(self._cfg, "memory_salience_weight_repetition", 0.14) or 0.14),
            project_relevance=float(
                getattr(self._cfg, "memory_salience_weight_project_relevance", 0.16) or 0.16
            ),
            task_relevance=float(getattr(self._cfg, "memory_salience_weight_task_relevance", 0.16) or 0.16),
            explicit_save_signal=float(
                getattr(self._cfg, "memory_salience_weight_explicit_save_signal", 0.12) or 0.12
            ),
        )

        self._load_state()
        self._cleanup_expired()

    def ingest_event(self, event: MemoryEvent) -> IngestResult:
        with self._lock:
            self._cleanup_expired()

            text = str(event.text or "").strip()
            now_ts = float(event.ts or time.time())
            event_id = f"evt:{uuid.uuid4().hex[:16]}"
            metadata = dict(event.metadata or {})
            namespace = str(event.namespace or "default")
            scope = event.scope
            memory_type = event.memory_type
            thinking = str(event.thinking or metadata.get("thinking") or "").strip()
            source_kind = self._policy.resolve_source_kind(
                role=str(event.role or ""),
                memory_type=memory_type,
                metadata=metadata,
                thinking=thinking,
            )
            metadata = self._policy.sanitize_metadata_for_storage(
                metadata=metadata,
                source_kind=source_kind,
                thinking=thinking,
            )
            source_fact_records_allowed = self._policy.allow_fact_records_for_source(source_kind=source_kind)
            metadata["source_fact_records_allowed"] = bool(source_fact_records_allowed)
            if not source_fact_records_allowed:
                metadata["source_fact_records_blocked"] = True

            if not text and event.scope != MemoryScope.PRIVATE_RUNTIME:
                return IngestResult(stored_ids=[])

            if memory_type == MemoryType.SUMMARY and scope == MemoryScope.CONVERSATION:
                scope = MemoryScope.SESSION

            if source_kind == MemorySourceKind.ASSISTANT_THOUGHT and scope != MemoryScope.PRIVATE_RUNTIME:
                decision = AssistantWriteDecision(
                    action="skip",
                    reason="assistant_thought_not_persisted",
                    allow_fact_records=False,
                    signals={"source_kind": str(source_kind.value)},
                )
                metadata["assistant_write_policy"] = decision.to_dict()
                metadata["assistant_write_blocked"] = True
                metadata["assistant_write_reason"] = str(decision.reason or "")
                self._record_ingest_skip(
                    event_id=event_id,
                    now_ts=now_ts,
                    event=event,
                    namespace=namespace,
                    metadata=metadata,
                    record_id=f"{memory_type.value}:{event_id}",
                    reason=str(decision.reason or "assistant_thought_not_persisted"),
                    decision=decision,
                )
                return IngestResult(
                    stored_ids=[],
                    dropped_ids=[f"{memory_type.value}:{event_id}"],
                    extracted_facts=[],
                )

            if memory_type == MemoryType.DOCUMENT and scope != MemoryScope.PRIVATE_RUNTIME:
                source = str(metadata.get("source") or metadata.get("path") or f"event:{event_id}").strip()
                title = str(metadata.get("title") or "").strip()
                doc_request = DocumentIngestRequest(
                    text=text,
                    source=source,
                    namespace=namespace,
                    scope=(scope if scope != MemoryScope.CONVERSATION else MemoryScope.PROJECT),
                    title=title,
                    metadata={**metadata, "event_id": event_id, "namespace": namespace},
                )
                doc_result = self._long_memory.ingest_document(doc_request)
                stored_ids = [doc_result.document.id] + [row.id for row in list(doc_result.chunks or [])]
                working_doc = MemoryRecord(
                    id=doc_result.document.id,
                    text=doc_result.document.summary,
                    memory_type=MemoryType.DOCUMENT,
                    level=MemoryLevel.L4_DOCUMENT,
                    scope=doc_result.document.scope,
                    namespace=namespace,
                    metadata=dict(doc_result.document.metadata or {}),
                    importance=float(doc_result.document.importance),
                    confidence=float(doc_result.document.confidence),
                    created_at=doc_result.document.created_at,
                    updated_at=doc_result.document.updated_at,
                    status=doc_result.document.status,
                    version=int(doc_result.document.version),
                )
                self._upsert_working_record(working_doc)
                self._event_store.append(
                    {
                        "event_id": event_id,
                        "ts": now_ts,
                        "type": "memory_document_ingest_v2",
                        "trace_id": str(metadata.get("trace_id") or ""),
                        "model": str(metadata.get("model") or ""),
                        "latency_ms": float(metadata.get("latency_ms") or 0.0),
                        "payload": {
                            "role": str(event.role or ""),
                            "scope": scope.value,
                            "memory_type": memory_type.value,
                            "record_id": doc_result.document.id,
                            "namespace": namespace,
                            "chunk_count": len(doc_result.chunks),
                            "request_id": str(metadata.get("request_id") or ""),
                            "turn_id": str(metadata.get("turn_id") or ""),
                            "conversation_id": str(metadata.get("conversation_id") or namespace),
                        },
                        "tags": [scope.value, memory_type.value, "document"],
                    }
                )
                self._save_state()
                log_json(
                    LOGGER,
                    "memory_v2_ingest_document",
                    summary=(
                        f"type={memory_type.value} scope={scope.value} stored={len(stored_ids)} "
                        f"source={source or '-'}"
                    ),
                    context=self._event_log_context(metadata=metadata, namespace=namespace),
                    namespace=namespace,
                    scope=scope.value,
                    source=source,
                    stored=len(stored_ids),
                )
                return IngestResult(stored_ids=stored_ids)

            preview_facts: list[FactRecordV2] = []
            ingest_analysis: IngestAnalysis | None = None
            if memory_type in {MemoryType.MESSAGE, MemoryType.SUMMARY} and scope != MemoryScope.PRIVATE_RUNTIME:
                ingest_analysis = analyze_message_for_memory(
                    text,
                    metadata={"event_id": event_id, "namespace": namespace, **metadata},
                )
                metadata = self._merge_ingest_analysis_into_metadata(metadata=metadata, analysis=ingest_analysis)
                if source_fact_records_allowed:
                    preview_facts = self._fact_extractor.extract_v2(
                        text=text,
                        metadata={"event_id": event_id, "namespace": namespace, **metadata},
                        speaker=str(event.role or "user"),
                        scope=scope,
                        mode=str(metadata.get("quality_profile") or "BALANCED"),
                        analysis=ingest_analysis,
                    )
                metadata = self._augment_message_metadata(
                    text=text,
                    metadata=metadata,
                    namespace=namespace,
                    preview_facts=preview_facts,
                )

            assistant_write_decision = AssistantWriteDecision(reason="not_applicable")
            if str(event.role or "").strip().lower() == "assistant":
                assistant_write_decision = self._policy.decide_assistant_message_write(
                    text=text,
                    metadata=metadata,
                    memory_type=memory_type,
                    requested_scope=scope,
                )
                metadata["assistant_write_policy"] = assistant_write_decision.to_dict()
                metadata["assistant_write_blocked"] = bool(not assistant_write_decision.allow_store)
                metadata["assistant_write_reason"] = str(assistant_write_decision.reason or "")
                if assistant_write_decision.target_scope is not None:
                    scope = assistant_write_decision.target_scope
                if scope == MemoryScope.TEMPORARY:
                    metadata.setdefault("ttl_sec", int(self._temporary_ttl_sec))
                if not assistant_write_decision.allow_fact_records or not source_fact_records_allowed:
                    metadata["assistant_fact_records_blocked"] = True

            level = self._initial_level(memory_type)
            importance = self._importance_score(text=text, metadata=metadata, namespace=namespace)
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
            if not assistant_write_decision.allow_store:
                self._record_ingest_skip(
                    event_id=event_id,
                    now_ts=now_ts,
                    event=event,
                    namespace=namespace,
                    metadata=metadata,
                    record_id=record.id,
                    reason=str(assistant_write_decision.reason or "assistant_write_policy_skip"),
                    decision=assistant_write_decision,
                )
                return IngestResult(
                    stored_ids=[],
                    dropped_ids=[record.id],
                    extracted_facts=[],
                )

            lifecycle_decision = self._lifecycle.decide(record, now_ts=now_ts)
            if not assistant_write_decision.allow_long_term:
                lifecycle_decision = self._without_promotion(
                    lifecycle_decision,
                    reason=str(assistant_write_decision.reason or "assistant_write_policy"),
                    scope=scope,
                    decision=assistant_write_decision,
                )
            record = self._attach_lifecycle_debug(
                record,
                lifecycle_decision=lifecycle_decision,
                extracted_facts_count=len(preview_facts),
            )

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

            if (
                scope != MemoryScope.PRIVATE_RUNTIME
                and lifecycle_decision.mark_status is not None
                and lifecycle_decision.mark_status != record.status
            ):
                status_row = self._status_transition(
                    record,
                    target=lifecycle_decision.mark_status,
                    now_ts=now_ts,
                    reason=str(lifecycle_decision.reason or "lifecycle"),
                )
                self._store.upsert(status_row)
            if (
                scope != MemoryScope.PRIVATE_RUNTIME
                and lifecycle_decision.promote_to is not None
                and lifecycle_decision.promote_to != record.level
            ):
                promoted = self._promote_record(record, target=lifecycle_decision.promote_to, now_ts=now_ts)
                self._store.upsert(promoted)
                promoted_ids.append(promoted.id)

            writeable_preview_facts = (
                list(preview_facts)
                if assistant_write_decision.allow_fact_records and source_fact_records_allowed
                else []
            )
            if scope != MemoryScope.PRIVATE_RUNTIME and writeable_preview_facts:
                extracted_facts = self._write_fact_records(
                    facts=writeable_preview_facts,
                    namespace=namespace,
                    now_ts=now_ts,
                    event_id=event_id,
                )

            self._event_store.append(
                {
                    "event_id": event_id,
                    "ts": now_ts,
                    "type": "memory_ingest_v2",
                    "trace_id": str(metadata.get("trace_id") or ""),
                    "model": str(metadata.get("model") or ""),
                    "latency_ms": float(metadata.get("latency_ms") or 0.0),
                    "payload": {
                        "role": str(event.role or ""),
                        "source_kind": str(source_kind.value),
                        "scope": scope.value,
                        "memory_type": memory_type.value,
                        "record_id": record.id,
                        "namespace": namespace,
                        "lifecycle_reason": str(lifecycle_decision.reason or ""),
                        "lifecycle_route": str(lifecycle_decision.route or ""),
                        "promote_to": (
                            str(lifecycle_decision.promote_to.value)
                            if lifecycle_decision.promote_to is not None
                            else ""
                        ),
                        "preview_facts_count": int(len(preview_facts)),
                        "assistant_write_policy": dict(metadata.get("assistant_write_policy") or {}),
                        "request_id": str(metadata.get("request_id") or ""),
                        "turn_id": str(metadata.get("turn_id") or ""),
                        "conversation_id": str(metadata.get("conversation_id") or namespace),
                    },
                    "tags": [scope.value, memory_type.value],
                }
                )
            self._save_state()

        assistant_policy = dict(metadata.get("assistant_write_policy") or {})
        log_json(
            LOGGER,
            "memory_v2_ingest",
            summary=(
                f"type={memory_type.value} scope={scope.value} stored={len(stored_ids)} "
                f"facts={len(extracted_facts)} promotions={len(promoted_ids)} "
                f"reason={str(lifecycle_decision.reason or '-')} "
                f"write_policy={str(assistant_policy.get('action') or 'allow')} "
                f"write_reason={str(assistant_policy.get('reason') or '-')}"
            ),
            context=self._event_log_context(metadata=metadata, namespace=namespace),
            namespace=namespace,
            scope=scope.value,
            memory_type=memory_type.value,
            source_kind=str(source_kind.value),
            stored=len(stored_ids),
            promoted=len(promoted_ids),
            facts=len(extracted_facts),
            promotion_reason=str(lifecycle_decision.reason or ""),
            promotion_route=str(lifecycle_decision.route or ""),
            promotion_score=float(
                dict(lifecycle_decision.decision_debug or {}).get("composite_score") or 0.0
            ),
            promotion_threshold=float(
                dict(lifecycle_decision.decision_debug or {}).get("composite_threshold") or 0.0
            ),
            assistant_write_blocked=bool(metadata.get("assistant_write_blocked")),
            assistant_write_reason=str(metadata.get("assistant_write_reason") or ""),
            assistant_write_policy=assistant_policy,
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
            raw_query_text = str(query.query_text or "")
            search_seed = str(query.search_text or raw_query_text)
            entity_keys = merge_projection_keys(
                list(query.entity_keys or []),
                extract_query_entity_keys(raw_query_text),
                extract_query_entity_keys(search_seed),
            )
            numeric_keys = merge_projection_keys(
                list(query.numeric_keys or []),
                extract_query_numeric_keys(raw_query_text),
                extract_query_numeric_keys(search_seed),
            )
            normalized_query = RetrievalQuery(
                query_text=raw_query_text,
                search_text=build_memory_search_text(
                    search_seed,
                    entity_keys=entity_keys,
                    numeric_keys=numeric_keys,
                ),
                entity_keys=entity_keys,
                numeric_keys=numeric_keys,
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
        def _cfg_budget(name: str, *, minimum: int) -> int:
            try:
                value = int(getattr(self._cfg, name))
            except Exception:
                value = minimum
            return max(minimum, value)

        def _pick_budget(value: Any, *, default: int, minimum: int) -> int:
            try:
                parsed = int(value)
            except Exception:
                parsed = 0
            if parsed <= 0:
                parsed = default
            return max(minimum, parsed)

        budget_total = _pick_budget(
            request.context_budget_total,
            default=_cfg_budget("memory_context_budget_total", minimum=256),
            minimum=256,
        )
        budget_memory = _pick_budget(
            request.context_budget_memory,
            default=_cfg_budget("memory_context_budget_memory", minimum=64),
            minimum=64,
        )
        budget_docs = _pick_budget(
            request.context_budget_docs,
            default=_cfg_budget("memory_context_budget_docs", minimum=64),
            minimum=64,
        )
        budget_tools = _pick_budget(
            request.context_budget_tools,
            default=_cfg_budget("memory_context_budget_tools", minimum=32),
            minimum=32,
        )
        budget_reserve = _pick_budget(
            request.context_budget_response_reserve,
            default=_cfg_budget("memory_context_budget_response_reserve", minimum=64),
            minimum=64,
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
            context_budget_total=budget_total,
            context_budget_memory=budget_memory,
            context_budget_docs=budget_docs,
            context_budget_tools=budget_tools,
            context_budget_response_reserve=budget_reserve,
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
            score_breakdowns=list(result.score_breakdowns or []),
            truncation_log=list(result.truncation_log or []),
            context_blocks=dict(result.blocks or {}),
        )
        return result

    def ingest_document(self, request: DocumentIngestRequest) -> DocumentIngestResult:
        with self._lock:
            result = self._long_memory.ingest_document(request)
            self._upsert_working_record(
                MemoryRecord(
                    id=result.document.id,
                    text=result.document.summary,
                    memory_type=MemoryType.DOCUMENT,
                    level=MemoryLevel.L4_DOCUMENT,
                    scope=result.document.scope,
                    namespace=result.document.namespace,
                    metadata=dict(result.document.metadata or {}),
                    importance=result.document.importance,
                    confidence=result.document.confidence,
                    created_at=result.document.created_at,
                    updated_at=result.document.updated_at,
                    status=result.document.status,
                    version=result.document.version,
                )
            )
            self._save_state()
            return result

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
            if hasattr(self._embedding_provider, "close"):
                try:
                    self._embedding_provider.close()
                except Exception:
                    pass

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
                action = str(decision.action or "").strip().lower()
                force_supersede_existing = bool(
                    action == "parallel" and self._is_singleton_fact_canonical(canonical)
                )

                if force_supersede_existing or (
                    decision.superseded_record_id and str(decision.superseded_record_id) == str(existing.id)
                ):
                    superseded = self._status_transition(
                        existing,
                        target=MemoryStatus.SUPERSEDED,
                        now_ts=now_ts,
                        reason=(
                            "singleton_fact_override"
                            if force_supersede_existing
                            else str(decision.reason or "conflict_supersede")
                        ),
                    )
                    self._store.upsert(superseded)
                if decision.archive_record_id and str(decision.archive_record_id) == str(existing.id):
                    archived = self._status_transition(
                        existing,
                        target=MemoryStatus.ARCHIVED,
                        now_ts=now_ts,
                        reason=str(decision.reason or "conflict_archive"),
                    )
                    self._store.upsert(archived)
                if action == "parallel" and not force_supersede_existing:
                    record = MemoryRecord(
                        id=record.id,
                        text=record.text,
                        memory_type=record.memory_type,
                        level=record.level,
                        scope=record.scope,
                        namespace=record.namespace,
                        metadata={
                            **dict(record.metadata or {}),
                            "conflict_resolution": {
                                "action": "parallel",
                                "parallel_with": decision.parallel_with_record_id,
                                "reason": decision.reason,
                                "score_delta": float(decision.score_delta),
                            },
                        },
                        embedding=list(record.embedding or []) if isinstance(record.embedding, list) else None,
                        importance=record.importance,
                        confidence=record.confidence,
                        created_at=record.created_at,
                        updated_at=record.updated_at,
                        expires_at=record.expires_at,
                        status=MemoryStatus.ACTIVE,
                        version=record.version,
                        parent_id=(record.parent_id or existing.id),
                        chunk_index=record.chunk_index,
                        source_event_id=record.source_event_id,
                        embedding_model=record.embedding_model,
                        embedding_fingerprint=record.embedding_fingerprint,
                        embedding_version=record.embedding_version,
                    )
                if not force_supersede_existing and decision.keep_record_id != record.id:
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

    @staticmethod
    def _is_singleton_fact_canonical(canonical_key: str) -> bool:
        key = str(canonical_key or "").strip().lower()
        if not key:
            return False
        predicate = key.split(".", 1)[1] if "." in key else key
        return predicate in {
            "preference",
            "identity_name",
            "identity_age_years",
            "project_name",
            "environment",
            "environment_os",
            "environment_tool",
            "environment_runtime_python",
            "environment_llm_model",
            "environment_gpu_model",
            "environment_cpu_model",
            "environment_gpu_vram_size",
            "environment_gpu_vram_gb",
            "environment_ram_size",
            "environment_ram_gb",
            "environment_memory_gb",
            "issue_status",
        }

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

    @staticmethod
    def _status_transition(record: MemoryRecord, *, target: MemoryStatus, now_ts: float, reason: str) -> MemoryRecord:
        return MemoryRecord(
            id=record.id,
            text=record.text,
            memory_type=record.memory_type,
            level=record.level,
            scope=record.scope,
            namespace=record.namespace,
            metadata={
                **dict(record.metadata or {}),
                "previous_status": str(record.status.value),
                "lifecycle_reason": str(reason or ""),
            },
            embedding=list(record.embedding or []) if isinstance(record.embedding, list) else None,
            importance=record.importance,
            confidence=record.confidence,
            created_at=record.created_at,
            updated_at=now_ts,
            expires_at=record.expires_at,
            status=target,
            version=int(record.version) + 1,
            parent_id=(record.parent_id or record.id),
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

        rows = self._store.iter_records()
        decayed = self._lifecycle.apply_decay(rows, now_ts=now_ts)

        # Mark expired records as deleted and apply stale/archive decay transitions.
        changed: list[MemoryRecord] = []
        for idx, row in enumerate(rows):
            if row.expires_at is None:
                pass
            elif float(row.expires_at) <= now_ts and row.status != MemoryStatus.DELETED:
                changed.append(
                    self._status_transition(
                        row,
                        target=MemoryStatus.DELETED,
                        now_ts=now_ts,
                        reason="ttl_expired",
                    )
                )
                continue

            next_row = decayed[idx] if idx < len(decayed) else row
            if (
                next_row.status != row.status
                or int(next_row.version) != int(row.version)
                or float(next_row.updated_at) != float(row.updated_at)
            ):
                changed.append(next_row)
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
    def _merge_ingest_analysis_into_metadata(
        *,
        metadata: dict[str, Any],
        analysis: IngestAnalysis | None,
    ) -> dict[str, Any]:
        out = dict(metadata or {})
        if analysis is None:
            return out

        def _merge_tags(existing: Any, incoming: list[str]) -> list[str]:
            seen: set[str] = set()
            tags: list[str] = []
            existing_items = (
                list(existing)
                if isinstance(existing, (list, tuple, set))
                else ([existing] if str(existing or "").strip() else [])
            )
            for raw in [*existing_items, *list(incoming or [])]:
                token = str(raw or "").strip().lower()
                if not token or token in seen:
                    continue
                seen.add(token)
                tags.append(token)
            return tags

        emotion_payload = analysis.emotion.to_dict() if analysis.emotion is not None else {}
        emotion_view = {**dict(emotion_payload or {})}
        if emotion_view and "label" not in emotion_view:
            emotion_view["label"] = str(emotion_view.get("primary") or "")
        out["analysis_version"] = "memory_ingest_v3"
        out["normalized_text"] = str(analysis.normalized_text or "")
        out["canonical_text"] = str(analysis.canonical_text or "")
        out["search_text"] = str(analysis.search_text or "")
        out["memory_views"] = dict(analysis.memory_views or {})
        out["memory_entities"] = [item.to_dict() for item in list(analysis.entities or [])]
        out["numeric_facts"] = [item.to_dict() for item in list(analysis.numeric_facts or [])]
        out["stable_facts"] = [item.to_dict() for item in list(analysis.stable_facts or [])]
        out["emotion_profile"] = emotion_view
        if emotion_view:
            out["emotion"] = emotion_view
        out["tags"] = _merge_tags(out.get("tags"), list(analysis.tags or []))
        out["memory_analysis"] = analysis.to_dict()
        return out

    def _augment_message_metadata(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
        namespace: str,
        preview_facts: list[FactRecordV2],
    ) -> dict[str, Any]:
        out = dict(metadata or {})
        recent = [
            str(row.text or "")
            for row in list(self._working_records or [])
            if str(row.namespace or "default") == str(namespace or "default")
        ][:80]
        signal = build_message_signal_breakdown(
            text=str(text or ""),
            metadata=out,
            recent_texts=recent,
        )
        fact_relations = sorted(
            {
                str(x.relation or "").strip().lower()
                for x in list(preview_facts or [])
                if str(x.relation or "").strip()
            }
        )
        facts_count = len(list(preview_facts or []))
        relation_weights = {
            "decision": 0.24,
            "issue": 0.22,
            "task": 0.20,
            "project": 0.18,
            "preference": 0.18,
            "environment": 0.14,
            "identity": 0.10,
            "temporary": 0.06,
            "resolved": 0.12,
            "unresolved": 0.12,
        }
        relation_bonus = sum(float(relation_weights.get(name, 0.08)) for name in fact_relations)
        fact_signal = self._clamp01((0.24 * min(3, facts_count)) + min(0.52, relation_bonus))
        fact_relation_diversity = self._clamp01(float(len(fact_relations)) / 3.0)

        out["promotion_project_signal"] = self._meta_float(out, "promotion_project_signal", signal.project_relevance)
        out["promotion_task_signal"] = self._meta_float(out, "promotion_task_signal", signal.task_intent)
        out["promotion_decision_signal"] = self._meta_float(out, "promotion_decision_signal", signal.decision_signal)
        out["promotion_preference_signal"] = self._meta_float(
            out, "promotion_preference_signal", signal.preference_signal
        )
        out["promotion_issue_signal"] = self._meta_float(out, "promotion_issue_signal", signal.issue_signal)
        out["promotion_technical_signal"] = self._meta_float(
            out, "promotion_technical_signal", signal.technical_relevance
        )
        out["promotion_repeated_topic_signal"] = self._meta_float(
            out, "promotion_repeated_topic_signal", signal.repeated_theme
        )
        out["promotion_smalltalk_signal"] = self._meta_float(out, "promotion_smalltalk_signal", signal.smalltalk)
        out["promotion_signal_score"] = self._meta_float(out, "promotion_signal_score", signal.meaningful_signal)
        out["promotion_stable_fact_signal"] = self._meta_float(
            out, "promotion_stable_fact_signal", signal.stable_fact_signal
        )
        out["promotion_fact_signal"] = self._meta_float(out, "promotion_fact_signal", fact_signal)
        out["promotion_fact_relation_diversity"] = self._meta_float(
            out, "promotion_fact_relation_diversity", fact_relation_diversity
        )
        out["extracted_facts_count"] = int(max(0, self._to_int(out.get("extracted_facts_count"), facts_count)))
        out["extracted_fact_relations"] = list(fact_relations)[:16]
        return out

    def _attach_lifecycle_debug(
        self,
        record: MemoryRecord,
        *,
        lifecycle_decision: LifecycleDecision,
        extracted_facts_count: int,
    ) -> MemoryRecord:
        meta = dict(record.metadata or {})
        debug_payload = dict(lifecycle_decision.decision_debug or {})
        lifecycle_payload = {
            "reason": str(lifecycle_decision.reason or ""),
            "route": str(lifecycle_decision.route or ""),
            "promote_to": (
                str(lifecycle_decision.promote_to.value) if lifecycle_decision.promote_to is not None else None
            ),
            "mark_status": (
                str(lifecycle_decision.mark_status.value) if lifecycle_decision.mark_status is not None else None
            ),
            "importance": float(record.importance),
            "confidence": float(record.confidence),
            "extracted_facts_count": int(max(0, extracted_facts_count)),
            "promotion_debug": dict(debug_payload or {}),
        }
        updated_meta = {
            **meta,
            "extracted_facts_count": int(
                max(0, self._to_int(meta.get("extracted_facts_count"), extracted_facts_count))
            ),
            "lifecycle_decision": lifecycle_payload,
            "promotion_debug": dict(debug_payload or {}),
        }
        return self._clone_record_with_metadata(record, metadata=updated_meta)

    @staticmethod
    def _clone_record_with_metadata(record: MemoryRecord, *, metadata: dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=record.id,
            text=record.text,
            memory_type=record.memory_type,
            level=record.level,
            scope=record.scope,
            namespace=record.namespace,
            metadata=dict(metadata or {}),
            embedding=list(record.embedding or []) if isinstance(record.embedding, list) else None,
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

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _meta_float(self, row: dict[str, Any], key: str, fallback: float) -> float:
        try:
            if key in row:
                return self._clamp01(float(row.get(key)))
        except Exception:
            pass
        return self._clamp01(fallback)

    @staticmethod
    def _event_log_context(*, metadata: dict[str, Any], namespace: str) -> dict[str, Any]:
        row = dict(metadata or {})
        return {
            "trace_id": str(row.get("trace_id") or "").strip(),
            "request_id": str(row.get("request_id") or "").strip(),
            "turn_id": str(row.get("turn_id") or "").strip(),
            "conversation_id": str(row.get("conversation_id") or namespace or "").strip(),
        }

    def _record_ingest_skip(
        self,
        *,
        event_id: str,
        now_ts: float,
        event: MemoryEvent,
        namespace: str,
        metadata: dict[str, Any],
        record_id: str,
        reason: str,
        decision: AssistantWriteDecision,
    ) -> None:
        self._event_store.append(
            {
                "event_id": event_id,
                "ts": now_ts,
                "type": "memory_ingest_skipped_v2",
                "trace_id": str(metadata.get("trace_id") or ""),
                "model": str(metadata.get("model") or ""),
                "latency_ms": float(metadata.get("latency_ms") or 0.0),
                "payload": {
                    "role": str(event.role or ""),
                    "source_kind": str(metadata.get("source_kind") or ""),
                    "scope": str(event.scope.value),
                    "effective_scope": str((decision.target_scope or event.scope).value),
                    "memory_type": str(event.memory_type.value),
                    "record_id": str(record_id or ""),
                    "namespace": str(namespace or "default"),
                    "reason": str(reason or ""),
                    "assistant_write_policy": decision.to_dict(),
                    "request_id": str(metadata.get("request_id") or ""),
                    "turn_id": str(metadata.get("turn_id") or ""),
                    "conversation_id": str(metadata.get("conversation_id") or namespace),
                },
                "tags": [str(event.scope.value), str(event.memory_type.value), "skipped"],
            }
        )
        log_json(
            LOGGER,
            "memory_v2_ingest_skipped",
            summary=(
                f"type={event.memory_type.value} scope={event.scope.value} "
                f"reason={str(reason or 'skipped')} "
                f"factual_mode={str(dict(decision.signals or {}).get('factual_mode') or '-')} "
                f"confidence={float(dict(decision.signals or {}).get('final_factual_confidence') or 0.0):.3f}"
            ),
            context=self._event_log_context(metadata=metadata, namespace=namespace),
            namespace=namespace,
            scope=str(event.scope.value),
            memory_type=str(event.memory_type.value),
            source_kind=str(metadata.get("source_kind") or ""),
            reason=str(reason or ""),
            factual_mode=str(dict(decision.signals or {}).get("factual_mode") or ""),
            final_factual_confidence=float(dict(decision.signals or {}).get("final_factual_confidence") or 0.0),
            conflict_severity=float(dict(decision.signals or {}).get("conflict_severity") or 0.0),
            assistant_write_policy=decision.to_dict(),
        )

    @staticmethod
    def _without_promotion(
        lifecycle_decision: LifecycleDecision,
        *,
        reason: str,
        scope: MemoryScope,
        decision: AssistantWriteDecision,
    ) -> LifecycleDecision:
        debug_payload = dict(lifecycle_decision.decision_debug or {})
        debug_payload["promotion_disabled_by_write_policy"] = True
        debug_payload["assistant_write_policy"] = decision.to_dict()
        debug_payload["effective_scope"] = str(scope.value)
        return LifecycleDecision(
            promote_to=None,
            mark_status=lifecycle_decision.mark_status,
            archive=bool(lifecycle_decision.archive),
            reason=str(reason or lifecycle_decision.reason or ""),
            route="assistant_write_policy",
            next_version=lifecycle_decision.next_version,
            chain_parent_id=lifecycle_decision.chain_parent_id,
            decision_debug=debug_payload,
        )

    def _importance_score(self, *, text: str, metadata: dict[str, Any], namespace: str) -> float:
        explicit = metadata.get("importance")
        if explicit is not None:
            try:
                value = float(explicit)
                return max(0.0, min(1.0, value))
            except Exception:
                pass

        src = str(text or "").lower()
        score = float(self._importance_weights.get("base", 0.42))
        if any(token in src for token in ("decide", "decision", "решили", "фикс", "issue", "error", "bug")):
            score += float(self._importance_weights.get("decision", 0.24))
        if any(token in src for token in ("remember", "важно", "save", "запомни")):
            score += float(self._importance_weights.get("remember", 0.18))
        if any(token in src for token in ("project", "архитект", "design", "release")):
            score += float(self._importance_weights.get("project", 0.10))
        legacy_score = max(0.0, min(1.0, score))
        recent = [
            str(row.text or "")
            for row in list(self._working_records or [])
            if str(row.namespace or "default") == str(namespace or "default")
        ][:80]
        salience = build_salience_score(
            text=str(text or ""),
            metadata=dict(metadata or {}),
            recent_texts=recent,
            weights=self._salience_weights,
        )
        signal_score = self._meta_float(metadata, "promotion_signal_score", 0.0)
        stable_fact_signal = self._meta_float(metadata, "promotion_stable_fact_signal", 0.0)
        fact_signal = self._meta_float(metadata, "promotion_fact_signal", 0.0)
        fact_relation_diversity = self._meta_float(metadata, "promotion_fact_relation_diversity", 0.0)
        project_signal = self._meta_float(metadata, "promotion_project_signal", 0.0)
        task_signal = self._meta_float(metadata, "promotion_task_signal", 0.0)
        decision_signal = self._meta_float(metadata, "promotion_decision_signal", 0.0)
        preference_signal = self._meta_float(metadata, "promotion_preference_signal", 0.0)
        issue_signal = self._meta_float(metadata, "promotion_issue_signal", 0.0)
        smalltalk_signal = self._meta_float(metadata, "promotion_smalltalk_signal", 0.0)
        blended = (
            (0.52 * legacy_score)
            + (0.40 * float(salience))
            + (0.20 * signal_score)
            + (0.08 * max(project_signal, task_signal))
            + (0.06 * max(decision_signal, issue_signal, preference_signal))
            + (0.08 * stable_fact_signal)
            + (0.07 * fact_signal)
            + (0.04 * fact_relation_diversity)
            - (0.18 * smalltalk_signal)
        )
        return max(0.0, min(1.0, blended))

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

