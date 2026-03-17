from __future__ import annotations

import json
import re
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
    RetrievalCandidate,
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
    build_score_breakdown,
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

_FACT_EXPECTATION_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...], str], ...] = (
    (
        re.compile(
            r"(?:"
            r"какая|какой|напомни|помнишь|помниш|помни|скажи|подскажи|"
            r"remember|what(?:'s| is)|what do i have|"
            r"что\s+у\s+меня\s+за|"
            r"моя|мою|мой"
            r")[^?.!\n]{0,64}(?:"
            r"видюх|видеокарт|видеокарта|gpu|graphics card|video card|"
            r"карточк|карта"
            r")",
            re.I,
        ),
        ("environment_gpu_model",),
        "gpu_model",
    ),
    (
        re.compile(
            r"(?:"
            r"какая|какой|напомни|помнишь|помниш|помни|скажи|подскажи|"
            r"remember|what(?:'s| is)|what do i have|"
            r"что\s+у\s+меня\s+за|"
            r"моя|мою|мой"
            r")[^?.!\n]{0,64}(?:"
            r"python|питон|пайтон|версия python|версия питона|версия пайтона"
            r")",
            re.I,
        ),
        ("environment_runtime_python",),
        "python_version",
    ),
    (
        re.compile(
            r"(?:"
            r"на\s+ч[её]м\s+я\s+(?:сейчас\s+)?сижу|"
            r"какая\s+у\s+меня\s+ос|"
            r"какая\s+у\s+меня\s+операционк|"
            r"что\s+у\s+меня\s+за\s+(?:ос|операционк|винд|систем)|"
            r"моя\s+(?:ос|операционк|винда|система)|"
            r"помнишь[^?.!\n]{0,64}(?:ос|операционк|винд|систем)|"
            r"помниш[^?.!\n]{0,64}(?:ос|операционк|винд|систем)|"
            r"версия\s+винды|"
            r"what\s+os|which\s+os"
            r")",
            re.I,
        ),
        ("environment_os",),
        "operating_system",
    ),
    (
        re.compile(
            r"(?:"
            r"сколько|какая|какой|напомни|помнишь|помниш|помни|скажи|подскажи|"
            r"remember|what(?:'s| is)|what do i have|"
            r"что\s+у\s+меня\s+с|"
            r"моя|мою|мой"
            r")[^?.!\n]{0,64}(?:"
            r"озу|ram|оператив|оперативк|memory"
            r")",
            re.I,
        ),
        ("environment_ram_gb", "environment_memory_gb"),
        "ram_amount",
    ),
)
_CONTEXTUAL_RECALL_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:о чем|о ч[её]м|what did we|what were we|what did we discuss)", re.I),
    re.compile(r"(?:что мы обсуждали|что обсуждали|что решили|what we decided|what did we decide)", re.I),
    re.compile(r"(?:почему так сделали|why did we do that|why we did that)", re.I),
    re.compile(r"(?:что ты советовала|what did you suggest|what did you advise)", re.I),
)


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
            source_preview_facts: list[FactRecordV2] = []
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
                    source_preview_facts = [
                        fact
                        for fact in list(preview_facts or [])
                        if self._policy.is_fact_allowed_for_source(
                            predicate=str(fact.predicate or ""),
                            source_role=str(event.role or "user"),
                            source_kind=source_kind,
                        )
                    ]
                metadata = self._augment_message_metadata(
                    text=text,
                    metadata=metadata,
                    namespace=namespace,
                    preview_facts=(source_preview_facts or preview_facts),
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
                decision_signals = dict(assistant_write_decision.signals or {})
                assistant_reply_kind = str(
                    decision_signals.get("assistant_reply_kind")
                    or metadata.get("assistant_reply_kind")
                    or ""
                ).strip()
                if assistant_reply_kind:
                    metadata["assistant_reply_kind"] = assistant_reply_kind
                if decision_signals.get("memory_help_noise") is not None:
                    metadata["assistant_memory_help_noise"] = bool(decision_signals.get("memory_help_noise"))
                if assistant_write_decision.target_scope is not None:
                    scope = assistant_write_decision.target_scope
                if scope == MemoryScope.TEMPORARY:
                    metadata.setdefault("ttl_sec", int(self._temporary_ttl_sec))
                if (
                    not assistant_write_decision.allow_fact_records
                    or not source_fact_records_allowed
                    or (preview_facts and not source_preview_facts)
                ):
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
                list(source_preview_facts or preview_facts)
                if assistant_write_decision.allow_fact_records and source_fact_records_allowed
                else []
            )
            if scope != MemoryScope.PRIVATE_RUNTIME and writeable_preview_facts:
                extracted_facts = self._write_fact_records(
                    facts=writeable_preview_facts,
                    namespace=namespace,
                    now_ts=now_ts,
                    event_id=event_id,
                    source_role=str(event.role or "user"),
                    source_kind=str(metadata.get("source_kind") or "structured_fact"),
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
        recall_mode = self._classify_memory_recall_mode(query_text=str(request.user_message or ""))
        fact_expectation = self._build_fact_expectation_check(
            query_text=str(request.user_message or ""),
            namespace=str(request.namespace or "default"),
        )
        exact_fact_candidates = self._exact_fact_candidates_for_expectation(
            query=retrieval_result.query,
            namespace=str(request.namespace or "default"),
            fact_expectation=fact_expectation,
        )
        prioritized_candidates = self._prioritize_retrieval_candidates_for_fact_expectation(
            query=retrieval_result.query,
            candidates=list(retrieval_result.candidates or []),
            exact_fact_candidates=exact_fact_candidates,
            fact_expectation=fact_expectation,
        )
        selected_candidates = self._select_candidates_for_recall_mode(
            query_text=str(request.user_message or ""),
            recall_mode=recall_mode,
            fallback_candidates=prioritized_candidates,
            exact_fact_candidates=exact_fact_candidates,
            fact_expectation=fact_expectation,
        )

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
            retrieved=selected_candidates,
            private_runtime_state=private_runtime,
        )
        self_facts_context = self._build_self_facts_context(
            query_text=str(request.user_message or ""),
            selected_candidates=list(result.selected or []),
            fact_expectation=fact_expectation,
        )
        if fact_expectation or self_facts_context:
            blocks = dict(result.blocks or {})
            blocks["fact_expectation_check"] = self._render_fact_expectation_check(fact_expectation)
            self_facts_block = self._render_self_facts(self_facts_context)
            if self_facts_block:
                blocks["self_facts"] = self_facts_block
            recall_mode_block = self._render_memory_recall_mode(recall_mode)
            if recall_mode_block:
                blocks["memory_recall_mode"] = recall_mode_block
            result = ContextBuildResult(
                blocks=blocks,
                selected=list(result.selected or []),
                dropped=[dict(x) for x in list(result.dropped or [])],
                score_breakdowns=[dict(x) for x in list(result.score_breakdowns or [])],
                truncation_log=[dict(x) for x in list(result.truncation_log or [])],
                fact_expectation=fact_expectation,
                self_facts_context=self_facts_context,
                recall_mode=recall_mode,
            )
        elif recall_mode:
            blocks = dict(result.blocks or {})
            recall_mode_block = self._render_memory_recall_mode(recall_mode)
            if recall_mode_block:
                blocks["memory_recall_mode"] = recall_mode_block
            result = ContextBuildResult(
                blocks=blocks,
                selected=list(result.selected or []),
                dropped=[dict(x) for x in list(result.dropped or [])],
                score_breakdowns=[dict(x) for x in list(result.score_breakdowns or [])],
                truncation_log=[dict(x) for x in list(result.truncation_log or [])],
                fact_expectation=fact_expectation,
                self_facts_context=self_facts_context,
                recall_mode=recall_mode,
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

    def _build_fact_expectation_check(self, *, query_text: str, namespace: str) -> dict[str, Any]:
        query = str(query_text or "").strip()
        if not query:
            return {}
        expected_predicates: list[str] = []
        intents: list[str] = []
        for pattern, predicates, intent_name in _FACT_EXPECTATION_RULES:
            if not pattern.search(query):
                continue
            intents.append(str(intent_name))
            for predicate in list(predicates or []):
                token = str(predicate or "").strip().lower()
                if token and token not in expected_predicates:
                    expected_predicates.append(token)
        if not expected_predicates and self._classify_memory_recall_mode(query_text=query) == "exact_fact_recall":
            fallback_predicates = self._relevant_self_fact_predicates(
                query_text=query,
                fact_expectation={},
            )
            for predicate in list(fallback_predicates or []):
                token = str(predicate or "").strip().lower()
                if token and token not in expected_predicates:
                    expected_predicates.append(token)
            if expected_predicates and "self_fact_fallback" not in intents:
                intents.append("self_fact_fallback")
        if not expected_predicates:
            return {}

        fact_rows = self._find_active_semantic_fact_rows(namespace=namespace, predicates=expected_predicates)
        found_by_predicate: dict[str, list[dict[str, Any]]] = {predicate: [] for predicate in expected_predicates}
        for row in list(fact_rows or []):
            fact = dict(dict(row.metadata or {}).get("fact") or {})
            predicate = str(fact.get("predicate") or "").strip().lower()
            if predicate not in found_by_predicate:
                continue
            found_by_predicate[predicate].append(
                {
                    "record_id": str(row.id or ""),
                    "value": fact.get("value"),
                    "confidence": float(fact.get("confidence") or row.confidence or 0.0),
                    "scope": str(row.scope.value),
                    "source_event_id": str(row.source_event_id or ""),
                }
            )

        found_predicates = [predicate for predicate in expected_predicates if found_by_predicate.get(predicate)]
        missing_predicates = [predicate for predicate in expected_predicates if not found_by_predicate.get(predicate)]
        return {
            "query": query,
            "intents": intents,
            "expected_predicates": expected_predicates,
            "found_predicates": found_predicates,
            "missing_predicates": missing_predicates,
            "found_facts": found_by_predicate,
            "exact_fact_required": True,
            "should_answer_cautiously": bool(missing_predicates),
        }

    def _exact_fact_candidates_for_expectation(
        self,
        *,
        query: RetrievalQuery,
        namespace: str,
        fact_expectation: dict[str, Any] | None,
    ) -> list[RetrievalCandidate]:
        row = dict(fact_expectation or {})
        expected_predicates = [
            str(x).strip().lower()
            for x in list(row.get("expected_predicates") or [])
            if str(x).strip()
        ]
        if not expected_predicates:
            return []

        exact_fact_rows = self._find_active_semantic_fact_rows(namespace=namespace, predicates=expected_predicates)
        exact_fact_candidates: list[RetrievalCandidate] = []
        for record in list(exact_fact_rows or []):
            breakdown = build_score_breakdown(
                query=query,
                record=record,
                semantic_similarity=1.0,
                lexical_score=1.0,
            )
            exact_fact_candidates.append(
                RetrievalCandidate(
                    record=record,
                    score_breakdown=breakdown,
                    source="fact_expectation_exact",
                )
            )
        return exact_fact_candidates

    def _prioritize_retrieval_candidates_for_fact_expectation(
        self,
        *,
        query: RetrievalQuery,
        candidates: list[RetrievalCandidate],
        exact_fact_candidates: list[RetrievalCandidate],
        fact_expectation: dict[str, Any] | None,
    ) -> list[RetrievalCandidate]:
        row = dict(fact_expectation or {})
        expected_predicates = [
            str(x).strip().lower()
            for x in list(row.get("expected_predicates") or [])
            if str(x).strip()
        ]
        if not expected_predicates:
            return list(candidates or [])

        existing_ids = {str(item.record.id or "") for item in list(exact_fact_candidates or [])}
        fallback_candidates = []
        for item in list(candidates or []):
            record_id = str(item.record.id or "")
            if record_id in existing_ids:
                continue
            fallback_candidates.append(item)
        all_candidates = list(exact_fact_candidates or []) + fallback_candidates

        def _priority(item: RetrievalCandidate) -> tuple[float, float, float, float]:
            record = item.record
            metadata = dict(record.metadata or {})
            fact = dict(metadata.get("fact") or {})
            predicate = str(fact.get("predicate") or "").strip().lower()
            is_expected_fact = 1.0 if record.memory_type == MemoryType.FACT and predicate in expected_predicates else 0.0
            is_any_fact = 1.0 if record.memory_type == MemoryType.FACT else 0.0
            episodic_bias = 1.0 if record.level in {MemoryLevel.L2_EPISODIC, MemoryLevel.L3_SEMANTIC} else 0.0
            working_penalty = 0.0 if record.level == MemoryLevel.L0_WORKING else 1.0
            return (
                is_expected_fact,
                is_any_fact,
                episodic_bias + working_penalty,
                float(item.final_score),
            )

        ordered = sorted(list(all_candidates), key=_priority, reverse=True)
        top_k = max(1, int(query.top_k or getattr(self, "_retrieval_top_k", 8) or 8))
        return ordered[: max(top_k, len(exact_fact_candidates))]

    @staticmethod
    def _select_candidates_for_self_fact_recall(
        *,
        fallback_candidates: list[RetrievalCandidate],
        exact_fact_candidates: list[RetrievalCandidate],
        fact_expectation: dict[str, Any] | None,
    ) -> list[RetrievalCandidate]:
        row = dict(fact_expectation or {})
        found_predicates = [str(x).strip() for x in list(row.get("found_predicates") or []) if str(x).strip()]
        if found_predicates:
            return list(exact_fact_candidates or [])
        expected_predicates = {
            str(x).strip().lower()
            for x in list(row.get("expected_predicates") or [])
            if str(x).strip()
        }
        if not expected_predicates:
            return list(fallback_candidates or [])
        filtered: list[RetrievalCandidate] = []
        for item in list(fallback_candidates or []):
            record = item.record
            if record.memory_type != MemoryType.FACT:
                filtered.append(item)
                continue
            fact = dict(dict(record.metadata or {}).get("fact") or {})
            predicate = str(fact.get("predicate") or "").strip().lower()
            if record.level == MemoryLevel.L3_SEMANTIC and predicate in expected_predicates:
                filtered.append(item)
        return filtered

    def _select_candidates_for_recall_mode(
        self,
        *,
        query_text: str,
        recall_mode: str,
        fallback_candidates: list[RetrievalCandidate],
        exact_fact_candidates: list[RetrievalCandidate],
        fact_expectation: dict[str, Any] | None,
    ) -> list[RetrievalCandidate]:
        mode = str(recall_mode or "").strip().lower()
        if mode == "exact_fact_recall":
            return self._select_candidates_for_self_fact_recall(
                fallback_candidates=fallback_candidates,
                exact_fact_candidates=exact_fact_candidates,
                fact_expectation=fact_expectation,
            )
        if mode == "contextual_recall":
            return self._prioritize_contextual_recall_candidates(
                query_text=query_text,
                candidates=fallback_candidates,
            )
        return list(fallback_candidates or [])

    @staticmethod
    def _prioritize_contextual_recall_candidates(
        *,
        query_text: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        _ = str(query_text or "")

        def _priority(item: RetrievalCandidate) -> tuple[float, float, float]:
            record = item.record
            meta = dict(record.metadata or {})
            fact = dict(meta.get("fact") or {})
            predicate = str(fact.get("predicate") or "").strip().lower()
            is_message = 1.0 if record.memory_type == MemoryType.MESSAGE else 0.0
            is_summary = 1.0 if record.memory_type == MemoryType.SUMMARY else 0.0
            is_context_fact = 1.0 if record.memory_type == MemoryType.FACT and predicate in {"decision", "task", "task_goal", "agreed_plan"} else 0.0
            is_other_fact = 1.0 if record.memory_type == MemoryType.FACT else 0.0
            docs_penalty = 0.0 if record.level == MemoryLevel.L4_DOCUMENT else 1.0
            return (
                (is_message * 5.0) + (is_summary * 4.0) + (is_context_fact * 3.0) + (docs_penalty * 0.2) - (is_other_fact * 0.5),
                float(item.final_score),
                float(record.updated_at or 0.0),
            )

        ordered = sorted(list(candidates or []), key=_priority, reverse=True)
        return ordered

    def _find_active_semantic_fact_rows(self, *, namespace: str, predicates: list[str]) -> list[MemoryRecord]:
        expected = {str(x or "").strip().lower() for x in list(predicates or []) if str(x or "").strip()}
        if not expected:
            return []
        rows: list[MemoryRecord] = []
        for row in list(self._store.iter_records(namespace=namespace)):
            if row.memory_type != MemoryType.FACT or row.status != MemoryStatus.ACTIVE:
                continue
            metadata = dict(row.metadata or {})
            fact = dict(metadata.get("fact") or {})
            predicate = str(fact.get("predicate") or "").strip().lower()
            subject = str(fact.get("subject") or "").strip().lower()
            if predicate not in expected or subject != "user":
                continue
            rows.append(row)
        rows.sort(
            key=lambda item: (
                float(dict(dict(item.metadata or {}).get("fact") or {}).get("confidence") or item.confidence or 0.0),
                float(item.updated_at or 0.0),
                float(item.created_at or 0.0),
            ),
            reverse=True,
        )
        return rows

    @staticmethod
    def _render_fact_expectation_check(row: dict[str, Any]) -> str:
        if not row:
            return ""
        expected = [str(x) for x in list(row.get("expected_predicates") or []) if str(x).strip()]
        missing = [str(x) for x in list(row.get("missing_predicates") or []) if str(x).strip()]
        found = dict(row.get("found_facts") or {})
        lines = [
            "- exact_fact_required: true",
            f"- expected_predicates: {', '.join(expected) if expected else 'none'}",
        ]
        if missing:
            lines.append(f"- missing_predicates: {', '.join(missing)}")
            lines.append("- response_rule: if an expected semantic fact is missing, say you do not see the exact fact in memory.")
            lines.append("- response_rule: do not infer an exact model/version/os from similar message records alone.")
        for predicate in expected:
            values = [str(dict(item).get('value') or '').strip() for item in list(found.get(predicate) or []) if str(dict(item).get("value") or "").strip()]
            if values:
                lines.append(f"- {predicate}: {', '.join(values)}")
        return "\n".join(lines).strip()

    @staticmethod
    def _render_self_facts(row: dict[str, Any]) -> str:
        found = dict(row.get("found_facts") or {})
        found_predicates = [str(x).strip() for x in list(row.get("found_predicates") or []) if str(x).strip()]
        if not found_predicates:
            return ""
        lines = [
            "- trust_level: exact_active_self_facts",
            "- response_rule: treat these as confirmed self facts for this turn.",
            "- response_rule: answer directly from these facts.",
            "- response_rule: do not suggest ways to check manually.",
            "- response_rule: do not say you cannot see it in memory.",
            "- response_rule: ignore weaker ordinary memory snippets if they conflict.",
        ]
        for predicate in found_predicates:
            items = list(found.get(predicate) or [])
            values = [str(dict(item).get("value") or "").strip() for item in items if str(dict(item).get("value") or "").strip()]
            if not values:
                continue
            lines.append(f"- {predicate}: {', '.join(values)}")
        return "\n".join(lines).strip()

    def _build_self_facts_context(
        self,
        *,
        query_text: str,
        selected_candidates: list[RetrievalCandidate],
        fact_expectation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        expectation = dict(fact_expectation or {})
        query = str(query_text or "").strip()
        relevant_predicates = set(self._relevant_self_fact_predicates(query_text=query, fact_expectation=expectation))
        if not relevant_predicates:
            return expectation if expectation.get("found_predicates") else {}
        found_facts: dict[str, list[dict[str, Any]]] = {}
        if expectation.get("found_facts"):
            for predicate, items in dict(expectation.get("found_facts") or {}).items():
                token = str(predicate or "").strip().lower()
                if relevant_predicates and token not in relevant_predicates:
                    continue
                rows = [dict(x) for x in list(items or []) if isinstance(x, dict)]
                if rows:
                    found_facts[token] = rows

        for candidate in list(selected_candidates or []):
            record = candidate.record
            if record.memory_type != MemoryType.FACT or record.status != MemoryStatus.ACTIVE:
                continue
            if record.level != MemoryLevel.L3_SEMANTIC:
                continue
            fact = dict(dict(record.metadata or {}).get("fact") or {})
            predicate = str(fact.get("predicate") or "").strip().lower()
            subject = str(fact.get("subject") or "").strip().lower()
            value = fact.get("value")
            if not predicate or subject != "user" or value in {"", None}:
                continue
            if relevant_predicates and predicate not in relevant_predicates:
                continue
            if float(candidate.final_score) < 0.58 and float(record.confidence or 0.0) < 0.72:
                continue
            payload = {
                "record_id": str(record.id or ""),
                "value": value,
                "confidence": float(fact.get("confidence") or record.confidence or 0.0),
                "scope": str(record.scope.value),
                "source_event_id": str(record.source_event_id or ""),
                "retrieval_score": float(candidate.final_score),
                "source": str(candidate.source or ""),
            }
            bucket = found_facts.setdefault(predicate, [])
            key = (str(payload["record_id"]), str(payload["value"]))
            existing_keys = {(str(dict(item).get("record_id") or ""), str(dict(item).get("value") or "")) for item in bucket}
            if key in existing_keys:
                continue
            bucket.append(payload)

        found_predicates = [predicate for predicate, items in found_facts.items() if list(items or [])]
        if not found_predicates:
            return expectation if expectation.get("found_predicates") else {}
        return {
            "query": query,
            "intents": list(expectation.get("intents") or []),
            "expected_predicates": [
                predicate
                for predicate in list(expectation.get("expected_predicates") or [])
                if not relevant_predicates or str(predicate or "").strip().lower() in relevant_predicates
            ],
            "found_predicates": found_predicates,
            "missing_predicates": [
                predicate
                for predicate in list(expectation.get("missing_predicates") or [])
                if predicate not in set(found_predicates)
                and (not relevant_predicates or str(predicate or "").strip().lower() in relevant_predicates)
            ],
            "found_facts": found_facts,
            "exact_fact_required": bool(expectation.get("exact_fact_required")),
            "should_answer_cautiously": bool(
                expectation.get("should_answer_cautiously")
                and not found_predicates
            ),
            "self_recall_like_query": bool(self._is_self_recall_like_query(query)),
        }

    @staticmethod
    def _classify_memory_recall_mode(*, query_text: str) -> str:
        text = str(query_text or "").strip()
        if not text:
            return ""
        if MemoryManager._is_self_recall_like_query(text):
            return "exact_fact_recall"
        low = text.lower()
        if any(pattern.search(low) for pattern in _CONTEXTUAL_RECALL_RULES):
            return "contextual_recall"
        return ""

    @staticmethod
    def _render_memory_recall_mode(recall_mode: str) -> str:
        mode = str(recall_mode or "").strip().lower()
        if mode == "exact_fact_recall":
            return "\n".join(
                [
                    "- mode: exact_fact_recall",
                    "- response_rule: prioritize exact active user facts over conversational snippets.",
                    "- response_rule: if the exact fact is missing, answer honestly that the exact fact is not visible in memory.",
                ]
            )
        if mode == "contextual_recall":
            return "\n".join(
                [
                    "- mode: contextual_recall",
                    "- response_rule: prioritize messages, summaries, and decision/task facts.",
                    "- response_rule: answer from remembered discussion context, not as an exact self-fact lookup.",
                ]
            )
        return ""

    @staticmethod
    def _relevant_self_fact_predicates(*, query_text: str, fact_expectation: dict[str, Any] | None) -> list[str]:
        expectation = dict(fact_expectation or {})
        preferred: list[str] = []
        for item in list(expectation.get("expected_predicates") or []):
            token = str(item or "").strip().lower()
            if token and token not in preferred:
                preferred.append(token)

        text = str(query_text or "").strip().lower()
        if not text:
            return preferred

        heuristic_groups: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
            (
                ("видюх", "видеокарт", "видеокарта", "gpu", "graphics card", "video card", "карточк", "карта"),
                ("environment_gpu_model", "environment_gpu_vram_gb"),
            ),
            (
                ("python", "питон", "пайтон", "версия python", "версия питона", "версия пайтона"),
                ("environment_runtime_python",),
            ),
            (
                ("ос", "операционк", "os", "operating system", "windows", "linux", "ubuntu", "macos", "винда", "винды", "система"),
                ("environment_os",),
            ),
            (
                ("озу", "ram", "оператив", "оперативк", "оперативы", "оперативка", "memory"),
                ("environment_ram_gb", "environment_memory_gb"),
            ),
            (
                ("имя", "name", "зовут"),
                ("identity_name",),
            ),
            (
                ("возраст", "age", "лет"),
                ("identity_age_years",),
            ),
        )
        for markers, predicates in heuristic_groups:
            if not any(marker in text for marker in markers):
                continue
            for predicate in predicates:
                token = str(predicate or "").strip().lower()
                if token and token not in preferred:
                    preferred.append(token)
        return preferred

    @staticmethod
    def _is_self_recall_like_query(query_text: str) -> bool:
        text = str(query_text or "").strip().lower()
        if not text:
            return False
        recall_markers = ("какая", "какой", "какое", "подскажи", "напомни", "скажи", "remember", "remind", "what", "which")
        self_markers = ("у меня", "мой ", "мою ", "моя ", "моё ", "my ", "mine", "me ")
        if any(marker in text for marker in self_markers) and any(marker in text for marker in recall_markers):
            return True
        return any(
            marker in text
            for marker in (
                "видюх",
                "видеокарт",
                "карточк",
                "карта",
                "python",
                "питон",
                "пайтон",
                "os",
                "операционк",
                "винда",
                "винды",
                "система",
                "имя",
                "возраст",
                "озу",
                "ram",
                "оператив",
                "оперативк",
                "оперативы",
            )
        )

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
        source_role: str = "user",
        source_kind: str = "structured_fact",
    ) -> list[FactRecordV2]:
        out: list[FactRecordV2] = []
        for fact in list(facts or []):
            canonical = str(fact.canonical_key or f"{fact.subject}.{fact.predicate}")
            candidates = self._find_active_fact_candidates_for_write(namespace=namespace, fact=fact)
            existing = self._pick_primary_fact_candidate(fact=fact, candidates=candidates)
            decision = self._policy.decide_fact_write(
                fact=fact,
                source_role=source_role,
                existing_record=existing,
            )

            if not decision.allow_write and str(decision.action or "").strip().lower() in {"skip", "keep_existing"}:
                continue

            fact_payload = {
                **dict(fact.to_dict() or {}),
                "source_role": str(source_role or "").strip().lower(),
                "source_kind": str(source_kind or dict(fact.metadata or {}).get("source_kind") or "structured_fact"),
            }
            record = MemoryRecord(
                id=f"fact:{uuid.uuid4().hex[:18]}",
                text=f"{fact.subject}.{fact.predicate}={fact.value}",
                memory_type=MemoryType.FACT,
                level=MemoryLevel.L3_SEMANTIC,
                scope=fact.scope,
                namespace=namespace,
                metadata={
                    "fact": fact_payload,
                    "canonical_key": canonical,
                    "relation": fact.relation,
                    "write_policy": decision.to_dict(),
                },
                importance=float(fact.importance),
                confidence=float(fact.confidence),
                created_at=now_ts,
                updated_at=now_ts,
                expires_at=fact.valid_to,
                status=fact.status,
                source_event_id=event_id,
            )

            if candidates and decision.allow_supersede:
                for previous in list(candidates):
                    superseded = self._status_transition(
                        previous,
                        target=MemoryStatus.SUPERSEDED,
                        now_ts=now_ts,
                        reason=str(decision.reason or "write_policy_supersede"),
                    )
                    self._store.upsert(superseded)
            if existing is not None and str(decision.action or "").strip().lower() == "parallel":
                record = MemoryRecord(
                    id=record.id,
                    text=record.text,
                    memory_type=record.memory_type,
                    level=record.level,
                    scope=record.scope,
                    namespace=record.namespace,
                    metadata={
                        **dict(record.metadata or {}),
                        "parallel_with": existing.id,
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

            self._store.upsert(record)
            out.append(fact)
        return out

    def _find_active_fact_candidates_for_write(self, *, namespace: str, fact: FactRecordV2) -> list[MemoryRecord]:
        rows = self._store.iter_records(namespace=namespace)
        canonical = str(fact.canonical_key or f"{fact.subject}.{fact.predicate}").strip().lower()
        subject = str(fact.subject or "").strip().lower()
        group = str(self._policy.fact_group(str(fact.predicate or "")) or "").strip().lower()
        exact: list[MemoryRecord] = []
        grouped: list[MemoryRecord] = []

        for row in rows:
            if row.memory_type != MemoryType.FACT or row.status != MemoryStatus.ACTIVE:
                continue
            metadata = dict(row.metadata or {})
            existing_canonical = str(metadata.get("canonical_key") or "").strip().lower()
            if canonical and existing_canonical == canonical:
                exact.append(row)
                continue
            if not self._policy.is_singleton_group(group):
                continue
            old_fact = dict(metadata.get("fact") or {})
            old_subject = str(old_fact.get("subject") or "").strip().lower()
            old_predicate = str(old_fact.get("predicate") or "").strip().lower()
            old_group = str(self._policy.fact_group(old_predicate) or "").strip().lower()
            if subject and old_subject == subject and old_group == group:
                grouped.append(row)

        return exact or sorted(
            grouped,
            key=lambda row: (
                float(dict(dict(row.metadata or {}).get("fact") or {}).get("confidence") or row.confidence or 0.0),
                float(row.updated_at or 0.0),
            ),
            reverse=True,
        )

    @staticmethod
    def _pick_primary_fact_candidate(*, fact: FactRecordV2, candidates: list[MemoryRecord]) -> MemoryRecord | None:
        _ = fact
        return candidates[0] if candidates else None

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
        return MemoryPolicy.is_singleton_predicate(predicate) or predicate in {"preference", "environment"}

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
        out["memory_tags"] = _merge_tags(out.get("memory_tags"), list(analysis.tags or []))
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

