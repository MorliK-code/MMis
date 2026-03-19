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
from memory.claim_promoter import promote_claim_candidates
from memory.claim_models import ClaimPromotionDecision, ClaimRecord
from memory.dialog_episode_retriever import DialogEpisodeHit, DialogEpisodeRetriever
from memory.document_memory import ChunkingConfig, DocumentMemory
from memory.document_retrieval import DocumentRetrievalHit, DocumentRetriever
from memory.embedding_provider import build_embedding_provider
from memory.event_store import EventStore
from memory.fact_extractor import FactExtractor
from memory.governor import GovernorProfileSnapshot, MemoryGovernor
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
from memory.recall_policy import classify_query_recall_profile
from memory.storage_profile import (
    DEBUG_ONLY_METADATA_FIELDS,
    DEFAULT_STORAGE_PROFILE,
    normalize_storage_profile,
    sanitize_storage_metadata,
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
_DOCUMENT_RECALL_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:что было в главе|what was in chapter|chapter\s+\d+|глава\s+\d+)", re.I),
    re.compile(r"(?:где .*вызывается|где .*объявляется|where .*called|where .*declared)", re.I),
    re.compile(r"(?:где в коде|where in (?:the )?code|класс|class|function|method|код|файл|документ|chapter|глава)", re.I),
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

        # Storage profile: "compact" (default) or "debug"
        # Can be set via config or environment variable MEMORY_STORAGE_PROFILE
        self._storage_profile = normalize_storage_profile(
            str(getattr(self._cfg, "memory_storage_profile", DEFAULT_STORAGE_PROFILE) or DEFAULT_STORAGE_PROFILE)
        )
        self._DEBUG_METADATA_FIELDS = set(DEBUG_ONLY_METADATA_FIELDS)

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
        self._governor = MemoryGovernor(
            parallel_margin=float(getattr(self._lifecycle, "parallel_margin", 0.03) or 0.03),
            lifecycle=self._lifecycle,
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
        self._dialog_episode_retriever = DialogEpisodeRetriever(store=self._store)
        self._document_retriever = DocumentRetriever(store=self._store)
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
        self._governor_profile_snapshots: dict[str, GovernorProfileSnapshot] = {}

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
                storage_profile=self._storage_profile,
                compact_for_storage=False,
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
                # Do not store assistant_write_policy - it's runtime-only
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
            preview_claims: list[ClaimRecord] = []
            ingest_analysis: IngestAnalysis | None = None
            if memory_type in {MemoryType.MESSAGE, MemoryType.SUMMARY} and scope != MemoryScope.PRIVATE_RUNTIME:
                # Long-term memory extraction must be text-first and memory-layer-owned.
                # Do not feed runtime/dialog metadata from metadata/* into the memory truth path.
                memory_analysis_metadata = {
                    "event_id": event_id,
                    "namespace": namespace,
                }
                ingest_analysis = analyze_message_for_memory(
                    text,
                    metadata=memory_analysis_metadata,
                )
                metadata = self._merge_ingest_analysis_into_metadata(metadata=metadata, analysis=ingest_analysis)
                source_claim_records_allowed = self._policy.allow_claim_records_for_source(source_kind=source_kind)
                if source_claim_records_allowed:
                    preview_claims = promote_claim_candidates(
                        list(ingest_analysis.claim_candidates or []),
                        event_id=event_id,
                        namespace=namespace,
                        scope=str(scope.value),
                    )
                if source_fact_records_allowed:
                    preview_facts = self._fact_extractor.extract_v2(
                        text=text,
                        metadata=memory_analysis_metadata,
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
                # Assistant replies should be explicitly marked as fact-record blocked
                # even when no preview facts were extracted, so storage/debug state stays clear.
                metadata["assistant_fact_records_blocked"] = True
                assistant_write_decision = self._policy.decide_assistant_message_write(
                    text=text,
                    metadata=metadata,
                    memory_type=memory_type,
                    requested_scope=scope,
                )
                # Do not store assistant_write_policy - it's runtime-only
                # Only extract and store essential signals
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
            extracted_claims: list[ClaimRecord] = []
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
                # Strip debug metadata before storage
                record = MemoryRecord(
                    id=record.id,
                    text=record.text,
                    memory_type=record.memory_type,
                    level=record.level,
                    scope=record.scope,
                    namespace=record.namespace,
                    metadata=self._strip_debug_metadata(record.metadata, memory_type=record.memory_type),
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
                )
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
                # Strip debug metadata from status transition record
                status_row = MemoryRecord(
                    id=status_row.id,
                    text=status_row.text,
                    memory_type=status_row.memory_type,
                    level=status_row.level,
                    scope=status_row.scope,
                    namespace=status_row.namespace,
                    metadata=self._strip_debug_metadata(status_row.metadata, memory_type=status_row.memory_type),
                    embedding=list(status_row.embedding or []) if isinstance(status_row.embedding, list) else None,
                    importance=status_row.importance,
                    confidence=status_row.confidence,
                    created_at=status_row.created_at,
                    updated_at=status_row.updated_at,
                    expires_at=status_row.expires_at,
                    status=status_row.status,
                    version=status_row.version,
                    parent_id=status_row.parent_id,
                    chunk_index=status_row.chunk_index,
                )
                self._store.upsert(status_row)
            if (
                scope != MemoryScope.PRIVATE_RUNTIME
                and lifecycle_decision.promote_to is not None
                and lifecycle_decision.promote_to != record.level
            ):
                promoted = self._promote_record(record, target=lifecycle_decision.promote_to, now_ts=now_ts)
                # Strip debug metadata from promoted record
                promoted = MemoryRecord(
                    id=promoted.id,
                    text=promoted.text,
                    memory_type=promoted.memory_type,
                    level=promoted.level,
                    scope=promoted.scope,
                    namespace=promoted.namespace,
                    metadata=self._strip_debug_metadata(promoted.metadata, memory_type=promoted.memory_type),
                    embedding=list(promoted.embedding or []) if isinstance(promoted.embedding, list) else None,
                    importance=promoted.importance,
                    confidence=promoted.confidence,
                    created_at=promoted.created_at,
                    updated_at=promoted.updated_at,
                    expires_at=promoted.expires_at,
                    status=promoted.status,
                    version=promoted.version,
                    parent_id=promoted.parent_id,
                    chunk_index=promoted.chunk_index,
                )
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
            if (
                scope != MemoryScope.PRIVATE_RUNTIME
                and str(source_kind.value) == MemorySourceKind.USER.value
                and preview_claims
            ):
                extracted_claims = self._write_claim_records(
                    claims=preview_claims,
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
                        "preview_claims_count": int(len(preview_claims)),
                        "assistant_write_action": str(assistant_write_decision.action or "allow"),
                        "assistant_write_reason": str(assistant_write_decision.reason or ""),
                        "request_id": str(metadata.get("request_id") or ""),
                        "turn_id": str(metadata.get("turn_id") or ""),
                        "conversation_id": str(metadata.get("conversation_id") or namespace),
                    },
                    "tags": [scope.value, memory_type.value],
                }
                )
            self._save_state()

        log_json(
            LOGGER,
            "memory_v2_ingest",
            summary=(
                f"type={memory_type.value} scope={scope.value} stored={len(stored_ids)} "
                f"facts={len(extracted_facts)} claims={len(extracted_claims)} promotions={len(promoted_ids)} "
                f"reason={str(lifecycle_decision.reason or '-')} "
                f"write_action={str(assistant_write_decision.action or 'allow')} "
                f"write_reason={str(assistant_write_decision.reason or '-')}"
            ),
            context=self._event_log_context(metadata=metadata, namespace=namespace),
            namespace=namespace,
            scope=scope.value,
            memory_type=memory_type.value,
            source_kind=str(source_kind.value),
            stored=len(stored_ids),
            promoted=len(promoted_ids),
            facts=len(extracted_facts),
            claims=len(extracted_claims),
            promotion_reason=str(lifecycle_decision.reason or ""),
            promotion_route=str(lifecycle_decision.route or ""),
            assistant_write_action=str(assistant_write_decision.action or "allow"),
            assistant_write_reason=str(assistant_write_decision.reason or ""),
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
        blocks = dict(result.blocks or {})
        relevant_claims_block = self._render_relevant_claims(list(result.selected or []))
        if relevant_claims_block:
            blocks["relevant_claims"] = relevant_claims_block

        dialog_hits = self._retrieve_dialog_episode_hits(
            query=retrieval_result.query,
            recall_mode=recall_mode,
        )
        recalled_dialog_block = self._render_recalled_dialog(dialog_hits)
        if recalled_dialog_block:
            blocks["recalled_dialog"] = recalled_dialog_block

        document_hits = self._retrieve_document_evidence_hits(
            query=retrieval_result.query,
            recall_mode=recall_mode,
        )
        document_evidence_block = self._render_document_evidence(document_hits)
        if document_evidence_block:
            blocks["document_evidence"] = document_evidence_block

        supporting_messages_block = self._render_supporting_messages(
            selected_candidates=list(result.selected or []),
            dialog_hits=dialog_hits,
        )
        if supporting_messages_block:
            blocks["supporting_messages"] = supporting_messages_block

        result = ContextBuildResult(
            blocks=blocks,
            selected=list(result.selected or []),
            dropped=[dict(x) for x in list(result.dropped or [])],
            score_breakdowns=[dict(x) for x in list(result.score_breakdowns or [])],
            truncation_log=[dict(x) for x in list(result.truncation_log or [])],
            fact_expectation=dict(result.fact_expectation or {}),
            self_facts_context=dict(result.self_facts_context or {}),
            recall_mode=str(result.recall_mode or ""),
        )
        self_facts_context = self._build_self_facts_context(
            query_text=str(request.user_message or ""),
            selected_candidates=list(result.selected or []),
            fact_expectation=fact_expectation,
        )
        if fact_expectation or self_facts_context:
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
        if mode == "document_recall":
            return self._prioritize_document_recall_candidates(
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
            is_episode = 1.0 if record.memory_type == MemoryType.EPISODE else 0.0
            is_message = 1.0 if record.memory_type == MemoryType.MESSAGE else 0.0
            is_summary = 1.0 if record.memory_type == MemoryType.SUMMARY else 0.0
            is_claim = 1.0 if record.memory_type == MemoryType.CLAIM else 0.0
            is_context_fact = 1.0 if record.memory_type == MemoryType.FACT and predicate in {"decision", "task", "task_goal", "agreed_plan"} else 0.0
            is_other_fact = 1.0 if record.memory_type == MemoryType.FACT else 0.0
            docs_penalty = -3.5 if record.level == MemoryLevel.L4_DOCUMENT else 0.0
            return (
                (is_episode * 6.0)
                + (is_message * 5.0)
                + (is_summary * 4.0)
                + (is_context_fact * 3.0)
                + (is_claim * 2.4)
                + docs_penalty
                - (is_other_fact * 0.5),
                float(item.final_score),
                float(record.updated_at or 0.0),
            )

        ordered = sorted(list(candidates or []), key=_priority, reverse=True)
        return ordered

    @staticmethod
    def _prioritize_document_recall_candidates(
        *,
        query_text: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        _ = str(query_text or "")

        def _priority(item: RetrievalCandidate) -> tuple[float, float, float]:
            record = item.record
            meta = dict(record.metadata or {})
            is_document = 1.0 if record.memory_type == MemoryType.DOCUMENT else 0.0
            is_chunk = 1.0 if record.memory_type == MemoryType.DOCUMENT_CHUNK else 0.0
            is_doc_summary = 1.0 if record.memory_type == MemoryType.SUMMARY and record.level == MemoryLevel.L4_DOCUMENT else 0.0
            is_doc_claim = 1.0 if record.memory_type == MemoryType.CLAIM and (record.level == MemoryLevel.L4_DOCUMENT or meta.get("document_id") or meta.get("doc_id")) else 0.0
            unrelated_penalty = -4.0 if record.level != MemoryLevel.L4_DOCUMENT and not is_doc_claim else 0.0
            return (
                (is_chunk * 6.0)
                + (is_doc_summary * 5.0)
                + (is_doc_claim * 4.5)
                + (is_document * 4.0)
                + unrelated_penalty,
                float(item.final_score),
                float(record.updated_at or 0.0),
            )

        return sorted(list(candidates or []), key=_priority, reverse=True)

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

    @staticmethod
    def _render_relevant_claims(candidates: list[RetrievalCandidate]) -> str:
        lines: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for item in list(candidates or []):
            record = item.record
            if record.memory_type != MemoryType.CLAIM or record.status != MemoryStatus.ACTIVE:
                continue
            if record.level == MemoryLevel.L4_DOCUMENT:
                continue
            claim = dict(dict(record.metadata or {}).get("claim") or {})
            subject = str(claim.get("subject") or "").strip().lower()
            predicate = str(claim.get("predicate") or "").strip().lower()
            obj = str(claim.get("object_surface") or claim.get("obj") or "").strip()
            if subject != "user" or not predicate or not obj:
                continue
            key = (subject, predicate, obj.lower())
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {subject} {predicate} {obj}")
            if len(lines) >= 4:
                break
        return "\n".join(lines).strip()

    def _retrieve_dialog_episode_hits(
        self,
        *,
        query: RetrievalQuery,
        recall_mode: str,
    ) -> list[DialogEpisodeHit]:
        if str(recall_mode or "").strip().lower() != "contextual_recall":
            return []
        hits = self._dialog_episode_retriever.retrieve(
            query=query,
            query_text=str(query.query_text or ""),
            namespace=str(query.namespace or "default"),
            scopes=list(query.scopes or self._default_retrieval_scopes()),
            top_k=2,
        )
        floor = 0.44 if str(recall_mode or "").strip().lower() == "contextual_recall" else 0.72
        return [row for row in list(hits or []) if float(row.score) >= floor]

    def _retrieve_document_evidence_hits(
        self,
        *,
        query: RetrievalQuery,
        recall_mode: str,
    ) -> list[DocumentRetrievalHit]:
        query_text = str(query.query_text or query.search_text or "").strip()
        doc_like = self._is_document_recall_like_query(query_text)
        if not doc_like and str(recall_mode or "").strip().lower() != "document_recall":
            return []
        hits = self._document_retriever.retrieve(
            query=query,
            query_text=query_text,
            namespace=str(query.namespace or "default"),
            scopes=list(query.scopes or self._default_retrieval_scopes()),
            top_k=2,
        )
        floor = 0.42 if doc_like else (0.48 if str(recall_mode or "").strip().lower() == "contextual_recall" else 0.74)
        return [row for row in list(hits or []) if float(row.score) >= floor]

    @staticmethod
    def _render_recalled_dialog(hits: list[DialogEpisodeHit]) -> str:
        if not hits:
            return ""
        top = hits[0]
        lines = []
        topic = str(top.episode.topic or "").strip()
        if topic:
            lines.append(f"Topic: {topic}")
        summary_short = str(top.summary_short or "").strip()
        if summary_short:
            lines.append(f"Summary: {summary_short}")
        summary_reasoning = str(top.summary_reasoning or "").strip()
        if summary_reasoning:
            lines.append(f"Reasoning: {summary_reasoning}")
        decisions = [str(x).strip() for x in list(top.decisions or []) if str(x).strip()]
        if decisions:
            lines.append("Decisions:")
            for item in decisions[:4]:
                lines.append(f"- {item}")
        open_questions = [str(x).strip() for x in list(top.episode.open_questions or []) if str(x).strip()]
        if open_questions:
            lines.append("Open questions:")
            for item in open_questions[:3]:
                lines.append(f"- {item}")
        return "\n".join(lines).strip()

    @staticmethod
    def _render_document_evidence(hits: list[DocumentRetrievalHit]) -> str:
        if not hits:
            return ""
        lines: list[str] = []
        for hit in list(hits or [])[:2]:
            document = hit.document_record
            title = str(dict(getattr(document, "metadata", {}) or {}).get("title") or getattr(document, "text", "") or hit.document_id).strip()
            if title:
                lines.append(f"Document: {title}")
            if hit.section_summary is not None:
                summary_text = str(hit.section_summary.text or "").strip()
                if summary_text:
                    lines.append(f"Section summary: {summary_text}")
            for chunk in list(hit.relevant_chunks or [])[:3]:
                chunk_index = dict(chunk.metadata or {}).get("chunk_index")
                prefix = f"Chunk {chunk_index}: " if chunk_index is not None else "Chunk: "
                lines.append(prefix + str(chunk.text or "").strip())
            for claim in list(hit.document_claims or [])[:2]:
                obj = str(claim.object_surface or claim.obj or "").strip()
                if obj:
                    lines.append(f"Claim: {claim.subject} {claim.predicate} {obj}")
        return "\n".join(lines).strip()

    @staticmethod
    def _render_supporting_messages(
        *,
        selected_candidates: list[RetrievalCandidate],
        dialog_hits: list[DialogEpisodeHit],
    ) -> str:
        lines: list[str] = []
        seen: set[str] = set()

        for hit in list(dialog_hits or [])[:1]:
            for turn in list(hit.supporting_turns or [])[:4]:
                role = str(turn.role or "message").strip().lower() or "message"
                text = str(turn.text or "").strip()
                if not text:
                    continue
                key = f"{role}:{text.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {role}: {text}")
                if len(lines) >= 4:
                    return "\n".join(lines).strip()

        for item in list(selected_candidates or []):
            record = item.record
            if record.memory_type != MemoryType.MESSAGE:
                continue
            meta = dict(record.metadata or {})
            source_kind = str(meta.get("source_kind") or "").strip().lower()
            if source_kind == "assistant_reply":
                continue
            text = str(record.text or "").strip()
            if not text:
                continue
            role = "user" if source_kind in {"", "user"} else source_kind
            key = f"{role}:{text.lower()}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {role}: {text}")
            if len(lines) >= 4:
                break
        return "\n".join(lines).strip()

    @staticmethod
    def _is_document_recall_like_query(query_text: str) -> bool:
        text = str(query_text or "").strip().lower()
        if not text:
            return False
        return any(pattern.search(text) for pattern in _DOCUMENT_RECALL_RULES)

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
        return str(classify_query_recall_profile(str(query_text or "")).mode or "")

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
                    "- response_rule: prioritize dialog episodes, supporting messages, summaries, and decision/task facts.",
                    "- response_rule: answer from remembered discussion context, not as an exact self-fact lookup.",
                ]
            )
        if mode == "document_recall":
            return "\n".join(
                [
                    "- mode: document_recall",
                    "- response_rule: prioritize document chunks, section summaries, and document claims.",
                    "- response_rule: do not answer document/code/file questions from ordinary dialog memory if document evidence is present.",
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
        profile = classify_query_recall_profile(text)
        if bool(profile.self_like):
            return True
        follow_up_markers = (
            "ещё раз",
            "еще раз",
            "а сколько",
            "а какая",
            "а какой",
            "а какая у меня",
            "а какой у меня",
        )
        domain_markers = (
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
        return any(marker in text for marker in follow_up_markers) and any(marker in text for marker in domain_markers)

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
            self._governor_profile_snapshots = {}
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

    def get_governor_profile_snapshot(self, namespace: str = "default") -> GovernorProfileSnapshot | None:
        cache = getattr(self, "_governor_profile_snapshots", None)
        if not isinstance(cache, dict):
            return None
        return cache.get(str(namespace or "default"))

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
                metadata=self._strip_debug_metadata(
                    {
                        "fact": fact_payload,
                        "canonical_key": canonical,
                        "relation": fact.relation,
                        "write_policy": decision.to_dict(),
                    },
                    memory_type=MemoryType.FACT,
                ),
                importance=float(fact.importance),
                confidence=float(fact.confidence),
                created_at=now_ts,
                updated_at=now_ts,
                expires_at=fact.valid_to,
                status=fact.status,
                source_event_id=event_id,
            )

            governor_decision = self._governor.decide_for_fact(
                new_record=record,
                active_candidates=list(candidates or []),
            )
            self._record_governor_decision_event(
                now_ts=now_ts,
                namespace=namespace,
                source_event_id=event_id,
                fact_record=record,
                governor_decision=governor_decision,
                active_candidates=list(candidates or []),
            )

            if str(governor_decision.action or "").strip().lower() == "noop":
                continue

            record_metadata = {
                **dict(record.metadata or {}),
                "governor_reason": str(governor_decision.reason or ""),
            }
            if str(governor_decision.parallel_with_record_id or "").strip():
                record_metadata["parallel_with"] = str(governor_decision.parallel_with_record_id or "").strip()
            record = MemoryRecord(
                id=record.id,
                text=record.text,
                memory_type=record.memory_type,
                level=record.level,
                scope=record.scope,
                namespace=record.namespace,
                metadata=self._strip_debug_metadata(record_metadata, memory_type=record.memory_type),
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

            governor_action = str(governor_decision.action or "").strip().lower()
            if candidates and governor_action in {"supersede_old", "archive_old"}:
                target_status = MemoryStatus.SUPERSEDED if governor_action == "supersede_old" else MemoryStatus.ARCHIVED
                for previous in list(candidates):
                    transitioned = self._status_transition(
                        previous,
                        target=target_status,
                        now_ts=now_ts,
                        reason=str(governor_decision.reason or "governor_transition"),
                    )
                    transitioned = MemoryRecord(
                        id=transitioned.id,
                        text=transitioned.text,
                        memory_type=transitioned.memory_type,
                        level=transitioned.level,
                        scope=transitioned.scope,
                        namespace=transitioned.namespace,
                        metadata=self._strip_debug_metadata(
                            dict(transitioned.metadata or {}),
                            memory_type=transitioned.memory_type,
                        ),
                        embedding=list(transitioned.embedding or []) if isinstance(transitioned.embedding, list) else None,
                        importance=transitioned.importance,
                        confidence=transitioned.confidence,
                        created_at=transitioned.created_at,
                        updated_at=transitioned.updated_at,
                        expires_at=transitioned.expires_at,
                        status=transitioned.status,
                        version=transitioned.version,
                        parent_id=transitioned.parent_id,
                        chunk_index=transitioned.chunk_index,
                        source_event_id=transitioned.source_event_id,
                        embedding_model=transitioned.embedding_model,
                        embedding_fingerprint=transitioned.embedding_fingerprint,
                        embedding_version=transitioned.embedding_version,
                    )
                    self._store.upsert(transitioned)

            if governor_action == "keep_parallel" and str(governor_decision.parallel_with_record_id or "").strip():
                record = MemoryRecord(
                    id=record.id,
                    text=record.text,
                    memory_type=record.memory_type,
                    level=record.level,
                    scope=record.scope,
                    namespace=record.namespace,
                    metadata=self._strip_debug_metadata(
                        {
                            **dict(record.metadata or {}),
                            "parallel_with": str(governor_decision.parallel_with_record_id or "").strip(),
                        },
                        memory_type=record.memory_type,
                    ),
                    embedding=list(record.embedding or []) if isinstance(record.embedding, list) else None,
                    importance=record.importance,
                    confidence=record.confidence,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    expires_at=record.expires_at,
                    status=MemoryStatus.ACTIVE,
                    version=record.version,
                    parent_id=(record.parent_id or str(governor_decision.parallel_with_record_id or "") or None),
                    chunk_index=record.chunk_index,
                    source_event_id=record.source_event_id,
                    embedding_model=record.embedding_model,
                    embedding_fingerprint=record.embedding_fingerprint,
                    embedding_version=record.embedding_version,
                )

            self._store.upsert(record)
            out.append(fact)
        if facts:
            self._refresh_governor_profile_snapshot(namespace=namespace)
        return out

    def _write_claim_records(
        self,
        *,
        claims: list[ClaimRecord],
        namespace: str,
        now_ts: float,
        event_id: str,
    ) -> list[ClaimRecord]:
        out: list[ClaimRecord] = []
        for claim in list(claims or []):
            decision = self._decide_claim_promotion(claim=claim, namespace=namespace)
            if not decision.allow_write:
                continue
            record = MemoryRecord(
                id=f"claim:{uuid.uuid4().hex[:18]}",
                text=f"{claim.subject}.{claim.predicate}={claim.obj}",
                memory_type=MemoryType.CLAIM,
                level=MemoryLevel.L3_SEMANTIC,
                scope=MemoryScope(str(claim.scope or MemoryScope.CONVERSATION.value)),
                namespace=namespace,
                metadata=self._strip_debug_metadata(
                    {
                        "claim": claim.to_dict(),
                        "canonical_key": str(claim.canonical_key or ""),
                        "topic_keys": list(claim.topic_keys or []),
                        "trigger_keys": list(claim.trigger_keys or []),
                        "write_policy": decision.to_dict(),
                    },
                    memory_type=MemoryType.CLAIM,
                ),
                importance=float(claim.salience or 0.0),
                confidence=float(claim.confidence or 0.0),
                created_at=now_ts,
                updated_at=now_ts,
                status=MemoryStatus(str(claim.status or MemoryStatus.ACTIVE.value)),
                source_event_id=str(event_id or claim.source_event_id or ""),
            )
            self._store.upsert(record)
            out.append(claim)
        return out

    def _decide_claim_promotion(self, *, claim: ClaimRecord, namespace: str) -> ClaimPromotionDecision:
        existing = self._find_active_claim_by_canonical(namespace=namespace, canonical_key=str(claim.canonical_key or ""))
        if existing is None:
            return ClaimPromotionDecision(action="allow", reason="new_claim", allow_write=True)

        old_claim = dict(dict(existing.metadata or {}).get("claim") or {})
        old_obj = str(old_claim.get("obj") or old_claim.get("object_surface") or "").strip().lower()
        new_obj = str(claim.obj or claim.object_surface or "").strip().lower()
        if old_obj and new_obj and old_obj == new_obj:
            old_conf = float(old_claim.get("confidence") or existing.confidence or 0.0)
            new_conf = float(claim.confidence or 0.0)
            if new_conf <= old_conf:
                return ClaimPromotionDecision(
                    action="keep_existing",
                    reason="same_claim_existing_confidence",
                    allow_write=False,
                    signals={"old_conf": old_conf, "new_conf": new_conf},
                )
        return ClaimPromotionDecision(action="allow", reason="claim_refresh", allow_write=True)

    def _find_active_fact_candidates_for_write(self, *, namespace: str, fact: FactRecordV2) -> list[MemoryRecord]:
        rows = self._latest_records(namespace=namespace)
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
        rows = self._latest_records(namespace=namespace)
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

    def _find_active_claim_by_canonical(self, *, namespace: str, canonical_key: str) -> MemoryRecord | None:
        rows = self._latest_records(namespace=namespace)
        key = str(canonical_key or "").strip().lower()
        if not key:
            return None
        for row in rows:
            if row.memory_type != MemoryType.CLAIM:
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

    def _latest_records(self, *, namespace: str | None = None) -> list[MemoryRecord]:
        latest_by_id: dict[str, MemoryRecord] = {}
        order: list[str] = []
        for row in list(self._store.iter_records(namespace=namespace) or []):
            row_id = str(getattr(row, "id", "") or "")
            if not row_id:
                continue
            if row_id not in latest_by_id:
                order.append(row_id)
            latest_by_id[row_id] = row
        return [latest_by_id[row_id] for row_id in order if row_id in latest_by_id]

    def _refresh_governor_profile_snapshot(self, *, namespace: str) -> GovernorProfileSnapshot | None:
        governor = getattr(self, "_governor", None)
        if governor is None:
            return None
        namespace_norm = str(namespace or "default")
        active_rows = [
            row
            for row in self._latest_records(namespace=namespace_norm)
            if row.memory_type == MemoryType.FACT and row.status == MemoryStatus.ACTIVE
        ]
        snapshot = governor.rebuild_profile_snapshot(
            namespace=namespace_norm,
            active_fact_rows=active_rows,
        )
        cache = getattr(self, "_governor_profile_snapshots", None)
        if not isinstance(cache, dict):
            cache = {}
            self._governor_profile_snapshots = cache
        cache[namespace_norm] = snapshot
        self._record_profile_snapshot_rebuilt_event(namespace=namespace_norm, snapshot=snapshot)
        return snapshot

    def _record_governor_decision_event(
        self,
        *,
        now_ts: float,
        namespace: str,
        source_event_id: str,
        fact_record: MemoryRecord,
        governor_decision: Any,
        active_candidates: list[MemoryRecord],
    ) -> None:
        fact_meta = dict(dict(fact_record.metadata or {}).get("fact") or {})
        self._event_store.append(
            {
                "ts": now_ts,
                "type": "memory_governor_decision",
                "payload": {
                    "namespace": str(namespace or "default"),
                    "source_event_id": str(source_event_id or ""),
                    "record_id": str(fact_record.id or ""),
                    "canonical_key": str(dict(fact_record.metadata or {}).get("canonical_key") or ""),
                    "predicate": str(fact_meta.get("predicate") or ""),
                    "value": fact_meta.get("value"),
                    "group": str(dict(governor_decision.debug or {}).get("group") or ""),
                    "group_mode": str(dict(governor_decision.debug or {}).get("group_mode") or ""),
                    "action": str(governor_decision.action or ""),
                    "reason": str(governor_decision.reason or ""),
                    "winner_record_id": str(governor_decision.winner_record_id or ""),
                    "loser_record_id": str(governor_decision.loser_record_id or ""),
                    "parallel_with_record_id": str(governor_decision.parallel_with_record_id or ""),
                    "active_candidate_ids": [str(row.id or "") for row in list(active_candidates or [])],
                    "superseded_record_ids": (
                        [str(row.id or "") for row in list(active_candidates or [])]
                        if str(governor_decision.action or "").strip().lower() == "supersede_old"
                        else []
                    ),
                },
                "tags": [
                    "memory",
                    "governor",
                    str(namespace or "default"),
                    str(governor_decision.action or "").strip().lower() or "unknown",
                ],
            }
        )

    def _record_profile_snapshot_rebuilt_event(
        self,
        *,
        namespace: str,
        snapshot: GovernorProfileSnapshot,
    ) -> None:
        self._event_store.append(
            {
                "ts": float(snapshot.updated_at or time.time()),
                "type": "memory_profile_snapshot_rebuilt",
                "payload": {
                    "namespace": str(namespace or "default"),
                    "rebuilt": True,
                    "active_fact_count": len(dict(snapshot.active_facts or {})),
                    "conflict_count": len(list(snapshot.conflicts or [])),
                    "conflict_groups": [
                        str(dict(item or {}).get("group") or "")
                        for item in list(snapshot.conflicts or [])
                        if str(dict(item or {}).get("group") or "").strip()
                    ],
                },
                "tags": ["memory", "profile_snapshot", str(namespace or "default")],
            }
        )

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
                # Strip debug metadata from decayed records
                next_row = MemoryRecord(
                    id=next_row.id,
                    text=next_row.text,
                    memory_type=next_row.memory_type,
                    level=next_row.level,
                    scope=next_row.scope,
                    namespace=next_row.namespace,
                    metadata=self._strip_debug_metadata(next_row.metadata, memory_type=next_row.memory_type),
                    embedding=list(next_row.embedding or []) if isinstance(next_row.embedding, list) else None,
                    importance=next_row.importance,
                    confidence=next_row.confidence,
                    created_at=next_row.created_at,
                    updated_at=next_row.updated_at,
                    expires_at=next_row.expires_at,
                    status=next_row.status,
                    version=next_row.version,
                    parent_id=next_row.parent_id,
                    chunk_index=next_row.chunk_index,
                )
                changed.append(next_row)
        if changed:
            self._store.batch_upsert(changed)

    def _strip_debug_metadata(self, metadata: dict[str, Any], *, memory_type: Any = None) -> dict[str, Any]:
        """Remove debug/runtime fields from metadata before storage.

        In compact profile, also apply full compaction via MemoryPolicy._compact_metadata().
        """
        if not isinstance(metadata, dict):
            return metadata
        return sanitize_storage_metadata(
            metadata=dict(metadata or {}),
            storage_profile=self._storage_profile,
            memory_type=memory_type,
        )

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
        if memory_type in {MemoryType.FACT, MemoryType.CLAIM}:
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
        """Merge ingest analysis into metadata.

        Store only essential fields for retrieval and search.
        Skip verbose analysis details to save storage space.
        """
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

        def _compact_claims() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            for item in list(analysis.claim_candidates or []):
                subject = str(item.subject or "").strip().lower()
                predicate = str(item.predicate or "").strip().lower()
                obj = str(item.normalized_object or item.object_surface or "").strip()
                if not subject or not predicate or not obj:
                    continue
                key = (subject, predicate, obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "subject": subject,
                    "predicate": predicate,
                    "obj": obj,
                    "object_surface": str(item.object_surface or "").strip(),
                    "object_type": str(item.object_type or "").strip().lower(),
                    "topic_keys": [str(x).strip().lower() for x in list(item.topic_keys or []) if str(x).strip()][:6],
                    "trigger_keys": [str(x).strip().lower() for x in list(item.trigger_keys or []) if str(x).strip()][:6],
                    "confidence": round(float(item.confidence or 0.0), 4),
                }
                rows.append(row)
            return rows

        def _compact_emotion_profile() -> dict[str, Any]:
            if analysis.emotion is None:
                return {}
            primary = str(analysis.emotion.primary or "").strip()
            label = primary
            out: dict[str, Any] = {}
            if primary:
                out["primary"] = primary
                out["label"] = label
            try:
                intensity = float(analysis.emotion.intensity or 0.0)
            except Exception:
                intensity = 0.0
            try:
                confidence = float(analysis.emotion.confidence or 0.0)
            except Exception:
                confidence = 0.0
            if intensity > 0.0:
                out["intensity"] = round(intensity, 4)
            if confidence > 0.0:
                out["confidence"] = round(confidence, 4)
            return out

        # Store only compact retrieval keys in memory_views.
        entity_keys = list(analysis.memory_views.get("entity_keys") or [])
        numeric_keys = list(analysis.memory_views.get("numeric_keys") or [])
        if entity_keys or numeric_keys:
            compact_views: dict[str, Any] = {}
            if entity_keys:
                compact_views["entity_keys"] = entity_keys
            if numeric_keys:
                compact_views["numeric_keys"] = numeric_keys
            if compact_views:
                out["memory_views"] = compact_views

        # Only store non-empty lists
        memory_entities = [item.to_dict() for item in list(analysis.entities or [])]
        if memory_entities:
            out["memory_entities"] = memory_entities

        numeric_facts = [item.to_dict() for item in list(analysis.numeric_facts or [])]
        if numeric_facts:
            out["numeric_facts"] = numeric_facts

        claims = _compact_claims()
        if claims:
            out["claims"] = claims

        stable_facts = [item.to_dict() for item in list(analysis.stable_facts or [])]
        if stable_facts:
            out["stable_facts"] = stable_facts

        emotion_view = _compact_emotion_profile()
        if emotion_view:
            out["emotion_profile"] = emotion_view

        out["memory_tags"] = _merge_tags(out.get("memory_tags"), list(analysis.tags or []))
        return out

    def _augment_message_metadata(
        self,
        *,
        text: str,
        metadata: dict[str, Any],
        namespace: str,
        preview_facts: list[FactRecordV2],
    ) -> dict[str, Any]:
        """Augment metadata with signal analysis for lifecycle decisions.

        Do NOT store promotion signals - they are computed for decision-making
        only and should not be persisted to save storage space.
        """
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

        # Do NOT store promotion signals - they are runtime-only for lifecycle decisions
        # Only store fact relations for reference (useful for debugging/analysis)
        if fact_relations:
            out["fact_relations"] = list(fact_relations)[:16]
        if facts_count > 0:
            out["facts_count"] = facts_count

        return out

    def _attach_lifecycle_debug(
        self,
        record: MemoryRecord,
        *,
        lifecycle_decision: LifecycleDecision,
        extracted_facts_count: int,
    ) -> MemoryRecord:
        """Attach lifecycle decision to metadata.

        Store only minimal decision info for runtime use.
        Full debug info is logged but not persisted to save storage.
        """
        meta = dict(record.metadata or {})
        # Store only essential lifecycle decision fields (no promotion_debug)
        lifecycle_payload = {
            "reason": str(lifecycle_decision.reason or ""),
            "route": str(lifecycle_decision.route or ""),
            "promote_to": (
                str(lifecycle_decision.promote_to.value) if lifecycle_decision.promote_to is not None else None
            ),
            "mark_status": (
                str(lifecycle_decision.mark_status.value) if lifecycle_decision.mark_status is not None else None
            ),
        }
        updated_meta = {
            **meta,
            "lifecycle_reason": lifecycle_payload["reason"],
            "lifecycle_route": lifecycle_payload["route"],
        }
        if lifecycle_decision.promote_to is not None:
            updated_meta["promoted_to"] = lifecycle_payload["promote_to"]
        if lifecycle_decision.mark_status is not None:
            updated_meta["mark_status"] = lifecycle_payload["mark_status"]
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
        """Record skipped ingest event (event store only, not in memory metadata)."""
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
                    "assistant_write_action": str(decision.action or "allow"),
                    "assistant_write_reason": str(decision.reason or ""),
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
                f"write_action={str(decision.action or 'allow')} "
                f"factual_mode={str(dict(decision.signals or {}).get('factual_mode') or '-')} "
                f"confidence={float(dict(decision.signals or {}).get('final_factual_confidence') or 0.0):.3f}"
            ),
            context=self._event_log_context(metadata=metadata, namespace=namespace),
            namespace=namespace,
            scope=str(event.scope.value),
            memory_type=str(event.memory_type.value),
            source_kind=str(metadata.get("source_kind") or ""),
            reason=str(reason or ""),
            write_action=str(decision.action or "allow"),
            write_reason=str(decision.reason or ""),
            factual_mode=str(dict(decision.signals or {}).get("factual_mode") or ""),
            final_factual_confidence=float(dict(decision.signals or {}).get("final_factual_confidence") or 0.0),
            conflict_severity=float(dict(decision.signals or {}).get("conflict_severity") or 0.0),
        )

    @staticmethod
    def _without_promotion(
        lifecycle_decision: LifecycleDecision,
        *,
        reason: str,
        scope: MemoryScope,
        decision: AssistantWriteDecision,
    ) -> LifecycleDecision:
        """Create lifecycle decision without promotion (runtime-only).

        Do not store debug payload - it's used for logging only.
        """
        return LifecycleDecision(
            promote_to=None,
            mark_status=lifecycle_decision.mark_status,
            archive=bool(lifecycle_decision.archive),
            reason=str(reason or lifecycle_decision.reason or ""),
            route="assistant_write_policy",
            next_version=lifecycle_decision.next_version,
            chain_parent_id=lifecycle_decision.chain_parent_id,
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
