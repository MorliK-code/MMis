"""Typed foundation models and contracts for Memory V2."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class MemoryLevel(str, Enum):
    L0_WORKING = "l0_working"
    L1_SESSION = "l1_session"
    L2_EPISODIC = "l2_episodic"
    L3_SEMANTIC = "l3_semantic"
    L4_DOCUMENT = "l4_document"


class MemoryScope(str, Enum):
    GLOBAL_USER = "global_user"
    CONVERSATION = "conversation"
    SESSION = "session"
    PROJECT = "project"
    CHARACTER = "character"
    TEMPORARY = "temporary"
    PRIVATE_RUNTIME = "private_runtime"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemoryType(str, Enum):
    MESSAGE = "message"
    SUMMARY = "summary"
    FACT = "fact"
    CLAIM = "claim"
    IDENTITY_CORE = "identity_core"
    EPISODE = "episode"
    SEMANTIC = "semantic"
    DOCUMENT = "document"
    DOCUMENT_CHUNK = "document_chunk"
    TASK_STATE = "task_state"
    TOOL_RESULT = "tool_result"
    RUNTIME_STATE = "runtime_state"


class MemorySourceKind(str, Enum):
    USER = "user"
    ASSISTANT_REPLY = "assistant_reply"
    ASSISTANT_THOUGHT = "assistant_thought"
    TOOL_RESULT = "tool_result"
    SYSTEM_DECISION = "system_decision"


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    text: str
    memory_type: MemoryType
    level: MemoryLevel
    scope: MemoryScope
    namespace: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    importance: float = 0.5
    confidence: float = 0.5
    created_at: float = field(default_factory=lambda: float(time.time()))
    updated_at: float = field(default_factory=lambda: float(time.time()))
    expires_at: float | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    parent_id: str | None = None
    chunk_index: int | None = None
    source_event_id: str = ""
    embedding_model: str = ""
    embedding_fingerprint: str = ""
    embedding_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "memory_type": str(self.memory_type.value),
            "level": str(self.level.value),
            "scope": str(self.scope.value),
            "namespace": str(self.namespace),
            "metadata": dict(self.metadata or {}),
            "embedding": list(self.embedding) if isinstance(self.embedding, list) else None,
            "importance": float(self.importance),
            "confidence": float(self.confidence),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "expires_at": (float(self.expires_at) if self.expires_at is not None else None),
            "status": str(self.status.value),
            "version": int(self.version),
            "parent_id": self.parent_id,
            "chunk_index": self.chunk_index,
            "source_event_id": str(self.source_event_id or ""),
            "embedding_model": str(self.embedding_model or ""),
            "embedding_fingerprint": str(self.embedding_fingerprint or ""),
            "embedding_version": str(self.embedding_version or ""),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=str(row.get("id") or ""),
            text=str(row.get("text") or ""),
            memory_type=MemoryType(str(row.get("memory_type") or MemoryType.MESSAGE.value)),
            level=MemoryLevel(str(row.get("level") or MemoryLevel.L0_WORKING.value)),
            scope=MemoryScope(str(row.get("scope") or MemoryScope.CONVERSATION.value)),
            namespace=str(row.get("namespace") or "default"),
            metadata=dict(row.get("metadata") or {}),
            embedding=list(row.get("embedding") or []) if isinstance(row.get("embedding"), list) else None,
            importance=float(row.get("importance") or 0.5),
            confidence=float(row.get("confidence") or 0.5),
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
            expires_at=(float(row.get("expires_at")) if row.get("expires_at") is not None else None),
            status=MemoryStatus(str(row.get("status") or MemoryStatus.ACTIVE.value)),
            version=max(1, int(row.get("version") or 1)),
            parent_id=(str(row.get("parent_id")) if row.get("parent_id") else None),
            chunk_index=(int(row.get("chunk_index")) if row.get("chunk_index") is not None else None),
            source_event_id=str(row.get("source_event_id") or ""),
            embedding_model=str(row.get("embedding_model") or ""),
            embedding_fingerprint=str(row.get("embedding_fingerprint") or ""),
            embedding_version=str(row.get("embedding_version") or ""),
        )


@dataclass(frozen=True)
class FactRecordV2:
    subject: str
    predicate: str
    value: Any
    scope: MemoryScope
    confidence: float
    importance: float
    evidence: str
    source_event_id: str
    valid_from: float | None = None
    valid_to: float | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    canonical_key: str = ""
    relation: str = ""
    id: str = ""
    text: str = ""
    memory_type: MemoryType = MemoryType.FACT
    level: MemoryLevel = MemoryLevel.L3_SEMANTIC
    namespace: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: float(time.time()))
    updated_at: float = field(default_factory=lambda: float(time.time()))
    parent_id: str | None = None
    chunk_index: int | None = None
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "scope": str(self.scope.value),
            "confidence": float(self.confidence),
            "importance": float(self.importance),
            "evidence": self.evidence,
            "source_event_id": self.source_event_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "status": str(self.status.value),
            "canonical_key": self.canonical_key,
            "relation": self.relation,
            "id": str(self.id or ""),
            "text": str(self.text or ""),
            "memory_type": str(self.memory_type.value),
            "level": str(self.level.value),
            "namespace": str(self.namespace or "default"),
            "metadata": dict(self.metadata or {}),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "parent_id": self.parent_id,
            "chunk_index": self.chunk_index,
            "version": max(1, int(self.version)),
        }


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    source: str
    text: str
    summary: str
    scope: MemoryScope
    namespace: str
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_type: MemoryType = MemoryType.DOCUMENT
    level: MemoryLevel = MemoryLevel.L4_DOCUMENT
    importance: float = 0.72
    confidence: float = 0.86
    created_at: float = field(default_factory=lambda: float(time.time()))
    updated_at: float = field(default_factory=lambda: float(time.time()))
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    parent_id: str | None = None
    chunk_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id or ""),
            "source": str(self.source or ""),
            "text": str(self.text or ""),
            "summary": str(self.summary or ""),
            "scope": str(self.scope.value),
            "namespace": str(self.namespace or "default"),
            "metadata": dict(self.metadata or {}),
            "memory_type": str(self.memory_type.value),
            "level": str(self.level.value),
            "importance": float(self.importance),
            "confidence": float(self.confidence),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "status": str(self.status.value),
            "version": max(1, int(self.version)),
            "parent_id": self.parent_id,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    document_id: str
    chunk_index: int
    text: str
    scope: MemoryScope
    namespace: str
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_type: MemoryType = MemoryType.DOCUMENT_CHUNK
    level: MemoryLevel = MemoryLevel.L4_DOCUMENT
    importance: float = 0.66
    confidence: float = 0.84
    created_at: float = field(default_factory=lambda: float(time.time()))
    updated_at: float = field(default_factory=lambda: float(time.time()))
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id or ""),
            "document_id": str(self.document_id or ""),
            "chunk_index": int(self.chunk_index),
            "text": str(self.text or ""),
            "scope": str(self.scope.value),
            "namespace": str(self.namespace or "default"),
            "metadata": dict(self.metadata or {}),
            "memory_type": str(self.memory_type.value),
            "level": str(self.level.value),
            "importance": float(self.importance),
            "confidence": float(self.confidence),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "status": str(self.status.value),
            "version": max(1, int(self.version)),
            "parent_id": self.parent_id,
        }


@dataclass(frozen=True)
class ScoreBreakdown:
    semantic_similarity: float = 0.0
    lexical_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    confidence_score: float = 0.0
    entity_overlap_score: float = 0.0
    numeric_overlap_score: float = 0.0
    exact_match_boost: float = 0.0
    scope_match_score: float = 0.0
    final_score: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "semantic_similarity": float(self.semantic_similarity),
            "lexical_score": float(self.lexical_score),
            "recency_score": float(self.recency_score),
            "importance_score": float(self.importance_score),
            "confidence_score": float(self.confidence_score),
            "entity_overlap_score": float(self.entity_overlap_score),
            "numeric_overlap_score": float(self.numeric_overlap_score),
            "exact_match_boost": float(self.exact_match_boost),
            "scope_match_score": float(self.scope_match_score),
            "final_score": float(self.final_score),
        }


@dataclass(frozen=True)
class RetrievalQuery:
    query_text: str
    search_text: str = ""
    entity_keys: list[str] = field(default_factory=list)
    numeric_keys: list[str] = field(default_factory=list)
    namespace: str = "default"
    scopes: list[MemoryScope] = field(default_factory=list)
    top_k: int = 8
    include_private_runtime: bool = False
    include_stale: bool = False
    metadata_filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalCandidate:
    record: MemoryRecord
    score_breakdown: ScoreBreakdown
    source: str = "hybrid"

    @property
    def final_score(self) -> float:
        return float(self.score_breakdown.final_score)

    def to_dict(self) -> dict[str, Any]:
        row = self.record.to_dict()
        row["score"] = float(self.final_score)
        row["score_breakdown"] = self.score_breakdown.to_dict()
        row["source"] = self.source
        return row


@dataclass(frozen=True)
class RetrievalResult:
    query: RetrievalQuery
    candidates: list[RetrievalCandidate]
    reindex_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": {
                "query_text": self.query.query_text,
                "search_text": self.query.search_text,
                "entity_keys": list(self.query.entity_keys or []),
                "numeric_keys": list(self.query.numeric_keys or []),
                "namespace": self.query.namespace,
                "scopes": [str(x.value) for x in list(self.query.scopes or [])],
                "top_k": int(self.query.top_k),
            },
            "reindex_required": bool(self.reindex_required),
            "candidates": [x.to_dict() for x in list(self.candidates or [])],
        }


@dataclass(frozen=True)
class ContextBuildRequest:
    system_prompt: str
    user_message: str
    namespace: str = "default"
    scopes: list[MemoryScope] = field(default_factory=list)
    top_k: int = 8
    session_summary: str = ""
    working_memory: list[MemoryRecord] = field(default_factory=list)
    tool_state: dict[str, Any] = field(default_factory=dict)
    unresolved_items: list[str] = field(default_factory=list)
    context_budget_total: int = 2048
    context_budget_memory: int = 700
    context_budget_docs: int = 600
    context_budget_tools: int = 200
    context_budget_response_reserve: int = 256


@dataclass(frozen=True)
class ContextBuildResult:
    blocks: dict[str, str]
    selected: list[RetrievalCandidate]
    dropped: list[dict[str, Any]] = field(default_factory=list)
    score_breakdowns: list[dict[str, Any]] = field(default_factory=list)
    truncation_log: list[dict[str, Any]] = field(default_factory=list)
    fact_expectation: dict[str, Any] = field(default_factory=dict)
    self_facts_context: dict[str, Any] = field(default_factory=dict)
    dialog_episode_hits: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    current_decisions: list[str] = field(default_factory=list)
    recall_mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocks": dict(self.blocks or {}),
            "selected": [x.to_dict() for x in list(self.selected or [])],
            "dropped": [dict(x) for x in list(self.dropped or [])],
            "score_breakdowns": [dict(x) for x in list(self.score_breakdowns or [])],
            "truncation_log": [dict(x) for x in list(self.truncation_log or [])],
            "fact_expectation": dict(self.fact_expectation or {}),
            "self_facts_context": dict(self.self_facts_context or {}),
            "dialog_episode_hits": [dict(x) for x in list(self.dialog_episode_hits or [])],
            "open_questions": [str(x) for x in list(self.open_questions or []) if str(x).strip()],
            "current_decisions": [str(x) for x in list(self.current_decisions or []) if str(x).strip()],
            "recall_mode": str(self.recall_mode or ""),
        }


@dataclass(frozen=True)
class LifecycleDecision:
    promote_to: MemoryLevel | None = None
    mark_status: MemoryStatus | None = None
    archive: bool = False
    reason: str = ""
    route: str = ""
    next_version: int | None = None
    chain_parent_id: str | None = None
    decision_debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConflictDecision:
    keep_record_id: str
    superseded_record_id: str | None
    reason: str
    status: MemoryStatus
    action: str = "supersede"
    archive_record_id: str | None = None
    parallel_with_record_id: str | None = None
    score_delta: float = 0.0


@dataclass(frozen=True)
class MemoryEvent:
    role: str
    text: str
    namespace: str = "default"
    scope: MemoryScope = MemoryScope.CONVERSATION
    memory_type: MemoryType = MemoryType.MESSAGE
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=lambda: float(time.time()))
    thinking: str = ""


@dataclass(frozen=True)
class IngestResult:
    stored_ids: list[str]
    promoted_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    extracted_facts: list[FactRecordV2] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentIngestRequest:
    text: str
    source: str
    namespace: str = "default"
    scope: MemoryScope = MemoryScope.PROJECT
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentIngestResult:
    document: DocumentRecord
    chunks: list[ChunkRecord]


@dataclass(frozen=True)
class DebugRequest:
    namespace: str = "default"
    scope: MemoryScope | None = None
    status: MemoryStatus | None = None
    limit: int = 100


class EmbeddingProvider(Protocol):
    model_name: str
    embedding_version: str

    def model_fingerprint(self) -> str:
        ...

    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class VectorIndexBackend(ABC):
    @abstractmethod
    def upsert(self, record: MemoryRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def batch_upsert(self, records: list[MemoryRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def iter_records(self, *, namespace: str | None = None) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class LexicalIndexBackend(ABC):
    @abstractmethod
    def upsert(self, record: MemoryRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def batch_upsert(self, records: list[MemoryRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def iter_records(self, *, namespace: str | None = None) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class Reranker(Protocol):
    def rerank(self, query: RetrievalQuery, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        ...


class ContextCompressor(Protocol):
    def compress(self, request: ContextBuildRequest, result: ContextBuildResult) -> ContextBuildResult:
        ...


class MemoryLifecycle(Protocol):
    def decide(self, record: MemoryRecord, *, now_ts: float) -> LifecycleDecision:
        ...


class MemoryDebugger(Protocol):
    def snapshot(self, request: DebugRequest) -> dict[str, Any]:
        ...
