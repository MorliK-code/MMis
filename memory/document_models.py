from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from memory.memory_models import (
    ChunkRecord as _BaseDocumentChunk,
    DocumentRecord as _BaseDocumentRecord,
    MemoryLevel,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)


@dataclass(frozen=True)
class DocumentRecord(_BaseDocumentRecord):
    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DocumentRecord":
        return cls(
            id=str(row.get("id") or ""),
            source=str(row.get("source") or ""),
            text=str(row.get("text") or ""),
            summary=str(row.get("summary") or ""),
            scope=MemoryScope(str(row.get("scope") or MemoryScope.PROJECT.value)),
            namespace=str(row.get("namespace") or "default"),
            metadata=dict(row.get("metadata") or {}),
            memory_type=MemoryType(str(row.get("memory_type") or MemoryType.DOCUMENT.value)),
            level=MemoryLevel(str(row.get("level") or MemoryLevel.L4_DOCUMENT.value)),
            importance=float(row.get("importance") or 0.72),
            confidence=float(row.get("confidence") or 0.86),
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
            status=MemoryStatus(str(row.get("status") or MemoryStatus.ACTIVE.value)),
            version=max(1, int(row.get("version") or 1)),
            parent_id=(str(row.get("parent_id")) if row.get("parent_id") else None),
            chunk_index=(int(row.get("chunk_index")) if row.get("chunk_index") is not None else None),
        )


@dataclass(frozen=True)
class DocumentChunk(_BaseDocumentChunk):
    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DocumentChunk":
        return cls(
            id=str(row.get("id") or ""),
            document_id=str(row.get("document_id") or ""),
            chunk_index=int(row.get("chunk_index") or 0),
            text=str(row.get("text") or ""),
            scope=MemoryScope(str(row.get("scope") or MemoryScope.PROJECT.value)),
            namespace=str(row.get("namespace") or "default"),
            metadata=dict(row.get("metadata") or {}),
            memory_type=MemoryType(str(row.get("memory_type") or MemoryType.DOCUMENT_CHUNK.value)),
            level=MemoryLevel(str(row.get("level") or MemoryLevel.L4_DOCUMENT.value)),
            importance=float(row.get("importance") or 0.66),
            confidence=float(row.get("confidence") or 0.84),
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
            status=MemoryStatus(str(row.get("status") or MemoryStatus.ACTIVE.value)),
            version=max(1, int(row.get("version") or 1)),
            parent_id=(str(row.get("parent_id")) if row.get("parent_id") else None),
        )


@dataclass(frozen=True)
class DocumentSummary:
    id: str
    document_id: str
    text: str
    summary_kind: str = "overview"
    source_chunk_ids: list[str] = field(default_factory=list)
    topic_keys: list[str] = field(default_factory=list)
    entity_keys: list[str] = field(default_factory=list)
    confidence: float = 0.8
    salience: float = 0.72
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: float(time.time()))
    updated_at: float = field(default_factory=lambda: float(time.time()))
    status: MemoryStatus = MemoryStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id or ""),
            "document_id": str(self.document_id or ""),
            "text": str(self.text or ""),
            "summary_kind": str(self.summary_kind or "overview"),
            "source_chunk_ids": [str(x).strip() for x in list(self.source_chunk_ids or []) if str(x).strip()],
            "topic_keys": [str(x).strip().lower() for x in list(self.topic_keys or []) if str(x).strip()],
            "entity_keys": [str(x).strip().lower() for x in list(self.entity_keys or []) if str(x).strip()],
            "confidence": float(self.confidence),
            "salience": float(self.salience),
            "metadata": dict(self.metadata or {}),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "status": str(self.status.value),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DocumentSummary":
        return cls(
            id=str(row.get("id") or ""),
            document_id=str(row.get("document_id") or ""),
            text=str(row.get("text") or ""),
            summary_kind=str(row.get("summary_kind") or "overview"),
            source_chunk_ids=[str(x).strip() for x in list(row.get("source_chunk_ids") or []) if str(x).strip()],
            topic_keys=[str(x).strip().lower() for x in list(row.get("topic_keys") or []) if str(x).strip()],
            entity_keys=[str(x).strip().lower() for x in list(row.get("entity_keys") or []) if str(x).strip()],
            confidence=float(row.get("confidence") or 0.8),
            salience=float(row.get("salience") or 0.72),
            metadata=dict(row.get("metadata") or {}),
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
            status=MemoryStatus(str(row.get("status") or MemoryStatus.ACTIVE.value)),
        )


@dataclass(frozen=True)
class DocumentClaim:
    id: str
    document_id: str
    chunk_id: str
    subject: str
    predicate: str
    obj: str
    subject_type: str = ""
    object_type: str = ""
    object_surface: str = ""
    qualifiers: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    salience: float = 0.0
    evidence_text: str = ""
    topic_keys: list[str] = field(default_factory=list)
    trigger_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: float(time.time()))
    updated_at: float = field(default_factory=lambda: float(time.time()))
    status: MemoryStatus = MemoryStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id or ""),
            "document_id": str(self.document_id or ""),
            "chunk_id": str(self.chunk_id or ""),
            "subject": str(self.subject or ""),
            "predicate": str(self.predicate or ""),
            "obj": str(self.obj or ""),
            "subject_type": str(self.subject_type or ""),
            "object_type": str(self.object_type or ""),
            "object_surface": str(self.object_surface or ""),
            "qualifiers": dict(self.qualifiers or {}),
            "confidence": float(self.confidence),
            "salience": float(self.salience),
            "evidence_text": str(self.evidence_text or ""),
            "topic_keys": [str(x).strip().lower() for x in list(self.topic_keys or []) if str(x).strip()],
            "trigger_keys": [str(x).strip().lower() for x in list(self.trigger_keys or []) if str(x).strip()],
            "metadata": dict(self.metadata or {}),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "status": str(self.status.value),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DocumentClaim":
        return cls(
            id=str(row.get("id") or ""),
            document_id=str(row.get("document_id") or ""),
            chunk_id=str(row.get("chunk_id") or ""),
            subject=str(row.get("subject") or ""),
            predicate=str(row.get("predicate") or ""),
            obj=str(row.get("obj") or ""),
            subject_type=str(row.get("subject_type") or ""),
            object_type=str(row.get("object_type") or ""),
            object_surface=str(row.get("object_surface") or ""),
            qualifiers=dict(row.get("qualifiers") or {}),
            confidence=float(row.get("confidence") or 0.0),
            salience=float(row.get("salience") or 0.0),
            evidence_text=str(row.get("evidence_text") or ""),
            topic_keys=[str(x).strip().lower() for x in list(row.get("topic_keys") or []) if str(x).strip()],
            trigger_keys=[str(x).strip().lower() for x in list(row.get("trigger_keys") or []) if str(x).strip()],
            metadata=dict(row.get("metadata") or {}),
            created_at=float(row.get("created_at") or time.time()),
            updated_at=float(row.get("updated_at") or time.time()),
            status=MemoryStatus(str(row.get("status") or MemoryStatus.ACTIVE.value)),
        )
