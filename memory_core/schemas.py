"""
Модели данных (schemas) для memory_core.
"""

from dataclasses import dataclass, field
from typing import Any
import time
import uuid


def _generate_id() -> str:
    """Генерирует уникальный ID."""
    return str(uuid.uuid4())


def _current_time() -> float:
    """Возвращает текущее время в секундах."""
    return time.time()


@dataclass(slots=True)
class MemoryEnvelope:
    """
    Конверт для входящего события памяти.
    
    Любое событие, попадающее в память, обёрнуто в этот конверт.
    """
    event_id: str = field(default_factory=_generate_id)
    source_kind: str = "user"  # user | assistant | tool | system | document
    payload_type: str = "message"  # message | tool_result | state_update | doc_text
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: str = "default"
    workspace_id: str = "global"
    session_id: str = "default"
    ts: float = field(default_factory=_current_time)
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "event_id": self.event_id,
            "source_kind": self.source_kind,
            "payload_type": self.payload_type,
            "text": self.text,
            "metadata": self.metadata,
            "namespace": self.namespace,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "ts": self.ts,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEnvelope":
        """Создаёт из словаря."""
        return cls(
            event_id=data.get("event_id", _generate_id()),
            source_kind=data.get("source_kind", "user"),
            payload_type=data.get("payload_type", "message"),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            namespace=data.get("namespace", "default"),
            workspace_id=data.get("workspace_id", "global"),
            session_id=data.get("session_id", "default"),
            ts=data.get("ts", _current_time()),
        )


@dataclass(slots=True)
class MemoryArtifact:
    """
    Артефакт памяти - результат обработки события.
    
    Артефакты создаются процессорами из сырых событий.
    """
    artifact_id: str = field(default_factory=_generate_id)
    artifact_type: str = "fact"  # fact | profile_fact | episode | task | document_chunk | etc.
    source_event_id: str = ""
    text: str = ""
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: str = "default"
    workspace_id: str = "global"
    status: str = "active"  # active | archived | superseded
    created_at: float = field(default_factory=_current_time)
    updated_at: float = field(default_factory=_current_time)
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "source_event_id": self.source_event_id,
            "text": self.text,
            "summary": self.summary,
            "metadata": self.metadata,
            "namespace": self.namespace,
            "workspace_id": self.workspace_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryArtifact":
        """Создаёт из словаря."""
        return cls(
            artifact_id=data.get("artifact_id", _generate_id()),
            artifact_type=data.get("artifact_type", "fact"),
            source_event_id=data.get("source_event_id", ""),
            text=data.get("text", ""),
            summary=data.get("summary", ""),
            metadata=data.get("metadata", {}),
            namespace=data.get("namespace", "default"),
            workspace_id=data.get("workspace_id", "global"),
            status=data.get("status", "active"),
            created_at=data.get("created_at", _current_time()),
            updated_at=data.get("updated_at", _current_time()),
        )


@dataclass(slots=True)
class MemoryQuery:
    """
    Запрос к памяти для retrieval.
    """
    text: str = ""
    namespace: str = "default"
    workspace_id: str = "global"
    artifact_types: list[str] = field(default_factory=list)
    top_k: int = 8
    include_citations: bool = True
    session_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "text": self.text,
            "namespace": self.namespace,
            "workspace_id": self.workspace_id,
            "artifact_types": self.artifact_types,
            "top_k": self.top_k,
            "include_citations": self.include_citations,
            "session_id": self.session_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryQuery":
        """Создаёт из словаря."""
        return cls(
            text=data.get("text", ""),
            namespace=data.get("namespace", "default"),
            workspace_id=data.get("workspace_id", "global"),
            artifact_types=data.get("artifact_types", []),
            top_k=data.get("top_k", 8),
            include_citations=data.get("include_citations", True),
            session_id=data.get("session_id"),
        )


@dataclass(slots=True)
class MemoryQueryResult:
    """
    Результат запроса к памяти.
    """
    hits: list[dict[str, Any]] = field(default_factory=list)
    context_blocks: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "hits": self.hits,
            "context_blocks": self.context_blocks,
            "citations": self.citations,
        }


@dataclass(slots=True)
class DocumentIngestRequest:
    """
    Запрос на ingest документа.
    """
    document_id: str = field(default_factory=_generate_id)
    title: str = ""
    content: str = ""
    source_kind: str = "file"  # file | url | text | code
    workspace_id: str = "global"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "source_kind": self.source_kind,
            "workspace_id": self.workspace_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentIngestRequest":
        """Создаёт из словаря."""
        return cls(
            document_id=data.get("document_id", _generate_id()),
            title=data.get("title", ""),
            content=data.get("content", ""),
            source_kind=data.get("source_kind", "file"),
            workspace_id=data.get("workspace_id", "global"),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class DocumentIngestResult:
    """
    Результат ingest документа.
    """
    document_id: str = ""
    chunks_created: int = 0
    summary_created: bool = False
    artifact_ids: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "document_id": self.document_id,
            "chunks_created": self.chunks_created,
            "summary_created": self.summary_created,
            "artifact_ids": self.artifact_ids,
        }


@dataclass(slots=True)
class MemoryInspectRequest:
    """
    Запрос на инспекцию памяти (для debug).
    """
    kind: str = "events"  # events | artifacts | trace | workspaces | profile
    event_id: str | None = None
    artifact_id: str | None = None
    workspace_id: str | None = None
    limit: int = 50
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "kind": self.kind,
            "event_id": self.event_id,
            "artifact_id": self.artifact_id,
            "workspace_id": self.workspace_id,
            "limit": self.limit,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryInspectRequest":
        """Создаёт из словаря."""
        return cls(
            kind=data.get("kind", "events"),
            event_id=data.get("event_id"),
            artifact_id=data.get("artifact_id"),
            workspace_id=data.get("workspace_id"),
            limit=data.get("limit", 50),
        )


@dataclass(slots=True)
class MemoryTrace:
    """
    Трассировка события: raw event -> порождённые artifacts.
    """
    event: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    indexed: bool = False
    retrieved: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "event": self.event,
            "artifacts": self.artifacts,
            "indexed": self.indexed,
            "retrieved": self.retrieved,
        }
