"""
Memory Trace Contract — единый контракт для trace.

Определяет структуру trace row, которую понимают:
- response_pipeline
- BackgroundWorker
- inspector_service
- UI panels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryTraceContract:
    """
    Контракт для trace row.

    Используется для:
    - persisted turn trace
    - worker trace
    - event trace
    - inspector API
    """

    # Идентификаторы
    trace_id: str = ""
    request_id: str = ""
    conversation_id: str = ""
    turn_id: int = 0
    created_at: str = ""
    route: str = ""

    # Пользовательский ввод
    user_text: str = ""

    # Pipeline trace (из response_pipeline)
    pipeline: dict[str, Any] = field(default_factory=dict)
    """
    pipeline = {
        "memory_retrieval": {...},
        "memory_governor": {...},
        "identity_core": {...},
        "persona_snapshot": {...},
        "active_task": {...},
        "prompt_pack": {...},
        "final_answer_meta": {...},
        "turn_log_summaries": {...},
        "turn_log_warnings": [...],
    }
    """

    # Memory trace (из worker/ingest)
    memory: dict[str, Any] = field(default_factory=dict)
    """
    memory = {
        "event_ids": [...],
        "job_ids": [...],
        "artifact_ids_created": [...],
        "artifact_ids_updated": [...],
        "artifact_ids_superseded": [...],
        "indexed_ids": [...],
    }
    """

    # Ссылки на связанные trace
    links: dict[str, Any] = field(default_factory=dict)
    """
    links = {
        "web_trace_detail_file_path": "...",
        "web_trace_compact_file_path": "...",
    }
    """

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
            "route": self.route,
            "user_text": self.user_text,
            "pipeline": self.pipeline,
            "memory": self.memory,
            "links": self.links,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryTraceContract":
        """Создаёт из словаря."""
        return cls(
            trace_id=str(data.get("trace_id", "")),
            request_id=str(data.get("request_id", "")),
            conversation_id=str(data.get("conversation_id", "")),
            turn_id=int(data.get("turn_id", 0)),
            created_at=str(data.get("created_at", "")),
            route=str(data.get("route", "")),
            user_text=str(data.get("user_text", "")),
            pipeline=dict(data.get("pipeline", {})),
            memory=dict(data.get("memory", {})),
            links=dict(data.get("links", {})),
        )


@dataclass(slots=True)
class WorkerTraceContract:
    """
    Контракт для worker trace row.

    Используется для:
    - worker stage trace
    - job processing trace
    - governor decisions trace
    """

    trace_id: str = ""
    stage: str = ""
    created_at: str = ""

    # Job данные
    job_id: str = ""
    event_id: str = ""
    job_type: str = ""
    status: str = ""

    # Processor данные
    proposal_count: int = 0
    proposals: list[dict[str, Any]] = field(default_factory=list)

    # Governor данные
    decisions: list[dict[str, Any]] = field(default_factory=list)
    created_ids: list[str] = field(default_factory=list)
    updated_ids: list[str] = field(default_factory=list)
    superseded_ids: list[str] = field(default_factory=list)

    # Vector index данные
    indexed_ids: list[str] = field(default_factory=list)

    # Ошибки
    error: str = ""
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "trace_id": self.trace_id,
            "stage": self.stage,
            "created_at": self.created_at,
            "job_id": self.job_id,
            "event_id": self.event_id,
            "job_type": self.job_type,
            "status": self.status,
            "proposal_count": self.proposal_count,
            "proposals": self.proposals,
            "decisions": self.decisions,
            "created_ids": self.created_ids,
            "updated_ids": self.updated_ids,
            "superseded_ids": self.superseded_ids,
            "indexed_ids": self.indexed_ids,
            "error": self.error,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerTraceContract":
        """Создаёт из словаря."""
        return cls(
            trace_id=str(data.get("trace_id", "")),
            stage=str(data.get("stage", "")),
            created_at=str(data.get("created_at", "")),
            job_id=str(data.get("job_id", "")),
            event_id=str(data.get("event_id", "")),
            job_type=str(data.get("job_type", "")),
            status=str(data.get("status", "")),
            proposal_count=int(data.get("proposal_count", 0)),
            proposals=list(data.get("proposals", [])),
            decisions=list(data.get("decisions", [])),
            created_ids=list(data.get("created_ids", [])),
            updated_ids=list(data.get("updated_ids", [])),
            superseded_ids=list(data.get("superseded_ids", [])),
            indexed_ids=list(data.get("indexed_ids", [])),
            error=str(data.get("error", "")),
            attempts=int(data.get("attempts", 0)),
        )
