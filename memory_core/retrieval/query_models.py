"""
Модели для retrieval запросов.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalFilters:
    """
    Фильтры для retrieval запроса.
    """
    artifact_types: list[str] = field(default_factory=list)
    workspace_id: str | None = None
    namespace: str | None = None
    status: str = "active"
    source_event_id: str | None = None
    min_confidence: float | None = None
    max_age_days: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "artifact_types": self.artifact_types,
            "workspace_id": self.workspace_id,
            "namespace": self.namespace,
            "status": self.status,
            "source_event_id": self.source_event_id,
            "min_confidence": self.min_confidence,
            "max_age_days": self.max_age_days,
        }


@dataclass(slots=True)
class ContextPack:
    """
    Пакет контекста для LLM.
    
    Содержит все блоки памяти, которые будут переданы в LLM.
    """
    profile_facts: list[str] = field(default_factory=list)
    active_tasks: list[str] = field(default_factory=list)
    recent_episodes: list[str] = field(default_factory=list)
    relevant_facts: list[str] = field(default_factory=list)
    document_chunks: list[str] = field(default_factory=list)
    workspace_info: str | None = None
    tone_hints: list[str] = field(default_factory=list)
    continuity_hints: list[str] = field(default_factory=list)
    answer_support: list[str] = field(default_factory=list)
    exact_recall: list[str] = field(default_factory=list)
    blocks: dict[str, str] = field(default_factory=dict)
    selected_memories: list[dict[str, Any]] = field(default_factory=list)
    dropped_memories: list[dict[str, Any]] = field(default_factory=list)
    recent_user_state: dict[str, Any] = field(default_factory=dict)
    response_bias: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
    
    def to_context_blocks(self) -> list[str]:
        """Преобразует в список текстовых блоков."""
        blocks = []
        
        if self.workspace_info:
            blocks.append(f"## Workspace\n{self.workspace_info}")
        
        if self.profile_facts:
            blocks.append(f"## Profile Facts\n" + "\n".join(f"- {f}" for f in self.profile_facts))
        
        if self.active_tasks:
            blocks.append(f"## Active Tasks\n" + "\n".join(f"- {t}" for t in self.active_tasks))
        
        if self.recent_episodes:
            blocks.append(f"## Recent Episodes\n" + "\n".join(f"- {e}" for e in self.recent_episodes))
        
        if self.relevant_facts:
            blocks.append(f"## Relevant Facts\n" + "\n".join(f"- {f}" for f in self.relevant_facts))
        
        if self.document_chunks:
            blocks.append(f"## Document Context\n" + "\n".join(self.document_chunks))

        if self.exact_recall:
            blocks.append(f"## Exact Recall\n" + "\n".join(f"- {f}" for f in self.exact_recall))

        if self.answer_support:
            blocks.append(f"## Answer Support\n" + "\n".join(f"- {f}" for f in self.answer_support))

        if self.continuity_hints:
            blocks.append(f"## Continuity Hints\n" + "\n".join(f"- {f}" for f in self.continuity_hints))

        if self.tone_hints:
            blocks.append(f"## Tone Hints\n" + "\n".join(f"- {f}" for f in self.tone_hints))
        
        return blocks
    
    def is_empty(self) -> bool:
        """Проверяет, пуст ли контекст."""
        return (
            not self.profile_facts
            and not self.active_tasks
            and not self.recent_episodes
            and not self.relevant_facts
            and not self.document_chunks
            and not self.tone_hints
            and not self.continuity_hints
            and not self.answer_support
            and not self.exact_recall
            and not self.blocks
            and not self.workspace_info
        )


@dataclass(slots=True)
class Citation:
    """
    Цитата для debug/tracing.
    
    Показывает источник информации в контексте.
    """
    artifact_id: str
    artifact_type: str
    source_event_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "source_event_id": self.source_event_id,
            "text": self.text,
            "metadata": self.metadata,
        }
    
    def to_short_ref(self) -> str:
        """Создаёт короткую ссылку для отображения."""
        return f"[{self.artifact_type}:{self.artifact_id[:8]}]"
