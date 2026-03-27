"""
PersonaContextBuilder — строитель контекста для формирования личности ассистента.

Pipeline:
memory_core.query()
 -> persona_context_builder
 -> persona snapshot
 -> prompt builder
 -> main LLM

Persona snapshot включает:
- кто пользователь
- кто ассистент
- стиль общения
- текущая задача
- текущий эпизод
- эмоциональный фон
- важные факты
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.event_store import EventStore
from memory_core.schemas import MemoryQuery
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class PersonaSnapshot:
    """
    Снимок контекста персоны для промпта.
    """
    # Пользователь
    user_name: str | None = None
    user_profile_facts: list[str] = field(default_factory=list)
    user_preferences: list[str] = field(default_factory=list)

    # Ассистент
    assistant_name: str | None = None
    assistant_identity: list[str] = field(default_factory=list)
    assistant_style: str = "friendly"  # friendly | formal | concise | detailed

    # Контекст
    current_task: str | None = None
    current_episode: str | None = None
    emotional_state: str | None = None

    # Факты
    important_facts: list[str] = field(default_factory=list)
    recent_events: list[str] = field(default_factory=list)

    # Мета
    workspace_id: str = "global"
    session_id: str = "default"
    retrieved_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "user_name": self.user_name,
            "user_profile_facts": self.user_profile_facts,
            "user_preferences": self.user_preferences,
            "assistant_name": self.assistant_name,
            "assistant_identity": self.assistant_identity,
            "assistant_style": self.assistant_style,
            "current_task": self.current_task,
            "current_episode": self.current_episode,
            "emotional_state": self.emotional_state,
            "important_facts": self.important_facts,
            "recent_events": self.recent_events,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "retrieved_at": self.retrieved_at,
        }

    def to_prompt(self) -> str:
        """
        Формирует текстовый промпт для LLM.

        Returns:
            Текстовый контекст.
        """
        blocks = []

        # Пользователь
        if self.user_name or self.user_profile_facts:
            user_block = "### Пользователь\n"
            if self.user_name:
                user_block += f"Имя: {self.user_name}\n"
            if self.user_profile_facts:
                user_block += "Факты:\n"
                for fact in self.user_profile_facts[:5]:
                    user_block += f"- {fact}\n"
            if self.user_preferences:
                user_block += "Предпочтения:\n"
                for pref in self.user_preferences[:5]:
                    user_block += f"- {pref}\n"
            blocks.append(user_block)

        # Ассистент
        if self.assistant_name or self.assistant_identity:
            assistant_block = "### Ассистент\n"
            if self.assistant_name:
                assistant_block += f"Имя: {self.assistant_name}\n"
            if self.assistant_identity:
                assistant_block += "Идентичность:\n"
                for identity in self.assistant_identity[:5]:
                    assistant_block += f"- {identity}\n"
            assistant_block += f"Стиль: {self.assistant_style}\n"
            blocks.append(assistant_block)

        # Текущий контекст
        context_block = "### Текущий контекст\n"
        if self.current_task:
            context_block += f"Задача: {self.current_task}\n"
        if self.current_episode:
            context_block += f"Эпизод: {self.current_episode}\n"
        if self.emotional_state:
            context_block += f"Эмоциональное состояние: {self.emotional_state}\n"
        if blocks or self.current_task or self.current_episode:
            blocks.append(context_block)

        # Важные факты
        if self.important_facts:
            facts_block = "### Важные факты\n"
            for fact in self.important_facts[:10]:
                facts_block += f"- {fact}\n"
            blocks.append(facts_block)

        # Недавние события
        if self.recent_events:
            events_block = "### Недавние события\n"
            for event in self.recent_events[:5]:
                events_block += f"- {event}\n"
            blocks.append(events_block)

        return "\n".join(blocks) if blocks else ""


class PersonaContextBuilder:
    """
    Строитель контекста персоны.

    Извлекает из памяти:
    - Identity Core (ядро личности)
    - Profile Facts (факты о пользователе)
    - Preferences (предпочтения)
    - Current Task (текущая задача)
    - Current Episode (текущий эпизод)
    - Emotional State (эмоциональное состояние)
    - Recent Events (недавние события)
    """

    # Приоритеты извлечения
    ARTIFACT_TYPE_IDENTITY = "identity_core"
    ARTIFACT_TYPE_PROFILE = "profile_fact"
    ARTIFACT_TYPE_PREFERENCE = "preference"
    ARTIFACT_TYPE_TASK = "task_state"
    ARTIFACT_TYPE_EPISODE = "episode_event"
    ARTIFACT_TYPE_EMOTION = "emotional_state"

    # Лимиты
    MAX_PROFILE_FACTS = 10
    MAX_PREFERENCES = 10
    MAX_IDENTITY_FACTS = 10
    MAX_RECENT_EVENTS = 10

    def __init__(
        self,
        artifact_store: ArtifactStore,
        event_store: EventStore,
    ):
        """
        Инициализирует строитель.

        Args:
            artifact_store: Хранилище артефактов.
            event_store: Хранилище событий.
        """
        self.artifact_store = artifact_store
        self.event_store = event_store

    def build(
        self,
        workspace_id: str = "global",
        session_id: str = "default",
        query_text: str | None = None,
    ) -> PersonaSnapshot:
        """
        Строит снимок контекста.

        Args:
            workspace_id: ID workspace.
            session_id: ID сессии.
            query_text: Текст запроса (для релевантного retrieval).

        Returns:
            PersonaSnapshot.
        """
        snapshot = PersonaSnapshot(
            workspace_id=workspace_id,
            session_id=session_id,
        )

        # Извлекаем Identity Core (всегда)
        self._extract_identity(snapshot, workspace_id)

        # Извлекаем Profile Facts
        self._extract_profile_facts(snapshot, workspace_id)

        # Извлекаем Preferences
        self._extract_preferences(snapshot, workspace_id)

        # Извлекаем Current Task
        self._extract_current_task(snapshot, workspace_id)

        # Извлекаем Current Episode
        self._extract_current_episode(snapshot, session_id)

        # Извлекаем Emotional State (если релевантно)
        self._extract_emotional_state(snapshot, session_id)

        # Извлекаем Recent Events
        self._extract_recent_events(snapshot, session_id)

        # Если есть query_text → добавляем релевантные факты
        if query_text:
            self._extract_relevant_facts(snapshot, query_text, workspace_id)

        return snapshot

    def _extract_identity(
        self,
        snapshot: PersonaSnapshot,
        workspace_id: str,
    ) -> None:
        """Извлекает Identity Core."""
        artifacts = self.artifact_store.list_artifacts(
            artifact_type=self.ARTIFACT_TYPE_IDENTITY,
            workspace_id=workspace_id,
            status="active",
            limit=self.MAX_IDENTITY_FACTS,
        )

        for artifact in artifacts:
            text = artifact.text
            metadata = artifact.metadata or {}

            # Определяем тип информации
            if "имя" in text.lower() or "name" in text.lower():
                if "пользователь" in text.lower():
                    # Извлекаем имя пользователя
                    snapshot.user_name = text.split(":")[-1].strip() if ":" in text else text
                elif "ассистент" in text.lower() or "assistant" in text.lower():
                    snapshot.assistant_name = text.split(":")[-1].strip() if ":" in text else text
            elif "стиль" in text.lower() or "style" in text.lower():
                if "friendly" in text.lower() or "дружелюбный" in text.lower():
                    snapshot.assistant_style = "friendly"
                elif "formal" in text.lower() or "формальный" in text.lower():
                    snapshot.assistant_style = "formal"
                elif "concise" in text.lower() or "краткий" in text.lower():
                    snapshot.assistant_style = "concise"
                elif "detailed" in text.lower() or "подробный" in text.lower():
                    snapshot.assistant_style = "detailed"
            else:
                snapshot.assistant_identity.append(text)

    def _extract_profile_facts(
        self,
        snapshot: PersonaSnapshot,
        workspace_id: str,
    ) -> None:
        """Извлекает факты о пользователе."""
        artifacts = self.artifact_store.list_artifacts(
            artifact_type=self.ARTIFACT_TYPE_PROFILE,
            workspace_id=workspace_id,
            status="active",
            limit=self.MAX_PROFILE_FACTS,
        )

        # Сортируем по confidence (в metadata)
        artifacts_sorted = sorted(
            artifacts,
            key=lambda a: float(a.metadata.get("confidence", 0.5)),
            reverse=True,
        )

        for artifact in artifacts_sorted[:self.MAX_PROFILE_FACTS]:
            snapshot.user_profile_facts.append(artifact.text)

    def _extract_preferences(
        self,
        snapshot: PersonaSnapshot,
        workspace_id: str,
    ) -> None:
        """Извлекает предпочтения."""
        artifacts = self.artifact_store.list_artifacts(
            artifact_type=self.ARTIFACT_TYPE_PREFERENCE,
            workspace_id=workspace_id,
            status="active",
            limit=self.MAX_PREFERENCES,
        )

        for artifact in artifacts:
            snapshot.user_preferences.append(artifact.text)

    def _extract_current_task(
        self,
        snapshot: PersonaSnapshot,
        workspace_id: str,
    ) -> None:
        """Извлекает текущую задачу."""
        artifacts = self.artifact_store.list_artifacts(
            artifact_type=self.ARTIFACT_TYPE_TASK,
            workspace_id=workspace_id,
            status="active",
            limit=1,
        )

        if artifacts:
            # Берём самую свежую
            latest = max(artifacts, key=lambda a: a.updated_at)
            snapshot.current_task = latest.text

    def _extract_current_episode(
        self,
        snapshot: PersonaSnapshot,
        session_id: str,
    ) -> None:
        """Извлекает текущий эпизод."""
        artifacts = self.artifact_store.list_artifacts(
            artifact_type=self.ARTIFACT_TYPE_EPISODE,
            workspace_id=snapshot.workspace_id,
            status="active",
            limit=5,
        )

        # Фильтруем по session_id (в metadata)
        session_artifacts = [
            a for a in artifacts
            if a.metadata.get("session_id") == session_id
        ]

        if session_artifacts:
            latest = max(session_artifacts, key=lambda a: a.updated_at)
            snapshot.current_episode = latest.text

    def _extract_emotional_state(
        self,
        snapshot: PersonaSnapshot,
        session_id: str,
    ) -> None:
        """Извлекает эмоциональное состояние."""
        artifacts = self.artifact_store.list_artifacts(
            artifact_type=self.ARTIFACT_TYPE_EMOTION,
            workspace_id=snapshot.workspace_id,
            status="active",
            limit=5,
        )

        # Фильтруем по session_id и decay
        session_artifacts = [
            a for a in artifacts
            if a.metadata.get("session_id") == session_id
            and a.metadata.get("decay") != "immediate"
        ]

        if session_artifacts:
            latest = max(session_artifacts, key=lambda a: a.updated_at)
            # Проверяем, не устарело ли (эмоции быстро устаревают)
            if time.time() - latest.updated_at < 3600:  # 1 час
                snapshot.emotional_state = latest.text

    def _extract_recent_events(
        self,
        snapshot: PersonaSnapshot,
        session_id: str,
    ) -> None:
        """Извлекает недавние события."""
        events = self.event_store.list_events(
            session_id=session_id,
            limit=self.MAX_RECENT_EVENTS,
        )

        for event in events:
            text = str(event.text or "")
            if text:
                snapshot.recent_events.append(f"[{event.source_kind or '?'}] {text}")

    def _extract_relevant_facts(
        self,
        snapshot: PersonaSnapshot,
        query_text: str,
        workspace_id: str,
    ) -> None:
        """
        Извлекает релевантные факты по запросу.

        Args:
            snapshot: Снимок для заполнения.
            query_text: Текст запроса.
            workspace_id: ID workspace.
        """
        # Здесь можно использовать векторный поиск
        # Для пока просто заглушка
        pass


def build_persona_context_builder(
    artifact_store: ArtifactStore,
    event_store: EventStore,
) -> PersonaContextBuilder:
    """
    Строит PersonaContextBuilder.

    Args:
        artifact_store: Хранилище артефактов.
        event_store: Хранилище событий.

    Returns:
        PersonaContextBuilder.
    """
    return PersonaContextBuilder(
        artifact_store=artifact_store,
        event_store=event_store,
    )
