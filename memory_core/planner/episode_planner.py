"""
Episode Planner — планирование и управление эпизодами.

Эпизод — это контекстуально связанная последовательность событий:
- одна тема разговора
- одна задача
- одна сессия работы

Episode Planner:
- определяет начало нового эпизода
- отслеживает контекст эпизода
- определяет конец эпизода
- создаёт episode artifacts
- управляет retrieval в рамках эпизода
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from memory_core.schemas import MemoryArtifact, MemoryEnvelope
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.event_store import EventStore
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class Episode:
    """
    Эпизод — контекстуально связанная последовательность событий.
    """
    episode_id: str
    session_id: str
    workspace_id: str
    title: str = ""
    summary: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    status: str = "active"  # active | completed | archived
    context_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "title": self.title,
            "summary": self.summary,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "context_tags": self.context_tags,
            "metadata": self.metadata,
        }

    def to_artifact(self) -> MemoryArtifact:
        """
        Преобразует в артефакт.

        Returns:
            MemoryArtifact.
        """
        return MemoryArtifact(
            artifact_type="episode_event",
            source_event_id="episode_planner",
            text=self.title or f"Episode {self.episode_id[:8]}",
            summary=self.summary,
            metadata={
                "episode_id": self.episode_id,
                "session_id": self.session_id,
                "workspace_id": self.workspace_id,
                "context_tags": self.context_tags,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "status": self.status,
            },
            namespace="episodes",
            workspace_id=self.workspace_id,
            status="active" if self.status == "active" else "archived",
            created_at=self.started_at,
            updated_at=self.updated_at,
        )


@dataclass(slots=True)
class EpisodeContext:
    """
    Контекст текущего эпизода.
    """
    episode: Episode
    event_count: int = 0
    last_event_at: float = field(default_factory=time.time)
    topic_keywords: list[str] = field(default_factory=list)
    task_description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "episode": self.episode.to_dict(),
            "event_count": self.event_count,
            "last_event_at": self.last_event_at,
            "topic_keywords": self.topic_keywords,
            "task_description": self.task_description,
        }


class EpisodePlanner:
    """
    Планировщик эпизодов.

    Управляет жизненным циклом эпизодов:
    - создание нового эпизода
    - обновление контекста
    - завершение эпизода
    - архивация
    """

    # Пороги для определения нового эпизода
    TOPIC_CHANGE_THRESHOLD = 0.6  # Порог смены темы
    INACTIVITY_THRESHOLD_SEC = 1800.0  # 30 минут неактивности

    # Максимальное количество активных эпизодов
    MAX_ACTIVE_EPISODES = 10

    # Максимальная длительность эпизода (3 часа)
    MAX_EPISODE_DURATION_SEC = 10800.0

    def __init__(
        self,
        artifact_store: ArtifactStore,
        event_store: EventStore,
    ):
        """
        Инициализирует планировщик.

        Args:
            artifact_store: Хранилище артефактов.
            event_store: Хранилище событий.
        """
        self.artifact_store = artifact_store
        self.event_store = event_store

        # Кэш активных эпизодов
        self._active_episodes: dict[str, EpisodeContext] = {}

    def get_or_create_episode(
        self,
        session_id: str,
        workspace_id: str = "global",
        envelope: MemoryEnvelope | None = None,
    ) -> Episode:
        """
        Получает или создаёт эпизод.

        Args:
            session_id: ID сессии.
            workspace_id: ID workspace.
            envelope: Конверт события.

        Returns:
            Активный эпизод.
        """
        # Проверяем кэш
        cache_key = f"{session_id}:{workspace_id}"
        if cache_key in self._active_episodes:
            ctx = self._active_episodes[cache_key]
            # Проверяем, не устарел ли эпизод
            if self._is_episode_stale(ctx):
                self._close_episode(ctx.episode)
                del self._active_episodes[cache_key]
            else:
                return ctx.episode

        # Ищем активный эпизод в хранилище
        episode = self._find_active_episode(session_id, workspace_id)
        if episode:
            ctx = EpisodeContext(episode=episode)
            self._active_episodes[cache_key] = ctx
            return episode

        # Создаём новый эпизод
        episode = self._create_episode(session_id, workspace_id, envelope)
        ctx = EpisodeContext(episode=episode)
        self._active_episodes[cache_key] = ctx

        return episode

    def _find_active_episode(
        self,
        session_id: str,
        workspace_id: str,
    ) -> Episode | None:
        """
        Ищет активный эпизод.

        Args:
            session_id: ID сессии.
            workspace_id: ID workspace.

        Returns:
            Эпизод или None.
        """
        artifacts = self.artifact_store.list_artifacts(
            artifact_type="episode_event",
            workspace_id=workspace_id,
            status="active",
            limit=20,
        )

        # Фильтруем по session_id
        for artifact in artifacts:
            meta = artifact.metadata or {}
            if meta.get("session_id") == session_id:
                # Проверяем, не устарел ли
                if time.time() - artifact.updated_at < self.INACTIVITY_THRESHOLD_SEC:
                    return Episode(
                        episode_id=meta.get("episode_id", artifact.artifact_id),
                        session_id=session_id,
                        workspace_id=workspace_id,
                        title=artifact.text,
                        summary=artifact.summary,
                        started_at=meta.get("started_at", artifact.created_at),
                        updated_at=artifact.updated_at,
                        status="active",
                        context_tags=meta.get("context_tags", []),
                        metadata=meta,
                    )

        return None

    def _create_episode(
        self,
        session_id: str,
        workspace_id: str,
        envelope: MemoryEnvelope | None = None,
    ) -> Episode:
        """
        Создаёт новый эпизод.

        Args:
            session_id: ID сессии.
            workspace_id: ID workspace.
            envelope: Конверт события.

        Returns:
            Новый эпизод.
        """
        import uuid
        now = time.time()

        episode = Episode(
            episode_id=f"ep_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            workspace_id=workspace_id,
            title=f"Episode {now:.0f}",
            summary="",
            started_at=now,
            updated_at=now,
            status="active",
        )

        # Извлекаем контекст из envelope
        if envelope:
            episode.title = self._extract_episode_title(envelope)
            episode.context_tags = self._extract_context_tags(envelope)

        # Сохраняем как артефакт
        self._save_episode(episode)

        LOGGER.info(f"Created new episode: {episode.episode_id}")
        return episode

    def _extract_episode_title(self, envelope: MemoryEnvelope) -> str:
        """
        Извлекает заголовок эпизода из события.

        Args:
            envelope: Конверт события.

        Returns:
            Заголовок эпизода.
        """
        # Простая эвристика: первые 50 символов текста
        text = envelope.text.strip()
        if len(text) > 50:
            text = text[:50] + "..."
        return text or f"Episode {envelope.event_id[:8]}"

    def _extract_context_tags(self, envelope: MemoryEnvelope) -> list[str]:
        """
        Извлекает теги контекста из события.

        Args:
            envelope: Конверт события.

        Returns:
            Список тегов.
        """
        tags = []

        # Извлекаем из metadata
        metadata = envelope.metadata or {}
        if "tags" in metadata:
            tags.extend(metadata["tags"])
        if "topic" in metadata:
            tags.append(metadata["topic"])
        if "intent" in metadata:
            tags.append(metadata["intent"])

        # Извлекаем из текста (простая эвристика)
        text_lower = envelope.text.lower()
        if "код" in text_lower or "code" in text_lower:
            tags.append("coding")
        if "баг" in text_lower or "bug" in text_lower:
            tags.append("debug")
        if "вопрос" in text_lower or "question" in text_lower:
            tags.append("question")

        return list(set(tags))

    def _save_episode(self, episode: Episode) -> None:
        """
        Сохраняет эпизод как артефакт.

        Args:
            episode: Эпизод.
        """
        artifact = episode.to_artifact()
        
        # Проверяем, существует ли уже артефакт
        existing = self.artifact_store.get_by_id(artifact.artifact_id)
        if existing:
            self.artifact_store.update(artifact)
        else:
            self.artifact_store.create(artifact)

    def _close_episode(self, episode: Episode) -> None:
        """
        Закрывает эпизод.

        Args:
            episode: Эпизод.
        """
        now = time.time()
        episode.ended_at = now
        episode.status = "completed"
        episode.updated_at = now

        # Обновляем артефакт
        artifacts = self.artifact_store.list_artifacts(
            artifact_type="episode_event",
            workspace_id=episode.workspace_id,
            status="active",
            limit=50,
        )

        for artifact in artifacts:
            meta = artifact.metadata or {}
            if meta.get("episode_id") == episode.episode_id:
                artifact.status = "archived"
                artifact.metadata["ended_at"] = now
                artifact.metadata["status"] = "completed"
                artifact.updated_at = now
                self.artifact_store.update(artifact)
                break

        LOGGER.info(f"Closed episode: {episode.episode_id}")

    def _is_episode_stale(self, ctx: EpisodeContext) -> bool:
        """
        Проверяет, устарел ли эпизод.

        Args:
            ctx: Контекст эпизода.

        Returns:
            True если эпизод устарел.
        """
        now = time.time()

        # Проверка по неактивности
        if now - ctx.last_event_at > self.INACTIVITY_THRESHOLD_SEC:
            return True

        # Проверка по длительности
        if now - ctx.episode.started_at > self.MAX_EPISODE_DURATION_SEC:
            return True

        return False

    def update_episode_context(
        self,
        episode: Episode,
        envelope: MemoryEnvelope,
    ) -> None:
        """
        Обновляет контекст эпизода.

        Args:
            episode: Эпизод.
            envelope: Конверт события.
        """
        cache_key = f"{episode.session_id}:{episode.workspace_id}"

        if cache_key not in self._active_episodes:
            self._active_episodes[cache_key] = EpisodeContext(episode=episode)

        ctx = self._active_episodes[cache_key]
        ctx.event_count += 1
        ctx.last_event_at = time.time()

        # Обновляем теги
        new_tags = self._extract_context_tags(envelope)
        for tag in new_tags:
            if tag not in ctx.topic_keywords:
                ctx.topic_keywords.append(tag)

        # Обновляем эпизод
        episode.updated_at = ctx.last_event_at
        episode.context_tags = ctx.topic_keywords

        # Периодически сохраняем
        if ctx.event_count % 10 == 0:
            self._save_episode(episode)

    def get_episode_summary(self, episode_id: str) -> dict[str, Any]:
        """
        Получает сводку эпизода.

        Args:
            episode_id: ID эпизода.

        Returns:
            Сводка эпизода.
        """
        # Ищем артефакт эпизода
        artifacts = self.artifact_store.list_artifacts(limit=200)
        episode_artifact = None

        for artifact in artifacts:
            meta = artifact.metadata or {}
            if meta.get("episode_id") == episode_id:
                episode_artifact = artifact
                break

        if not episode_artifact:
            return {"error": f"Episode {episode_id} not found"}

        # Считаем события эпизода по metadata
        all_artifacts = self.artifact_store.list_artifacts(limit=500)
        event_count = sum(
            1 for a in all_artifacts
            if a.metadata.get("episode_id") == episode_id
        )

        # Считаем длительность
        started_at = episode_artifact.metadata.get("started_at", episode_artifact.created_at)
        ended_at = episode_artifact.metadata.get("ended_at") or episode_artifact.updated_at
        duration_sec = ended_at - started_at if ended_at and started_at else None

        return {
            "episode": episode_artifact.to_dict(),
            "event_count": event_count,
            "duration_sec": duration_sec,
            "started_at": started_at,
            "ended_at": ended_at,
        }

    def get_active_episodes(
        self,
        session_id: str | None = None,
    ) -> list[Episode]:
        """
        Получает активные эпизоды.

        Args:
            session_id: ID сессии (опционально).

        Returns:
            Список активных эпизодов.
        """
        episodes = []

        for ctx in self._active_episodes.values():
            if session_id is None or ctx.episode.session_id == session_id:
                episodes.append(ctx.episode)

        return episodes

    def cleanup_stale_episodes(self) -> int:
        """
        Очищает устаревшие эпизоды.

        Returns:
            Количество закрытых эпизодов.
        """
        closed_count = 0
        stale_keys = []

        for key, ctx in self._active_episodes.items():
            if self._is_episode_stale(ctx):
                self._close_episode(ctx.episode)
                stale_keys.append(key)
                closed_count += 1

        for key in stale_keys:
            del self._active_episodes[key]

        return closed_count


def build_episode_planner(
    artifact_store: ArtifactStore,
    event_store: EventStore,
) -> EpisodePlanner:
    """
    Строит EpisodePlanner.

    Args:
        artifact_store: Хранилище артефактов.
        event_store: Хранилище событий.

    Returns:
        EpisodePlanner.
    """
    return EpisodePlanner(
        artifact_store=artifact_store,
        event_store=event_store,
    )
