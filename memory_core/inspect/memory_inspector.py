"""
Memory Inspector UI — интерфейс для инспекции и отладки памяти.

Поддерживает различные виды инспекции:
- events: список сырых событий
- artifacts: список артефактов
- workspaces: список workspace
- profile: профильные факты
- stats: статистика
- trace: трассировка события
- jobs: очередь задач
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from memory_core.storage.event_store import EventStore
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.workspace_store import WorkspaceStore
from memory_core.storage.job_queue_store import JobQueueStore
from memory_core.identity.identity_core import IdentityCore
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class InspectorStats:
    """
    Статистика памяти.
    """
    events_count: int = 0
    artifacts_count: int = 0
    artifacts_by_type: dict[str, int] = field(default_factory=dict)
    workspaces_count: int = 0
    jobs_queued: int = 0
    jobs_processing: int = 0
    jobs_done: int = 0
    jobs_dead: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "events_count": self.events_count,
            "artifacts_count": self.artifacts_count,
            "artifacts_by_type": self.artifacts_by_type,
            "workspaces_count": self.workspaces_count,
            "jobs_queued": self.jobs_queued,
            "jobs_processing": self.jobs_processing,
            "jobs_done": self.jobs_done,
            "jobs_dead": self.jobs_dead,
        }


class MemoryInspector:
    """
    Инспектор памяти для отладки.

    Предоставляет методы для инспекции различных аспектов
    системы памяти.
    """

    def __init__(
        self,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        workspace_store: WorkspaceStore,
        job_queue: JobQueueStore,
        identity_core: IdentityCore | None = None,
    ):
        """
        Инициализирует инспектор.

        Args:
            event_store: Хранилище событий.
            artifact_store: Хранилище артефактов.
            workspace_store: Хранилище workspace.
            job_queue: Очередь задач.
            identity_core: Identity Core (опционально).
        """
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.workspace_store = workspace_store
        self.job_queue = job_queue
        self.identity_core = identity_core

    def inspect(
        self,
        kind: str = "events",
        limit: int = 50,
        workspace_id: str | None = None,
        event_id: str | None = None,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Инспектирует память.

        Args:
            kind: Тип инспекции (events | artifacts | workspaces | profile | stats | trace | jobs).
            limit: Максимальное количество результатов.
            workspace_id: ID workspace для фильтрации.
            event_id: ID события для trace.
            artifact_id: ID артефакта для trace.

        Returns:
            Результат инспекции.
        """
        if kind == "events":
            return self._inspect_events(limit, workspace_id)
        elif kind == "artifacts":
            return self._inspect_artifacts(limit, workspace_id)
        elif kind == "workspaces":
            return self._inspect_workspaces(limit)
        elif kind == "profile":
            return self._inspect_profile(workspace_id)
        elif kind == "stats":
            return self._inspect_stats()
        elif kind == "trace" and event_id:
            return self._inspect_trace(event_id)
        elif kind == "jobs":
            return self._inspect_jobs(limit)
        else:
            return {"error": f"Unknown inspect kind: {kind}"}

    def _inspect_events(
        self,
        limit: int = 50,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Инспектирует события."""
        if workspace_id:
            events = self.event_store.get_by_workspace(workspace_id, limit=limit)
        else:
            events = self.event_store.get_recent(limit=limit)

        return {
            "items": [dict(e) for e in events],
            "count": len(events),
            "limit": limit,
        }

    def _inspect_artifacts(
        self,
        limit: int = 50,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Инспектирует артефакты."""
        if workspace_id:
            artifacts = self.artifact_store.get_by_workspace(workspace_id, limit=limit)
        else:
            artifacts = self.artifact_store.get_all(limit=limit)

        return {
            "items": [a.to_dict() for a in artifacts],
            "count": len(artifacts),
            "limit": limit,
        }

    def _inspect_workspaces(self, limit: int = 50) -> dict[str, Any]:
        """Инспектирует workspace."""
        workspaces = self.workspace_store.get_all(limit=limit)

        return {
            "workspaces": [dict(w) for w in workspaces],
            "count": len(workspaces),
            "limit": limit,
        }

    def _inspect_profile(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Инспектирует профиль (Identity Core + Profile Facts)."""
        ws_id = workspace_id or "global"

        result = {
            "workspace_id": ws_id,
        }

        # Identity Core
        if self.identity_core:
            profile = self.identity_core.get_profile(ws_id)
            result["identity"] = profile.to_dict()
            result["identity_artifacts"] = [
                a.to_dict() for a in self.identity_core.get_artifacts(ws_id)
            ]

        # Profile Facts
        profile_facts = self.artifact_store.get_by_type(
            artifact_type="profile_fact",
            workspace_id=ws_id,
            status="active",
            limit=20,
        )
        result["profile_facts"] = [a.to_dict() for a in profile_facts]

        # Preferences
        preferences = self.artifact_store.get_by_type(
            artifact_type="preference",
            workspace_id=ws_id,
            status="active",
            limit=20,
        )
        result["preferences"] = [a.to_dict() for a in preferences]

        return result

    def _inspect_stats(self) -> dict[str, Any]:
        """Инспектирует статистику."""
        stats = InspectorStats()

        # Считаем события
        stats.events_count = self.event_store.count()

        # Считаем артефакты
        stats.artifacts_count = self.artifact_store.count()

        # Считаем артефакты по типам
        all_artifacts = self.artifact_store.list_artifacts(limit=1000)
        by_type: dict[str, int] = {}
        for artifact in all_artifacts:
            if hasattr(artifact, 'artifact_type'):
                t = artifact.artifact_type
            else:
                t = artifact.get("artifact_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        stats.artifacts_by_type = by_type

        # Считаем workspace
        stats.workspaces_count = len(self.workspace_store.list_workspaces())

        # Считаем задачи
        if self.job_queue:
            job_stats = self.job_queue.get_stats()
            stats.jobs_queued = job_stats.get("queued", 0)
            stats.jobs_processing = job_stats.get("processing", 0)
            stats.jobs_done = job_stats.get("done", 0)
            stats.jobs_dead = job_stats.get("dead", 0)

        return stats.to_dict()

    def get_stats(self) -> dict[str, Any]:
        """
        Получает статистику памяти.

        Returns:
            Словарь статистики.
        """
        return self._inspect_stats()

    def _inspect_trace(self, event_id: str) -> dict[str, Any]:
        """
        Инспектирует трассировку события.

        Показывает:
        - raw event
        - порождённые artifacts
        - статус индексации
        """
        # Находим событие
        event = self.event_store.get_by_id(event_id)
        if not event:
            return {"error": f"Event {event_id} not found"}

        # Находим связанные артефакты
        artifacts = self.artifact_store.get_by_event(event_id)

        return {
            "event": dict(event) if event else None,
            "artifacts": [a.to_dict() for a in artifacts],
            "artifact_count": len(artifacts),
            "indexed": True,  # TODO: проверить векторный индекс
        }

    def _inspect_jobs(self, limit: int = 50) -> dict[str, Any]:
        """Инспектирует очередь задач."""
        # Получаем задачи по статусам
        jobs_queued = self.job_queue.list_jobs(status="queued", limit=limit)
        jobs_processing = self.job_queue.list_jobs(status="processing", limit=limit)
        jobs_done = self.job_queue.list_jobs(status="done", limit=limit)
        jobs_dead = self.job_queue.list_jobs(status="dead", limit=limit)

        return {
            "queued": [j.to_dict() for j in jobs_queued],
            "processing": [j.to_dict() for j in jobs_processing],
            "done": [j.to_dict() for j in jobs_done],
            "dead": [j.to_dict() for j in jobs_dead],
            "stats": self.job_queue.get_stats(),
        }

    def get_debug_snapshot(self, limit: int = 50) -> dict[str, Any]:
        """
        Получает отладочный снимок памяти.

        Args:
            limit: Максимальное количество результатов.

        Returns:
            Снимок для отладки.
        """
        return {
            "timestamp": time.time(),
            "stats": self._inspect_stats(),
            "recent_events": self._inspect_events(limit=limit),
            "recent_artifacts": self._inspect_artifacts(limit=limit),
            "jobs": self._inspect_jobs(limit=limit),
        }


def build_memory_inspector(
    event_store: EventStore,
    artifact_store: ArtifactStore,
    workspace_store: WorkspaceStore,
    job_queue: JobQueueStore,
    identity_core: IdentityCore | None = None,
) -> MemoryInspector:
    """
    Строит MemoryInspector.

    Args:
        event_store: Хранилище событий.
        artifact_store: Хранилище артефактов.
        workspace_store: Хранилище workspace.
        job_queue: Очередь задач.
        identity_core: Identity Core.

    Returns:
        MemoryInspector.
    """
    return MemoryInspector(
        event_store=event_store,
        artifact_store=artifact_store,
        workspace_store=workspace_store,
        job_queue=job_queue,
        identity_core=identity_core,
    )
