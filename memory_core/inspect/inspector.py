"""
Memory Inspector - диагностика и просмотр памяти.
"""

from typing import Any
from memory_core.storage.sqlite_db import Database
from memory_core.storage.event_store import EventStore
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.workspace_store import WorkspaceStore
from memory_core.schemas import MemoryTrace


class MemoryInspector:
    """
    Инспектор памяти для диагностики.
    
    Умеет:
    - список events
    - список artifacts
    - trace event -> artifacts
    - просмотр workspace sources
    - просмотр profile facts
    - debug what was retrieved
    """
    
    def __init__(self, db: Database):
        """
        Инициализирует инспектор.
        
        Args:
            db: Экземпляр Database.
        """
        self.db = db
        self.event_store = EventStore(db)
        self.artifact_store = ArtifactStore(db)
        self.workspace_store = WorkspaceStore(db)
    
    def list_events(
        self,
        workspace_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Получает список событий.
        
        Args:
            workspace_id: Фильтр по workspace.
            limit: Максимальное количество.
            
        Returns:
            Список событий.
        """
        events = self.event_store.list_events(
            workspace_id=workspace_id,
            limit=limit,
        )
        return [e.to_dict() for e in events]
    
    def list_artifacts(
        self,
        artifact_type: str | None = None,
        workspace_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Получает список артефактов.
        
        Args:
            artifact_type: Фильтр по типу.
            workspace_id: Фильтр по workspace.
            limit: Максимальное количество.
            
        Returns:
            Список артефактов.
        """
        artifacts = self.artifact_store.list_artifacts(
            artifact_type=artifact_type,
            workspace_id=workspace_id,
            limit=limit,
        )
        return [a.to_dict() for a in artifacts]
    
    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """
        Получает событие по ID.
        
        Args:
            event_id: ID события.
            
        Returns:
            Словарь события или None.
        """
        event = self.event_store.get_by_id(event_id)
        return event.to_dict() if event else None
    
    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """
        Получает артефакт по ID.
        
        Args:
            artifact_id: ID артефакта.
            
        Returns:
            Словарь артефакта или None.
        """
        artifact = self.artifact_store.get_by_id(artifact_id)
        return artifact.to_dict() if artifact else None
    
    def trace_event(self, event_id: str) -> MemoryTrace:
        """
        Трассирует событие: raw event -> порождённые artifacts.
        
        Args:
            event_id: ID события.
            
        Returns:
            MemoryTrace.
        """
        # Получаем событие
        event = self.event_store.get_by_id(event_id)
        event_dict = event.to_dict() if event else None
        
        # Получаем артефакты
        artifacts = self.artifact_store.get_by_source_event(event_id)
        artifact_dicts = [a.to_dict() for a in artifacts]
        
        # Проверяем, индексировано ли
        indexed = len(artifacts) > 0
        
        return MemoryTrace(
            event=event_dict,
            artifacts=artifact_dicts,
            indexed=indexed,
            retrieved=False,  # Нужно проверять отдельно
        )
    
    def list_workspaces(self) -> list[dict[str, Any]]:
        """
        Получает список workspace.
        
        Returns:
            Список workspace.
        """
        return self.workspace_store.list_workspaces()
    
    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        """
        Получает workspace по ID.
        
        Args:
            workspace_id: ID workspace.
            
        Returns:
            Словарь workspace или None.
        """
        return self.workspace_store.get_workspace(workspace_id)
    
    def list_workspace_sources(
        self,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        """
        Получает список источников workspace.
        
        Args:
            workspace_id: ID workspace.
            
        Returns:
            Список источников.
        """
        return self.workspace_store.list_sources(workspace_id=workspace_id)
    
    def get_profile_facts(
        self,
        workspace_id: str = "global",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Получает профильные факты.
        
        Args:
            workspace_id: ID workspace.
            limit: Максимальное количество.
            
        Returns:
            Список профильных фактов.
        """
        artifacts = self.artifact_store.list_artifacts(
            artifact_type="profile_fact",
            workspace_id=workspace_id,
            status="active",
            limit=limit,
        )
        return [a.to_dict() for a in artifacts]
    
    def get_stats(self) -> dict[str, Any]:
        """
        Получает статистику памяти.
        
        Returns:
            Словарь статистики.
        """
        return {
            "events_count": self.event_store.count(),
            "artifacts_count": self.artifact_store.count(),
            "workspaces_count": len(self.workspace_store.list_workspaces()),
        }
    
    def search_artifacts(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Ищет артефакты по тексту.
        
        Args:
            query: Поисковый запрос.
            workspace_id: Фильтр по workspace.
            limit: Максимальное количество.
            
        Returns:
            Список найденных артефактов.
        """
        artifacts = self.artifact_store.search_by_text(
            query=query,
            workspace_id=workspace_id,
            limit=limit,
        )
        return [a.to_dict() for a in artifacts]
