"""
Retrieval Service - сервис поиска и извлечения памяти.
"""

import time
from memory_core.storage.sqlite_db import Database
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.schemas import MemoryQuery, MemoryArtifact
from memory_core.retrieval.query_models import RetrievalFilters, ContextPack, Citation
from memory_core.retrieval.context_builder import ContextBuilder
from memory_core.retrieval.reranker import Reranker
from memory_core.retrieval.filters import apply_filters, filter_by_text_similarity
from memory_core.errors import RetrievalError


class RetrievalService:
    """
    Сервис для hybrid retrieval.
    
    Ищет по:
    - profile facts
    - active tasks
    - recent episodes
    - relevant facts
    - document chunks
    """
    
    def __init__(self, db: Database):
        """
        Инициализирует Retrieval Service.
        
        Args:
            db: Экземпляр Database.
        """
        self.db = db
        self.artifact_store = ArtifactStore(db)
        self.context_builder = ContextBuilder()
        self.reranker = Reranker()
    
    def query(self, query: MemoryQuery) -> tuple[ContextPack, list[Citation]]:
        """
        Выполняет запрос к памяти.
        
        Args:
            query: Запрос к памяти.
            
        Returns:
            Кортеж (ContextPack, список Citation).
        """
        # Получаем артефакты
        artifacts = self._retrieve_artifacts(query)
        
        # Строим контекст
        context_pack, citations = self.context_builder.build(artifacts, query)
        
        return context_pack, citations
    
    def _retrieve_artifacts(self, query: MemoryQuery) -> list[MemoryArtifact]:
        """
        Извлекает артефакты по запросу.
        
        Args:
            query: Запрос к памяти.
            
        Returns:
            Список релевантных артефактов.
        """
        # Создаём фильтры
        filters = RetrievalFilters(
            artifact_types=query.artifact_types if query.artifact_types else [],
            workspace_id=query.workspace_id,
            namespace=query.namespace,
        )
        
        # Получаем артефакты из хранилища
        all_artifacts = self.artifact_store.list_artifacts(
            workspace_id=query.workspace_id,
            namespace=query.namespace,
            limit=100,  # Берём с запасом для rerank
        )
        
        # Применяем фильтры
        filtered = apply_filters(all_artifacts, filters)
        
        # Если есть текст запроса, фильтруем по схожести
        if query.text:
            # Поиск по тексту
            text_matches = self.artifact_store.search_by_text(
                query=query.text,
                workspace_id=query.workspace_id,
                artifact_types=query.artifact_types if query.artifact_types else None,
                limit=50,
            )
            
            # Объединяем результаты
            artifact_ids = {a.artifact_id for a in filtered}
            for artifact in text_matches:
                if artifact.artifact_id not in artifact_ids:
                    filtered.append(artifact)
        
        # Rerank по релевантности
        if query.text:
            filtered = self.reranker.rerank(
                filtered,
                query.text,
                top_k=query.top_k,
            )
        else:
            # Если нет текста, просто берём top_k по recency
            filtered = sorted(
                filtered,
                key=lambda a: a.created_at,
                reverse=True,
            )[:query.top_k]
        
        return filtered
    
    def get_profile_facts(
        self,
        workspace_id: str = "global",
        limit: int = 20,
    ) -> list[MemoryArtifact]:
        """
        Получает профильные факты.
        
        Args:
            workspace_id: ID workspace.
            limit: Максимальное количество.
            
        Returns:
            Список профильных фактов.
        """
        return self.artifact_store.list_artifacts(
            artifact_type="profile_fact",
            workspace_id=workspace_id,
            status="active",
            limit=limit,
        )
    
    def get_active_tasks(
        self,
        workspace_id: str = "global",
        limit: int = 10,
    ) -> list[MemoryArtifact]:
        """
        Получает активные задачи.
        
        Args:
            workspace_id: ID workspace.
            limit: Максимальное количество.
            
        Returns:
            Список активных задач.
        """
        all_tasks = self.artifact_store.list_artifacts(
            artifact_type="task",
            workspace_id=workspace_id,
            status="active",
            limit=limit * 2,  # Берём с запасом
        )
        
        # Фильтруем только открытые
        open_tasks = [
            t for t in all_tasks
            if t.metadata.get("task_status") == "open"
        ]
        
        return open_tasks[:limit]
    
    def get_recent_episodes(
        self,
        workspace_id: str = "global",
        limit: int = 5,
    ) -> list[MemoryArtifact]:
        """
        Получает недавние эпизоды.
        
        Args:
            workspace_id: ID workspace.
            limit: Максимальное количество.
            
        Returns:
            Список недавних эпизодов.
        """
        return self.artifact_store.list_artifacts(
            artifact_type="episode",
            workspace_id=workspace_id,
            status="active",
            limit=limit,
        )
    
    def search_documents(
        self,
        query: str,
        workspace_id: str = "global",
        limit: int = 10,
    ) -> list[MemoryArtifact]:
        """
        Ищет по документам.
        
        Args:
            query: Поисковый запрос.
            workspace_id: ID workspace.
            limit: Максимальное количество.
            
        Returns:
            Список документ чанков.
        """
        return self.artifact_store.search_by_text(
            query=query,
            workspace_id=workspace_id,
            artifact_types=["document_chunk", "document_summary"],
            limit=limit,
        )
    
    def get_workspace_summary(self, workspace_id: str) -> str:
        """
        Получает краткую информацию о workspace.
        
        Args:
            workspace_id: ID workspace.
            
        Returns:
            Текстовое описание workspace.
        """
        from memory_core.storage.workspace_store import WorkspaceStore
        
        workspace_store = WorkspaceStore(self.db)
        workspace = workspace_store.get_workspace(workspace_id)
        
        if workspace:
            title = workspace.get("title", "Unknown")
            goal = workspace.get("goal", "")
            
            if goal:
                return f"Workspace: {title}\nGoal: {goal}"
            return f"Workspace: {title}"
        
        return f"Workspace: {workspace_id}"
