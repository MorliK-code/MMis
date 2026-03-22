"""
Service Factory - фабрика для создания MemoryService.
"""

import os
from dataclasses import dataclass
from memory_core.facade import MemoryService
from memory_core.storage.sqlite_db import Database
from memory_core.storage.event_store import EventStore
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.workspace_store import WorkspaceStore
from memory_core.storage.state_store import StateStore
from memory_core.indexing.embeddings import EmbeddingProvider
from memory_core.indexing.vector_index import VectorIndex
from memory_core.retrieval.retrieval_service import RetrievalService


@dataclass
class MemoryServiceConfig:
    """
    Конфигурация для MemoryService.
    """
    db_path: str = "data/memory_core/memory.db"
    vector_path: str = "data/memory_core/vector"
    vector_backend: str = "memory"  # memory | chroma
    default_namespace: str = "default"
    default_workspace: str = "global"
    top_k: int = 8
    enable_profile_memory: bool = True
    enable_task_memory: bool = True
    enable_document_memory: bool = True
    enable_debug_inspector: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


def build_memory_service(config: MemoryServiceConfig | None = None) -> MemoryService:
    """
    Создаёт и настраивает MemoryService.
    
    Args:
        config: Конфигурация сервиса.
        
    Returns:
        Настроенный MemoryService.
    """
    if config is None:
        config = MemoryServiceConfig()
    
    # Создаём базу данных
    db = Database(config.db_path)
    
    # Создаём хранилища
    event_store = EventStore(db)
    artifact_store = ArtifactStore(db)
    workspace_store = WorkspaceStore(db)
    state_store = StateStore(db)
    
    # Создаём embedding provider
    embedding_provider = EmbeddingProvider(config.embedding_model)
    
    # Создаём vector index
    vector_index = VectorIndex(
        index_path=config.vector_path,
        embedding_provider=embedding_provider,
    )
    
    # Создаём retrieval service
    retrieval_service = RetrievalService(db)
    
    # Создаём главный сервис
    memory_service = MemoryService(
        event_store=event_store,
        artifact_store=artifact_store,
        workspace_store=workspace_store,
        state_store=state_store,
        retrieval_service=retrieval_service,
        vector_index=vector_index,
        embedding_provider=embedding_provider,
        config=config,
    )
    
    # Инициализируем default workspace
    _ensure_default_workspace(workspace_store, config.default_workspace)
    
    return memory_service


def _ensure_default_workspace(
    workspace_store: WorkspaceStore,
    default_workspace: str,
) -> None:
    """Гарантирует существование default workspace."""
    if not workspace_store.workspace_exists(default_workspace):
        workspace_store.create_workspace(
            workspace_id=default_workspace,
            title="Default Workspace",
            goal="Default workspace for general memory",
        )
