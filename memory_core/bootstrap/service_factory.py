"""
Service Factory - фабрика для создания MemoryService.
"""

import os
from dataclasses import dataclass
from typing import Any
from memory_core.facade import MemoryService
from memory_core.storage.sqlite_db import Database
from memory_core.storage.event_store import EventStore
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.workspace_store import WorkspaceStore
from memory_core.storage.state_store import StateStore
from memory_core.storage.job_queue_store import JobQueueStore
from memory_core.runtime_session_store import RuntimeSessionStore
from memory_core.indexing.embeddings import EmbeddingProvider
from memory_core.indexing.vector_index import VectorIndex
from memory_core.retrieval.retrieval_service import RetrievalService
from memory_core.inspect.memory_inspector import build_memory_inspector
from memory_core.worker.background_worker import BackgroundWorker, WorkerConfig
from utils.logger import get_logger


LOGGER = get_logger(__name__)


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
    enable_background_worker: bool = True
    worker_poll_interval: float = 2.0
    worker_shutdown_idle_timeout: float = 300.0  # Таймаут простоя worker (сек)
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
    runtime_session_store = RuntimeSessionStore(state_store)
    job_queue = JobQueueStore(db)

    # Создаём embedding provider
    embedding_provider = EmbeddingProvider(config.embedding_model)

    # Создаём vector index
    vector_index = VectorIndex(
        index_path=config.vector_path,
        embedding_provider=embedding_provider,
    )

    # Создаём retrieval service
    from config.settings import load_config
    from memory_core.config_manager import get_memory_core_config
    from memory_core.inspect.trace_store import MemoryTraceStore
    
    memory_cfg = get_memory_core_config()
    app_cfg = load_config()
    
    # Создаём trace store
    trace_store = MemoryTraceStore()
    
    retrieval_service = RetrievalService(
        db=db,
        vector_index=vector_index,
        retrieval_config=memory_cfg.retrieval,
        runtime_session_store=runtime_session_store,
        include_pending_facts_in_retrieval=bool(
            getattr(getattr(app_cfg, "memory", None), "include_pending_facts_in_retrieval", False)
        ),
    )

    # Создаём главный сервис
    memory_service = MemoryService(
        event_store=event_store,
        artifact_store=artifact_store,
        workspace_store=workspace_store,
        state_store=state_store,
        runtime_session_store=runtime_session_store,
        retrieval_service=retrieval_service,
        vector_index=vector_index,
        embedding_provider=embedding_provider,
        config=config,
        job_queue=job_queue,
    )

    # Добавляем компоненты v2
    memory_service.worker = None
    memory_service.trace_store = trace_store  # Добавляем trace_store

    # Создаём inspector
    memory_service.inspector = build_memory_inspector(
        event_store=event_store,
        artifact_store=artifact_store,
        workspace_store=workspace_store,
        job_queue=job_queue,
    )

    # Создаём и запускаем background worker если включено
    # ЗАМЕЧАНИЕ: worker запускается явно в api/app.py lifespan startup
    if config.enable_background_worker:
        worker = _create_background_worker(
            job_queue=job_queue,
            event_store=event_store,
            artifact_store=artifact_store,
            vector_index=vector_index,
            config=config,
            trace_store=trace_store,  # Добавляем trace_store
        )
        memory_service.worker = worker
        # worker.start()  # Не запускаем автоматически!

        # Worker будет запущен явно в api/app.py lifespan startup
        # Worker будет остановлен через lifespan shutdown в api/app.py
        # atexit не используется, так как не срабатывает при taskkill /F

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


def _create_background_worker(
    job_queue: JobQueueStore,
    event_store: EventStore,
    artifact_store: ArtifactStore,
    vector_index: VectorIndex,
    config: MemoryServiceConfig,
    trace_store: Any | None = None,
) -> BackgroundWorker:
    """
    Создаёт и настраивает BackgroundWorker.

    Args:
        job_queue: Очередь задач.
        event_store: Хранилище событий.
        artifact_store: Хранилище артефактов.
        vector_index: Векторный индекс.
        config: Конфигурация.

    Returns:
        Настроенный BackgroundWorker.
    """
    # Создаём Memory LLM Processor с TaskRouter
    from llm.task_router import TaskModelRouter
    from llm.task_models import TaskModelRegistry, TaskModelProfile
    from memory_core.processors.memory_llm_processor import MemoryLLMProcessor
    from memory_core.governor.governor import build_governor
    from memory_core.config_manager import get_memory_core_config
    from config.settings import load_config

    cfg = load_config()
    mc_config = get_memory_core_config()

    # Создаём TaskModelRegistry
    # Загружаем профили из главного конфига
    registry = TaskModelRegistry.from_settings(cfg)

    # Добавляем профиль memory_llm_process из memory_core конфига (task_model_profiles)
    # Это основной источник настроек LLM для memory_core
    memory_llm_profile_data = mc_config.task_model_profiles.get("memory_llm_process", {})
    if memory_llm_profile_data and "memory_llm_process" not in registry._profiles:
        registry._profiles["memory_llm_process"] = TaskModelProfile(
            name=str(memory_llm_profile_data.get("name", "memory_llm_process")),
            provider=str(memory_llm_profile_data.get("provider", "ollama")),
            model=str(memory_llm_profile_data.get("model", "qwen3:4b")),
            temperature=float(memory_llm_profile_data.get("temperature", 0.1)),
            max_tokens=int(memory_llm_profile_data.get("max_tokens", 1024)),
            timeout=float(memory_llm_profile_data.get("timeout", 60.0)),
            enabled=bool(memory_llm_profile_data.get("enabled", True)),
        )

    task_router = TaskModelRouter(registry=registry)
    
    # Получаем timeout из task_model_profiles
    memory_llm_timeout = float(memory_llm_profile_data.get("timeout", 60.0))
    memory_llm_keep_alive = memory_llm_profile_data.get("keep_alive")
    memory_llm_processor = MemoryLLMProcessor(
        task_router=task_router,
        timeout_sec=memory_llm_timeout,
        keep_alive=memory_llm_keep_alive,
        system_prompt=str(mc_config.llm.system_prompt or ""),
    )
    
    # Создаём Governor
    governor = build_governor(artifact_store)

    def vector_index_updater(artifact):
        """Обновление векторного индекса."""
        import time  # Импортируем time здесь
        
        # artifact может быть dict или MemoryArtifact
        if hasattr(artifact, 'artifact_id'):
            # MemoryArtifact
            artifact_id = artifact.artifact_id
            text = artifact.text
            artifact_type = artifact.artifact_type
            workspace_id = artifact.workspace_id
            created_at = artifact.created_at
        else:
            # dict
            artifact_id = artifact.get("artifact_id", "")
            text = artifact.get("text", "")
            artifact_type = artifact.get("artifact_type", "fact")
            workspace_id = artifact.get("workspace_id", "global")
            created_at = artifact.get("created_at", time.time())

        if artifact_id and text:
            vector_index.add(
                artifact_id=artifact_id,
                text=text,
                metadata={
                    "artifact_type": artifact_type,
                    "workspace_id": workspace_id,
                    "created_at": created_at,
                },
            )

    worker_config = WorkerConfig(
        poll_interval_sec=config.worker_poll_interval,
        enabled=True,
        shutdown_idle_timeout_sec=config.worker_shutdown_idle_timeout,
    )

    worker = BackgroundWorker(
        job_queue=job_queue,
        event_store=event_store,
        artifact_store=artifact_store,  # Добавляем artifact_store для episode planner
        trace_store=trace_store,  # Добавляем trace_store для inspector
        memory_llm_processor=memory_llm_processor,
        governor=lambda proposals, env: governor.decide(proposals, env),
        vector_index_updater=vector_index_updater,
        config=worker_config,
    )

    return worker
