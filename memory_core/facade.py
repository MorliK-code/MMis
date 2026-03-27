"""
Memory Service - единый фасад для работы с памятью.
"""

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_core.bootstrap.service_factory import MemoryServiceConfig

from memory_core.schemas import (
    MemoryEnvelope,
    MemoryArtifact,
    MemoryQuery,
    MemoryQueryResult,
    DocumentIngestRequest,
    DocumentIngestResult,
    MemoryInspectRequest,
    MemoryTrace,
)
from memory_core.storage.event_store import EventStore
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.workspace_store import WorkspaceStore
from memory_core.storage.state_store import StateStore
from memory_core.storage.job_queue_store import JobQueueStore
from memory_core.retrieval.retrieval_service import RetrievalService
from memory_core.retrieval.query_models import ContextPack, Citation
from memory_core.indexing.embeddings import EmbeddingProvider
from memory_core.indexing.vector_index import VectorIndex
from memory_core.processors.ingest_analyzer import IngestAnalyzer
from memory_core.processors.fact_processor import FactProcessor
from memory_core.processors.profile_processor import ProfileProcessor
from memory_core.processors.episode_processor import EpisodeProcessor
from memory_core.processors.task_processor import TaskProcessor
from memory_core.processors.document_processor import DocumentProcessor
from memory_core.processors.dedupe_processor import DedupeProcessor
from memory_core.processors.memory_llm_processor import MemoryLLMProcessor
from memory_core.governor.governor import Governor
from memory_core.inspect.inspector import MemoryInspector
from memory_core.errors import MemoryError, IngestError


class MemoryService:
    """
    Единый фасад для работы с памятью MMis.
    
    Весь проект знает только эти методы:
    - ingest_event(...)
    - query(...)
    - ingest_document(...)
    - inspect(...)
    """
    
    def __init__(
        self,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        workspace_store: WorkspaceStore,
        state_store: StateStore,
        retrieval_service: RetrievalService,
        vector_index: VectorIndex,
        embedding_provider: EmbeddingProvider,
        config: "MemoryServiceConfig",
        job_queue: JobQueueStore | None = None,
    ):
        """
        Инициализирует MemoryService.

        Args:
            event_store: Хранилище событий.
            artifact_store: Хранилище артефактов.
            workspace_store: Хранилище workspace.
            state_store: Хранилище состояния.
            retrieval_service: Сервис retrieval.
            vector_index: Векторный индекс.
            embedding_provider: Провайдер embeddings.
            config: Конфигурация.
            job_queue: Очередь задач (опционально).
        """
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.workspace_store = workspace_store
        self.state_store = state_store
        self.retrieval_service = retrieval_service
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self.config = config
        self.job_queue = job_queue

        # Процессоры
        self.analyzer = IngestAnalyzer()
        self.fact_processor = FactProcessor()
        self.profile_processor = ProfileProcessor()
        self.episode_processor = EpisodeProcessor()
        self.task_processor = TaskProcessor()
        self.document_processor = DocumentProcessor()
        self.dedupe_processor = DedupeProcessor()

        # Memory LLM Processor и Governor будут созданы в worker (service_factory)
        # Здесь не создаём, чтобы избежать дублирования
        self.memory_llm_processor = None
        self.governor = None

        # Inspector
        self.inspector = MemoryInspector(event_store.db)
    
    def ingest_event(self, envelope: MemoryEnvelope) -> dict[str, Any]:
        """
        Принимает событие на обработку.

        Переведено в enqueue-only режим:
        - записывает raw event
        - создаёт job в очереди
        - worker обработает в фоне

        Args:
            envelope: Конверт события.

        Returns:
            Результат ingest.
        """
        try:
            # Шаг 1: Записываем raw event (канон)
            self.event_store.append(envelope)

            # Шаг 2: Анализируем событие (быстрый фильтр)
            analysis = self.analyzer.analyze(envelope)

            if not analysis["should_process"]:
                return {
                    "event_id": envelope.event_id,
                    "processed": False,
                    "reason": "skipped_by_analyzer",
                    "queued": False,
                    "artifacts_created": 0,
                }

            # Шаг 3: Создаём job в очереди (enqueue-only!)
            job_id = self.job_queue.enqueue(
                event_id=envelope.event_id,
                job_type=self.job_queue.TYPE_MEMORY_LLM_PROCESS,
                payload={
                    "workspace_id": envelope.workspace_id,
                    "session_id": envelope.session_id,
                    "source_kind": envelope.source_kind,
                    "payload_type": envelope.payload_type,
                },
                priority=5,
            )

            worker = getattr(self, "worker", None)
            if worker is not None and hasattr(worker, "is_running") and hasattr(worker, "start"):
                try:
                    is_memory_locked = False
                    try:
                        from memory_core.adapter import _memory_llm_lock
                        is_memory_locked = bool(_memory_llm_lock is not None and _memory_llm_lock.locked())
                    except Exception:
                        is_memory_locked = False

                    if not worker.is_running() and not is_memory_locked:
                        worker.start()
                    elif worker.is_running():
                        wake = getattr(worker, "wake", None)
                        if callable(wake):
                            wake()
                except Exception:
                    # Не роняем ingest, если worker не поднялся с первого раза.
                    pass

            return {
                "event_id": envelope.event_id,
                "processed": True,
                "queued": True,
                "job_id": job_id,
                "artifacts_created": 0,  # Будут созданы worker-ом в фоне
            }

        except Exception as e:
            raise IngestError(f"Failed to ingest event: {e}")
    
    def query(self, query: MemoryQuery) -> MemoryQueryResult:
        """
        Выполняет запрос к памяти.
        
        Args:
            query: Запрос к памяти.
            
        Returns:
            Результат запроса.
        """
        # Выполняем retrieval
        context_pack, citations = self.retrieval_service.query(query)
        
        # Формируем результат
        hits = []
        selected_rows = [dict(x) for x in list(getattr(context_pack, "selected_memories", []) or []) if isinstance(x, dict)]
        if selected_rows:
            for row in selected_rows:
                hits.append({
                    "artifact_id": row.get("artifact_id"),
                    "artifact_type": row.get("artifact_type"),
                    "text": row.get("text", ""),
                    "summary": row.get("summary", ""),
                    "prompt_view": row.get("prompt_view", ""),
                    "exposure_mode": row.get("exposure_mode", ""),
                    "sensitivity": row.get("sensitivity", ""),
                    "channel": row.get("channel", ""),
                    "score": row.get("score", 1.0),
                    "confidence": row.get("confidence", 0.5),
                    "metadata": row.get("metadata", {}),
                })
        else:
            for citation in citations:
                hits.append({
                    "artifact_id": citation.artifact_id,
                    "artifact_type": citation.artifact_type,
                    "text": citation.text,
                    "score": 1.0,  # Score можно добавить из retrieval
                    "metadata": citation.metadata,
                    "prompt_view": dict(citation.metadata or {}).get("prompt_view", ""),
                    "exposure_mode": dict(citation.metadata or {}).get("exposure_mode", ""),
                })

        context_blocks = context_pack.to_context_blocks()
        
        return MemoryQueryResult(
            hits=hits,
            context_blocks=context_blocks,
            citations=[c.to_dict() for c in citations],
            blocks=dict(getattr(context_pack, "blocks", {}) or {}),
            selected=selected_rows,
            dropped=[dict(x) for x in list(getattr(context_pack, "dropped_memories", []) or []) if isinstance(x, dict)],
            recent_user_state=dict(getattr(context_pack, "recent_user_state", {}) or {}),
            response_bias=dict(getattr(context_pack, "response_bias", {}) or {}),
            debug=dict(getattr(context_pack, "debug", {}) or {}),
        )
    
    def ingest_document(
        self,
        request: DocumentIngestRequest,
    ) -> DocumentIngestResult:
        """
        Принимает документ на обработку.
        
        Args:
            request: Запрос на ingest документа.
            
        Returns:
            Результат ingest.
        """
        # Создаём envelope для документа
        envelope = MemoryEnvelope(
            source_kind="document",
            payload_type="doc_text",
            text=request.content,
            metadata={
                "document_id": request.document_id,
                "title": request.title,
                "source_kind": request.source_kind,
            },
            workspace_id=request.workspace_id,
        )
        
        # Запускаем document processor
        artifacts = self.document_processor.process_document(request, envelope)
        
        # Сохраняем raw event
        self.event_store.append(envelope)
        
        # Сохраняем артефакты
        if artifacts:
            self.artifact_store.create_many(artifacts)
            self._index_artifacts(artifacts)
        
        # Регистрируем source в workspace
        self.workspace_store.add_source(
            source_id=request.document_id,
            workspace_id=request.workspace_id,
            source_kind=request.source_kind,
            title=request.title,
            content_ref=f"document:{request.document_id}",
            metadata=request.metadata,
        )
        
        return DocumentIngestResult(
            document_id=request.document_id,
            chunks_created=len([a for a in artifacts if a.artifact_type == "document_chunk"]),
            summary_created=any(a.artifact_type == "document_summary" for a in artifacts),
            artifact_ids=[a.artifact_id for a in artifacts],
        )
    
    def inspect(self, request: MemoryInspectRequest) -> dict[str, Any]:
        """
        Инспектирует память (для debug).
        
        Args:
            request: Запрос на инспекцию.
            
        Returns:
            Результат инспекции.
        """
        if request.kind == "events":
            return {
                "items": self.inspector.list_events(
                    workspace_id=request.workspace_id,
                    limit=request.limit,
                ),
            }
        
        elif request.kind == "artifacts":
            return {
                "items": self.inspector.list_artifacts(
                    workspace_id=request.workspace_id,
                    limit=request.limit,
                ),
            }
        
        elif request.kind == "trace" and request.event_id:
            trace = self.inspector.trace_event(request.event_id)
            return trace.to_dict()
        
        elif request.kind == "workspaces":
            return {
                "workspaces": self.inspector.list_workspaces(),
            }
        
        elif request.kind == "profile":
            return {
                "profile_facts": self.inspector.get_profile_facts(
                    workspace_id=request.workspace_id or "global",
                    limit=request.limit,
                ),
            }
        
        elif request.kind == "stats":
            return {
                "stats": self.inspector.get_stats(),
            }
        
        return {"error": f"Unknown inspect kind: {request.kind}"}
    
    def _run_processors(
        self,
        envelope: MemoryEnvelope,
        analysis: dict,
    ) -> list[MemoryArtifact]:
        """Запускает процессоры для события."""
        artifacts = []
        
        # Определяем, какие процессоры запускать
        processors_to_run = analysis.get("processors", [])
        
        # Fact processor
        if "fact_processor" in processors_to_run or not processors_to_run:
            artifacts.extend(self.fact_processor.process(envelope))
        
        # Profile processor
        if self.config.enable_profile_memory:
            artifacts.extend(self.profile_processor.process(envelope))
        
        # Episode processor
        artifacts.extend(self.episode_processor.process(envelope))
        
        # Task processor
        if self.config.enable_task_memory:
            artifacts.extend(self.task_processor.process(envelope))
        
        # Document processor
        if envelope.source_kind == "document" and self.config.enable_document_memory:
            artifacts.extend(self.document_processor.process(envelope))
        
        return artifacts

    def _index_artifacts(self, artifacts: list[MemoryArtifact]) -> None:
        """Индексирует артефакты в vector index."""
        for artifact in artifacts:
            self.vector_index.add(
                artifact_id=artifact.artifact_id,
                text=artifact.text,
                metadata={
                    "artifact_type": artifact.artifact_type,
                    "workspace_id": artifact.workspace_id,
                    "created_at": artifact.created_at,
                },
            )

    def _index_artifact_data(self, artifact_data: dict[str, Any]) -> None:
        """Индексирует артефакт из данных Governor."""
        artifact_id = artifact_data.get("artifact_id", "")
        text = artifact_data.get("text", "")
        artifact_type = artifact_data.get("artifact_type", "fact")
        workspace_id = artifact_data.get("workspace_id", "global")
        created_at = artifact_data.get("created_at", time.time())

        if artifact_id and text:
            self.vector_index.add(
                artifact_id=artifact_id,
                text=text,
                metadata={
                    "artifact_type": artifact_type,
                    "workspace_id": workspace_id,
                    "created_at": created_at,
                },
            )

    def get_current_workspace(self) -> str:
        """Получает текущий workspace."""
        return self.state_store.get_current_workspace(self.config.default_workspace)
    
    def set_current_workspace(self, workspace_id: str) -> None:
        """Устанавливает текущий workspace."""
        self.state_store.set_current_workspace(workspace_id)
    
    def get_stats(self) -> dict[str, Any]:
        """Получает статистику памяти."""
        return self.inspector.get_stats()
