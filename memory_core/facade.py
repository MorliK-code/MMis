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
from memory_core.episode_manager import EpisodeManager
from memory_core.memory_types import (
    classify_event_memory_type,
    enrich_metadata_with_memory_type,
    normalize_memory_type,
)
from memory_core.runtime_session_store import RuntimeSessionStore


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

        # Fast runtime memory. This is updated synchronously in ingest_event()
        # and merged into query() before the background worker finishes.
        self.runtime_session_store = RuntimeSessionStore()
        self.episode_manager = EpisodeManager()
    
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
            runtime_update = self._update_runtime_layer(envelope)
            memory_type = classify_event_memory_type(
                envelope.source_kind,
                envelope.payload_type,
                envelope.metadata,
            )
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
                    "artifact_ids": [],
                    "memory_type": memory_type,
                    **runtime_update,
                }

            # Шаг 3: Создаём job в очереди (enqueue-only!)
            if self.job_queue is None:
                return {
                    "event_id": envelope.event_id,
                    "processed": True,
                    "queued": False,
                    "job_id": None,
                    "artifacts_created": 0,
                    "artifact_ids": [],
                    "memory_type": memory_type,
                    **runtime_update,
                }

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
                    scheduler_mode = "strict"
                    try:
                        from memory_core.config_manager import get_memory_core_config
                        scheduler_mode = str(get_memory_core_config().memory_llm_scheduler_mode or "strict").strip().lower()
                    except Exception:
                        scheduler_mode = "strict"
                    is_memory_locked = False
                    try:
                        from memory_core.adapter import _memory_llm_lock
                        is_memory_locked = bool(_memory_llm_lock is not None and _memory_llm_lock.locked())
                    except Exception:
                        is_memory_locked = False

                    can_start_worker = (scheduler_mode != "strict") or (not is_memory_locked)
                    if scheduler_mode == "strict" and is_memory_locked:
                        try:
                            from utils.logger import get_logger

                            get_logger(__name__).info(
                                "MemoryService: strict scheduler keeps worker stopped while main lock is held"
                            )
                        except Exception:
                            pass
                    if not worker.is_running() and can_start_worker:
                        if scheduler_mode == "cooperative" and is_memory_locked:
                            try:
                                from utils.logger import get_logger

                                get_logger(__name__).info(
                                    "MemoryService: cooperative scheduler starts worker under main lock"
                                )
                            except Exception:
                                pass
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
                "artifact_ids": [],
                "memory_type": memory_type,
                **runtime_update,
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
        self._merge_runtime_layers(context_pack, query)
        
        # Формируем результат
        hits = []
        selected_rows = [
            self._enrich_selected_row(dict(x))
            for x in list(getattr(context_pack, "selected_memories", []) or [])
            if isinstance(x, dict)
        ]
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
                metadata = enrich_metadata_with_memory_type(
                    citation.metadata,
                    artifact_type=citation.artifact_type,
                )
                hits.append({
                    "artifact_id": citation.artifact_id,
                    "artifact_type": citation.artifact_type,
                    "text": citation.text,
                    "score": 1.0,  # Score можно добавить из retrieval
                    "metadata": metadata,
                    "prompt_view": metadata.get("prompt_view", ""),
                    "exposure_mode": metadata.get("exposure_mode", ""),
                })

        context_blocks = context_pack.to_context_blocks()
        
        return MemoryQueryResult(
            hits=hits,
            context_blocks=context_blocks,
            citations=[self._enrich_citation_dict(c) for c in citations],
            blocks=dict(getattr(context_pack, "blocks", {}) or {}),
            selected=selected_rows,
            dropped=[dict(x) for x in list(getattr(context_pack, "dropped_memories", []) or []) if isinstance(x, dict)],
            recent_user_state=dict(getattr(context_pack, "recent_user_state", {}) or {}),
            response_bias=dict(getattr(context_pack, "response_bias", {}) or {}),
            debug=dict(getattr(context_pack, "debug", {}) or {}),
        )

    def _update_runtime_layer(self, envelope: MemoryEnvelope) -> dict[str, Any]:
        runtime_update: dict[str, Any] = {"runtime_updated": False}
        try:
            episode_manager = getattr(self, "episode_manager", None)
            episode_result: dict[str, Any] = {}
            if episode_manager is not None and hasattr(episode_manager, "update_from_event"):
                episode_result = dict(episode_manager.update_from_event(envelope) or {})

            runtime_store = getattr(self, "runtime_session_store", None)
            if runtime_store is not None and hasattr(runtime_store, "update_from_event"):
                runtime_update = dict(
                    runtime_store.update_from_event(
                        envelope,
                        current_episode_id=str(episode_result.get("episode_id") or ""),
                    )
                    or {}
                )

            if episode_result:
                runtime_update["episode_updated"] = bool(episode_result.get("episode_updated", True))
                runtime_update["current_episode_id"] = str(episode_result.get("episode_id") or "")
        except Exception as exc:
            runtime_update = {"runtime_updated": False, "runtime_error": str(exc)}
        return runtime_update

    def _merge_runtime_layers(self, context_pack: ContextPack, query: MemoryQuery) -> None:
        if context_pack is None:
            return

        include_runtime = self._include_pending_facts_in_retrieval()
        layers: list[dict[str, Any]] = []
        runtime_store = getattr(self, "runtime_session_store", None)
        if runtime_store is not None and hasattr(runtime_store, "build_query_layer"):
            try:
                layers.append(dict(runtime_store.build_query_layer(query.workspace_id, query.session_id) or {}))
            except Exception as exc:
                layers.append({"debug": {"source": "runtime_session_store", "error": str(exc)}})

        episode_manager = getattr(self, "episode_manager", None)
        if episode_manager is not None and hasattr(episode_manager, "build_query_layer"):
            try:
                layers.append(dict(episode_manager.build_query_layer(query.workspace_id, query.session_id) or {}))
            except Exception as exc:
                layers.append({"debug": {"source": "episode_manager", "error": str(exc)}})

        if not layers:
            return

        existing_selected = [
            self._enrich_selected_row(dict(x))
            for x in list(getattr(context_pack, "selected_memories", []) or [])
            if isinstance(x, dict)
        ]
        runtime_selected: list[dict[str, Any]] = []
        runtime_selected_dropped: list[dict[str, Any]] = []
        debug_sources: dict[str, Any] = {}
        active_episode: dict[str, Any] = {}
        recent_user_state: dict[str, Any] = dict(getattr(context_pack, "recent_user_state", {}) or {})

        for layer in layers:
            layer_debug = dict(layer.get("debug") or {})
            layer_source = str(layer_debug.get("source") or "runtime").strip() or "runtime"
            is_runtime_layer = layer_source == "runtime_session_store"
            blocks = dict(layer.get("blocks") or {})
            if (not is_runtime_layer) or include_runtime:
                for key, value in blocks.items():
                    self._merge_context_block(context_pack, key, str(value or "").strip())

            for row in list(layer.get("selected") or []):
                if isinstance(row, dict):
                    enriched = self._enrich_selected_row(dict(row))
                    if is_runtime_layer and not include_runtime:
                        runtime_selected_dropped.append(enriched)
                    elif is_runtime_layer and not self._runtime_row_matches_query(enriched, query):
                        runtime_selected_dropped.append(enriched)
                    else:
                        runtime_selected.append(enriched)

            state = dict(layer.get("recent_user_state") or {})
            if state and ((not is_runtime_layer) or include_runtime):
                recent_user_state.update(state)

            debug = layer_debug
            source = layer_source
            debug_sources[source] = debug
            if isinstance(debug.get("active_episode"), dict):
                active_episode = dict(debug.get("active_episode") or {})

        if include_runtime:
            context_pack.selected_memories = runtime_selected + existing_selected
            context_pack.recent_user_state = recent_user_state

        debug = dict(getattr(context_pack, "debug", {}) or {})
        debug.setdefault("sources", {})
        debug_sources_existing = dict(debug.get("sources") or {})
        debug_sources_existing.update(debug_sources)
        debug["sources"] = debug_sources_existing
        debug["selected_by_source"] = {
            "runtime": len(runtime_selected),
            "long_term": len(existing_selected),
        }
        debug["dropped_runtime_rows"] = [dict(row) for row in runtime_selected_dropped]
        debug["include_pending_facts_in_retrieval"] = include_runtime
        if active_episode:
            debug["active_episode"] = active_episode
        context_pack.debug = debug

    def _runtime_row_matches_query(self, row: dict[str, Any], query: MemoryQuery) -> bool:
        metadata = dict(row.get("metadata") or {})
        row_session = str(metadata.get("session_id") or "").strip()
        query_session = str(query.session_id or "").strip()
        if row_session and query_session and row_session == query_session:
            return True

        row_episode = str(metadata.get("current_episode_id") or "").strip()
        query_episode = str(query.topic_thread_id or "").strip()
        if row_episode and query_episode and row_episode == query_episode:
            return True

        row_text = " ".join(
            [
                str(row.get("text") or ""),
                str(row.get("summary") or ""),
                str(metadata.get("active_topic") or ""),
            ]
        ).lower()
        query_text = str(query.text or "").lower()
        if not row_text or not query_text:
            return False

        query_tokens = {token for token in query_text.split() if len(token) >= 4}
        if not query_tokens:
            return False
        overlap = sum(1 for token in query_tokens if token in row_text)
        return overlap >= 1

    def _merge_context_block(self, context_pack: ContextPack, key: str, value: str) -> None:
        if not value:
            return
        blocks = dict(getattr(context_pack, "blocks", {}) or {})
        existing = str(blocks.get(key) or "").strip()
        if existing and value not in existing:
            blocks[key] = f"{value}\n{existing}"
        elif not existing:
            blocks[key] = value
        context_pack.blocks = blocks

        items = self._block_to_items(value)
        if not items:
            return
        if key == "continuity_hints":
            context_pack.continuity_hints = self._merge_unique(items, list(context_pack.continuity_hints or []))

    def _enrich_selected_row(self, row: dict[str, Any]) -> dict[str, Any]:
        artifact_type = str(row.get("artifact_type") or "").strip() or "fact"
        metadata = enrich_metadata_with_memory_type(
            dict(row.get("metadata") or {}),
            artifact_type=artifact_type,
        )
        row["metadata"] = metadata
        row.setdefault("summary", "")
        row.setdefault("prompt_view", metadata.get("prompt_view") or row.get("text") or row.get("summary") or "")
        row.setdefault("exposure_mode", metadata.get("exposure_mode") or "prompt_safe")
        row.setdefault("sensitivity", metadata.get("sensitivity") or "low")
        row.setdefault("confidence", metadata.get("confidence") or 0.5)
        row.setdefault("score", metadata.get("score") or 1.0)
        return row

    def _enrich_citation_dict(self, citation: Citation) -> dict[str, Any]:
        row = citation.to_dict()
        row["metadata"] = enrich_metadata_with_memory_type(
            dict(row.get("metadata") or {}),
            artifact_type=row.get("artifact_type"),
        )
        return row

    def _include_pending_facts_in_retrieval(self) -> bool:
        try:
            from config.settings import load_config

            cfg = load_config()
            memory_cfg = getattr(cfg, "memory", None)
            if isinstance(memory_cfg, dict):
                return bool(memory_cfg.get("include_pending_facts_in_retrieval", False))
            return bool(getattr(memory_cfg, "include_pending_facts_in_retrieval", False))
        except Exception:
            return False

    @staticmethod
    def _block_to_items(value: str) -> list[str]:
        items: list[str] = []
        for line in str(value or "").splitlines():
            text = line.strip()
            if text.startswith("- "):
                text = text[2:].strip()
            if text:
                items.append(text)
        return items

    @staticmethod
    def _merge_unique(first: list[str], second: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in [*first, *second]:
            text = str(value or "").strip()
            marker = text.lower()
            if not text or marker in seen:
                continue
            seen.add(marker)
            out.append(text)
        return out
    
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
            stats = dict(self.inspector.get_stats() or {})
            runtime_store = getattr(self, "runtime_session_store", None)
            episode_manager = getattr(self, "episode_manager", None)
            worker = getattr(self, "worker", None)
            if runtime_store is not None and hasattr(runtime_store, "list_states"):
                stats["runtime_sessions_count"] = len(runtime_store.list_states(limit=10000))
            if episode_manager is not None and hasattr(episode_manager, "list_episodes"):
                stats["runtime_episodes_count"] = len(episode_manager.list_episodes(limit=10000))
            if worker is not None and hasattr(worker, "get_stats"):
                try:
                    worker_stats = dict(worker.get_stats() or {})
                    worker_config = dict(worker_stats.get("config") or {})
                    worker_inner_stats = dict(worker_stats.get("stats") or {})
                    stats["scheduler_mode"] = str(worker_config.get("scheduler_mode") or "")
                    stats["interrupt_count"] = int(worker_inner_stats.get("interrupt_count") or 0)
                    stats["requeue_count"] = int(worker_inner_stats.get("requeue_count") or 0)
                except Exception:
                    pass
            return {"stats": stats}

        elif request.kind == "runtime":
            runtime_store = getattr(self, "runtime_session_store", None)
            episode_manager = getattr(self, "episode_manager", None)
            states = []
            episodes = []
            if runtime_store is not None and hasattr(runtime_store, "list_states"):
                states = runtime_store.list_states(
                    workspace_id=request.workspace_id,
                    limit=request.limit,
                )
            if episode_manager is not None and hasattr(episode_manager, "list_episodes"):
                episodes = episode_manager.list_episodes(
                    workspace_id=request.workspace_id,
                    limit=request.limit,
                )
            return {
                "items": states,
                "episodes": episodes,
                "sources": ["runtime_session_store", "episode_manager"],
            }

        elif request.kind == "episodes":
            episode_manager = getattr(self, "episode_manager", None)
            episodes = []
            if episode_manager is not None and hasattr(episode_manager, "list_episodes"):
                episodes = episode_manager.list_episodes(
                    workspace_id=request.workspace_id,
                    limit=request.limit,
                )
            return {"items": episodes}
        
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
                    "memory_type": normalize_memory_type(artifact.artifact_type),
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
                    "memory_type": normalize_memory_type(artifact_type),
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
