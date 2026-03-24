"""
BackgroundWorker - фоновый воркер для обработки очереди ingest_jobs.

Алгоритм работы:
1. Берёт job из очереди (dequeue)
2. Ставит status = processing
3. Загружает raw event из event_store
4. Отправляет событие в Memory LLM processor
5. Получает proposals
6. Передаёт proposals в Governor
7. Governor решает: создать/обновить/объединить/отклонить artifact
8. Обновляется vector index
9. Job → done

Если ошибка:
- attempts++
- retry_wait с backoff
- после max_attempts → dead
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from memory_core.storage.job_queue_store import JobQueueStore, IngestJob
from memory_core.storage.event_store import EventStore
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.schemas import MemoryEnvelope
from memory_core.inspect.trace_store import MemoryTraceStore
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class WorkerConfig:
    """
    Конфигурация воркера.
    """
    worker_id: str = field(default_factory=lambda: f"worker_{uuid.uuid4().hex[:8]}")
    poll_interval_sec: float = 2.0  # Интервал опроса очереди
    max_jobs_per_cycle: int = 10  # Максимум задач за один цикл
    job_types: list[str] = field(default_factory=lambda: ["memory_llm_process"])
    enabled: bool = True


@dataclass(slots=True)
class WorkerStats:
    """
    Статистика воркера.
    """
    jobs_processed: int = 0
    jobs_succeeded: int = 0
    jobs_failed: int = 0
    jobs_retried: int = 0
    last_job_at: float | None = None
    last_error: str | None = None
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "jobs_processed": self.jobs_processed,
            "jobs_succeeded": self.jobs_succeeded,
            "jobs_failed": self.jobs_failed,
            "jobs_retried": self.jobs_retried,
            "last_job_at": self.last_job_at,
            "last_error": self.last_error,
            "uptime_sec": time.time() - self.started_at,
        }


# Типы для callback-функций
MemoryLLMProcessorFn = Callable[[MemoryEnvelope], dict[str, Any]]
GovernorFn = Callable[[list[dict[str, Any]], MemoryEnvelope], dict[str, Any]]
VectorIndexFn = Callable[[dict[str, Any]], None]


class BackgroundWorker:
    """
    Фоновый воркер для обработки очереди памяти.

    Запускается в отдельном потоке и обрабатывает задачи из ingest_jobs.
    """

    def __init__(
        self,
        job_queue: JobQueueStore,
        event_store: EventStore,
        memory_llm_processor: MemoryLLMProcessorFn,
        governor: GovernorFn,
        vector_index_updater: VectorIndexFn | None = None,
        config: WorkerConfig | None = None,
        artifact_store: ArtifactStore | None = None,
        trace_store: MemoryTraceStore | None = None,
    ):
        """
        Инициализирует воркер.

        Args:
            job_queue: Хранилище очереди задач.
            event_store: Хранилище событий.
            memory_llm_processor: Функция обработки Memory LLM.
            governor: Функция Governor для принятия решений.
            vector_index_updater: Функция обновления векторного индекса.
            config: Конфигурация воркера.
            artifact_store: Хранилище артефактов (для episode planner).
            trace_store: Хранилище trace (для inspector).
        """
        self.job_queue = job_queue
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.trace_store = trace_store
        self.memory_llm_processor = memory_llm_processor
        self.governor = governor
        self.vector_index_updater = vector_index_updater

        self.config = config or WorkerConfig()
        self.stats = WorkerStats()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self) -> None:
        """Запускает воркер в фоновом потоке."""
        if self._running:
            LOGGER.warning(f"Worker {self.config.worker_id} already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"MemoryWorker-{self.config.worker_id}",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        LOGGER.info(f"Worker {self.config.worker_id} started")

    def stop(self, timeout_sec: float = 5.0) -> None:
        """
        Останавливает воркер.

        Args:
            timeout_sec: Таймаут ожидания остановки.
        """
        if not self._running:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout_sec)
        self._running = False
        LOGGER.info(f"Worker {self.config.worker_id} stopped")

    def is_running(self) -> bool:
        """Проверяет, запущен ли воркер."""
        return self._running

    def get_stats(self) -> dict[str, Any]:
        """Получает статистику воркера."""
        return {
            "worker_id": self.config.worker_id,
            "running": self._running,
            "config": {
                "poll_interval_sec": self.config.poll_interval_sec,
                "max_jobs_per_cycle": self.config.max_jobs_per_cycle,
                "job_types": self.config.job_types,
            },
            "stats": self.stats.to_dict(),
        }

    def process_one_job(self) -> bool:
        """
        Обрабатывает одну задачу из очереди.

        Returns:
            True если задача обработана, False если задач нет.
        """
        LOGGER.info(f"Worker {self.config.worker_id}: Attempting to dequeue job...")
        
        # Получаем задачу из очереди
        job = self.job_queue.dequeue(
            worker_id=self.config.worker_id,
            job_type=self.config.job_types[0] if len(self.config.job_types) == 1 else None,
        )
        
        LOGGER.info(f"Worker {self.config.worker_id}: Dequeue result: {job is not None}")

        if not job:
            return False

        try:
            LOGGER.info(f"Worker {self.config.worker_id}: Processing job {job.job_id[:8]}...")
            self._process_job(job)
            self.stats.jobs_succeeded += 1
            LOGGER.info(f"Worker {self.config.worker_id}: Job {job.job_id[:8]}... succeeded")
            return True
        except Exception as e:
            LOGGER.exception(f"Job {job.job_id} failed: {e}")
            self.stats.jobs_failed += 1
            self.stats.last_error = str(e)

            # Пытаемся сделать retry
            retried = self.job_queue.fail(job.job_id, str(e), retry=True)
            if retried:
                self.stats.jobs_retried += 1
            return True
        finally:
            self.stats.jobs_processed += 1
            self.stats.last_job_at = time.time()
            LOGGER.info(f"Worker {self.config.worker_id}: Total processed: {self.stats.jobs_processed}")

    def _process_job(self, job: IngestJob) -> None:
        """
        Обрабатывает задачу.

        Args:
            job: Задача для обработки.
        """
        from datetime import datetime
        
        trace_id = f"trace_{job.job_id}"
        LOGGER.debug(f"Processing job {job.job_id} for event {job.event_id}")

        # Загружаем raw event
        event = self.event_store.get_by_id(job.event_id)
        if not event:
            raise ValueError(f"Event {job.event_id} not found")

        # Пишем trace: event_loaded
        if self.trace_store:
            self.trace_store.append_worker_trace(trace_id, {
                "trace_id": trace_id,
                "stage": "event_loaded",
                "created_at": datetime.now().isoformat(),
                "job_id": job.job_id,
                "event_id": job.event_id,
                "job_type": job.job_type,
                "status": job.status,
            })

        # Создаём envelope из события (MemoryEnvelope — это dataclass)
        envelope = MemoryEnvelope(
            event_id=event.event_id,
            source_kind=event.source_kind,
            payload_type=event.payload_type,
            text=event.text,
            metadata=event.metadata,
            namespace=event.namespace,
            workspace_id=event.workspace_id,
            session_id=event.session_id,
            ts=event.ts,
        )

        # Получаем active episode для continuity
        episode_id = self._get_active_episode_id(envelope)
        if episode_id:
            envelope.metadata["episode_id"] = episode_id

        # Отправляем в Memory LLM processor
        llm_result = self.memory_llm_processor(envelope)

        # Пишем trace: memory_llm_done
        if self.trace_store:
            self.trace_store.append_worker_trace(trace_id, {
                "trace_id": trace_id,
                "stage": "memory_llm_done",
                "created_at": datetime.now().isoformat(),
                "job_id": job.job_id,
                "event_id": envelope.event_id,
                "proposal_count": len(llm_result.proposals) if hasattr(llm_result, 'proposals') else 0,
                "proposals": [p.to_dict() if hasattr(p, 'to_dict') else p for p in llm_result.proposals] if hasattr(llm_result, 'proposals') else [],
            })

        # Проверяем, нужно ли обрабатывать
        if not llm_result.should_process or not llm_result.proposals:
            LOGGER.debug(f"Job {job.job_id}: no proposals, skipping")
            self.job_queue.complete(job.job_id)
            return

        # Передаём proposals в Governor
        governor_result = self.governor(llm_result.proposals, envelope)

        # Пишем trace: governor_done
        if self.trace_store:
            artifacts_list = governor_result.artifacts if hasattr(governor_result, 'artifacts') else []
            self.trace_store.append_worker_trace(trace_id, {
                "trace_id": trace_id,
                "stage": "governor_done",
                "created_at": datetime.now().isoformat(),
                "job_id": job.job_id,
                "event_id": envelope.event_id,
                "decisions": [d.to_dict() if hasattr(d, 'to_dict') else d for d in governor_result.decisions] if hasattr(governor_result, 'decisions') else [],
                "created_ids": [a.get("artifact_id") if isinstance(a, dict) else a.artifact_id for a in artifacts_list if hasattr(a, 'artifact_id') or isinstance(a, dict)],
                "updated_ids": [],
                "superseded_ids": [],
            })

        # Обновляем векторный индекс (если есть)
        # governor_result.artifacts может быть list[dict] или list[MemoryArtifact]
        artifacts_list = governor_result.artifacts if hasattr(governor_result, 'artifacts') else []
        indexed_ids = []

        if self.vector_index_updater and artifacts_list:
            for artifact in artifacts_list:
                # Проверяем статус — может быть атрибут или ключ dict
                status = None
                if hasattr(artifact, 'status'):
                    # MemoryArtifact
                    status = artifact.status
                elif isinstance(artifact, dict) and 'status' in artifact:
                    # dict
                    status = artifact.get('status')

                if status == "active":
                    self.vector_index_updater(artifact)
                    indexed_ids.append(
                        artifact.artifact_id if hasattr(artifact, 'artifact_id') else artifact.get("artifact_id", "")
                    )

        # Пишем trace: vector_index_done
        if self.trace_store and indexed_ids:
            self.trace_store.append_worker_trace(trace_id, {
                "trace_id": trace_id,
                "stage": "vector_index_done",
                "created_at": datetime.now().isoformat(),
                "job_id": job.job_id,
                "event_id": envelope.event_id,
                "indexed_ids": indexed_ids,
            })

        # Обновляем episode context после обработки
        if episode_id:
            self._update_episode_context(episode_id, envelope)

        # Помечаем задачу как выполненную
        self.job_queue.complete(job.job_id)

        # Пишем trace: job_done
        if self.trace_store:
            self.trace_store.append_worker_trace(trace_id, {
                "trace_id": trace_id,
                "stage": "job_done",
                "created_at": datetime.now().isoformat(),
                "job_id": job.job_id,
                "event_id": envelope.event_id,
                "status": "done",
            })

        LOGGER.debug(
            f"Job {job.job_id} completed: "
            f"artifacts={len(artifacts_list)}"
        )

    def _get_active_episode_id(self, envelope: MemoryEnvelope) -> str | None:
        """
        Получает ID активного эпизода.

        Args:
            envelope: Конверт события.

        Returns:
            ID эпизода или None.
        """
        try:
            from memory_core.planner.episode_planner import EpisodePlanner
            
            # Используем self.artifact_store если есть
            artifact_store = self.artifact_store
            if not artifact_store:
                return None
            
            planner = EpisodePlanner(
                artifact_store=artifact_store,
                event_store=self.event_store,
            )
            
            # Получаем или создаём эпизод
            episode = planner.get_or_create_episode(
                session_id=envelope.session_id,
                workspace_id=envelope.workspace_id,
                envelope=envelope,
            )
            
            if episode:
                return episode.episode_id
        except Exception as e:
            LOGGER.debug(f"Failed to get active episode: {e}")
        
        return None

    def _update_episode_context(self, episode_id: str, envelope: MemoryEnvelope) -> None:
        """
        Обновляет контекст эпизода.

        Args:
            episode_id: ID эпизода.
            envelope: Конверт события.
        """
        try:
            from memory_core.planner.episode_planner import EpisodePlanner
            
            # Здесь можно обновить last_event_at и context_tags
            # Для пока просто логируем
            LOGGER.debug(f"Episode {episode_id[:8]}... updated with event {envelope.event_id[:8]}...")
        except Exception as e:
            LOGGER.debug(f"Failed to update episode context: {e}")

    def _run_loop(self) -> None:
        """Основной цикл воркера."""
        LOGGER.info(f"Worker {self.config.worker_id} run loop started")
        
        loop_count = 0

        while not self._stop_event.is_set():
            loop_count += 1
            
            if not self.config.enabled:
                LOGGER.debug(f"Worker {self.config.worker_id}: Disabled, sleeping...")
                time.sleep(self.config.poll_interval_sec)
                continue

            # Обрабатываем пакет задач
            jobs_count = 0
            LOGGER.debug(f"Worker {self.config.worker_id}: Attempting to process jobs (loop {loop_count})...")
            
            while jobs_count < self.config.max_jobs_per_cycle:
                if self._stop_event.is_set():
                    break

                if not self.process_one_job():
                    # Нет задач — выходим из внутреннего цикла
                    break

                jobs_count += 1

            # Если задач не было — ждём следующего опроса
            if jobs_count == 0:
                LOGGER.debug(f"Worker {self.config.worker_id}: No jobs, sleeping for {self.config.poll_interval_sec}s...")
                time.sleep(self.config.poll_interval_sec)

        LOGGER.info(f"Worker {self.config.worker_id} run loop exited after {loop_count} loops")


class BackgroundWorkerPool:
    """
    Пул воркеров для параллельной обработки.

    Управляет несколькими BackgroundWorker.
    """

    def __init__(self, num_workers: int = 2):
        """
        Инициализирует пул.

        Args:
            num_workers: Количество воркеров.
        """
        self.num_workers = num_workers
        self._workers: list[BackgroundWorker] = []

    def add_worker(self, worker: BackgroundWorker) -> None:
        """Добавляет воркера в пул."""
        if len(self._workers) >= self.num_workers:
            raise ValueError(f"Max workers ({self.num_workers}) reached")
        self._workers.append(worker)

    def start_all(self) -> None:
        """Запускает все воркеры."""
        for worker in self._workers:
            worker.start()

    def stop_all(self, timeout_sec: float = 5.0) -> None:
        """Останавливает все воркеры."""
        for worker in self._workers:
            worker.stop(timeout_sec=timeout_sec)

    def get_stats(self) -> list[dict[str, Any]]:
        """Получает статистику всех воркеров."""
        return [w.get_stats() for w in self._workers]

    @property
    def workers(self) -> list[BackgroundWorker]:
        """Возвращает список воркеров."""
        return self._workers
