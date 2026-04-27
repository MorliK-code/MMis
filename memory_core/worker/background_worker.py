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
import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from memory_core.storage.job_queue_store import JobQueueStore, IngestJob
from memory_core.storage.event_store import EventStore
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.schemas import MemoryEnvelope
from memory_core.inspect.trace_store import MemoryTraceStore
from memory_core.topic import TopicMaintenanceService, TopicStore
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
    shutdown_idle_timeout_sec: float = 0.0  # 0 = отключено, >0 = таймаут простоя в секундах


@dataclass(slots=True)
class WorkerStats:
    """
    Статистика воркера.
    """
    jobs_processed: int = 0
    jobs_succeeded: int = 0
    jobs_failed: int = 0
    jobs_retried: int = 0
    interrupt_count: int = 0
    requeue_count: int = 0
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
            "interrupt_count": self.interrupt_count,
            "requeue_count": self.requeue_count,
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
        memory_llm_processor: MemoryLLMProcessorFn | Any,
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
        self._pause_event = threading.Event()  # Event для паузы (установлен = пауза)
        self._wake_event = threading.Event()
        self._running = False
        self._memory_llm_provider_unloaded = False
        self._scheduler_mode = self._resolve_scheduler_mode()
        self.on_memory_queue_idle: Callable[[], None] | None = None

    def _memory_llm_control_method(self, method_name: str):
        processor = getattr(self, "memory_llm_processor", None)
        if processor is None:
            return None
        method = getattr(processor, method_name, None)
        if callable(method):
            return method
        return None

    @staticmethod
    def _resolve_scheduler_mode() -> str:
        try:
            from memory_core.config_manager import get_memory_core_config

            mode = str(get_memory_core_config().memory_llm_scheduler_mode or "strict").strip().lower()
            if mode in {"strict", "cooperative"}:
                return mode
        except Exception:
            pass
        return "strict"

    def _is_strict_scheduler_mode(self) -> bool:
        return self._scheduler_mode == "strict"

    def _get_memory_llm_interrupt_epoch(self) -> int:
        try:
            from memory_core.adapter import _get_memory_llm_interrupt_epoch as runtime_interrupt_epoch
            return int(runtime_interrupt_epoch())
        except Exception:
            return 0

    def _get_memory_llm_preemption(self, *, since_epoch: int | None = None) -> InterruptedError | None:
        try:
            from memory_core.processors.memory_llm_processor import _get_memory_llm_preemption as runtime_preemption
        except Exception:
            return None
        try:
            return runtime_preemption(since_epoch=since_epoch)
        except Exception:
            return None

    def _raise_if_memory_llm_preempted(self, *, since_epoch: int | None = None, stage: str = "") -> None:
        preemption = self._get_memory_llm_preemption(since_epoch=since_epoch)
        if preemption is None:
            return
        if stage:
            LOGGER.info(f"Worker {self.config.worker_id}: preempted at stage '{stage}', requeueing job")
        raise preemption

    def start(self) -> None:
        """Запускает воркер в фоновом потоке."""
        if self._running and self._thread is not None and self._thread.is_alive():
            LOGGER.warning(f"Worker {self.config.worker_id} already running")
            return

        self._running = False
        self._thread = None

        self._stop_event.clear()
        self._pause_event.clear()  # Не на паузе по умолчанию
        self._wake_event.set()
        
        # НЕ daemon поток — должен завершиться корректно перед выходом
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"MemoryWorker-{self.config.worker_id}",
            daemon=False,  # Важно: False для корректного shutdown
        )
        self._thread.start()
        self._running = True
        LOGGER.info(f"Worker {self.config.worker_id} started")

    def stop(self, timeout_sec: float = 5.0) -> None:
        """
        Останавливает воркер и Memory LLM provider.

        Args:
            timeout_sec: Таймаут ожидания остановки.
        """
        if not self._running:
            return

        LOGGER.info(f"Worker {self.config.worker_id}: Stop requested (timeout={timeout_sec}s)")

        # Сначала ставим на паузу, чтобы остановить новые задачи
        self._pause_event.set()

        # Сигнал остановки
        self._stop_event.set()

        # Ждём завершения потока
        if self._thread:
            self._thread.join(timeout=timeout_sec)
            
            # Проверяем, завершился ли поток
            if self._thread.is_alive():
                LOGGER.warning(
                    f"Worker {self.config.worker_id}: Thread did not terminate gracefully, "
                    f"forcing stop"
                )
        
        self._running = False
        
        # Критически важно: останавливаем Memory LLM provider для освобождения VRAM
        self._shutdown_memory_llm_provider()
        
        LOGGER.info(f"Worker {self.config.worker_id} stopped")

    def _shutdown_memory_llm_provider(self) -> None:
        """Останавливает Memory LLM provider для освобождения VRAM."""
        try:
            shutdown = self._memory_llm_control_method("shutdown")
            if shutdown is not None:
                LOGGER.info(f"Worker {self.config.worker_id}: Shutting down Memory LLM provider...")
                shutdown()
                LOGGER.info(f"Worker {self.config.worker_id}: Memory LLM provider shutdown complete")
                self._memory_llm_provider_unloaded = True
        except Exception as exc:
            LOGGER.warning(f"Worker {self.config.worker_id}: Failed to shutdown Memory LLM provider: {exc}")

    def pause(self) -> None:
        """
        Ставит воркер на паузу и уступает управление Memory LLM.
        
        Воркер продолжит работу, но не будет брать новые задачи из очереди.
        """
        self._pause_event.set()
        LOGGER.debug(f"Worker {self.config.worker_id} paused")
        
        self._yield_memory_llm_provider()

    def resume(self) -> None:
        """
        Снимает воркер с паузы и возобновляет Memory LLM.
        
        Воркер продолжит обработку задач из очереди.
        Memory LLM готов к обработке.
        """
        self._pause_event.clear()
        self._wake_event.set()
        LOGGER.debug(f"Worker {self.config.worker_id} resumed")
        
        # Возобновляем Memory LLM
        self._resume_memory_llm_provider()

    def resume_idle(self) -> None:
        """Resume worker polling without touching the Memory LLM provider."""
        self._pause_event.clear()
        self._wake_event.set()
        LOGGER.debug(f"Worker {self.config.worker_id} resumed idle without Memory LLM provider wake")

    def wake(self) -> None:
        """Будит воркер для немедленной проверки очереди."""
        self._wake_event.set()

    def _wait_or_wake(self, timeout_sec: float) -> None:
        timeout = max(0.0, float(timeout_sec or 0.0))
        if timeout <= 0.0:
            return
        self._wake_event.wait(timeout=timeout)
        self._wake_event.clear()

    def _yield_memory_llm_provider(self) -> None:
        """Yield Memory LLM provider without unloading it."""
        try:
            yield_control = self._memory_llm_control_method("yield_control")
            if yield_control is not None:
                LOGGER.info(f"Worker {self.config.worker_id}: Yielding Memory LLM provider...")
                yield_control()
                LOGGER.info(f"Worker {self.config.worker_id}: Memory LLM provider yielded")
                self._memory_llm_provider_unloaded = False
        except Exception as exc:
            LOGGER.debug(f"Worker {self.config.worker_id}: Failed to yield Memory LLM provider: {exc}")

    def _unload_memory_llm_provider(self) -> None:
        """Unload Memory LLM provider explicitly."""
        try:
            unload = self._memory_llm_control_method("unload")
            if unload is not None:
                LOGGER.info(f"Worker {self.config.worker_id}: Unloading Memory LLM provider...")
                unload()
                LOGGER.info(f"Worker {self.config.worker_id}: Memory LLM provider unloaded")
                self._memory_llm_provider_unloaded = True
        except Exception as exc:
            LOGGER.debug(f"Worker {self.config.worker_id}: Failed to unload Memory LLM provider: {exc}")

    def _pause_memory_llm_provider(self) -> None:
        self._yield_memory_llm_provider()

    def _resume_memory_llm_provider(self) -> None:
        """Возобновляет Memory LLM provider после паузы."""
        try:
            resume = self._memory_llm_control_method("resume")
            if resume is not None:
                LOGGER.info(f"Worker {self.config.worker_id}: Resuming Memory LLM provider...")
                resume()
                LOGGER.info(f"Worker {self.config.worker_id}: Memory LLM provider resumed")
                self._memory_llm_provider_unloaded = False
        except Exception as exc:
            LOGGER.debug(f"Worker {self.config.worker_id}: Failed to resume Memory LLM provider: {exc}")

    def is_paused(self) -> bool:
        """Проверяет, на паузе ли воркер."""
        return self._pause_event.is_set()

    def is_running(self) -> bool:
        """Проверяет, запущен ли воркер."""
        if self._running and self._thread is not None and not self._thread.is_alive():
            self._running = False
            self._thread = None
        return self._running

    def get_stats(self) -> dict[str, Any]:
        """Получает статистику воркера."""
        return {
            "worker_id": self.config.worker_id,
            "running": self._running,
            "paused": self.is_paused(),
            "memory_llm_provider_unloaded": bool(self._memory_llm_provider_unloaded),
            "memory_llm_locked": self._is_memory_llm_locked(),
            "config": {
                "poll_interval_sec": self.config.poll_interval_sec,
                "max_jobs_per_cycle": self.config.max_jobs_per_cycle,
                "job_types": self.config.job_types,
                "enabled": bool(self.config.enabled),
                "scheduler_mode": self._scheduler_mode,
            },
            "stats": self.stats.to_dict(),
        }

    def process_one_job(self) -> bool:
        """
        Обрабатывает одну задачу из очереди.

        Returns:
            True если задача обработана, False если задач нет.
        """
        import sys
        print(f"[WORKER] {self.config.worker_id}: Attempting to dequeue job...", flush=True)
        sys.stdout.flush()

        # Проверяем сигнал остановки перед получением задачи
        if self._stop_event.is_set():
            print(f"[WORKER] {self.config.worker_id}: Stop requested", flush=True)
            sys.stdout.flush()
            return False

        # Проверяем глобальную блокировку Memory LLM
        # Если lock установлен — основная модель отвечает, не обрабатываем
        if self._is_strict_scheduler_mode() and self._is_memory_llm_locked():
            print(f"[WORKER] {self.config.worker_id}: Memory LLM locked", flush=True)
            sys.stdout.flush()
            LOGGER.debug(
                "Worker %s: strict scheduler blocked dequeue by memory lock",
                self.config.worker_id,
            )
            return False

        # Получаем задачу из очереди
        job = self.job_queue.dequeue(
            worker_id=self.config.worker_id,
            job_type=self.config.job_types[0] if len(self.config.job_types) == 1 else None,
        )

        print(f"[WORKER] {self.config.worker_id}: Dequeue result: {job is not None}", flush=True)
        sys.stdout.flush()

        if not job:
            return False

        try:
            if self._memory_llm_provider_unloaded:
                self._resume_memory_llm_provider()
            print(f"[WORKER] {self.config.worker_id}: Processing job {job.job_id[:8]}...", flush=True)
            sys.stdout.flush()
            self._process_job(job)
            
            # Проверяем сигнал остановки после обработки
            if self._stop_event.is_set():
                print(f"[WORKER] {self.config.worker_id}: Stop requested after job", flush=True)
                sys.stdout.flush()
                return False
                
            self.stats.jobs_succeeded += 1
            print(f"[WORKER] {self.config.worker_id}: Job succeeded", flush=True)
            sys.stdout.flush()
            return True
        except InterruptedError as e:
            # Прервано основной моделью — возвращаем задачу в очередь
            print(f"[WORKER] {self.config.worker_id}: Job interrupted, requeuing", flush=True)
            sys.stdout.flush()
            LOGGER.info(
                "Worker %s: job %s interrupted by main priority, requeue immediately",
                self.config.worker_id,
                job.job_id,
            )
            requeue_now = getattr(self.job_queue, "requeue_immediately", None)
            self.stats.interrupt_count += 1
            if callable(requeue_now):
                if requeue_now(job.job_id, ""):
                    self.stats.requeue_count += 1
            else:
                self.job_queue.fail(job.job_id, "", retry=True)
                self.stats.requeue_count += 1
            self.stats.jobs_retried += 1
            return False  # FIXED: Stop burst on interruption
        except Exception as e:
            print(f"[WORKER] {self.config.worker_id}: Job failed: {e}", flush=True)
            sys.stdout.flush()
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
            print(f"[WORKER] {self.config.worker_id}: Total processed: {self.stats.jobs_processed}", flush=True)
            sys.stdout.flush()
            self._notify_memory_queue_idle_if_needed()

    def _notify_memory_queue_idle_if_needed(self) -> None:
        callback = getattr(self, "on_memory_queue_idle", None)
        if not callable(callback):
            return
        try:
            stats = dict(self.job_queue.get_stats() or {})
            by_type = dict(stats.get("by_type") or {})
            pending = int(by_type.get("memory_llm_process") or 0)
        except Exception:
            return
        if pending > 0:
            return
        try:
            callback()
        except Exception as exc:
            LOGGER.debug(f"Worker {self.config.worker_id}: memory idle callback failed: {exc}")

    def _is_memory_llm_locked(self) -> bool:
        """Проверяет, заблокирован ли Memory LLM основной моделью."""
        try:
            from memory_core.adapter import _memory_llm_lock
            if _memory_llm_lock is not None:
                return _memory_llm_lock.locked()
        except Exception:
            pass
        return False

    def _process_job(self, job: IngestJob) -> None:
        """
        Обрабатывает задачу.

        Args:
            job: Задача для обработки.
        """
        from datetime import datetime
        
        trace_id = f"trace_{job.job_id}"
        interrupt_epoch = self._get_memory_llm_interrupt_epoch()
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
        if hasattr(self.memory_llm_processor, "process"):
            llm_result = self.memory_llm_processor.process(envelope)
        else:
            llm_result = self.memory_llm_processor(envelope)

        self._raise_if_memory_llm_preempted(since_epoch=interrupt_epoch, stage="after_memory_llm")

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
            self._raise_if_memory_llm_preempted(since_epoch=interrupt_epoch, stage="before_complete_empty_result")
            LOGGER.debug(f"Job {job.job_id}: no proposals, skipping")
            self.job_queue.complete(job.job_id)
            return

        # Передаём proposals в Governor
        self._raise_if_memory_llm_preempted(since_epoch=interrupt_epoch, stage="before_governor")
        governor_result = self.governor(llm_result.proposals, envelope)
        self._raise_if_memory_llm_preempted(since_epoch=interrupt_epoch, stage="after_governor")

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
        self._raise_if_memory_llm_preempted(since_epoch=interrupt_epoch, stage="before_topic_summary_refresh")
        self._refresh_topic_thread(envelope)
        self._maintain_topic_threads(envelope)

        artifacts_list = governor_result.artifacts if hasattr(governor_result, 'artifacts') else []
        indexed_ids = []

        if self.vector_index_updater and artifacts_list:
            for artifact in artifacts_list:
                self._raise_if_memory_llm_preempted(since_epoch=interrupt_epoch, stage="before_vector_index")
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
            self._raise_if_memory_llm_preempted(since_epoch=interrupt_epoch, stage="before_episode_update")
            self._update_episode_context(episode_id, envelope)

        # Помечаем задачу как выполненную
        self._raise_if_memory_llm_preempted(since_epoch=interrupt_epoch, stage="before_complete")
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

    def _refresh_topic_thread(self, envelope: MemoryEnvelope) -> None:
        topic_thread_id = str(dict(envelope.metadata or {}).get("topic_thread_id") or "").strip()
        if not topic_thread_id or self.artifact_store is None:
            return
        try:
            TopicMaintenanceService(TopicStore(self.artifact_store)).maintain_thread(
                topic_thread_id,
                workspace_id=envelope.workspace_id,
                allow_summary_rebuild=True,
                trigger="background_worker",
            )
        except Exception as exc:
            LOGGER.debug(f"Failed to refresh topic thread {topic_thread_id}: {exc}")

    def _maintain_topic_threads(self, envelope: MemoryEnvelope) -> None:
        if self.artifact_store is None:
            return
        metadata = dict(envelope.metadata or {})
        visible_chat_id = str(metadata.get("visible_chat_id") or metadata.get("conversation_id") or "").strip()
        topic_thread_id = str(metadata.get("topic_thread_id") or "").strip()
        try:
            TopicMaintenanceService(TopicStore(self.artifact_store)).maintain_scope(
                visible_chat_id=visible_chat_id,
                workspace_id=envelope.workspace_id,
                session_id=envelope.session_id,
                active_thread_id=topic_thread_id,
                limit=32,
            )
        except Exception as exc:
            LOGGER.debug(f"Failed to maintain topic scope for {visible_chat_id or envelope.session_id}: {exc}")

    def _run_loop(self) -> None:
        """Основной цикл воркера."""
        LOGGER.info(f"Worker {self.config.worker_id} run loop started")

        loop_count = 0
        idle_start_time: float | None = None  # Время начала простоя

        try:
            while not self._stop_event.is_set():
                loop_count += 1

                if not self.config.enabled:
                    LOGGER.debug(f"Worker {self.config.worker_id}: Disabled, sleeping...")
                    self._wait_or_wake(self.config.poll_interval_sec)
                    continue

                # Проверяем паузу - не обрабатываем задачи, но продолжаем цикл
                if self._pause_event.is_set():
                    LOGGER.debug(f"Worker {self.config.worker_id}: Paused, waiting...")
                    self._wait_or_wake(0.5)
                    continue

                # Обрабатываем пакет задач
                jobs_count = 0
                LOGGER.debug(f"Worker {self.config.worker_id}: Attempting to process jobs (loop {loop_count})...")

                while jobs_count < self.config.max_jobs_per_cycle:
                    if self._stop_event.is_set():
                        break

                    if self._pause_event.is_set():
                        # Пауза во время обработки - прерываем цикл
                        break

                    try:
                        processed = self.process_one_job()
                    except Exception as exc:
                        self.stats.jobs_failed += 1
                        self.stats.last_error = str(exc)
                        LOGGER.exception(f"Worker {self.config.worker_id}: Unhandled process_one_job error: {exc}")
                        break

                    if not processed:
                        # Нет задач — выходим из внутреннего цикла
                        break

                    jobs_count += 1
                    idle_start_time = None  # Сбрасываем простой при обработке задачи

                # Если задач не было — проверяем idle timeout
                if jobs_count == 0:
                    # Проверяем настройку shutdown_idle_timeout
                    if self.config.shutdown_idle_timeout_sec > 0:
                        if idle_start_time is None:
                            idle_start_time = time.time()

                        idle_duration = time.time() - idle_start_time
                        if idle_duration >= self.config.shutdown_idle_timeout_sec:
                            if not self._memory_llm_provider_unloaded:
                                LOGGER.info(
                                    f"Worker {self.config.worker_id}: Idle timeout "
                                    f"({idle_duration:.1f}s >= {self.config.shutdown_idle_timeout_sec}s), unloading Memory LLM provider..."
                                )
                                self._unload_memory_llm_provider()
                        else:
                            LOGGER.debug(
                                f"Worker {self.config.worker_id}: Idle for {idle_duration:.1f}s, "
                                f"timeout in {self.config.shutdown_idle_timeout_sec - idle_duration:.1f}s"
                            )
                    # Не сбрасываем idle_start_time здесь, чтобы отслеживать общий простой

                    LOGGER.info(f"Worker {self.config.worker_id}: No jobs, sleeping for {self.config.poll_interval_sec}s...")
                    self._wait_or_wake(self.config.poll_interval_sec)
                else:
                    # Были задачи — спим небольшое время перед следующим циклом
                    # чтобы не перегружать CPU если задач очень много
                    self._wait_or_wake(0.05)
        except Exception as exc:
            self.stats.last_error = str(exc)
            LOGGER.exception(f"Worker {self.config.worker_id}: Run loop crashed: {exc}")

        self._running = False
        self._thread = None
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
