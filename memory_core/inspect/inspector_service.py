"""
Memory Inspector Service — сервисный слой для Inspector UI.

Интегрирован с memory_core v2:
- JobQueueStore для очереди задач
- ArtifactStore для артефактов
- EventStore для событий
- MemoryInspector для инспекции
- BackgroundWorker для worker health
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MemoryInspectorService:
    """
    Тонкий адаптер для inspector UI.
    """

    memory_core: Any
    _job_queue: Any = None
    _artifact_store: Any = None
    _event_store: Any = None
    _inspector: Any = None
    _worker: Any = None
    _vector_index: Any = None
    _trace_store: Any = None

    def __init__(self, memory_core: Any):
        """Инициализирует сервис."""
        self.memory_core = memory_core
        self._job_queue = None
        self._artifact_store = None
        self._event_store = None
        self._inspector = None
        self._worker = None
        self._vector_index = None
        self._trace_store = None

        # Пытаемся получить доступ к внутренним компонентам memory_core
        if hasattr(self.memory_core, "service"):
            service = self.memory_core.service

            # Компоненты v2
            if hasattr(service, "job_queue"):
                self._job_queue = service.job_queue
            if hasattr(service, "worker"):
                self._worker = service.worker
            if hasattr(service, "inspector"):
                self._inspector = service.inspector
            if hasattr(service, "vector_index"):
                self._vector_index = service.vector_index
            if hasattr(service, "trace_store"):
                self._trace_store = service.trace_store

            # Хранилища
            if hasattr(service, "artifact_store"):
                self._artifact_store = service.artifact_store
            if hasattr(service, "event_store"):
                self._event_store = service.event_store

    def get_overview(self) -> dict[str, Any]:
        """Возвращает общую сводку."""
        overview = {
            "artifacts_total": 0,
            "artifacts_active": 0,
            "artifacts_superseded": 0,
            "artifacts_by_type": {},
            "events_total": 0,
            "jobs_total": 0,
            "jobs_queued": 0,
            "jobs_processing": 0,
            "jobs_retry_wait": 0,
            "jobs_dead": 0,
            "jobs_done": 0,
            "memory_llm_status": "idle",
            "worker_health": 0,
            "worker_enabled": False,
        }

        # Используем inspector если доступен
        if self._inspector:
            try:
                stats = self._inspector.inspect(kind="stats")
                overview["artifacts_total"] = stats.get("artifacts_count", 0)
                overview["events_total"] = stats.get("events_count", 0)
                overview["jobs_queued"] = stats.get("jobs_queued", 0)
                overview["jobs_processing"] = stats.get("jobs_processing", 0)
                overview["jobs_dead"] = stats.get("jobs_dead", 0)
            except Exception:
                pass

        # Получаем статистику jobs из job_queue
        if self._job_queue:
            try:
                job_stats = self._job_queue.get_stats()
                overview["jobs_total"] = sum([
                    job_stats.get("queued", 0),
                    job_stats.get("processing", 0),
                    job_stats.get("retry_wait", 0),
                    job_stats.get("done", 0),
                    job_stats.get("dead", 0),
                ])
                overview["jobs_queued"] = job_stats.get("queued", 0)
                overview["jobs_processing"] = job_stats.get("processing", 0)
                overview["jobs_retry_wait"] = job_stats.get("retry_wait", 0)
                overview["jobs_dead"] = job_stats.get("dead", 0)
                overview["jobs_done"] = job_stats.get("done", 0)
                overview["jobs_by_type"] = job_stats.get("by_type", {})
            except Exception:
                pass

        # Считаем артефакты по статусам и типам
        if self._artifact_store:
            try:
                artifacts = self._artifact_store.list_artifacts(limit=1000)
                overview["artifacts_total"] = len(artifacts)
                overview["artifacts_active"] = sum(1 for a in artifacts if a.status == "active")
                overview["artifacts_superseded"] = sum(1 for a in artifacts if a.status == "superseded")

                # По типам
                by_type: dict[str, int] = {}
                for a in artifacts:
                    t = a.artifact_type
                    by_type[t] = by_type.get(t, 0) + 1
                overview["artifacts_by_type"] = by_type
            except Exception as e:
                print(f"Error counting artifacts: {e}")
                pass

        # Worker health
        if self._worker:
            try:
                worker_stats = self._worker.get_stats()
                overview["worker_enabled"] = True
                running = worker_stats.get("running", False)
                overview["worker_health"] = 100 if running else 0
                overview["memory_llm_status"] = "processing" if running else "idle"
                
                # Детали воркера
                stats_data = worker_stats.get("stats", {})
                overview["worker_jobs_processed"] = stats_data.get("jobs_processed", 0)
                overview["worker_jobs_succeeded"] = stats_data.get("jobs_succeeded", 0)
                overview["worker_jobs_failed"] = stats_data.get("jobs_failed", 0)
            except Exception:
                overview["worker_enabled"] = False
        else:
            overview["worker_enabled"] = False

        return overview

    def list_artifacts(
        self,
        *,
        query: str = "",
        artifact_type: str = "all",
        include_superseded: bool = True,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Возвращает список артефактов."""
        if not self._artifact_store:
            return []

        try:
            if artifact_type == "all":
                artifacts = self._artifact_store.list_artifacts(limit=limit)
            else:
                artifacts = self._artifact_store.list_artifacts(
                    artifact_type=artifact_type,
                    limit=limit,
                )

            result = []
            for artifact in artifacts:
                # Фильтр по query
                if query and query.lower() not in artifact.text.lower():
                    continue

                # Фильтр по superseded
                if not include_superseded and artifact.status == "superseded":
                    continue

                # Конвертируем MemoryArtifact в dict
                result.append({
                    "id": artifact.artifact_id,
                    "type": artifact.artifact_type,
                    "text": artifact.text,
                    "summary": artifact.summary,
                    "status": artifact.status,
                    "confidence": float(artifact.metadata.get("confidence", 0.5)) if hasattr(artifact, 'metadata') and artifact.metadata else 0.5,
                    "scope": artifact.metadata.get("scope", "global") if hasattr(artifact, 'metadata') and artifact.metadata else "global",
                    "decay": artifact.metadata.get("decay", "slow") if hasattr(artifact, 'metadata') and artifact.metadata else "slow",
                    "tags": artifact.metadata.get("retrieve_when", []) if hasattr(artifact, 'metadata') and artifact.metadata else [],
                    "created_at": artifact.created_at,
                    "updated_at": artifact.updated_at,
                    "source_event_id": artifact.source_event_id,
                })

            return result
        except Exception as e:
            print(f"Error in list_artifacts: {e}")
            return []

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Возвращает список задач очереди."""
        if not self._job_queue:
            return []

        try:
            jobs = self._job_queue.list_jobs(status=status, limit=limit)
            return [
                {
                    "id": job.job_id,
                    "event_id": job.event_id,
                    "type": job.job_type,
                    "status": job.status,
                    "priority": job.priority,
                    "attempts": job.attempts,
                    "max_attempts": job.max_attempts,
                    "locked_by": job.locked_by,
                    "locked_at": job.locked_at,
                    "available_at": job.available_at,
                    "error_text": job.error_text,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                    "summary": f"Job {job.job_type} for event {job.event_id}",
                }
                for job in jobs
            ]
        except Exception:
            return []

    def get_pipeline_trace(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """
        Возвращает trace pipeline из trace_store.

        Args:
            limit: Максимальное количество.

        Returns:
            Список trace rows.
        """
        if not self._trace_store:
            return []
        
        try:
            rows = self._trace_store.list_traces(limit=limit)
            return [
                {
                    "trace_id": row.get("trace_id"),
                    "request_id": row.get("request_id"),
                    "conversation_id": row.get("conversation_id"),
                    "turn_id": row.get("turn_id"),
                    "created_at": row.get("created_at"),
                    "route": row.get("route"),
                    "user_text": row.get("user_text"),
                    "retrieval_summary": dict(row.get("pipeline", {})).get("memory_retrieval", {}),
                    "final_answer_meta": dict(row.get("pipeline", {})).get("final_answer_meta", {}),
                }
                for row in rows
            ]
        except Exception:
            return []

    def get_raw_event(self, event_id: str) -> dict[str, Any] | None:
        """Получает сырое событие."""
        if not self._event_store:
            return None

        try:
            event = self._event_store.get_by_id(event_id)
            return dict(event) if event else None
        except Exception:
            return None

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        """Получает артефакт по ID."""
        if not self._artifact_store:
            return None

        try:
            artifact = self._artifact_store.get_by_id(artifact_id)
            if not artifact:
                return None

            meta = artifact.metadata or {}
            return {
                "id": artifact.artifact_id,
                "type": artifact.artifact_type,
                "text": artifact.text,
                "summary": artifact.summary,
                "status": artifact.status,
                "confidence": float(meta.get("confidence", 0.5)),
                "scope": meta.get("scope", "global"),
                "decay": meta.get("decay", "slow"),
                "tags": meta.get("retrieve_when", []),
                "created_at": artifact.created_at,
                "updated_at": artifact.updated_at,
                "source_event_id": artifact.source_event_id,
                "metadata": meta,
            }
        except Exception:
            return None

    def get_event_trace(self, event_id: str) -> dict[str, Any] | None:
        """
        Получает полную трассировку события.

        Args:
            event_id: ID события.

        Returns:
            Трассировка с событием и связанными артефактами.
        """
        if not self._event_store or not self._artifact_store:
            return None

        try:
            # Получаем событие
            event = self._event_store.get_by_id(event_id)
            if not event:
                return None

            # Получаем связанные артефакты
            artifacts = self._artifact_store.get_by_event(event_id)

            # Получаем связанные jobs
            jobs = []
            if self._job_queue:
                all_jobs = self._job_queue.list_jobs(limit=500)
                jobs = [j for j in all_jobs if j.event_id == event_id]

            # Получаем trace из trace_store
            trace = None
            worker_trace = []
            if self._trace_store:
                trace = self._trace_store.find_by_event_id(event_id)
                if trace:
                    worker_trace = self._trace_store.get_worker_trace(trace.get("trace_id", ""))

            return {
                "event": dict(event) if event else None,
                "artifacts": [a.to_dict() for a in artifacts],
                "jobs": [
                    {
                        "id": j.job_id,
                        "type": j.job_type,
                        "status": j.status,
                        "attempts": j.attempts,
                        "error_text": j.error_text,
                    }
                    for j in jobs
                ],
                "trace": trace,
                "worker_trace": worker_trace,
                "artifact_count": len(artifacts),
                "job_count": len(jobs),
            }
        except Exception:
            return None

    def retry_job(self, job_id: str) -> dict[str, Any]:
        """
        Повторяет задачу.

        Args:
            job_id: ID задачи.

        Returns:
            Результат операции.
        """
        if not self._job_queue:
            return {"success": False, "error": "JobQueue not available"}

        try:
            job = self._job_queue.list_jobs(status="dead", limit=1)
            if not job or job[0].job_id != job_id:
                # Пытаемся найти в других статусах
                for status in ["queued", "processing", "retry_wait"]:
                    jobs = self._job_queue.list_jobs(status=status, limit=100)
                    for j in jobs:
                        if j.job_id == job_id:
                            job = [j]
                            break

            if not job:
                return {"success": False, "error": f"Job {job_id} not found"}

            # Сбрасываем задачу в queued
            self._job_queue.db.execute(
                """
                UPDATE ingest_jobs
                SET status = 'queued', attempts = 0, error_text = NULL,
                    locked_by = NULL, locked_at = NULL, available_at = ?
                WHERE job_id = ?
                """,
                (job[0].created_at, job_id),
            )

            return {"success": True, "job_id": job_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_job(self, job_id: str) -> dict[str, Any]:
        """
        Удаляет задачу.

        Args:
            job_id: ID задачи.

        Returns:
            Результат операции.
        """
        if not self._job_queue:
            return {"success": False, "error": "JobQueue not available"}

        try:
            self._job_queue.db.execute(
                "DELETE FROM ingest_jobs WHERE job_id = ?",
                (job_id,),
            )
            return {"success": True, "job_id": job_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_checks(self) -> list[dict[str, Any]]:
        """Возвращает список проверок."""
        artifacts = self.list_artifacts(limit=1000)
        jobs = self.list_jobs(limit=1000)

        checks = [
            {
                "name": "Artifacts loaded",
                "ok": isinstance(artifacts, list),
            },
            {
                "name": "Jobs loaded",
                "ok": isinstance(jobs, list),
            },
            {
                "name": "Artifact confidence within range",
                "ok": all(
                    isinstance(item.get("confidence", 0), (float, int)) and 0 <= float(item.get("confidence", 0)) <= 1
                    for item in artifacts
                ),
            },
            {
                "name": "Job statuses valid",
                "ok": all(
                    item.get("status") in {"queued", "processing", "retry_wait", "done", "dead"}
                    for item in jobs
                ),
            },
        ]

        # Дополнительные проверки для memory_core v2
        if self._job_queue:
            checks.append({
                "name": "JobQueue initialized",
                "ok": True,
            })

        if self._artifact_store:
            checks.append({
                "name": "ArtifactStore initialized",
                "ok": True,
            })

        if self._event_store:
            checks.append({
                "name": "EventStore initialized",
                "ok": True,
            })

        if self._worker:
            checks.append({
                "name": "BackgroundWorker enabled",
                "ok": True,
            })

        return checks
