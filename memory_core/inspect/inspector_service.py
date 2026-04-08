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

from memory_core.topic import TopicStore, TopicSummaryBuilder


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
    _topic_store: TopicStore | None = None

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
        self._topic_store = None

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
                self._topic_store = TopicStore(self._artifact_store)
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
            "worker_jobs_processed": 0,
            "worker_jobs_succeeded": 0,
            "worker_jobs_failed": 0,
            "worker_jobs_retried": 0,
            "jobs_done_confirmed": 0,
            "jobs_done_first_run": 0,
            "jobs_done_after_restart": 0,
            "jobs_restarted_total": 0,
            "jobs_inferred_interruptions_total": 0,
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

            try:
                all_jobs = self._job_queue.list_jobs(limit=5000)
                for job in all_jobs:
                    lifecycle = self._get_job_lifecycle(job.job_id, attempts=job.attempts)
                    restart_count = int(lifecycle.get("restart_count", 0) or 0)
                    inferred_interruptions = int(lifecycle.get("inferred_interruptions", 0) or 0)
                    if restart_count > 0:
                        overview["jobs_restarted_total"] += 1
                    if inferred_interruptions > 0:
                        overview["jobs_inferred_interruptions_total"] += inferred_interruptions
                    if str(job.status or "") == "done" and bool(lifecycle.get("has_job_done")):
                        overview["jobs_done_confirmed"] += 1
                        if restart_count > 0:
                            overview["jobs_done_after_restart"] += 1
                        else:
                            overview["jobs_done_first_run"] += 1
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
                overview["worker_jobs_retried"] = stats_data.get("jobs_retried", 0)
            except Exception:
                overview["worker_enabled"] = False
        else:
            overview["worker_enabled"] = False

        return overview

    def _get_job_lifecycle(self, job_id: str, *, attempts: int = 0) -> dict[str, Any]:
        if not self._trace_store or not str(job_id or "").strip():
            return {
                "run_count": 0,
                "restart_count": 0,
                "inferred_interruptions": 0,
                "has_memory_llm_done": False,
                "has_governor_done": False,
                "has_job_done": False,
                "last_stage": "",
                "completed_at": None,
            }

        try:
            rows = list(self._trace_store.get_worker_trace(f"trace_{job_id}") or [])
        except Exception:
            rows = []

        stages = [str(row.get("stage") or "") for row in rows if isinstance(row, dict)]
        run_count = stages.count("event_loaded")
        restart_count = max(0, run_count - 1)
        retry_count = max(0, int(attempts or 0))
        inferred_interruptions = max(0, restart_count - retry_count)

        completed_at = None
        for row in reversed(rows):
            if isinstance(row, dict) and str(row.get("stage") or "") == "job_done":
                completed_at = row.get("created_at")
                break

        return {
            "run_count": run_count,
            "restart_count": restart_count,
            "inferred_interruptions": inferred_interruptions,
            "has_memory_llm_done": "memory_llm_done" in stages,
            "has_governor_done": "governor_done" in stages,
            "has_job_done": "job_done" in stages,
            "last_stage": stages[-1] if stages else "",
            "completed_at": completed_at,
        }

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
                    "prompt_view": artifact.metadata.get("prompt_view", "") if hasattr(artifact, 'metadata') and artifact.metadata else "",
                    "exposure_mode": artifact.metadata.get("exposure_mode", "") if hasattr(artifact, 'metadata') and artifact.metadata else "",
                    "sensitivity": artifact.metadata.get("sensitivity", "") if hasattr(artifact, 'metadata') and artifact.metadata else "",
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
                self._build_job_payload(job)
                for job in jobs
            ]
        except Exception:
            return []

    def _build_job_payload(self, job: Any) -> dict[str, Any]:
        lifecycle = self._get_job_lifecycle(job.job_id, attempts=job.attempts)
        run_count = int(lifecycle.get("run_count", 0) or 0)
        restart_count = int(lifecycle.get("restart_count", 0) or 0)
        inferred_interruptions = int(lifecycle.get("inferred_interruptions", 0) or 0)
        has_job_done = bool(lifecycle.get("has_job_done"))

        completion_label = ""
        if str(job.status or "") == "done":
            if has_job_done:
                completion_label = "confirmed_done_after_restart" if restart_count > 0 else "confirmed_done_first_run"
            else:
                completion_label = "done_without_worker_trace"
        elif restart_count > 0:
            completion_label = "restarted_not_finished"

        return {
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
            "run_count": run_count,
            "restart_count": restart_count,
            "inferred_interruptions": inferred_interruptions,
            "trace_last_stage": lifecycle.get("last_stage"),
            "trace_has_job_done": has_job_done,
            "trace_completed_at": lifecycle.get("completed_at"),
            "completion_label": completion_label,
        }

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
                "prompt_view": meta.get("prompt_view", ""),
                "exposure_mode": meta.get("exposure_mode", ""),
                "sensitivity": meta.get("sensitivity", ""),
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
            artifacts = self._artifact_store.get_by_source_event(event_id)

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

    def list_topics(
        self,
        *,
        visible_chat_id: str = "",
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if not self._topic_store:
            return []

        threads = self._topic_store.list_threads(
            visible_chat_id=str(visible_chat_id or "").strip(),
            status=status,
            limit=limit,
        )
        return [self._serialize_topic(self._ensure_topic_summary(thread)) for thread in threads]

    def get_topic_details(self, thread_id: str) -> dict[str, Any] | None:
        if not self._topic_store:
            return None

        thread = self._topic_store.get_thread(thread_id)
        if thread is None:
            return None
        thread = self._ensure_topic_summary(thread)

        recent_artifacts = self._topic_store.list_thread_artifacts(
            thread.thread_id,
            workspace_id=thread.workspace_id,
            limit=50,
        )
        episode_count = sum(1 for row in recent_artifacts if str(row.get("artifact_type") or "") == "episode_event")
        thread_meta = dict(thread.metadata or {})
        open_questions: list[str] = [
            str(item).strip()
            for item in list(thread_meta.get("open_questions") or [])
            if str(item).strip()
        ]
        current_decisions: list[str] = [
            str(item).strip()
            for item in list(thread_meta.get("current_decisions") or [])
            if str(item).strip()
        ]
        linked_tasks: list[dict[str, Any]] = []
        for row in recent_artifacts:
            metadata = dict(row.get("metadata") or {})
            for question in list(metadata.get("open_questions") or []):
                text = str(question or "").strip()
                if text and text not in open_questions:
                    open_questions.append(text)
            for decision in list(metadata.get("decisions") or metadata.get("current_decisions") or []):
                text = str(decision or "").strip()
                if text and text not in current_decisions:
                    current_decisions.append(text)
            if str(row.get("artifact_type") or "") in {"task", "task_state"}:
                linked_tasks.append(
                    {
                        "artifact_id": row.get("artifact_id"),
                        "text": row.get("text"),
                        "summary": row.get("summary"),
                        "status": metadata.get("task_status") or row.get("status"),
                    }
                )

        related_topics = [
            self._serialize_topic(related)
            for related_id in list(thread.related_thread_ids or [])
            for related in [self._topic_store.get_thread(related_id)]
            if related is not None
        ]
        return {
            **self._serialize_topic(thread, recent_artifacts=recent_artifacts),
            "episode_count": int(episode_count),
            "open_questions": open_questions[:20],
            "current_decisions": current_decisions[:20],
            "linked_tasks": linked_tasks[:20],
            "related_topics": related_topics,
            "recent_artifacts": recent_artifacts[:50],
        }

    def _ensure_topic_summary(self, thread):
        if thread is None or not self._topic_store:
            return thread
        if not TopicSummaryBuilder.needs_refresh(thread):
            return thread
        try:
            TopicSummaryBuilder(self._topic_store).rebuild_thread(
                thread.thread_id,
                workspace_id=str(thread.workspace_id or "").strip(),
            )
        except Exception:
            return thread
        refreshed = self._topic_store.get_thread(thread.thread_id)
        return refreshed or thread

    def _serialize_topic(
        self,
        thread: Any,
        *,
        recent_artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if thread is None:
            return {}
        artifacts = list(recent_artifacts or self._topic_store.list_thread_artifacts(
            thread.thread_id,
            workspace_id=thread.workspace_id,
            limit=100,
        )) if self._topic_store else list(recent_artifacts or [])
        return {
            "thread_id": thread.thread_id,
            "visible_chat_id": thread.visible_chat_id,
            "workspace_id": thread.workspace_id,
            "session_id": thread.session_id,
            "topic_key": thread.topic_key,
            "title": thread.title,
            "status": thread.status,
            "summary": thread.summary,
            "tags": list(thread.tags or []),
            "related_thread_ids": list(thread.related_thread_ids or []),
            "created_at": thread.created_at,
            "updated_at": thread.updated_at,
            "metadata": dict(thread.metadata or {}),
            "artifact_count": len(artifacts),
            "recent_artifact_ids": [row.get("artifact_id") for row in artifacts[:5]],
        }

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
