"""
Memory Inspector Router — FastAPI router для UI инспектора памяти.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from memory_core.inspect.inspector_service import MemoryInspectorService


def create_memory_inspector_router(memory_core: Any) -> APIRouter:
    """
    Создаёт router для Memory Inspector UI.

    Args:
        memory_core: Экземпляр MemoryCoreAdapter.

    Returns:
        FastAPI router.
    """
    router = APIRouter(tags=["memory-inspector"])
    service = MemoryInspectorService(memory_core=memory_core)
    page_path = Path(__file__).parent.parent / "memory_core" / "inspect" / "inspector_page.html"

    @router.get("/memory-core", response_class=HTMLResponse)
    def memory_inspector_page() -> HTMLResponse:
        """Страница Memory Inspector UI."""
        return HTMLResponse(
            content=page_path.read_text(encoding="utf-8"),
            headers={
                "Cache-Control": "no-store, no-cache, max-age=0, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @router.get("/api/memory-core/overview")
    def overview() -> dict[str, Any]:
        """Общая сводка по памяти."""
        return service.get_overview()

    @router.get("/api/memory-core/artifacts")
    def artifacts(
        query: str = Query(default=""),
        artifact_type: str = Query(default="all"),
        include_superseded: bool = Query(default=True),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        """Список артефактов с фильтрами."""
        return service.list_artifacts(
            query=query,
            artifact_type=artifact_type,
            include_superseded=include_superseded,
            limit=limit,
        )

    @router.get("/api/memory-core/jobs")
    def jobs(
        status: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        """Список задач очереди."""
        return service.list_jobs(status=status, limit=limit)

    @router.get("/api/memory-core/trace")
    def trace(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
        """Trace pipeline memory LLM."""
        return service.get_pipeline_trace(limit=limit)

    @router.get("/api/memory-core/runtime")
    def runtime(limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        """Runtime session state and hidden episodes."""
        return service.get_runtime(limit=limit)

    @router.get("/api/memory-core/topics")
    def topics(
        visible_chat_id: str = Query(default=""),
        status: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        """Список скрытых topic threads."""
        return service.list_topics(
            visible_chat_id=visible_chat_id,
            status=status,
            limit=limit,
        )

    @router.get("/api/memory-core/topics/{thread_id}")
    def topic_details(thread_id: str) -> dict[str, Any] | None:
        """Детали одной скрытой темы."""
        return service.get_topic_details(thread_id)

    @router.get("/api/memory-core/checks")
    def checks() -> list[dict[str, Any]]:
        """Встроенные проверки."""
        return service.get_checks()

    @router.get("/api/memory-core/events/{event_id}")
    def raw_event(event_id: str) -> dict[str, Any] | None:
        """Сырое событие по ID."""
        return service.get_raw_event(event_id)

    @router.get("/api/memory-core/artifacts/{artifact_id}")
    def artifact_by_id(artifact_id: str) -> dict[str, Any] | None:
        """Артефакт по ID."""
        return service.get_artifact(artifact_id)

    @router.get("/api/memory-core/events/{event_id}/trace")
    def event_trace(event_id: str) -> dict[str, Any] | None:
        """Полная трассировка события."""
        return service.get_event_trace(event_id)

    @router.post("/api/memory-core/jobs/{job_id}/retry")
    def retry_job(job_id: str) -> dict[str, Any]:
        """Повтор задачи."""
        return service.retry_job(job_id)

    @router.delete("/api/memory-core/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, Any]:
        """Удаление задачи."""
        return service.delete_job(job_id)

    return router
