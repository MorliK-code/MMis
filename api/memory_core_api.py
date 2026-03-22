"""
Memory Core API - endpoints для работы с новой памятью MMis.
"""

from __future__ import annotations

from typing import Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from memory_core.adapter import MemoryCoreAdapter, get_memory_core_adapter, init_memory_core
from utils.logger import get_logger


LOGGER = get_logger(__name__)


# === Request/Response Models ===

class MemoryIngestRequest(BaseModel):
    """Запрос на добавление события в память."""
    text: str = Field(..., description="Текст события")
    source_kind: str = Field(default="user", description="Тип источника (user|assistant|tool|system)")
    payload_type: str = Field(default="message", description="Тип контента")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Метаданные")
    session_id: str | None = Field(default=None, description="ID сессии")
    workspace_id: str | None = Field(default=None, description="ID workspace")


class MemoryIngestResponse(BaseModel):
    """Ответ на ingest события."""
    event_id: str
    processed: bool
    artifacts_created: int
    artifact_ids: list[str]
    error: str | None = None


class MemoryQueryRequest(BaseModel):
    """Запрос к памяти."""
    text: str = Field(..., description="Текст запроса")
    workspace_id: str | None = Field(default=None, description="ID workspace")
    top_k: int = Field(default=8, description="Количество результатов")
    include_citations: bool = Field(default=True, description="Включать цитаты")


class MemoryQueryResponse(BaseModel):
    """Ответ на запрос к памяти."""
    context_blocks: list[str]
    hits: list[dict[str, Any]]
    citations: list[dict[str, Any]]


class MemoryInspectRequest(BaseModel):
    """Запрос на инспекцию памяти."""
    kind: str = Field(default="events", description="Тип инспекции (events|artifacts|workspaces|profile|stats)")
    workspace_id: str | None = Field(default=None, description="ID workspace")
    limit: int = Field(default=50, description="Максимальное количество")
    event_id: str | None = Field(default=None, description="ID события для trace")


class MemoryInspectResponse(BaseModel):
    """Ответ инспекции."""
    items: list[dict[str, Any]] | None = None
    workspaces: list[dict[str, Any]] | None = None
    profile_facts: list[dict[str, Any]] | None = None
    stats: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None


class MemoryStatsResponse(BaseModel):
    """Статистика памяти."""
    events_count: int
    artifacts_count: int
    workspaces_count: int


class MemoryWorkspaceRequest(BaseModel):
    """Запрос на управление workspace."""
    workspace_id: str = Field(..., description="ID workspace")
    title: str | None = Field(default=None, description="Заголовок")
    goal: str | None = Field(default=None, description="Цель")


class MemoryWorkspaceResponse(BaseModel):
    """Ответ workspace."""
    workspace_id: str
    title: str
    goal: str
    metadata: dict[str, Any]


# === API Endpoints ===

def create_memory_core_router(adapter: MemoryCoreAdapter | None = None) -> Any:
    """
    Создаёт API router для memory_core.
    
    Args:
        adapter: Экземпляр адаптера (создаётся автоматически, если None).
        
    Returns:
        FastAPI router.
    """
    from fastapi import APIRouter
    
    if adapter is None:
        adapter = get_memory_core_adapter()
    
    router = APIRouter(prefix="/memory", tags=["memory_core"])
    
    @router.post("/ingest", response_model=MemoryIngestResponse)
    async def ingest_event(request: MemoryIngestRequest):
        """
        Добавляет событие в память.
        
        Принимает текст и метаданные, сохраняет в raw events
        и создаёт артефакты через процессоры.
        """
        try:
            result = adapter.ingest_event(
                text=request.text,
                source_kind=request.source_kind,
                payload_type=request.payload_type,
                metadata=request.metadata,
                session_id=request.session_id,
                workspace_id=request.workspace_id,
            )
            
            return MemoryIngestResponse(
                event_id=result["event_id"],
                processed=result["processed"],
                artifacts_created=result["artifacts_created"],
                artifact_ids=result.get("artifact_ids", []),
                error=None,
            )
        except Exception as e:
            LOGGER.exception("Failed to ingest event")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/query", response_model=MemoryQueryResponse)
    async def query_memory(request: MemoryQueryRequest):
        """
        Выполняет запрос к памяти.
        
        Ищет релевантные артефакты и возвращает контекст для LLM.
        """
        try:
            result = adapter.query(
                text=request.text,
                workspace_id=request.workspace_id,
                top_k=request.top_k,
                include_citations=request.include_citations,
            )
            
            return MemoryQueryResponse(
                context_blocks=result["context_blocks"],
                hits=result["hits"],
                citations=result["citations"],
            )
        except Exception as e:
            LOGGER.exception("Failed to query memory")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/context")
    async def get_context(
        text: str = Query(..., description="Текст запроса"),
        workspace_id: str | None = Query(None, description="ID workspace"),
    ):
        """
        Получает контекст для LLM.
        
        Упрощённый endpoint для быстрого получения контекста.
        """
        try:
            context = adapter.get_context(text, workspace_id)
            return {"context": context, "length": len(context) if context else 0}
        except Exception as e:
            LOGGER.exception("Failed to get context")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/inspect", response_model=MemoryInspectResponse)
    async def inspect_memory(request: MemoryInspectRequest):
        """
        Инспектирует память (для debug).
        
        Поддерживает различные виды инспекции:
        - events: список сырых событий
        - artifacts: список артефактов
        - workspaces: список workspace
        - profile: профильные факты
        - stats: статистика
        - trace: трассировка события
        """
        try:
            result = adapter.inspect(
                kind=request.kind,
                limit=request.limit,
            )
            
            response = MemoryInspectResponse()
            
            if request.kind == "workspaces":
                response.workspaces = result.get("workspaces", [])
            elif request.kind == "profile":
                response.profile_facts = result.get("profile_facts", [])
            elif request.kind == "stats":
                response.stats = result.get("stats", {})
            elif request.kind == "trace" and request.event_id:
                response.trace = result
            else:
                response.items = result.get("items", [])
            
            return response
        except Exception as e:
            LOGGER.exception("Failed to inspect memory")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/stats", response_model=MemoryStatsResponse)
    async def get_stats():
        """
        Получает статистику памяти.
        
        Возвращает количество событий, артефактов и workspace.
        """
        try:
            stats = adapter.get_stats()
            return MemoryStatsResponse(
                events_count=stats.get("events_count", 0),
                artifacts_count=stats.get("artifacts_count", 0),
                workspaces_count=stats.get("workspaces_count", 0),
            )
        except Exception as e:
            LOGGER.exception("Failed to get stats")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/workspaces")
    async def list_workspaces():
        """
        Получает список всех workspace.
        """
        try:
            result = adapter.inspect(kind="workspaces")
            return {"workspaces": result.get("workspaces", [])}
        except Exception as e:
            LOGGER.exception("Failed to list workspaces")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/workspace")
    async def set_workspace(request: MemoryWorkspaceRequest):
        """
        Устанавливает текущий workspace.
        """
        try:
            adapter.set_current_workspace(request.workspace_id)
            return {"workspace_id": request.workspace_id, "status": "ok"}
        except Exception as e:
            LOGGER.exception("Failed to set workspace")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/workspace/current")
    async def get_current_workspace():
        """
        Получает текущий workspace.
        """
        try:
            workspace_id = adapter.get_current_workspace()
            return {"workspace_id": workspace_id}
        except Exception as e:
            LOGGER.exception("Failed to get current workspace")
            raise HTTPException(status_code=500, detail=str(e))
    
    return router


def register_memory_core_api(app: FastAPI, adapter: MemoryCoreAdapter | None = None) -> None:
    """
    Регистрирует memory_core API в приложении.
    
    Args:
        app: FastAPI приложение.
        adapter: Экземпляр адаптера (опционально).
    """
    router = create_memory_core_router(adapter)
    app.include_router(router)
    LOGGER.info("Memory Core API registered at /memory")
