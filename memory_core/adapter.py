"""
Memory Core Adapter - адаптер для интеграции memory_core в существующий MMis.

Этот файл позволяет постепенно переводить проект на новую память,
сохраняя обратную совместимость со старыми интерфейсами.
"""

from typing import Any
from memory_core.bootstrap.service_factory import build_memory_service, MemoryServiceConfig
from memory_core.schemas import MemoryEnvelope, MemoryQuery
from memory_core.facade import MemoryService


class MemoryCoreAdapter:
    """
    Адаптер memory_core для использования в main.py и brain.py.

    Предоставляет упрощённый интерфейс для миграции со старого MemoryManager.
    """

    def __init__(
        self,
        db_path: str = "data/memory_core/memory.db",
        vector_path: str = "data/memory_core/vector",
        default_workspace: str = "global",
        default_namespace: str = "default",
        top_k: int = 8,
    ):
        """
        Инициализирует адаптер.

        Args:
            db_path: Путь к SQLite базе данных.
            vector_path: Путь к векторному индексу.
            default_workspace: Workspace по умолчанию.
            default_namespace: Namespace по умолчанию.
            top_k: Количество результатов по умолчанию.
        """
        self.config = MemoryServiceConfig(
            db_path=db_path,
            vector_path=vector_path,
            default_workspace=default_workspace,
            default_namespace=default_namespace,
            top_k=top_k,
        )

        self.service = build_memory_service(self.config)
        self._current_workspace = default_workspace
        self._current_session = "default"

    def ingest_event(
        self,
        text: str,
        source_kind: str = "user",
        payload_type: str = "message",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Добавляет событие в память.

        Args:
            text: Текст события.
            source_kind: Тип источника (user | assistant | tool | system).
            payload_type: Тип контента (message | tool_result | state_update).
            metadata: Дополнительные метаданные.
            session_id: ID сессии.
            workspace_id: ID workspace.

        Returns:
            Результат ingest.
        """
        envelope = MemoryEnvelope(
            source_kind=source_kind,
            payload_type=payload_type,
            text=text,
            metadata=metadata or {},
            workspace_id=workspace_id or self._current_workspace,
            session_id=session_id or self._current_session,
        )

        return self.service.ingest_event(envelope)

    def query(
        self,
        text: str,
        workspace_id: str | None = None,
        top_k: int | None = None,
        include_citations: bool = True,
    ) -> dict[str, Any]:
        """
        Выполняет запрос к памяти.

        Args:
            text: Текст запроса.
            workspace_id: ID workspace.
            top_k: Количество результатов.
            include_citations: Включать ли цитаты.

        Returns:
            Результат запроса с context_blocks.
        """
        query = MemoryQuery(
            text=text,
            workspace_id=workspace_id or self._current_workspace,
            top_k=top_k or self.config.top_k,
            include_citations=include_citations,
        )

        result = self.service.query(query)

        return {
            "context_blocks": result.context_blocks,
            "hits": result.hits,
            "citations": result.citations,
        }

    def get_context(
        self,
        query_text: str,
        workspace_id: str | None = None,
    ) -> str:
        """
        Получает контекст для LLM.

        Args:
            query_text: Текст запроса.
            workspace_id: ID workspace.

        Returns:
            Текстовый контекст.
        """
        result = self.query(query_text, workspace_id)

        if not result["context_blocks"]:
            return ""

        return "\n\n".join(result["context_blocks"])

    def set_current_workspace(self, workspace_id: str) -> None:
        """Устанавливает текущий workspace."""
        self._current_workspace = workspace_id
        self.service.set_current_workspace(workspace_id)

    def get_current_workspace(self) -> str:
        """Получает текущий workspace."""
        return self._current_workspace

    def set_current_session(self, session_id: str) -> None:
        """Устанавливает текущую сессию."""
        self._current_session = session_id

    def get_stats(self) -> dict[str, Any]:
        """Получает статистику памяти."""
        return self.service.get_stats()

    def inspect(self, kind: str = "events", limit: int = 50) -> dict[str, Any]:
        """
        Инспектирует память.

        Args:
            kind: Тип инспекции (events | artifacts | workspaces | profile | stats).
            limit: Максимальное количество результатов.

        Returns:
            Результат инспекции.
        """
        from memory_core.schemas import MemoryInspectRequest

        request = MemoryInspectRequest(kind=kind, limit=limit)
        return self.service.inspect(request)

    def debug_snapshot(self, limit: int = 50) -> dict[str, Any]:
        """
        Метод для совместимости со старым debug_snapshot.

        Args:
            limit: Максимальное количество результатов.

        Returns:
            Снимок состояния памяти.
        """
        return {
            "events": self.inspect(kind="events", limit=limit),
            "artifacts": self.inspect(kind="artifacts", limit=limit),
            "stats": self.inspect(kind="stats"),
        }

    def close(self) -> None:
        """Закрывает соединения (если требуется)."""
        # SQLite не требует явного закрытия в большинстве случаев
        pass


# Глобальный экземпляр для использования в main.py
_memory_core_adapter: MemoryCoreAdapter | None = None


def get_memory_core_adapter() -> MemoryCoreAdapter:
    """Получает глобальный экземпляр адаптера."""
    global _memory_core_adapter
    if _memory_core_adapter is None:
        _memory_core_adapter = MemoryCoreAdapter()
    return _memory_core_adapter


def init_memory_core(
    db_path: str = "data/memory_core/memory.db",
    vector_path: str = "data/memory_core/vector",
    **kwargs,
) -> MemoryCoreAdapter:
    """
    Инициализирует глобальный экземпляр memory_core.

    Args:
        db_path: Путь к базе данных.
        vector_path: Путь к векторному индексу.
        **kwargs: Дополнительные аргументы для MemoryCoreAdapter.

    Returns:
        Инициализированный адаптер.
    """
    global _memory_core_adapter
    _memory_core_adapter = MemoryCoreAdapter(
        db_path=db_path,
        vector_path=vector_path,
        **kwargs,
    )
    return _memory_core_adapter
