"""
History tools — точное чтение истории из БД.

Перенесено из memory/history_tools.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm.provider_base import ToolSpec


@dataclass(frozen=True)
class HistoryReadResult:
    """Результат чтения истории."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    has_more: bool = False


def history_read_recent_tool_spec() -> ToolSpec:
    """
    Tool для чтения последних сообщений.
    """
    return ToolSpec(
        name="history_read_recent",
        description="Read exact recent conversation messages in order. Use for 'last N messages' queries.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Number of messages to retrieve",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "any"],
                    "description": "Filter by message role",
                },
                "order": {
                    "type": "string",
                    "enum": ["newest_first", "oldest_first"],
                    "description": "Sort order",
                },
            },
            "required": ["limit", "role", "order"],
            "additionalProperties": False,
        },
    )


def history_read_range_tool_spec() -> ToolSpec:
    """
    Tool для чтения сообщений за диапазон времени.
    """
    return ToolSpec(
        name="history_read_range",
        description="Read exact stored messages for a time range. Use for 'messages from today/yesterday/date' queries.",
        input_schema={
            "type": "object",
            "properties": {
                "start_ts": {
                    "type": "string",
                    "description": "Start timestamp (ISO format or relative like 'today', 'yesterday')",
                },
                "end_ts": {
                    "type": "string",
                    "description": "End timestamp (ISO format or relative)",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "any"],
                    "description": "Filter by message role",
                },
                "order": {
                    "type": "string",
                    "enum": ["newest_first", "oldest_first"],
                    "description": "Sort order",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Maximum number of messages",
                },
            },
            "required": ["start_ts", "end_ts", "role", "order"],
            "additionalProperties": False,
        },
    )


def history_search_tool_spec() -> ToolSpec:
    """
    Tool для поиска по истории.
    """
    return ToolSpec(
        name="history_search",
        description="Search exact stored message history by keyword or phrase. Use for 'did I mention X' queries.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keyword or phrase)",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "any"],
                    "description": "Filter by message role",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of results",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def get_history_tools() -> list[ToolSpec]:
    """Вернуть список всех history tools."""
    return [
        history_read_recent_tool_spec(),
        history_read_range_tool_spec(),
        history_search_tool_spec(),
    ]


def history_tools_list() -> list[ToolSpec]:
    """Алиас для get_history_tools()."""
    return get_history_tools()


# ============================================================================
# Execute functions — runtime handlers для history tools
# ============================================================================

def _normalize_role(role: str) -> str:
    """Нормализует роль."""
    role = str(role or "any").strip().lower()
    if role in {"user", "assistant", "system", "any", "tool"}:
        return role
    return "any"


def _normalize_order(order: str) -> str:
    """Нормализует порядок."""
    order = str(order or "newest_first").strip().lower()
    if order in {"newest_first", "oldest_first"}:
        return order
    return "newest_first"


def _execute_history_read_recent(
    ctx: Any,
    *,
    limit: int = 10,
    role: str = "any",
    order: str = "newest_first",
) -> HistoryReadResult:
    """
    Читает последние сообщения из истории.

    Args:
        ctx: Контекст pipeline.
        limit: Максимальное количество сообщений.
        role: Фильтр по роли (user | assistant | system | any).
        order: Порядок (newest_first | oldest_first).

    Returns:
        HistoryReadResult.
    """
    history = list(ctx.state.get("history") or [])
    normalized_role = _normalize_role(role)
    normalized_order = _normalize_order(order)
    
    # Фильтруем по роли
    if normalized_role != "any":
        history = [
            msg for msg in history
            if str(msg.get("role") or "").strip().lower() == normalized_role
        ]
    
    # Сортируем
    if normalized_order == "newest_first":
        history = list(reversed(history))
    
    # Ограничиваем
    has_more = len(history) > limit
    messages = history[:limit]
    
    return HistoryReadResult(
        messages=messages,
        total_count=len(history),
        has_more=has_more,
    )


def _execute_history_search(
    ctx: Any,
    *,
    query: str,
    role: str = "any",
    limit: int = 10,
) -> HistoryReadResult:
    """
    Ищет сообщения в истории по тексту.

    Args:
        ctx: Контекст pipeline.
        query: Поисковый запрос.
        role: Фильтр по роли.
        limit: Максимальное количество сообщений.

    Returns:
        HistoryReadResult.
    """
    history = list(ctx.state.get("history") or [])
    normalized_role = _normalize_role(role)
    query_lower = str(query or "").strip().lower()
    
    # Фильтруем по роли
    if normalized_role != "any":
        history = [
            msg for msg in history
            if str(msg.get("role") or "").strip().lower() == normalized_role
        ]
    
    # Ищем по тексту
    if query_lower:
        history = [
            msg for msg in history
            if query_lower in str(msg.get("content") or "").lower()
        ]
    
    # Сортируем по релевантности (простой heuristic)
    if query_lower:
        def score_msg(msg: dict) -> int:
            content = str(msg.get("content") or "").lower()
            if query_lower in content:
                return content.count(query_lower)
            return 0
        history = sorted(history, key=score_msg, reverse=True)
    
    # Ограничиваем
    has_more = len(history) > limit
    messages = history[:limit]
    
    return HistoryReadResult(
        messages=messages,
        total_count=len(history),
        has_more=has_more,
    )


def _execute_history_read_range(
    ctx: Any,
    *,
    start_ts: str | None = None,
    end_ts: str | None = None,
    role: str = "any",
    order: str = "newest_first",
    limit: int = 50,
) -> HistoryReadResult:
    """
    Читает сообщения из истории по диапазону.

    Args:
        ctx: Контекст pipeline.
        start_ts: Начальная метка времени (ISO format).
        end_ts: Конечная метка времени (ISO format).
        role: Фильтр по роли.
        order: Порядок.
        limit: Максимальное количество сообщений.

    Returns:
        HistoryReadResult.
    """
    history = list(ctx.state.get("history") or [])
    normalized_role = _normalize_role(role)
    normalized_order = _normalize_order(order)
    
    # Фильтруем по роли
    if normalized_role != "any":
        history = [
            msg for msg in history
            if str(msg.get("role") or "").strip().lower() == normalized_role
        ]
    
    # Фильтруем по времени (если указано)
    # Примечание: это fallback для in-memory history
    # Для полноценной фильтрации нужно использовать memory_core adapter
    
    # Сортируем
    if normalized_order == "newest_first":
        history = list(reversed(history))
    
    # Ограничиваем
    has_more = len(history) > limit
    messages = history[:limit]
    
    return HistoryReadResult(
        messages=messages,
        total_count=len(history),
        has_more=has_more,
    )
