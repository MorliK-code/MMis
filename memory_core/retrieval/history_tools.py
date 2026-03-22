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
