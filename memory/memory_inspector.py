"""
Memory Inspector — отладка и инспекция memory retrieval.

Показывает:
- был ли memory tool доступен
- вызвала ли его модель
- почему не вызвала
- сработал ли fallback
- какие queries были собраны
- какие hits вернулись
- что попало в memory_native_state
- откуда появился dialog_summary
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentLoopMemoryInspection:
    """
    Инспекция работы memory tool в agent loop.
    """
    agent_loop_enabled: bool = False
    agent_loop_triggered: bool = False
    memory_gate_triggered: bool = False
    memory_tool_available: bool = False
    memory_tool_called: bool = False
    memory_tool_call_count: int = 0
    fallback_used: bool = False
    fallback_reason: str = ""
    
    queries: list[dict[str, Any]] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)
    native_state: dict[str, Any] = field(default_factory=dict)
    dialog_summary_source: str = ""
    
    timing: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_loop_enabled": bool(self.agent_loop_enabled),
            "agent_loop_triggered": bool(self.agent_loop_triggered),
            "memory_gate_triggered": bool(self.memory_gate_triggered),
            "memory_tool_available": bool(self.memory_tool_available),
            "memory_tool_called": bool(self.memory_tool_called),
            "memory_tool_call_count": int(self.memory_tool_call_count),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": str(self.fallback_reason),
            "queries": list(self.queries or []),
            "hits": list(self.hits or []),
            "native_state": dict(self.native_state or {}),
            "dialog_summary_source": str(self.dialog_summary_source),
            "timing": dict(self.timing or {}),
            "errors": list(self.errors or []),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AgentLoopMemoryInspection":
        d = dict(data or {})
        return cls(
            agent_loop_enabled=bool(d.get("agent_loop_enabled", False)),
            agent_loop_triggered=bool(d.get("agent_loop_triggered", False)),
            memory_gate_triggered=bool(d.get("memory_gate_triggered", False)),
            memory_tool_available=bool(d.get("memory_tool_available", False)),
            memory_tool_called=bool(d.get("memory_tool_called", False)),
            memory_tool_call_count=int(d.get("memory_tool_call_count", 0)),
            fallback_used=bool(d.get("fallback_used", False)),
            fallback_reason=str(d.get("fallback_reason", "")),
            queries=list(d.get("queries") or []),
            hits=list(d.get("hits") or []),
            native_state=dict(d.get("native_state") or {}),
            dialog_summary_source=str(d.get("dialog_summary_source", "")),
            timing=dict(d.get("timing") or {}),
            errors=list(d.get("errors") or []),
        )


class MemoryInspector:
    """
    Инспектор для отладки memory retrieval.
    
    Собирает информацию о работе memory tool и fallback.
    """
    
    def __init__(self):
        self._inspections: dict[str, AgentLoopMemoryInspection] = {}
        self._current_turn_id: str | None = None
        self._start_times: dict[str, float] = {}
    
    def start_turn(self, turn_id: str) -> None:
        """Начать новый turn."""
        self._current_turn_id = turn_id
        self._start_times[turn_id] = time.time()
    
    def end_turn(self, turn_id: str | None = None) -> AgentLoopMemoryInspection | None:
        """Завершить turn и вернуть инспекцию."""
        turn_id = turn_id or self._current_turn_id
        if not turn_id:
            return None
        
        # Добавляем timing
        if turn_id in self._inspections:
            inspection = self._inspections[turn_id]
            start_time = self._start_times.get(turn_id, 0.0)
            elapsed = time.time() - start_time
            # Обновляем timing с использованием replace для frozen dataclass
            timing = dict(inspection.timing)
            timing["total_elapsed_sec"] = round(elapsed, 3)
            object.__setattr__(inspection, 'timing', timing)
        
        return self._inspections.get(turn_id)
    
    def record_agent_loop_state(
        self,
        *,
        enabled: bool,
        triggered: bool = False,
        gate_triggered: bool = False,
    ) -> None:
        """Записать состояние agent loop."""
        turn_id = self._current_turn_id
        if not turn_id:
            return
        
        inspection = self._inspections.get(turn_id, AgentLoopMemoryInspection())
        # Обновляем поля
        object.__setattr__(inspection, 'agent_loop_enabled', bool(enabled))
        object.__setattr__(inspection, 'agent_loop_triggered', bool(triggered))
        object.__setattr__(inspection, 'memory_gate_triggered', bool(gate_triggered))
        self._inspections[turn_id] = inspection
    
    def record_memory_tool_state(
        self,
        *,
        available: bool,
        called: bool,
        call_count: int = 0,
        queries: list[dict[str, Any]] | None = None,
    ) -> None:
        """Записать состояние memory tool."""
        turn_id = self._current_turn_id
        if not turn_id:
            return
        
        inspection = self._inspections.get(turn_id, AgentLoopMemoryInspection())
        object.__setattr__(inspection, 'memory_tool_available', bool(available))
        object.__setattr__(inspection, 'memory_tool_called', bool(called))
        object.__setattr__(inspection, 'memory_tool_call_count', int(call_count))
        if queries:
            existing_queries = list(inspection.queries)
            existing_queries.extend(list(queries))
            object.__setattr__(inspection, 'queries', existing_queries)
        self._inspections[turn_id] = inspection
    
    def record_fallback_state(
        self,
        *,
        used: bool,
        reason: str = "",
        hits: list[dict[str, Any]] | None = None,
    ) -> None:
        """Записать состояние fallback."""
        turn_id = self._current_turn_id
        if not turn_id:
            return
        
        inspection = self._inspections.get(turn_id, AgentLoopMemoryInspection())
        object.__setattr__(inspection, 'fallback_used', bool(used))
        object.__setattr__(inspection, 'fallback_reason', str(reason))
        if hits:
            object.__setattr__(inspection, 'hits', list(hits))
        self._inspections[turn_id] = inspection
    
    def record_native_state(self, native_state: dict[str, Any]) -> None:
        """Записать memory native state."""
        turn_id = self._current_turn_id
        if not turn_id:
            return
        
        inspection = self._inspections.get(turn_id, AgentLoopMemoryInspection())
        object.__setattr__(inspection, 'native_state', dict(native_state or {}))
        self._inspections[turn_id] = inspection
    
    def record_dialog_summary_source(self, source: str) -> None:
        """Записать источник dialog_summary."""
        turn_id = self._current_turn_id
        if not turn_id:
            return
        
        inspection = self._inspections.get(turn_id, AgentLoopMemoryInspection())
        object.__setattr__(inspection, 'dialog_summary_source', str(source or ""))
        self._inspections[turn_id] = inspection
    
    def record_error(self, error: str) -> None:
        """Записать ошибку."""
        turn_id = self._current_turn_id
        if not turn_id:
            return
        
        inspection = self._inspections.get(turn_id, AgentLoopMemoryInspection())
        errors = list(inspection.errors)
        errors.append(str(error))
        object.__setattr__(inspection, 'errors', errors)
        self._inspections[turn_id] = inspection
    
    def get_inspection(self, turn_id: str | None = None) -> AgentLoopMemoryInspection | None:
        """Получить инспекцию для turn."""
        turn_id = turn_id or self._current_turn_id
        if not turn_id:
            return None
        return self._inspections.get(turn_id)
    
    def get_all_inspections(self) -> dict[str, AgentLoopMemoryInspection]:
        """Получить все инспекции."""
        return dict(self._inspections)
    
    def clear(self) -> None:
        """Очистить все инспекции."""
        self._inspections.clear()
        self._start_times.clear()
        self._current_turn_id = None
    
    def to_json(self, turn_id: str | None = None) -> str:
        """Экспортировать инспекцию в JSON."""
        if turn_id:
            inspection = self._inspections.get(turn_id)
            if inspection:
                return json.dumps(inspection.to_dict(), indent=2, ensure_ascii=False)
        else:
            all_inspections = {
                tid: insp.to_dict()
                for tid, insp in self._inspections.items()
            }
            return json.dumps(all_inspections, indent=2, ensure_ascii=False)
        return "{}"


def build_memory_inspector() -> MemoryInspector:
    """Создать memory inspector."""
    return MemoryInspector()
