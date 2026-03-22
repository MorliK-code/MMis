"""
Memory Governor — управление записью в память.

Read-path (чтение) — свободный для LLM:
- memory_retrieve (semantic search)
- history_read_recent (точные последние сообщения)
- history_search (поиск по истории)

Write-path (запись) — под governor:
- запись фактов
- обновление identity
- conflict resolution
- profile evolution
- state reducer merge

LLM не должна сама свободно писать всё подряд в память.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WriteAction(Enum):
    """Действие governor для записи."""
    ALLOW = "allow"
    BLOCK = "block"
    DEFER = "defer"  # Отложить до подтверждения
    MERGE = "merge"  # Объединить с существующим


@dataclass(frozen=True)
class WriteRequest:
    """Запрос на запись в память."""
    text: str
    memory_type: str  # fact, claim, identity_core, preference, decision
    namespace: str
    confidence: float = 0.5
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WriteDecision:
    """Решение governor."""
    action: WriteAction
    reason: str
    confidence: float = 0.0
    merged_text: str | None = None


def govern_write_request(request: WriteRequest, current_state: dict[str, Any]) -> WriteDecision:
    """
    Принять решение о записи в память.
    
    Правила:
    1. Факты о пользователе (identity_core) — только с высокой уверенностью
    2. Предпочтения — разрешены, но дедуплицируются
    3. Решения — разрешены, если есть явные маркеры
    4. Случайные утверждения — блокируются
    
    Args:
        request: Запрос на запись
        current_state: Текущее состояние памяти
    
    Returns:
        WriteDecision: Решение governor
    """
    text = request.text.strip()
    memory_type = request.memory_type.lower()
    
    # === 1. Проверка уверенности ===
    if request.confidence < 0.3:
        return WriteDecision(
            action=WriteAction.BLOCK,
            reason=f"confidence_too_low: {request.confidence:.2f} < 0.3",
            confidence=request.confidence,
        )
    
    # === 2. Проверка важности ===
    if request.importance < 0.2:
        return WriteDecision(
            action=WriteAction.BLOCK,
            reason=f"importance_too_low: {request.importance:.2f} < 0.2",
            confidence=request.confidence,
        )
    
    # === 3. Тип-специфичные правила ===
    if memory_type == "identity_core":
        # Факты о пользователе — только с очень высокой уверенностью
        if request.confidence < 0.8:
            return WriteDecision(
                action=WriteAction.DEFER,
                reason="identity_core_requires_high_confidence",
                confidence=request.confidence,
            )
        
        # Проверка на конфликты с существующими фактами
        existing_facts = _get_existing_identity_facts(current_state)
        if _has_conflict(text, existing_facts):
            return WriteDecision(
                action=WriteAction.MERGE,
                reason="conflict_with_existing_facts",
                confidence=request.confidence,
                merged_text=_merge_facts(text, existing_facts),
            )
        
        return WriteDecision(
            action=WriteAction.ALLOW,
            reason="identity_core_high_confidence",
            confidence=request.confidence,
        )
    
    elif memory_type == "preference":
        # Предпочтения — разрешены, но дедуплицируются
        existing_prefs = _get_existing_preferences(current_state)
        if _is_duplicate(text, existing_prefs):
            return WriteDecision(
                action=WriteAction.BLOCK,
                reason="duplicate_preference",
                confidence=request.confidence,
            )
        
        return WriteDecision(
            action=WriteAction.ALLOW,
            reason="preference_allowed",
            confidence=request.confidence,
        )
    
    elif memory_type == "decision":
        # Решения — разрешены если есть явные маркеры
        decision_markers = ["решили", "договорились", "пришли к выводу", "итог"]
        has_marker = any(m in text.lower() for m in decision_markers)
        
        if not has_marker:
            return WriteDecision(
                action=WriteAction.BLOCK,
                reason="no_decision_marker",
                confidence=request.confidence,
            )
        
        return WriteDecision(
            action=WriteAction.ALLOW,
            reason="decision_with_marker",
            confidence=request.confidence,
        )
    
    elif memory_type == "fact":
        # Общие факты — разрешены с умеренной уверенностью
        if request.confidence >= 0.5:
            return WriteDecision(
                action=WriteAction.ALLOW,
                reason="fact_allowed",
                confidence=request.confidence,
            )
        else:
            return WriteDecision(
                action=WriteAction.DEFER,
                reason="fact_requires_higher_confidence",
                confidence=request.confidence,
            )
    
    elif memory_type == "claim":
        # Утверждения — разрешены с умеренной уверенностью
        if request.confidence >= 0.5:
            return WriteDecision(
                action=WriteAction.ALLOW,
                reason="claim_allowed",
                confidence=request.confidence,
            )
        else:
            return WriteDecision(
                action=WriteAction.DEFER,
                reason="claim_requires_higher_confidence",
                confidence=request.confidence,
            )
    
    # По умолчанию — блокируем
    return WriteDecision(
        action=WriteAction.BLOCK,
        reason=f"unknown_memory_type: {memory_type}",
        confidence=request.confidence,
    )


def _get_existing_identity_facts(state: dict[str, Any]) -> list[str]:
    """Получить существующие факты identity."""
    identity_core = state.get("identity_core", {})
    facts = []
    
    # Из address
    addressing = identity_core.get("addressing", {})
    if addressing.get("canonical_name"):
        facts.append(f"name: {addressing['canonical_name']}")
    
    # Из boundaries
    boundaries = identity_core.get("boundaries", {})
    for key, value in boundaries.items():
        if value:
            facts.append(f"boundary: {key}")
    
    return facts


def _get_existing_preferences(state: dict[str, Any]) -> list[str]:
    """Получить существующие предпочтения."""
    return list(state.get("active_preferences") or [])


def _has_conflict(new_fact: str, existing_facts: list[str]) -> bool:
    """Проверить конфликт нового факта с существующими."""
    new_lower = new_fact.lower()
    for existing in existing_facts:
        existing_lower = existing.lower()
        # Простая проверка на противоречие
        if "не " in new_lower and "не " not in existing_lower:
            return True
        if "не " not in new_lower and "не " in existing_lower:
            return True
    return False


def _merge_facts(new_fact: str, existing_facts: list[str]) -> str:
    """Объединить новый факт с существующими."""
    # Простое объединение
    return f"{new_fact}; existing: {', '.join(existing_facts)}"


def _is_duplicate(new_pref: str, existing_prefs: list[str]) -> bool:
    """Проверить дубликат предпочтения."""
    new_lower = new_pref.lower()
    for existing in existing_prefs:
        if new_lower in existing.lower() or existing.lower() in new_lower:
            return True
    return False


def apply_write_decision(
    decision: WriteDecision,
    request: WriteRequest,
    current_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Применить решение governor к state.
    
    Args:
        decision: Решение governor
        request: Исходный запрос
        current_state: Текущее состояние
    
    Returns:
        Новое состояние (или неизмененное если BLOCK)
    """
    if decision.action == WriteAction.BLOCK:
        return current_state
    
    new_state = dict(current_state or {})
    
    if decision.action == WriteAction.ALLOW:
        # Применяем запись
        if request.memory_type.lower() == "preference":
            prefs = list(new_state.get("active_preferences") or [])
            if request.text not in prefs:
                prefs.append(request.text)
            new_state["active_preferences"] = prefs
        
        elif request.memory_type.lower() == "decision":
            decisions = list(new_state.get("current_decisions") or [])
            if request.text not in decisions:
                decisions.append(request.text)
            new_state["current_decisions"] = decisions
    
    elif decision.action == WriteAction.MERGE:
        # Применяем merged текст
        if decision.merged_text:
            if request.memory_type.lower() == "preference":
                prefs = list(new_state.get("active_preferences") or [])
                if decision.merged_text not in prefs:
                    prefs.append(decision.merged_text)
                new_state["active_preferences"] = prefs
    
    elif decision.action == WriteAction.DEFER:
        # Откладываем — записываем в pending
        pending = list(new_state.get("pending_writes") or [])
        pending.append({
            "text": request.text,
            "type": request.memory_type,
            "reason": decision.reason,
            "confidence": request.confidence,
        })
        new_state["pending_writes"] = pending[-10:]  # Max 10 pending
    
    return new_state
