"""
State Reducer — нормализация runtime state.

Перенесено из memory/state_reducer.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StateUpdate:
    """Результат работы state reducer."""
    active_task_patch: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    current_decisions: list[str] = field(default_factory=list)
    active_preferences: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    relation_state_patch: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_task_patch": dict(self.active_task_patch or {}),
            "open_questions": list(self.open_questions or []),
            "current_decisions": list(self.current_decisions or []),
            "active_preferences": list(self.active_preferences or []),
            "recent_topics": list(self.recent_topics or []),
            "relation_state_patch": dict(self.relation_state_patch or {}),
        }


def reduce_state_for_turn(
    user_text: str,
    assistant_text: str,
    current_state: dict[str, Any],
) -> StateUpdate:
    """
    Нормализовать state для текущего turn.

    Это НЕ LLM-операция, а набор детерминированных правил.

    Args:
        user_text: Текст пользователя
        assistant_text: Текст ассистента
        current_state: Текущее состояние

    Returns:
        StateUpdate: Обновления для state
    """
    update = StateUpdate()

    # === 1. open_questions — только явные вопросы без ответа ===
    if "?" in user_text and not _is_rhetorical_question(user_text):
        if not _question_answered_in_reply(user_text, assistant_text):
            update.open_questions.append(_extract_question(user_text))

    # === 2. current_decisions — только явные маркеры решений ===
    decision_markers = [
        "решили что", "договорились", "пришли к выводу",
        "итог:", "вывод:", "значит",
    ]
    for marker in decision_markers:
        if marker in assistant_text.lower():
            decision = _extract_sentence_with_marker(assistant_text, marker)
            if decision and decision not in update.current_decisions:
                update.current_decisions.append(decision)

    # === 3. active_preferences — только явные предпочтения ===
    preference_markers = [
        "я предпочитаю", "я люблю", "мне нравится",
        "i prefer", "i like", "i love",
    ]
    for marker in preference_markers:
        if marker in user_text.lower():
            pref = _extract_sentence_with_marker(user_text, marker)
            if pref and pref not in update.active_preferences:
                update.active_preferences.append(pref)

    # === 4. recent_topics — каноничные labels, не сырой текст ===
    context_tags = current_state.get("context_tags", {})
    topic = context_tags.get("topic", "")
    if topic and topic not in update.recent_topics:
        update.recent_topics.append(_normalize_topic_label(topic))

    # === 5. active_task — обновляем если есть явные маркеры задачи ===
    task_markers = ["задача:", "цель:", "нужно:", "надо:"]
    for marker in task_markers:
        if marker in user_text.lower() or marker in assistant_text.lower():
            text = user_text if marker in user_text.lower() else assistant_text
            task_summary = _extract_sentence_with_marker(text, marker)
            if task_summary:
                update.active_task_patch["summary_short"] = task_summary

    return update


def _is_rhetorical_question(text: str) -> bool:
    rhetorical_patterns = [
        "как дела", "что нового", "понимаешь",
        "you know", "right?", "okay?",
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in rhetorical_patterns)


def _question_answered_in_reply(question: str, reply: str) -> bool:
    if len(reply) > 50 and "?" not in reply:
        return True
    return False


def _extract_question(text: str) -> str:
    sentences = text.replace("?", "?\n").split("\n")
    for s in sentences:
        if "?" in s:
            return s.strip()
    return text.strip()[:100]


def _extract_sentence_with_marker(text: str, marker: str) -> str | None:
    text_lower = text.lower()
    idx = text_lower.find(marker)
    if idx < 0:
        return None

    start = text.rfind(". ", 0, idx)
    if start < 0:
        start = 0
    else:
        start += 2

    end = text.find(". ", idx)
    if end < 0:
        end = len(text)

    sentence = text[start:end].strip()
    return sentence[:200] if len(sentence) > 200 else sentence


def _normalize_topic_label(topic: str) -> str:
    label = topic.strip().lower()
    label = label.replace(" ", "_")
    label = "".join(c for c in label if c.isalnum() or c == "_")
    return label[:50]


def merge_state_updates(
    current_state: dict[str, Any],
    update: StateUpdate,
) -> dict[str, Any]:
    """
    Применить StateUpdate к current_state.

    Args:
        current_state: Текущее состояние
        update: Обновления

    Returns:
        Новое состояние
    """
    new_state = dict(current_state or {})

    if update.active_task_patch:
        active_task = dict(new_state.get("active_task") or {})
        active_task.update(update.active_task_patch)
        new_state["active_task"] = active_task

    if update.open_questions:
        new_state["open_questions"] = list(update.open_questions)

    if update.current_decisions:
        new_state["current_decisions"] = list(update.current_decisions)

    if update.active_preferences:
        existing = list(new_state.get("active_preferences") or [])
        for pref in update.active_preferences:
            if pref not in existing:
                existing.append(pref)
        new_state["active_preferences"] = existing

    if update.recent_topics:
        existing = list(new_state.get("recent_topics") or [])
        for topic in update.recent_topics:
            if topic not in existing:
                existing.append(topic)
        new_state["recent_topics"] = existing[-10:]

    if update.relation_state_patch:
        relation_state = dict(new_state.get("relation_state") or {})
        relation_state.update(update.relation_state_patch)
        new_state["relation_state"] = relation_state

    return new_state
