"""
State Reducer — нормализация runtime state.

Вместо эвристик ("?" → open_questions, "решили" → current_decisions),
state reducer использует чёткие правила для обновления:
- active_task
- open_questions
- current_decisions
- active_preferences
- recent_topics
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
    LLM-операция будет в state_reducer_llm.py (отдельно).
    
    Args:
        user_text: Текст пользователя
        assistant_text: Текст ассистента
        current_state: Текущее состояние
    
    Returns:
        StateUpdate: Обновления для state
    """
    update = StateUpdate()
    
    # === 1. open_questions — только явные вопросы без ответа ===
    # Не добавляем каждый "?" — только если это явный незакрытый вопрос
    if "?" in user_text and not _is_rhetorical_question(user_text):
        # Проверяем, не был ли вопрос уже закрыт в assistant_text
        if not _question_answered_in_reply(user_text, assistant_text):
            update.open_questions.append(_extract_question(user_text))
    
    # === 2. current_decisions — только явные маркеры решений ===
    decision_markers = [
        "решили что", "договорились", "пришли к выводу",
        "итог:", "вывод:", "значит",
    ]
    for marker in decision_markers:
        if marker in assistant_text.lower():
            # Извлекаем предложение с решением
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
    # Извлекаем из context_tags если есть
    context_tags = current_state.get("context_tags", {})
    topic = context_tags.get("topic", "")
    if topic and topic not in update.recent_topics:
        # Нормализуем topic label
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
    """Проверить, риторический ли вопрос."""
    rhetorical_patterns = [
        "как дела", "что нового", "понимаешь",
        "you know", "right?", "okay?",
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in rhetorical_patterns)


def _question_answered_in_reply(question: str, reply: str) -> bool:
    """Проверить, закрыт ли вопрос в ответе."""
    # Простая эвристика: если reply длинный и нет "?" — вопрос закрыт
    if len(reply) > 50 and "?" not in reply:
        return True
    return False


def _extract_question(text: str) -> str:
    """Извлечь вопрос из текста."""
    # Находим предложение с "?"
    sentences = text.replace("?", "?\n").split("\n")
    for s in sentences:
        if "?" in s:
            return s.strip()
    return text.strip()[:100]


def _extract_sentence_with_marker(text: str, marker: str) -> str | None:
    """Извлечь предложение с маркером."""
    text_lower = text.lower()
    idx = text_lower.find(marker)
    if idx < 0:
        return None
    
    # Ищем начало предложения
    start = text.rfind(". ", 0, idx)
    if start < 0:
        start = 0
    else:
        start += 2
    
    # Ищем конец предложения
    end = text.find(". ", idx)
    if end < 0:
        end = len(text)
    
    sentence = text[start:end].strip()
    return sentence[:200] if len(sentence) > 200 else sentence


def _normalize_topic_label(topic: str) -> str:
    """Нормализовать topic label."""
    # Убираем лишние символы, приводим к lowercase
    label = topic.strip().lower()
    # Заменяем пробелы на подчёркивания
    label = label.replace(" ", "_")
    # Убираем специальные символы
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
    
    # active_task patch
    if update.active_task_patch:
        active_task = dict(new_state.get("active_task") or {})
        active_task.update(update.active_task_patch)
        new_state["active_task"] = active_task
    
    # open_questions — заменяем полностью
    if update.open_questions:
        new_state["open_questions"] = list(update.open_questions)
    
    # current_decisions — заменяем полностью
    if update.current_decisions:
        new_state["current_decisions"] = list(update.current_decisions)
    
    # active_preferences — добавляем к существующим
    if update.active_preferences:
        existing = list(new_state.get("active_preferences") or [])
        for pref in update.active_preferences:
            if pref not in existing:
                existing.append(pref)
        new_state["active_preferences"] = existing
    
    # recent_topics — добавляем к существующим (max 10)
    if update.recent_topics:
        existing = list(new_state.get("recent_topics") or [])
        for topic in update.recent_topics:
            if topic not in existing:
                existing.append(topic)
        new_state["recent_topics"] = existing[-10:]
    
    # relation_state patch
    if update.relation_state_patch:
        relation_state = dict(new_state.get("relation_state") or {})
        relation_state.update(update.relation_state_patch)
        new_state["relation_state"] = relation_state
    
    return new_state
