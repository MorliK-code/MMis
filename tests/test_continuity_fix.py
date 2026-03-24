"""
Тесты на исправление липкой continuity и topic shift.

См. tasks/MMis_continuity_fix_detailed.md
"""

import pytest
from modules.character.persona_snapshot_builder import PersonaSnapshotBuilder


def test_persona_snapshot_builder_does_not_mark_followup_on_clear_topic_shift() -> None:
    """
    Тест: Persona builder не должен считать follow-up при явной смене темы.
    
    Сценарий:
    - Есть активная задача про "memory loop"
    - Пользователь задаёт вопрос про "глаза и зрение" — это явная смена темы
    - is_followup должен быть False
    - topic_shift должен быть True
    """
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot=None,
        memory_context={},
        state={
            "active_task": {
                "task_id": "task:memory-loop",
                "topic": "memory loop",
                "status": "active",
                "current_goal": "Fix continuity in memory system.",
            }
        },
        meta={
            "emotion": "neutral",
            "same_calendar_day": "true",
            "minutes_since_previous": "10",
            "current_user_message": "а как тебе идея сделать тебе глаза чтобы ты видела как я работаю",
        },
    )

    assert snapshot.relation_continuity.get("is_followup") is False, \
        "is_followup должен быть False при явной смене темы"
    assert snapshot.relation_continuity.get("topic_shift") is True, \
        "topic_shift должен быть True при смене темы"


def test_persona_snapshot_prefers_current_turn_emotion_over_old_state_mood() -> None:
    """
    Тест: Текущий turn mood должен побеждать старый state mood.
    
    Сценарий:
    - В state mood = "frustrated" (старое состояние)
    - В meta emotion = "neutral" (текущий turn)
    - snapshot.mood должен быть "neutral"
    """
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot=None,
        memory_context={},
        state={"mood": "frustrated"},
        meta={
            "emotion": "neutral",
            "current_user_message": "как тебе такая идея",
        },
    )

    assert snapshot.mood == "neutral", \
        "mood должен браться из текущего turn (meta.emotion), а не из старого state"


def test_looks_like_topic_shift_with_low_overlap() -> None:
    """
    Тест: _looks_like_topic_shift возвращает True при низком overlap.
    """
    active_task = {
        "topic": "memory loop",
        "summary": "Fix continuity bug in memory system",
        "current_goal": "Debug memory core",
    }
    
    # Сообщение с очень низким overlap
    message = "а как тебе идея сделать тебе глаза чтобы ты видела"
    
    assert PersonaSnapshotBuilder._looks_like_topic_shift(message, active_task) is True, \
        "Должен определить смену темы при низком overlap"


def test_looks_like_topic_shift_with_high_overlap() -> None:
    """
    Тест: _looks_like_topic_shift возвращает False при высоком overlap.
    """
    active_task = {
        "topic": "memory loop",
        "summary": "Fix continuity bug in memory system",
        "current_goal": "Debug memory core",
    }
    
    # Сообщение с высоким overlap (продолжение темы)
    message = "как исправить continuity bug в memory системе"
    
    assert PersonaSnapshotBuilder._looks_like_topic_shift(message, active_task) is False, \
        "Не должен определять смену темы при высоком overlap"


def test_looks_like_topic_shift_with_empty_message() -> None:
    """
    Тест: _looks_like_topic_shift возвращает False для пустого сообщения.
    """
    active_task = {"topic": "test"}
    
    assert PersonaSnapshotBuilder._looks_like_topic_shift("", active_task) is False, \
        "Пустое сообщение не должно считаться сменой темы"


def test_looks_like_topic_shift_with_empty_task() -> None:
    """
    Тест: _looks_like_topic_shift возвращает False для пустой задачи.
    """
    active_task = {}
    
    assert PersonaSnapshotBuilder._looks_like_topic_shift("test message", active_task) is False, \
        "Пустая задача не должна считаться сменой темы"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
