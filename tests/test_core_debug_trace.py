from __future__ import annotations

from core import DebugTrace


def test_debug_trace_to_dict_returns_expected_shape() -> None:
    trace = DebugTrace(
        request_id="req-123",
        user_text="how are we doing on memory?",
        memory_retrieval={"selected_count": 4},
        memory_governor={"action": "supersede_old"},
        active_profile={"identity.name": {"value": "Pasha"}},
        identity_core={"snapshot": {"addressing": {"canonical_name": "Паша"}}},
        persona_snapshot={"mood": "neutral"},
        active_task={"task_id": "task:episode:memory-plan", "status": "active"},
        prompt_pack={"memory_blocks": ["SELF_FACTS", "ACTIVE_TASK"]},
        final_answer_meta={"route": "chat"},
    )

    row = trace.to_dict()

    assert row["request_id"] == "req-123"
    assert row["user_text"] == "how are we doing on memory?"
    assert dict(row["memory_retrieval"] or {}) == {"selected_count": 4}
    assert dict(row["memory_governor"] or {}) == {"action": "supersede_old"}
    assert dict(row["active_profile"] or {}) == {"identity.name": {"value": "Pasha"}}
    assert dict(row["identity_core"] or {}) == {"snapshot": {"addressing": {"canonical_name": "Паша"}}}
    assert dict(row["persona_snapshot"] or {}) == {"mood": "neutral"}
    assert dict(row["active_task"] or {}) == {
        "task_id": "task:episode:memory-plan",
        "status": "active",
    }
    assert dict(row["prompt_pack"] or {}) == {"memory_blocks": ["SELF_FACTS", "ACTIVE_TASK"]}
    assert dict(row["final_answer_meta"] or {}) == {"route": "chat"}


def test_debug_trace_defaults_are_compact_dicts() -> None:
    row = DebugTrace().to_dict()

    assert row == {
        "request_id": "",
        "user_text": "",
        "memory_retrieval": {},
        "memory_governor": {},
        "active_profile": {},
        "identity_core": {},
        "persona_snapshot": {},
        "active_task": {},
        "prompt_pack": {},
        "final_answer_meta": {},
    }
