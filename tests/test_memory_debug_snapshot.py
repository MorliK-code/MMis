from __future__ import annotations

from types import SimpleNamespace

from core import DebugTrace
from memory import build_memory_debug_snapshot


def test_build_memory_debug_snapshot_returns_expected_projection() -> None:
    ctx = SimpleNamespace(
        state={
            "debug_trace": DebugTrace(
                request_id="req-42",
                user_text="return to the memory plan",
                memory_retrieval={"selected_total": 3},
                memory_governor={"decisions": [{"action": "supersede_old"}]},
                active_profile={"namespace": "conv-1"},
                identity_core={"protected_keys": ["addressing.canonical_name"]},
                persona_snapshot={"mood": "neutral"},
                active_task={"task_id": "task:episode:memory-plan", "status": "active"},
                prompt_pack={"token_estimate": 321},
                final_answer_meta={
                    "route": "chat",
                    "agent_loop": True,
                    "agent_tool_calls": 1,
                    "agent_passes": 2,
                    "memory_reasoning_used": True,
                    "memory_reasoning_snapshot": {"profile_facts": {"prefers_short_answers": True}},
                },
            )
        }
    )

    row = build_memory_debug_snapshot(ctx)

    assert row == {
        "request_id": "req-42",
        "user_text": "return to the memory plan",
        "retrieval": {"selected_total": 3},
        "governor": {"decisions": [{"action": "supersede_old"}]},
        "active_profile": {"namespace": "conv-1"},
        "identity_core": {"protected_keys": ["addressing.canonical_name"]},
        "persona_snapshot": {"mood": "neutral"},
        "active_task": {"task_id": "task:episode:memory-plan", "status": "active"},
        "prompt_pack": {"token_estimate": 321},
        "tool_loop": {
            "agent_loop": True,
            "agent_tool_calls": 1,
            "agent_passes": 2,
            "memory_reasoning_used": True,
            "memory_reasoning_snapshot": {"profile_facts": {"prefers_short_answers": True}},
        },
        "final_answer_meta": {
            "route": "chat",
            "agent_loop": True,
            "agent_tool_calls": 1,
            "agent_passes": 2,
            "memory_reasoning_used": True,
            "memory_reasoning_snapshot": {"profile_facts": {"prefers_short_answers": True}},
        },
    }


def test_build_memory_debug_snapshot_returns_empty_dict_without_trace() -> None:
    ctx = SimpleNamespace(state={})

    assert build_memory_debug_snapshot(ctx) == {}
