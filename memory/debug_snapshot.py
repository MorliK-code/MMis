from __future__ import annotations

from typing import Any


def build_memory_debug_snapshot(ctx) -> dict[str, Any]:
    state = dict(getattr(ctx, "state", {}) or {})
    trace = state.get("debug_trace")
    if trace is None:
        return {}

    data = trace.to_dict() if hasattr(trace, "to_dict") else dict(trace or {})
    final_answer_meta = dict(data.get("final_answer_meta") or {})
    tool_loop = dict(
        data.get("tool_loop")
        or final_answer_meta.get("tool_loop")
        or {}
    )
    if not tool_loop:
        tool_loop = {
            "agent_loop": bool(final_answer_meta.get("agent_loop", False)),
            "agent_tool_calls": int(final_answer_meta.get("agent_tool_calls") or 0),
            "agent_passes": int(final_answer_meta.get("agent_passes") or 0),
            "memory_reasoning_used": bool(final_answer_meta.get("memory_reasoning_used", False)),
            "memory_reasoning_sections": [
                str(x).strip()
                for x in list(final_answer_meta.get("memory_reasoning_sections") or [])
                if str(x).strip()
            ],
            "memory_reasoning_snapshot": dict(final_answer_meta.get("memory_reasoning_snapshot") or {}),
        }
        tool_loop = {
            key: value
            for key, value in tool_loop.items()
            if value not in ({}, [], "", None)
        }

    return {
        "request_id": str(data.get("request_id") or ""),
        "user_text": str(data.get("user_text") or ""),
        "retrieval": dict(data.get("memory_retrieval") or {}),
        "governor": dict(data.get("memory_governor") or {}),
        "active_profile": dict(data.get("active_profile") or {}),
        "identity_core": dict(data.get("identity_core") or {}),
        "persona_snapshot": dict(data.get("persona_snapshot") or {}),
        "active_task": dict(data.get("active_task") or {}),
        "prompt_pack": dict(data.get("prompt_pack") or {}),
        "tool_loop": tool_loop,
        "final_answer_meta": dict(data.get("final_answer_meta") or {}),
    }
