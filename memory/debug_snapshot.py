from __future__ import annotations

from typing import Any


def build_memory_debug_snapshot(ctx) -> dict[str, Any]:
    state = dict(getattr(ctx, "state", {}) or {})
    trace = state.get("debug_trace")
    if trace is None:
        return {}

    data = trace.to_dict() if hasattr(trace, "to_dict") else dict(trace or {})

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
        "final_answer_meta": dict(data.get("final_answer_meta") or {}),
    }
