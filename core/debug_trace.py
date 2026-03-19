from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DebugTrace:
    request_id: str = ""
    user_text: str = ""

    memory_retrieval: dict[str, Any] = field(default_factory=dict)
    memory_governor: dict[str, Any] = field(default_factory=dict)
    active_profile: dict[str, Any] = field(default_factory=dict)
    identity_core: dict[str, Any] = field(default_factory=dict)
    persona_snapshot: dict[str, Any] = field(default_factory=dict)
    active_task: dict[str, Any] = field(default_factory=dict)
    prompt_pack: dict[str, Any] = field(default_factory=dict)
    final_answer_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": str(self.request_id or ""),
            "user_text": str(self.user_text or ""),
            "memory_retrieval": dict(self.memory_retrieval or {}),
            "memory_governor": dict(self.memory_governor or {}),
            "active_profile": dict(self.active_profile or {}),
            "identity_core": dict(self.identity_core or {}),
            "persona_snapshot": dict(self.persona_snapshot or {}),
            "active_task": dict(self.active_task or {}),
            "prompt_pack": dict(self.prompt_pack or {}),
            "final_answer_meta": dict(self.final_answer_meta or {}),
        }
