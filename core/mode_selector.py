from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from metadata.taxonomy import MODES

_MODE_SET = set(str(x).strip().lower() for x in MODES)
_LEGACY_TO_MODE = {
    "chat": "friend_chat",
    "task": "helper",
    "coding": "engineer",
    "debug": "debugger",
}


@dataclass(frozen=True)
class ModeDecision:
    mode: str
    confidence: float
    reason: str
    locked: bool = False


def normalize_mode_name(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in _MODE_SET:
        return text
    mapped = _LEGACY_TO_MODE.get(text, "")
    if mapped:
        return mapped
    return "friend_chat"


class ModeSelector:
    def __init__(self, *, confidence_threshold: float = 0.8):
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))

    def decide(
        self,
        *,
        active_mode: str,
        mode_lock: bool,
        intent: str,
        emotion: str,
        tags: Iterable[str] | None = None,
    ) -> ModeDecision:
        current = normalize_mode_name(active_mode)
        if bool(mode_lock):
            return ModeDecision(mode=current, confidence=1.0, reason="mode_lock", locked=True)

        intent_key = str(intent or "").strip().lower()
        emotion_key = str(emotion or "").strip().lower()
        tag_set = {str(x or "").strip().lower() for x in list(tags or []) if str(x or "").strip()}
        signal_tags = {
            tag
            for tag in tag_set
            if not tag.startswith(("lang_", "intent_", "emotion_", "mode_hint_", "tone_", "is_", "needs_"))
        }
        trace_tags = {"has_traceback", "has_logs", "has_stacktrace"}

        if trace_tags & signal_tags:
            reason = "bug_report_traceback" if intent_key == "bug_report" else "traceback_or_logs"
            confidence = 0.98 if intent_key == "bug_report" else 0.96
            return ModeDecision(mode="debugger", confidence=confidence, reason=reason)

        if intent_key == "bug_report" and {"has_code", "topic_python"} & signal_tags:
            return ModeDecision(mode="debugger", confidence=0.91, reason="bug_report_code")

        if intent_key == "planning":
            return ModeDecision(mode="planner", confidence=0.92, reason="planning_intent")

        if "has_code" in signal_tags and intent_key in {"task", "bug_report", "code_review"}:
            return ModeDecision(mode="engineer", confidence=0.9, reason="code_task")

        if emotion_key == "frustrated":
            return ModeDecision(mode="helper", confidence=0.82, reason="frustrated_emotion")

        if intent_key in {"chat", "clarification"} and not signal_tags:
            return ModeDecision(mode="friend_chat", confidence=0.85, reason="casual_chat")

        return ModeDecision(mode="friend_chat", confidence=0.72, reason="default")

    def should_switch(self, *, current_mode: str, decision: ModeDecision) -> bool:
        if decision.locked:
            return False
        current = normalize_mode_name(current_mode)
        if decision.mode == current:
            return False
        return float(decision.confidence) >= self.confidence_threshold
