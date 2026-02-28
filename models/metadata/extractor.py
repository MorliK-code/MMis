from __future__ import annotations

import re
from typing import Any


def _fallback(text: str, role: str, context_text: str = "") -> dict[str, Any]:
    clean = str(text or "").strip()
    ctx = str(context_text or "").strip()
    combined = f"{clean} {ctx}".strip()

    has_question = "?" in clean
    has_code = bool(re.search(r"```|Traceback|Exception|def\s+\w+\(|class\s+\w+\s*:|[A-Za-z]:\\", combined, re.I))
    has_link = bool(re.search(r"https?://\S+|www\.\S+", combined, re.I))

    lang = "ru" if re.search(r"[\u0400-\u04FF]", combined) else "en"
    intent = "coding_help" if has_code else ("question" if has_question else "chat")
    speech_act = "question" if has_question else "statement"

    tone = "neutral"
    sentiment = "neutral"
    if re.search(r"\b(thanks|спасибо|класс|great|awesome)\b", combined, re.I):
        tone = "friendly"
        sentiment = "positive"

    topic = []
    if has_code:
        topic.append("coding")
    if has_link:
        topic.append("links")

    return {
        "summary": clean[:220],
        "intent": intent,
        "speech_act": speech_act,
        "user_goal": clean[:160] if role == "user" else "",
        "assistant_action": clean[:160] if role == "assistant" else "",
        "who": "user" if role == "user" else "assistant",
        "what": clean[:140],
        "why": "",
        "tone": tone,
        "sentiment": sentiment,
        "sentiment_score": 0.7 if sentiment == "positive" else 0.0,
        "emotion_primary": "neutral",
        "emotion_secondary": "",
        "valence": 0.4 if sentiment == "positive" else 0.0,
        "arousal": 0.3,
        "politeness": 0.7,
        "urgency": 0.2,
        "confidence": 0.55,
        "toxicity_risk": 0.0,
        "language": lang,
        "topics": topic,
        "entities_people": [],
        "entities_orgs": [],
        "entities_places": [],
        "temporal_refs": [],
        "entities": [],
        "facts": [],
        "needs_followup": bool(has_question),
        "requires_clarification": False,
        "memory_priority": 0.3,
        "role": str(role or "assistant"),
    }


def extract_message_metadata(
    *,
    model: str,
    fallback_models: list[str] | None = None,
    role: str,
    message_text: str,
    context_text: str = "",
) -> dict[str, Any]:
    # model/fallback_models kept in signature for compatibility.
    _ = model
    _ = fallback_models

    target_role = str(role or "assistant").strip().lower()
    target_text = str(message_text or "").strip()
    if not target_text:
        return _fallback("", target_role, context_text=context_text)
    return _fallback(target_text, target_role, context_text=context_text)
