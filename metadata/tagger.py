from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from metadata.emotion_detector import EmotionResult, detect
from metadata.intent_classifier import IntentResult, classify
from metadata.language_detector import LanguageResult, analyze


@dataclass
class Tagger:
    def build(
        self,
        text: str,
        lang,
        intent,
        emotion,
        extra_flags: dict[str, Any] | None = None,
    ) -> list[str]:
        src = str(text or "")
        lower = src.lower()
        flags = dict(extra_flags or {})

        lang_code = _extract_lang(lang)
        intent_label = _extract_intent(intent)
        emotion_label = _extract_emotion(emotion)
        tags: list[str] = []

        if _has_code(src) or bool(flags.get("is_code_like")):
            tags.append("has_code")
        if "traceback" in lower or "stack trace" in lower:
            tags.append("has_traceback")
        if re.search(r"[A-Za-z]:\\", src):
            tags.append("has_path_windows")
        if _is_json_like(src):
            tags.append("has_json")
        if re.search(r"https?://\S+|www\.\S+", src, re.I):
            tags.append("contains_link")
        if re.search(r"\b\d+\b", src):
            tags.append("contains_numbers")
        if re.search(r"\b\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\b|\b\d{4}-\d{2}-\d{2}\b", src):
            tags.append("contains_date")

        if re.search(r"^\s*(как|how)\b", lower):
            tags.append("question_howto")
        if intent_label in {"coding_help", "task_request", "ui_request"}:
            tags.append("needs_steps")
        if intent_label in {"question", "search"} and len(src) <= 120:
            tags.append("needs_short_answer")

        if intent_label == "coding_help":
            tags.append("response_structured")
            tags.append("no_flirt")
        if emotion_label == "frustrated_angry":
            tags.append("user_frustrated")
            tags.append("tone_soft")
            tags.append("supportive_priority")
        elif emotion_label in {"sad_tired", "confused"}:
            tags.append("tone_soft")
        else:
            tags.append("tone_strict")

        if any(token in lower for token in ("llm", "model", "prompt", "ollama", "openai", "qwen", "token")):
            tags.append("topic_llm")
        if any(token in lower for token in ("git", "branch", "commit", "merge", "rebase")):
            tags.append("topic_git")
        if any(token in lower for token in ("ui", "button", "css", "layout", "icon", "theme", "интерфейс", "кнопк")):
            tags.append("topic_ui")
        if any(token in lower for token in ("python", "traceback", "pytest", "pip", ".py")):
            tags.append("topic_python")

        if lang_code:
            tags.append(f"lang_{lang_code}")
        if intent_label:
            tags.append(f"intent_{intent_label}")
        if emotion_label:
            tags.append(f"emotion_{emotion_label}")

        return _dedupe(tags)


def build(text: str, lang, intent, emotion, extra_flags: dict[str, Any] | None = None) -> list[str]:
    return Tagger().build(text=text, lang=lang, intent=intent, emotion=emotion, extra_flags=extra_flags)


def tag_message(text: str) -> dict:
    # Backward-compatible helper used by old call sites.
    lang_result: LanguageResult = analyze(text)
    intent_result: IntentResult = classify(text, lang=lang_result.lang, context_tags=None)
    emotion_result: EmotionResult = detect(text, lang=lang_result.lang)
    tags = build(
        text=text,
        lang=lang_result,
        intent=intent_result,
        emotion=emotion_result,
        extra_flags={
            "is_code_like": lang_result.is_code_like,
            "has_url": lang_result.has_url,
            "has_emoji": lang_result.has_emoji,
        },
    )
    return {
        "intent": intent_result.label,
        "intent_conf": float(intent_result.conf),
        "emotion": {
            "label": emotion_result.label,
            "scores": dict(emotion_result.scores),
            "intensity": float(emotion_result.intensity),
            "arousal": float(emotion_result.arousal),
        },
        "language": lang_result.lang,
        "language_conf": float(lang_result.conf),
        "tags": tags,
    }


def _extract_lang(value) -> str:
    if isinstance(value, LanguageResult):
        return value.lang
    if isinstance(value, dict):
        return str(value.get("lang") or value.get("language") or "").strip()
    return str(value or "").strip()


def _extract_intent(value) -> str:
    if isinstance(value, IntentResult):
        return value.label
    if isinstance(value, dict):
        return str(value.get("label") or value.get("intent") or "").strip()
    return str(value or "").strip()


def _extract_emotion(value) -> str:
    if isinstance(value, EmotionResult):
        return value.label
    if isinstance(value, dict):
        return str(value.get("label") or value.get("primary") or value.get("emotion") or "").strip()
    return str(value or "").strip()


def _has_code(text: str) -> bool:
    src = str(text or "")
    return bool(
        re.search(
            r"```|`[^`]+`|def\s+\w+\(|class\s+\w+\s*:|import\s+\w+|#include\s*<|Traceback|Exception|{.*}",
            src,
            re.I | re.S,
        )
    )


def _is_json_like(text: str) -> bool:
    src = str(text or "").strip()
    return (src.startswith("{") and src.endswith("}")) or (src.startswith("[") and src.endswith("]"))


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value or "").strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out

