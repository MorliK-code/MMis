from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.spec_registry import load_spec
from metadata.emotion_detector import EmotionResult, detect
from metadata.intent_classifier import IntentResult, classify
from metadata.language_detector import LanguageResult, analyze
from metadata.taxonomy import (
    STRUCTURAL_TAGS,
    TOPICS,
    normalize_emotion,
    normalize_intent,
    normalize_topic_list,
)


@dataclass
class Tagger:
    metadata_spec: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        spec = dict(self.metadata_spec or {})
        if not spec:
            spec = load_spec("metadata", required=False)
        self._spec = dict(spec or {})
        tagging = dict(self._spec.get("tagging") or {})
        self._allowed_tags = {
            str(x).strip().lower()
            for x in list(tagging.get("allowed_tags") or [])
            if str(x).strip()
        }
        self._soft_filter = bool(tagging.get("soft_filter", True))
        heuristics = dict(self._spec.get("heuristics") or {})
        self._topic_rules = [dict(x) for x in list(heuristics.get("topics") or []) if isinstance(x, dict)]
        self._structure_rules = [dict(x) for x in list(heuristics.get("structure") or []) if isinstance(x, dict)]

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
        intent_label = normalize_intent(_extract_intent(intent))
        emotion_label = normalize_emotion(_extract_emotion(emotion))

        tags: set[str] = set()
        topics: set[str] = set()

        if _has_code(src) or bool(flags.get("is_code_like")):
            tags.add("has_code")
        if "traceback" in lower or "stack trace" in lower or "internal server error" in lower:
            tags.add("has_traceback")
            tags.add("has_stacktrace")
        if any(token in lower for token in ("[log]", "logs:", "stderr", "stdout")):
            tags.add("has_logs")
        if _is_json_like(src):
            tags.add("has_json")
        if re.search(r"https?://\S+|www\.\S+", src, re.I):
            tags.add("has_link")
        if re.search(r"\b(?:docker|git|python|pip|pytest|npm|node|uvicorn|curl|powershell|cmd|bash)\b", lower):
            tags.add("has_command")
        if re.search(r"[A-Za-z]:\\", src):
            tags.add("has_config")

        if re.search(r"\bquestion\b|\?$|\bкак\b|\bwhat\b|\bhow\b", lower):
            tags.add("is_question")
        if intent_label in {"task", "bug_report", "code_review"}:
            tags.add("is_task")
            tags.add("needs_steps")
        if any(token in lower for token in ("follow-up", "ещё", "далее", "продолжим")):
            tags.add("is_followup")
        if intent_label in {"question", "clarification"} and len(src) <= 160:
            tags.add("needs_short_answer")

        if emotion_label in {"frustrated", "angry", "sad", "anxious", "tired"}:
            tags.add("tone_soft")
        elif emotion_label in {"happy", "excited"} and intent_label == "chat":
            tags.add("tone_teasing")
        elif intent_label in {"task", "bug_report", "code_review"}:
            tags.add("tone_strict")
        else:
            tags.add("tone_neutral")

        if any(token in lower for token in ("docker compose", "container", "image ", "dockerfile")):
            topics.add("docker")
            tags.add("has_command")
        if any(token in lower for token in ("traceback", "exception", "pytest", "pip", ".py", "python")):
            topics.add("python")
        if any(token in lower for token in ("llm", "model", "prompt", "ollama", "openai", "qwen", "token")):
            topics.add("llm")
        if any(token in lower for token in ("git", "branch", "commit", "merge", "rebase")):
            topics.add("git")
        if any(token in lower for token in ("ui", "button", "css", "layout", "icon", "theme", "интерфейс", "кноп")):
            topics.add("ui")
        if any(token in lower for token in ("vs code", "vscode", "settings.json")):
            topics.add("vscode")
            tags.add("has_config")
        if "c:\\users" in lower:
            topics.add("windows")
        if any(token in lower for token in ("linux", "ubuntu", "debian", "systemd")):
            topics.add("linux")
        if any(token in lower for token in ("chromadb", "chroma")):
            topics.add("chromadb")
        if "rag" in lower:
            topics.add("rag")
        if any(token in lower for token in ("database", "sql", "sqlite", "postgres", "mysql")):
            topics.add("db")
        if any(token in lower for token in ("network", "http", "tcp", "dns", "api")):
            topics.add("network")
        if "pyside" in lower or "qt" in lower:
            topics.add("qt")
        if "ollama" in lower:
            topics.add("ollama")

        self._apply_rules(text=src, tags=tags)

        for topic in normalize_topic_list(topics):
            if topic in TOPICS:
                tags.add(f"topic_{topic}")

        if lang_code:
            tags.add(f"lang_{lang_code}")
        if intent_label:
            tags.add(f"intent_{intent_label}")
        if emotion_label:
            tags.add(f"emotion_{emotion_label}")

        normalized: list[str] = []
        seen: set[str] = set()
        for tag in sorted(tags):
            item = str(tag or "").strip().lower()
            if not item or item in seen:
                continue
            if (
                item in STRUCTURAL_TAGS
                or item.startswith(("lang_", "intent_", "emotion_", "topic_", "is_", "tone_", "mode_hint_", "needs_", "entity_"))
            ):
                if self._soft_filter and self._allowed_tags and not self._is_allowed(item):
                    continue
                normalized.append(item)
                seen.add(item)
        return normalized

    def _apply_rules(self, *, text: str, tags: set[str]) -> None:
        lower = str(text or "").lower()
        for row in [*self._topic_rules, *self._structure_rules]:
            if not _rule_matches(lower=lower, source=text, rule=row):
                continue
            for tag in list(row.get("add") or []):
                token = str(tag or "").strip().lower()
                if token:
                    tags.add(token)

    def _is_allowed(self, tag: str) -> bool:
        if tag in self._allowed_tags:
            return True
        if tag.startswith(("lang_", "intent_", "emotion_", "topic_", "mode_hint_", "is_", "tone_", "needs_", "entity_")):
            return True
        if tag in STRUCTURAL_TAGS:
            return True
        return False


def build(text: str, lang, intent, emotion, extra_flags: dict[str, Any] | None = None) -> list[str]:
    return Tagger().build(text=text, lang=lang, intent=intent, emotion=emotion, extra_flags=extra_flags)


def tag_message(text: str) -> dict:
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


def _rule_matches(*, lower: str, source: str, rule: dict[str, Any]) -> bool:
    contains = [str(x).strip().lower() for x in list(rule.get("if_contains") or []) if str(x).strip()]
    if contains and not any(item in lower for item in contains):
        return False
    pattern = str(rule.get("if_regex") or "").strip()
    if pattern:
        try:
            if not bool(re.search(pattern, source, flags=re.I | re.S)):
                return False
        except re.error:
            return False
    return bool(contains or pattern)


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

