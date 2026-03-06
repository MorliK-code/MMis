from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.mode_selector import normalize_mode_name
from metadata.taxonomy import normalize_lang
from modules.character.feedback_detector import detect_feedback


@dataclass(frozen=True)
class CharacterSignals:
    lang: str = "unknown"
    intent: str = "chat"
    emotion: str = "neutral"
    mode: str = "chatting"
    topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    user_feedback: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lang": str(self.lang or "unknown"),
            "intent": str(self.intent or "chat"),
            "emotion": str(self.emotion or "neutral"),
            "mode": str(self.mode or "chatting"),
            "topics": [str(x) for x in list(self.topics or []) if str(x).strip()],
            "tags": [str(x) for x in list(self.tags or []) if str(x).strip()],
            "user_feedback": [str(x) for x in list(self.user_feedback or []) if str(x).strip()],
        }


def build_character_signals(*, text: str, metadata: dict[str, Any] | None = None) -> CharacterSignals:
    meta = dict(metadata or {})
    raw_tags = _normalize_tags(meta.get("metadata_tags") or meta.get("tags"))
    lang = _resolve_lang(meta=meta, tags=raw_tags)
    tags = _canonicalize_lang_tag(tags=raw_tags, lang=lang)
    topics = _extract_topics(tags=tags, metadata=meta)
    feedback = detect_feedback(text)
    return CharacterSignals(
        lang=lang,
        intent=str(meta.get("intent") or "chat").strip().lower() or "chat",
        emotion=str(meta.get("mood") or meta.get("emotion") or "neutral").strip().lower() or "neutral",
        mode=_resolve_mode(meta=meta),
        topics=topics,
        tags=tags,
        user_feedback=feedback,
    )


def _normalize_tags(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in list(values or []):
        key = str(row or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_topics(*, tags: list[str], metadata: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tag in list(tags or []):
        if tag.startswith("topic_"):
            topic = tag
            if topic not in seen:
                seen.add(topic)
                out.append(topic)
    for key in ("topic", "topics"):
        raw = metadata.get(key)
        if key == "topic":
            values = [raw] if raw is not None else []
        else:
            values = list(raw or [])
        for value in values:
            topic = str(value or "").strip().lower()
            if not topic:
                continue
            if not topic.startswith("topic_"):
                topic = f"topic_{topic}"
            if topic in seen:
                continue
            seen.add(topic)
            out.append(topic)
    return out


def _resolve_lang(*, meta: dict[str, Any], tags: list[str]) -> str:
    explicit = normalize_lang(meta.get("lang"))
    if explicit != "unknown":
        return explicit
    for tag in list(tags or []):
        token = str(tag or "").strip().lower()
        if not token.startswith("lang_"):
            continue
        value = normalize_lang(token.replace("lang_", "", 1))
        if value != "unknown":
            return value
    return "unknown"


def _canonicalize_lang_tag(*, tags: list[str], lang: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in list(tags or []):
        token = str(row or "").strip().lower()
        if not token or token.startswith("lang_"):
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)

    lang_token = normalize_lang(lang)
    if lang_token != "unknown":
        key = f"lang_{lang_token}"
        if key not in seen:
            out.append(key)
    return out


def _resolve_mode(*, meta: dict[str, Any]) -> str:
    source = str(meta.get("mode") or meta.get("active_mode") or "chatting").strip().lower()
    return normalize_mode_name(source, allow_custom=True)
