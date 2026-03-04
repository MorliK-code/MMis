from __future__ import annotations

from typing import Iterable

from core.spec_registry import load_spec


def load_taxonomy(*, force_reload: bool = False) -> dict:
    _ = force_reload  # kept for API compatibility
    payload = load_spec("taxonomy", required=False)
    return dict(payload or {})


_TAXONOMY = load_taxonomy()
_ALIASES = dict(_TAXONOMY.get("aliases") or {})

LANGS = set(str(x).strip().lower() for x in list(_TAXONOMY.get("langs") or []))
INTENTS = set(str(x).strip().lower() for x in list(_TAXONOMY.get("intents") or []))
EMOTIONS = set(str(x).strip().lower() for x in list(_TAXONOMY.get("emotions") or []))
MODES = set(str(x).strip().lower() for x in list(_TAXONOMY.get("modes") or []))
TOPICS = set(str(x).strip().lower() for x in list(_TAXONOMY.get("topics") or []))
STRUCTURAL_TAGS = set(str(x).strip().lower() for x in list(_TAXONOMY.get("structural_tags") or []))
TONE_TAGS = set(str(x).strip().lower() for x in list(_TAXONOMY.get("tone_tags") or []))


_LANG_ALIASES = {
    str(k).strip().lower(): str(v).strip().lower()
    for k, v in dict(_ALIASES.get("langs") or {}).items()
    if str(k).strip() and str(v).strip()
}
_INTENT_ALIASES = {
    str(k).strip().lower(): str(v).strip().lower()
    for k, v in dict(_ALIASES.get("intents") or {}).items()
    if str(k).strip() and str(v).strip()
}
_EMOTION_ALIASES = {
    str(k).strip().lower(): str(v).strip().lower()
    for k, v in dict(_ALIASES.get("emotions") or {}).items()
    if str(k).strip() and str(v).strip()
}
_TOPIC_ALIASES = {
    str(k).strip().lower(): str(v).strip().lower()
    for k, v in dict(_ALIASES.get("topics") or {}).items()
    if str(k).strip() and str(v).strip()
}


def normalize_mode(value) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "friend_chat"
    aliases = {str(k).strip().lower(): str(v).strip().lower() for k, v in dict(_ALIASES.get("modes") or {}).items()}
    token = aliases.get(token, token)
    return token if token in MODES else "friend_chat"


def normalize_lang(value) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "unknown"
    token = _LANG_ALIASES.get(token, token)
    return token if token in LANGS else "unknown"


def normalize_intent(value) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "chat"
    token = _INTENT_ALIASES.get(token, token)
    return token if token in INTENTS else "chat"


def normalize_emotion(value) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "neutral"
    token = _EMOTION_ALIASES.get(token, token)
    return token if token in EMOTIONS else "neutral"


def normalize_topic_list(values: Iterable | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        token = str(value or "").strip().lower()
        if not token:
            continue
        token = token.replace("topic_", "", 1)
        token = _TOPIC_ALIASES.get(token, token)
        if token not in TOPICS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out

