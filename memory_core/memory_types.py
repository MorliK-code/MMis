from __future__ import annotations

from typing import Any


PROFILE_FACT = "profile_fact"
PREFERENCE = "preference"
EPISODE_EVENT = "episode_event"
TASK_STATE = "task_state"
EMOTIONAL_STATE = "emotional_state"
IDENTITY_CORE = "identity_core"
SYSTEM_NOTE = "system_note"
PROJECT_CONTEXT = "project_context"
FACT = "fact"
DOCUMENT_CHUNK = "document_chunk"
DOCUMENT_SUMMARY = "document_summary"


MEMORY_TYPE_DEFAULTS: dict[str, dict[str, Any]] = {
    PROFILE_FACT: {
        "scope": "profile",
        "priority": 5,
        "decay": "none",
        "retrieve_when": ["profile", "identity", "memory"],
    },
    PREFERENCE: {
        "scope": "profile",
        "priority": 5,
        "decay": "slow",
        "retrieve_when": ["preference", "style", "memory"],
    },
    EPISODE_EVENT: {
        "scope": "episode",
        "priority": 3,
        "decay": "fast",
        "retrieve_when": ["continuity", "recent", "memory"],
    },
    TASK_STATE: {
        "scope": "task",
        "priority": 4,
        "decay": "slow",
        "retrieve_when": ["task", "project", "memory"],
    },
    EMOTIONAL_STATE: {
        "scope": "episode",
        "priority": 2,
        "decay": "fast",
        "retrieve_when": ["tone", "emotion", "continuity"],
    },
    IDENTITY_CORE: {
        "scope": "global",
        "priority": 10,
        "decay": "none",
        "retrieve_when": ["identity", "always"],
    },
    SYSTEM_NOTE: {
        "scope": "global",
        "priority": 7,
        "decay": "none",
        "retrieve_when": ["system", "policy"],
    },
    PROJECT_CONTEXT: {
        "scope": "project",
        "priority": 4,
        "decay": "slow",
        "retrieve_when": ["project", "task", "memory"],
    },
    FACT: {
        "scope": "profile",
        "priority": 3,
        "decay": "slow",
        "retrieve_when": ["fact", "memory"],
    },
    DOCUMENT_CHUNK: {
        "scope": "document",
        "priority": 3,
        "decay": "slow",
        "retrieve_when": ["document", "evidence"],
    },
    DOCUMENT_SUMMARY: {
        "scope": "document",
        "priority": 3,
        "decay": "slow",
        "retrieve_when": ["document", "summary"],
    },
}

ARTIFACT_TO_MEMORY_TYPE: dict[str, str] = {
    "profile_fact": PROFILE_FACT,
    "preference": PREFERENCE,
    "episode_event": EPISODE_EVENT,
    "episode": EPISODE_EVENT,
    "task": TASK_STATE,
    "task_state": TASK_STATE,
    "emotional_state": EMOTIONAL_STATE,
    "identity_core": IDENTITY_CORE,
    "system_note": SYSTEM_NOTE,
    "project_context": PROJECT_CONTEXT,
    "fact": FACT,
    "document_chunk": DOCUMENT_CHUNK,
    "document_summary": DOCUMENT_SUMMARY,
}


def normalize_memory_type(value: Any, fallback: str = FACT) -> str:
    raw = str(value or "").strip().lower()
    if raw in MEMORY_TYPE_DEFAULTS:
        return raw
    return ARTIFACT_TO_MEMORY_TYPE.get(raw, fallback)


def memory_type_defaults(memory_type: Any, fallback: str = FACT) -> dict[str, Any]:
    normalized = normalize_memory_type(memory_type, fallback=fallback)
    defaults = dict(MEMORY_TYPE_DEFAULTS.get(normalized) or MEMORY_TYPE_DEFAULTS[FACT])
    defaults["memory_type"] = normalized
    return defaults


def enrich_metadata_with_memory_type(
    metadata: dict[str, Any] | None,
    *,
    artifact_type: Any = None,
    memory_type: Any = None,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    normalized = normalize_memory_type(memory_type or meta.get("memory_type") or artifact_type)
    defaults = memory_type_defaults(normalized)
    meta.setdefault("memory_type", normalized)
    meta.setdefault("type", normalized)
    meta.setdefault("scope", defaults.get("scope"))
    meta.setdefault("priority", defaults.get("priority"))
    meta.setdefault("decay", defaults.get("decay"))
    meta.setdefault("retrieve_when", list(defaults.get("retrieve_when") or []))
    return meta


def classify_event_memory_type(source_kind: Any, payload_type: Any, metadata: dict[str, Any] | None = None) -> str:
    meta = dict(metadata or {})
    explicit = meta.get("memory_type") or meta.get("artifact_type")
    if explicit:
        return normalize_memory_type(explicit)

    payload = str(payload_type or "").strip().lower()
    source = str(source_kind or "").strip().lower()
    if payload == "state_update":
        return TASK_STATE
    if source == "assistant":
        return EPISODE_EVENT
    if source == "tool":
        return PROJECT_CONTEXT
    if source == "document":
        return DOCUMENT_SUMMARY
    return EPISODE_EVENT
