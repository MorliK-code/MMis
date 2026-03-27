from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory_core.schemas import MemoryArtifact


_EXPOSURE_BY_TYPE = {
    "emotional_state": "latent",
    "identity_core": "prompt_safe",
    "profile_fact": "exact_quote",
    "preference": "prompt_safe",
    "task_state": "prompt_safe",
    "task": "prompt_safe",
    "episode_event": "prompt_safe",
    "episode": "prompt_safe",
    "fact": "prompt_safe",
    "document_chunk": "exact_quote",
    "document_summary": "prompt_safe",
}

_EMOTION_KEYWORDS = {
    "sad": ("sad", "грусть", "груст", "уныл", "печал"),
    "tired": ("tired", "устал", "усталость", "измотан", "выгор"),
    "frustrated": ("frustrated", "раздраж", "злит", "бесит", "фрустр"),
    "anxious": ("anxious", "тревог", "нервн", "паник"),
    "angry": ("angry", "злой", "злость", "сердит"),
}

_SELF_RECALL_MARKERS = (
    "что ты про меня помнишь",
    "что ты обо мне помнишь",
    "что ты помнишь",
    "what do you remember about me",
    "what do you remember",
)


@dataclass(slots=True)
class DistilledMemoryPack:
    selected: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    blocks: dict[str, str] = field(default_factory=dict)
    recent_user_state: dict[str, Any] = field(default_factory=dict)
    response_bias: dict[str, Any] = field(default_factory=dict)
    legacy_lists: dict[str, list[str]] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


def distill_memory_artifacts(
    artifacts: list[MemoryArtifact],
    *,
    query_text: str = "",
) -> DistilledMemoryPack:
    exact_recall: list[str] = []
    answer_support: list[str] = []
    continuity_hints: list[str] = []
    tone_hints: list[str] = []
    docs: list[str] = []
    legacy_profile: list[str] = []
    legacy_tasks: list[str] = []
    legacy_episodes: list[str] = []
    legacy_facts: list[str] = []
    selected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    recent_user_state: dict[str, Any] = {}
    response_bias: dict[str, Any] = {}
    self_recall_mode = _is_self_recall_query(query_text)

    for artifact in list(artifacts or []):
        row = _distill_artifact(artifact, self_recall_mode=self_recall_mode)
        selected.append(row)
        prompt_text = str(row.get("prompt_text") or "").strip()
        channel = str(row.get("channel") or "").strip().lower()
        exposure_mode = str(row.get("exposure_mode") or "latent").strip().lower()

        if row.get("recent_user_state"):
            recent_user_state = _merge_recent_user_state(recent_user_state, dict(row.get("recent_user_state") or {}))
        if row.get("response_bias"):
            response_bias = _merge_numeric_maps(response_bias, dict(row.get("response_bias") or {}))

        if not prompt_text:
            dropped.append(
                {
                    "artifact_id": row.get("artifact_id"),
                    "artifact_type": row.get("artifact_type"),
                    "reason": "no_prompt_safe_view",
                    "exposure_mode": exposure_mode,
                }
            )
            continue

        if exposure_mode == "latent":
            dropped.append(
                {
                    "artifact_id": row.get("artifact_id"),
                    "artifact_type": row.get("artifact_type"),
                    "reason": "latent_hidden_from_prompt",
                    "exposure_mode": exposure_mode,
                    "prompt_view": prompt_text,
                }
            )
            continue

        if channel == "exact_recall":
            exact_recall.append(prompt_text)
        elif channel == "answer_support":
            answer_support.append(prompt_text)
        elif channel == "continuity_hints":
            continuity_hints.append(prompt_text)
        elif channel == "tone_hints":
            tone_hints.append(prompt_text)
        elif channel == "document_context":
            docs.append(prompt_text)

        artifact_type = str(row.get("artifact_type") or "").strip().lower()
        if artifact_type == "profile_fact":
            legacy_profile.append(prompt_text)
        elif artifact_type in {"task_state", "task"}:
            legacy_tasks.append(prompt_text)
        elif artifact_type in {"episode_event", "episode"}:
            legacy_episodes.append(prompt_text)
        elif artifact_type in {"fact", "preference"}:
            legacy_facts.append(prompt_text)

    blocks = {
        "exact_recall": _render_list_block(exact_recall),
        "answer_support": _render_list_block(answer_support),
        "continuity_hints": _render_list_block(continuity_hints),
        "tone_hints": _render_list_block(tone_hints),
        "retrieved_docs": _render_list_block(docs),
    }

    return DistilledMemoryPack(
        selected=selected,
        dropped=dropped,
        blocks={key: value for key, value in blocks.items() if value},
        recent_user_state=recent_user_state,
        response_bias=response_bias,
        legacy_lists={
            "profile_facts": legacy_profile,
            "active_tasks": legacy_tasks,
            "recent_episodes": legacy_episodes,
            "relevant_facts": legacy_facts,
            "document_chunks": docs,
        },
        debug={
            "self_recall_mode": self_recall_mode,
            "selected_count": len(selected),
            "dropped_count": len(dropped),
            "visible_prompt_items": len(exact_recall) + len(answer_support) + len(continuity_hints) + len(tone_hints) + len(docs),
        },
    )


def resolve_artifact_exposure_mode(artifact_type: str, metadata: dict[str, Any] | None = None) -> str:
    meta = dict(metadata or {})
    explicit = str(meta.get("exposure_mode") or "").strip().lower()
    if explicit in {"latent", "prompt_safe", "exact_quote"}:
        return explicit
    return str(_EXPOSURE_BY_TYPE.get(str(artifact_type or "").strip().lower(), "prompt_safe"))


def resolve_artifact_prompt_view(
    artifact_type: str,
    *,
    text: str,
    summary: str,
    metadata: dict[str, Any] | None = None,
    exposure_mode: str | None = None,
) -> str:
    meta = dict(metadata or {})
    for key in ("prompt_view", "distilled_text", "safe_text"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value

    normalized_type = str(artifact_type or "").strip().lower()
    if normalized_type == "emotional_state":
        return _derive_emotional_prompt_view(text=text, summary=summary, metadata=meta)

    safe_summary = str(summary or "").strip()
    if safe_summary:
        return safe_summary

    mode = str(exposure_mode or resolve_artifact_exposure_mode(normalized_type, meta)).strip().lower()
    safe_text = str(text or "").strip()
    if mode == "exact_quote" and _allow_exact_text_fallback(normalized_type, safe_text):
        return safe_text
    return ""


def resolve_artifact_sensitivity(metadata: dict[str, Any] | None = None) -> str:
    meta = dict(metadata or {})
    raw = str(meta.get("sensitivity") or "").strip().lower()
    if raw in {"low", "medium", "high"}:
        return raw
    return "medium"


def _distill_artifact(artifact: MemoryArtifact, *, self_recall_mode: bool) -> dict[str, Any]:
    metadata = dict(artifact.metadata or {})
    artifact_type = str(artifact.artifact_type or "").strip().lower()
    raw_text = str(artifact.text or "").strip()
    summary = str(artifact.summary or "").strip()
    exposure_mode = resolve_artifact_exposure_mode(artifact_type, metadata)
    prompt_view = resolve_artifact_prompt_view(
        artifact_type,
        text=raw_text,
        summary=summary,
        metadata=metadata,
        exposure_mode=exposure_mode,
    )
    sensitivity = resolve_artifact_sensitivity(metadata)
    channel = _choose_channel(
        artifact_type=artifact_type,
        exposure_mode=exposure_mode,
        self_recall_mode=self_recall_mode,
    )
    recent_user_state = {}
    response_bias = {}
    if artifact_type == "emotional_state":
        recent_user_state = _extract_recent_user_state(raw_text, summary, metadata)
        response_bias = _build_response_bias(recent_user_state)

    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact_type,
        "text": raw_text,
        "summary": summary,
        "prompt_view": prompt_view,
        "prompt_text": prompt_view,
        "metadata": metadata,
        "confidence": float(metadata.get("confidence", 0.5)),
        "score": float(metadata.get("semantic_score", metadata.get("score", 0.0)) or 0.0),
        "source_event_id": artifact.source_event_id,
        "updated_at": artifact.updated_at,
        "exposure_mode": exposure_mode,
        "sensitivity": sensitivity,
        "channel": channel,
        "relevant": exposure_mode != "latent" or bool(recent_user_state or response_bias),
        "recent_user_state": recent_user_state,
        "response_bias": response_bias,
    }


def _choose_channel(*, artifact_type: str, exposure_mode: str, self_recall_mode: bool) -> str:
    if artifact_type == "emotional_state":
        return "tone_hints"
    if artifact_type in {"document_chunk", "document_summary"}:
        return "document_context"
    if self_recall_mode and artifact_type in {"profile_fact", "preference", "task_state", "task"}:
        return "exact_recall"
    if artifact_type in {"profile_fact", "fact"} and exposure_mode == "exact_quote":
        return "exact_recall"
    if artifact_type in {"episode_event", "episode", "identity_core"}:
        return "continuity_hints"
    return "answer_support"


def _allow_exact_text_fallback(artifact_type: str, text: str) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    if artifact_type == "emotional_state":
        return False
    forbidden = (
        "user emotion:",
        "пользователь чувствует",
        "пользователь раздраж",
        "the user is frustrated",
        "the user feels",
    )
    return not any(token in low for token in forbidden)


def _derive_emotional_prompt_view(*, text: str, summary: str, metadata: dict[str, Any]) -> str:
    if bool(metadata.get("low_bandwidth")):
        return "user currently low-bandwidth; keep answer calm and concise"
    emotion = str(metadata.get("emotion") or "").strip().lower()
    if not emotion:
        emotion = _infer_emotion_from_text(" ".join(part for part in (text, summary) if part))
    if emotion in {"sad", "tired", "anxious"}:
        return "user currently low-bandwidth; keep tone steady and concise"
    if emotion in {"frustrated", "angry"}:
        return "user appears tense; keep tone calm, direct, and reduce teasing"
    return "treat as soft tone guidance only; do not quote this memory literally"


def _extract_recent_user_state(text: str, summary: str, metadata: dict[str, Any]) -> dict[str, Any]:
    emotion = str(metadata.get("emotion") or "").strip().lower()
    if not emotion:
        emotion = _infer_emotion_from_text(" ".join(part for part in (text, summary) if part))
    frustrated = emotion in {"frustrated", "angry"}
    low_bandwidth = bool(metadata.get("low_bandwidth")) or emotion in {"sad", "tired", "anxious"}
    intensity = _to_float(metadata.get("intensity"), None)
    arousal = _to_float(metadata.get("arousal"), None)
    sources = ["memory.emotional_state"]
    if metadata.get("emotion"):
        sources.append("memory.metadata.emotion")
    elif emotion:
        sources.append("memory.text_inference")
    row = {
        "emotion": emotion,
        "frustrated": frustrated,
        "low_bandwidth": low_bandwidth,
        "intensity": intensity,
        "arousal": arousal,
        "sources": sources,
    }
    return {
        key: value
        for key, value in row.items()
        if value not in (None, "", [], {})
    }


def _build_response_bias(recent_user_state: dict[str, Any]) -> dict[str, Any]:
    low_bandwidth = bool(recent_user_state.get("low_bandwidth"))
    frustrated = bool(recent_user_state.get("frustrated"))
    out = {
        "needs_short_answer": 1.0 if low_bandwidth else 0.0,
        "frustration_softening": 1.0 if frustrated else 0.0,
        "playfulness_downshift": 0.75 if frustrated else 0.0,
        "warmth_upshift": 0.18 if frustrated else 0.0,
    }
    return {key: value for key, value in out.items() if float(value or 0.0) > 0.0}


def _infer_emotion_from_text(text: str) -> str:
    low = str(text or "").strip().lower()
    if not low:
        return ""
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        if any(token in low for token in keywords):
            return emotion
    return ""


def _merge_recent_user_state(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key in ("frustrated", "low_bandwidth"):
        merged[key] = bool(merged.get(key)) or bool(extra.get(key))
    for key in ("emotion", "intensity", "arousal"):
        if key not in merged and key in extra:
            merged[key] = extra.get(key)
    merged["sources"] = _dedupe_list(list(merged.get("sources") or []) + list(extra.get("sources") or []))
    return {
        key: value
        for key, value in merged.items()
        if value not in (None, "", [], {})
    }


def _merge_numeric_maps(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in dict(extra or {}).items():
        merged[key] = max(_to_float(merged.get(key), 0.0) or 0.0, _to_float(value, 0.0) or 0.0)
    return merged


def _render_list_block(items: list[str]) -> str:
    rows = [str(item).strip() for item in list(items or []) if str(item).strip()]
    if not rows:
        return ""
    return "\n".join(f"- {item}" for item in _dedupe_list(rows))


def _dedupe_list(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        text = str(item or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _to_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _is_self_recall_query(query_text: str) -> bool:
    low = str(query_text or "").strip().lower()
    if not low:
        return False
    return any(marker in low for marker in _SELF_RECALL_MARKERS)
