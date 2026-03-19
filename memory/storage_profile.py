from __future__ import annotations

from typing import Any


DEFAULT_STORAGE_PROFILE = "compact"
VALID_STORAGE_PROFILES = {"compact", "debug"}

DEBUG_ONLY_METADATA_FIELDS = {
    "web_used",
    "web_factual_mode",
    "web_query_intent",
    "web_search_mode",
    "web_primary_category",
    "web_evidence_quality_score",
    "web_evidence_quality",
    "web_sources_scanned",
    "web_sources_selected",
    "web_evidence_count",
    "web_conflicting_sources",
    "web_low_evidence_quality",
    "web_selected_avg_quality",
    "web_topical_filtered_sources",
    "web_conflict_severity",
    "web_evidence_strength",
    "web_final_factual_confidence",
    "web_cautious_synthesis",
    "web_numeric_candidates_selected",
    "web_numeric_candidates_rejected",
    "_persona_snapshot_debug",
    "persona_snapshot",
    "promotion_project_signal",
    "promotion_task_signal",
    "promotion_decision_signal",
    "promotion_preference_signal",
    "promotion_issue_signal",
    "promotion_technical_signal",
    "promotion_repeated_topic_signal",
    "promotion_smalltalk_signal",
    "promotion_signal_score",
    "promotion_stable_fact_signal",
    "promotion_fact_signal",
    "promotion_fact_relation_diversity",
    "extracted_facts_count",
    "extracted_fact_relations",
    "memory_analysis",
    "memory_views_debug",
    "claim_candidates",
    "lifecycle_decision",
    "decision_debug",
    "assistant_write_policy",
    "assistant_write_blocked",
}

_TEXT_PROJECTION_FIELDS = {
    "normalized_text",
    "canonical_text",
    "search_text",
}

_POLICY_KEYS = {
    "action",
    "reason",
    "target_scope",
    "allow_store",
    "allow_long_term",
    "allow_fact_records",
    "allow_claim_records",
    "allow_write",
    "allow_supersede",
    "group",
}

_LIFECYCLE_KEYS = {
    "reason",
    "route",
    "promote_to",
    "mark_status",
}

_FACT_KEYS = {
    "subject",
    "predicate",
    "value",
    "scope",
    "confidence",
    "importance",
    "source_event_id",
    "valid_from",
    "valid_to",
    "status",
    "canonical_key",
    "relation",
    "source_role",
    "source_kind",
}

_CLAIM_KEYS = {
    "subject",
    "predicate",
    "obj",
    "subject_type",
    "object_type",
    "object_surface",
    "qualifiers",
    "confidence",
    "salience",
    "topic_keys",
    "trigger_keys",
    "recall_mode",
    "spontaneous_recall",
    "status",
    "promotion_level",
    "canonical_key",
    "source_event_id",
    "scope",
    "namespace",
}

_COMPACT_CLAIM_LIST_KEYS = {
    "subject",
    "predicate",
    "obj",
    "object_surface",
    "object_type",
    "topic_keys",
    "trigger_keys",
    "confidence",
}

_DIALOG_EPISODE_KEYS = {
    "id",
    "topic",
    "turn_ids",
    "summary_short",
    "summary_reasoning",
    "decisions",
    "open_questions",
    "participants",
    "salience",
    "topic_keys",
    "entity_keys",
    "created_at",
    "updated_at",
}

_DOCUMENT_SUMMARY_KEYS = {
    "id",
    "document_id",
    "text",
    "summary_kind",
    "source_chunk_ids",
    "topic_keys",
    "entity_keys",
    "confidence",
    "salience",
    "status",
}

_DOCUMENT_CLAIM_KEYS = {
    "id",
    "document_id",
    "chunk_id",
    "subject",
    "predicate",
    "obj",
    "subject_type",
    "object_type",
    "object_surface",
    "confidence",
    "salience",
    "topic_keys",
    "trigger_keys",
    "status",
}

_MESSAGE_METADATA_KEYS = {
    "memory_views",
    "memory_entities",
    "numeric_facts",
    "stable_facts",
    "claims",
    "emotion_profile",
    "memory_tags",
    "meta",
    "source_kind",
    "source_role",
    "source_fact_records_allowed",
    "source_fact_records_blocked",
    "assistant_fact_records_blocked",
    "assistant_thinking_stripped",
    "assistant_reply_kind",
    "assistant_memory_help_noise",
    "assistant_write_reason",
    "assistant_noise_archived",
    "previous_status",
    "lifecycle_reason",
    "write_policy",
    "fact_relations",
    "facts_count",
}

_FACT_METADATA_KEYS = {
    "fact",
    "canonical_key",
    "governor_reason",
    "relation",
    "parallel_with",
    "write_policy",
}

_CLAIM_METADATA_KEYS = {
    "claim",
    "canonical_key",
    "topic_keys",
    "trigger_keys",
    "write_policy",
    "document_id",
    "doc_id",
    "chunk_id",
    "document_claim",
}

_EPISODE_METADATA_KEYS = {
    "dialog_episode",
    "topic",
    "summary_short",
    "summary_reasoning",
    "decisions",
    "open_questions",
    "participants",
    "turn_ids",
    "topic_keys",
    "entity_keys",
    "salience",
}

_DOCUMENT_METADATA_KEYS = {
    "title",
    "path",
    "source_path",
    "filename",
    "extension",
    "document_outline",
    "section_summary_ids",
    "document_claim_ids",
}

_DOCUMENT_CHUNK_METADATA_KEYS = {
    "document_id",
    "doc_id",
    "chunk_id",
    "chunk_index",
    "section_label",
    "memory_views",
    "memory_entities",
    "numeric_facts",
    "stable_facts",
    "claims",
    "memory_tags",
    "title",
    "path",
    "source_path",
    "filename",
    "extension",
}


def normalize_storage_profile(profile: str) -> str:
    token = str(profile or "").strip().lower()
    return token if token in VALID_STORAGE_PROFILES else DEFAULT_STORAGE_PROFILE


def storage_schema_for_memory_type(memory_type: Any) -> dict[str, Any]:
    token = _normalize_memory_type(memory_type)
    always = sorted(_allowed_top_level_keys(token, {}))
    return {
        "memory_type": token,
        "always": always,
        "debug_only": sorted(DEBUG_ONLY_METADATA_FIELDS | _TEXT_PROJECTION_FIELDS),
    }


def compact_metadata_payload(metadata: dict[str, Any] | None, *, memory_type: Any = None) -> dict[str, Any]:
    out = dict(metadata or {})

    for field in DEBUG_ONLY_METADATA_FIELDS:
        out.pop(field, None)
    for field in _TEXT_PROJECTION_FIELDS:
        out.pop(field, None)

    memory_views = out.get("memory_views")
    if isinstance(memory_views, dict):
        compact_views: dict[str, Any] = {}
        if memory_views.get("entity_keys"):
            compact_views["entity_keys"] = list(memory_views.get("entity_keys") or [])
        if memory_views.get("numeric_keys"):
            compact_views["numeric_keys"] = list(memory_views.get("numeric_keys") or [])
        if compact_views:
            out["memory_views"] = compact_views
        else:
            out.pop("memory_views", None)

    out.pop("emotion", None)

    if isinstance(out.get("assistant_write_policy"), dict):
        out["assistant_write_policy"] = _compact_dict(out["assistant_write_policy"], _POLICY_KEYS)
    if isinstance(out.get("write_policy"), dict):
        out["write_policy"] = _compact_dict(out["write_policy"], _POLICY_KEYS)
    if isinstance(out.get("lifecycle_decision"), dict):
        out["lifecycle_decision"] = _compact_dict(out["lifecycle_decision"], _LIFECYCLE_KEYS)
        if not out["lifecycle_decision"]:
            out.pop("lifecycle_decision", None)

    if isinstance(out.get("fact"), dict):
        out["fact"] = _compact_dict(out["fact"], _FACT_KEYS)
    if isinstance(out.get("claim"), dict):
        out["claim"] = _compact_dict(out["claim"], _CLAIM_KEYS)
    if isinstance(out.get("dialog_episode"), dict):
        out["dialog_episode"] = _compact_dict(out["dialog_episode"], _DIALOG_EPISODE_KEYS)
    if isinstance(out.get("document_outline"), dict):
        out["document_outline"] = _compact_dict(out["document_outline"], _DOCUMENT_SUMMARY_KEYS)
    if isinstance(out.get("document_summary"), dict):
        out["document_summary"] = _compact_dict(out["document_summary"], _DOCUMENT_SUMMARY_KEYS)
    if isinstance(out.get("document_claim"), dict):
        out["document_claim"] = _compact_dict(out["document_claim"], _DOCUMENT_CLAIM_KEYS)

    claims = out.get("claims")
    if isinstance(claims, list):
        compact_claims = []
        for item in list(claims or []):
            if not isinstance(item, dict):
                continue
            compact_item = _compact_dict(item, _COMPACT_CLAIM_LIST_KEYS)
            if compact_item:
                compact_claims.append(compact_item)
        if compact_claims:
            out["claims"] = compact_claims
        else:
            out.pop("claims", None)

    meta = out.get("meta")
    if isinstance(meta, dict):
        compact_meta = {}
        lang_conf = meta.get("lang_conf")
        if lang_conf is not None:
            compact_meta["lang_conf"] = lang_conf
        intent = meta.get("intent")
        if isinstance(intent, dict):
            compact_meta["intent_label"] = intent.get("label", "")
            compact_meta["intent_conf"] = intent.get("conf", 0.0)
        if meta.get("source"):
            compact_meta["source"] = meta["source"]
        if meta.get("ts"):
            compact_meta["ts"] = meta["ts"]
        if meta.get("chars"):
            compact_meta["chars"] = meta["chars"]
        if meta.get("words"):
            compact_meta["words"] = meta["words"]
        if compact_meta:
            out["meta"] = compact_meta
        else:
            out.pop("meta", None)

    out = {key: value for key, value in out.items() if not _is_empty_value(value)}

    allowed_keys = _allowed_top_level_keys(_normalize_memory_type(memory_type), out)
    if allowed_keys:
        out = _compact_dict(out, allowed_keys)

    return {key: value for key, value in out.items() if not _is_empty_value(value)}


def sanitize_storage_metadata(*, metadata: dict[str, Any] | None, storage_profile: str, memory_type: Any = None) -> dict[str, Any]:
    profile = normalize_storage_profile(storage_profile)
    out = dict(metadata or {})
    if profile == "compact":
        return compact_metadata_payload(out, memory_type=memory_type)
    return out


def prepare_storage_metadata(
    *,
    metadata: dict[str, Any] | None,
    storage_profile: str,
    memory_type: Any = None,
    source_kind: Any = None,
    thinking: str = "",
) -> dict[str, Any]:
    """Single entry point for write-path metadata sanitation.

    This helper normalizes write-time metadata first, then applies the
    storage-profile compaction rules. The goal is to keep the storage contract
    centralized in this module instead of scattering ad-hoc cleanups across the
    write path.
    """
    out = dict(metadata or {})
    hidden = str(thinking or out.get("thinking") or "").strip()
    legacy_entities = out.pop("entities", None)
    if legacy_entities and "runtime_entities" not in out:
        out["runtime_entities"] = legacy_entities
        out["legacy_runtime_entities_stripped"] = True
    out.pop("thinking", None)
    out.pop("tags", None)

    source_value = getattr(source_kind, "value", source_kind)
    source_token = str(source_value or "").strip().lower()
    if source_token:
        out["source_kind"] = source_token
    if source_token in {"assistant_reply", "assistant_thought"} and hidden:
        out["assistant_thinking_stripped"] = True

    return sanitize_storage_metadata(
        metadata=out,
        storage_profile=storage_profile,
        memory_type=memory_type,
    )


def _normalize_memory_type(memory_type: Any) -> str:
    if memory_type is None:
        return ""
    value = getattr(memory_type, "value", memory_type)
    return str(value or "").strip().lower()


def _allowed_top_level_keys(memory_type: str, metadata: dict[str, Any]) -> set[str]:
    token = str(memory_type or "").strip().lower()
    if token == "message":
        return set(_MESSAGE_METADATA_KEYS)
    if token == "fact":
        return set(_FACT_METADATA_KEYS)
    if token == "claim":
        return set(_CLAIM_METADATA_KEYS)
    if token == "episode":
        return set(_EPISODE_METADATA_KEYS)
    if token == "document":
        return set(_DOCUMENT_METADATA_KEYS)
    if token == "document_chunk":
        return set(_DOCUMENT_CHUNK_METADATA_KEYS)
    if token == "summary" and any(key in dict(metadata or {}) for key in {"document_summary", "document_id", "doc_id"}):
        return {
            "document_id",
            "doc_id",
            "summary_kind",
            "source_chunk_ids",
            "topic_keys",
            "entity_keys",
            "document_summary",
            "section_label",
            "title",
            "path",
            "source_path",
            "filename",
            "extension",
        }
    return set()


def _compact_dict(row: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in list(allowed_keys or set()):
        value = row.get(key)
        if _is_empty_value(value):
            continue
        out[key] = value
    return out


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False
