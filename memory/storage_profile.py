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
    "assistant_write_reason",
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


def normalize_storage_profile(profile: str) -> str:
    token = str(profile or "").strip().lower()
    return token if token in VALID_STORAGE_PROFILES else DEFAULT_STORAGE_PROFILE


def compact_metadata_payload(metadata: dict[str, Any] | None) -> dict[str, Any]:
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

    return {key: value for key, value in out.items() if not _is_empty_value(value)}


def sanitize_storage_metadata(*, metadata: dict[str, Any] | None, storage_profile: str) -> dict[str, Any]:
    profile = normalize_storage_profile(storage_profile)
    out = dict(metadata or {})
    if profile == "compact":
        return compact_metadata_payload(out)
    return out


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
