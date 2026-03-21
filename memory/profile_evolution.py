from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from memory.memory_models import MemoryRecord, MemoryScope

PROFILE_LAYER_PRIORITY: tuple[str, ...] = (
    "persistent_traits",
    "volatile_preferences",
    "session_preferences",
)

PROFILE_FACT_GROUPS: dict[str, str] = {
    "identity_name": "addressing.canonical_name",
    "name_allowed_forms": "addressing.allowed_forms",
    "name_forbidden_forms": "addressing.forbidden_forms",
    "use_name_by_default": "addressing.use_name_by_default",
    "allow_diminutives": "addressing.allow_diminutives",
    "assistant_warmth": "assistant.warmth",
    "assistant_directness": "assistant.directness",
    "assistant_sarcasm": "assistant.sarcasm",
    "assistant_empathy": "assistant.empathy",
    "assistant_professionalism": "assistant.professionalism",
    "assistant_verbosity": "assistant.verbosity",
    "assistant_teasing": "assistant.teasing",
    "preferred_answer_brevity": "response.brevity",
    "prefers_short_answers": "response.brevity",
    "prefers_examples_on_user_code": "response.examples_on_user_code",
    "allow_light_teasing": "tone.teasing_permission",
    "avoid_baby_talk": "tone.boundary.baby_talk",
    "avoid_baby_tone": "tone.boundary.baby_talk",
    "avoid_repeating_question": "tone.boundary.repeating_question",
    "avoid_overformal_tone": "tone.boundary.overformal_tone",
    "do_not_invent_user_facts": "tone.boundary.invent_user_facts",
    "avoid_inventing_user_facts": "tone.boundary.invent_user_facts",
    "relation_familiarity": "relation.familiarity",
    "relation_trust": "relation.trust",
    "relation_technical_collaboration": "relation.technical_collaboration",
    "preferred_editor": "preferences.editor",
    "preferred_language": "preferences.language",
    "preferred_tool": "preferences.tool",
    "preferred_style": "preferences.style",
    "interest": "preferences.interests",
    "project_name": "project.name",
    "task_goal": "task.goal",
    "environment_os": "environment.os",
    "environment_runtime_python": "environment.python",
    "environment_ram_gb": "environment.ram",
    "environment_memory_gb": "environment.ram",
    "environment_gpu_model": "environment.gpu_model",
    "environment_gpu_vram_gb": "environment.gpu_vram",
}

PROFILE_GROUP_POLICY: dict[str, str] = {
    "addressing.canonical_name": "singleton",
    "addressing.allowed_forms": "multi",
    "addressing.forbidden_forms": "multi",
    "addressing.use_name_by_default": "singleton",
    "addressing.allow_diminutives": "singleton",
    "assistant.warmth": "soft_singleton",
    "assistant.directness": "soft_singleton",
    "assistant.sarcasm": "soft_singleton",
    "assistant.empathy": "soft_singleton",
    "assistant.professionalism": "soft_singleton",
    "assistant.verbosity": "soft_singleton",
    "assistant.teasing": "soft_singleton",
    "response.brevity": "soft_singleton",
    "response.examples_on_user_code": "singleton",
    "tone.teasing_permission": "soft_singleton",
    "tone.boundary.baby_talk": "singleton",
    "tone.boundary.repeating_question": "singleton",
    "tone.boundary.overformal_tone": "singleton",
    "tone.boundary.invent_user_facts": "singleton",
    "relation.familiarity": "soft_singleton",
    "relation.trust": "soft_singleton",
    "relation.technical_collaboration": "soft_singleton",
    "preferences.editor": "soft_singleton",
    "preferences.language": "soft_singleton",
    "preferences.tool": "multi",
    "preferences.style": "soft_singleton",
    "preferences.interests": "multi",
    "project.name": "multi",
    "task.goal": "multi",
    "environment.os": "singleton",
    "environment.python": "singleton",
    "environment.ram": "singleton",
    "environment.gpu_model": "singleton",
    "environment.gpu_vram": "singleton",
}

_RESERVED_SNAPSHOT_KEYS = {
    "namespace",
    "active_facts",
    "conflicts",
    "persistent_traits",
    "volatile_preferences",
    "session_preferences",
    "resolved_profile",
    "updated_at",
    "debug",
}


@dataclass(frozen=True)
class ProfileFactRule:
    profile_type: str
    group: str
    group_mode: str
    half_life_days: float
    protected: bool = False
    requires_confirmation: bool = False
    override_min_signal: float = 0.82
    override_margin: float = 0.10


_PROFILE_RULES: dict[str, ProfileFactRule] = {
    "identity_name": ProfileFactRule(
        profile_type="persistent_traits",
        group="addressing.canonical_name",
        group_mode="singleton",
        half_life_days=720.0,
        protected=True,
        requires_confirmation=True,
        override_min_signal=0.90,
        override_margin=0.14,
    ),
    "name_allowed_forms": ProfileFactRule(
        profile_type="persistent_traits",
        group="addressing.allowed_forms",
        group_mode="multi",
        half_life_days=720.0,
        protected=True,
        requires_confirmation=True,
        override_min_signal=0.88,
    ),
    "name_forbidden_forms": ProfileFactRule(
        profile_type="persistent_traits",
        group="addressing.forbidden_forms",
        group_mode="multi",
        half_life_days=720.0,
        protected=True,
        requires_confirmation=False,
        override_min_signal=0.86,
    ),
    "use_name_by_default": ProfileFactRule(
        profile_type="persistent_traits",
        group="addressing.use_name_by_default",
        group_mode="singleton",
        half_life_days=720.0,
        protected=True,
        requires_confirmation=True,
        override_min_signal=0.88,
    ),
    "allow_diminutives": ProfileFactRule(
        profile_type="persistent_traits",
        group="addressing.allow_diminutives",
        group_mode="singleton",
        half_life_days=720.0,
        protected=True,
        requires_confirmation=True,
        override_min_signal=0.88,
    ),
    "assistant_warmth": ProfileFactRule(
        profile_type="persistent_traits",
        group="assistant.warmth",
        group_mode="soft_singleton",
        half_life_days=540.0,
        protected=True,
        override_min_signal=0.84,
        override_margin=0.08,
    ),
    "assistant_directness": ProfileFactRule(
        profile_type="persistent_traits",
        group="assistant.directness",
        group_mode="soft_singleton",
        half_life_days=540.0,
        protected=True,
        override_min_signal=0.84,
        override_margin=0.08,
    ),
    "assistant_sarcasm": ProfileFactRule(
        profile_type="persistent_traits",
        group="assistant.sarcasm",
        group_mode="soft_singleton",
        half_life_days=540.0,
        protected=True,
        override_min_signal=0.84,
        override_margin=0.08,
    ),
    "assistant_empathy": ProfileFactRule(
        profile_type="persistent_traits",
        group="assistant.empathy",
        group_mode="soft_singleton",
        half_life_days=540.0,
        protected=True,
        override_min_signal=0.84,
        override_margin=0.08,
    ),
    "assistant_professionalism": ProfileFactRule(
        profile_type="persistent_traits",
        group="assistant.professionalism",
        group_mode="soft_singleton",
        half_life_days=540.0,
        protected=True,
        override_min_signal=0.84,
        override_margin=0.08,
    ),
    "assistant_verbosity": ProfileFactRule(
        profile_type="persistent_traits",
        group="assistant.verbosity",
        group_mode="soft_singleton",
        half_life_days=540.0,
        protected=True,
        override_min_signal=0.84,
        override_margin=0.08,
    ),
    "assistant_teasing": ProfileFactRule(
        profile_type="persistent_traits",
        group="assistant.teasing",
        group_mode="soft_singleton",
        half_life_days=540.0,
        protected=True,
        override_min_signal=0.84,
        override_margin=0.08,
    ),
    "prefers_examples_on_user_code": ProfileFactRule(
        profile_type="persistent_traits",
        group="response.examples_on_user_code",
        group_mode="singleton",
        half_life_days=540.0,
        protected=True,
        override_min_signal=0.88,
    ),
    "allow_light_teasing": ProfileFactRule(
        profile_type="volatile_preferences",
        group="tone.teasing_permission",
        group_mode="soft_singleton",
        half_life_days=45.0,
        protected=False,
        override_min_signal=0.78,
        override_margin=0.06,
    ),
    "preferred_answer_brevity": ProfileFactRule(
        profile_type="volatile_preferences",
        group="response.brevity",
        group_mode="soft_singleton",
        half_life_days=28.0,
        protected=False,
        override_min_signal=0.76,
    ),
    "prefers_short_answers": ProfileFactRule(
        profile_type="volatile_preferences",
        group="response.brevity",
        group_mode="soft_singleton",
        half_life_days=28.0,
        protected=False,
        override_min_signal=0.76,
    ),
    "avoid_baby_talk": ProfileFactRule(
        profile_type="persistent_traits",
        group="tone.boundary.baby_talk",
        group_mode="singleton",
        half_life_days=720.0,
        protected=True,
        override_min_signal=0.90,
    ),
    "avoid_baby_tone": ProfileFactRule(
        profile_type="persistent_traits",
        group="tone.boundary.baby_talk",
        group_mode="singleton",
        half_life_days=720.0,
        protected=True,
        override_min_signal=0.90,
    ),
    "avoid_repeating_question": ProfileFactRule(
        profile_type="persistent_traits",
        group="tone.boundary.repeating_question",
        group_mode="singleton",
        half_life_days=720.0,
        protected=True,
        override_min_signal=0.90,
    ),
    "avoid_overformal_tone": ProfileFactRule(
        profile_type="persistent_traits",
        group="tone.boundary.overformal_tone",
        group_mode="singleton",
        half_life_days=720.0,
        protected=True,
        override_min_signal=0.90,
    ),
    "do_not_invent_user_facts": ProfileFactRule(
        profile_type="persistent_traits",
        group="tone.boundary.invent_user_facts",
        group_mode="singleton",
        half_life_days=720.0,
        protected=True,
        override_min_signal=0.90,
    ),
    "avoid_inventing_user_facts": ProfileFactRule(
        profile_type="persistent_traits",
        group="tone.boundary.invent_user_facts",
        group_mode="singleton",
        half_life_days=720.0,
        protected=True,
        override_min_signal=0.90,
    ),
    "relation_familiarity": ProfileFactRule(
        profile_type="persistent_traits",
        group="relation.familiarity",
        group_mode="soft_singleton",
        half_life_days=240.0,
        protected=False,
        override_min_signal=0.76,
    ),
    "relation_trust": ProfileFactRule(
        profile_type="persistent_traits",
        group="relation.trust",
        group_mode="soft_singleton",
        half_life_days=240.0,
        protected=False,
        override_min_signal=0.76,
    ),
    "relation_technical_collaboration": ProfileFactRule(
        profile_type="persistent_traits",
        group="relation.technical_collaboration",
        group_mode="soft_singleton",
        half_life_days=240.0,
        protected=False,
        override_min_signal=0.76,
    ),
    "preferred_editor": ProfileFactRule(
        profile_type="volatile_preferences",
        group="preferences.editor",
        group_mode="soft_singleton",
        half_life_days=60.0,
        override_min_signal=0.74,
    ),
    "preferred_language": ProfileFactRule(
        profile_type="volatile_preferences",
        group="preferences.language",
        group_mode="soft_singleton",
        half_life_days=60.0,
        override_min_signal=0.74,
    ),
    "preferred_tool": ProfileFactRule(
        profile_type="volatile_preferences",
        group="preferences.tool",
        group_mode="multi",
        half_life_days=45.0,
        override_min_signal=0.72,
    ),
    "preferred_style": ProfileFactRule(
        profile_type="volatile_preferences",
        group="preferences.style",
        group_mode="soft_singleton",
        half_life_days=45.0,
        override_min_signal=0.74,
    ),
    "interest": ProfileFactRule(
        profile_type="volatile_preferences",
        group="preferences.interests",
        group_mode="multi",
        half_life_days=120.0,
        override_min_signal=0.70,
    ),
    "project_name": ProfileFactRule(
        profile_type="session_preferences",
        group="project.name",
        group_mode="multi",
        half_life_days=7.0,
        override_min_signal=0.68,
    ),
    "task_goal": ProfileFactRule(
        profile_type="session_preferences",
        group="task.goal",
        group_mode="multi",
        half_life_days=2.0,
        override_min_signal=0.66,
    ),
    "environment_os": ProfileFactRule(
        profile_type="persistent_traits",
        group="environment.os",
        group_mode="singleton",
        half_life_days=540.0,
        override_min_signal=0.82,
    ),
    "environment_runtime_python": ProfileFactRule(
        profile_type="persistent_traits",
        group="environment.python",
        group_mode="singleton",
        half_life_days=240.0,
        override_min_signal=0.80,
    ),
    "environment_ram_gb": ProfileFactRule(
        profile_type="persistent_traits",
        group="environment.ram",
        group_mode="singleton",
        half_life_days=360.0,
        override_min_signal=0.82,
    ),
    "environment_memory_gb": ProfileFactRule(
        profile_type="persistent_traits",
        group="environment.ram",
        group_mode="singleton",
        half_life_days=360.0,
        override_min_signal=0.82,
    ),
    "environment_gpu_model": ProfileFactRule(
        profile_type="persistent_traits",
        group="environment.gpu_model",
        group_mode="singleton",
        half_life_days=360.0,
        override_min_signal=0.82,
    ),
    "environment_gpu_vram_gb": ProfileFactRule(
        profile_type="persistent_traits",
        group="environment.gpu_vram",
        group_mode="singleton",
        half_life_days=360.0,
        override_min_signal=0.82,
    ),
}


def profile_fact_rule(predicate: str, *, record: MemoryRecord | None = None) -> ProfileFactRule:
    pred = str(predicate or "").strip().lower()
    if pred in _PROFILE_RULES:
        return _PROFILE_RULES[pred]

    scope = getattr(record, "scope", None)
    scope_token = str(scope.value if isinstance(scope, MemoryScope) else scope or "").strip().lower()
    if scope_token in {MemoryScope.SESSION.value, MemoryScope.TEMPORARY.value}:
        return ProfileFactRule(
            profile_type="session_preferences",
            group=str(PROFILE_FACT_GROUPS.get(pred) or pred or "session.fact"),
            group_mode=str(PROFILE_GROUP_POLICY.get(str(PROFILE_FACT_GROUPS.get(pred) or pred or "session.fact")) or "soft_singleton"),
            half_life_days=3.0,
            override_min_signal=0.66,
        )
    if pred.startswith("preferred_"):
        group = str(PROFILE_FACT_GROUPS.get(pred) or f"preferences.{pred[10:]}")
        return ProfileFactRule(
            profile_type="volatile_preferences",
            group=group,
            group_mode=str(PROFILE_GROUP_POLICY.get(group) or "soft_singleton"),
            half_life_days=45.0,
            override_min_signal=0.72,
        )
    if pred.startswith("assistant_") or pred.startswith("relation_") or pred.startswith("identity_"):
        group = str(PROFILE_FACT_GROUPS.get(pred) or pred)
        return ProfileFactRule(
            profile_type="persistent_traits",
            group=group,
            group_mode=str(PROFILE_GROUP_POLICY.get(group) or "soft_singleton"),
            half_life_days=360.0,
            protected=pred.startswith(("assistant_", "identity_")),
            override_min_signal=0.80,
            override_margin=0.08,
        )
    group = str(PROFILE_FACT_GROUPS.get(pred) or pred)
    return ProfileFactRule(
        profile_type="persistent_traits",
        group=group,
        group_mode=str(PROFILE_GROUP_POLICY.get(group) or "multi"),
        half_life_days=180.0,
        override_min_signal=0.72,
    )


def explicit_confirmation_present(record: MemoryRecord | None) -> bool:
    if record is None:
        return False
    meta = dict(getattr(record, "metadata", {}) or {})
    fact = dict(meta.get("fact") or {})
    fact_meta = dict(fact.get("metadata") or {})
    write_policy = dict(meta.get("write_policy") or {})
    signals = dict(write_policy.get("signals") or {})
    return any(
        bool(value)
        for value in (
            meta.get("explicit_confirmation"),
            meta.get("identity_core_explicit_confirmation"),
            fact.get("explicit_confirmation"),
            fact.get("identity_core_explicit_confirmation"),
            fact_meta.get("explicit_confirmation"),
            fact_meta.get("identity_core_explicit_confirmation"),
            signals.get("explicit_confirmation"),
        )
    )


def protected_profile_override_reason(
    *,
    new_record: MemoryRecord,
    incumbent: MemoryRecord | None,
) -> str:
    predicate = _predicate_for_record(new_record)
    rule = profile_fact_rule(predicate, record=new_record)
    if not rule.protected or incumbent is None:
        return ""
    if _value_norm(new_record) == _value_norm(incumbent):
        return ""
    if explicit_confirmation_present(new_record):
        return ""
    new_signal = _override_signal(new_record, rule=rule)
    old_signal = _record_confidence(incumbent)
    required_signal = min(0.98, max(float(rule.override_min_signal), old_signal + float(rule.override_margin)))
    if rule.requires_confirmation:
        required_signal = max(required_signal, 0.995)
    if new_signal >= required_signal:
        return ""
    return (
        "protected_profile_requires_confirmation"
        if rule.requires_confirmation
        else "protected_profile_requires_strong_signal"
    )


def build_profile_entry(record: MemoryRecord, *, now_ts: float | None = None) -> dict[str, Any]:
    predicate = _predicate_for_record(record)
    rule = profile_fact_rule(predicate, record=record)
    fact = dict(dict(getattr(record, "metadata", {}) or {}).get("fact") or {})
    value = fact.get("value")
    created_at = _safe_float(getattr(record, "created_at", 0.0))
    updated_at = _safe_float(getattr(record, "updated_at", 0.0)) or created_at
    now_value = float(now_ts or updated_at or time.time())
    age_days = max(0.0, (now_value - updated_at) / 86400.0) if updated_at > 0.0 else 0.0
    decay_factor = _decay_factor(age_days=age_days, half_life_days=rule.half_life_days)
    base_confidence = _record_confidence(record)
    effective_confidence = _clamp01(base_confidence * decay_factor)
    source_kind = str(fact.get("source_kind") or "").strip().lower()
    source_role = str(fact.get("source_role") or "").strip().lower()
    return {
        "record_id": str(getattr(record, "id", "") or ""),
        "canonical_key": str(dict(getattr(record, "metadata", {}) or {}).get("canonical_key") or ""),
        "predicate": predicate,
        "value": value,
        "confidence": base_confidence,
        "effective_confidence": effective_confidence,
        "decay_factor": decay_factor,
        "age_days": round(age_days, 6),
        "profile_type": rule.profile_type,
        "group": rule.group,
        "group_mode": rule.group_mode,
        "protected": bool(rule.protected),
        "requires_confirmation": bool(rule.requires_confirmation),
        "explicit_confirmation": bool(explicit_confirmation_present(record)),
        "source_kind": source_kind,
        "source_role": source_role,
        "scope": str(getattr(record, "scope", "") or ""),
        "updated_at": updated_at,
        "status": str(getattr(getattr(record, "status", None), "value", getattr(record, "status", "")) or ""),
    }


def flatten_governor_profile_snapshot(
    snapshot: dict[str, Any] | None,
    *,
    layer_priority: tuple[str, ...] = PROFILE_LAYER_PRIORITY,
    include_active_facts: bool = True,
) -> dict[str, Any]:
    row = dict(snapshot or {})
    flat: dict[str, Any] = {
        key: value
        for key, value in row.items()
        if key not in _RESERVED_SNAPSHOT_KEYS
    }

    for layer_name in layer_priority:
        _merge_entry_layer(
            flat,
            row.get(layer_name),
        )

    if include_active_facts:
        _merge_entry_layer(
            flat,
            row.get("active_facts"),
        )
    return flat


def resolve_snapshot_now_ts(active_fact_rows: list[MemoryRecord]) -> float:
    latest = 0.0
    for row in list(active_fact_rows or []):
        latest = max(
            latest,
            _safe_float(getattr(row, "updated_at", 0.0)),
            _safe_float(getattr(row, "created_at", 0.0)),
        )
    if latest <= 0.0:
        return float(time.time())
    real_now = float(time.time())
    if latest < 1_000_000_000.0 and real_now >= 1_000_000_000.0:
        return latest
    return max(real_now, latest)


def _merge_entry_layer(flat: dict[str, Any], layer: Any) -> None:
    entries = _normalized_layer_entries(layer)
    for item in entries:
        predicate = str(item.get("predicate") or "").strip().lower()
        if not predicate:
            continue
        value = item.get("value")
        if _is_empty(value):
            continue
        group_mode = str(item.get("group_mode") or profile_fact_rule(predicate).group_mode).strip().lower()
        if group_mode == "multi":
            existing = flat.get(predicate)
            values = _to_clean_list(existing)
            for extra in _to_clean_list(value):
                if extra not in values:
                    values.append(extra)
            flat[predicate] = values
            continue
        if predicate not in flat:
            flat[predicate] = value


def _normalized_layer_entries(layer: Any) -> list[dict[str, Any]]:
    rows = [
        dict(item)
        for item in list(dict(layer or {}).values())
        if isinstance(item, dict)
    ]
    if not rows:
        return []
    singles: dict[str, dict[str, Any]] = {}
    multi: list[dict[str, Any]] = []
    for item in rows:
        predicate = str(item.get("predicate") or "").strip().lower()
        if not predicate:
            continue
        group_mode = str(item.get("group_mode") or profile_fact_rule(predicate).group_mode).strip().lower()
        if group_mode == "multi":
            multi.append(item)
            continue
        previous = singles.get(predicate)
        if previous is None or _entry_score(item) >= _entry_score(previous):
            singles[predicate] = item
    return [*singles.values(), *multi]


def _entry_score(item: dict[str, Any]) -> float:
    return _safe_float(item.get("effective_confidence") or item.get("confidence") or 0.0)


def _override_signal(record: MemoryRecord, *, rule: ProfileFactRule) -> float:
    meta = dict(getattr(record, "metadata", {}) or {})
    fact = dict(meta.get("fact") or {})
    source_kind = str(fact.get("source_kind") or "").strip().lower()
    source_role = str(fact.get("source_role") or "").strip().lower()
    relation = str(meta.get("relation") or fact.get("relation") or "").strip().lower()
    score = _record_confidence(record)
    if source_role == "user":
        score += 0.02
    if source_kind in {"identity_core_rule", "user_direct_preference", "user_direct_prohibition"}:
        score += 0.05
    if relation in {"identity", "preference", "style", "relation"}:
        score += 0.01
    if str(getattr(record, "scope", "") or "") in {MemoryScope.GLOBAL_USER.value, MemoryScope.CHARACTER.value}:
        score += 0.02
    if explicit_confirmation_present(record):
        score += 0.10
    return _clamp01(score)


def _record_confidence(record: MemoryRecord) -> float:
    meta = dict(getattr(record, "metadata", {}) or {})
    fact = dict(meta.get("fact") or {})
    return _clamp01(_safe_float(fact.get("confidence") or getattr(record, "confidence", 0.0)))


def _predicate_for_record(record: MemoryRecord | None) -> str:
    meta = dict(getattr(record, "metadata", {}) or {})
    fact = dict(meta.get("fact") or {})
    return str(fact.get("predicate") or "").strip().lower()


def _value_norm(record: MemoryRecord | None) -> str:
    if record is None:
        return ""
    meta = dict(getattr(record, "metadata", {}) or {})
    fact = dict(meta.get("fact") or {})
    return str(fact.get("value") or "").strip().lower()


def _decay_factor(*, age_days: float, half_life_days: float) -> float:
    age = max(0.0, float(age_days or 0.0))
    half_life = max(0.001, float(half_life_days or 0.001))
    return max(0.05, math.pow(0.5, age / half_life))


def _to_clean_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            if _is_empty(item):
                continue
            if item not in out:
                out.append(item)
        return out
    if _is_empty(value):
        return []
    return [value]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return str(value).strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value or 0.0)))
