from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.mode_selector import normalize_mode_name
from modules.character.mode_profile import (
    MODE_BLEND_ALPHA,
    blend_mode_dialog_levels,
    resolve_mode_profile,
)
from modules.character.persona_compiler import compile_system_persona
from modules.character.storage import CharacterStorage

_MOOD_MODIFIERS: dict[str, dict[str, float]] = {
    "focused": {"strictness": 0.20, "warmth": -0.08, "sarcasm": -0.10, "verbosity": -0.12},
    "thoughtful": {"warmth": 0.08, "verbosity": 0.06},
    "teasing": {"sarcasm": 0.14, "warmth": 0.02},
    "ironic": {"sarcasm": 0.10, "warmth": -0.04},
    "soft_supportive": {"warmth": 0.16, "sarcasm": -0.14, "strictness": -0.04},
    "playful": {"sarcasm": 0.08, "warmth": 0.04},
}


@dataclass(frozen=True)
class CharacterComposeResult:
    prompt: str
    mood: str
    used_files: list[str] = field(default_factory=list)
    active_traits: list[str] = field(default_factory=list)
    effective_traits: dict[str, float] = field(default_factory=dict)
    style_coefficients: dict[str, float] = field(default_factory=dict)


class CharacterComposer:
    def __init__(self, *, trait_threshold: float = 0.55, max_trait_overlays: int = 4):
        self.trait_threshold = max(0.0, min(1.0, float(trait_threshold)))
        self.max_trait_overlays = max(1, int(max_trait_overlays))

    def compose(
        self,
        *,
        storage: CharacterStorage,
        character_id: str,
        character: dict[str, Any],
        state: dict[str, Any],
        traits: dict[str, Any],
        dialog_mode: dict[str, Any] | None = None,
        context_meta: dict[str, Any] | None = None,
        persona_snapshot: dict[str, Any] | None = None,
        identity_core_snapshot: dict[str, Any] | None = None,
    ) -> CharacterComposeResult:
        snapshot = dict(persona_snapshot or {})
        identity_core = dict(identity_core_snapshot or {})
        snapshot_mood = str(snapshot.get("mood") or "").strip().lower()
        mood = str(
            snapshot_mood
            or state.get("mood")
            or character.get("default_mood")
            or "thoughtful"
        ).strip().lower()
        if mood == "romantic_soft":
            mood = "soft_supportive"
        active = set(str(x).strip().lower() for x in list(state.get("active_traits") or []) if str(x).strip())
        disabled = set(str(x).strip().lower() for x in list(state.get("disabled_traits") or []) if str(x).strip())

        dm = dict(dialog_mode or {})
        meta = dict(context_meta or {})
        context_mods = compute_context_trait_modifiers(
            dialog_mode=dm,
            intent=str(meta.get("intent") or meta.get("meta", {}).get("intent_label") or ""),
            is_technical=bool(meta.get("is_technical", False)),
        )
        overlay_mods = _overlay_modifiers(traits)
        style_coefficients = _resolve_style_coefficients(
            traits=traits,
            dialog_mode=dm,
            context_mods=context_mods,
            mood=mood,
            overlay_mods=overlay_mods,
        )
        active_mode = normalize_mode_name(
            str(meta.get("active_mode") or dm.get("active_mode") or "chatting"),
            allow_custom=True,
        )
        mode_profile = resolve_mode_profile(mode=active_mode, character_id=character_id)
        style_coefficients = blend_mode_dialog_levels(
            base_levels=style_coefficients,
            dialog_targets=mode_profile.dialog_targets,
            alpha=MODE_BLEND_ALPHA,
        )
        style_coefficients = _apply_response_bias(
            style_coefficients=style_coefficients,
            response_bias=_as_float_map(snapshot.get("response_bias")),
        )

        effective_traits: dict[str, float] = {}
        persona_traits: dict[str, float] = {}
        for name, payload in dict(traits or {}).items():
            trait_name = str(name or "").strip().lower()
            if not trait_name or trait_name.startswith("_") or trait_name in disabled:
                continue
            row = dict(payload or {})
            if active and trait_name not in active:
                continue
            if bool(row.get("disabled", False)):
                continue
            ttype = str(row.get("type") or "scalar").strip().lower()
            value = row.get("value")
            include = False
            score = 0.0
            if ttype == "flag":
                include = bool(value)
                score = 1.0 if include else 0.0
            else:
                base_score = _to_float(value, 0.0)
                score = compute_effective_trait_value(
                    trait_name,
                    base_trait=base_score,
                    mood=mood,
                    overlay_modifier=float(overlay_mods.get(trait_name, 0.0)),
                    context_modifier=float(context_mods.get(trait_name, 0.0)),
                )
                include = score >= self.trait_threshold
                effective_traits[trait_name] = float(score)
            if not include:
                continue
            if trait_name in {"warmth", "sarcasm", "verbosity", "strictness", "teasing", "empathy"}:
                persona_traits[trait_name] = float(score)
            elif trait_name == "playfulness":
                persona_traits.setdefault("teasing", float(score))
            elif trait_name == "thoughtfulness":
                persona_traits.setdefault("empathy", float(score))

        persona_payload = dict(storage.load_persona_state(character_id) or {})
        stable_traits_payload = dict(persona_payload.get("traits") or {})
        if not stable_traits_payload:
            stable_traits_payload.update(persona_traits)
        stable_traits_payload = _apply_identity_core_trait_policy(
            current=stable_traits_payload,
            identity_core=identity_core,
        )
        stable_traits_payload = _overlay_stable_traits(
            current=stable_traits_payload,
            overlay=_as_float_map(snapshot.get("stable_traits")),
        )
        persona_payload["traits"] = stable_traits_payload
        persona_payload["mood"] = mood
        persona_payload["emotional_state"] = dict(storage.load_emotion_state(character_id) or {})
        if not dict(persona_payload.get("emotional_state") or {}).get("mood"):
            persona_payload["emotional_state"] = dict(persona_payload.get("emotional_state") or {})
            persona_payload["emotional_state"]["mood"] = mood
        persona_payload["relation_state"] = _merge_relation_state(
            current=persona_payload.get("relation_state"),
            overlay=snapshot.get("relation_state"),
        )
        persona_payload["boundaries"] = _merge_boundaries(
            current=persona_payload.get("boundaries"),
            overlay=snapshot.get("boundaries"),
        )
        persona_payload["emotional_handling"] = _merge_emotional_handling(
            current=persona_payload.get("emotional_handling"),
            overlay=snapshot.get("emotional_handling"),
        )
        user_addressing = _overlay_user_addressing(
            current=storage.load_user_addressing(character_id),
            overlay=dict(identity_core.get("addressing") or {}),
        )
        user_addressing = _overlay_user_addressing(
            current=user_addressing,
            overlay=snapshot.get("user_addressing"),
        )
        for key in ("warmth", "sarcasm", "strictness", "verbosity", "empathy", "teasing"):
            if key in stable_traits_payload:
                effective_traits[key] = float(_to_float(stable_traits_payload.get(key), 0.5))
        prompt, _ = compile_system_persona(
            character_id=character_id,
            persona_state=persona_payload,
            active_mode=active_mode,
            user_addressing=user_addressing,
        )
        used = [
            f"spec:characters/{character_id}/persona_spec.json",
            f"runtime:characters_runtime/{character_id}/persona_state.json",
            f"runtime:characters_runtime/{character_id}/emotion_state.json",
            f"runtime:characters_runtime/{character_id}/user_addressing.json",
        ]
        active_traits = sorted([k for k in ("warmth", "sarcasm", "strictness", "verbosity", "empathy", "teasing") if k in stable_traits_payload])
        return CharacterComposeResult(
            prompt=prompt,
            mood=mood,
            used_files=used,
            active_traits=active_traits,
            effective_traits=effective_traits,
            style_coefficients=style_coefficients,
        )


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def compute_effective_trait_value(
    trait_name: str,
    *,
    base_trait: float,
    mood: str = "",
    overlay_modifier: float = 0.0,
    context_modifier: float = 0.0,
) -> float:
    key = str(trait_name or "").strip().lower()
    mood_key = str(mood or "").strip().lower()
    mood_modifier = 0.0
    if mood_key in _MOOD_MODIFIERS:
        mood_modifier = float(_MOOD_MODIFIERS[mood_key].get(key, 0.0))
    value = float(base_trait) + mood_modifier + float(overlay_modifier) + float(context_modifier)
    return max(0.0, min(1.0, value))


def compute_context_trait_modifiers(
    *,
    dialog_mode: dict[str, Any] | None = None,
    intent: str = "",
    is_technical: bool = False,
) -> dict[str, float]:
    mode = dict(dialog_mode or {})
    out = {
        "warmth": 0.0,
        "sarcasm": 0.0,
        "strictness": 0.0,
        "verbosity": 0.0,
    }

    for key in ("warmth", "sarcasm", "strictness", "verbosity"):
        target = mode.get(f"{key}_level")
        if target is None:
            continue
        try:
            value = float(target)
        except Exception:
            continue
        baseline = 0.5
        out[key] += (value - baseline) * 0.6

    intent_key = str(intent or "").strip().lower()
    if intent_key in {"bug_report", "task", "code_review", "question"}:
        out["strictness"] += 0.18
        out["verbosity"] -= 0.06
        out["sarcasm"] -= 0.10
    if is_technical:
        out["sarcasm"] -= 0.16
        out["strictness"] += 0.20
        out["verbosity"] -= 0.08

    for key in list(out.keys()):
        out[key] = max(-0.5, min(0.5, float(out[key])))
    return out


def _trait_scalar(traits: dict[str, Any], key: str) -> float:
    row = dict(traits.get(str(key).strip().lower()) or {})
    value = row.get("value")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _to_float(value, 0.0)


def _overlay_modifiers(traits: dict[str, Any]) -> dict[str, float]:
    sarcasm = _trait_scalar(traits, "sarcasm")
    warmth = _trait_scalar(traits, "warmth")
    romance = _trait_scalar(traits, "romance")
    out = {
        "sarcasm": 0.0,
        "warmth": 0.0,
        "strictness": 0.0,
        "verbosity": 0.0,
    }
    if sarcasm >= 0.72:
        out["sarcasm"] += 0.12
        out["warmth"] -= 0.04
    if warmth >= 0.72:
        out["warmth"] += 0.12
        out["sarcasm"] -= 0.04
    if romance >= 0.66:
        out["warmth"] += 0.08
        out["strictness"] -= 0.05
    return out


def _resolve_style_coefficients(
    *,
    traits: dict[str, Any],
    dialog_mode: dict[str, Any],
    context_mods: dict[str, float],
    mood: str,
    overlay_mods: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("warmth", "sarcasm", "strictness", "verbosity"):
        base = _trait_scalar(traits, key) if key in traits else 0.5
        mood_modifier = 0.0
        if mood == "focused" and key in {"strictness"}:
            mood_modifier = 0.14
        elif mood == "focused" and key == "verbosity":
            mood_modifier = -0.10
        elif mood == "teasing" and key == "sarcasm":
            mood_modifier = 0.12
        elif mood == "soft_supportive" and key == "warmth":
            mood_modifier = 0.15
        elif mood == "playful" and key == "sarcasm":
            mood_modifier = 0.08
        dialog_target = _to_float(dialog_mode.get(f"{key}_level"), base)
        context_modifier = _to_float(context_mods.get(key), 0.0)
        overlay_modifier = _to_float(overlay_mods.get(key), 0.0)
        effective = compute_effective_trait_value(
            key,
            base_trait=base,
            mood=mood,
            overlay_modifier=overlay_modifier + mood_modifier,
            context_modifier=context_modifier + ((dialog_target - base) * 0.6),
        )
        out[key] = max(0.0, min(1.0, float(effective)))
    return out


def _as_float_map(value: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_key, raw_value in dict(value or {}).items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        try:
            out[key] = max(0.0, min(1.0, float(raw_value)))
        except Exception:
            continue
    return out


def _overlay_stable_traits(*, current: dict[str, Any] | None, overlay: dict[str, float] | None) -> dict[str, float]:
    merged: dict[str, float] = {}
    for raw_key, raw_value in dict(current or {}).items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        try:
            merged[key] = max(0.0, min(1.0, float(raw_value)))
        except Exception:
            continue
    for key, value in dict(overlay or {}).items():
        if key in merged:
            merged[key] = max(0.0, min(1.0, (float(merged[key]) * 0.35) + (float(value) * 0.65)))
        else:
            merged[key] = max(0.0, min(1.0, float(value)))
    return merged


def _apply_identity_core_trait_policy(*, current: dict[str, Any] | None, identity_core: dict[str, Any] | None) -> dict[str, float]:
    merged: dict[str, float] = {}
    for raw_key, raw_value in dict(current or {}).items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        try:
            merged[key] = max(0.0, min(1.0, float(raw_value)))
        except Exception:
            continue

    baseline = dict(dict(identity_core or {}).get("assistant_trait_baseline") or {})
    if not baseline:
        return merged

    if "warmth_baseline" in baseline:
        merged["warmth"] = max(0.0, min(1.0, _to_float(baseline.get("warmth_baseline"), merged.get("warmth", 0.5))))
    if "directness_baseline" in baseline:
        merged["directness"] = max(0.0, min(1.0, _to_float(baseline.get("directness_baseline"), merged.get("directness", 0.5))))
    if "empathy_floor" in baseline:
        merged["empathy"] = max(
            max(0.0, min(1.0, _to_float(merged.get("empathy"), 0.5))),
            max(0.0, min(1.0, _to_float(baseline.get("empathy_floor"), 0.0))),
        )
    if "professionalism_floor" in baseline:
        merged["professionalism"] = max(
            max(0.0, min(1.0, _to_float(merged.get("professionalism"), 0.5))),
            max(0.0, min(1.0, _to_float(baseline.get("professionalism_floor"), 0.0))),
        )
    if "sarcasm_ceiling" in baseline:
        merged["sarcasm"] = min(
            max(0.0, min(1.0, _to_float(merged.get("sarcasm"), 0.5))),
            max(0.0, min(1.0, _to_float(baseline.get("sarcasm_ceiling"), 1.0))),
        )
    return merged


def _merge_relation_state(*, current: Any, overlay: Any) -> dict[str, Any]:
    merged = dict(current or {})
    for raw_key, raw_value in dict(overlay or {}).items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        merged[key] = raw_value
    return merged


def _overlay_user_addressing(*, current: Any, overlay: Any) -> dict[str, Any]:
    merged = dict(current or {})
    row = dict(overlay or {})
    for key in ("canonical_name", "use_name_by_default", "allow_diminutives"):
        if key in row:
            merged[key] = row.get(key)
    for key in ("allowed_forms", "forbidden_forms"):
        values = [str(x).strip() for x in list(row.get(key) or []) if str(x).strip()]
        if values:
            merged[key] = values
    return merged


def _merge_boundaries(*, current: Any, overlay: Any) -> dict[str, Any]:
    merged = dict(current or {})
    for raw_key, raw_value in dict(overlay or {}).items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        merged[key] = bool(raw_value)
    return merged


def _merge_emotional_handling(*, current: Any, overlay: Any) -> dict[str, Any]:
    merged = dict(current or {})
    for raw_key, raw_value in dict(overlay or {}).items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        if key in {"deescalate_on_irritation", "treat_short_replies_as_low_bandwidth"}:
            merged[key] = bool(raw_value)
            continue
        try:
            merged[key] = max(0.0, min(1.0, float(raw_value)))
        except Exception:
            continue
    return merged


def _apply_response_bias(*, style_coefficients: dict[str, float], response_bias: dict[str, float]) -> dict[str, float]:
    out = {str(k): max(0.0, min(1.0, float(v))) for k, v in dict(style_coefficients or {}).items()}
    if not response_bias:
        return out
    technical_mode = float(response_bias.get("technical_mode", 0.0) or 0.0)
    if technical_mode > 0.0:
        out["strictness"] = max(0.0, min(1.0, out.get("strictness", 0.5) + (0.18 * technical_mode)))
        out["sarcasm"] = max(0.0, min(1.0, out.get("sarcasm", 0.5) - (0.16 * technical_mode)))
    short_answer = float(response_bias.get("needs_short_answer", 0.0) or 0.0)
    if short_answer > 0.0:
        out["verbosity"] = max(0.0, min(1.0, out.get("verbosity", 0.5) - (0.22 * short_answer)))
    frustration_softening = float(response_bias.get("frustration_softening", 0.0) or 0.0)
    if frustration_softening > 0.0:
        out["warmth"] = max(0.0, min(1.0, out.get("warmth", 0.5) + (0.18 * frustration_softening)))
        out["sarcasm"] = max(0.0, min(1.0, out.get("sarcasm", 0.5) - (0.18 * frustration_softening)))
    warmth_upshift = float(response_bias.get("warmth_upshift", 0.0) or 0.0)
    if warmth_upshift > 0.0:
        out["warmth"] = max(0.0, min(1.0, out.get("warmth", 0.5) + (0.24 * warmth_upshift)))
    playfulness_downshift = float(response_bias.get("playfulness_downshift", 0.0) or 0.0)
    if playfulness_downshift > 0.0:
        out["sarcasm"] = max(0.0, min(1.0, out.get("sarcasm", 0.5) - (0.22 * playfulness_downshift)))
    return out
