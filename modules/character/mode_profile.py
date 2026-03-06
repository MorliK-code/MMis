from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.mode_selector import normalize_mode_name
from core.spec_registry import load_character_spec, load_spec

MODE_BLEND_ALPHA = 0.70
_TRAIT_KEYS = ("warmth", "sarcasm", "strictness", "verbosity", "empathy", "teasing")
_DIALOG_KEYS = ("warmth", "sarcasm", "strictness", "verbosity")


@dataclass(frozen=True)
class ModeProfile:
    mode_id: str
    profile_id: str
    source: str
    trait_targets: dict[str, float]
    dialog_targets: dict[str, float]
    prompt_lines: list[str]


_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "chatting": {
        "trait_targets": {"warmth": 0.72, "sarcasm": 0.22, "strictness": 0.38, "verbosity": 0.62, "empathy": 0.66, "teasing": 0.38},
        "dialog_targets": {"warmth": 0.72, "sarcasm": 0.30, "strictness": 0.38, "verbosity": 0.60},
        "prompt_lines": ["Friendly conversational mode.", "Natural warm tone with light humor when relevant."],
    },
    "helper": {
        "trait_targets": {"warmth": 0.68, "sarcasm": 0.08, "strictness": 0.74, "verbosity": 0.48, "empathy": 0.70, "teasing": 0.05},
        "dialog_targets": {"warmth": 0.66, "sarcasm": 0.06, "strictness": 0.78, "verbosity": 0.44},
        "prompt_lines": ["Helper mode.", "Prioritize clear support, concise steps, and practical guidance."],
    },
    "engineer": {
        "trait_targets": {"warmth": 0.52, "sarcasm": 0.04, "strictness": 0.86, "verbosity": 0.54, "empathy": 0.52, "teasing": 0.00},
        "dialog_targets": {"warmth": 0.46, "sarcasm": 0.05, "strictness": 0.88, "verbosity": 0.42},
        "prompt_lines": ["Engineer mode.", "Use technical structure, verification steps, and concrete details."],
    },
    "debugger": {
        "trait_targets": {"warmth": 0.56, "sarcasm": 0.02, "strictness": 0.90, "verbosity": 0.58, "empathy": 0.56, "teasing": 0.00},
        "dialog_targets": {"warmth": 0.48, "sarcasm": 0.03, "strictness": 0.92, "verbosity": 0.48},
        "prompt_lines": ["Debugger mode.", "Start with diagnostics, then hypotheses, then minimal fixes."],
    },
    "planner": {
        "trait_targets": {"warmth": 0.58, "sarcasm": 0.04, "strictness": 0.84, "verbosity": 0.52, "empathy": 0.56, "teasing": 0.02},
        "dialog_targets": {"warmth": 0.52, "sarcasm": 0.06, "strictness": 0.86, "verbosity": 0.50},
        "prompt_lines": ["Planner mode.", "Respond with phased plans and explicit acceptance criteria."],
    },
    "spicy_chat": {
        "trait_targets": {"warmth": 0.70, "sarcasm": 0.42, "strictness": 0.26, "verbosity": 0.68, "empathy": 0.50, "teasing": 0.74},
        "dialog_targets": {"warmth": 0.70, "sarcasm": 0.44, "strictness": 0.28, "verbosity": 0.66},
        "prompt_lines": ["Spicy chat mode.", "Use a bold playful tone while respecting safety boundaries."],
    },
}

_SEMANTIC_MAP: list[tuple[str, tuple[str, ...]]] = [
    ("helper", ("help", "assist", "support", "pom", "assistant", "пом", "ассист")),
    ("engineer", ("code", "coding", "dev", "engineer", "код", "програм")),
    ("debugger", ("debug", "bug", "trace", "log", "ошиб", "диаг")),
    ("planner", ("plan", "roadmap", "strategy", "план")),
    ("spicy_chat", ("spicy", "naughty", "vulg", "flirt", "hot", "sexy", "пошл", "вульг", "флирт")),
]


def resolve_mode_profile(
    *,
    mode: str,
    character_id: str = "",
    persona_spec: dict[str, Any] | None = None,
) -> ModeProfile:
    mode_id = normalize_mode_name(mode, allow_custom=True)
    spec = dict(persona_spec or {})
    if not spec and str(character_id or "").strip():
        spec = load_character_spec(str(character_id or ""), "persona_spec", required=False)
    spec = dict(spec or {})

    modes_spec = load_spec("modes", required=False)
    mode_entry = dict(dict(modes_spec.get("modes") or {}).get(mode_id) or {})
    profile_id = _resolve_profile_id(mode_id=mode_id, mode_entry=mode_entry)
    preset = dict(_BUILTIN_PROFILES.get(profile_id) or _BUILTIN_PROFILES["chatting"])

    trait_targets = _coerce_targets(preset.get("trait_targets"), keys=_TRAIT_KEYS)
    dialog_targets = _coerce_targets(preset.get("dialog_targets"), keys=_DIALOG_KEYS)
    prompt_lines = _coerce_lines(preset.get("prompt_lines"))
    source = "builtin_preset" if mode_id in _BUILTIN_PROFILES else "semantic_preset"

    effects = dict(mode_entry.get("persona_effects") or {})
    if effects:
        trait_targets = _merge_targets(trait_targets, effects.get("trait_targets"), keys=_TRAIT_KEYS)
        dialog_targets = _merge_targets(dialog_targets, effects.get("dialog_targets"), keys=_DIALOG_KEYS)
        effect_lines = _coerce_lines(effects.get("prompt_lines"))
        if effect_lines:
            prompt_lines = effect_lines
        source = "modes_spec.persona_effects"

    persona_lines = _coerce_lines(dict(spec.get("modes") or {}).get(mode_id))
    if persona_lines:
        prompt_lines = persona_lines
        source = "persona_spec.override"

    if not prompt_lines:
        prompt_lines = [f"Active mode {mode_id}. Keep behavior consistent."]

    return ModeProfile(
        mode_id=mode_id,
        profile_id=profile_id,
        source=source,
        trait_targets=trait_targets,
        dialog_targets=dialog_targets,
        prompt_lines=prompt_lines,
    )


def blend_mode_traits(
    *,
    base_traits: dict[str, Any] | None,
    trait_targets: dict[str, Any] | None,
    alpha: float = MODE_BLEND_ALPHA,
) -> dict[str, float]:
    out = {str(k).strip().lower(): _clamp01(v) for k, v in dict(base_traits or {}).items() if str(k).strip()}
    a = _clamp01(alpha)
    for key in _TRAIT_KEYS:
        target = _to_float(dict(trait_targets or {}).get(key), None)
        if target is None:
            continue
        base = _to_float(out.get(key), 0.5)
        out[key] = _clamp01((base * (1.0 - a)) + (target * a))
    return out


def blend_mode_dialog_levels(
    *,
    base_levels: dict[str, Any] | None,
    dialog_targets: dict[str, Any] | None,
    alpha: float = MODE_BLEND_ALPHA,
) -> dict[str, float]:
    out = {str(k).strip().lower(): _clamp01(v) for k, v in dict(base_levels or {}).items() if str(k).strip()}
    a = _clamp01(alpha)
    for key in _DIALOG_KEYS:
        target = _to_float(dict(dialog_targets or {}).get(key), None)
        if target is None:
            continue
        base = _to_float(out.get(key), 0.5)
        out[key] = _clamp01((base * (1.0 - a)) + (target * a))
    return out


def _resolve_profile_id(*, mode_id: str, mode_entry: dict[str, Any]) -> str:
    if mode_id in _BUILTIN_PROFILES:
        return mode_id
    hint = " ".join(
        [
            str(mode_id or ""),
            str(mode_entry.get("description") or ""),
            str(mode_entry.get("legacy_mode") or ""),
        ]
    ).lower()
    for profile_id, tokens in _SEMANTIC_MAP:
        if any(token in hint for token in tokens):
            return profile_id
    return "chatting"


def _coerce_targets(value: Any, *, keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    src = dict(value or {}) if isinstance(value, dict) else {}
    for key in keys:
        if key not in src:
            continue
        score = _to_float(src.get(key), None)
        if score is None:
            continue
        out[key] = _clamp01(score)
    return out


def _merge_targets(base: dict[str, float], extra: Any, *, keys: tuple[str, ...]) -> dict[str, float]:
    out = dict(base or {})
    for key, value in _coerce_targets(extra, keys=keys).items():
        out[key] = value
    return out


def _coerce_lines(value: Any) -> list[str]:
    out: list[str] = []
    for raw in list(value or []):
        text = str(raw or "").strip()
        if text:
            out.append(text)
    return out


def _to_float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: Any) -> float:
    num = _to_float(value, 0.0)
    if num is None:
        return 0.0
    return max(0.0, min(1.0, float(num)))
