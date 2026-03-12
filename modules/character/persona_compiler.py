from __future__ import annotations

from typing import Any

from core.mode_selector import normalize_mode_name
from core.spec_registry import load_character_spec
from modules.character.mode_profile import MODE_BLEND_ALPHA, resolve_mode_profile
from modules.character.trait_policy import clamp_trait_map, clamp_trait_scalar

_CORE_STABLE_TRAITS = {
    "warmth",
    "professionalism",
    "empathy",
    "strictness",
    "sarcasm",
    "directness",
}


def compile_system_persona(
    character_id: str,
    persona_state: dict[str, Any] | None,
    active_mode: str,
    user_addressing: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    cid = str(character_id or "assistant").strip().lower() or "assistant"
    spec = load_character_spec(cid, "persona_spec", required=False)
    state = _sanitize_persona_state(_merge_state(
        base=load_character_spec(cid, "persona_state", required=False),
        current=dict(persona_state or {}),
    ))
    traits = dict(state.get("traits") or {})
    relation_state = dict(state.get("relation_state") or {})
    emotional_state = dict(state.get("emotional_state") or {})
    locks = dict(state.get("locks") or {})
    bans = [str(x).strip() for x in list(state.get("bans") or []) if str(x).strip()]
    mood = str(state.get("mood") or emotional_state.get("mood") or "neutral").strip().lower() or "neutral"
    mode = normalize_mode_name(active_mode, allow_custom=True)
    addressing = _coerce_user_addressing(user_addressing)

    identity_lines = _build_identity(spec=spec, character_id=cid, locks=locks)
    stable_trait_lines, stable_debug = _build_stable_traits_lines(spec=spec, traits=traits)
    relation_lines = _build_relation_state_lines(relation_state=relation_state)
    emotional_lines = _build_emotional_state_lines(emotional_state=emotional_state, mood=mood)
    addressing_lines = _build_addressing_lines(addressing=addressing)
    mode_lines, mode_debug = _build_mode_lines(spec=spec, mode=mode, character_id=cid)
    ban_lines = _build_ban_lines(spec=spec, bans=bans)

    blocks = [
        _render_block("PERSONA_IDENTITY", identity_lines),
        _render_block("PERSONA_STABLE_TRAITS", stable_trait_lines),
        _render_block("PERSONA_RELATION_STATE", relation_lines),
        _render_block("PERSONA_EMOTIONAL_STATE", emotional_lines),
        _render_block("PERSONA_ADDRESSING", addressing_lines),
        _render_block("PERSONA_MODE", mode_lines),
        _render_block("PERSONA_BANS", ban_lines),
    ]

    debug = {
        "active_mode": mode,
        "mood": mood,
        "traits": stable_debug,
        "relation_state": dict(relation_state),
        "emotional_state": dict(emotional_state),
        "locks": dict(locks),
        "bans": list(bans),
        "user_addressing": dict(addressing),
        "layers": {
            "identity": list(identity_lines),
            "stable_traits": list(stable_trait_lines),
            "relation_state": list(relation_lines),
            "emotional_state": list(emotional_lines),
            "addressing": list(addressing_lines),
            "mode": list(mode_lines),
            "bans": list(ban_lines),
        },
        "priority_order": [
            "identity",
            "stable_traits",
            "relation_state",
            "emotional_state",
            "addressing",
            "mode",
            "bans",
        ],
        "spec_version": int(spec.get("schema_version") or 1),
        "mode_profile_source": str(mode_debug.get("mode_profile_source") or ""),
        "mode_profile_id": str(mode_debug.get("mode_profile_id") or mode),
        "mode_blend_alpha": float(mode_debug.get("mode_blend_alpha") or MODE_BLEND_ALPHA),
    }
    return "\n\n".join(blocks).strip(), debug


def _merge_state(*, base: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in dict(current or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            child = dict(merged.get(key) or {})
            child.update(dict(value))
            merged[key] = child
            continue
        merged[key] = value
    merged.setdefault("traits", {})
    merged.setdefault("locks", {"feminine": True, "informal_you": True})
    merged.setdefault("relation_state", {})
    merged.setdefault("bans", [])
    merged.setdefault("mood", "neutral")
    return merged


def _sanitize_persona_state(value: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(value or {})
    traits = dict(clamp_trait_map(row.get("traits")))
    row["traits"] = traits
    row["locks"] = dict(row.get("locks") or {"feminine": True, "informal_you": True})
    row["relation_state"] = _coerce_relation_state(row.get("relation_state"), traits=traits)
    row["bans"] = [str(x).strip() for x in list(row.get("bans") or []) if str(x).strip()]
    row["emotional_state"] = _coerce_emotional_state(
        row.get("emotional_state"),
        fallback_mood=row.get("mood"),
    )
    mood = str(
        row.get("mood")
        or dict(row.get("emotional_state") or {}).get("mood")
        or "neutral"
    ).strip().lower() or "neutral"
    if mood == "romantic_soft":
        mood = "soft_supportive"
    row["mood"] = mood
    return row


def _build_identity(*, spec: dict[str, Any], character_id: str, locks: dict[str, Any]) -> list[str]:
    character_name = str(
        spec.get("character_name")
        or load_character_spec(character_id, "character", required=False).get("name")
        or character_id
    ).strip() or character_id
    templates = [str(x).strip() for x in list(spec.get("identity") or []) if str(x).strip()]
    if not templates:
        templates = [
            "You are {character_name}.",
            "Use Russian by default unless the user requests another language.",
            "Address the user informally unless explicitly requested otherwise.",
        ]
    lines = [
        _render_template(
            tpl,
            {
                "{character_name}": character_name,
                "{name}": character_name,
                "{character_id}": character_id,
            },
        )
        for tpl in templates
    ]
    lock_map = dict(spec.get("locks_map") or {})
    for key, template in lock_map.items():
        if bool(locks.get(str(key), False)):
            line = str(template or "").strip()
            if line:
                lines.append(line)
    return _dedupe_lines(lines)


def _build_stable_traits_lines(*, spec: dict[str, Any], traits: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    order = _stable_trait_order(spec=spec, traits=traits)
    selected: list[tuple[str, float]] = []
    for key in order:
        if key not in traits:
            continue
        value = _trait_value(traits, key, default=0.5)
        if key in _CORE_STABLE_TRAITS or abs(value - 0.5) >= 0.08:
            selected.append((key, value))
        if len(selected) >= 6:
            break
    if not selected:
        for key in order[:4]:
            if key in traits:
                selected.append((key, _trait_value(traits, key, default=0.5)))

    if not selected:
        return ["- baseline: balanced and practical, without theatrical swings."], {}

    lines = [f"- {key}: {_trait_descriptor(key, value)}" for key, value in selected]
    debug = {key: value for key, value in selected}
    return lines, debug


def _build_relation_state_lines(*, relation_state: dict[str, Any]) -> list[str]:
    row = _coerce_relation_state(relation_state, traits={})
    return [
        f"- familiarity: {_relation_descriptor('familiarity', row['familiarity'])}",
        f"- trust: {_relation_descriptor('trust', row['trust'])}",
        f"- teasing_permission: {_relation_descriptor('teasing_permission', row['teasing_permission'])}",
        f"- softness_bias: {_relation_descriptor('softness_bias', row['softness_bias'])}",
    ]


def _build_emotional_state_lines(*, emotional_state: dict[str, Any], mood: str) -> list[str]:
    row = _coerce_emotional_state(emotional_state, fallback_mood=mood)
    trigger = str(row.get("trigger") or "").strip().lower()
    trigger_text = trigger or "none"
    intensity_text = _axis_band(float(row.get("intensity", 0.0)))
    arousal_text = _axis_band(float(row.get("arousal", 0.0)))
    return [
        f"- current_mood: {str(row.get('mood') or mood or 'neutral')}",
        f"- trigger: {trigger_text}; intensity: {intensity_text}; arousal: {arousal_text}.",
        f"- temporary response bias: {_mood_response_bias(str(row.get('mood') or mood or 'neutral'))}",
        "- Treat this as short-term state only; do not rewrite identity or stable traits from it.",
    ]


def _build_addressing_lines(*, addressing: dict[str, Any]) -> list[str]:
    row = _coerce_user_addressing(addressing)
    canonical = str(row.get("canonical_name") or "").strip()
    allowed = [str(x).strip() for x in list(row.get("allowed_forms") or []) if str(x).strip()]
    forbidden = [str(x).strip() for x in list(row.get("forbidden_forms") or []) if str(x).strip()]

    lines = [
        "- invent_new_name_forms: forbidden",
        f"- canonical_name: {canonical or '<unset>'}",
        "- allowed_forms: " + (", ".join(allowed) if allowed else "<none>"),
        "- forbidden_forms: " + (", ".join(forbidden) if forbidden else "<none>"),
        f"- use_name_by_default: {'yes' if bool(row.get('use_name_by_default', False)) else 'no'}",
        f"- diminutives: {'allowed only if explicitly allowed' if bool(row.get('allow_diminutives', False)) else 'forbidden unless explicitly allowed'}",
    ]
    if not canonical:
        lines.append("- If no explicit name is configured, avoid using a name at all.")
    return lines


def _build_mode_lines(*, spec: dict[str, Any], mode: str, character_id: str) -> tuple[list[str], dict[str, Any]]:
    profile = resolve_mode_profile(mode=mode, character_id=character_id, persona_spec=spec)
    selected = _dedupe_lines([str(x).strip() for x in list(profile.prompt_lines or []) if str(x).strip()])
    selected = selected[:2]
    lines = [f"- active_mode: {mode}"]
    lines.extend(f"- {line}" for line in selected)
    lines.append("- This mode is task-local; it must not override identity, stable traits, relation state, or name policy.")
    return lines, {
        "mode_profile_source": profile.source,
        "mode_profile_id": profile.profile_id,
        "mode_blend_alpha": MODE_BLEND_ALPHA,
    }


def _build_ban_lines(*, spec: dict[str, Any], bans: list[str]) -> list[str]:
    clean = [str(x).strip() for x in list(bans or []) if str(x).strip()]
    if not clean:
        return ["- no explicit bans."]
    ban_template = str(spec.get("bans_template") or "Do not use banned word: {term}.").strip()
    rendered: list[str] = []
    for term in clean[:8]:
        rendered.append(
            "- " + _render_template(
                ban_template,
                {
                    "{term}": term,
                    "{word}": term,
                    "{ban-word}": term,
                    "{ban_word}": term,
                },
            )
        )
    return rendered


def _render_block(title: str, lines: list[str]) -> str:
    body = "\n".join([str(x).strip() for x in list(lines or []) if str(x).strip()] or ["- none"])
    return f"[{title}]\n{body}"


def _stable_trait_order(*, spec: dict[str, Any], traits: dict[str, Any]) -> list[str]:
    order = [_normalize_trait_key(x) for x in list(spec.get("trait_order") or []) if _normalize_trait_key(x)]
    if order:
        return order
    fallback = [
        "professionalism",
        "warmth",
        "empathy",
        "directness",
        "strictness",
        "sarcasm",
        "humor",
        "playfulness",
        "verbosity",
        "curiosity",
        "energy",
        "emoji_rate",
    ]
    seen = set(fallback)
    out = list(fallback)
    for key in dict(traits or {}).keys():
        name = _normalize_trait_key(key)
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _trait_descriptor(key: str, value: float) -> str:
    band = _level_band(value)
    if key == "warmth":
        return {
            "low": "reserved baseline.",
            "mid": "calm, neutral-warm baseline.",
            "high": "supportive and warm baseline.",
        }[band]
    if key == "professionalism":
        return {
            "low": "conversational, not formal by default.",
            "mid": "worklike and pragmatic.",
            "high": "precise, rigorous, and worklike.",
        }[band]
    if key == "empathy":
        return {
            "low": "emotionally restrained.",
            "mid": "attentive without overdoing comfort.",
            "high": "notices strain and responds supportively.",
        }[band]
    if key == "strictness":
        return {
            "low": "flexible, not procedural by default.",
            "mid": "structured without rigidity.",
            "high": "prefers strict structure and clear boundaries.",
        }[band]
    if key == "sarcasm":
        return {
            "low": "avoid sarcasm.",
            "mid": "light irony only when clearly safe.",
            "high": "sharp wit is available, but keep it safe.",
        }[band]
    if key == "directness":
        return {
            "low": "soft phrasing before direct pushes.",
            "mid": "balanced directness.",
            "high": "blunt and concise when needed.",
        }[band]
    if key in {"playfulness", "teasing"}:
        return {
            "low": "playfulness is restrained.",
            "mid": "lightly playful when appropriate.",
            "high": "playful by default, but still bounded.",
        }[band]
    if key == "humor":
        return {
            "low": "mostly serious tone.",
            "mid": "occasional light humor.",
            "high": "humor is available when it helps.",
        }[band]
    if key == "verbosity":
        return {
            "low": "prefers concise answers.",
            "mid": "balanced amount of detail.",
            "high": "comfortable with detailed explanations.",
        }[band]
    if key == "emoji_rate":
        return {
            "low": "avoid emoji.",
            "mid": "rare emoji only when clearly fitting.",
            "high": "emoji may appear sparingly when fitting.",
        }[band]
    if key == "curiosity":
        return {
            "low": "asks only critical questions.",
            "mid": "asks focused clarifying questions when needed.",
            "high": "proactively clarifies before committing.",
        }[band]
    if key == "energy":
        return {
            "low": "calm, low-tempo delivery.",
            "mid": "balanced tempo.",
            "high": "brisk, high-tempo delivery.",
        }[band]
    return f"{band} baseline."


def _relation_descriptor(key: str, value: float) -> str:
    band = _level_band(value)
    if key == "familiarity":
        return {
            "low": "limited familiarity; keep assumptions modest.",
            "mid": "working familiarity; concise direct dialogue is fine.",
            "high": "high familiarity; nuanced shorthand is acceptable.",
        }[band]
    if key == "trust":
        return {
            "low": "trust is limited; verify and avoid overreach.",
            "mid": "baseline working trust.",
            "high": "strong trust; collaborate directly without ceremonial hedging.",
        }[band]
    if key == "teasing_permission":
        return {
            "low": "avoid teasing unless the user clearly invites it.",
            "mid": "light teasing is allowed only when context stays safe.",
            "high": "playful teasing is allowed, but never over the line.",
        }[band]
    if key == "softness_bias":
        return {
            "low": "keep tone firmer than soothing.",
            "mid": "balanced between softness and directness.",
            "high": "default to calm, soft-supportive phrasing when tension appears.",
        }[band]
    return f"{band}."


def _mood_response_bias(mood: str) -> str:
    key = str(mood or "").strip().lower()
    if key == "focused":
        return "be calm, direct, and structured."
    if key == "thoughtful":
        return "be reflective, causal, and patient."
    if key == "soft_supportive":
        return "be gentle, containing, and non-pushy."
    if key == "teasing":
        return "be lightly playful only within relation limits."
    if key == "ironic":
        return "keep irony light and never hostile."
    return "rely on stable persona without extra emotional coloring."


def _coerce_relation_state(value: dict[str, Any] | None, *, traits: dict[str, Any]) -> dict[str, float]:
    warmth = _trait_value(traits, "warmth", default=0.58)
    empathy = _trait_value(traits, "empathy", default=0.62)
    teasing = _trait_value(
        traits,
        "teasing",
        default=_trait_value(traits, "playfulness", default=0.35),
    )
    defaults = {
        "familiarity": _clamp01(0.28 + max(0.0, warmth - 0.5) * 0.12 + max(0.0, empathy - 0.5) * 0.08),
        "trust": _clamp01(0.54 + max(0.0, empathy - 0.5) * 0.16),
        "teasing_permission": _clamp01(0.08 + max(0.0, teasing - 0.3) * 0.55),
        "softness_bias": _clamp01((warmth * 0.55) + (empathy * 0.45)),
    }
    row = dict(defaults)
    for key in ("familiarity", "trust", "teasing_permission", "softness_bias"):
        if isinstance(value, dict) and key in value:
            row[key] = _clamp01(_to_float(value.get(key), row[key]))
    return {k: float(v) for k, v in row.items()}


def _coerce_emotional_state(value: dict[str, Any] | None, *, fallback_mood: Any) -> dict[str, Any]:
    mood = str(fallback_mood or "neutral").strip().lower() or "neutral"
    row = dict(value or {})
    mood = str(row.get("mood") or mood).strip().lower() or "neutral"
    if mood == "romantic_soft":
        mood = "soft_supportive"
    trigger = str(row.get("trigger") or "").strip().lower()
    if trigger == "romantic_soft":
        trigger = "soft_supportive"
    return {
        "mood": mood,
        "trigger": trigger,
        "valence": float(max(-1.0, min(1.0, _to_float(row.get("valence"), 0.0)))),
        "arousal": float(_clamp01(_to_float(row.get("arousal"), 0.0))),
        "intensity": float(_clamp01(_to_float(row.get("intensity"), 0.0))),
        "last_update_ts": str(row.get("last_update_ts") or "").strip(),
    }


def _trait_value(traits: dict[str, Any], key: str, *, default: float) -> float:
    value = traits.get(str(key).strip().lower(), default)
    return float(clamp_trait_scalar(key, value, minimum=0.0, maximum=1.0))


def _level_band(value: float) -> str:
    current = float(max(0.0, min(1.0, value)))
    if current < 0.34:
        return "low"
    if current < 0.67:
        return "mid"
    return "high"


def _axis_band(value: float) -> str:
    current = float(max(0.0, min(1.0, value)))
    if current < 0.20:
        return "low"
    if current < 0.45:
        return "moderate"
    if current < 0.72:
        return "elevated"
    return "high"


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_trait_key(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})


def _dedupe_lines(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        item = str(value or "").strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _render_template(template: str, values: dict[str, str]) -> str:
    out = str(template or "")
    for token, value in dict(values or {}).items():
        out = out.replace(str(token), str(value))
    return out.strip()


def _coerce_user_addressing(value: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(value or {})
    return {
        "canonical_name": str(row.get("canonical_name") or "").strip(),
        "allowed_forms": [str(x).strip() for x in list(row.get("allowed_forms") or []) if str(x).strip()],
        "forbidden_forms": [str(x).strip() for x in list(row.get("forbidden_forms") or []) if str(x).strip()],
        "allow_diminutives": bool(row.get("allow_diminutives", False)),
        "use_name_by_default": bool(row.get("use_name_by_default", False)),
        "updated_at": str(row.get("updated_at") or "").strip(),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
