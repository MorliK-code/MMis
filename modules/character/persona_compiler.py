from __future__ import annotations

from typing import Any

from core.mode_selector import normalize_mode_name
from core.spec_registry import load_character_spec


def compile_system_persona(
    character_id: str,
    persona_state: dict[str, Any] | None,
    active_mode: str,
) -> tuple[str, dict[str, Any]]:
    cid = str(character_id or "assistant").strip().lower() or "assistant"
    spec = load_character_spec(cid, "persona_spec", required=False)
    state = _merge_state(
        base=load_character_spec(cid, "persona_state", required=False),
        current=dict(persona_state or {}),
    )
    traits = dict(state.get("traits") or {})
    locks = dict(state.get("locks") or {})
    bans = [str(x).strip() for x in list(state.get("bans") or []) if str(x).strip()]
    mood = str(state.get("mood") or "neutral").strip().lower() or "neutral"
    mode = normalize_mode_name(active_mode)

    identity = _build_identity(spec=spec, character_id=cid, locks=locks)
    constraints = _build_mode_constraints(spec=spec, mode=mode)
    tone_lines, tone_debug = _build_tone_block(spec=spec, traits=traits)
    lock_lines = _build_lock_ban_block(spec=spec, locks=locks, bans=bans)
    mood_lines = _build_mood_lines(spec=spec, mood=mood)

    blocks: list[str] = []
    blocks.append("[PERSONA_IDENTITY]\n" + "\n".join(identity))
    if lock_lines:
        blocks.append("[PERSONA_LOCKS]\n" + "\n".join(lock_lines))
    blocks.append("[PERSONA_MODE]\n" + "\n".join(constraints))
    blocks.append("[PERSONA_TONE]\n" + "\n".join(tone_lines))
    if mood_lines:
        blocks.append("[PERSONA_MOOD]\n" + "\n".join(mood_lines))
    blocks.append(f"[PERSONA_STATE]\n- mood: {mood}\n- active_mode: {mode}")

    debug = {
        "active_mode": mode,
        "mood": mood,
        "traits": tone_debug,
        "locks": dict(locks),
        "bans": list(bans),
        "priority_order": ["locks", "mode", "persona", "mood"],
        "spec_version": int(spec.get("schema_version") or 1),
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
    merged.setdefault("bans", [])
    merged.setdefault("mood", "neutral")
    return merged


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
    return lines


def _build_mode_constraints(*, spec: dict[str, Any], mode: str) -> list[str]:
    modes = dict(spec.get("modes") or {})
    selected = [str(x).strip() for x in list(modes.get(mode) or []) if str(x).strip()]
    if not selected:
        selected = [str(x).strip() for x in list(modes.get("friend_chat") or []) if str(x).strip()]
    if not selected:
        selected = ["Keep friendly conversational tone."]
    selected.insert(0, f"Active mode: {mode}.")
    return selected


def _build_mood_lines(*, spec: dict[str, Any], mood: str) -> list[str]:
    moods = dict(spec.get("moods") or {})
    lines = [str(x).strip() for x in list(moods.get(mood) or []) if str(x).strip()]
    if lines:
        return lines
    fallback = [str(x).strip() for x in list(moods.get("neutral") or []) if str(x).strip()]
    return fallback


def _build_tone_block(*, spec: dict[str, Any], traits: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    rules = dict(spec.get("traits_rules") or {})
    order = [_normalize_trait_key(x) for x in list(spec.get("trait_order") or []) if _normalize_trait_key(x)]
    if not order:
        rule_keys = [_normalize_trait_key(x) for x in list(rules.keys()) if _normalize_trait_key(x)]
        trait_keys = [_normalize_trait_key(x) for x in list(dict(traits or {}).keys()) if _normalize_trait_key(x)]
        order = _merge_unique(rule_keys + trait_keys)
    if not order:
        order = ["warmth", "sarcasm", "verbosity", "strictness", "teasing", "empathy"]
    lines: list[str] = []
    debug: dict[str, float] = {}
    for key in order:
        value = _trait_value(traits, key, default=0.5)
        selected = _select_trait_lines(key=key, value=value, rules=rules)
        if selected:
            lines.extend(selected)
        debug[key] = value
    return lines, debug


def _select_trait_lines(*, key: str, value: float, rules: dict[str, Any]) -> list[str]:
    rows = [dict(x) for x in list(rules.get(key) or []) if isinstance(x, dict)]
    if not rows:
        return []
    rows.sort(key=lambda row: _to_float(row.get("min"), 0.0), reverse=True)
    for row in rows:
        if value >= _to_float(row.get("min"), 0.0):
            lines_payload = row.get("lines")
            if isinstance(lines_payload, list):
                out = [str(x).strip() for x in lines_payload if str(x).strip()]
                if out:
                    return out
            if isinstance(lines_payload, str):
                rendered = str(lines_payload).strip()
                if rendered:
                    return [rendered]
            line = str(row.get("line") or "").strip()
            if line:
                return [line]
            return []
    return []


def _build_lock_ban_block(*, spec: dict[str, Any], locks: dict[str, Any], bans: list[str]) -> list[str]:
    lines: list[str] = []
    lock_map = dict(spec.get("locks_map") or {})
    for key, template in lock_map.items():
        if bool(locks.get(str(key), False)):
            line = str(template or "").strip()
            if line:
                lines.append(f"Hard rule: {line}")
    ban_template = str(spec.get("bans_template") or "Do not use banned word: {term}.").strip()
    for term in bans:
        rendered = _render_template(
            ban_template,
            {
                "{term}": term,
                "{word}": term,
                "{ban-word}": term,
                "{ban_word}": term,
            },
        )
        lines.append(f"Hard rule: {rendered}")
    return lines


def _trait_value(traits: dict[str, Any], key: str, *, default: float) -> float:
    value = traits.get(str(key).strip().lower(), default)
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return float(default)


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


def _merge_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        key = _normalize_trait_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _render_template(template: str, values: dict[str, str]) -> str:
    out = str(template or "")
    for token, value in dict(values or {}).items():
        out = out.replace(str(token), str(value))
    return out.strip()
