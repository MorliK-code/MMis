from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modules.character.storage import CharacterStorage

<<<<<<< HEAD
_MOOD_MODIFIERS: dict[str, dict[str, float]] = {
    "focused": {"strictness": 0.20, "warmth": -0.08, "sarcasm": -0.10, "verbosity": -0.12},
    "thoughtful": {"warmth": 0.08, "verbosity": 0.06},
    "teasing": {"sarcasm": 0.14, "warmth": 0.02},
    "ironic": {"sarcasm": 0.10, "warmth": -0.04},
    "romantic_soft": {"warmth": 0.18, "sarcasm": -0.12},
}

=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04

@dataclass(frozen=True)
class CharacterComposeResult:
    prompt: str
    mood: str
    used_files: list[str] = field(default_factory=list)
    active_traits: list[str] = field(default_factory=list)
<<<<<<< HEAD
    effective_traits: dict[str, float] = field(default_factory=dict)
    style_coefficients: dict[str, float] = field(default_factory=dict)
=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04


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
<<<<<<< HEAD
        dialog_mode: dict[str, Any] | None = None,
        context_meta: dict[str, Any] | None = None,
=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
    ) -> CharacterComposeResult:
        prompt_files = dict(character.get("prompt_files") or {})
        base_rel = str(prompt_files.get("base") or "prompts/base.txt")
        base_text = storage.read_prompt(character_id, base_rel)
        used = [base_rel] if base_text else []

        mood = str(state.get("mood") or character.get("default_mood") or "thoughtful").strip().lower()
        mood_rel = f"{str(prompt_files.get('moods_dir') or 'prompts/moods').strip().strip('/')}/{mood}.txt"
        mood_text = storage.read_prompt(character_id, mood_rel)
        if mood_text:
            used.append(mood_rel)

        active = set(str(x).strip().lower() for x in list(state.get("active_traits") or []) if str(x).strip())
        disabled = set(str(x).strip().lower() for x in list(state.get("disabled_traits") or []) if str(x).strip())

<<<<<<< HEAD
        dm = dict(dialog_mode or {})
        meta = dict(context_meta or {})
        context_mods = compute_context_trait_modifiers(
            dialog_mode=dm,
            intent=str(meta.get("intent") or ""),
            is_technical=bool(meta.get("is_technical", False)),
        )
        overlay_mods = _overlay_modifiers(traits)
        style_coefficients = _resolve_style_coefficients(traits=traits, dialog_mode=dm, context_mods=context_mods, mood=mood, overlay_mods=overlay_mods)

        trait_blocks: list[tuple[float, str, str]] = []
        effective_traits: dict[str, float] = {}
=======
        trait_blocks: list[tuple[float, str, str]] = []
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
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
<<<<<<< HEAD
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
=======
                score = _to_float(value, 0.0)
                include = score >= self.trait_threshold
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
            if not include:
                continue
            rel = str(row.get("prompt_file") or "").strip().strip("/")
            if not rel:
                continue
            text = storage.read_prompt(character_id, rel)
            if not text:
                continue
            trait_blocks.append((score, trait_name, text))
            used.append(rel)

        trait_blocks.sort(key=lambda x: x[0], reverse=True)
        trait_blocks = trait_blocks[: self.max_trait_overlays]

<<<<<<< HEAD
=======
        overlays: list[str] = []
        overlays_dir = str(prompt_files.get("overlays_dir") or "prompts/overlays").strip().strip("/")
        sarcasm = _trait_scalar(traits, "sarcasm")
        warmth = _trait_scalar(traits, "warmth")
        if sarcasm >= 0.72:
            rel = f"{overlays_dir}/high_sarcasm.txt"
            text = storage.read_prompt(character_id, rel)
            if text:
                overlays.append(text)
                used.append(rel)
        if warmth >= 0.72:
            rel = f"{overlays_dir}/high_warmth.txt"
            text = storage.read_prompt(character_id, rel)
            if text:
                overlays.append(text)
                used.append(rel)

>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
        blocks: list[str] = []
        if base_text:
            blocks.append(f"[CHAR_BASE]\n{base_text}")
        blocks.append(f"[CHAR_META]\ncharacter={character_id}\nmood={mood}")
<<<<<<< HEAD
        blocks.append(
            "[CHAR_DIALOG_MODE]\n"
            f"greeting_allowed={str(bool(dm.get('greeting_allowed', dm.get('allow_greeting', False)))).lower()}\n"
            f"smalltalk_allowed={str(bool(dm.get('smalltalk_allowed', True))).lower()}"
        )
        blocks.append(
            "[CHAR_STYLE_COEFFICIENTS]\n"
            f"warmth={style_coefficients['warmth']:.3f}\n"
            f"sarcasm={style_coefficients['sarcasm']:.3f}\n"
            f"strictness={style_coefficients['strictness']:.3f}\n"
            f"verbosity={style_coefficients['verbosity']:.3f}"
        )
=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
        if mood_text:
            blocks.append(f"[CHAR_MOOD]\n{mood_text}")
        if trait_blocks:
            text = "\n\n".join([row[2] for row in trait_blocks if row[2]])
            blocks.append(f"[CHAR_TRAITS]\n{text}")
<<<<<<< HEAD
        if not base_text and not mood_text and not trait_blocks:
=======
        if overlays:
            overlays_text = "\n\n".join(overlays)
            blocks.append(f"[CHAR_OVERLAYS]\n{overlays_text}")
        if not base_text and not mood_text and not trait_blocks and not overlays:
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
            blocks.append("[CHAR_FALLBACK]\nKeep responses adaptive, warm, and concise.")

        active_traits = [row[1] for row in trait_blocks]
        prompt = "\n\n".join([x.strip() for x in blocks if str(x).strip()]).strip()
<<<<<<< HEAD
        return CharacterComposeResult(
            prompt=prompt,
            mood=mood,
            used_files=used,
            active_traits=active_traits,
            effective_traits=effective_traits,
            style_coefficients=style_coefficients,
        )
=======
        return CharacterComposeResult(prompt=prompt, mood=mood, used_files=used, active_traits=active_traits)
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


<<<<<<< HEAD
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
    if intent_key in {"coding_help", "coding", "task_request", "implementation", "debug"}:
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


=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
def _trait_scalar(traits: dict[str, Any], key: str) -> float:
    row = dict(traits.get(str(key).strip().lower()) or {})
    value = row.get("value")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _to_float(value, 0.0)
<<<<<<< HEAD


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
        elif mood == "romantic_soft" and key == "warmth":
            mood_modifier = 0.15
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
=======
>>>>>>> 51b8456b510b4061cb471a6e7b7574d205e99e04
