from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modules.character.storage import CharacterStorage


@dataclass(frozen=True)
class CharacterComposeResult:
    prompt: str
    mood: str
    used_files: list[str] = field(default_factory=list)
    active_traits: list[str] = field(default_factory=list)


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

        trait_blocks: list[tuple[float, str, str]] = []
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
                score = _to_float(value, 0.0)
                include = score >= self.trait_threshold
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

        blocks: list[str] = []
        if base_text:
            blocks.append(f"[CHAR_BASE]\n{base_text}")
        blocks.append(f"[CHAR_META]\ncharacter={character_id}\nmood={mood}")
        if mood_text:
            blocks.append(f"[CHAR_MOOD]\n{mood_text}")
        if trait_blocks:
            text = "\n\n".join([row[2] for row in trait_blocks if row[2]])
            blocks.append(f"[CHAR_TRAITS]\n{text}")
        if overlays:
            overlays_text = "\n\n".join(overlays)
            blocks.append(f"[CHAR_OVERLAYS]\n{overlays_text}")
        if not base_text and not mood_text and not trait_blocks and not overlays:
            blocks.append("[CHAR_FALLBACK]\nKeep responses adaptive, warm, and concise.")

        active_traits = [row[1] for row in trait_blocks]
        prompt = "\n\n".join([x.strip() for x in blocks if str(x).strip()]).strip()
        return CharacterComposeResult(prompt=prompt, mood=mood, used_files=used, active_traits=active_traits)


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _trait_scalar(traits: dict[str, Any], key: str) -> float:
    row = dict(traits.get(str(key).strip().lower()) or {})
    value = row.get("value")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _to_float(value, 0.0)
