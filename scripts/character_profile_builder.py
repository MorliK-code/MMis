from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.character.storage import CharacterStorage
from utils.datetime_local import now_local_iso


PRESET_DEFAULT = "starter_plus"
PRESETS = {"starter_plus", "asya_like", "default"}


def _safe_id(value: str) -> str:
    raw = str(value or "").strip().lower()
    out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
    return out or "default"


def _display_name(character_id: str) -> str:
    cleaned = str(character_id or "").strip().replace("_", " ").replace("-", " ")
    parts = [x for x in cleaned.split(" ") if x]
    if not parts:
        return "Character"
    return " ".join(p[:1].upper() + p[1:] for p in parts)


def _default_traits_seed() -> dict[str, float]:
    return {
        "warmth": 0.62,
        "sarcasm": 0.2,
        "teasing": 0.26,
        "verbosity": 0.5,
        "strictness": 0.54,
        "empathy": 0.64,
        "professionalism": 0.6,
        "directness": 0.56,
        "patience": 0.62,
        "humor": 0.4,
        "emoji_rate": 0.14,
        "energy": 0.57,
        "curiosity": 0.58,
        "playfulness": 0.34,
        "assertiveness": 0.52,
    }


def _default_persona_spec_seed() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": [
            "Ты - ассистентка по имени {character_name}.",
            "Пиши на языке пользователя, если он явно не попросил другой язык.",
            "Сохраняй живой, человечный тон без токсичности и без унижения.",
            "Если это задача или ошибка - быстро переходи к структуре и практическим шагам.",
            "Если данных не хватает - задай 1-2 ключевых уточнения и явно обозначь допущения.",
        ],
        "locks_map": {
            "feminine": "Всегда используй женский род.",
            "informal_you": "Всегда обращайся на 'ты'.",
        },
        "bans_template": "Никогда не используй слово: '{ban-word}'.",
        "moods": {
            "neutral": ["Тон спокойный, дружелюбный, без лишней театральности."],
            "playful": ["Тон живой и чуть игривый, но без грубости и давления."],
            "focused": ["Тон собранный и практичный: минимум воды, максимум пользы."],
            "thoughtful": ["Тон вдумчивый: объясняй причинно-следственные связи."],
            "teasing": ["Легкие подколы допустимы только в безопасном, позитивном контексте."],
        },
        "modes": {
            "friend_chat": [
                "Живой диалог, короткие естественные реакции, уместная теплота.",
                "Флирт и игривость - только по явному запросу пользователя и без 18+.",
            ],
            "helper": [
                "Поясняй понятно, мягко веди к решению, предлагай следующий шаг.",
            ],
            "engineer": [
                "Давай структуру: диагностика, шаги, проверка результата.",
                "Если есть код, опирайся на код пользователя и показывай точечные правки.",
            ],
            "debugger": [
                "Сначала причина и воспроизведение, потом гипотезы, потом фикс.",
                "Если есть traceback или логи, отталкивайся от них в первую очередь.",
            ],
            "planner": [
                "Сначала дай план по пунктам, затем риски и критерии готовности.",
            ],
        },
        "traits_rules": {
            "warmth": [
                {"min": 0.7, "lines": ["Тон заметно теплый и поддерживающий."]},
                {"min": 0.4, "lines": ["Тон дружелюбный, но без сюсюканья."]},
                {"min": 0.0, "lines": ["Тон сдержанный и деловой."]},
            ],
            "empathy": [
                {"min": 0.65, "lines": ["Учитывай состояние пользователя, но держи фокус на решении."]},
                {"min": 0.0, "lines": ["Эмпатия умеренная, фокус на фактах и шагах."]},
            ],
            "patience": [
                {"min": 0.65, "lines": ["Терпеливо объясняй и не раздражайся на повторные вопросы."]},
                {"min": 0.0, "lines": ["Если контекст повторяется - проси короткое уточнение."]},
            ],
            "humor": [
                {"min": 0.65, "lines": ["Легкий юмор допустим, если он помогает контакту."]},
                {"min": 0.0, "lines": ["Юмор минимальный."]},
            ],
            "sarcasm": [
                {"min": 0.45, "lines": ["Ирония допустима только мягкая и без уколов."]},
                {"min": 0.2, "lines": ["Сарказм редкий и очень мягкий."]},
                {"min": 0.0, "lines": ["Сарказм отключен."]},
            ],
            "teasing": [
                {"min": 0.45, "lines": ["Дружеские подколы допустимы только в позитивном контексте."]},
                {"min": 0.2, "lines": ["Подколы редкие."]},
                {"min": 0.0, "lines": ["Без подколов."]},
            ],
            "playfulness": [
                {"min": 0.55, "lines": ["Манера общения более живая и игровая."]},
                {"min": 0.0, "lines": ["Манера спокойная."]},
            ],
            "energy": [
                {"min": 0.65, "lines": ["Реагируй бодро и инициативно."]},
                {"min": 0.0, "lines": ["Реагируй спокойно."]},
            ],
            "emoji_rate": [
                {"min": 0.45, "lines": ["Эмодзи допустимы иногда, но не в каждом сообщении."]},
                {"min": 0.2, "lines": ["Эмодзи редкие."]},
                {"min": 0.0, "lines": ["Без эмодзи."]},
            ],
            "directness": [
                {"min": 0.7, "lines": ["Формулируй прямее и короче."]},
                {"min": 0.0, "lines": ["Формулируй мягче, с вариантами."]},
            ],
            "assertiveness": [
                {"min": 0.65, "lines": ["Бери инициативу: предлагай конкретный следующий шаг."]},
                {"min": 0.0, "lines": ["Инициатива умеренная."]},
            ],
            "strictness": [
                {"min": 0.65, "lines": ["Будь требовательнее к вводным и проверяй гипотезы."]},
                {"min": 0.0, "lines": ["Будь мягче к ошибкам пользователя."]},
            ],
            "professionalism": [
                {"min": 0.65, "lines": ["В задачах держи деловой стиль и структуру."]},
                {"min": 0.0, "lines": ["Профессионализм умеренный."]},
            ],
            "verbosity": [
                {"min": 0.7, "lines": ["Отвечай подробно: шаги, примеры, проверка."]},
                {"min": 0.3, "lines": ["Отвечай средней длины."]},
                {"min": 0.0, "lines": ["Отвечай кратко и по делу."]},
            ],
            "curiosity": [
                {"min": 0.65, "lines": ["Задавай уместные уточнения и предлагай альтернативы."]},
                {"min": 0.0, "lines": ["Уточнения только при необходимости."]},
            ],
        },
    }


def _default_evolution_seed() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rules": [
            {
                "when": {"emotion": ["frustrated", "angry"]},
                "apply": [
                    {"trait": "patience", "op": "add", "value": 0.04},
                    {"trait": "warmth", "op": "add", "value": 0.02},
                    {"trait": "sarcasm", "op": "add", "value": -0.04},
                    {"trait": "teasing", "op": "add", "value": -0.04},
                    {"set_mood": "focused"},
                ],
            },
            {
                "when": {"intent": ["bug_report", "task", "code_review"]},
                "apply": [
                    {"set_mood": "focused"},
                    {"trait": "professionalism", "op": "add", "value": 0.03},
                    {"trait": "strictness", "op": "add", "value": 0.03},
                    {"trait": "directness", "op": "add", "value": 0.02},
                    {"trait": "emoji_rate", "op": "add", "value": -0.02},
                ],
            },
            {
                "when": {"intent": ["chat"]},
                "apply": [
                    {"set_mood": "playful"},
                    {"trait": "humor", "op": "add", "value": 0.02},
                    {"trait": "playfulness", "op": "add", "value": 0.03},
                ],
            },
            {
                "when": {"tags_any": ["needs_short_answer"]},
                "apply": [
                    {"trait": "verbosity", "op": "add", "value": -0.03},
                    {"trait": "directness", "op": "add", "value": 0.02},
                ],
            },
            {
                "when": {"tags_any": ["has_traceback", "has_stacktrace", "has_logs"]},
                "apply": [
                    {"set_mood": "focused"},
                    {"trait": "professionalism", "op": "add", "value": 0.03},
                    {"trait": "strictness", "op": "add", "value": 0.03},
                    {"trait": "sarcasm", "op": "add", "value": -0.02},
                ],
            },
        ],
        "cleanup": {
            "remove_if_confidence_below": 0.25,
            "remove_if_unused_days": 45,
        },
    }


def _starter_plus_blueprint(character_id: str) -> dict[str, Any]:
    name = _display_name(character_id)
    return {
        "schema_version": 1,
        "preset": "starter_plus",
        "profile": {
            "vibe": "balanced",
            "technicality": 0.6,
            "energy": 0.55,
        },
        "character": {
            "schema_version": 1,
            "character_id": character_id,
            "id": character_id,
            "name": name,
            "version": "1.0.0",
            "default_mood": "neutral",
            "default_mode": "friend_chat",
            "llm_profile": "BALANCED",
            "locks": {"feminine": True, "informal_you": True},
        },
        "persona_state": {
            "schema_version": 1,
            "character_id": character_id,
            "name": name,
            "mood": "neutral",
            "traits": _default_traits_seed(),
            "locks": {"feminine": True, "informal_you": True},
            "bans": [],
            "learned": {
                "preferences_confirmed": [],
                "preferences_pending": [],
                "style_bias": {},
            },
            "baseline_traits": {},
        },
        "persona_spec": _default_persona_spec_seed(),
        "evolution_spec": _default_evolution_seed(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload or {}), ensure_ascii=False, indent=2), encoding="utf-8")


def _clamp01(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _coerce_traits(value: Any, *, fallback: dict[str, Any] | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    src = dict(fallback or {})
    src.update(dict(value or {}))
    for key, raw in src.items():
        name = str(key or "").strip().lower()
        if not name:
            continue
        out[name] = _clamp01(raw, 0.5)
    return out


def _coerce_string_list(value: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in list(value or []):
        item = str(row or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _coerce_locks(value: Any) -> dict[str, bool]:
    row = dict(value or {})
    out = {str(k).strip(): bool(v) for k, v in row.items() if str(k).strip()}
    if "feminine" not in out:
        out["feminine"] = True
    if "informal_you" not in out:
        out["informal_you"] = True
    return out


def _coerce_learned(value: Any) -> dict[str, Any]:
    row = dict(value or {})
    out = {
        "preferences_confirmed": _coerce_string_list(row.get("preferences_confirmed")),
        "preferences_pending": _coerce_string_list(row.get("preferences_pending")),
        "style_bias": dict(row.get("style_bias") or {}),
    }
    if isinstance(row.get("baseline_traits"), dict):
        out["baseline_traits"] = _coerce_traits(row.get("baseline_traits"))
    return out


def _coerce_baseline_traits(value: Any) -> dict[str, float]:
    return _coerce_traits(value)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in dict(patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out.get(key) or {}), dict(value))
            continue
        out[key] = value
    return out


def _load_character_specs(storage: CharacterStorage, character_id: str) -> dict[str, dict[str, Any]]:
    spec_dir = storage.character_spec_dir(character_id)
    return {
        "character": _read_json(spec_dir / "character.json"),
        "persona_state": _read_json(spec_dir / "persona_state.json"),
        "persona_spec": _read_json(spec_dir / "persona_spec.json"),
        "evolution_spec": _read_json(spec_dir / "evolution_spec.json"),
    }


def _load_preset(storage: CharacterStorage, character_id: str, preset: str) -> dict[str, Any]:
    name = str(preset or PRESET_DEFAULT).strip().lower()
    if name == "starter_plus":
        return _starter_plus_blueprint(character_id)

    if name == "asya_like":
        base_id = "asya"
        storage.ensure_character_structure(base_id)
        src = _load_character_specs(storage, base_id)
        return {
            "schema_version": 1,
            "preset": "asya_like",
            "character": dict(src.get("character") or {}),
            "persona_state": dict(src.get("persona_state") or {}),
            "persona_spec": dict(src.get("persona_spec") or {}),
            "evolution_spec": dict(src.get("evolution_spec") or {}),
        }

    storage.ensure_character_structure(character_id)
    src = _load_character_specs(storage, character_id)
    return {
        "schema_version": 1,
        "preset": "default",
        "character": dict(src.get("character") or {}),
        "persona_state": dict(src.get("persona_state") or {}),
        "persona_spec": dict(src.get("persona_spec") or {}),
        "evolution_spec": dict(src.get("evolution_spec") or {}),
    }


def _normalize_blueprint(
    blueprint: dict[str, Any],
    *,
    character_id: str,
    name_override: str = "",
) -> dict[str, Any]:
    row = dict(blueprint or {})
    character = dict(row.get("character") or {})
    persona_state = dict(row.get("persona_state") or {})
    persona_spec = dict(row.get("persona_spec") or {})
    evolution_spec = dict(row.get("evolution_spec") or {})
    profile = dict(row.get("profile") or {})

    char_name = str(name_override or character.get("name") or persona_state.get("name") or _display_name(character_id)).strip()
    default_mood = str(character.get("default_mood") or persona_state.get("mood") or "neutral").strip().lower() or "neutral"
    default_mode = str(character.get("default_mode") or "friend_chat").strip().lower() or "friend_chat"
    llm_profile = str(character.get("llm_profile") or "BALANCED").strip().upper() or "BALANCED"

    locks = _coerce_locks(_pick_first(persona_state.get("locks"), character.get("locks"), {}))
    traits = _coerce_traits(persona_state.get("traits"), fallback=_default_traits_seed())
    traits = _apply_profile_knobs(traits, profile)
    learned = _coerce_learned(persona_state.get("learned"))
    baseline_traits = _coerce_baseline_traits(persona_state.get("baseline_traits"))
    bans = _coerce_string_list(persona_state.get("bans"))

    normalized_character = _deep_merge(
        {
            "schema_version": 1,
            "id": character_id,
            "character_id": character_id,
            "name": char_name,
            "version": str(character.get("version") or "1.0.0"),
            "default_mood": default_mood,
            "default_mode": default_mode,
            "llm_profile": llm_profile,
            "locks": dict(locks),
        },
        character,
    )
    normalized_character["schema_version"] = 1
    normalized_character["id"] = character_id
    normalized_character["character_id"] = character_id
    normalized_character["name"] = char_name
    normalized_character["default_mood"] = default_mood
    normalized_character["default_mode"] = default_mode
    normalized_character["llm_profile"] = llm_profile
    normalized_character["locks"] = dict(locks)

    normalized_state = _deep_merge(
        {
            "schema_version": 1,
            "character_id": character_id,
            "name": char_name,
            "mood": str(persona_state.get("mood") or default_mood).strip().lower() or default_mood,
            "traits": dict(traits),
            "locks": dict(locks),
            "bans": list(bans),
            "learned": dict(learned),
            "baseline_traits": dict(baseline_traits),
        },
        persona_state,
    )
    normalized_state["schema_version"] = 1
    normalized_state["character_id"] = character_id
    normalized_state["name"] = char_name
    normalized_state["mood"] = str(normalized_state.get("mood") or default_mood).strip().lower() or default_mood
    normalized_state["traits"] = dict(traits)
    normalized_state["locks"] = dict(locks)
    normalized_state["bans"] = list(bans)
    normalized_state["learned"] = dict(learned)
    normalized_state["baseline_traits"] = dict(baseline_traits)

    normalized_spec = _deep_merge(_default_persona_spec_seed(), persona_spec)
    normalized_spec["schema_version"] = int(normalized_spec.get("schema_version") or 1)

    normalized_evolution = _deep_merge(_default_evolution_seed(), evolution_spec)
    normalized_evolution["schema_version"] = int(normalized_evolution.get("schema_version") or 1)
    normalized_evolution["rules"] = list(normalized_evolution.get("rules") or [])
    cleanup = dict(normalized_evolution.get("cleanup") or {})
    cleanup.setdefault("remove_if_confidence_below", 0.25)
    cleanup.setdefault("remove_if_unused_days", 45)
    normalized_evolution["cleanup"] = cleanup

    return {
        "schema_version": int(row.get("schema_version") or 1),
        "preset": str(row.get("preset") or PRESET_DEFAULT),
        "updated_at": now_local_iso(),
        "profile": dict(profile),
        "character": normalized_character,
        "persona_state": normalized_state,
        "persona_spec": normalized_spec,
        "evolution_spec": normalized_evolution,
    }


def _apply_profile_knobs(traits: dict[str, float], profile: dict[str, Any]) -> dict[str, float]:
    out = dict(traits or {})
    row = dict(profile or {})
    vibe = str(row.get("vibe") or "").strip().lower()

    if vibe == "playful":
        _add_trait_delta(out, "humor", 0.08)
        _add_trait_delta(out, "playfulness", 0.1)
        _add_trait_delta(out, "teasing", 0.05)
        _add_trait_delta(out, "emoji_rate", 0.05)
    elif vibe == "strict":
        _add_trait_delta(out, "professionalism", 0.08)
        _add_trait_delta(out, "strictness", 0.08)
        _add_trait_delta(out, "directness", 0.06)
        _add_trait_delta(out, "humor", -0.06)
        _add_trait_delta(out, "emoji_rate", -0.06)
    elif vibe == "soft":
        _add_trait_delta(out, "warmth", 0.08)
        _add_trait_delta(out, "empathy", 0.08)
        _add_trait_delta(out, "patience", 0.06)
        _add_trait_delta(out, "strictness", -0.05)
        _add_trait_delta(out, "sarcasm", -0.05)

    technicality = _clamp01(row.get("technicality"), 0.5)
    tech_delta = (technicality - 0.5) * 0.22
    _add_trait_delta(out, "professionalism", tech_delta)
    _add_trait_delta(out, "strictness", tech_delta * 0.8)
    _add_trait_delta(out, "directness", tech_delta * 0.9)
    _add_trait_delta(out, "emoji_rate", -tech_delta * 0.8)

    energy = _clamp01(row.get("energy"), out.get("energy", 0.55))
    out["energy"] = _clamp01(energy, 0.55)
    return out


def _add_trait_delta(traits: dict[str, float], key: str, delta: float) -> None:
    name = str(key or "").strip().lower()
    if not name:
        return
    current = _clamp01(traits.get(name), 0.5)
    traits[name] = _clamp01(current + float(delta), current)


def _pick_first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        return value
    return None


def build_spec_payloads(blueprint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    row = dict(blueprint or {})
    return {
        "character.json": dict(row.get("character") or {}),
        "persona_state.json": dict(row.get("persona_state") or {}),
        "persona_spec.json": dict(row.get("persona_spec") or {}),
        "evolution_spec.json": dict(row.get("evolution_spec") or {}),
    }


def _set_active_character(storage: CharacterStorage, character_id: str) -> None:
    manifest = storage.load_manifest()
    prev = str(manifest.get("active_character_id") or "").strip().lower()
    manifest["active_character_id"] = character_id
    storage.save_manifest(manifest)
    storage.append_manifest_event(
        {
            "type": "active_character_changed",
            "from": prev,
            "to": character_id,
            "source": "character_profile_builder",
        }
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one character profile from a single blueprint JSON.")
    parser.add_argument("--id", required=True, help="Character id (folder name).")
    parser.add_argument("--name", default="", help="Display name override.")
    parser.add_argument("--preset", default=PRESET_DEFAULT, choices=sorted(PRESETS), help="Blueprint preset.")
    parser.add_argument("--blueprint", default="", help="Path to blueprint JSON. Default: data/specs/characters/<id>/profile_blueprint.json")
    parser.add_argument("--reset-blueprint", action="store_true", help="Recreate blueprint from preset even if it exists.")
    parser.add_argument("--set-active", action="store_true", help="Set created character as active in manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files.")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or [])
    character_id = _safe_id(args.id)
    name_override = str(args.name or "").strip()
    preset = str(args.preset or PRESET_DEFAULT).strip().lower()

    storage = CharacterStorage()
    storage.ensure_character_structure(character_id)
    spec_dir = storage.character_spec_dir(character_id)

    blueprint_path = Path(args.blueprint).expanduser() if str(args.blueprint or "").strip() else (spec_dir / "profile_blueprint.json")
    blueprint_path = blueprint_path.resolve()

    if args.reset_blueprint or not blueprint_path.exists():
        seed = _load_preset(storage, character_id, preset)
        seed = _normalize_blueprint(seed, character_id=character_id, name_override=name_override)
        if args.dry_run:
            print(f"[dry-run] create blueprint: {blueprint_path}")
        else:
            _write_json(blueprint_path, seed)
            print(f"[ok] blueprint: {blueprint_path}")
        blueprint = seed
    else:
        loaded = _read_json(blueprint_path)
        if not loaded:
            print(f"[error] invalid blueprint json: {blueprint_path}")
            return 1
        blueprint = _normalize_blueprint(loaded, character_id=character_id, name_override=name_override)
        if args.dry_run:
            print(f"[dry-run] normalize blueprint: {blueprint_path}")
        else:
            _write_json(blueprint_path, blueprint)
            print(f"[ok] blueprint normalized: {blueprint_path}")

    payloads = build_spec_payloads(blueprint)
    output_paths = {
        "character.json": spec_dir / "character.json",
        "persona_state.json": spec_dir / "persona_state.json",
        "persona_spec.json": spec_dir / "persona_spec.json",
        "evolution_spec.json": spec_dir / "evolution_spec.json",
    }

    for name, path in output_paths.items():
        if args.dry_run:
            print(f"[dry-run] write {name}: {path}")
            continue
        _write_json(path, payloads.get(name) or {})
        print(f"[ok] {name}: {path}")

    if not args.dry_run:
        storage.sync_manifest()
        if bool(args.set_active):
            _set_active_character(storage, character_id)
            print(f"[ok] active_character_id={character_id}")

    print(f"[done] character_id={character_id} preset={preset} dry_run={int(bool(args.dry_run))}")
    return 0


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
