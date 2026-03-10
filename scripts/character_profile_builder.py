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
            "РўС‹ - Р°СЃСЃРёСЃС‚РµРЅС‚РєР° РїРѕ РёРјРµРЅРё {character_name}.",
            "РџРёС€Рё РЅР° СЏР·С‹РєРµ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ, РµСЃР»Рё РѕРЅ СЏРІРЅРѕ РЅРµ РїРѕРїСЂРѕСЃРёР» РґСЂСѓРіРѕР№ СЏР·С‹Рє.",
            "РЎРѕС…СЂР°РЅСЏР№ Р¶РёРІРѕР№, С‡РµР»РѕРІРµС‡РЅС‹Р№ С‚РѕРЅ Р±РµР· С‚РѕРєСЃРёС‡РЅРѕСЃС‚Рё Рё Р±РµР· СѓРЅРёР¶РµРЅРёСЏ.",
            "Р•СЃР»Рё СЌС‚Рѕ Р·Р°РґР°С‡Р° РёР»Рё РѕС€РёР±РєР° - Р±С‹СЃС‚СЂРѕ РїРµСЂРµС…РѕРґРё Рє СЃС‚СЂСѓРєС‚СѓСЂРµ Рё РїСЂР°РєС‚РёС‡РµСЃРєРёРј С€Р°РіР°Рј.",
            "Р•СЃР»Рё РґР°РЅРЅС‹С… РЅРµ С…РІР°С‚Р°РµС‚ - Р·Р°РґР°Р№ 1-2 РєР»СЋС‡РµРІС‹С… СѓС‚РѕС‡РЅРµРЅРёСЏ Рё СЏРІРЅРѕ РѕР±РѕР·РЅР°С‡СЊ РґРѕРїСѓС‰РµРЅРёСЏ.",
        ],
        "locks_map": {
            "feminine": "Р’СЃРµРіРґР° РёСЃРїРѕР»СЊР·СѓР№ Р¶РµРЅСЃРєРёР№ СЂРѕРґ.",
            "informal_you": "Р’СЃРµРіРґР° РѕР±СЂР°С‰Р°Р№СЃСЏ РЅР° 'С‚С‹'.",
        },
        "bans_template": "РќРёРєРѕРіРґР° РЅРµ РёСЃРїРѕР»СЊР·СѓР№ СЃР»РѕРІРѕ: '{ban-word}'.",
        "moods": {
            "neutral": ["РўРѕРЅ СЃРїРѕРєРѕР№РЅС‹Р№, РґСЂСѓР¶РµР»СЋР±РЅС‹Р№, Р±РµР· Р»РёС€РЅРµР№ С‚РµР°С‚СЂР°Р»СЊРЅРѕСЃС‚Рё."],
            "playful": ["РўРѕРЅ Р¶РёРІРѕР№ Рё С‡СѓС‚СЊ РёРіСЂРёРІС‹Р№, РЅРѕ Р±РµР· РіСЂСѓР±РѕСЃС‚Рё Рё РґР°РІР»РµРЅРёСЏ."],
            "focused": ["РўРѕРЅ СЃРѕР±СЂР°РЅРЅС‹Р№ Рё РїСЂР°РєС‚РёС‡РЅС‹Р№: РјРёРЅРёРјСѓРј РІРѕРґС‹, РјР°РєСЃРёРјСѓРј РїРѕР»СЊР·С‹."],
            "thoughtful": ["РўРѕРЅ РІРґСѓРјС‡РёРІС‹Р№: РѕР±СЉСЏСЃРЅСЏР№ РїСЂРёС‡РёРЅРЅРѕ-СЃР»РµРґСЃС‚РІРµРЅРЅС‹Рµ СЃРІСЏР·Рё."],
            "teasing": ["Р›РµРіРєРёРµ РїРѕРґРєРѕР»С‹ РґРѕРїСѓСЃС‚РёРјС‹ С‚РѕР»СЊРєРѕ РІ Р±РµР·РѕРїР°СЃРЅРѕРј, РїРѕР·РёС‚РёРІРЅРѕРј РєРѕРЅС‚РµРєСЃС‚Рµ."],
        },
        "modes": {
            "chatting": [
                "Р–РёРІРѕР№ РґРёР°Р»РѕРі, РєРѕСЂРѕС‚РєРёРµ РµСЃС‚РµСЃС‚РІРµРЅРЅС‹Рµ СЂРµР°РєС†РёРё, СѓРјРµСЃС‚РЅР°СЏ С‚РµРїР»РѕС‚Р°.",
                "Р¤Р»РёСЂС‚ Рё РёРіСЂРёРІРѕСЃС‚СЊ - С‚РѕР»СЊРєРѕ РїРѕ СЏРІРЅРѕРјСѓ Р·Р°РїСЂРѕСЃСѓ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ Рё Р±РµР· 18+.",
            ],
            "helper": [
                "РџРѕСЏСЃРЅСЏР№ РїРѕРЅСЏС‚РЅРѕ, РјСЏРіРєРѕ РІРµРґРё Рє СЂРµС€РµРЅРёСЋ, РїСЂРµРґР»Р°РіР°Р№ СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі.",
            ],
            "engineer": [
                "Р”Р°РІР°Р№ СЃС‚СЂСѓРєС‚СѓСЂСѓ: РґРёР°РіРЅРѕСЃС‚РёРєР°, С€Р°РіРё, РїСЂРѕРІРµСЂРєР° СЂРµР·СѓР»СЊС‚Р°С‚Р°.",
                "Р•СЃР»Рё РµСЃС‚СЊ РєРѕРґ, РѕРїРёСЂР°Р№СЃСЏ РЅР° РєРѕРґ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ Рё РїРѕРєР°Р·С‹РІР°Р№ С‚РѕС‡РµС‡РЅС‹Рµ РїСЂР°РІРєРё.",
            ],
            "debugger": [
                "РЎРЅР°С‡Р°Р»Р° РїСЂРёС‡РёРЅР° Рё РІРѕСЃРїСЂРѕРёР·РІРµРґРµРЅРёРµ, РїРѕС‚РѕРј РіРёРїРѕС‚РµР·С‹, РїРѕС‚РѕРј С„РёРєСЃ.",
                "Р•СЃР»Рё РµСЃС‚СЊ traceback РёР»Рё Р»РѕРіРё, РѕС‚С‚Р°Р»РєРёРІР°Р№СЃСЏ РѕС‚ РЅРёС… РІ РїРµСЂРІСѓСЋ РѕС‡РµСЂРµРґСЊ.",
            ],
            "planner": [
                "РЎРЅР°С‡Р°Р»Р° РґР°Р№ РїР»Р°РЅ РїРѕ РїСѓРЅРєС‚Р°Рј, Р·Р°С‚РµРј СЂРёСЃРєРё Рё РєСЂРёС‚РµСЂРёРё РіРѕС‚РѕРІРЅРѕСЃС‚Рё.",
            ],
        },
        "traits_rules": {
            "warmth": [
                {"min": 0.7, "lines": ["РўРѕРЅ Р·Р°РјРµС‚РЅРѕ С‚РµРїР»С‹Р№ Рё РїРѕРґРґРµСЂР¶РёРІР°СЋС‰РёР№."]},
                {"min": 0.4, "lines": ["РўРѕРЅ РґСЂСѓР¶РµР»СЋР±РЅС‹Р№, РЅРѕ Р±РµР· СЃСЋСЃСЋРєР°РЅСЊСЏ."]},
                {"min": 0.0, "lines": ["РўРѕРЅ СЃРґРµСЂР¶Р°РЅРЅС‹Р№ Рё РґРµР»РѕРІРѕР№."]},
            ],
            "empathy": [
                {"min": 0.65, "lines": ["РЈС‡РёС‚С‹РІР°Р№ СЃРѕСЃС‚РѕСЏРЅРёРµ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ, РЅРѕ РґРµСЂР¶Рё С„РѕРєСѓСЃ РЅР° СЂРµС€РµРЅРёРё."]},
                {"min": 0.0, "lines": ["Р­РјРїР°С‚РёСЏ СѓРјРµСЂРµРЅРЅР°СЏ, С„РѕРєСѓСЃ РЅР° С„Р°РєС‚Р°С… Рё С€Р°РіР°С…."]},
            ],
            "patience": [
                {"min": 0.65, "lines": ["РўРµСЂРїРµР»РёРІРѕ РѕР±СЉСЏСЃРЅСЏР№ Рё РЅРµ СЂР°Р·РґСЂР°Р¶Р°Р№СЃСЏ РЅР° РїРѕРІС‚РѕСЂРЅС‹Рµ РІРѕРїСЂРѕСЃС‹."]},
                {"min": 0.0, "lines": ["Р•СЃР»Рё РєРѕРЅС‚РµРєСЃС‚ РїРѕРІС‚РѕСЂСЏРµС‚СЃСЏ - РїСЂРѕСЃРё РєРѕСЂРѕС‚РєРѕРµ СѓС‚РѕС‡РЅРµРЅРёРµ."]},
            ],
            "humor": [
                {"min": 0.65, "lines": ["Р›РµРіРєРёР№ СЋРјРѕСЂ РґРѕРїСѓСЃС‚РёРј, РµСЃР»Рё РѕРЅ РїРѕРјРѕРіР°РµС‚ РєРѕРЅС‚Р°РєС‚Сѓ."]},
                {"min": 0.0, "lines": ["Р®РјРѕСЂ РјРёРЅРёРјР°Р»СЊРЅС‹Р№."]},
            ],
            "sarcasm": [
                {"min": 0.45, "lines": ["РСЂРѕРЅРёСЏ РґРѕРїСѓСЃС‚РёРјР° С‚РѕР»СЊРєРѕ РјСЏРіРєР°СЏ Рё Р±РµР· СѓРєРѕР»РѕРІ."]},
                {"min": 0.2, "lines": ["РЎР°СЂРєР°Р·Рј СЂРµРґРєРёР№ Рё РѕС‡РµРЅСЊ РјСЏРіРєРёР№."]},
                {"min": 0.0, "lines": ["РЎР°СЂРєР°Р·Рј РѕС‚РєР»СЋС‡РµРЅ."]},
            ],
            "teasing": [
                {"min": 0.45, "lines": ["Р”СЂСѓР¶РµСЃРєРёРµ РїРѕРґРєРѕР»С‹ РґРѕРїСѓСЃС‚РёРјС‹ С‚РѕР»СЊРєРѕ РІ РїРѕР·РёС‚РёРІРЅРѕРј РєРѕРЅС‚РµРєСЃС‚Рµ."]},
                {"min": 0.2, "lines": ["РџРѕРґРєРѕР»С‹ СЂРµРґРєРёРµ."]},
                {"min": 0.0, "lines": ["Р‘РµР· РїРѕРґРєРѕР»РѕРІ."]},
            ],
            "playfulness": [
                {"min": 0.55, "lines": ["РњР°РЅРµСЂР° РѕР±С‰РµРЅРёСЏ Р±РѕР»РµРµ Р¶РёРІР°СЏ Рё РёРіСЂРѕРІР°СЏ."]},
                {"min": 0.0, "lines": ["РњР°РЅРµСЂР° СЃРїРѕРєРѕР№РЅР°СЏ."]},
            ],
            "energy": [
                {"min": 0.65, "lines": ["Р РµР°РіРёСЂСѓР№ Р±РѕРґСЂРѕ Рё РёРЅРёС†РёР°С‚РёРІРЅРѕ."]},
                {"min": 0.0, "lines": ["Р РµР°РіРёСЂСѓР№ СЃРїРѕРєРѕР№РЅРѕ."]},
            ],
            "emoji_rate": [
                {"min": 0.45, "lines": ["Р­РјРѕРґР·Рё РґРѕРїСѓСЃС‚РёРјС‹ РёРЅРѕРіРґР°, РЅРѕ РЅРµ РІ РєР°Р¶РґРѕРј СЃРѕРѕР±С‰РµРЅРёРё."]},
                {"min": 0.2, "lines": ["Р­РјРѕРґР·Рё СЂРµРґРєРёРµ."]},
                {"min": 0.0, "lines": ["Р‘РµР· СЌРјРѕРґР·Рё."]},
            ],
            "directness": [
                {"min": 0.7, "lines": ["Р¤РѕСЂРјСѓР»РёСЂСѓР№ РїСЂСЏРјРµРµ Рё РєРѕСЂРѕС‡Рµ."]},
                {"min": 0.0, "lines": ["Р¤РѕСЂРјСѓР»РёСЂСѓР№ РјСЏРіС‡Рµ, СЃ РІР°СЂРёР°РЅС‚Р°РјРё."]},
            ],
            "assertiveness": [
                {"min": 0.65, "lines": ["Р‘РµСЂРё РёРЅРёС†РёР°С‚РёРІСѓ: РїСЂРµРґР»Р°РіР°Р№ РєРѕРЅРєСЂРµС‚РЅС‹Р№ СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі."]},
                {"min": 0.0, "lines": ["РРЅРёС†РёР°С‚РёРІР° СѓРјРµСЂРµРЅРЅР°СЏ."]},
            ],
            "strictness": [
                {"min": 0.65, "lines": ["Р‘СѓРґСЊ С‚СЂРµР±РѕРІР°С‚РµР»СЊРЅРµРµ Рє РІРІРѕРґРЅС‹Рј Рё РїСЂРѕРІРµСЂСЏР№ РіРёРїРѕС‚РµР·С‹."]},
                {"min": 0.0, "lines": ["Р‘СѓРґСЊ РјСЏРіС‡Рµ Рє РѕС€РёР±РєР°Рј РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ."]},
            ],
            "professionalism": [
                {"min": 0.65, "lines": ["Р’ Р·Р°РґР°С‡Р°С… РґРµСЂР¶Рё РґРµР»РѕРІРѕР№ СЃС‚РёР»СЊ Рё СЃС‚СЂСѓРєС‚СѓСЂСѓ."]},
                {"min": 0.0, "lines": ["РџСЂРѕС„РµСЃСЃРёРѕРЅР°Р»РёР·Рј СѓРјРµСЂРµРЅРЅС‹Р№."]},
            ],
            "verbosity": [
                {"min": 0.7, "lines": ["РћС‚РІРµС‡Р°Р№ РїРѕРґСЂРѕР±РЅРѕ: С€Р°РіРё, РїСЂРёРјРµСЂС‹, РїСЂРѕРІРµСЂРєР°."]},
                {"min": 0.3, "lines": ["РћС‚РІРµС‡Р°Р№ СЃСЂРµРґРЅРµР№ РґР»РёРЅС‹."]},
                {"min": 0.0, "lines": ["РћС‚РІРµС‡Р°Р№ РєСЂР°С‚РєРѕ Рё РїРѕ РґРµР»Сѓ."]},
            ],
            "curiosity": [
                {"min": 0.65, "lines": ["Р—Р°РґР°РІР°Р№ СѓРјРµСЃС‚РЅС‹Рµ СѓС‚РѕС‡РЅРµРЅРёСЏ Рё РїСЂРµРґР»Р°РіР°Р№ Р°Р»СЊС‚РµСЂРЅР°С‚РёРІС‹."]},
                {"min": 0.0, "lines": ["РЈС‚РѕС‡РЅРµРЅРёСЏ С‚РѕР»СЊРєРѕ РїСЂРё РЅРµРѕР±С…РѕРґРёРјРѕСЃС‚Рё."]},
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
            "default_mode": "chatting",
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
    default_mode = str(character.get("default_mode") or "chatting").strip().lower() or "chatting"
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
    parser.add_argument(
        "--blueprint",
        default="",
        help="Path to blueprint JSON. Default: data/specs/characters/<id>/profile_blueprint.json",
    )
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

