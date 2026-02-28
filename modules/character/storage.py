from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config.paths import DATA_DIR


def _safe_id(value: str) -> str:
    raw = str(value or "").strip().lower()
    out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
    return out or "asya"


class CharacterStorage:
    def __init__(self, root: str | Path | None = None):
        self.root = (Path(root).expanduser() if root is not None else (DATA_DIR / "characters")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def ensure_defaults(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self.save_manifest(
                {
                    "version": 1,
                    "active_character_id": "asya",
                    "characters": [
                        {"id": "asya", "name": "Asya", "enabled": True},
                    ],
                    "updated_at": time.time(),
                }
            )
        else:
            manifest = self.load_manifest()
            chars = [x for x in list(manifest.get("characters") or []) if isinstance(x, dict)]
            if not any(str(x.get("id") or "").strip().lower() == "asya" for x in chars):
                chars.append({"id": "asya", "name": "Asya", "enabled": True})
            manifest["characters"] = chars
            manifest.setdefault("active_character_id", "asya")
            self.save_manifest(manifest)

        self._ensure_asya_tree()

    def load_manifest(self) -> dict[str, Any]:
        payload = _read_json(self.manifest_path)
        if not isinstance(payload, dict):
            return {"version": 1, "active_character_id": "asya", "characters": []}
        payload.setdefault("version", 1)
        payload.setdefault("active_character_id", "asya")
        payload.setdefault("characters", [])
        return payload

    def save_manifest(self, payload: dict[str, Any]) -> None:
        item = dict(payload or {})
        item["updated_at"] = float(item.get("updated_at") or time.time())
        _write_json(self.manifest_path, item)

    def list_character_ids(self, *, include_disabled: bool = False) -> list[str]:
        manifest = self.load_manifest()
        out: list[str] = []
        seen: set[str] = set()
        for row in list(manifest.get("characters") or []):
            if not isinstance(row, dict):
                continue
            cid = _safe_id(row.get("id"))
            if not cid or cid in seen:
                continue
            enabled = bool(row.get("enabled", True))
            if enabled or include_disabled:
                seen.add(cid)
                out.append(cid)
        if not out:
            out = [p.name for p in sorted(self.root.iterdir()) if p.is_dir()]
        return out

    def character_dir(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        path = (self.root / cid).resolve()
        try:
            path.relative_to(self.root)
        except Exception as exc:
            raise ValueError(f"character path escapes root: {path}") from exc
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_character(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        path = self.character_dir(cid) / "character.json"
        payload = _read_json(path)
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("id", cid)
        payload.setdefault("name", cid)
        payload.setdefault("version", "1.0.0")
        payload.setdefault("default_mood", "thoughtful")
        payload.setdefault("llm_profile", "BALANCED")
        payload.setdefault(
            "prompt_files",
            {
                "base": "prompts/base.txt",
                "moods_dir": "prompts/moods",
                "overlays_dir": "prompts/overlays",
            },
        )
        return payload

    def load_state(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        payload = _read_json(self.character_dir(cid) / "state.json")
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("mood", "thoughtful")
        payload.setdefault("last_update_ts", 0.0)
        payload.setdefault("active_traits", [])
        payload.setdefault("disabled_traits", [])
        payload.setdefault("counters", {"banter_hits": 0, "comfort_hits": 0})
        return payload

    def save_state(self, character_id: str, state: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        _write_json(self.character_dir(cid) / "state.json", dict(state or {}))

    def load_builtin_traits(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        payload = _read_json(self.character_dir(cid) / "traits" / "builtin.json")
        if not isinstance(payload, dict):
            return {}
        return dict(payload.get("traits") or {})

    def load_learned_traits(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        payload = _read_json(self.character_dir(cid) / "traits" / "learned.json")
        if not isinstance(payload, dict):
            return {}
        return dict(payload.get("traits") or {})

    def save_learned_traits(self, character_id: str, traits: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        path = self.character_dir(cid) / "traits" / "learned.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, {"traits": dict(traits or {})})

    def load_rules(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        payload = _read_json(self.character_dir(cid) / "rules" / "evolution.json")
        if not isinstance(payload, dict):
            return {"rules": [], "cleanup": {"remove_if_confidence_below": 0.25, "remove_if_unused_days": 45}}
        payload.setdefault("rules", [])
        payload.setdefault("cleanup", {"remove_if_confidence_below": 0.25, "remove_if_unused_days": 45})
        return payload

    def read_prompt(self, character_id: str, rel_path: str) -> str:
        cid = _safe_id(character_id)
        rel = str(rel_path or "").replace("\\", "/").strip().strip("/")
        if not rel:
            return ""
        root = self.character_dir(cid)
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except Exception:
            return ""
        if not path.exists() or not path.is_file():
            return ""
        try:
            return str(path.read_text(encoding="utf-8-sig")).strip()
        except Exception:
            return ""

    def append_event(self, character_id: str, event: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        root = self.character_dir(cid)
        path = root / "events.jsonl"
        row = dict(event or {})
        row.setdefault("ts", time.time())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _ensure_asya_tree(self) -> None:
        cid = "asya"
        root = self.character_dir(cid)
        for rel in [
            "traits",
            "rules",
            "prompts",
            "prompts/moods",
            "prompts/overlays",
            "prompts/traits",
        ]:
            (root / rel).mkdir(parents=True, exist_ok=True)

        _ensure_json(
            root / "character.json",
            {
                "id": "asya",
                "name": "Asya",
                "version": "1.0.0",
                "default_mood": "thoughtful",
                "llm_profile": "BALANCED",
                "prompt_files": {
                    "base": "prompts/base.txt",
                    "moods_dir": "prompts/moods",
                    "overlays_dir": "prompts/overlays",
                },
            },
        )
        _ensure_json(
            root / "state.json",
            {
                "mood": "thoughtful",
                "last_update_ts": 0.0,
                "active_traits": ["warmth", "thoughtfulness"],
                "disabled_traits": [],
                "counters": {"banter_hits": 0, "comfort_hits": 0},
            },
        )
        _ensure_json(
            root / "traits" / "builtin.json",
            {
                "traits": {
                    "sarcasm": {
                        "type": "scalar",
                        "value": 0.32,
                        "min": 0.0,
                        "max": 1.0,
                        "decay_per_day": 0.01,
                        "confidence": 0.9,
                        "tags": ["tone", "humor"],
                        "prompt_file": "prompts/traits/sarcasm.txt",
                    },
                    "warmth": {
                        "type": "scalar",
                        "value": 0.68,
                        "min": 0.0,
                        "max": 1.0,
                        "decay_per_day": 0.01,
                        "confidence": 0.9,
                        "tags": ["tone", "support"],
                        "prompt_file": "prompts/traits/warmth.txt",
                    },
                    "romance": {
                        "type": "scalar",
                        "value": 0.24,
                        "min": 0.0,
                        "max": 1.0,
                        "decay_per_day": 0.02,
                        "confidence": 0.6,
                        "tags": ["tone", "warm"],
                        "prompt_file": "prompts/traits/romance.txt",
                    },
                    "playfulness": {
                        "type": "scalar",
                        "value": 0.46,
                        "min": 0.0,
                        "max": 1.0,
                        "decay_per_day": 0.01,
                        "confidence": 0.7,
                        "tags": ["tone", "humor"],
                        "prompt_file": "prompts/traits/teasing.txt",
                    },
                    "thoughtfulness": {
                        "type": "scalar",
                        "value": 0.72,
                        "min": 0.0,
                        "max": 1.0,
                        "decay_per_day": 0.005,
                        "confidence": 0.85,
                        "tags": ["tone", "care"],
                        "prompt_file": "prompts/traits/warmth.txt",
                    },
                }
            },
        )
        _ensure_json(root / "traits" / "learned.json", {"traits": {}})
        _ensure_json(
            root / "rules" / "evolution.json",
            {
                "rules": [
                    {
                        "when": {"emotion": ["frustrated_angry", "sad_tired", "confused"]},
                        "apply": [
                            {"trait": "warmth", "op": "add", "value": 0.08},
                            {"trait": "sarcasm", "op": "add", "value": -0.1},
                            {"set_mood": "romantic_soft"},
                        ],
                    },
                    {
                        "when": {"intent": ["coding_help", "task_request", "ui_request"]},
                        "apply": [
                            {"set_mood": "focused"},
                            {"trait": "playfulness", "op": "add", "value": -0.08},
                        ],
                    },
                    {
                        "when": {"intent": ["chat"]},
                        "apply": [
                            {"trait": "playfulness", "op": "add", "value": 0.04},
                            {"set_mood": "teasing"},
                        ],
                    },
                ],
                "cleanup": {
                    "remove_if_confidence_below": 0.25,
                    "remove_if_unused_days": 45,
                },
            },
        )

        _ensure_text(
            root / "prompts" / "base.txt",
            "You are Asya. Keep responses practical, warm, and concise.\n"
            "Stay adaptive to user mood and intent.\n",
        )
        _ensure_text(root / "prompts" / "moods" / "ironic.txt", "Mood: ironic. Keep humor light and non-hostile.\n")
        _ensure_text(root / "prompts" / "moods" / "teasing.txt", "Mood: teasing. Playful tone, but keep it respectful.\n")
        _ensure_text(
            root / "prompts" / "moods" / "romantic_soft.txt",
            "Mood: romantic_soft. Use soft, supportive, caring tone.\n",
        )
        _ensure_text(
            root / "prompts" / "moods" / "focused.txt",
            "Mood: focused. Prioritize clarity, steps, and direct answers.\n",
        )
        _ensure_text(
            root / "prompts" / "moods" / "thoughtful.txt",
            "Mood: thoughtful. Show empathy and reasoning before conclusion.\n",
        )
        _ensure_text(
            root / "prompts" / "overlays" / "high_sarcasm.txt",
            "Overlay high_sarcasm: brief witty comments are allowed, avoid harsh sarcasm.\n",
        )
        _ensure_text(
            root / "prompts" / "overlays" / "high_warmth.txt",
            "Overlay high_warmth: be kind, reassuring, and emotionally supportive.\n",
        )
        _ensure_text(
            root / "prompts" / "traits" / "sarcasm.txt",
            "Trait sarcasm: use subtle irony, do not mock user mistakes.\n",
        )
        _ensure_text(
            root / "prompts" / "traits" / "warmth.txt",
            "Trait warmth: keep language soft, validating, and calm.\n",
        )
        _ensure_text(
            root / "prompts" / "traits" / "romance.txt",
            "Trait romance: use gentle affection, stay tasteful and safe.\n",
        )
        _ensure_text(
            root / "prompts" / "traits" / "teasing.txt",
            "Trait teasing: playful short banter, avoid crossing boundaries.\n",
        )


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_json(path: Path, payload) -> None:
    if path.exists():
        return
    _write_json(path, payload)


def _ensure_text(path: Path, text: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")

