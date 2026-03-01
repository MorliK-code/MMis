from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.paths import DATA_DIR
from utils.datetime_local import now_local_iso, now_local_ts

DEFAULT_CHARACTER_ID = "default"
DEFAULT_CHARACTER_NAME = "Default"
RESERVED_CHARACTER_DIRS: set[str] = set()


def _safe_id(value: str) -> str:
    raw = str(value or "").strip().lower()
    out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
    return out or DEFAULT_CHARACTER_ID


class CharacterStorage:
    def __init__(self, root: str | Path | None = None):
        self.root = (Path(root).expanduser() if root is not None else (DATA_DIR / "characters")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def ensure_defaults(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.ensure_character_structure(DEFAULT_CHARACTER_ID)
        self.ensure_character_structure("asya")
        self.ensure_all_character_structures()
        self._sync_manifest_with_directories()

    def load_manifest(self) -> dict[str, Any]:
        payload = _read_json(self.manifest_path)
        if not isinstance(payload, dict):
            return {"version": 1, "active_character_id": DEFAULT_CHARACTER_ID, "characters": []}
        payload.setdefault("version", 1)
        payload.setdefault("active_character_id", DEFAULT_CHARACTER_ID)
        payload.setdefault("characters", [])
        return payload

    def save_manifest(self, payload: dict[str, Any]) -> None:
        item = dict(payload or {})
        item["updated_at"] = now_local_iso()
        _write_json(self.manifest_path, item)

    def list_character_ids(self, *, include_disabled: bool = False) -> list[str]:
        self._sync_manifest_with_directories()
        manifest = self.load_manifest()
        out: list[str] = []
        seen: set[str] = set()
        enabled_map: dict[str, bool] = {}
        for row in list(manifest.get("characters") or []):
            if not isinstance(row, dict):
                continue
            cid = _safe_id(row.get("id"))
            if cid in RESERVED_CHARACTER_DIRS:
                continue
            if not cid or cid in seen:
                continue
            enabled = bool(row.get("enabled", True))
            enabled_map[cid] = enabled
            if enabled or include_disabled:
                seen.add(cid)
                out.append(cid)
        for cid in self._discover_character_dirs():
            if not cid or cid in seen:
                continue
            explicitly_disabled = enabled_map.get(cid) is False
            if explicitly_disabled and not include_disabled:
                continue
            seen.add(cid)
            out.append(cid)
        if not out:
            out = [DEFAULT_CHARACTER_ID]
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

    def ensure_character_structure(self, character_id: str) -> None:
        cid = _safe_id(character_id)
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
                "id": cid,
                "name": _character_name(cid),
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
                "last_update_ts": "",
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
            f"You are {_character_name(cid)}. Keep responses practical, warm, and concise.\n"
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

    def ensure_all_character_structures(self) -> None:
        ids: set[str] = set()
        for cid in self.list_character_ids(include_disabled=True):
            if str(cid).startswith("_"):
                continue
            ids.add(_safe_id(cid))
        for row in self.root.iterdir():
            if row.is_dir() and not row.name.startswith("_"):
                ids.add(_safe_id(row.name))
        if not ids:
            ids.add(DEFAULT_CHARACTER_ID)
        for cid in sorted(ids):
            self.ensure_character_structure(cid)

    def load_character(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        self.ensure_character_structure(cid)
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
        self.ensure_character_structure(cid)
        payload = _read_json(self.character_dir(cid) / "state.json")
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("mood", "thoughtful")
        payload.setdefault("last_update_ts", "")
        payload.setdefault("active_traits", [])
        payload.setdefault("disabled_traits", [])
        payload.setdefault("counters", {"banter_hits": 0, "comfort_hits": 0})
        return payload

    def save_state(self, character_id: str, state: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        _write_json(self.character_dir(cid) / "state.json", dict(state or {}))

    def load_builtin_traits(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        self.ensure_character_structure(cid)
        payload = _read_json(self.character_dir(cid) / "traits" / "builtin.json")
        if not isinstance(payload, dict):
            return {}
        return dict(payload.get("traits") or {})

    def load_learned_traits(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        self.ensure_character_structure(cid)
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
        self.ensure_character_structure(cid)
        payload = _read_json(self.character_dir(cid) / "rules" / "evolution.json")
        if not isinstance(payload, dict):
            return {"rules": [], "cleanup": {"remove_if_confidence_below": 0.25, "remove_if_unused_days": 45}}
        payload.setdefault("rules", [])
        payload.setdefault("cleanup", {"remove_if_confidence_below": 0.25, "remove_if_unused_days": 45})
        return payload

    def read_prompt(self, character_id: str, rel_path: str) -> str:
        cid = _safe_id(character_id)
        self.ensure_character_structure(cid)
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
        row.setdefault("ts", now_local_ts())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def append_manifest_event(self, event: dict[str, Any]) -> None:
        path = self.root / "manifest_events.jsonl"
        row = dict(event or {})
        row.setdefault("ts", now_local_ts())
        row.setdefault("at", now_local_iso())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def sync_manifest(self) -> dict[str, Any]:
        self._sync_manifest_with_directories()
        return self.load_manifest()

    def _discover_character_dirs(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for row in sorted(self.root.iterdir(), key=lambda p: p.name.lower()):
            if not row.is_dir() or row.name.startswith("_"):
                continue
            if row.name.strip().lower() in RESERVED_CHARACTER_DIRS:
                continue
            cid = _safe_id(row.name)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
        return out

    def _sync_manifest_with_directories(self) -> None:
        manifest = self.load_manifest()
        existing_rows = [x for x in list(manifest.get("characters") or []) if isinstance(x, dict)]
        by_id: dict[str, dict[str, Any]] = {}
        old_ids: set[str] = set()
        for row in existing_rows:
            cid = _safe_id(row.get("id"))
            if not cid:
                continue
            if cid in RESERVED_CHARACTER_DIRS:
                continue
            clean = dict(row)
            clean["id"] = cid
            clean.setdefault("name", _character_name(cid))
            clean.setdefault("enabled", True)
            by_id[cid] = clean
            old_ids.add(cid)

        discovered = set(self._discover_character_dirs())
        discovered.add(DEFAULT_CHARACTER_ID)
        all_ids = sorted(discovered.union(set(by_id.keys())))

        chars: list[dict[str, Any]] = []
        for cid in all_ids:
            row = dict(by_id.get(cid) or {})
            row["id"] = cid
            row.setdefault("name", _character_name(cid))
            row.setdefault("enabled", True)
            chars.append(row)

        old_active = _safe_id(manifest.get("active_character_id"))
        active = old_active
        if active not in {str(x.get("id") or "").strip().lower() for x in chars}:
            active = DEFAULT_CHARACTER_ID

        manifest["version"] = int(manifest.get("version") or 1)
        manifest["active_character_id"] = active
        manifest["characters"] = chars
        new_ids = {str(x.get("id") or "").strip().lower() for x in chars if isinstance(x, dict)}
        changed = (
            old_ids != new_ids
            or old_active != active
            or len(existing_rows) != len(chars)
        )
        if changed:
            self.save_manifest(manifest)
            self.append_manifest_event(
                {
                    "type": "manifest_sync",
                    "active_before": old_active,
                    "active_after": active,
                    "added": sorted(new_ids - old_ids),
                    "removed": sorted(old_ids - new_ids),
                    "total": len(chars),
                }
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


def _character_name(cid: str) -> str:
    cleaned = str(cid or "").strip().replace("_", " ").replace("-", " ")
    if not cleaned:
        return DEFAULT_CHARACTER_NAME
    parts = [x for x in cleaned.split(" ") if x]
    if not parts:
        return DEFAULT_CHARACTER_NAME
    return " ".join(p[:1].upper() + p[1:] for p in parts)
