from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR
from config.settings import load_config
from utils.datetime_local import now_local_iso, now_local_ts

DEFAULT_CHARACTER_ID = "default"
DEFAULT_CHARACTER_NAME = "Default"
RESERVED_CHARACTER_DIRS: set[str] = set()


def _safe_id(value: str) -> str:
    raw = str(value or "").strip().lower()
    out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
    return out or DEFAULT_CHARACTER_ID


class CharacterStorage:
    def __init__(self, root: str | Path | None = None, logs_root: str | Path | None = None):
        cfg = load_config()
        default_root = Path(cfg.memory_dir) / "characters_runtime"
        requested_root = (Path(root).expanduser() if root is not None else default_root).resolve()
        legacy_root = (DATA_DIR / "characters").resolve()
        # Hard guard: runtime character storage must not use legacy data/characters root.
        self.root = default_root.resolve() if requested_root == legacy_root else requested_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_root = (Path(logs_root).expanduser() if logs_root is not None else Path(cfg.log_dir)).resolve()
        self.character_logs_root = (self.logs_root / "characters").resolve()
        self.character_logs_root.mkdir(parents=True, exist_ok=True)
        self.spec_root = (DATA_DIR / "specs" / "characters").resolve()
        self.spec_root.mkdir(parents=True, exist_ok=True)
        self.legacy_spec_root = (DATA_DIR / "specs" / "rules_for_all" / "characters").resolve()
        self._migrate_legacy_character_specs()

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def character_dir(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        path = (self.root / cid).resolve()
        try:
            path.relative_to(self.root)
        except Exception as exc:
            raise ValueError(f"character path escapes root: {path}") from exc
        path.mkdir(parents=True, exist_ok=True)
        return path

    def character_spec_dir(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        path = (self.spec_root / cid).resolve()
        try:
            path.relative_to(self.spec_root)
        except Exception as exc:
            raise ValueError(f"character spec path escapes root: {path}") from exc
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_defaults(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec_root.mkdir(parents=True, exist_ok=True)
        self.ensure_character_structure(DEFAULT_CHARACTER_ID)
        self.ensure_character_structure("asya")
        self.ensure_all_character_structures()
        self._sync_manifest_with_directories()

    def _migrate_legacy_character_specs(self) -> None:
        src_root = self.legacy_spec_root
        dst_root = self.spec_root
        if not src_root.exists() or src_root == dst_root:
            return
        try:
            for row in src_root.iterdir():
                if not row.is_dir():
                    continue
                if row.name.startswith("_"):
                    continue
                src_dir = row.resolve()
                dst_dir = (dst_root / row.name).resolve()
                dst_dir.mkdir(parents=True, exist_ok=True)
                for item in src_dir.iterdir():
                    if item.is_file():
                        target = dst_dir / item.name
                        if not target.exists():
                            try:
                                shutil.copy2(item, target)
                            except Exception:
                                continue
        except Exception:
            return

    def ensure_character_structure(self, character_id: str) -> None:
        cid = _safe_id(character_id)
        root = self.character_dir(cid)
        for rel in ["traits", "rules"]:
            (root / rel).mkdir(parents=True, exist_ok=True)

        _ensure_json(
            root / "character.json",
            {
                "id": cid,
                "name": _character_name(cid),
                "version": "1.0.0",
                "default_mood": "thoughtful",
                "llm_profile": "BALANCED",
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
                    },
                    "warmth": {
                        "type": "scalar",
                        "value": 0.68,
                        "min": 0.0,
                        "max": 1.0,
                        "decay_per_day": 0.01,
                        "confidence": 0.9,
                        "tags": ["tone", "support"],
                    },
                    "romance": {
                        "type": "scalar",
                        "value": 0.24,
                        "min": 0.0,
                        "max": 1.0,
                        "decay_per_day": 0.02,
                        "confidence": 0.6,
                        "tags": ["tone", "warm"],
                    },
                    "playfulness": {
                        "type": "scalar",
                        "value": 0.46,
                        "min": 0.0,
                        "max": 1.0,
                        "decay_per_day": 0.01,
                        "confidence": 0.7,
                        "tags": ["tone", "humor"],
                    },
                    "thoughtfulness": {
                        "type": "scalar",
                        "value": 0.72,
                        "min": 0.0,
                        "max": 1.0,
                        "decay_per_day": 0.005,
                        "confidence": 0.85,
                        "tags": ["tone", "care"],
                    },
                }
            },
        )
        _ensure_json(root / "traits" / "learned.json", {"traits": {}})
        _ensure_json(
            root / "rules" / "evolution.json",
            _default_evolution_rules(),
        )
        self.ensure_character_specs(cid)

    def ensure_character_specs(self, character_id: str) -> None:
        cid = _safe_id(character_id)
        root = self.character_spec_dir(cid)
        _ensure_json(
            root / "character.json",
            {
                "schema_version": 1,
                "id": cid,
                "name": _character_name(cid),
                "version": "1.0.0",
                "default_mood": "thoughtful",
                "llm_profile": "BALANCED",
            },
        )
        _ensure_json(root / "persona_state.json", _default_persona_state())
        _ensure_json(root / "persona_spec.json", _default_persona_spec())
        _ensure_json(root / "evolution_spec.json", _default_evolution_rules())

    def ensure_all_character_structures(self) -> None:
        ids: set[str] = set()
        for cid in self.list_character_ids(include_disabled=True):
            if str(cid).startswith("_"):
                continue
            ids.add(_safe_id(cid))
        for row in self.root.iterdir():
            if row.is_dir() and not row.name.startswith("_"):
                ids.add(_safe_id(row.name))
        for row in self.spec_root.iterdir():
            if row.is_dir() and not row.name.startswith("_"):
                ids.add(_safe_id(row.name))
        if not ids:
            ids.add(DEFAULT_CHARACTER_ID)
        for cid in sorted(ids):
            self.ensure_character_structure(cid)

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
        for cid in self._discover_character_spec_dirs():
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
        if not out:
            out = [DEFAULT_CHARACTER_ID]
        return out

    def load_character(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        self.ensure_character_structure(cid)
        payload = self._sync_runtime_character_from_specs(cid)
        if not isinstance(payload, dict):
            payload = {}
        payload["id"] = _safe_id(payload.get("id") or payload.get("character_id") or cid)
        payload["character_id"] = str(payload.get("character_id") or payload.get("id") or cid)
        payload.setdefault("name", _character_name(cid))
        payload.setdefault("version", "1.0.0")
        payload.setdefault("default_mood", "thoughtful")
        llm_profile = str(payload.get("llm_profile") or payload.get("model_profile") or "BALANCED").strip().upper() or "BALANCED"
        payload["llm_profile"] = llm_profile
        payload["model_profile"] = str(payload.get("model_profile") or llm_profile).strip().upper() or llm_profile
        return payload

    def _sync_runtime_character_from_specs(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        runtime_path = self.character_dir(cid) / "character.json"
        spec_path = self.character_spec_dir(cid) / "character.json"
        runtime_payload = _read_json(runtime_path)
        if not isinstance(runtime_payload, dict):
            runtime_payload = {}
        spec_payload = _read_json(spec_path)
        if not isinstance(spec_payload, dict):
            spec_payload = {}
        spec_normalized = self._normalize_character_payload(spec_payload, fallback_id=cid)

        merged = dict(runtime_payload)
        if spec_normalized:
            # specs/characters is source-of-truth for character settings.
            merged.update(spec_normalized)
        merged = self._normalize_character_payload(merged, fallback_id=cid)

        if not runtime_payload or merged != runtime_payload:
            _write_json(runtime_path, merged)
        return merged

    @staticmethod
    def _normalize_character_payload(payload: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
        row = dict(payload or {})
        cid = _safe_id(row.get("id") or row.get("character_id") or fallback_id)
        name = str(row.get("name") or row.get("display_name") or _character_name(cid)).strip() or _character_name(cid)
        version = str(row.get("version") or "1.0.0").strip() or "1.0.0"
        default_mood = str(row.get("default_mood") or "thoughtful").strip() or "thoughtful"
        llm_profile = str(row.get("llm_profile") or row.get("model_profile") or "BALANCED").strip().upper() or "BALANCED"

        out = dict(row)
        out["id"] = cid
        out["character_id"] = str(row.get("character_id") or cid)
        out["name"] = name
        out["version"] = version
        out["default_mood"] = default_mood
        out["llm_profile"] = llm_profile
        out["model_profile"] = str(row.get("model_profile") or llm_profile).strip().upper() or llm_profile

        if "default_mode" in row:
            mode = str(row.get("default_mode") or "").strip().lower()
            if mode:
                out["default_mode"] = mode
        locks = row.get("locks")
        if isinstance(locks, dict):
            out["locks"] = dict(locks)
        return out

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
        self.ensure_character_specs(cid)
        spec_payload = _read_json(self.character_spec_dir(cid) / "evolution_spec.json")
        payload = spec_payload if isinstance(spec_payload, dict) else _read_json(self.character_dir(cid) / "rules" / "evolution.json")
        if not isinstance(payload, dict):
            return {"rules": [], "cleanup": {"remove_if_confidence_below": 0.25, "remove_if_unused_days": 45}}
        payload.setdefault("rules", [])
        payload.setdefault("cleanup", {"remove_if_confidence_below": 0.25, "remove_if_unused_days": 45})
        return payload

    def load_persona_state(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        self.ensure_character_specs(cid)
        payload = _read_json(self.character_spec_dir(cid) / "persona_state.json")
        return _normalize_persona_state_payload(payload if isinstance(payload, dict) else {})

    def save_persona_state(self, character_id: str, payload: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        self.ensure_character_specs(cid)
        row = self.load_persona_state(cid)
        row.update(dict(payload or {}))
        _write_json(self.character_spec_dir(cid) / "persona_state.json", _normalize_persona_state_payload(row))

    def load_persona_spec(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        self.ensure_character_specs(cid)
        payload = _read_json(self.character_spec_dir(cid) / "persona_spec.json")
        if not isinstance(payload, dict):
            payload = {}
        base = _default_persona_spec()
        merged = dict(base)
        merged.update(payload)
        return merged

    def read_prompt(self, character_id: str, rel_path: str) -> str:
        _ = (character_id, rel_path)
        raise ValueError("TXT prompts are disabled in runtime. Use JSON specs under data/specs/rules_for_all.")

    def append_event(self, character_id: str, event: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        path = self.character_events_path(cid)
        legacy_path = self.character_dir(cid) / "events.jsonl"
        row = dict(event or {})
        row.setdefault("ts", now_local_ts())
        self._append_jsonl(path, row)
        if legacy_path != path:
            self._append_jsonl(legacy_path, row)

    def append_manifest_event(self, event: dict[str, Any]) -> None:
        path = self.manifest_events_path()
        legacy_path = self.root / "manifest_events.jsonl"
        row = dict(event or {})
        row.setdefault("ts", now_local_ts())
        row.setdefault("at", now_local_iso())
        self._append_jsonl(path, row)
        if legacy_path != path:
            self._append_jsonl(legacy_path, row)

    def character_events_path(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        root = (self.character_logs_root / cid).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root / "events.jsonl"

    def manifest_events_path(self) -> Path:
        self.character_logs_root.mkdir(parents=True, exist_ok=True)
        return self.character_logs_root / "manifest_events.jsonl"

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(row or {}), ensure_ascii=False) + "\n")

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

    def _discover_character_spec_dirs(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for row in sorted(self.spec_root.iterdir(), key=lambda p: p.name.lower()):
            if not row.is_dir() or row.name.startswith("_"):
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

        discovered = set(self._discover_character_dirs()).union(set(self._discover_character_spec_dirs()))
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
        changed = old_ids != new_ids or old_active != active or len(existing_rows) != len(chars)
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


def _default_evolution_rules() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rules": [
            {
                "when": {"emotion": ["frustrated", "angry", "sad", "anxious", "tired"]},
                "apply": [
                    {"trait": "warmth", "op": "add", "value": 0.08},
                    {"trait": "sarcasm", "op": "add", "value": -0.1},
                    {"set_mood": "romantic_soft"},
                ],
            },
            {
                "when": {"intent": ["task", "bug_report", "code_review", "planning"]},
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
    }


def _default_persona_state() -> dict[str, Any]:
    traits = {
        "warmth": 0.68,
        "sarcasm": 0.32,
        "teasing": 0.46,
        "strictness": 0.52,
        "verbosity": 0.55,
        "empathy": 0.72,
    }
    baselines = {str(k): float(v) for k, v in traits.items()}
    return {
        "schema_version": 1,
        "traits": dict(traits),
        "mood": "thoughtful",
        "locks": {
            "feminine": True,
            "informal_you": True,
        },
        "bans": [],
        "learned": {
            "preferences_confirmed": [],
            "preferences_pending": [],
            "style_bias": {},
            "baseline_traits": dict(baselines),
        },
        "baseline_traits": dict(baselines),
    }


def _default_persona_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": [
            "You are {character_name}.",
            "Use Russian by default unless the user requests another language.",
            "Address the user informally unless explicitly requested otherwise.",
        ],
        "locks_map": {
            "feminine": "Always use feminine grammatical gender for self-reference.",
            "informal_you": "Use informal address form ('ты').",
        },
        "bans_template": "Do not use banned word: {term}.",
        "moods": {
            "focused": ["Mood: focused. Prioritize clarity, concrete steps, and concise delivery."],
            "thoughtful": ["Mood: thoughtful. Show empathy and reasoning before conclusion."],
            "teasing": ["Mood: teasing. Keep playful tone but stay respectful."],
            "ironic": ["Mood: ironic. Keep humor light and non-hostile."],
            "romantic_soft": ["Mood: romantic_soft. Keep soft and supportive tone."],
            "neutral": ["Mood: neutral. Keep balanced and practical tone."],
        },
        "modes": {
            "chatting": ["Keep friendly conversational tone.", "Light humor is allowed when relevant."],
            "helper": ["Prioritize support and clarity.", "Keep warm tone and reduce sarcasm."],
            "engineer": ["Respond in structured technical format.", "Prefer concrete steps and concise explanations."],
            "debugger": ["Use hypothesis-driven debugging flow.", "Ask for diagnostics and exact reproduction steps."],
            "planner": ["Build plans with phases and checkpoints.", "Present deliverables and acceptance criteria."],
        },
        "trait_order": ["warmth", "sarcasm", "verbosity", "strictness", "teasing", "empathy"],
        "traits_rules": {
            "warmth": [
                {"min": 0.75, "line": "Tone: very warm and supportive."},
                {"min": 0.45, "line": "Tone: neutral-warm."},
                {"min": 0.0, "line": "Tone: restrained and concise."},
            ],
            "sarcasm": [
                {"min": 0.6, "line": "Sarcasm: light sarcasm is acceptable, avoid hostility."},
                {"min": 0.2, "line": "Sarcasm: minimal, only if context clearly allows it."},
                {"min": 0.0, "line": "Sarcasm: avoid sarcasm."},
            ],
            "verbosity": [
                {"min": 0.7, "line": "Verbosity: provide detailed explanations."},
                {"min": 0.3, "line": "Verbosity: medium detail with actionable focus."},
                {"min": 0.0, "line": "Verbosity: keep answers short."},
            ],
            "strictness": [
                {"min": 0.7, "line": "Structure: strict, procedural, and concrete."},
                {"min": 0.3, "line": "Structure: balanced between clarity and flexibility."},
                {"min": 0.0, "line": "Structure: flexible conversational flow."},
            ],
            "teasing": [
                {"min": 0.6, "line": "Teasing: light playful teasing is allowed."},
                {"min": 0.2, "line": "Teasing: minimal and only when user tone is positive."},
                {"min": 0.0, "line": "Teasing: avoid teasing."},
            ],
            "empathy": [
                {"min": 0.7, "line": "Empathy: explicitly validate user frustration and effort."},
                {"min": 0.3, "line": "Empathy: maintain calm supportive baseline."},
                {"min": 0.0, "line": "Empathy: keep neutral professional tone."},
            ],
        },
    }


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


def _character_name(cid: str) -> str:
    cleaned = str(cid or "").strip().replace("_", " ").replace("-", " ")
    if not cleaned:
        return DEFAULT_CHARACTER_NAME
    parts = [x for x in cleaned.split(" ") if x]
    if not parts:
        return DEFAULT_CHARACTER_NAME
    return " ".join(p[:1].upper() + p[1:] for p in parts)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _coerce_baseline_map(value: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(value, dict):
        return out
    for key, raw in value.items():
        name = str(key or "").strip().lower()
        if not name:
            continue
        try:
            out[name] = float(_clamp01(float(raw)))
        except Exception:
            continue
    return out


def _normalize_persona_state_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    input_row = dict(value or {}) if isinstance(value, dict) else {}
    input_learned = dict(input_row.get("learned") or {})
    has_root_baseline = isinstance(input_row.get("baseline_traits"), dict)
    has_learned_baseline = isinstance(input_learned.get("baseline_traits"), dict)

    payload = dict(_default_persona_state())
    payload.update(input_row)

    payload["traits"] = dict(payload.get("traits") or {})
    payload["locks"] = dict(payload.get("locks") or {"feminine": True, "informal_you": True})
    payload["bans"] = [str(x).strip() for x in list(payload.get("bans") or []) if str(x).strip()]

    learned = dict(payload.get("learned") or {})
    learned.setdefault("preferences_confirmed", [])
    learned.setdefault("preferences_pending", [])
    learned.setdefault("style_bias", {})

    root_baseline = _coerce_baseline_map(payload.get("baseline_traits")) if has_root_baseline else {}
    learned_baseline = _coerce_baseline_map(learned.get("baseline_traits")) if has_learned_baseline else {}
    trait_seed = _coerce_baseline_map(payload.get("traits"))
    baselines = dict(root_baseline or learned_baseline or trait_seed)
    if not baselines:
        baselines = _coerce_baseline_map(_default_persona_state().get("baseline_traits"))

    # Anchor any newly seen numeric trait to baseline on first sight.
    for key, raw in dict(payload.get("traits") or {}).items():
        name = str(key or "").strip().lower()
        if not name or name in baselines:
            continue
        try:
            baselines[name] = float(_clamp01(float(raw)))
        except Exception:
            continue

    learned["baseline_traits"] = dict(baselines)
    payload["learned"] = learned
    # Keep legacy mirror for backward compatibility with readers that still use root key.
    payload["baseline_traits"] = dict(baselines)
    return payload
