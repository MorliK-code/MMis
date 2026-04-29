from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR
from config.settings import load_config
from metadata.taxonomy import normalize_emotion
from modules.character.trait_policy import clamp_trait_map, clamp_trait_scalar, normalize_trait_name, normalize_trait_record
from utils.datetime_local import now_local_iso, now_local_ts

DEFAULT_CHARACTER_ID = "default"
DEFAULT_CHARACTER_NAME = "Assistant"
RESERVED_CHARACTER_DIRS: set[str] = set()


def _safe_id(value: str) -> str:
    raw = str(value or "").strip().lower()
    out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
    return out or DEFAULT_CHARACTER_ID


class CharacterStorage:
    def __init__(self, root: str | Path | None = None, logs_root: str | Path | None = None):
        cfg = load_config()
        default_root = Path(cfg.memory_dir).expanduser().resolve().parent / "characters_runtime"
        requested_root = (Path(root).expanduser() if root is not None else default_root).resolve()
        legacy_root = (DATA_DIR / "characters").resolve()
        # Hard guard: runtime character storage must not use legacy data/characters root.
        self.root = default_root.resolve() if requested_root == legacy_root else requested_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_root = (Path(logs_root).expanduser() if logs_root is not None else Path(cfg.log_dir)).resolve()
        self.character_logs_root = (self.logs_root / "characters").resolve()
        self.character_logs_root.mkdir(parents=True, exist_ok=True)
        account_spec_root = getattr(cfg, "character_specs_dir", None)
        if account_spec_root is None:
            account_spec_root = Path(cfg.memory_dir).parent / "specs" / "characters"
        self.spec_root = Path(account_spec_root).expanduser().resolve()
        self.spec_root.mkdir(parents=True, exist_ok=True)
        self.global_spec_root = (DATA_DIR / "specs" / "characters").resolve()
        self.legacy_spec_root = (DATA_DIR / "specs" / "rules_for_all" / "characters").resolve()
        self._seed_default_character_specs()
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

    def persona_state_runtime_path(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        return (self.character_dir(cid) / "persona_state.json").resolve()

    def persona_state_spec_path(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        path = (self.spec_root / cid / "persona_state.json").resolve()
        try:
            path.relative_to(self.spec_root)
        except Exception as exc:
            raise ValueError(f"persona_state spec path escapes root: {path}") from exc
        return path

    def persona_spec_path(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        path = (self.spec_root / cid / "persona_spec.json").resolve()
        try:
            path.relative_to(self.spec_root)
        except Exception as exc:
            raise ValueError(f"persona_spec path escapes root: {path}") from exc
        return path

    def emotion_state_runtime_path(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        return (self.character_dir(cid) / "emotion_state.json").resolve()

    def user_addressing_runtime_path(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        return (self.character_dir(cid) / "user_addressing.json").resolve()

    def identity_core_runtime_path(self, character_id: str) -> Path:
        cid = _safe_id(character_id)
        return (self.character_dir(cid) / "identity_core.json").resolve()

    def ensure_defaults(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec_root.mkdir(parents=True, exist_ok=True)
        self.ensure_character_structure(DEFAULT_CHARACTER_ID)
        self.ensure_character_structure("asya")
        self.ensure_all_character_structures()
        self._sync_manifest_with_directories()

    def _migrate_legacy_character_specs(self) -> None:
        for src_root in (self.legacy_spec_root,):
            dst_root = self.spec_root
            if not src_root.exists() or src_root == dst_root:
                continue
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
                continue

    def _seed_default_character_specs(self) -> None:
        for cid in (DEFAULT_CHARACTER_ID, "asya"):
            src_dir = (DATA_DIR / "specs" / "characters" / cid).resolve()
            dst_dir = (self.spec_root / cid).resolve()
            if not src_dir.exists() or src_dir == dst_dir:
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            for item in src_dir.iterdir():
                if item.is_file():
                    target = dst_dir / item.name
                    if not target.exists():
                        try:
                            shutil.copy2(item, target)
                        except Exception:
                            pass

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
                "mood": "neutral",
                "last_update_ts": "",
                "active_traits": ["warmth", "thoughtfulness"],
                "disabled_traits": [],
                "counters": {"banter_hits": 0, "comfort_hits": 0},
                "last_signals": {},
                "applied_rules": [],
                "emotional_state_before": {},
                "emotional_state_after": {},
                "persona_feedback_applied": [],
            },
        )
        _ensure_json(root / "emotion_state.json", _default_emotion_state())
        _ensure_json(root / "user_addressing.json", _default_user_addressing())
        _ensure_json(root / "identity_core.json", _default_identity_core())
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

        return payload

    def update_character(self, character_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Обновить настройки персонажа."""
        cid = _safe_id(character_id)
        self.ensure_character_structure(cid)
        
        # Обновляем runtime конфиг
        runtime_path = self.character_dir(cid) / "character.json"
        payload = _read_json(runtime_path) or {}
        payload.update(updates)
        normalized = self._normalize_character_payload(payload, fallback_id=cid)
        _write_json(runtime_path, normalized)
        
        # Синхронизируем со спеками если нужно
        spec_path = self.character_spec_dir(cid) / "character.json"
        if spec_path.exists():
            spec_payload = _read_json(spec_path) or {}
            spec_payload.update(updates)
            _write_json(spec_path, self._normalize_character_payload(spec_payload, fallback_id=cid))
            
        self._sync_manifest_with_directories()
        return normalized

    def delete_character(self, character_id: str) -> bool:
        """Удалить персонажа."""
        cid = _safe_id(character_id)
        if cid in {DEFAULT_CHARACTER_ID, "asya", "default"}:
            return False
            
        # Удаляем данные рантайма
        runtime_dir = self.root / cid
        if runtime_dir.exists() and runtime_dir.is_dir():
            shutil.rmtree(runtime_dir, ignore_errors=True)
            
        # Удаляем спеки
        spec_dir = self.spec_root / cid
        if spec_dir.exists() and spec_dir.is_dir():
            shutil.rmtree(spec_dir, ignore_errors=True)
            
        self._sync_manifest_with_directories()
        return True

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

    def load_character(self, character_id: str) -> dict[str, Any]:
        """Загрузить основные данные персонажа (character.json)."""
        cid = _safe_id(character_id)
        synced = self._sync_runtime_character_from_specs(cid)
        if synced:
            return synced
        path = self.character_dir(cid) / "character.json"
        payload = _read_json(path)
        if not isinstance(payload, dict):
            return {
                "id": cid,
                "name": _character_name(cid),
                "version": "1.0.0",
                "default_mood": "thoughtful",
                "llm_profile": "BALANCED",
            }
        return self._normalize_character_payload(payload, fallback_id=cid)

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
        normalized = _normalize_runtime_state_payload(payload if isinstance(payload, dict) else {})
        if normalized != payload:
            _write_json(self.character_dir(cid) / "state.json", normalized)
        return normalized

    def save_state(self, character_id: str, state: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        _write_json(self.character_dir(cid) / "state.json", _normalize_runtime_state_payload(state))

    def load_emotion_state(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        path = self.emotion_state_runtime_path(cid)
        payload = _read_json(path)
        normalized = _normalize_emotion_state_payload(payload if isinstance(payload, dict) else {})
        if normalized != payload:
            _write_json(path, normalized)
        return normalized

    def save_emotion_state(self, character_id: str, payload: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        row = self.load_emotion_state(cid)
        row.update(dict(payload or {}))
        _write_json(self.emotion_state_runtime_path(cid), _normalize_emotion_state_payload(row))

    def load_user_addressing(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        path = self.user_addressing_runtime_path(cid)
        payload = _read_json(path)
        normalized = _normalize_user_addressing_payload(payload if isinstance(payload, dict) else {})
        if normalized != payload:
            _write_json(path, normalized)
        return normalized

    def save_user_addressing(self, character_id: str, payload: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        row = self.load_user_addressing(cid)
        row.update(dict(payload or {}))
        normalized = _normalize_user_addressing_payload(row)
        _write_json(self.user_addressing_runtime_path(cid), normalized)
        identity_core = self.load_identity_core(cid)
        identity_core["addressing"] = dict(normalized)
        if str(normalized.get("updated_at") or "").strip():
            identity_core["updated_at"] = str(normalized.get("updated_at") or "").strip()
        _write_json(self.identity_core_runtime_path(cid), _normalize_identity_core_payload(identity_core))

    def load_identity_core(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        path = self.identity_core_runtime_path(cid)
        payload = _read_json(path)
        normalized = _normalize_identity_core_payload(payload if isinstance(payload, dict) else {})
        if normalized != payload:
            _write_json(path, normalized)
        return normalized

    def save_identity_core(self, character_id: str, payload: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        row = self.load_identity_core(cid)
        row.update(dict(payload or {}))
        normalized = _normalize_identity_core_payload(row)
        _write_json(self.identity_core_runtime_path(cid), normalized)

    def load_builtin_traits(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        self.ensure_character_structure(cid)
        payload = _read_json(self.character_dir(cid) / "traits" / "builtin.json")
        if not isinstance(payload, dict):
            return {}
        traits = _normalize_trait_payload_map(payload.get("traits"))
        if traits != dict(payload.get("traits") or {}):
            _write_json(self.character_dir(cid) / "traits" / "builtin.json", {"traits": traits})
        return traits

    def save_builtin_traits(self, character_id: str, traits: dict[str, Any]) -> None:
        """Сохранить встроенные черты персонажа."""
        cid = _safe_id(character_id)
        path = self.character_dir(cid) / "traits" / "builtin.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, {"traits": _normalize_trait_payload_map(traits)})

    def load_learned_traits(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        self.ensure_character_structure(cid)
        payload = _read_json(self.character_dir(cid) / "traits" / "learned.json")
        if not isinstance(payload, dict):
            return {}
        traits = _normalize_trait_payload_map(payload.get("traits"))
        if traits != dict(payload.get("traits") or {}):
            _write_json(self.character_dir(cid) / "traits" / "learned.json", {"traits": traits})
        return traits

    def save_learned_traits(self, character_id: str, traits: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        path = self.character_dir(cid) / "traits" / "learned.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, {"traits": _normalize_trait_payload_map(traits)})

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
        runtime_path = self.persona_state_runtime_path(cid)
        runtime_payload = _read_json(runtime_path)
        if isinstance(runtime_payload, dict):
            normalized = _normalize_persona_state_payload(runtime_payload)
            if normalized != runtime_payload:
                _write_json(runtime_path, normalized)
            return normalized

        seed_payload = _read_json(self.persona_state_spec_path(cid))
        normalized_seed = _normalize_persona_state_payload(seed_payload if isinstance(seed_payload, dict) else {})
        _write_json(runtime_path, normalized_seed)
        return normalized_seed

    def save_persona_state(self, character_id: str, payload: dict[str, Any]) -> None:
        cid = _safe_id(character_id)
        row = self.load_persona_state(cid)
        row.update(dict(payload or {}))
        _write_json(self.persona_state_runtime_path(cid), _normalize_persona_state_payload(row))

    def load_persona_spec(self, character_id: str) -> dict[str, Any]:
        cid = _safe_id(character_id)
        payload = _read_json(self.persona_spec_path(cid))
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
            if cid not in discovered:
                continue
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
                    {"set_mood": "soft_supportive"},
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
        "relation_state": _default_relation_state(traits),
        "bans": [],
        "learned": {
            "preferences_confirmed": [],
            "preferences_pending": [],
            "style_bias": {},
            "baseline_traits": dict(baselines),
        },
        # Stabilizer state для медленной эволюции личности
        "stabilizer": {
            "counters": {},
            "last_promotion_at": "",
            "last_decay_at": "",
        },
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
            "soft_supportive": ["Mood: soft_supportive. Keep calm, supportive, and steady tone."],
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


def _default_emotion_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mood": "neutral",
        "valence": 0.0,
        "arousal": 0.0,
        "intensity": 0.0,
        "trigger": "",
        "last_update_ts": "",
        "cooldown_until_ts": "",
    }


def _default_user_addressing() -> dict[str, Any]:
    return {
        "canonical_name": "",
        "allowed_forms": [],
        "forbidden_forms": [],
        "allow_diminutives": False,
        "use_name_by_default": False,
        "updated_at": "",
    }


def _default_identity_core() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "addressing": _default_user_addressing(),
        "interaction_style": {},
        "boundaries": {},
        "emotional_handling": {},
        "assistant_trait_baseline": {},
        "updated_at": "",
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_json(path: Path, payload) -> None:
    if path.exists():
        return
    _write_json(path, payload)


def _character_name(cid: str) -> str:
    cleaned = str(cid or "").strip().lower()
    if cleaned == "asya":
        return "Ассистент"
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    if not cleaned:
        return DEFAULT_CHARACTER_NAME
    parts = [x for x in cleaned.split(" ") if x]
    if not parts:
        return DEFAULT_CHARACTER_NAME
    return " ".join(p[:1].upper() + p[1:] for p in parts)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    low = float(min(minimum, maximum))
    high = float(max(minimum, maximum))
    return max(low, min(high, float(value)))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_baseline_map(value: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(value, dict):
        return out
    for key, raw in value.items():
        name = normalize_trait_name(key)
        if not name:
            continue
        out[name] = float(clamp_trait_scalar(name, raw, minimum=0.0, maximum=1.0))
    return out


def _normalize_persona_state_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    input_row = dict(value or {}) if isinstance(value, dict) else {}
    input_learned = dict(input_row.get("learned") or {})
    has_root_baseline = isinstance(input_row.get("baseline_traits"), dict)
    has_learned_baseline = isinstance(input_learned.get("baseline_traits"), dict)

    payload = dict(_default_persona_state())
    payload.update(input_row)

    payload["traits"] = dict(clamp_trait_map(payload.get("traits")))
    payload["locks"] = dict(payload.get("locks") or {"feminine": True, "informal_you": True})
    payload["relation_state"] = _normalize_relation_state_payload(
        payload.get("relation_state"),
        traits=payload.get("traits"),
    )
    payload["bans"] = [str(x).strip() for x in list(payload.get("bans") or []) if str(x).strip()]

    learned = dict(payload.get("learned") or {})
    learned.setdefault("preferences_confirmed", [])
    learned.setdefault("preferences_pending", [])
    learned.setdefault("style_bias", {})

    # Backward-compatible read: old payloads may still keep baseline at root.
    root_baseline = _coerce_baseline_map(payload.get("baseline_traits")) if has_root_baseline else {}
    learned_baseline = _coerce_baseline_map(learned.get("baseline_traits")) if has_learned_baseline else {}
    trait_seed = _coerce_baseline_map(payload.get("traits"))
    baselines = dict(learned_baseline or root_baseline or trait_seed)
    if not baselines:
        baselines = _coerce_baseline_map(dict(_default_persona_state().get("learned") or {}).get("baseline_traits"))

    # Anchor any newly seen numeric trait to baseline on first sight.
    for key, raw in dict(payload.get("traits") or {}).items():
        name = normalize_trait_name(key)
        if not name or name in baselines:
            continue
        baselines[name] = float(clamp_trait_scalar(name, raw, minimum=0.0, maximum=1.0))

    next_style_bias: dict[str, float] = {}
    for key, current in dict(payload.get("traits") or {}).items():
        name = normalize_trait_name(key)
        if not name:
            continue
        baseline = float(baselines.get(name, clamp_trait_scalar(name, current, minimum=0.0, maximum=1.0)))
        current_value = float(clamp_trait_scalar(name, current, minimum=0.0, maximum=1.0))
        diff = current_value - baseline
        if abs(diff) <= 1e-9:
            continue
        next_style_bias[name] = diff

    learned["style_bias"] = next_style_bias
    learned["baseline_traits"] = dict(baselines)
    payload["learned"] = learned
    # Canonical storage keeps baseline only under learned.baseline_traits.
    payload.pop("baseline_traits", None)
    
    # Normalize stabilizer state
    stabilizer = dict(payload.get("stabilizer") or {})
    stabilizer.setdefault("counters", {})
    stabilizer.setdefault("last_promotion_at", "")
    stabilizer.setdefault("last_decay_at", "")
    
    # Ensure counters is a dict and clean invalid entries
    counters = dict(stabilizer.get("counters") or {})
    cleaned_counters = {}
    for key, counter in counters.items():
        if isinstance(counter, dict):
            cleaned_counters[key] = counter
    stabilizer["counters"] = cleaned_counters
    
    payload["stabilizer"] = stabilizer
    
    return payload


def _normalize_runtime_state_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(value or {})
    mood = str(payload.get("mood") or "neutral").strip().lower() or "neutral"
    if mood == "romantic_soft":
        mood = "soft_supportive"
    payload["mood"] = mood
    payload["last_update_ts"] = str(payload.get("last_update_ts") or "")
    payload["active_traits"] = sorted(
        {
            str(x).strip().lower()
            for x in list(payload.get("active_traits") or [])
            if str(x).strip()
        }
    )
    payload["disabled_traits"] = sorted(
        {
            str(x).strip().lower()
            for x in list(payload.get("disabled_traits") or [])
            if str(x).strip()
        }
    )
    counters = dict(payload.get("counters") or {})
    payload["counters"] = {
        "banter_hits": max(0, int(_to_float(counters.get("banter_hits"), 0))),
        "comfort_hits": max(0, int(_to_float(counters.get("comfort_hits"), 0))),
    }
    payload["last_signals"] = dict(payload.get("last_signals") or {})
    payload["applied_rules"] = [dict(x) for x in list(payload.get("applied_rules") or []) if isinstance(x, dict)]
    payload["emotional_state_before"] = dict(payload.get("emotional_state_before") or {})
    payload["emotional_state_after"] = dict(payload.get("emotional_state_after") or {})
    payload["persona_feedback_applied"] = [
        str(x).strip()
        for x in list(payload.get("persona_feedback_applied") or [])
        if str(x).strip()
    ]
    emotion = str(payload.get("emotion") or "").strip().lower()
    payload["emotion"] = "" if emotion == "romantic_soft" else emotion
    return payload


def _default_relation_state(traits: dict[str, Any] | None = None) -> dict[str, float]:
    row = dict(traits or {})
    warmth = _clamp01(_to_float(row.get("warmth"), 0.58))
    empathy = _clamp01(_to_float(row.get("empathy"), 0.62))
    teasing = _clamp01(_to_float(row.get("teasing"), _to_float(row.get("playfulness"), 0.35)))
    return {
        "familiarity": float(_clamp01(0.28 + max(0.0, warmth - 0.5) * 0.12 + max(0.0, empathy - 0.5) * 0.08)),
        "trust": float(_clamp01(0.54 + max(0.0, empathy - 0.5) * 0.16)),
        "teasing_permission": float(_clamp01(0.08 + max(0.0, teasing - 0.3) * 0.55)),
        "softness_bias": float(_clamp01((warmth * 0.55) + (empathy * 0.45))),
    }


def _normalize_relation_state_payload(value: dict[str, Any] | None, *, traits: dict[str, Any] | None = None) -> dict[str, float]:
    payload = dict(_default_relation_state(traits))
    input_row = dict(value or {})
    for key in ("familiarity", "trust", "teasing_permission", "softness_bias"):
        if key not in input_row:
            continue
        payload[key] = float(_clamp01(_to_float(input_row.get(key), payload[key])))
    return payload


def _normalize_emotion_state_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(_default_emotion_state())
    payload.update(dict(value or {}))
    mood = str(payload.get("mood") or "neutral").strip().lower() or "neutral"
    if mood == "romantic_soft":
        mood = "soft_supportive"
    payload["schema_version"] = 1
    payload["mood"] = mood
    payload["valence"] = float(_clamp(_to_float(payload.get("valence"), 0.0), -1.0, 1.0))
    payload["arousal"] = float(_clamp01(_to_float(payload.get("arousal"), 0.0)))
    payload["intensity"] = float(_clamp01(_to_float(payload.get("intensity"), 0.0)))
    trigger = normalize_emotion(payload.get("trigger"))
    payload["trigger"] = "" if trigger == "neutral" and payload["intensity"] <= 0.0 else str(trigger or "")
    payload["last_update_ts"] = str(payload.get("last_update_ts") or "")
    payload["cooldown_until_ts"] = str(payload.get("cooldown_until_ts") or "")
    return payload


def _normalize_user_addressing_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(_default_user_addressing())
    payload.update(dict(value or {}))
    payload["canonical_name"] = _normalize_name_form(payload.get("canonical_name"))
    payload["allowed_forms"] = _normalize_name_form_list(payload.get("allowed_forms"))
    payload["forbidden_forms"] = _normalize_name_form_list(payload.get("forbidden_forms"))
    payload["allow_diminutives"] = bool(payload.get("allow_diminutives", False))
    payload["use_name_by_default"] = bool(payload.get("use_name_by_default", False))
    payload["updated_at"] = str(payload.get("updated_at") or "")

    canonical_key = payload["canonical_name"].casefold()
    if canonical_key:
        if all(str(x).casefold() != canonical_key for x in payload["allowed_forms"]):
            payload["allowed_forms"].insert(0, payload["canonical_name"])
        payload["forbidden_forms"] = [x for x in payload["forbidden_forms"] if str(x).casefold() != canonical_key]
    return payload


def _normalize_identity_core_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(value or {})
    addressing = _normalize_user_addressing_payload(row.get("addressing"))
    payload = {
        "schema_version": 1,
        "addressing": addressing,
        "interaction_style": _normalize_identity_core_interaction_style(row.get("interaction_style")),
        "boundaries": _normalize_identity_core_boundaries(row.get("boundaries")),
        "emotional_handling": _normalize_identity_core_emotional_handling(row.get("emotional_handling")),
        "assistant_trait_baseline": _normalize_identity_core_assistant_trait_baseline(row.get("assistant_trait_baseline")),
        "updated_at": str(
            row.get("updated_at")
            or dict(addressing or {}).get("updated_at")
            or ""
        ).strip(),
    }
    return payload


def _normalize_identity_core_interaction_style(value: Any) -> dict[str, Any]:
    row = dict(value or {})
    out: dict[str, Any] = {}
    if "prefers_directness" in row:
        out["prefers_directness"] = float(_clamp01(_to_float(row.get("prefers_directness"), 0.0)))
    if "prefers_short_answers" in row:
        out["prefers_short_answers"] = float(_clamp01(_to_float(row.get("prefers_short_answers"), 0.0)))
    if "allows_light_teasing" in row:
        out["allows_light_teasing"] = bool(row.get("allows_light_teasing"))
    style = str(row.get("technical_collaboration_style") or "").strip().lower()
    if style in {"low", "medium", "high"}:
        out["technical_collaboration_style"] = style
    return out


def _normalize_identity_core_boundaries(value: Any) -> dict[str, Any]:
    row = dict(value or {})
    out: dict[str, Any] = {}
    for key in (
        "avoid_overloaded_intros",
        "avoid_baby_talk",
        "avoid_overformal_tone",
        "do_not_invent_user_facts",
    ):
        if key in row:
            out[key] = bool(row.get(key))
    return out


def _normalize_identity_core_emotional_handling(value: Any) -> dict[str, Any]:
    row = dict(value or {})
    out: dict[str, Any] = {}
    if "deescalate_on_irritation" in row:
        out["deescalate_on_irritation"] = bool(row.get("deescalate_on_irritation"))
    if "treat_short_replies_as_low_bandwidth" in row:
        out["treat_short_replies_as_low_bandwidth"] = bool(row.get("treat_short_replies_as_low_bandwidth"))
    if "warmth_upshift_on_user_distress" in row:
        out["warmth_upshift_on_user_distress"] = float(
            _clamp01(_to_float(row.get("warmth_upshift_on_user_distress"), 0.0))
        )
    if "playfulness_downshift_on_user_distress" in row:
        out["playfulness_downshift_on_user_distress"] = float(
            _clamp01(_to_float(row.get("playfulness_downshift_on_user_distress"), 0.0))
        )
    return out


def _normalize_identity_core_assistant_trait_baseline(value: Any) -> dict[str, Any]:
    row = dict(value or {})
    out: dict[str, Any] = {}
    for key in (
        "warmth_baseline",
        "directness_baseline",
        "empathy_floor",
        "professionalism_floor",
        "sarcasm_ceiling",
    ):
        if key in row:
            out[key] = float(_clamp01(_to_float(row.get(key), 0.0)))
    return out


def _normalize_name_form(value: Any) -> str:
    text = str(value or "").strip()
    text = text.strip(" \t\r\n.,!?;:()[]{}\"'`«»")
    text = " ".join(text.split())
    if not text:
        return ""
    if len(text) > 40:
        return ""
    if any(ch.isdigit() for ch in text):
        return ""
    return text


def _normalize_name_form_list(value: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in list(value or []):
        item = _normalize_name_form(row)
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_trait_payload_map(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(value, dict):
        return out
    for raw_name, raw_row in dict(value).items():
        name = normalize_trait_name(raw_name)
        if not name:
            continue
        out[name] = normalize_trait_record(name, raw_row if isinstance(raw_row, dict) else {})
    return out
