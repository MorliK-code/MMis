from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from config.paths import DATA_DIR


_SPEC_FILES = {
    "system": "system_spec.json",
    "metadata": "metadata_spec.json",
    "output": "output_spec.json",
    "memory": "memory_spec.json",
    "modes": "modes_spec.json",
    "taxonomy": "taxonomy.json",
}

_CHARACTER_FILES = {
    "character": "character.json",
    "persona_state": "persona_state.json",
    "persona_spec": "persona_spec.json",
    "evolution": "evolution_spec.json",
}


def _normalize_name(value: str) -> str:
    return str(value or "").strip().lower()


def _safe_character_id(value: str) -> str:
    raw = str(value or "").strip().lower()
    out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
    return out or "default"


def _assert_json_path(path: Path, *, label: str = "spec path") -> None:
    if str(path.suffix or "").strip().lower() != ".json":
        raise ValueError(f"{label} must point to .json file, got: {path}")


class SpecRegistry:
    def __init__(self, root: str | Path | None = None):
        base = Path(root).expanduser() if root is not None else (DATA_DIR / "specs")
        self.root = base.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    @property
    def character_root(self) -> Path:
        return self.root / "characters"

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def load_spec(self, name: str, *, required: bool = True) -> dict[str, Any]:
        key = _normalize_name(name)
        if key not in _SPEC_FILES:
            raise KeyError(f"unknown spec name: {name}")
        path = self.root / _SPEC_FILES[key]
        return self._load_json(path=path, cache_key=f"spec:{key}", required=required)

    def load_character_spec(self, character_id: str, name: str, *, required: bool = True) -> dict[str, Any]:
        cid = _safe_character_id(character_id)
        key = _normalize_name(name)
        if key not in _CHARACTER_FILES:
            raise KeyError(f"unknown character spec name: {name}")
        path = self.character_root / cid / _CHARACTER_FILES[key]
        return self._load_json(path=path, cache_key=f"character:{cid}:{key}", required=required)

    def list_prompt_keys(self) -> list[str]:
        keys: set[str] = set()
        for spec_name in ("system", "memory", "metadata"):
            spec = self.load_spec(spec_name, required=False)
            prompts = dict(spec.get("prompts") or {})
            keys.update(str(x).strip() for x in prompts.keys() if str(x).strip())
        return sorted(keys)

    def load_prompt_document(self, key: str) -> dict[str, Any]:
        name = str(key or "").strip()
        if not name:
            raise ValueError("prompt key is empty")
        for spec_name in ("system", "memory", "metadata"):
            spec = self.load_spec(spec_name, required=False)
            prompts = dict(spec.get("prompts") or {})
            if name not in prompts:
                continue
            row = dict(prompts.get(name) or {})
            return {
                "key": name,
                "id": str(row.get("id") or name),
                "version": str(row.get("version") or "0.0.0"),
                "role": str(row.get("role") or "system"),
                "tags": [str(x).strip().lower() for x in list(row.get("tags") or []) if str(x).strip()],
                "min_ctx": _to_int(row.get("min_ctx"), default=0, minimum=0),
                "text": str(row.get("text") or ""),
                "source_path": str(self.root / _SPEC_FILES[spec_name]),
                "rel_path": f"{_SPEC_FILES[spec_name]}#prompts.{name}",
                "spec_name": spec_name,
            }
        raise KeyError(f"unknown prompt key: {name}")

    def validate_no_txt_refs(self) -> None:
        problems: list[str] = []
        for path in sorted(self.root.rglob("*.json")):
            payload = self._load_json(path=path, cache_key=f"scan:{path}", required=False)
            problems.extend(_collect_txt_refs(payload=payload, path_hint=str(path)))
        if problems:
            raise ValueError("TXT references are forbidden in specs:\n- " + "\n- ".join(sorted(set(problems))))

    def _load_json(self, *, path: Path, cache_key: str, required: bool) -> dict[str, Any]:
        _assert_json_path(path, label="spec path")
        with self._lock:
            if cache_key in self._cache:
                return dict(self._cache[cache_key])
            if not path.exists():
                if required:
                    raise FileNotFoundError(f"spec file not found: {path}")
                self._cache[cache_key] = {}
                return {}
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                raise ValueError(f"invalid spec json: {path} ({exc})") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"spec must contain object at root: {path}")
            self._cache[cache_key] = dict(payload)
            return dict(payload)


def _to_int(value: Any, *, default: int, minimum: int = 0) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(int(minimum), out)


def _collect_txt_refs(*, payload: Any, path_hint: str, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_collect_txt_refs(payload=value, path_hint=path_hint, prefix=child_prefix))
        return out
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            child_prefix = f"{prefix}[{idx}]"
            out.extend(_collect_txt_refs(payload=value, path_hint=path_hint, prefix=child_prefix))
        return out
    if not isinstance(payload, str):
        return out
    value = str(payload).strip().lower()
    if ".txt" not in value:
        return out
    tail = prefix.split(".")[-1].lower() if prefix else ""
    if tail.endswith(("path", "file", "prompt", "source", "template")) or value.endswith(".txt"):
        out.append(f"{path_hint}:{prefix}={payload}")
    return out


_REGISTRY_LOCK = RLock()
_REGISTRY_SINGLETON: SpecRegistry | None = None


def get_spec_registry(*, force_reload: bool = False) -> SpecRegistry:
    global _REGISTRY_SINGLETON
    with _REGISTRY_LOCK:
        if _REGISTRY_SINGLETON is None or force_reload:
            _REGISTRY_SINGLETON = SpecRegistry()
        return _REGISTRY_SINGLETON


def load_spec(name: str, *, required: bool = True) -> dict[str, Any]:
    return get_spec_registry().load_spec(name, required=required)


def load_character_spec(character_id: str, name: str, *, required: bool = True) -> dict[str, Any]:
    return get_spec_registry().load_character_spec(character_id=character_id, name=name, required=required)


def invalidate_spec_cache() -> None:
    global _REGISTRY_SINGLETON
    with _REGISTRY_LOCK:
        if _REGISTRY_SINGLETON is None:
            return
        _REGISTRY_SINGLETON.invalidate()


def validate_no_txt_paths(config: Any = None) -> None:
    registry = get_spec_registry()
    registry.validate_no_txt_refs()
    if config is None:
        return
    cfg_map = _to_mapping(config)
    problems = _collect_cfg_txt_paths(cfg_map)
    if problems:
        raise ValueError("TXT runtime paths are forbidden:\n- " + "\n- ".join(sorted(set(problems))))


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _collect_cfg_txt_paths(payload: dict[str, Any], *, prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, value in dict(payload or {}).items():
        name = str(key or "")
        path_name = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            out.extend(_collect_cfg_txt_paths(value, prefix=path_name))
            continue
        if isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    out.extend(_collect_cfg_txt_paths(item, prefix=f"{path_name}[{idx}]"))
                elif isinstance(item, str) and ".txt" in item.lower():
                    out.append(f"{path_name}[{idx}]={item}")
            continue
        if not isinstance(value, str):
            continue
        low = value.strip().lower()
        if ".txt" not in low:
            continue
        k = name.lower()
        if any(token in k for token in ("prompt", "spec", "path", "file", "template")):
            out.append(f"{path_name}={value}")
    return out
