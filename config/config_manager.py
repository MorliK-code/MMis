from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


LEGACY_KEY_MAP: dict[str, str] = {
    "app_name": "app.name",
    "debug": "app.debug",
    "locale": "app.locale",
    "default_language": "app.default_language",
    "startup_mode": "startup.mode",
    "active_profile": "startup.active_profile",
    "safety_mode": "startup.safety_mode",
    "host": "api.host",
    "port": "api.port",
    "llm_default_provider": "llm.provider",
    "model_name": "llm.model_name",
    "model_fallbacks": "llm.model_fallbacks",
    "thinking_enabled": "llm.thinking_enabled",
    "json_mode_enabled": "llm.json_mode_enabled",
    "llm_max_tokens_lower_bound": "llm.max_tokens.lower_bound",
    "llm_max_tokens_upper_bound": "llm.max_tokens.upper_bound",
    "metadata_model": "llm.metadata.model",
    "metadata_model_fallbacks": "llm.metadata.fallbacks",
    "ollama_base_url": "llm.providers.ollama.base_url",
    "ollama_timeout_sec": "llm.providers.ollama.timeout_sec",
    "ollama_retries": "llm.providers.ollama.retries",
    "openai_api_key": "llm.providers.openai.api_key",
    "openai_api_url": "llm.providers.openai.api_url",
    "openai_timeout_sec": "llm.providers.openai.timeout_sec",
    "openai_max_retries": "llm.providers.openai.max_retries",
    "internet_enabled": "internet.enabled",
    "web_mode": "internet.web_mode",
    "search_api_url": "internet.search.api_url",
    "search_provider": "internet.search.provider",
    "search_strict_endpoint": "internet.search.strict_endpoint",
    "search_timeout_sec": "internet.search.timeout_sec",
    "web_fetch_timeout_sec": "internet.fetch.timeout_sec",
    "web_fetch_retries": "internet.fetch.retries",
    "web_clean_max_chars": "internet.fetch.clean_max_chars",
    "web_clean_min_chars": "internet.fetch.clean_min_chars",
    "web_clean_language_hint": "internet.fetch.clean_language_hint",
    "memory_dir": "memory.memory_dir",
    "cache_dir": "memory.cache_dir",
    "log_dir": "memory.log_dir",
    "db_path": "memory.db_path",
    "data_dir": "paths.data_dir",
    "models_dir": "paths.models_dir",
    "chat_recall_results": "memory.chat_recall_results",
    "chat_events_limit": "memory.chat_events_limit",
    "chat_proofread": "memory.chat_proofread",
    "chat_proofread_strict": "memory.chat_proofread_strict",
    "dialog_new_session_after_min": "dialog.new_session_after_min",
    "dialog_greeting_max_words": "dialog.greeting_max_words",
    "dialog_greeting_max_chars": "dialog.greeting_max_chars",
    "dialog_greetings": "dialog.greetings",
    "dialog_greeting_exclusions": "dialog.greeting_exclusions",
    "voice_enabled": "voice.enabled",
    "voice_tts_voice": "voice.tts.voice",
    "voice_tts_rate": "voice.tts.rate",
    "voice_tts_volume": "voice.tts.volume",
    "voice_input_dir": "voice.paths.input_dir",
    "voice_output_dir": "voice.paths.output_dir",
    "automation_enabled": "modules.automation_enabled",
    "screen_enabled": "modules.screen_enabled",
    "prompt_response_safety_filter_enabled": "prompt.response_safety_filter_enabled",
    "prompt_response_formatting_enabled": "prompt.response_formatting_enabled",
    "log_level": "logging.level",
    "log_file": "logging.file",
    "log_colors": "logging.colors",
    "log_max_bytes": "logging.max_bytes",
    "log_backup_count": "logging.backup_count",
    "log_format": "logging.format",
    "log_channels": "logging.channels",
    "log_web_trace_enabled": "logging.web_trace_enabled",
    "log_web_trace_logger": "logging.web_trace_logger",
    "gpu_vram_gb": "hardware.gpu_vram_gb",
    "feature_flags": "features.flags",
    "console_model": "llm.model_name",
    "console_timeout_sec": "ui.console.timeout_sec",
    "console_stream_timeout_sec": "ui.console.stream_timeout_sec",
    "console_store_turn": "ui.console.store_turn",
    "console_show_thinking": "ui.console.show_thinking",
    "console_thinking_first": "ui.console.thinking_first",
    "console_json_mode_enabled": "llm.json_mode_enabled",
    "console_auto_start_api": "ui.console.auto_start_api",
    "console_auto_start_ollama": "ui.console.auto_start_ollama",
    "console_mode_lock": "ui.console.runtime.mode_lock",
    "console_active_mode": "ui.console.runtime.active_mode",
    "console_output_parameters": "ui.console.runtime.output_parameters",
    "console_output_summary": "ui.console.runtime.output_summary",
    "console_runtime": "ui.console.runtime",
    "model_profiles": "llm.profiles",
}

_DEPRECATED_TOP_LEVEL_KEYS: set[str] = {
    "read_only_tools",
    "base_dir",
    "config_dir",
    "default_memory_dir",
    "legacy_memory_dir",
    "console_last_api_base_url",
    "web_auto_profile",
}

_DEPRECATED_DOTTED_KEYS: tuple[str, ...] = (
    "startup.read_only_tools",
    "internet.web_auto_profile",
    "ui.console.model",
    "ui.console.json_mode_enabled",
    "ui.console.last_api_base_url",
    "ui.console.runtime.think_enabled",
    "ui.console.runtime.web_mode",
    "ui.console.runtime.web_auto_profile",
    "ui.console.runtime.json_mode_enabled",
    "ui.console.runtime.model",
    "features.extra_legacy.console_runtime_state",
    "paths.base_dir",
    "paths.config_dir",
    "paths.default_memory_dir",
    "paths.legacy_memory_dir",
)

DIR_PATH_TOKEN = "{dir_path}"
_DIR_PATH_TOKEN_LOW = DIR_PATH_TOKEN.lower()
_PROJECT_PATH_KEYS: tuple[str, ...] = (
    "memory.memory_dir",
    "memory.cache_dir",
    "memory.log_dir",
    "memory.db_path",
    "voice.paths.input_dir",
    "voice.paths.output_dir",
    "paths.data_dir",
    "paths.models_dir",
)


@dataclass
class ConfigManager:
    path: Path
    defaults: dict[str, Any]
    project_dir: Path | None = None

    def load_or_create(self, *, bootstrap_seed: dict[str, Any] | None = None) -> dict[str, Any]:
        defaults = copy.deepcopy(self.defaults if isinstance(self.defaults, dict) else {})
        cfg_path = Path(self.path).expanduser().resolve()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)

        exists = cfg_path.exists()
        raw_text = ""
        if exists:
            raw_text = cfg_path.read_text(encoding="utf-8-sig")
            payload = self._parse_json(raw_text)
        else:
            payload = {}

        migrated, migrated_changed = self.migrate_legacy_flat_keys(payload)
        normalized, normalized_changed = self.normalize_and_fill_defaults(migrated, defaults=defaults)

        if not exists and isinstance(bootstrap_seed, dict) and bootstrap_seed:
            merged = copy.deepcopy(normalized)
            _deep_merge(merged, dict(bootstrap_seed))
            normalized, _ = self.normalize_and_fill_defaults(merged, defaults=defaults)
            normalized_changed = True

        if exists and migrated_changed:
            self._write_migration_backup(raw_text)

        if (not exists) or migrated_changed or normalized_changed:
            self.save_atomic(normalized)

        return normalized

    def _project_root(self) -> Path:
        if self.project_dir is not None:
            return Path(self.project_dir).expanduser().resolve()
        cfg_path = Path(self.path).expanduser().resolve()
        try:
            return cfg_path.parent.parent.resolve()
        except Exception:
            return cfg_path.parent.resolve()

    def normalize_and_fill_defaults(
        self,
        payload: dict[str, Any] | None,
        *,
        defaults: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        base = copy.deepcopy(defaults if isinstance(defaults, dict) else self.defaults)
        src = payload if isinstance(payload, dict) else {}
        out = copy.deepcopy(base)
        _deep_merge(out, src)
        return out, (out != src)

    def migrate_legacy_flat_keys(self, payload: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
        src = copy.deepcopy(payload if isinstance(payload, dict) else {})
        if not isinstance(src, dict):
            return {}, True
        out = copy.deepcopy(src)
        changed = False

        known_sections = set(self.defaults.keys()) if isinstance(self.defaults, dict) else set()
        extra_legacy = _as_dict(_get_dotted(out, "features.extra_legacy"))

        # Special handling for old `console_runtime_state`.
        runtime_state = _as_dict(out.get("console_runtime_state"))
        if runtime_state:
            if str(runtime_state.get("model") or "").strip() and _get_dotted(out, "llm.model_name") in {None, ""}:
                _set_dotted(out, "llm.model_name", str(runtime_state.get("model")).strip())
            if runtime_state.get("think_enabled") is not None and _get_dotted(out, "llm.thinking_enabled") is None:
                _set_dotted(out, "llm.thinking_enabled", bool(runtime_state.get("think_enabled")))
            if runtime_state.get("json_mode_enabled") is not None and _get_dotted(out, "llm.json_mode_enabled") is None:
                _set_dotted(out, "llm.json_mode_enabled", bool(runtime_state.get("json_mode_enabled")))
            web_mode = str(runtime_state.get("web_mode") or "").strip().lower()
            if web_mode in {"on", "off", "auto"} and _get_dotted(out, "internet.web_mode") is None:
                _set_dotted(out, "internet.web_mode", web_mode)
            if runtime_state.get("mode_lock") is not None and _get_dotted(out, "ui.console.runtime.mode_lock") is None:
                _set_dotted(out, "ui.console.runtime.mode_lock", bool(runtime_state.get("mode_lock")))
            if str(runtime_state.get("active_mode") or "").strip() and _get_dotted(out, "ui.console.runtime.active_mode") in {None, ""}:
                _set_dotted(out, "ui.console.runtime.active_mode", str(runtime_state.get("active_mode")).strip())
            if runtime_state.get("output_parameters") is not None and _get_dotted(out, "ui.console.runtime.output_parameters") is None:
                _set_dotted(out, "ui.console.runtime.output_parameters", bool(runtime_state.get("output_parameters")))
            if runtime_state.get("output_summary") is not None and _get_dotted(out, "ui.console.runtime.output_summary") is None:
                _set_dotted(out, "ui.console.runtime.output_summary", bool(runtime_state.get("output_summary")))
            out.pop("console_runtime_state", None)
            changed = True

        for key in list(out.keys()):
            if key in known_sections:
                continue
            if key in _DEPRECATED_TOP_LEVEL_KEYS:
                out.pop(key, None)
                changed = True
                continue
            if str(key) == "read_only_tools":
                _migrate_read_only_tools(out, out.get(key))
                out.pop(key, None)
                changed = True
                continue
            mapped = LEGACY_KEY_MAP.get(str(key))
            if mapped:
                if _get_dotted(out, mapped) is None:
                    _set_dotted(out, mapped, out.get(key))
                out.pop(key, None)
                changed = True
                continue
            if key.startswith("_"):
                continue
            extra_legacy[key] = out.get(key)
            out.pop(key, None)
            changed = True

        changed = _migrate_nested_deprecated_keys(out) or changed
        changed = _normalize_project_paths(out, project_root=self._project_root()) or changed

        extra_legacy.pop("console_runtime_state", None)
        if extra_legacy:
            _set_dotted(out, "features.extra_legacy", extra_legacy)
        return out, changed

    def save_atomic(self, payload: dict[str, Any]) -> None:
        cfg_path = Path(self.path).expanduser().resolve()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(cfg_path)

    def update(self, dotted_path: str, value: Any) -> dict[str, Any]:
        return self.update_many({str(dotted_path or ""): value})

    def update_many(self, updates: dict[str, Any]) -> dict[str, Any]:
        payload = self.load_or_create()
        changed = False
        for path, value in dict(updates or {}).items():
            key = str(path or "").strip()
            if not key:
                continue
            before = _get_dotted(payload, key)
            if before != value:
                _set_dotted(payload, key, value)
                changed = True

        migrated, migrated_changed = self.migrate_legacy_flat_keys(payload)
        normalized, normalized_changed = self.normalize_and_fill_defaults(migrated)
        if changed or migrated_changed or normalized_changed:
            self.save_atomic(normalized)
        return normalized

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_migration_backup(self, raw_text: str) -> None:
        cfg_path = Path(self.path).expanduser().resolve()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = cfg_path.with_name(f"{cfg_path.stem}.pre_migration.{stamp}{cfg_path.suffix}")
        backup.write_text(str(raw_text or ""), encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _get_dotted(payload: dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for part in [x for x in str(dotted or "").split(".") if x]:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def _set_dotted(payload: dict[str, Any], dotted: str, value: Any) -> None:
    parts = [x for x in str(dotted or "").split(".") if x]
    if not parts:
        return
    cur: dict[str, Any] = payload
    for part in parts[:-1]:
        node = cur.get(part)
        if not isinstance(node, dict):
            node = {}
            cur[part] = node
        cur = node
    cur[parts[-1]] = value


def _remove_dotted(payload: dict[str, Any], dotted: str) -> bool:
    parts = [x for x in str(dotted or "").split(".") if x]
    if not parts:
        return False
    cur: Any = payload
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur.get(part)
    if not isinstance(cur, dict):
        return False
    if parts[-1] in cur:
        cur.pop(parts[-1], None)
        return True
    return False


def _migrate_read_only_tools(payload: dict[str, Any], value: Any) -> bool:
    if _get_dotted(payload, "startup.safety_mode") not in {None, ""}:
        return False
    flag = bool(value)
    _set_dotted(payload, "startup.safety_mode", "read_only_tools" if flag else "allow_os_actions")
    return True


def _migrate_nested_deprecated_keys(payload: dict[str, Any]) -> bool:
    changed = False

    if str(_get_dotted(payload, "ui.console.model") or "").strip() and _get_dotted(payload, "llm.model_name") in {None, ""}:
        _set_dotted(payload, "llm.model_name", str(_get_dotted(payload, "ui.console.model")).strip())
        changed = True
    if _get_dotted(payload, "ui.console.json_mode_enabled") is not None and _get_dotted(payload, "llm.json_mode_enabled") is None:
        _set_dotted(payload, "llm.json_mode_enabled", bool(_get_dotted(payload, "ui.console.json_mode_enabled")))
        changed = True

    if _migrate_read_only_tools(payload, _get_dotted(payload, "startup.read_only_tools")):
        changed = True

    runtime = _as_dict(_get_dotted(payload, "ui.console.runtime"))
    if runtime:
        if runtime.get("think_enabled") is not None and _get_dotted(payload, "llm.thinking_enabled") is None:
            _set_dotted(payload, "llm.thinking_enabled", bool(runtime.get("think_enabled")))
            changed = True
        if runtime.get("json_mode_enabled") is not None and _get_dotted(payload, "llm.json_mode_enabled") is None:
            _set_dotted(payload, "llm.json_mode_enabled", bool(runtime.get("json_mode_enabled")))
            changed = True
        web_mode = str(runtime.get("web_mode") or "").strip().lower()
        if web_mode in {"on", "off", "auto"} and _get_dotted(payload, "internet.web_mode") is None:
            _set_dotted(payload, "internet.web_mode", web_mode)
            changed = True
        if str(runtime.get("model") or "").strip() and _get_dotted(payload, "llm.model_name") in {None, ""}:
            _set_dotted(payload, "llm.model_name", str(runtime.get("model")).strip())
            changed = True

    for dotted in _DEPRECATED_DOTTED_KEYS:
        if _remove_dotted(payload, dotted):
            changed = True

    return changed


def _normalize_project_paths(payload: dict[str, Any], *, project_root: Path) -> bool:
    changed = False
    root = Path(project_root).expanduser().resolve()
    for dotted in _PROJECT_PATH_KEYS:
        before = _get_dotted(payload, dotted)
        if before is None:
            continue
        after = _normalize_project_path_value(before, project_root=root)
        if after != before:
            _set_dotted(payload, dotted, after)
            changed = True
    return changed


def _normalize_project_path_value(value: Any, *, project_root: Path) -> Any:
    if value is None:
        return value
    if not isinstance(value, (str, Path)):
        return value
    src = str(value).strip()
    if not src:
        return value
    expanded = _expand_dir_path_token(src, project_root=project_root)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = (project_root / path)
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(project_root)
    except Exception:
        return str(resolved)

    rel_text = str(rel).replace("/", "\\")
    if not rel_text or rel_text == ".":
        return DIR_PATH_TOKEN
    return f"{DIR_PATH_TOKEN}\\{rel_text}"


def _expand_dir_path_token(value: str, *, project_root: Path) -> str:
    src = str(value or "")
    if not src:
        return src
    low = src.lower()
    if _DIR_PATH_TOKEN_LOW not in low:
        return src
    out: list[str] = []
    idx = 0
    token_len = len(DIR_PATH_TOKEN)
    root_text = str(Path(project_root).expanduser().resolve())
    while idx < len(src):
        if low[idx : idx + token_len] == _DIR_PATH_TOKEN_LOW:
            out.append(root_text)
            idx += token_len
            continue
        out.append(src[idx])
        idx += 1
    return "".join(out)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in dict(patch or {}).items():
        if key not in base:
            base[key] = copy.deepcopy(value)
            continue
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_merge(current, value)
            continue
        base[key] = copy.deepcopy(value)
