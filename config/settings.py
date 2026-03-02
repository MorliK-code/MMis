from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config.paths import BASE_DIR, DATA_DIR, MODELS_DIR, ensure_dirs, resolve_memory_dir


VALID_PROFILES = {"FAST", "BALANCED", "QUALITY", "ECONOM"}
VALID_PROVIDERS = {"ollama", "auto"}
VALID_SAFETY_MODES = {"read_only_tools", "allow_os_actions"}


@dataclass(frozen=True)
class AppSettings:
    app_name: str
    debug: bool
    locale: str
    default_language: str
    startup_mode: str
    active_profile: str
    llm_default_provider: str
    model_name: str
    host: str
    port: int
    thinking_enabled: bool
    web_mode: str
    json_mode_enabled: bool
    internet_enabled: bool
    automation_enabled: bool
    screen_enabled: bool
    voice_enabled: bool
    safety_mode: str
    read_only_tools: bool
    data_dir: Path
    models_dir: Path
    memory_dir: Path
    cache_dir: Path
    log_dir: Path
    db_path: Path
    dialog_new_session_after_min: int
    dialog_greeting_max_words: int
    dialog_greeting_max_chars: int
    dialog_greetings: list[str] = field(default_factory=list)
    dialog_greeting_exclusions: list[str] = field(default_factory=list)
    config_file: Path | None = None
    feature_flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key in ("data_dir", "models_dir", "memory_dir", "cache_dir", "log_dir", "db_path", "config_file"):
            if row.get(key) is not None:
                row[key] = str(row[key])
        return row


# Backward-compatible alias.
AppConfig = AppSettings


_SETTINGS_CACHE: AppSettings | None = None


def load_config(force_reload: bool = False) -> AppSettings:
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None and not force_reload:
        return _SETTINGS_CACHE

    config_file = _resolve_config_file()
    dotenv_file = _resolve_dotenv_file()
    json_cfg = _read_json_file(config_file) if config_file is not None else {}
    dotenv_cfg = _read_dotenv_file(dotenv_file) if dotenv_file is not None else {}

    active_profile = _norm_upper(_pick("MMIS_ACTIVE_PROFILE", json_cfg, dotenv_cfg, "BALANCED"))
    llm_provider = _norm_lower(_pick("MMIS_LLM_PROVIDER", json_cfg, dotenv_cfg, "ollama"))
    safety_mode = _norm_lower(_pick("MMIS_SAFETY_MODE", json_cfg, dotenv_cfg, "read_only_tools"))

    memory_dir = _resolve_memory_path(json_cfg=json_cfg, dotenv_cfg=dotenv_cfg)
    dirs = ensure_dirs(memory_dir=memory_dir)
    db_default = str(memory_dir / "memory.db")

    settings = AppSettings(
        app_name=_norm_str(_pick("MMIS_APP_NAME", json_cfg, dotenv_cfg, "MMis")),
        debug=_to_bool(_pick("MMIS_DEBUG", json_cfg, dotenv_cfg, False)),
        locale=_norm_str(_pick("MMIS_LOCALE", json_cfg, dotenv_cfg, "ru_RU")),
        default_language=_norm_str(_pick("MMIS_DEFAULT_LANGUAGE", json_cfg, dotenv_cfg, "ru")),
        startup_mode=_norm_lower(_pick("MMIS_START_MODE", json_cfg, dotenv_cfg, "api")),
        active_profile=active_profile,
        llm_default_provider=llm_provider,
        model_name=_norm_str(_pick("MMIS_MODEL_NAME", json_cfg, dotenv_cfg, "qcwind/qwen3-8b-instruct-Q4-K-M")),
        host=_norm_str(_pick("MMIS_API_HOST", json_cfg, dotenv_cfg, "127.0.0.1")),
        port=_to_int(_pick("MMIS_API_PORT", json_cfg, dotenv_cfg, 8040), default=8040),
        thinking_enabled=_to_bool(_pick("MMIS_THINKING_ENABLED", json_cfg, dotenv_cfg, False)),
        web_mode=_norm_lower(_pick("MMIS_WEB_MODE", json_cfg, dotenv_cfg, "auto")),
        json_mode_enabled=_to_bool(_pick("MMIS_JSON_MODE", json_cfg, dotenv_cfg, False)),
        internet_enabled=_to_bool(_pick("MMIS_INTERNET_ENABLED", json_cfg, dotenv_cfg, True)),
        automation_enabled=_to_bool(_pick("MMIS_AUTOMATION_ENABLED", json_cfg, dotenv_cfg, True)),
        screen_enabled=_to_bool(_pick("MMIS_SCREEN_ENABLED", json_cfg, dotenv_cfg, True)),
        voice_enabled=_to_bool(_pick("MMIS_VOICE_ENABLED", json_cfg, dotenv_cfg, True)),
        safety_mode=safety_mode,
        read_only_tools=(safety_mode != "allow_os_actions"),
        data_dir=DATA_DIR,
        models_dir=MODELS_DIR,
        memory_dir=memory_dir,
        cache_dir=Path(_norm_str(_pick("MMIS_CACHE_DIR", json_cfg, dotenv_cfg, str(dirs["cache"]))))
        .expanduser()
        .resolve(),
        log_dir=Path(_norm_str(_pick("MMIS_LOG_DIR", json_cfg, dotenv_cfg, str(dirs["logs"])))).expanduser().resolve(),
        db_path=Path(_norm_str(_pick("MMIS_DB_PATH", json_cfg, dotenv_cfg, db_default))).expanduser().resolve(),
        dialog_new_session_after_min=max(
            1,
            _to_int(_pick("MMIS_DIALOG_NEW_SESSION_AFTER_MIN", json_cfg, dotenv_cfg, 360), default=360),
        ),
        dialog_greeting_max_words=max(
            1,
            _to_int(_pick("MMIS_DIALOG_GREETING_MAX_WORDS", json_cfg, dotenv_cfg, 6), default=6),
        ),
        dialog_greeting_max_chars=max(
            8,
            _to_int(_pick("MMIS_DIALOG_GREETING_MAX_CHARS", json_cfg, dotenv_cfg, 35), default=35),
        ),
        dialog_greetings=_to_csv_list(
            _pick(
                "MMIS_DIALOG_GREETINGS",
                json_cfg,
                dotenv_cfg,
                "привет,приветик,здарова,здравствуйте,доброе утро,добрый день,добрый вечер,hi,hello,hey,yo",
            )
        ),
        dialog_greeting_exclusions=_to_csv_list(
            _pick(
                "MMIS_DIALOG_GREETING_EXCLUSIONS",
                json_cfg,
                dotenv_cfg,
                "слово привет,передай привет,передайте привет,приветствие,в коде привет,обсуждение слова привет,перевод привет",
            )
        ),
        config_file=config_file,
        feature_flags=_collect_feature_flags(json_cfg=json_cfg, dotenv_cfg=dotenv_cfg),
    )
    _validate_settings(settings)

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    _SETTINGS_CACHE = settings
    return settings


def load_settings(force_reload: bool = False) -> AppSettings:
    return load_config(force_reload=force_reload)


def _resolve_config_file() -> Path | None:
    env_raw = str(os.getenv("MMIS_CONFIG_FILE", "")).strip()
    if env_raw:
        path = Path(env_raw).expanduser()
        if not path.exists():
            raise ValueError(f"MMIS_CONFIG_FILE does not exist: {path}")
        return path
    default = BASE_DIR / "config" / "config.json"
    return default if default.exists() else None


def _resolve_dotenv_file() -> Path | None:
    env_raw = str(os.getenv("MMIS_DOTENV_FILE", "")).strip()
    if env_raw:
        path = Path(env_raw).expanduser()
        if not path.exists():
            raise ValueError(f"MMIS_DOTENV_FILE does not exist: {path}")
        return path
    default = BASE_DIR / ".env"
    return default if default.exists() else None


def _resolve_memory_path(*, json_cfg: dict[str, Any], dotenv_cfg: dict[str, str]) -> Path:
    raw = _pick("MMIS_MEMORY_DIR", json_cfg, dotenv_cfg, "")
    if _norm_str(raw):
        return Path(_norm_str(raw)).expanduser().resolve()
    return resolve_memory_dir().expanduser().resolve()


def _collect_feature_flags(*, json_cfg: dict[str, Any], dotenv_cfg: dict[str, str]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    ff_json = _json_get(json_cfg, "feature_flags", default={})
    if isinstance(ff_json, dict):
        for key, value in ff_json.items():
            row_key = _norm_lower(key)
            if row_key:
                result[row_key] = _to_bool(value)

    for key, value in dotenv_cfg.items():
        if key.startswith("MMIS_FF_"):
            result[_norm_lower(key.replace("MMIS_FF_", "", 1))] = _to_bool(value)
    for key, value in os.environ.items():
        if key.startswith("MMIS_FF_"):
            result[_norm_lower(key.replace("MMIS_FF_", "", 1))] = _to_bool(value)
    return result


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Invalid config json file: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain an object: {path}")
    return payload


def _read_dotenv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = str(raw_line or "").strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            k = key.strip()
            if not k:
                continue
            out[k] = _strip_quotes(value.strip())
    except Exception:
        return {}
    return out


def _strip_quotes(value: str) -> str:
    src = str(value or "")
    if len(src) >= 2 and ((src[0] == '"' and src[-1] == '"') or (src[0] == "'" and src[-1] == "'")):
        return src[1:-1]
    return src


def _to_csv_list(value) -> list[str]:
    if isinstance(value, list):
        raw_items = [str(x or "").strip() for x in value]
    else:
        raw = str(value or "").strip()
        raw_items = [x.strip() for x in raw.split(",")] if raw else []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if not item:
            continue
        low = item.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(item)
    return out


def _pick(env_key: str, json_cfg: dict[str, Any], dotenv_cfg: dict[str, str], default):
    if env_key in os.environ:
        return os.environ.get(env_key)
    if env_key in dotenv_cfg:
        return dotenv_cfg.get(env_key)

    json_key = _norm_lower(env_key.replace("MMIS_", "", 1))
    aliases = {
        "llm_provider": ("llm_default_provider", "default_provider"),
        "active_profile": ("profile",),
        "start_mode": ("mode",),
        "api_host": ("host",),
        "api_port": ("port",),
        "default_language": ("language",),
    }
    for key in (json_key, *aliases.get(json_key, ())):
        from_json = _json_get(json_cfg, key, default=None)
        if from_json is not None:
            return from_json
    return default


def _json_get(payload: dict[str, Any], key: str, default=None):
    keys = [
        key,
        key.lower(),
        key.upper(),
        f"app.{key}",
        f"llm.{key}",
        f"api.{key}",
        f"modules.{key}",
        f"paths.{key}",
    ]
    for full_key in keys:
        value = _lookup_dotted(payload, full_key)
        if value is not None:
            return value
    return default


def _lookup_dotted(payload: dict[str, Any], dotted: str):
    cur: Any = payload
    for part in str(dotted).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def _validate_settings(settings: AppSettings) -> None:
    errors: list[str] = []
    if settings.active_profile not in VALID_PROFILES:
        errors.append(f"MMIS_ACTIVE_PROFILE must be one of {sorted(VALID_PROFILES)}, got: {settings.active_profile}")
    if settings.llm_default_provider not in VALID_PROVIDERS:
        errors.append(f"MMIS_LLM_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got: {settings.llm_default_provider}")
    if settings.safety_mode not in VALID_SAFETY_MODES:
        errors.append(f"MMIS_SAFETY_MODE must be one of {sorted(VALID_SAFETY_MODES)}, got: {settings.safety_mode}")
    if not settings.host:
        errors.append("MMIS_API_HOST cannot be empty")
    if settings.port < 1 or settings.port > 65535:
        errors.append(f"MMIS_API_PORT must be in range 1..65535, got: {settings.port}")
    if not settings.model_name:
        errors.append("MMIS_MODEL_NAME cannot be empty")
    if not settings.startup_mode:
        errors.append("MMIS_START_MODE cannot be empty")
    if errors:
        raise ValueError("Invalid application settings:\n- " + "\n- ".join(errors))


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    raw = str(value).strip().lower()
    return raw in {"1", "true", "yes", "on", "y", "t"}


def _to_int(value, *, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _norm_str(value) -> str:
    return str(value or "").strip()


def _norm_lower(value) -> str:
    return _norm_str(value).lower()


def _norm_upper(value) -> str:
    return _norm_str(value).upper()
