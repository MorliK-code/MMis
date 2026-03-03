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
    app_name: str = "MMis"
    debug: bool = False
    locale: str = "ru_RU"
    default_language: str = "ru"
    startup_mode: str = "api"
    active_profile: str = "BALANCED"
    llm_default_provider: str = "ollama"
    model_name: str = "qcwind/qwen3-8b-instruct-Q4-K-M"
    host: str = "127.0.0.1"
    port: int = 8000
    thinking_enabled: bool = True
    web_mode: str = "auto"
    json_mode_enabled: bool = False
    internet_enabled: bool = True
    automation_enabled: bool = True
    screen_enabled: bool = True
    voice_enabled: bool = True
    safety_mode: str = "read_only_tools"
    read_only_tools: bool = True
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    memory_dir: Path = field(default_factory=lambda: DATA_DIR / "memory")
    cache_dir: Path = field(default_factory=lambda: Path(".cache").resolve())
    log_dir: Path = field(default_factory=lambda: Path("logs").resolve())
    db_path: Path = field(default_factory=lambda: DATA_DIR / "memory" / "memory.db")
    dialog_new_session_after_min: int = 360
    dialog_greeting_max_words: int = 6
    dialog_greeting_max_chars: int = 35
    dialog_greetings: list[str] = field(default_factory=list)
    dialog_greeting_exclusions: list[str] = field(default_factory=list)
    config_file: Path | None = None
    feature_flags: dict[str, bool] = field(default_factory=dict)
    
    # --- Consolidated Settings ---
    # Logging
    log_level: str = "INFO"
    log_file: Path | str | None = None
    log_colors: bool = True
    log_max_bytes: int = 10485760
    log_backup_count: int = 5
    
    # Metadata
    metadata_model: str = "qwen3:1.7b"
    metadata_model_fallbacks: list[str] = field(default_factory=list)
    
    # LLM Providers
    llm_max_tokens_lower_bound: int = 2048
    llm_max_tokens_upper_bound: int = 8192
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_sec: float = 120.0
    ollama_retries: int = 1
    openai_api_key: str = ""
    openai_api_url: str = "https://api.openai.com/v1"
    openai_timeout_sec: float = 120.0
    openai_max_retries: int = 2
    
    # Tools & Search
    search_api_url: str = "https://www.googleapis.com/customsearch/v1?key=API_KEY&cx=SEARCH_ENGINE_ID"
    
    # Voice
    voice_tts_voice: str = "ru-RU-DmitryNeural"
    voice_tts_rate: str = "+0%"
    voice_tts_volume: str = "+0%"
    voice_input_dir: Path | None = None
    voice_output_dir: Path | None = None
    
    # Chat & Memory
    short_memory_limit: int = 10
    chat_recall_results: int = 3
    chat_events_limit: int = 10
    chat_proofread: bool = False
    chat_proofread_strict: bool = False
    model_fallbacks: list[str] = field(default_factory=list)
    
    llm_max_tokens: int = 2048
    llm_max_tokens_lower_bound: int = 2048
    llm_max_tokens_upper_bound: int = 8192

    # Hardware
    gpu_vram_gb: int | None = None
    
    # UI Console
    console_model: str = ""
    console_timeout_sec: float = 2.5
    console_stream_timeout_sec: float = 600.0
    console_store_turn: bool = True
    console_show_thinking: bool = True  
    console_json_mode_enabled: bool = False
    console_auto_start_api: bool = True
    console_auto_start_ollama: bool = True

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}"

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
        port=_to_int(_pick("MMIS_API_PORT", json_cfg, dotenv_cfg, 8000), default=8000),
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
        log_level=str(_pick("MMIS_LOG_LEVEL", json_cfg, dotenv_cfg, "DEBUG" if _to_bool(_pick("MMIS_DEBUG", json_cfg, dotenv_cfg, False)) else "INFO")).strip().upper(),
        log_file=Path(str(_pick("MMIS_LOG_FILE", json_cfg, dotenv_cfg, ""))).expanduser() if str(_pick("MMIS_LOG_FILE", json_cfg, dotenv_cfg, "")).strip() else None,
        log_colors=_to_bool(_pick("MMIS_LOG_COLORS", json_cfg, dotenv_cfg, True)),
        log_max_bytes=max(262144, _to_int(_pick("MMIS_LOG_MAX_BYTES", json_cfg, dotenv_cfg, 10485760), default=10485760)),
        log_backup_count=max(1, _to_int(_pick("MMIS_LOG_BACKUP_COUNT", json_cfg, dotenv_cfg, 5), default=5)),
        metadata_model=str(_pick("MMIS_METADATA_MODEL", json_cfg, dotenv_cfg, "qwen3:1.7b")).strip(),
        metadata_model_fallbacks=_to_csv_list(_pick("MMIS_METADATA_MODEL_FALLBACKS", json_cfg, dotenv_cfg, "phi3:mini,llama3.2:1b")),
        llm_max_tokens_lower_bound=max(1, _to_int(_pick("MMIS_LLM_MAX_TOKENS_LOWER_BOUND", json_cfg, dotenv_cfg, llm_max_tokens_lower_bound), default=2048)),
        llm_max_tokens_upper_bound=max(1, _to_int(_pick("MMIS_LLM_MAX_TOKENS_UPPER_BOUND", json_cfg, dotenv_cfg, llm_max_tokens_upper_bound), default=8192)),
        ollama_base_url=str(_pick("OLLAMA_HOST", json_cfg, dotenv_cfg, "http://127.0.0.1:11434")).strip(),
        ollama_timeout_sec=float(_pick("OLLAMA_TIMEOUT_SEC", json_cfg, dotenv_cfg, 120.0)),
        ollama_retries=max(0, _to_int(_pick("OLLAMA_RETRIES", json_cfg, dotenv_cfg, 1), default=1)),
        openai_api_key=str(_pick("OPENAI_API_KEY", json_cfg, dotenv_cfg, "")).strip(),
        openai_api_url=str(_pick("OPENAI_BASE_URL", json_cfg, dotenv_cfg, "https://api.openai.com/v1")).strip(),
        openai_timeout_sec=float(_pick("OPENAI_TIMEOUT_SEC", json_cfg, dotenv_cfg, 120.0)),
        openai_max_retries=max(0, _to_int(_pick("OPENAI_MAX_RETRIES", json_cfg, dotenv_cfg, 2), default=2)),
        search_api_url=str(_pick("MMIS_SEARCH_API_URL", json_cfg, dotenv_cfg, "")).strip(),
        voice_tts_voice=str(_pick("MMIS_VOICE_TTS_VOICE", json_cfg, dotenv_cfg, "ru-RU-DmitryNeural")).strip(),
        voice_tts_rate=str(_pick("MMIS_VOICE_TTS_RATE", json_cfg, dotenv_cfg, "+0%")).strip(),
        voice_tts_volume=str(_pick("MMIS_VOICE_TTS_VOLUME", json_cfg, dotenv_cfg, "+0%")).strip(),
        voice_input_dir=Path(str(_pick("MMIS_VOICE_INPUT_DIR", json_cfg, dotenv_cfg, str(memory_dir / "voice" / "input")))).expanduser(),
        voice_output_dir=Path(str(_pick("MMIS_VOICE_OUTPUT_DIR", json_cfg, dotenv_cfg, str(memory_dir / "voice" / "output")))).expanduser(),
        short_memory_limit=_to_int(_pick("MMIS_SHORT_MEMORY_LIMIT", json_cfg, dotenv_cfg, 10), default=10),
        chat_recall_results=_to_int(_pick("MMIS_CHAT_RECALL_RESULTS", json_cfg, dotenv_cfg, 3), default=3),
        chat_events_limit=_to_int(_pick("MMIS_CHAT_EVENTS_LIMIT", json_cfg, dotenv_cfg, 10), default=10),
        chat_proofread=_to_bool(_pick("MMIS_CHAT_PROOFREAD", json_cfg, dotenv_cfg, False)),
        chat_proofread_strict=_to_bool(_pick("MMIS_CHAT_PROOFREAD_STRICT", json_cfg, dotenv_cfg, False)),
        model_fallbacks=_to_csv_list(_pick("MMIS_MODEL_FALLBACKS", json_cfg, dotenv_cfg, "")),
        gpu_vram_gb=_to_int_or_none(_pick("MMIS_GPU_VRAM_GB", json_cfg, dotenv_cfg, None)),
        console_model=str(_pick("MMIS_CONSOLE_MODEL", json_cfg, dotenv_cfg, "")).strip(),
        console_timeout_sec=float(_pick("MMIS_CONSOLE_TIMEOUT_SEC", json_cfg, dotenv_cfg, 2.5)),
        console_stream_timeout_sec=float(_pick("MMIS_CONSOLE_STREAM_TIMEOUT_SEC", json_cfg, dotenv_cfg, 600.0)),
        console_store_turn=_to_bool(_pick("MMIS_CONSOLE_STORE_TURN", json_cfg, dotenv_cfg, True)),
        console_show_thinking=_to_bool(_pick("MMIS_CONSOLE_SHOW_THINKING", json_cfg, dotenv_cfg, True)),
        console_json_mode_enabled=_to_bool(_pick("MMIS_CONSOLE_JSON_MODE", json_cfg, dotenv_cfg, False)),
        console_auto_start_api=_to_bool(_pick("MMIS_CONSOLE_AUTO_API", json_cfg, dotenv_cfg, True)),
        console_auto_start_ollama=_to_bool(_pick("MMIS_CONSOLE_AUTO_OLLAMA", json_cfg, dotenv_cfg, True)),
    )
    _validate_settings(settings)

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.memory_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if settings.voice_input_dir:
        settings.voice_input_dir.mkdir(parents=True, exist_ok=True)
    if settings.voice_output_dir:
        settings.voice_output_dir.mkdir(parents=True, exist_ok=True)

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


def _to_bool_or_none(value) -> bool | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw or raw == "none":
        return None
    return raw in {"1", "true", "yes", "on", "y", "t"}


def _to_int(value, *, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _to_int_or_none(value) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _norm_str(value) -> str:
    return str(value or "").strip()


def _norm_lower(value) -> str:
    return _norm_str(value).lower()


def _norm_upper(value) -> str:
    return _norm_str(value).upper()
