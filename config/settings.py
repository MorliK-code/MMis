from __future__ import annotations

import copy
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

from config.config_manager import ConfigManager


VALID_PROFILES = {"FAST", "BALANCED", "QUALITY", "ECONOM", "AUTONOMOUS", "ASYA"}
VALID_PROVIDERS = {"ollama", "auto", "openai"}
VALID_SAFETY_MODES = {"read_only_tools", "allow_os_actions"}


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
DEFAULT_MEMORY_DIR = DATA_DIR / "memory_storage"
LEGACY_MEMORY_DIR = BASE_DIR / "memory_storage"
CACHE_DIR = DEFAULT_MEMORY_DIR / "cache"
LEGACY_CACHE_DIR = DATA_DIR / "cache"
DIR_PATH_TOKEN = "{dir_path}"
_DIR_PATH_TOKEN_LOW = DIR_PATH_TOKEN.lower()


def _expand_dir_path_token(value: Any) -> str:
    src = str(value or "")
    if not src:
        return src
    low = src.lower()
    if _DIR_PATH_TOKEN_LOW not in low:
        return src
    out: list[str] = []
    idx = 0
    token_len = len(DIR_PATH_TOKEN)
    base_text = str(BASE_DIR)
    while idx < len(src):
        if low[idx : idx + token_len] == _DIR_PATH_TOKEN_LOW:
            out.append(base_text)
            idx += token_len
            continue
        out.append(src[idx])
        idx += 1
    return "".join(out)


def _resolve_path_value(value: Any, default: Path | None = None) -> Path:
    src = str(value or "").strip()
    if not src:
        if default is None:
            return BASE_DIR.resolve()
        return Path(default).expanduser().resolve()
    expanded = _expand_dir_path_token(src)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path)
    return path.resolve()


def _path_to_config_string(value: str | Path) -> str:
    path = Path(value).expanduser().resolve()
    try:
        rel = path.relative_to(BASE_DIR.resolve())
        rel_text = str(rel).replace("/", "\\")
        if not rel_text or rel_text == ".":
            return DIR_PATH_TOKEN
        return f"{DIR_PATH_TOKEN}\\{rel_text}"
    except Exception:
        return str(path)


def _from_env_path(name: str) -> Path | None:
    raw = str(os.getenv(name, "")).strip()
    return Path(raw).expanduser() if raw else None


def _resolve_memory_dir_default() -> Path:
    env_path = _from_env_path("MMIS_MEMORY_DIR")
    if env_path is not None:
        return env_path
    use_legacy = str(os.getenv("MMIS_USE_LEGACY_MEMORY_DIR", "")).strip().lower() in {"1", "true", "yes", "on"}
    if use_legacy:
        return LEGACY_MEMORY_DIR
    return DEFAULT_MEMORY_DIR


def _resolve_cache_dir_default(memory_dir: Path | None = None) -> Path:
    env_path = _from_env_path("MMIS_CACHE_DIR")
    if env_path is not None:
        return env_path
    if memory_dir is not None:
        root = Path(memory_dir).expanduser().resolve()
    else:
        root = _resolve_memory_dir_default().expanduser().resolve()
    return (root / "cache").resolve()


def resolve_memory_dir() -> Path:
    cache = globals().get("_SETTINGS_CACHE")
    if cache is not None and getattr(cache, "memory_dir", None):
        return _resolve_path_value(cache.memory_dir, _resolve_memory_dir_default())
    cfg_raw = str(os.getenv("MMIS_CONFIG_FILE", "")).strip()
    if cfg_raw:
        cfg_file = Path(cfg_raw).expanduser().resolve()
    else:
        cfg_file = (BASE_DIR / "config" / "config.json").resolve()
    try:
        if cfg_file.exists():
            payload = json.loads(cfg_file.read_text(encoding="utf-8-sig") or "{}")
            if isinstance(payload, dict):
                memory_row = payload.get("memory")
                value = memory_row.get("memory_dir") if isinstance(memory_row, dict) else None
                if str(value or "").strip():
                    return _resolve_path_value(value, _resolve_memory_dir_default())
    except Exception:
        pass
    return _resolve_path_value(None, _resolve_memory_dir_default())


MEMORY_DIR = resolve_memory_dir()


def ensure_dirs(memory_dir: str | Path | None = None) -> dict[str, Path]:
    mem_dir = _to_path(memory_dir, resolve_memory_dir()) if memory_dir is not None else resolve_memory_dir()
    cfg = globals().get("_SETTINGS_CACHE")
    default_cache_dir = _resolve_cache_dir_default(mem_dir)
    cache_dir = _to_path(cfg.cache_dir, default_cache_dir) if cfg is not None and cfg.cache_dir else default_cache_dir
    logs_dir = _to_path(cfg.log_dir, LOG_DIR.resolve()) if cfg is not None and cfg.log_dir else LOG_DIR.resolve()
    data_dir = _to_path(cfg.data_dir, DATA_DIR.resolve()) if cfg is not None and cfg.data_dir else DATA_DIR.resolve()
    models_dir = _to_path(cfg.models_dir, MODELS_DIR.resolve()) if cfg is not None and cfg.models_dir else MODELS_DIR.resolve()
    dirs = {
        "base": BASE_DIR,
        "config": CONFIG_DIR,
        "data": data_dir,
        "models": models_dir,
        "memory": mem_dir,
        "logs": logs_dir,
        "cache": cache_dir,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_join(base: str | Path, user_path: str | Path) -> Path:
    root = Path(base).expanduser().resolve()
    target = (root / Path(user_path)).resolve()
    try:
        target.relative_to(root)
    except Exception as exc:
        raise ValueError(f"Path escapes base directory: {target}") from exc
    return target


ROOT_DIR = BASE_DIR
LOGS_DIR = LOG_DIR
NEW_MEMORY_DIR = DEFAULT_MEMORY_DIR


def ensure_data_dirs() -> None:
    ensure_dirs()


@dataclass(frozen=True)
class GenerationProfile:
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class OllamaProfile:
    num_thread: int = 6
    num_ctx: int = 8192
    num_gpu: int = 1
    num_batch: int = 128
    keep_alive: str = "5m"


@dataclass(frozen=True)
class OpenAIProfile:
    model: str = ""
    reasoning_effort: str = "medium"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    generation: GenerationProfile
    ollama: OllamaProfile
    openai: OpenAIProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generation": asdict(self.generation),
            "ollama": asdict(self.ollama),
            "openai": asdict(self.openai),
        }


_LOG_FORMAT_DEFAULT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LOG_CHANNEL_PREFIXES_DEFAULT: dict[str, list[str]] = {
    "llm": ["llm"],
    "memory": ["memory", "metadata"],
    "tools": ["modules", "tools"],
    "ui": ["ui", "api", "ui_console", "ui_pyside6"],
    "web": ["web", "modules.internet.web", "tools.modules.internet"],
}
_LOG_WEB_TRACE_LOGGER_DEFAULT = "web.trace"
_LOG_SETUP_DONE = False


class _LoggerPrefixFilter(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self.prefixes = tuple(str(x or "").strip().lower() for x in prefixes if str(x or "").strip())

    def filter(self, record: logging.LogRecord) -> bool:
        name = str(getattr(record, "name", "") or "").strip().lower()
        if not name:
            return False
        for prefix in self.prefixes:
            if name == prefix or name.startswith(prefix + "."):
                return True
        return False


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
    port: int = 8027
    thinking_enabled: bool = True
    web_mode: str = "auto"
    web_v2: dict[str, Any] = field(default_factory=dict)
    json_mode_enabled: bool = False
    internet_enabled: bool = True
    automation_enabled: bool = True
    screen_enabled: bool = True
    voice_enabled: bool = True
    safety_mode: str = "read_only_tools"
    prompt_response_safety_filter_enabled: bool = False
    prompt_response_formatting_enabled: bool = True
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    memory_dir: Path = field(default_factory=lambda: DEFAULT_MEMORY_DIR.resolve())
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR.resolve())
    log_dir: Path = field(default_factory=lambda: Path("logs").resolve())
    db_path: Path = field(default_factory=lambda: DEFAULT_MEMORY_DIR.resolve() / "memory.db")
    dialog_new_session_after_min: int = 360
    dialog_greeting_max_words: int = 6
    dialog_greeting_max_chars: int = 35
    dialog_greetings: list[str] = field(default_factory=list)
    dialog_greeting_exclusions: list[str] = field(default_factory=list)
    config_file: Path | None = None
    feature_flags: dict[str, bool] = field(default_factory=dict)

    # Logging
    log_level: str = "INFO"
    log_file: Path | str | None = None
    log_colors: bool = True
    log_max_bytes: int = 10485760
    log_backup_count: int = 5
    log_format: str = _LOG_FORMAT_DEFAULT
    log_channels: dict[str, list[str]] = field(default_factory=lambda: copy.deepcopy(_LOG_CHANNEL_PREFIXES_DEFAULT))
    log_web_trace_enabled: bool = True
    log_web_trace_logger: str = _LOG_WEB_TRACE_LOGGER_DEFAULT

    # Metadata
    metadata_model: str = "qwen3:1.7b"
    metadata_model_fallbacks: list[str] = field(default_factory=list)

    # LLM Providers
    llm_max_tokens_lower_bound: int = 2048
    llm_max_tokens_upper_bound: int = 8192
    llm_profiles: dict[str, Any] = field(default_factory=dict)
    task_model_profiles: dict[str, Any] = field(default_factory=dict)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_sec: float = 120.0
    ollama_retries: int = 1
    openai_api_key: str = ""
    openai_api_url: str = "https://api.openai.com/v1"
    openai_timeout_sec: float = 120.0
    openai_max_retries: int = 2

    # Tools & Search
    search_api_url: str = "http://127.0.0.1:8080/search?format=json"
    search_provider: str = "searxng"
    search_strict_endpoint: bool = True
    search_timeout_sec: float = 12.0
    web_fetch_timeout_sec: int = 12
    web_fetch_retries: int = 1
    web_clean_max_chars: int = 4000
    web_clean_min_chars: int = 200
    web_clean_language_hint: str = ""

    # Voice
    voice_tts_voice: str = "ru-RU-DmitryNeural"
    voice_tts_rate: str = "+0%"
    voice_tts_volume: str = "+0%"
    voice_input_dir: Path | None = None
    voice_output_dir: Path | None = None

    # Chat & Memory
    chat_recall_results: int = 3
    chat_events_limit: int = 10
    chat_proofread: bool = False
    chat_proofread_strict: bool = False
    memory_facts_scope: str = "user_only"
    memory_include_pending_facts_in_retrieval: bool = False
    memory_confirmation_ttl_sec: int = 300
    memory_migration_auto_on_start: bool = True
    memory_migration_schema_version: int = 2
    memory_version: str = "v2"
    memory_backend: str = "chroma"
    memory_embedding_backend: str = "sentence_transformers"
    memory_embedding_model: str = "all-MiniLM-L6-v2"
    memory_embedding_dim: int = 384
    memory_retrieval_top_k: int = 8
    memory_rerank_top_k: int = 8
    memory_retrieval_weight_semantic_similarity: float = 0.34
    memory_retrieval_weight_lexical_score: float = 0.25
    memory_retrieval_weight_recency_score: float = 0.10
    memory_retrieval_weight_importance_score: float = 0.09
    memory_retrieval_weight_confidence_score: float = 0.08
    memory_retrieval_weight_entity_overlap_score: float = 0.07
    memory_retrieval_weight_exact_match_boost: float = 0.04
    memory_retrieval_weight_scope_match_score: float = 0.03
    memory_chunk_size: int = 1200
    memory_chunk_overlap: int = 160
    memory_summary_trigger: int = 60
    memory_summary_target_tokens: int = 220
    memory_context_budget_total: int = 2200
    memory_context_budget_memory: int = 700
    memory_context_budget_docs: int = 600
    memory_context_budget_tools: int = 220
    memory_context_budget_response_reserve: int = 260
    memory_stale_after_days: int = 30
    memory_archive_after_days: int = 90
    memory_promotion_message_importance_threshold: float = 0.55
    memory_promotion_message_confidence_threshold: float = 0.50
    memory_promotion_project_signal_boost: float = 0.12
    memory_promotion_fact_signal_boost: float = 0.16
    memory_promotion_decision_signal_boost: float = 0.12
    memory_promotion_smalltalk_penalty: float = 0.20
    memory_importance_weight_base: float = 0.42
    memory_importance_weight_decision: float = 0.24
    memory_importance_weight_remember: float = 0.18
    memory_importance_weight_project: float = 0.10
    memory_salience_weight_novelty: float = 0.22
    memory_salience_weight_permanence: float = 0.20
    memory_salience_weight_repetition: float = 0.14
    memory_salience_weight_project_relevance: float = 0.16
    memory_salience_weight_task_relevance: float = 0.16
    memory_salience_weight_explicit_save_signal: float = 0.12
    memory_temporary_ttl_sec: int = 3600
    memory_private_runtime_ttl_sec: int = 900
    memory_working_limit: int = 120
    model_fallbacks: list[str] = field(default_factory=list)

    # Hardware
    gpu_vram_gb: int | None = None

    # UI Console
    console_timeout_sec: float = 2.5
    console_stream_timeout_sec: float = 600.0
    console_store_turn: bool = True
    console_show_thinking: bool = False
    console_thinking_first: bool = True
    console_auto_start_api: bool = True
    console_auto_start_ollama: bool = True
    console_runtime: dict[str, Any] = field(default_factory=dict)

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key in (
            "data_dir",
            "models_dir",
            "memory_dir",
            "cache_dir",
            "log_dir",
            "db_path",
            "config_file",
            "voice_input_dir",
            "voice_output_dir",
            "log_file",
        ):
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

    manager = get_config_manager()
    bootstrap_seed = None
    if not manager.path.exists():
        dotenv_file = _resolve_dotenv_file()
        dotenv_cfg = _read_dotenv_file(dotenv_file) if dotenv_file is not None else {}
        bootstrap_seed = _bootstrap_seed_from_env(dotenv_cfg=dotenv_cfg)
    payload = manager.load_or_create(bootstrap_seed=bootstrap_seed)

    settings = _settings_from_payload(payload=payload, config_file=manager.path)
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


def get_config_manager() -> ConfigManager:
    cfg_path = _resolve_config_file()
    return ConfigManager(path=cfg_path, defaults=_default_config_tree(), project_dir=BASE_DIR)


def get_config_payload(force_reload: bool = False) -> dict[str, Any]:
    manager = get_config_manager()
    bootstrap_seed = None
    if force_reload:
        global _SETTINGS_CACHE
        _SETTINGS_CACHE = None
    if not manager.path.exists():
        dotenv_file = _resolve_dotenv_file()
        dotenv_cfg = _read_dotenv_file(dotenv_file) if dotenv_file is not None else {}
        bootstrap_seed = _bootstrap_seed_from_env(dotenv_cfg=dotenv_cfg)
    return manager.load_or_create(bootstrap_seed=bootstrap_seed)


def update_config_value(dotted_path: str, value: Any) -> AppSettings:
    manager = get_config_manager()
    manager.update(str(dotted_path or ""), value)
    return load_config(force_reload=True)


def update_config_values(updates: dict[str, Any]) -> AppSettings:
    manager = get_config_manager()
    manager.update_many(dict(updates or {}))
    return load_config(force_reload=True)


def _default_model_profiles_tree() -> dict[str, Any]:
    return {
        "FAST": {
            "name": "FAST",
            "generation": {
                "temperature": 0.55,
                "top_p": 0.9,
                "repeat_penalty": 1.05,
                "max_tokens": 512,
                "stop": [],
            },
            "ollama": {
                "num_thread": 8,
                "num_ctx": 4096,
                "num_gpu": 1,
                "num_batch": 64,
                "keep_alive": "2m",
            },
            "openai": {
                "model": "",
                "reasoning_effort": "low",
            },
        },
        "BALANCED": {
            "name": "BALANCED",
            "generation": {
                "temperature": 0.7,
                "top_p": 0.92,
                "repeat_penalty": 1.1,
                "max_tokens": 1024,
                "stop": [],
            },
            "ollama": {
                "num_thread": 6,
                "num_ctx": 8192,
                "num_gpu": 1,
                "num_batch": 128,
                "keep_alive": "5m",
            },
            "openai": {
                "model": "",
                "reasoning_effort": "medium",
            },
        },
        "QUALITY": {
            "name": "QUALITY",
            "generation": {
                "temperature": 0.82,
                "top_p": 0.95,
                "repeat_penalty": 1.2,
                "max_tokens": 2048,
                "stop": [],
            },
            "ollama": {
                "num_thread": 6,
                "num_ctx": 12288,
                "num_gpu": 1,
                "num_batch": 160,
                "keep_alive": "10m",
            },
            "openai": {
                "model": "",
                "reasoning_effort": "high",
            },
        },
        "ECONOM": {
            "name": "ECONOM",
            "generation": {
                "temperature": 0.45,
                "top_p": 0.88,
                "repeat_penalty": 1.12,
                "max_tokens": 384,
                "stop": [],
            },
            "ollama": {
                "num_thread": 4,
                "num_ctx": 3072,
                "num_gpu": 0,
                "num_batch": 48,
                "keep_alive": "1m",
            },
            "openai": {
                "model": "",
                "reasoning_effort": "low",
            },
        },
        "ASYA": {
            "name": "ASYA",
            "generation": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "max_tokens": -1,
                "stop": [],
            },
            "ollama": {
                "num_thread": 6,
                "num_ctx": 8192,
                "num_gpu": 1,
                "num_batch": 128,
                "keep_alive": "5m",
            },
            "openai": {
                "model": "",
                "reasoning_effort": "medium",
            },
        },
        "AUTONOMOUS": {
            "name": "AUTONOMOUS",
            "generation": {
                "temperature": 0.6,
                "top_p": 0.9,
                "repeat_penalty": 1.08,
                "max_tokens": -1,
                "stop": [],
            },
            "ollama": {
                "num_thread": 6,
                "num_ctx": 16384,
                "num_gpu": 1,
                "num_batch": 192,
                "keep_alive": "15m",
            },
            "openai": {
                "model": "",
                "reasoning_effort": "high",
            },
        },
    }


def _default_task_model_profiles_tree(
    *,
    provider: str = "ollama",
    model: str = "qcwind/qwen3-8b-instruct-Q4-K-M",
) -> dict[str, Any]:
    provider_name = _norm_lower(provider) or "ollama"
    model_name = _norm_str(model) or "qcwind/qwen3-8b-instruct-Q4-K-M"
    return {
        "emotion": {
            "name": "emotion",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.15,
            "max_tokens": 128,
            "timeout": 20.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "tagging": {
            "name": "tagging",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.10,
            "max_tokens": 128,
            "timeout": 20.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "intent_judge": {
            "name": "intent_judge",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.10,
            "max_tokens": 160,
            "timeout": 20.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "query_rewrite": {
            "name": "query_rewrite",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.20,
            "max_tokens": 192,
            "timeout": 25.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "source_relevance": {
            "name": "source_relevance",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.10,
            "max_tokens": 160,
            "timeout": 20.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "summary_mini_pass": {
            "name": "summary_mini_pass",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.20,
            "max_tokens": 96,
            "timeout": 20.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "studio_seed_extract": {
            "name": "studio_seed_extract",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.35,
            "max_tokens": 540,
            "timeout": 45.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "studio_options": {
            "name": "studio_options",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.55,
            "max_tokens": 220,
            "timeout": 25.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "studio_pack_blueprint": {
            "name": "studio_pack_blueprint",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.55,
            "max_tokens": 4096,
            "timeout": 90.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "fact_filter": {
            "name": "fact_filter",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.10,
            "max_tokens": 192,
            "timeout": 20.0,
            "enabled": True,
            "fallback_profile": "",
        },
        "final_response": {
            "name": "final_response",
            "provider": provider_name,
            "model": model_name,
            "temperature": 0.70,
            "max_tokens": 1024,
            "timeout": 120.0,
            "enabled": True,
            "fallback_profile": "",
        },
    }


def _default_config_tree() -> dict[str, Any]:
    memory_dir = _resolve_memory_dir_default().expanduser().resolve()
    cache_dir = _resolve_cache_dir_default(memory_dir).resolve()
    log_dir = LOG_DIR.resolve()
    return {
        "app": {
            "name": "MMis",
            "debug": False,
            "locale": "ru_RU",
            "default_language": "ru",
        },
        "startup": {
            "mode": "api",
            "active_profile": "BALANCED",
            "safety_mode": "read_only_tools",
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8027,
        },
        "llm": {
            "provider": "ollama",
            "model_name": "qcwind/qwen3-8b-instruct-Q4-K-M",
            "model_fallbacks": [],
            "thinking_enabled": True,
            "json_mode_enabled": False,
            "profiles": _default_model_profiles_tree(),
            "task_models": _default_task_model_profiles_tree(),
            "max_tokens": {
                "lower_bound": 2048,
                "upper_bound": 8192,
            },
            "metadata": {
                "model": "qwen3:1.7b",
                "fallbacks": [],
            },
            "providers": {
                "ollama": {
                    "base_url": "http://127.0.0.1:11434",
                    "timeout_sec": 120.0,
                    "retries": 1,
                },
                "openai": {
                    "api_key": "",
                    "api_url": "https://api.openai.com/v1",
                    "timeout_sec": 120.0,
                    "max_retries": 2,
                },
            },
        },
        "internet": {
            "enabled": True,
            "web_mode": "auto",
            "web_v2": {
                "enabled": True,
                "thresholds": {
                    "no_search_max": 0.26,
                    "verify_max": 0.46,
                    "soft_max": 0.66,
                    "targeted_max": 0.84,
                },
                "weights": {
                    "base": 0.14,
                    "external_fact": 0.42,
                    "mixed_query": 0.24,
                    "ambiguous_query": 0.22,
                    "temporal_risk": 0.48,
                    "confidence_penalty": 0.45,
                    "freshness_required": 0.24,
                    "stakes_medium": 0.08,
                    "stakes_high": 0.22,
                    "explicit_search_intent": 0.24,
                    "force_keyword_boost": 0.34,
                    "local_scope_penalty": 0.50,
                    "local_scope_depth_cap_threshold": 0.33,
                    "category_penalty_default": 0.36,
                },
                "budgets": {
                    "no_search": {"max_queries": 0, "max_sources": 0, "max_pages": 0, "max_fetches": 0},
                    "verify_only": {"max_queries": 1, "max_sources": 2, "max_pages": 1, "max_fetches": 1},
                    "soft_search": {"max_queries": 2, "max_sources": 3, "max_pages": 2, "max_fetches": 2},
                    "targeted_search": {"max_queries": 3, "max_sources": 5, "max_pages": 3, "max_fetches": 3},
                    "deep_search": {"max_queries": 5, "max_sources": 8, "max_pages": 5, "max_fetches": 5},
                },
                "cooldown_seconds": 45,
                "retry_policy": {
                    "default_attempts": 1,
                    "verify_only": 1,
                    "soft_search": 1,
                    "targeted_search": 2,
                    "deep_search": 2,
                    "backoff_ms": 250,
                },
                "continuation": {
                    "ttl_minutes": 20,
                    "max_user_turns": 6,
                    "short_followup_max_tokens": 9,
                },
                "evidence_quality": {
                    "min_score": 0.46,
                    "min_usable_results": 2,
                    "min_unique_domains": 2,
                    "min_trusted_count": 1,
                },
                "force_search_keywords": [
                    "сейчас",
                    "актуально",
                    "latest",
                    "today",
                    "2026",
                    "последняя версия",
                    "цена",
                    "сколько стоит",
                    "новости",
                ],
                "never_search_categories": ["reasoning", "architecture", "rewrite"],
                "category_penalties": {
                    "reasoning": 0.32,
                    "architecture": 0.30,
                    "refactor": 0.35,
                    "local": 0.42,
                },
                "ttl_days": {
                    "default": 7,
                    "prices": 1,
                    "versions": 14,
                    "news": 2,
                    "docs_summary": 30,
                    "market_compare": 7,
                },
                "preferred_domains": [],
                "blocked_domains": [],
                "trust_policy": {
                    "trusted_allowlist": [],
                    "preferred_domains": [],
                    "blocked_domains": [],
                    "risky_domains": [
                        "medium.com",
                        "substack.com",
                    ],
                    "degraded_domains": [
                        "blogspot.com",
                    ],
                    "trusted_allowlist_by_category": {
                        "docs": [
                            "docs.python.org",
                            "developer.mozilla.org",
                            "openai.com",
                            "docs.docker.com",
                            "kubernetes.io",
                        ],
                        "finance": [
                            "bank.gov.ua",
                        ],
                        "news": [
                            "reuters.com",
                            "apnews.com",
                        ],
                        "generic": [],
                    },
                    "preferred_domains_by_category": {
                        "docs": [
                            "github.com",
                        ],
                        "finance": [
                            "minfin.com.ua",
                            "finance.ua",
                        ],
                        "weather": [
                            "sinoptik.ua",
                            "meteo.ua",
                        ],
                        "news": [
                            "ukrinform.ua",
                        ],
                        "generic": [],
                    },
                    "manual_overrides": {
                        "trusted": [],
                        "preferred": [],
                        "blocked": [],
                        "risky": [],
                        "degraded": [],
                    },
                },
                "citations": {
                    "enabled": True,
                    "style": "compact",
                    "max_items_fact": 1,
                    "max_items_compare": 3,
                    "max_items_news": 2,
                    "include_full_urls_on_request": True,
                },
            },
            "search": {
                "api_url": "http://127.0.0.1:8080/search?format=json",
                "provider": "searxng",
                "strict_endpoint": True,
                "timeout_sec": 12.0,
            },
            "fetch": {
                "timeout_sec": 12,
                "retries": 1,
                "clean_max_chars": 4000,
                "clean_min_chars": 200,
                "clean_language_hint": "",
            },
        },
        "modules": {
            "automation_enabled": True,
            "screen_enabled": True,
        },
        "prompt": {
            "response_safety_filter_enabled": False,
            "response_formatting_enabled": True,
        },
        "memory": {
            "memory_dir": _path_to_config_string(memory_dir),
            "cache_dir": _path_to_config_string(cache_dir),
            "log_dir": _path_to_config_string(log_dir),
            "db_path": _path_to_config_string(memory_dir / "memory.db"),
            "version": "v2",
            "backend": "chroma",
            "embedding": {
                "backend": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
                "dim": 384,
            },
            "retrieval": {
                "top_k": 8,
                "rerank_top_k": 8,
                "fusion_weights": {
                    "semantic_similarity": 0.34,
                    "lexical_score": 0.25,
                    "recency_score": 0.10,
                    "importance_score": 0.09,
                    "confidence_score": 0.08,
                    "entity_overlap_score": 0.07,
                    "exact_match_boost": 0.04,
                    "scope_match_score": 0.03,
                },
            },
            "documents": {
                "chunk_size": 1200,
                "chunk_overlap": 160,
            },
            "summary": {
                "trigger": 60,
                "target_tokens": 220,
            },
            "context_budget": {
                "total": 2200,
                "memory": 700,
                "docs": 600,
                "tools": 220,
                "response_reserve": 260,
            },
            "lifecycle": {
                "stale_after_days": 30,
                "archive_after_days": 90,
                "promotion_thresholds": {
                    "message_importance": 0.55,
                    "message_confidence": 0.50,
                },
                "promotion_signal_boosts": {
                    "project": 0.12,
                    "fact": 0.16,
                    "decision": 0.12,
                    "smalltalk_penalty": 0.20,
                },
                "temporary_ttl_sec": 3600,
                "private_runtime_ttl_sec": 900,
                "working_limit": 120,
            },
            "scoring": {
                "importance_weights": {
                    "base": 0.42,
                    "decision": 0.24,
                    "remember": 0.18,
                    "project": 0.10,
                },
                "salience_weights": {
                    "novelty": 0.22,
                    "permanence": 0.20,
                    "repetition": 0.14,
                    "project_relevance": 0.16,
                    "task_relevance": 0.16,
                    "explicit_save_signal": 0.12,
                },
            },
            "chat_recall_results": 3,
            "chat_events_limit": 10,
            "chat_proofread": False,
            "chat_proofread_strict": False,
            "facts_scope": "user_only",
            "include_pending_facts_in_retrieval": False,
            "confirmation_ttl_sec": 300,
            "migration": {
                "auto_on_start": True,
                "schema_version": 2,
            },
        },
        "dialog": {
            "new_session_after_min": 360,
            "greeting_max_words": 6,
            "greeting_max_chars": 35,
            "greetings": [],
            "greeting_exclusions": [],
        },
        "voice": {
            "enabled": True,
            "tts": {
                "voice": "ru-RU-DmitryNeural",
                "rate": "+0%",
                "volume": "+0%",
            },
            "paths": {
                "input_dir": _path_to_config_string(memory_dir / "voice" / "input"),
                "output_dir": _path_to_config_string(memory_dir / "voice" / "output"),
            },
        },
        "logging": {
            "level": "INFO",
            "file": "",
            "colors": True,
            "max_bytes": 10485760,
            "backup_count": 5,
            "format": _LOG_FORMAT_DEFAULT,
            "channels": copy.deepcopy(_LOG_CHANNEL_PREFIXES_DEFAULT),
            "web_trace_enabled": True,
            "web_trace_logger": _LOG_WEB_TRACE_LOGGER_DEFAULT,
        },
        "ui": {
            "console": {
                "timeout_sec": 2.5,
                "stream_timeout_sec": 600.0,
                "store_turn": True,
                "show_thinking": False,
                "thinking_first": True,
                "auto_start_api": True,
                "auto_start_ollama": True,
                "runtime": {
                    "mode_lock": False,
                    "active_mode": "",
                    "output_parameters": False,
                    "output_summary": False,
                },
            },
        },
        "features": {
            "flags": {},
            "extra_legacy": {},
        },
        "hardware": {
            "gpu_vram_gb": None,
        },
        "paths": {
            "data_dir": _path_to_config_string(DATA_DIR),
            "models_dir": _path_to_config_string(MODELS_DIR),
        },
    }


def _settings_from_payload(payload: dict[str, Any], *, config_file: Path) -> AppSettings:
    row = dict(payload or {})

    memory_dir = _to_path(_get_dotted(row, "memory.memory_dir"), _resolve_memory_dir_default().expanduser().resolve())
    dirs = ensure_dirs(memory_dir=memory_dir)
    cache_dir = _to_path(_get_dotted(row, "memory.cache_dir"), dirs["cache"])
    if cache_dir.resolve() == LEGACY_CACHE_DIR.resolve():
        cache_dir = _resolve_cache_dir_default(memory_dir).resolve()
    log_dir = _to_path(_get_dotted(row, "memory.log_dir"), dirs["logs"])
    db_path = _to_path(_get_dotted(row, "memory.db_path"), memory_dir / "memory.db")

    voice_input = _to_path(_get_dotted(row, "voice.paths.input_dir"), memory_dir / "voice" / "input")
    voice_output = _to_path(_get_dotted(row, "voice.paths.output_dir"), memory_dir / "voice" / "output")

    log_file_raw = _get_dotted(row, "logging.file")
    log_file = None
    if str(log_file_raw or "").strip():
        log_file = Path(str(log_file_raw)).expanduser()

    safety_mode = _norm_lower(_pick_value(_get_dotted(row, "startup.safety_mode"), "read_only_tools"))
    llm_default_provider = _norm_lower(_get_dotted(row, "llm.provider") or "ollama")
    model_name = _norm_str(_get_dotted(row, "llm.model_name") or "qcwind/qwen3-8b-instruct-Q4-K-M")

    runtime = _normalize_ui_console_runtime(_as_dict(_get_dotted(row, "ui.console.runtime")))
    features_flags = _as_dict(_get_dotted(row, "features.flags"))
    profile_rows = _as_dict(_get_dotted(row, "llm.profiles"))
    if not profile_rows:
        profile_rows = copy.deepcopy(_default_model_profiles_tree())
    profile_rows = _normalize_profile_rows(profile_rows)
    task_model_rows = _as_dict(_get_dotted(row, "llm.task_models"))
    if not task_model_rows:
        task_model_rows = copy.deepcopy(
            _default_task_model_profiles_tree(provider=llm_default_provider, model=model_name)
        )
    task_model_rows = _normalize_task_model_profile_rows(
        task_model_rows,
        default_provider=llm_default_provider,
        default_model=model_name,
    )
    log_channels = _normalize_log_channels(_as_dict(_get_dotted(row, "logging.channels")))

    settings = AppSettings(
        app_name=_norm_str(_get_dotted(row, "app.name") or "MMis"),
        debug=_to_bool(_get_dotted(row, "app.debug")),
        locale=_norm_str(_get_dotted(row, "app.locale") or "ru_RU"),
        default_language=_norm_str(_get_dotted(row, "app.default_language") or "ru"),
        startup_mode=_norm_lower(_get_dotted(row, "startup.mode") or "api"),
        active_profile=_norm_upper(_get_dotted(row, "startup.active_profile") or "BALANCED"),
        llm_default_provider=llm_default_provider,
        model_name=model_name,
        host=_norm_str(_get_dotted(row, "api.host") or "127.0.0.1"),
        port=_to_int(_get_dotted(row, "api.port"), default=8027),
        thinking_enabled=_to_bool(_get_dotted(row, "llm.thinking_enabled")),
        web_mode=_norm_lower(_get_dotted(row, "internet.web_mode") or "auto"),
        web_v2=_as_dict(_get_dotted(row, "internet.web_v2")),
        json_mode_enabled=_to_bool(_get_dotted(row, "llm.json_mode_enabled")),
        internet_enabled=_to_bool(_get_dotted(row, "internet.enabled")),
        automation_enabled=_to_bool(_get_dotted(row, "modules.automation_enabled")),
        screen_enabled=_to_bool(_get_dotted(row, "modules.screen_enabled")),
        voice_enabled=_to_bool(_get_dotted(row, "voice.enabled")),
        safety_mode=safety_mode,
        prompt_response_safety_filter_enabled=_to_bool(_get_dotted(row, "prompt.response_safety_filter_enabled")),
        prompt_response_formatting_enabled=_to_bool(_get_dotted(row, "prompt.response_formatting_enabled")),
        data_dir=_to_path(_get_dotted(row, "paths.data_dir"), DATA_DIR),
        models_dir=_to_path(_get_dotted(row, "paths.models_dir"), MODELS_DIR),
        memory_dir=memory_dir,
        cache_dir=cache_dir,
        log_dir=log_dir,
        db_path=db_path,
        dialog_new_session_after_min=max(1, _to_int(_get_dotted(row, "dialog.new_session_after_min"), default=360)),
        dialog_greeting_max_words=max(1, _to_int(_get_dotted(row, "dialog.greeting_max_words"), default=6)),
        dialog_greeting_max_chars=max(8, _to_int(_get_dotted(row, "dialog.greeting_max_chars"), default=35)),
        dialog_greetings=_to_csv_list(_get_dotted(row, "dialog.greetings")),
        dialog_greeting_exclusions=_to_csv_list(_get_dotted(row, "dialog.greeting_exclusions")),
        config_file=Path(config_file).expanduser().resolve(),
        feature_flags={str(k): _to_bool(v) for k, v in features_flags.items()},
        log_level=_norm_upper(_get_dotted(row, "logging.level") or "INFO"),
        log_file=log_file,
        log_colors=_to_bool(_get_dotted(row, "logging.colors")),
        log_max_bytes=max(262144, _to_int(_get_dotted(row, "logging.max_bytes"), default=10485760)),
        log_backup_count=max(1, _to_int(_get_dotted(row, "logging.backup_count"), default=5)),
        log_format=_norm_str(_get_dotted(row, "logging.format") or _LOG_FORMAT_DEFAULT),
        log_channels=log_channels,
        log_web_trace_enabled=_to_bool(_pick_value(_get_dotted(row, "logging.web_trace_enabled"), True)),
        log_web_trace_logger=_norm_str(_get_dotted(row, "logging.web_trace_logger") or _LOG_WEB_TRACE_LOGGER_DEFAULT),
        metadata_model=_norm_str(_get_dotted(row, "llm.metadata.model") or "qwen3:1.7b"),
        metadata_model_fallbacks=_to_csv_list(_get_dotted(row, "llm.metadata.fallbacks")),
        llm_max_tokens_lower_bound=max(1, _to_int(_get_dotted(row, "llm.max_tokens.lower_bound"), default=2048)),
        llm_max_tokens_upper_bound=max(1, _to_int(_get_dotted(row, "llm.max_tokens.upper_bound"), default=8192)),
        llm_profiles=profile_rows,
        task_model_profiles=task_model_rows,
        ollama_base_url=_norm_str(_get_dotted(row, "llm.providers.ollama.base_url") or "http://127.0.0.1:11434"),
        ollama_timeout_sec=float(_pick_value(_get_dotted(row, "llm.providers.ollama.timeout_sec"), 120.0)),
        ollama_retries=max(0, _to_int(_get_dotted(row, "llm.providers.ollama.retries"), default=1)),
        openai_api_key=_norm_str(_get_dotted(row, "llm.providers.openai.api_key")),
        openai_api_url=_norm_str(_get_dotted(row, "llm.providers.openai.api_url") or "https://api.openai.com/v1"),
        openai_timeout_sec=float(_pick_value(_get_dotted(row, "llm.providers.openai.timeout_sec"), 120.0)),
        openai_max_retries=max(0, _to_int(_get_dotted(row, "llm.providers.openai.max_retries"), default=2)),
        search_api_url=_norm_str(_get_dotted(row, "internet.search.api_url") or "http://127.0.0.1:8080/search?format=json"),
        search_provider=_norm_lower(_get_dotted(row, "internet.search.provider") or "searxng"),
        search_strict_endpoint=_to_bool(_get_dotted(row, "internet.search.strict_endpoint")),
        search_timeout_sec=max(3.0, float(_pick_value(_get_dotted(row, "internet.search.timeout_sec"), 12.0))),
        web_fetch_timeout_sec=max(3, _to_int(_get_dotted(row, "internet.fetch.timeout_sec"), default=12)),
        web_fetch_retries=max(0, _to_int(_get_dotted(row, "internet.fetch.retries"), default=1)),
        web_clean_max_chars=max(256, _to_int(_get_dotted(row, "internet.fetch.clean_max_chars"), default=4000)),
        web_clean_min_chars=max(40, _to_int(_get_dotted(row, "internet.fetch.clean_min_chars"), default=200)),
        web_clean_language_hint=_norm_str(_get_dotted(row, "internet.fetch.clean_language_hint")),
        voice_tts_voice=_norm_str(_get_dotted(row, "voice.tts.voice") or "ru-RU-DmitryNeural"),
        voice_tts_rate=_norm_str(_get_dotted(row, "voice.tts.rate") or "+0%"),
        voice_tts_volume=_norm_str(_get_dotted(row, "voice.tts.volume") or "+0%"),
        voice_input_dir=voice_input,
        voice_output_dir=voice_output,
        chat_recall_results=max(1, _to_int(_get_dotted(row, "memory.chat_recall_results"), default=3)),
        chat_events_limit=max(1, _to_int(_get_dotted(row, "memory.chat_events_limit"), default=10)),
        chat_proofread=_to_bool(_get_dotted(row, "memory.chat_proofread")),
        chat_proofread_strict=_to_bool(_get_dotted(row, "memory.chat_proofread_strict")),
        memory_facts_scope=_norm_lower(_pick_value(_get_dotted(row, "memory.facts_scope"), "user_only")),
        memory_include_pending_facts_in_retrieval=_to_bool(
            _pick_value(_get_dotted(row, "memory.include_pending_facts_in_retrieval"), False)
        ),
        memory_confirmation_ttl_sec=max(1, _to_int(_pick_value(_get_dotted(row, "memory.confirmation_ttl_sec"), 300), default=300)),
        memory_migration_auto_on_start=_to_bool(
            _pick_value(_get_dotted(row, "memory.migration.auto_on_start"), True)
        ),
        memory_migration_schema_version=max(
            1,
            _to_int(_pick_value(_get_dotted(row, "memory.migration.schema_version"), 2), default=2),
        ),
        memory_version=_norm_lower(_pick_value(_get_dotted(row, "memory.version"), "v2")),
        memory_backend=_norm_lower(_pick_value(_get_dotted(row, "memory.backend"), "chroma")),
        memory_embedding_backend=_norm_lower(
            _pick_value(_get_dotted(row, "memory.embedding.backend"), "sentence_transformers")
        ),
        memory_embedding_model=_norm_str(
            _pick_value(_get_dotted(row, "memory.embedding.model"), "all-MiniLM-L6-v2")
        ),
        memory_embedding_dim=max(32, _to_int(_pick_value(_get_dotted(row, "memory.embedding.dim"), 384), default=384)),
        memory_retrieval_top_k=max(1, _to_int(_pick_value(_get_dotted(row, "memory.retrieval.top_k"), 8), default=8)),
        memory_rerank_top_k=max(1, _to_int(_pick_value(_get_dotted(row, "memory.retrieval.rerank_top_k"), 8), default=8)),
        memory_retrieval_weight_semantic_similarity=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.retrieval.fusion_weights.semantic_similarity"), 0.34),
                    default=0.34,
                ),
            ),
        ),
        memory_retrieval_weight_lexical_score=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.retrieval.fusion_weights.lexical_score"), 0.25),
                    default=0.25,
                ),
            ),
        ),
        memory_retrieval_weight_recency_score=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.retrieval.fusion_weights.recency_score"), 0.10),
                    default=0.10,
                ),
            ),
        ),
        memory_retrieval_weight_importance_score=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.retrieval.fusion_weights.importance_score"), 0.09),
                    default=0.09,
                ),
            ),
        ),
        memory_retrieval_weight_confidence_score=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.retrieval.fusion_weights.confidence_score"), 0.08),
                    default=0.08,
                ),
            ),
        ),
        memory_retrieval_weight_entity_overlap_score=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.retrieval.fusion_weights.entity_overlap_score"), 0.07),
                    default=0.07,
                ),
            ),
        ),
        memory_retrieval_weight_exact_match_boost=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.retrieval.fusion_weights.exact_match_boost"), 0.04),
                    default=0.04,
                ),
            ),
        ),
        memory_retrieval_weight_scope_match_score=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.retrieval.fusion_weights.scope_match_score"), 0.03),
                    default=0.03,
                ),
            ),
        ),
        memory_chunk_size=max(200, _to_int(_pick_value(_get_dotted(row, "memory.documents.chunk_size"), 1200), default=1200)),
        memory_chunk_overlap=max(
            0,
            _to_int(_pick_value(_get_dotted(row, "memory.documents.chunk_overlap"), 160), default=160),
        ),
        memory_summary_trigger=max(1, _to_int(_pick_value(_get_dotted(row, "memory.summary.trigger"), 60), default=60)),
        memory_summary_target_tokens=max(
            32,
            _to_int(_pick_value(_get_dotted(row, "memory.summary.target_tokens"), 220), default=220),
        ),
        memory_context_budget_total=max(
            256,
            _to_int(_pick_value(_get_dotted(row, "memory.context_budget.total"), 2200), default=2200),
        ),
        memory_context_budget_memory=max(
            64,
            _to_int(_pick_value(_get_dotted(row, "memory.context_budget.memory"), 700), default=700),
        ),
        memory_context_budget_docs=max(
            64,
            _to_int(_pick_value(_get_dotted(row, "memory.context_budget.docs"), 600), default=600),
        ),
        memory_context_budget_tools=max(
            32,
            _to_int(_pick_value(_get_dotted(row, "memory.context_budget.tools"), 220), default=220),
        ),
        memory_context_budget_response_reserve=max(
            64,
            _to_int(
                _pick_value(_get_dotted(row, "memory.context_budget.response_reserve"), 260),
                default=260,
            ),
        ),
        memory_stale_after_days=max(
            1,
            _to_int(_pick_value(_get_dotted(row, "memory.lifecycle.stale_after_days"), 30), default=30),
        ),
        memory_archive_after_days=max(
            1,
            _to_int(_pick_value(_get_dotted(row, "memory.lifecycle.archive_after_days"), 90), default=90),
        ),
        memory_promotion_message_importance_threshold=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.lifecycle.promotion_thresholds.message_importance"), 0.55),
                    default=0.55,
                ),
            ),
        ),
        memory_promotion_message_confidence_threshold=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.lifecycle.promotion_thresholds.message_confidence"), 0.50),
                    default=0.50,
                ),
            ),
        ),
        memory_promotion_project_signal_boost=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.lifecycle.promotion_signal_boosts.project"), 0.12),
                    default=0.12,
                ),
            ),
        ),
        memory_promotion_fact_signal_boost=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.lifecycle.promotion_signal_boosts.fact"), 0.16),
                    default=0.16,
                ),
            ),
        ),
        memory_promotion_decision_signal_boost=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.lifecycle.promotion_signal_boosts.decision"), 0.12),
                    default=0.12,
                ),
            ),
        ),
        memory_promotion_smalltalk_penalty=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.lifecycle.promotion_signal_boosts.smalltalk_penalty"), 0.20),
                    default=0.20,
                ),
            ),
        ),
        memory_importance_weight_base=max(
            0.0,
            min(
                1.0,
                _to_float(_pick_value(_get_dotted(row, "memory.scoring.importance_weights.base"), 0.42), default=0.42),
            ),
        ),
        memory_importance_weight_decision=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.scoring.importance_weights.decision"), 0.24),
                    default=0.24,
                ),
            ),
        ),
        memory_importance_weight_remember=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.scoring.importance_weights.remember"), 0.18),
                    default=0.18,
                ),
            ),
        ),
        memory_importance_weight_project=max(
            0.0,
            min(
                1.0,
                _to_float(_pick_value(_get_dotted(row, "memory.scoring.importance_weights.project"), 0.10), default=0.10),
            ),
        ),
        memory_salience_weight_novelty=max(
            0.0,
            min(
                1.0,
                _to_float(_pick_value(_get_dotted(row, "memory.scoring.salience_weights.novelty"), 0.22), default=0.22),
            ),
        ),
        memory_salience_weight_permanence=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.scoring.salience_weights.permanence"), 0.20),
                    default=0.20,
                ),
            ),
        ),
        memory_salience_weight_repetition=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.scoring.salience_weights.repetition"), 0.14),
                    default=0.14,
                ),
            ),
        ),
        memory_salience_weight_project_relevance=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.scoring.salience_weights.project_relevance"), 0.16),
                    default=0.16,
                ),
            ),
        ),
        memory_salience_weight_task_relevance=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.scoring.salience_weights.task_relevance"), 0.16),
                    default=0.16,
                ),
            ),
        ),
        memory_salience_weight_explicit_save_signal=max(
            0.0,
            min(
                1.0,
                _to_float(
                    _pick_value(_get_dotted(row, "memory.scoring.salience_weights.explicit_save_signal"), 0.12),
                    default=0.12,
                ),
            ),
        ),
        memory_temporary_ttl_sec=max(
            30,
            _to_int(_pick_value(_get_dotted(row, "memory.lifecycle.temporary_ttl_sec"), 3600), default=3600),
        ),
        memory_private_runtime_ttl_sec=max(
            30,
            _to_int(
                _pick_value(_get_dotted(row, "memory.lifecycle.private_runtime_ttl_sec"), 900),
                default=900,
            ),
        ),
        memory_working_limit=max(
            20,
            _to_int(_pick_value(_get_dotted(row, "memory.lifecycle.working_limit"), 120), default=120),
        ),
        model_fallbacks=_to_csv_list(_get_dotted(row, "llm.model_fallbacks")),
        gpu_vram_gb=_to_int_or_none(_get_dotted(row, "hardware.gpu_vram_gb")),
        console_timeout_sec=float(_pick_value(_get_dotted(row, "ui.console.timeout_sec"), 2.5)),
        console_stream_timeout_sec=float(_pick_value(_get_dotted(row, "ui.console.stream_timeout_sec"), 600.0)),
        console_store_turn=_to_bool(_get_dotted(row, "ui.console.store_turn")),
        console_show_thinking=_to_bool(_get_dotted(row, "ui.console.show_thinking")),
        console_thinking_first=_to_bool(_get_dotted(row, "ui.console.thinking_first")),
        console_auto_start_api=_to_bool(_get_dotted(row, "ui.console.auto_start_api")),
        console_auto_start_ollama=_to_bool(_get_dotted(row, "ui.console.auto_start_ollama")),
        console_runtime=runtime,
    )
    return settings


def get_model_profiles(*, force_reload: bool = False) -> dict[str, ModelProfile]:
    settings = load_config(force_reload=force_reload)
    rows = _normalize_profile_rows(settings.llm_profiles)
    out: dict[str, ModelProfile] = {}
    for key, payload in rows.items():
        out[key] = _profile_from_row(name=key, payload=payload)
    return out


def get_profile(name: str | None) -> ModelProfile:
    profiles = get_model_profiles()
    key = str(name or "BALANCED").strip().upper()
    if key not in profiles:
        key = "BALANCED"
    return _apply_hardware_guards(profiles[key])


def merge_profile(profile: ModelProfile | str, overrides: dict[str, Any] | None = None) -> ModelProfile:
    base = get_profile(profile if isinstance(profile, str) else profile.name)
    if isinstance(profile, ModelProfile):
        base = profile
    patch = dict(overrides or {})
    if not patch:
        return _apply_hardware_guards(base)

    payload = base.to_dict()
    _deep_merge(payload, patch)
    merged = _profile_from_row(name=str(payload.get("name") or base.name), payload=payload)
    return _apply_hardware_guards(merged)


def build_ollama_options(task_type: str) -> dict[str, Any]:
    _ = task_type
    profile = get_profile(load_config().active_profile)
    max_tokens = profile.generation.max_tokens
    num_predict = int(max_tokens if max_tokens is not None else 768)
    return {
        "temperature": float(profile.generation.temperature),
        "top_p": float(profile.generation.top_p),
        "repeat_penalty": float(profile.generation.repeat_penalty),
        "num_predict": num_predict,
    }


def setup_logging(settings: AppSettings | None = None, *, force: bool = False) -> None:
    global _LOG_SETUP_DONE
    if _LOG_SETUP_DONE and not force:
        return

    cfg = settings or load_config()
    log_level_name = str(cfg.log_level).strip().upper()
    level = getattr(logging, log_level_name, logging.INFO)
    max_bytes = int(cfg.log_max_bytes)
    backup_count = int(cfg.log_backup_count)
    fmt = str(cfg.log_format or _LOG_FORMAT_DEFAULT)

    log_dir = Path(str(cfg.log_dir or LOG_DIR)).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    if not _has_stream_handler(root):
        root.addHandler(_console_handler(level, fmt=fmt))
    if not _has_file_handler(root, "app.log"):
        root.addHandler(_file_handler(log_dir / "app.log", level, fmt=fmt, max_bytes=max_bytes, backup_count=backup_count))

    channels = _normalize_log_channels(cfg.log_channels)
    for channel, prefixes in channels.items():
        filename = f"{channel}.log"
        if _has_file_handler(root, filename):
            continue
        handler = _file_handler(log_dir / filename, level, fmt=fmt, max_bytes=max_bytes, backup_count=backup_count)
        handler.addFilter(_LoggerPrefixFilter(tuple(prefixes)))
        root.addHandler(handler)

    # Web trace v2 writes per-trace JSON files from pipeline code.
    # Keep logger channels only; no dedicated jsonl sink.

    for channel, prefixes in channels.items():
        logging.getLogger(channel).setLevel(level)
        for prefix in prefixes:
            logging.getLogger(prefix).setLevel(level)

    logging.captureWarnings(True)
    _LOG_SETUP_DONE = True


configure_logging = setup_logging


def _profile_from_row(*, name: str, payload: dict[str, Any]) -> ModelProfile:
    defaults = copy.deepcopy(_default_model_profiles_tree().get(str(name or "BALANCED").strip().upper(), _default_model_profiles_tree()["BALANCED"]))
    src = copy.deepcopy(payload if isinstance(payload, dict) else {})
    _deep_merge(defaults, src)
    generation_row = _as_dict(defaults.get("generation"))
    ollama_row = _as_dict(defaults.get("ollama"))
    openai_row = _as_dict(defaults.get("openai"))

    stop_values = generation_row.get("stop")
    if isinstance(stop_values, list):
        stop_tuple = tuple(str(x) for x in stop_values if str(x or "").strip())
    elif isinstance(stop_values, tuple):
        stop_tuple = tuple(str(x) for x in stop_values if str(x or "").strip())
    else:
        stop_tuple = ()

    generation = GenerationProfile(
        temperature=float(_pick_value(generation_row.get("temperature"), 0.7)),
        top_p=float(_pick_value(generation_row.get("top_p"), 0.9)),
        repeat_penalty=float(_pick_value(generation_row.get("repeat_penalty"), 1.1)),
        max_tokens=_to_int_or_none(generation_row.get("max_tokens")),
        stop=stop_tuple,
    )
    ollama = OllamaProfile(
        num_thread=max(1, _to_int(ollama_row.get("num_thread"), default=6)),
        num_ctx=max(512, _to_int(ollama_row.get("num_ctx"), default=8192)),
        num_gpu=max(0, _to_int(ollama_row.get("num_gpu"), default=1)),
        num_batch=max(1, _to_int(ollama_row.get("num_batch"), default=128)),
        keep_alive=_norm_str(_pick_value(ollama_row.get("keep_alive"), "5m")) or "5m",
    )
    openai = OpenAIProfile(
        model=_norm_str(openai_row.get("model")),
        reasoning_effort=_norm_lower(_pick_value(openai_row.get("reasoning_effort"), "medium")) or "medium",
    )
    profile_name = _norm_upper(_pick_value(defaults.get("name"), name, "BALANCED")) or "BALANCED"
    return ModelProfile(name=profile_name, generation=generation, ollama=ollama, openai=openai)


def _apply_hardware_guards(profile: ModelProfile) -> ModelProfile:
    app_settings = load_config()
    vram = app_settings.gpu_vram_gb
    if vram is None:
        return profile
    ollama = profile.ollama
    if vram <= 4:
        ollama = OllamaProfile(
            num_thread=min(ollama.num_thread, 6),
            num_ctx=min(ollama.num_ctx, 4096),
            num_gpu=min(ollama.num_gpu, 1),
            num_batch=min(ollama.num_batch, 64),
            keep_alive=ollama.keep_alive,
        )
    elif vram <= 6:
        ollama = OllamaProfile(
            num_thread=min(ollama.num_thread, 8),
            num_ctx=min(ollama.num_ctx, 6144),
            num_gpu=min(ollama.num_gpu, 1),
            num_batch=min(ollama.num_batch, 96),
            keep_alive=ollama.keep_alive,
        )
    return ModelProfile(name=profile.name, generation=profile.generation, ollama=ollama, openai=profile.openai)


def _normalize_profile_rows(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    src = dict(value or {})
    out: dict[str, dict[str, Any]] = {}
    for key, raw in src.items():
        name = _norm_upper(key)
        if not name:
            continue
        if isinstance(raw, ModelProfile):
            out[name] = raw.to_dict()
            continue
        if not isinstance(raw, dict):
            continue
        payload = copy.deepcopy(raw)
        payload["name"] = _norm_upper(_pick_value(payload.get("name"), name)) or name
        out[name] = payload
    if not out:
        return copy.deepcopy(_default_model_profiles_tree())
    return out


def _normalize_task_model_profile_rows(
    value: dict[str, Any],
    *,
    default_provider: str = "ollama",
    default_model: str = "qcwind/qwen3-8b-instruct-Q4-K-M",
) -> dict[str, dict[str, Any]]:
    src = dict(value or {})
    out: dict[str, dict[str, Any]] = {}
    defaults = _default_task_model_profiles_tree(provider=default_provider, model=default_model)
    generic_default = {
        "name": "",
        "provider": _norm_lower(default_provider) or "ollama",
        "model": _norm_str(default_model) or "qcwind/qwen3-8b-instruct-Q4-K-M",
        "temperature": 0.2,
        "max_tokens": 256,
        "timeout": 30.0,
        "enabled": True,
        "fallback_profile": "",
    }
    for key, raw in src.items():
        name = _norm_lower(key)
        if not name:
            continue
        if not isinstance(raw, dict):
            continue
        payload = copy.deepcopy(defaults.get(name, generic_default))
        payload.update(copy.deepcopy(raw))
        payload["name"] = _norm_lower(_pick_value(payload.get("name"), name)) or name
        payload["provider"] = _norm_lower(_pick_value(payload.get("provider"), payload.get("provider") or default_provider))
        payload["model"] = _norm_str(_pick_value(payload.get("model"), payload.get("model") or default_model))
        payload["temperature"] = float(_pick_value(payload.get("temperature"), payload.get("temperature") or 0.2))
        payload["max_tokens"] = max(1, _to_int(_pick_value(payload.get("max_tokens"), payload.get("max_tokens") or 256), default=256))
        payload["timeout"] = max(1.0, float(_pick_value(payload.get("timeout"), payload.get("timeout") or 30.0)))
        payload["enabled"] = bool(_pick_value(payload.get("enabled"), True))
        payload["fallback_profile"] = _norm_lower(payload.get("fallback_profile"))
        out[name] = payload
    if not out:
        return copy.deepcopy(defaults)
    return out


def _normalize_log_channels(value: dict[str, Any]) -> dict[str, list[str]]:
    src = dict(value or {})
    out: dict[str, list[str]] = {}
    for channel, prefixes in src.items():
        key = _norm_lower(channel)
        if not key:
            continue
        items: list[str] = []
        if isinstance(prefixes, list):
            items = [_norm_str(x) for x in prefixes]
        elif isinstance(prefixes, tuple):
            items = [_norm_str(x) for x in prefixes]
        elif isinstance(prefixes, str):
            items = [_norm_str(prefixes)]
        items = [x for x in items if x]
        if items:
            out[key] = items
    if not out:
        return copy.deepcopy(_LOG_CHANNEL_PREFIXES_DEFAULT)
    return out


def _normalize_ui_console_runtime(value: dict[str, Any]) -> dict[str, Any]:
    src = dict(value or {})
    out: dict[str, Any] = {}
    if isinstance(src.get("mode_lock"), bool):
        out["mode_lock"] = bool(src.get("mode_lock"))
    active_mode = _norm_str(src.get("active_mode"))
    if active_mode:
        out["active_mode"] = active_mode
    if isinstance(src.get("output_parameters"), bool):
        out["output_parameters"] = bool(src.get("output_parameters"))
    if isinstance(src.get("output_summary"), bool):
        out["output_summary"] = bool(src.get("output_summary"))
    return out


def _console_handler(level: int, *, fmt: str) -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(str(fmt or _LOG_FORMAT_DEFAULT)))
    return handler


def _file_handler(path: Path, level: int, *, fmt: str, max_bytes: int, backup_count: int) -> logging.Handler:
    handler = RotatingFileHandler(
        path,
        mode="a",
        maxBytes=max(1024, int(max_bytes)),
        backupCount=max(1, int(backup_count)),
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(str(fmt or _LOG_FORMAT_DEFAULT)))
    return handler


def _has_file_handler(logger: logging.Logger, filename: str) -> bool:
    needle = str(filename).lower()
    for handler in logger.handlers:
        base = getattr(handler, "baseFilename", "")
        if base and str(base).lower().endswith(needle):
            return True
    return False


def _has_stream_handler(logger: logging.Logger) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return True
    return False


def _bootstrap_seed_from_env(*, dotenv_cfg: dict[str, str]) -> dict[str, Any]:
    seed: dict[str, Any] = {}
    mappings: list[tuple[str, str, Callable[[Any], Any]]] = [
        ("MMIS_APP_NAME", "app.name", _norm_str),
        ("MMIS_DEBUG", "app.debug", _to_bool),
        ("MMIS_LOCALE", "app.locale", _norm_str),
        ("MMIS_DEFAULT_LANGUAGE", "app.default_language", _norm_str),
        ("MMIS_START_MODE", "startup.mode", _norm_lower),
        ("MMIS_ACTIVE_PROFILE", "startup.active_profile", _norm_upper),
        ("MMIS_SAFETY_MODE", "startup.safety_mode", _norm_lower),
        ("MMIS_API_HOST", "api.host", _norm_str),
        ("MMIS_API_PORT", "api.port", lambda x: _to_int(x, default=8027)),
        ("MMIS_LLM_PROVIDER", "llm.provider", _norm_lower),
        ("MMIS_MODEL_NAME", "llm.model_name", _norm_str),
        ("MMIS_THINKING_ENABLED", "llm.thinking_enabled", _to_bool),
        ("MMIS_JSON_MODE", "llm.json_mode_enabled", _to_bool),
        ("MMIS_WEB_MODE", "internet.web_mode", _norm_lower),
        ("MMIS_INTERNET_ENABLED", "internet.enabled", _to_bool),
        ("MMIS_AUTOMATION_ENABLED", "modules.automation_enabled", _to_bool),
        ("MMIS_SCREEN_ENABLED", "modules.screen_enabled", _to_bool),
        ("MMIS_VOICE_ENABLED", "voice.enabled", _to_bool),
        ("MMIS_DATA_DIR", "paths.data_dir", _norm_str),
        ("MMIS_MODELS_DIR", "paths.models_dir", _norm_str),
        ("MMIS_MEMORY_DIR", "memory.memory_dir", _norm_str),
        ("MMIS_CACHE_DIR", "memory.cache_dir", _norm_str),
        ("MMIS_LOG_DIR", "memory.log_dir", _norm_str),
        ("MMIS_DB_PATH", "memory.db_path", _norm_str),
        ("MMIS_LOG_LEVEL", "logging.level", _norm_upper),
        ("MMIS_LOG_FILE", "logging.file", _norm_str),
        ("MMIS_LOG_COLORS", "logging.colors", _to_bool),
        ("MMIS_LOG_MAX_BYTES", "logging.max_bytes", lambda x: _to_int(x, default=10485760)),
        ("MMIS_LOG_BACKUP_COUNT", "logging.backup_count", lambda x: _to_int(x, default=5)),
        ("MMIS_METADATA_MODEL", "llm.metadata.model", _norm_str),
        ("MMIS_METADATA_MODEL_FALLBACKS", "llm.metadata.fallbacks", _to_csv_list),
        ("MMIS_LLM_MAX_TOKENS_LOWER_BOUND", "llm.max_tokens.lower_bound", lambda x: _to_int(x, default=2048)),
        ("MMIS_LLM_MAX_TOKENS_UPPER_BOUND", "llm.max_tokens.upper_bound", lambda x: _to_int(x, default=8192)),
        ("OLLAMA_HOST", "llm.providers.ollama.base_url", _norm_str),
        ("OLLAMA_TIMEOUT_SEC", "llm.providers.ollama.timeout_sec", float),
        ("OLLAMA_RETRIES", "llm.providers.ollama.retries", lambda x: _to_int(x, default=1)),
        ("OPENAI_API_KEY", "llm.providers.openai.api_key", _norm_str),
        ("OPENAI_BASE_URL", "llm.providers.openai.api_url", _norm_str),
        ("OPENAI_TIMEOUT_SEC", "llm.providers.openai.timeout_sec", float),
        ("OPENAI_MAX_RETRIES", "llm.providers.openai.max_retries", lambda x: _to_int(x, default=2)),
    ]
    for env_key, dotted, parser in mappings:
        value = _env_pick(env_key, dotenv_cfg=dotenv_cfg)
        if value is None:
            continue
        parsed = parser(value)
        if parsed is None:
            continue
        _set_dotted(seed, dotted, parsed)
    use_legacy = _env_pick("MMIS_USE_LEGACY_MEMORY_DIR", dotenv_cfg=dotenv_cfg)
    if str(use_legacy or "").strip().lower() in {"1", "true", "yes", "on"}:
        if _get_dotted(seed, "memory.memory_dir") is None:
            _set_dotted(seed, "memory.memory_dir", _path_to_config_string(LEGACY_MEMORY_DIR))
    return seed


def _resolve_config_file() -> Path:
    env_raw = str(os.getenv("MMIS_CONFIG_FILE", "")).strip()
    if env_raw:
        return Path(env_raw).expanduser().resolve()
    return (BASE_DIR / "config" / "config.json").resolve()


def _resolve_dotenv_file() -> Path | None:
    env_raw = str(os.getenv("MMIS_DOTENV_FILE", "")).strip()
    if env_raw:
        return Path(env_raw).expanduser().resolve()
    default = BASE_DIR / ".env"
    return default if default.exists() else None


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


def _validate_settings(settings: AppSettings) -> None:
    errors: list[str] = []
    known_profiles = {str(x).upper() for x in dict(settings.llm_profiles or {}).keys()}
    if not known_profiles:
        known_profiles = set(VALID_PROFILES)
    if settings.active_profile not in known_profiles:
        errors.append(f"startup.active_profile must be one of {sorted(known_profiles)}, got: {settings.active_profile}")
    if settings.llm_default_provider not in VALID_PROVIDERS:
        errors.append(f"MMIS_LLM_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got: {settings.llm_default_provider}")
    if settings.safety_mode not in VALID_SAFETY_MODES:
        errors.append(f"MMIS_SAFETY_MODE must be one of {sorted(VALID_SAFETY_MODES)}, got: {settings.safety_mode}")
    errors.extend(_validate_task_model_profiles(settings.task_model_profiles))
    if not settings.host:
        errors.append("api.host cannot be empty")
    if settings.port < 1 or settings.port > 65535:
        errors.append(f"api.port must be in range 1..65535, got: {settings.port}")
    if not settings.model_name:
        errors.append("llm.model_name cannot be empty")
    if not settings.startup_mode:
        errors.append("startup.mode cannot be empty")
    if not str(settings.log_format or "").strip():
        errors.append("logging.format cannot be empty")
    if str(settings.web_mode or "").strip().lower() not in {"on", "off", "auto"}:
        errors.append(f"internet.web_mode must be one of ['auto', 'off', 'on'], got: {settings.web_mode}")
    errors.extend(_validate_web_v2_settings(settings.web_v2))
    if str(settings.memory_facts_scope or "").strip().lower() not in {"user_only", "all"}:
        errors.append(
            f"memory.facts_scope must be one of ['all', 'user_only'], got: {settings.memory_facts_scope}"
        )
    if int(settings.memory_confirmation_ttl_sec) < 1:
        errors.append("memory.confirmation_ttl_sec must be >= 1")
    if int(settings.memory_migration_schema_version) < 1:
        errors.append("memory.migration.schema_version must be >= 1")
    if str(settings.memory_version or "").strip().lower() not in {"v2"}:
        errors.append(f"memory.version must be 'v2', got: {settings.memory_version}")
    if str(settings.memory_backend or "").strip().lower() not in {"chroma", "chromadb"}:
        errors.append(
            "memory.backend must be one of ['chroma', 'chromadb']; "
            f"got: {settings.memory_backend}. Switch memory.backend to 'chroma' or 'chromadb'."
        )
    if int(settings.memory_retrieval_top_k) < 1:
        errors.append("memory.retrieval.top_k must be >= 1")
    if int(settings.memory_rerank_top_k) < 1:
        errors.append("memory.retrieval.rerank_top_k must be >= 1")
    if not (0.0 <= float(settings.memory_retrieval_weight_semantic_similarity) <= 1.0):
        errors.append("memory.retrieval.fusion_weights.semantic_similarity must be in [0, 1]")
    if not (0.0 <= float(settings.memory_retrieval_weight_lexical_score) <= 1.0):
        errors.append("memory.retrieval.fusion_weights.lexical_score must be in [0, 1]")
    if not (0.0 <= float(settings.memory_retrieval_weight_recency_score) <= 1.0):
        errors.append("memory.retrieval.fusion_weights.recency_score must be in [0, 1]")
    if not (0.0 <= float(settings.memory_retrieval_weight_importance_score) <= 1.0):
        errors.append("memory.retrieval.fusion_weights.importance_score must be in [0, 1]")
    if not (0.0 <= float(settings.memory_retrieval_weight_confidence_score) <= 1.0):
        errors.append("memory.retrieval.fusion_weights.confidence_score must be in [0, 1]")
    if not (0.0 <= float(settings.memory_retrieval_weight_entity_overlap_score) <= 1.0):
        errors.append("memory.retrieval.fusion_weights.entity_overlap_score must be in [0, 1]")
    if not (0.0 <= float(settings.memory_retrieval_weight_exact_match_boost) <= 1.0):
        errors.append("memory.retrieval.fusion_weights.exact_match_boost must be in [0, 1]")
    if not (0.0 <= float(settings.memory_retrieval_weight_scope_match_score) <= 1.0):
        errors.append("memory.retrieval.fusion_weights.scope_match_score must be in [0, 1]")
    if int(settings.memory_chunk_size) < 200:
        errors.append("memory.documents.chunk_size must be >= 200")
    if int(settings.memory_chunk_overlap) < 0:
        errors.append("memory.documents.chunk_overlap must be >= 0")
    if not (0.0 <= float(settings.memory_promotion_message_importance_threshold) <= 1.0):
        errors.append("memory.lifecycle.promotion_thresholds.message_importance must be in [0, 1]")
    if not (0.0 <= float(settings.memory_promotion_message_confidence_threshold) <= 1.0):
        errors.append("memory.lifecycle.promotion_thresholds.message_confidence must be in [0, 1]")
    if not (0.0 <= float(settings.memory_promotion_project_signal_boost) <= 1.0):
        errors.append("memory.lifecycle.promotion_signal_boosts.project must be in [0, 1]")
    if not (0.0 <= float(settings.memory_promotion_fact_signal_boost) <= 1.0):
        errors.append("memory.lifecycle.promotion_signal_boosts.fact must be in [0, 1]")
    if not (0.0 <= float(settings.memory_promotion_decision_signal_boost) <= 1.0):
        errors.append("memory.lifecycle.promotion_signal_boosts.decision must be in [0, 1]")
    if not (0.0 <= float(settings.memory_promotion_smalltalk_penalty) <= 1.0):
        errors.append("memory.lifecycle.promotion_signal_boosts.smalltalk_penalty must be in [0, 1]")
    if not (0.0 <= float(settings.memory_importance_weight_base) <= 1.0):
        errors.append("memory.scoring.importance_weights.base must be in [0, 1]")
    if not (0.0 <= float(settings.memory_importance_weight_decision) <= 1.0):
        errors.append("memory.scoring.importance_weights.decision must be in [0, 1]")
    if not (0.0 <= float(settings.memory_importance_weight_remember) <= 1.0):
        errors.append("memory.scoring.importance_weights.remember must be in [0, 1]")
    if not (0.0 <= float(settings.memory_importance_weight_project) <= 1.0):
        errors.append("memory.scoring.importance_weights.project must be in [0, 1]")
    if not (0.0 <= float(settings.memory_salience_weight_novelty) <= 1.0):
        errors.append("memory.scoring.salience_weights.novelty must be in [0, 1]")
    if not (0.0 <= float(settings.memory_salience_weight_permanence) <= 1.0):
        errors.append("memory.scoring.salience_weights.permanence must be in [0, 1]")
    if not (0.0 <= float(settings.memory_salience_weight_repetition) <= 1.0):
        errors.append("memory.scoring.salience_weights.repetition must be in [0, 1]")
    if not (0.0 <= float(settings.memory_salience_weight_project_relevance) <= 1.0):
        errors.append("memory.scoring.salience_weights.project_relevance must be in [0, 1]")
    if not (0.0 <= float(settings.memory_salience_weight_task_relevance) <= 1.0):
        errors.append("memory.scoring.salience_weights.task_relevance must be in [0, 1]")
    if not (0.0 <= float(settings.memory_salience_weight_explicit_save_signal) <= 1.0):
        errors.append("memory.scoring.salience_weights.explicit_save_signal must be in [0, 1]")
    if int(settings.memory_temporary_ttl_sec) < 30:
        errors.append("memory.lifecycle.temporary_ttl_sec must be >= 30")
    if int(settings.memory_private_runtime_ttl_sec) < 30:
        errors.append("memory.lifecycle.private_runtime_ttl_sec must be >= 30")
    if errors:
        raise ValueError("Invalid application settings:\n- " + "\n- ".join(errors))


def _validate_task_model_profiles(payload: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    rows = _normalize_task_model_profile_rows(_as_dict(payload))
    if not rows:
        errors.append("llm.task_models must define at least one task profile")
        return errors
    known_names = {str(name or "").strip().lower() for name in rows.keys()}
    for task_name, raw in rows.items():
        profile = _as_dict(raw)
        label = f"llm.task_models.{task_name}"
        provider = _norm_lower(profile.get("provider"))
        if provider not in VALID_PROVIDERS:
            errors.append(f"{label}.provider must be one of {sorted(VALID_PROVIDERS)}, got: {provider or '<empty>'}")
        if not _norm_str(profile.get("model")):
            errors.append(f"{label}.model cannot be empty")
        try:
            temperature = float(profile.get("temperature"))
        except Exception:
            errors.append(f"{label}.temperature must be a number")
            temperature = 0.0
        if not (0.0 <= temperature <= 2.0):
            errors.append(f"{label}.temperature must be in [0, 2], got: {temperature}")
        try:
            max_tokens = int(profile.get("max_tokens"))
        except Exception:
            errors.append(f"{label}.max_tokens must be an integer")
            max_tokens = 0
        if max_tokens < 1:
            errors.append(f"{label}.max_tokens must be >= 1, got: {max_tokens}")
        try:
            timeout = float(profile.get("timeout"))
        except Exception:
            errors.append(f"{label}.timeout must be a number")
            timeout = 0.0
        if timeout <= 0.0:
            errors.append(f"{label}.timeout must be > 0, got: {timeout}")
        enabled = profile.get("enabled")
        if not isinstance(enabled, bool):
            errors.append(f"{label}.enabled must be a boolean")
        fallback_profile = _norm_lower(profile.get("fallback_profile"))
        if fallback_profile:
            if fallback_profile == task_name:
                errors.append(f"{label}.fallback_profile cannot point to itself")
            elif fallback_profile not in known_names:
                errors.append(
                    f"{label}.fallback_profile must reference an existing task profile, got: {fallback_profile}"
                )
    return errors


def _validate_web_v2_settings(payload: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    cfg = _as_dict(payload)
    if not cfg:
        return errors

    enabled = cfg.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        errors.append("internet.web_v2.enabled must be a boolean")

    thresholds = _as_dict(cfg.get("thresholds"))
    threshold_keys = ("no_search_max", "verify_max", "soft_max", "targeted_max")
    parsed_thresholds: dict[str, float] = {}
    for key in threshold_keys:
        if key not in thresholds:
            continue
        try:
            value = float(thresholds.get(key))
        except Exception:
            errors.append(f"internet.web_v2.thresholds.{key} must be a number in [0, 1]")
            continue
        if not (0.0 <= value <= 1.0):
            errors.append(f"internet.web_v2.thresholds.{key} must be in [0, 1]")
            continue
        parsed_thresholds[key] = value
    if all(key in parsed_thresholds for key in threshold_keys):
        if not (
            parsed_thresholds["no_search_max"]
            <= parsed_thresholds["verify_max"]
            <= parsed_thresholds["soft_max"]
            <= parsed_thresholds["targeted_max"]
        ):
            errors.append(
                "internet.web_v2.thresholds must be non-decreasing: "
                "no_search_max <= verify_max <= soft_max <= targeted_max"
            )

    budgets = _as_dict(cfg.get("budgets"))
    for mode_key, row in budgets.items():
        item = _as_dict(row)
        if not item:
            continue
        for field in ("max_queries", "max_sources", "max_pages", "max_fetches"):
            if field not in item:
                continue
            try:
                value = int(item.get(field))
            except Exception:
                errors.append(f"internet.web_v2.budgets.{mode_key}.{field} must be an integer >= 0")
                continue
            if value < 0:
                errors.append(f"internet.web_v2.budgets.{mode_key}.{field} must be >= 0")

    cooldown = cfg.get("cooldown_seconds")
    if cooldown is not None:
        try:
            cooldown_val = int(cooldown)
        except Exception:
            errors.append("internet.web_v2.cooldown_seconds must be an integer >= 0")
        else:
            if cooldown_val < 0:
                errors.append("internet.web_v2.cooldown_seconds must be >= 0")

    retry_policy = _as_dict(cfg.get("retry_policy"))
    for field in ("default_attempts", "verify_only", "soft_search", "targeted_search", "deep_search"):
        if field not in retry_policy:
            continue
        try:
            attempts = int(retry_policy.get(field))
        except Exception:
            errors.append(f"internet.web_v2.retry_policy.{field} must be an integer >= 1")
            continue
        if attempts < 1:
            errors.append(f"internet.web_v2.retry_policy.{field} must be >= 1")
    if "backoff_ms" in retry_policy:
        try:
            backoff_ms = int(retry_policy.get("backoff_ms"))
        except Exception:
            errors.append("internet.web_v2.retry_policy.backoff_ms must be an integer >= 0")
        else:
            if backoff_ms < 0:
                errors.append("internet.web_v2.retry_policy.backoff_ms must be >= 0")

    for key in ("force_search_keywords", "preferred_domains", "blocked_domains", "trusted_allowlist", "risky_domains", "degraded_domains"):
        value = cfg.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            errors.append(f"internet.web_v2.{key} must be a list of strings")
            continue
        for idx, item in enumerate(value):
            if not str(item or "").strip():
                errors.append(f"internet.web_v2.{key}[{idx}] must be a non-empty string")

    ttl = _as_dict(cfg.get("ttl_days") or cfg.get("ttl"))
    for key, value in ttl.items():
        try:
            days = int(value)
        except Exception:
            errors.append(f"internet.web_v2.ttl_days.{key} must be an integer >= 1")
            continue
        if days < 1:
            errors.append(f"internet.web_v2.ttl_days.{key} must be >= 1")

    trust_policy = _as_dict(cfg.get("trust_policy"))
    for key in (
        "trusted_allowlist",
        "preferred_domains",
        "blocked_domains",
        "risky_domains",
        "degraded_domains",
    ):
        value = trust_policy.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            errors.append(f"internet.web_v2.trust_policy.{key} must be a list of strings")
            continue
        for idx, item in enumerate(value):
            if not str(item or "").strip():
                errors.append(f"internet.web_v2.trust_policy.{key}[{idx}] must be a non-empty string")

    for key in (
        "trusted_allowlist_by_category",
        "preferred_domains_by_category",
        "blocked_domains_by_category",
        "risky_domains_by_category",
        "degraded_domains_by_category",
    ):
        value = trust_policy.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            errors.append(f"internet.web_v2.trust_policy.{key} must be a dict of category -> list[str]")
            continue
        for category, domains in value.items():
            if not str(category or "").strip():
                errors.append(f"internet.web_v2.trust_policy.{key} contains an empty category key")
                continue
            if not isinstance(domains, list):
                errors.append(f"internet.web_v2.trust_policy.{key}.{category} must be a list of strings")
                continue
            for idx, item in enumerate(domains):
                if not str(item or "").strip():
                    errors.append(f"internet.web_v2.trust_policy.{key}.{category}[{idx}] must be a non-empty string")

    manual_overrides = trust_policy.get("manual_overrides")
    if manual_overrides is not None:
        if not isinstance(manual_overrides, dict):
            errors.append("internet.web_v2.trust_policy.manual_overrides must be a dict")
        else:
            valid_states = {"trusted", "preferred", "blocked", "risky", "degraded"}
            keys = {str(k or "").strip().lower() for k in manual_overrides.keys()}
            grouped_shape = bool(keys) and keys.issubset(valid_states)
            if grouped_shape:
                for state, domains in manual_overrides.items():
                    token = str(state or "").strip().lower()
                    if token not in valid_states:
                        errors.append(f"internet.web_v2.trust_policy.manual_overrides.{state} is not a valid state")
                        continue
                    if not isinstance(domains, list):
                        errors.append(f"internet.web_v2.trust_policy.manual_overrides.{state} must be a list of strings")
                        continue
                    for idx, item in enumerate(domains):
                        if not str(item or "").strip():
                            errors.append(
                                f"internet.web_v2.trust_policy.manual_overrides.{state}[{idx}] must be a non-empty string"
                            )
            else:
                for domain, state in manual_overrides.items():
                    if not str(domain or "").strip():
                        errors.append("internet.web_v2.trust_policy.manual_overrides contains an empty domain key")
                    token = str(state or "").strip().lower()
                    if token not in valid_states:
                        errors.append(
                            f"internet.web_v2.trust_policy.manual_overrides[{domain!r}] must be one of {sorted(valid_states)}"
                        )

    return errors


def _env_pick(key: str, *, dotenv_cfg: dict[str, str]) -> str | None:
    if key in os.environ:
        return os.environ.get(key)
    return dotenv_cfg.get(key)


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


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _pick_value(*values):
    for value in values:
        if value is not None:
            return value
    return None


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


def _to_float(value, *, default: float) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def _to_int_or_none(value) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


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


def _strip_quotes(value: str) -> str:
    src = str(value or "")
    if len(src) >= 2 and ((src[0] == '"' and src[-1] == '"') or (src[0] == "'" and src[-1] == "'")):
        return src[1:-1]
    return src


def _norm_str(value) -> str:
    return str(value or "").strip()


def _norm_lower(value) -> str:
    return _norm_str(value).lower()


def _norm_upper(value) -> str:
    return _norm_str(value).upper()


def _to_path(value: Any, default: Path) -> Path:
    return _resolve_path_value(value, default)
