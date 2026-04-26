from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SettingKind = Literal["bool", "int", "float", "text", "select", "json"]


@dataclass(frozen=True)
class SettingSpec:
    path: str
    title: str
    kind: SettingKind = "text"
    description: str = ""
    example: str = ""
    options: tuple[str, ...] = ()
    restart_required: bool = False
    dangerous: bool = False
    live: bool = False
    placeholder: str = ""


@dataclass(frozen=True)
class SettingCard:
    title: str
    tag: str
    settings: tuple[SettingSpec, ...]
    dangerous: bool = False


@dataclass(frozen=True)
class SettingCategory:
    key: str
    title: str
    group: str
    description: str
    cards: tuple[SettingCard, ...] = field(default_factory=tuple)


SETTINGS_CATEGORIES: tuple[SettingCategory, ...] = (
    SettingCategory(
        key="main",
        title="Main config",
        group="Core",
        description="Application, startup, API and path settings.",
        cards=(
            SettingCard(
                title="Приложение",
                tag="app",
                settings=(
                    SettingSpec("app.name", "Name", description="Displayed in logs and UI.", example="MMis"),
                    SettingSpec("app.locale", "Locale", description="Application locale hint.", example="ru_RU"),
                    SettingSpec("app.default_language", "Default language", options=("ru", "en", "uk"), kind="select", example="ru"),
                    SettingSpec("app.debug", "Debug mode", kind="bool", description="Enables extra diagnostics.", example="false"),
                ),
            ),
            SettingCard(
                title="Запуск",
                tag="startup",
                settings=(
                    SettingSpec("startup.mode", "Mode", kind="select", options=("api", "ui", "console"), restart_required=True, example="api"),
                    SettingSpec("startup.active_profile", "Active profile", description="Performance profile name.", example="BALANCED", live=True),
                ),
            ),
            SettingCard(
                title="API",
                tag="api",
                settings=(
                    SettingSpec("api.host", "Host", description="0.0.0.0 accepts LAN connections.", example="127.0.0.1", restart_required=True),
                    SettingSpec("api.port", "Port", kind="int", description="FastAPI server port.", example="8027", restart_required=True),
                ),
            ),
            SettingCard(
                title="Подключение к API",
                tag="ui.api",
                settings=(
                    SettingSpec(
                        "ui.api.active_endpoint",
                        "Active endpoint",
                        kind="select",
                        options=("local", "public"),
                        description="Which API endpoint the desktop UI calls.",
                        example="local",
                        live=True,
                    ),
                    SettingSpec(
                        "ui.api.local_base_url",
                        "Local API URL",
                        description="Endpoint for the API running on this machine.",
                        example="http://127.0.0.1:8027",
                        live=True,
                    ),
                    SettingSpec(
                        "ui.api.public_base_url",
                        "Public API URL",
                        description="Endpoint for connecting to an API exposed outside this machine.",
                        example="http://0.0.0.0:8027",
                        live=True,
                    ),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="llm",
        title="LLM / Models",
        group="Core",
        description="Main model provider and native Ollama/OpenAI options. Generation length is controlled by num_ctx and num_batch only.",
        cards=(
            SettingCard(
                title="Runtime",
                tag="llm",
                settings=(
                    SettingSpec("llm.provider", "Provider", kind="select", options=("ollama", "openai", "auto"), example="ollama", restart_required=True),
                    SettingSpec("llm.model_name", "Model name", description="Main chat model.", example="qcwind/qwen3-8b-instruct-Q4-K-M", live=True),
                    SettingSpec("llm.thinking_enabled", "Thinking", kind="bool", description="Allows model reasoning mode when supported.", example="true", live=True),
                    SettingSpec("llm.json_mode_enabled", "JSON mode", kind="bool", description="Default JSON mode flag.", example="false", live=True),
                    SettingSpec("llm.model_fallbacks", "Fallbacks", kind="json", description="Ordered fallback model list.", example='["model-a", "model-b"]'),
                ),
            ),
            SettingCard(
                title="Ollama",
                tag="providers.ollama",
                settings=(
                    SettingSpec(
                        "ui.console.auto_start_ollama",
                        "Auto start Ollama",
                        kind="bool",
                        description="Automatically start `ollama serve` from the desktop UI when Ollama is unavailable.",
                        example="true",
                        live=True,
                    ),
                    SettingSpec("llm.providers.ollama.base_url", "Base URL", example="http://127.0.0.1:11434", restart_required=True),
                    SettingSpec("llm.providers.ollama.timeout_sec", "Timeout sec", kind="float", example="120.0"),
                    SettingSpec("llm.providers.ollama.retries", "Retries", kind="int", example="1"),
                ),
            ),
            SettingCard(
                title="OpenAI",
                tag="providers.openai",
                settings=(
                    SettingSpec("llm.providers.openai.api_url", "API URL", example="https://api.openai.com/v1", restart_required=True),
                    SettingSpec("llm.providers.openai.api_key", "API key", description="Stored in config if edited here.", example="sk-...", dangerous=True),
                    SettingSpec("llm.providers.openai.timeout_sec", "Timeout sec", kind="float", example="120.0"),
                    SettingSpec("llm.providers.openai.max_retries", "Max retries", kind="int", example="2"),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="memory",
        title="Memory Core",
        group="Core",
        description="Memory paths, worker and chat retrieval settings.",
        cards=(
            SettingCard(
                title="Storage",
                tag="memory",
                settings=(
                    SettingSpec("memory.enabled", "Enabled", kind="bool", example="true", restart_required=True),
                    SettingSpec("memory.memory_dir", "Memory dir", example="{dir_path}\\data\\memory_core", restart_required=True),
                    SettingSpec("memory.cache_dir", "Cache dir", example="{dir_path}\\data\\cache", restart_required=True),
                    SettingSpec("memory.db_path", "DB path", example="{dir_path}\\data\\memory_core\\memory.db", restart_required=True),
                ),
            ),
            SettingCard(
                title="Recall",
                tag="memory",
                settings=(
                    SettingSpec("memory.chat_recall_results", "Chat recall results", kind="int", example="3"),
                    SettingSpec("memory.chat_events_limit", "Chat events limit", kind="int", example="10"),
                    SettingSpec("memory.chat_proofread", "Proofread memory", kind="bool", example="false"),
                    SettingSpec("memory.chat_proofread_strict", "Strict proofread", kind="bool", example="false"),
                ),
            ),
            SettingCard(
                title="Worker",
                tag="memory_core",
                settings=(
                    SettingSpec("memory_core.enabled", "Memory Core enabled", kind="bool", example="true", restart_required=True),
                    SettingSpec("memory_core.worker_enabled", "Background worker", kind="bool", example="true", restart_required=True),
                    SettingSpec("memory_core.worker_poll_interval", "Worker poll interval", kind="float", example="2.0"),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="web",
        title="Web / Internet",
        group="Modules",
        description="Search, fetch and evidence policy settings.",
        cards=(
            SettingCard(
                title="Поиск",
                tag="internet",
                settings=(
                    SettingSpec("internet.enabled", "Enabled", kind="bool", example="true", live=True),
                    SettingSpec("internet.web_mode", "Web mode", kind="select", options=("off", "auto", "on", "aggressive"), example="auto", live=True),
                    SettingSpec("internet.search.provider", "Provider", kind="select", options=("searxng", "auto"), example="searxng"),
                    SettingSpec("internet.search.api_url", "Search API URL", example="http://127.0.0.1:8080/search?format=json"),
                    SettingSpec("internet.search.timeout_sec", "Search timeout", kind="float", example="12.0"),
                ),
            ),
            SettingCard(
                title="Загрузка страниц",
                tag="fetch",
                settings=(
                    SettingSpec("internet.fetch.timeout_sec", "Fetch timeout", kind="int", example="12"),
                    SettingSpec("internet.fetch.retries", "Fetch retries", kind="int", example="1"),
                    SettingSpec("internet.fetch.clean_max_chars", "Clean max chars", kind="int", example="4000"),
                    SettingSpec("internet.fetch.clean_min_chars", "Clean min chars", kind="int", example="200"),
                ),
            ),
            SettingCard(
                title="Web v2",
                tag="web_v2",
                settings=(
                    SettingSpec("internet.web_v2", "Advanced web config", kind="json", description="Nested web policy object.", example='{"continuation": {"ttl_minutes": 20}}'),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="voice",
        title="Voice",
        group="Modules",
        description="Speech-to-text, text-to-speech and voice mode settings.",
        cards=(
            SettingCard(
                title="Режим работы",
                tag="voice",
                settings=(
                    SettingSpec("voice.enabled", "Enabled", kind="bool", example="true", restart_required=True),
                    SettingSpec("voice.mode", "Mode", kind="select", options=("push_to_talk", "toggle"), example="push_to_talk"),
                    SettingSpec("voice.open_mode_from_rail", "Open from rail", kind="bool", example="true"),
                    SettingSpec("voice.auto_speak_replies", "Auto speak replies", kind="bool", example="true", live=True),
                    SettingSpec("voice.barge_in", "Barge-in", kind="bool", example="true"),
                ),
            ),
            SettingCard(
                title="STT",
                tag="stt",
                settings=(
                    SettingSpec("voice.stt_engine", "STT engine", example="faster_whisper", restart_required=True),
                    SettingSpec("voice.stt_model", "STT model", example="small", restart_required=True),
                    SettingSpec("voice.stt_device", "STT device", kind="select", options=("cuda", "cpu", "auto"), example="cuda", restart_required=True),
                    SettingSpec("voice.stt_compute_type", "Compute type", example="int8_float16", restart_required=True),
                    SettingSpec("voice.stt_language_hint", "Language hint", example="ru"),
                ),
            ),
            SettingCard(
                title="TTS",
                tag="tts",
                settings=(
                    SettingSpec("voice.tts_engine", "TTS engine", example="qwen", restart_required=True),
                    SettingSpec("voice.tts_model", "TTS model", example="Qwen3-TTS-0.6B", restart_required=True),
                    SettingSpec("voice.tts_device", "TTS device", kind="select", options=("cuda", "cpu", "auto"), example="cuda", restart_required=True),
                    SettingSpec("voice.tts.voice", "Voice", example="ru-RU-DmitryNeural"),
                    SettingSpec("voice.tts.rate", "Rate", example="+0%"),
                    SettingSpec("voice.tts.volume", "Volume", example="+0%"),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="dialog_ui",
        title="Dialog / UI",
        group="Modules",
        description="Greeting, console and visible runtime toggles.",
        cards=(
            SettingCard(
                title="Диалог",
                tag="dialog",
                settings=(
                    SettingSpec("dialog.new_session_after_min", "New session after min", kind="int", example="360"),
                    SettingSpec("dialog.greeting_max_words", "Greeting max words", kind="int", example="6"),
                    SettingSpec("dialog.greeting_max_chars", "Greeting max chars", kind="int", example="35"),
                    SettingSpec("dialog.greetings", "Greetings", kind="json", example='["Привет"]'),
                    SettingSpec("dialog.greeting_exclusions", "Greeting exclusions", kind="json", example='["ошибка"]'),
                ),
            ),
            SettingCard(
                title="Консоль",
                tag="ui.console",
                settings=(
                    SettingSpec("ui.console.timeout_sec", "Timeout sec", kind="float", example="2.5"),
                    SettingSpec("ui.console.stream_timeout_sec", "Stream timeout sec", kind="float", example="600.0"),
                    SettingSpec("ui.console.store_turn", "Store turns", kind="bool", example="true"),
                    SettingSpec("ui.console.show_thinking", "Show thinking", kind="bool", example="true", live=True),
                    SettingSpec("ui.console.thinking_first", "Thinking first", kind="bool", example="false", live=True),
                    SettingSpec("ui.console.auto_start_api", "Auto start API", kind="bool", example="true", restart_required=True),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="debug",
        title="Debug / Inspector",
        group="System",
        description="Inspector panels and prompt diagnostics.",
        cards=(
            SettingCard(
                title="Inspector",
                tag="debug",
                settings=(
                    SettingSpec("debug.memory_inspector_enabled", "Memory inspector", kind="bool", example="true", live=True),
                    SettingSpec("debug.show_raw_scores", "Show raw scores", kind="bool", example="false", live=True),
                    SettingSpec("debug.show_filtered_items", "Show filtered items", kind="bool", example="false", live=True),
                    SettingSpec("debug.show_prompt_blocks", "Show prompt blocks", kind="bool", example="false", live=True),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="logging",
        title="Logging",
        group="System",
        description="Log files, rotation and trace channels.",
        cards=(
            SettingCard(
                title="Files",
                tag="logging",
                settings=(
                    SettingSpec("logging.level", "Level", kind="select", options=("DEBUG", "INFO", "WARNING", "ERROR"), example="INFO", live=True),
                    SettingSpec("logging.file", "File", example="{dir_path}\\data\\logs\\mmis.log", restart_required=True),
                    SettingSpec("logging.colors", "Colors", kind="bool", example="true"),
                    SettingSpec("logging.max_bytes", "Max bytes", kind="int", example="10485760"),
                    SettingSpec("logging.backup_count", "Backup count", kind="int", example="5"),
                    SettingSpec("logging.format", "Format", example="%(asctime)s %(levelname)s %(message)s"),
                ),
            ),
            SettingCard(
                title="Trace",
                tag="web_trace",
                settings=(
                    SettingSpec("logging.web_trace_enabled", "Web trace", kind="bool", example="true", live=True),
                    SettingSpec("logging.web_trace_logger", "Web trace logger", example="mmis.web.trace"),
                    SettingSpec("logging.channels", "Channels", kind="json", example='{"api": true}'),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="safety",
        title="Safety / Tools",
        group="System",
        description="Dangerous automation and prompt safety switches.",
        cards=(
            SettingCard(
                title="Опасные действия",
                tag="safety",
                dangerous=True,
                settings=(
                    SettingSpec("startup.safety_mode", "Safety mode", kind="select", options=("read_only_tools", "allow_os_actions"), example="read_only_tools", dangerous=True, live=True),
                    SettingSpec("modules.automation_enabled", "Automation", kind="bool", example="false", dangerous=True, restart_required=True),
                    SettingSpec("modules.screen_enabled", "Screen tools", kind="bool", example="false", dangerous=True, live=True),
                    SettingSpec("prompt.response_safety_filter_enabled", "Response safety filter", kind="bool", example="true", dangerous=True, live=True),
                    SettingSpec("prompt.response_formatting_enabled", "Response formatting", kind="bool", example="true", live=True),
                ),
            ),
        ),
    ),
)


def iter_settings() -> tuple[SettingSpec, ...]:
    rows: list[SettingSpec] = []
    for category in SETTINGS_CATEGORIES:
        for card in category.cards:
            rows.extend(card.settings)
    return tuple(rows)


def get_category(key: str) -> SettingCategory:
    for category in SETTINGS_CATEGORIES:
        if category.key == key:
            return category
    return SETTINGS_CATEGORIES[0]


def dotted_get(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in [chunk for chunk in str(path or "").split(".") if chunk]:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
