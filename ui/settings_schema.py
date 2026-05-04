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
    admin_only: bool = False


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
        key="characters",
        title="Персонажи",
        group="Core",
        description="Управление личностями, выбор активного персонажа и настройка их поведения.",
        cards=(),
    ),
    SettingCategory(
        key="main",
        title="Main config",
        group="Core",
        description="Настройки приложения, запуска, API и путей.",
        cards=(
            SettingCard(
                title="Приложение",
                tag="app",
                settings=(
                    SettingSpec("app.name", "Name", description="Отображается в логах и интерфейсе.", example="MMis"),
                    SettingSpec("app.locale", "Locale", description="Подсказка локали приложения.", example="ru_RU"),
                    SettingSpec("app.default_language", "Default language", options=("ru", "en", "uk"), kind="select", description="Язык по умолчанию для ответов.", example="ru"),
                    SettingSpec("app.debug", "Debug mode", kind="bool", description="Включает расширенную диагностику.", example="false"),
                ),
            ),
            SettingCard(
                title="Запуск",
                tag="startup",
                settings=(
                    SettingSpec("startup.mode", "Mode", kind="select", options=("api", "ui", "console"), description="Режим запуска приложения.", restart_required=True, example="api"),
                    SettingSpec("startup.active_profile", "Active profile", description="Название профиля производительности.", example="BALANCED", live=True),
                ),
            ),
            SettingCard(
                title="API",
                tag="api",
                settings=(
                    SettingSpec("api.host", "Host", description="0.0.0.0 разрешает подключения по локальной сети.", example="127.0.0.1", restart_required=True),
                    SettingSpec("api.port", "Port", kind="int", description="Порт сервера FastAPI.", example="8027", restart_required=True),
                    SettingSpec(
                        "api.access_lock.enabled",
                        "API lock",
                        kind="bool",
                        description="Полностью закрывает API без правильного X-MMis-Access-Key.",
                        example="true",
                        restart_required=True,
                        dangerous=True,
                        admin_only=True,
                    ),
                    SettingSpec(
                        "api.access_lock.key_hash",
                        "API access key SHA256",
                        description="SHA256 от длинного ключа доступа. Сам ключ здесь не хранить.",
                        example="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
                        restart_required=True,
                        dangerous=True,
                        admin_only=True,
                    ),
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
                        description="Какой эндпоинт API использует Desktop UI.",
                        example="local",
                        live=True,
                    ),
                    SettingSpec(
                        "ui.api.local_base_url",
                        "Local API URL",
                        description="Эндпоинт для API, запущенного на этой машине.",
                        example="http://127.0.0.1:8027  или  http://192.168.1.2:8027",
                        live=True,
                    ),
                    SettingSpec(
                        "ui.api.public_base_url",
                        "Public API URL",
                        description="Эндпоинт для подключения к API, доступному извне.",
                        example="http://45.82.9.3:8027  или  http://your-domain.example:8027",
                        live=True,
                    ),
                    SettingSpec(
                        "ui.api.api_access_key",
                        "API access key",
                        description="Длинный ключ, который Desktop UI отправляет в X-MMis-Access-Key.",
                        example="mmis_7JfQ2QK4cM8nN9vR6sT1wX3yZ5aB0cD2",
                        dangerous=True,
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
        description="Выбор провайдера моделей и настройки Ollama/OpenAI. Длина генерации контролируется только параметрами контекста.",
        cards=(
            SettingCard(
                title="Runtime",
                tag="llm",
                settings=(
                    SettingSpec("llm.provider", "Provider", kind="select", options=("ollama", "openai", "auto"), description="Провайдер LLM.", example="ollama", restart_required=True),
                    SettingSpec("llm.model_name", "Model name", description="Основная модель чата.", example="qwen3.5:8b", live=True),
                    SettingSpec("llm.json_mode_enabled", "JSON mode", kind="bool", description="Флаг режима JSON по умолчанию.", example="false", live=True),
                    SettingSpec("llm.model_fallbacks", "Fallbacks", kind="json", description="Упорядоченный список резервных моделей.", example='["model-a", "model-b"]'),
                ),
            ),
            SettingCard(
                title="Ollama",
                tag="providers.ollama",
                settings=(
                    SettingSpec(
                        "ui.console.auto_start_ollama",
                        "Автозапуск Ollama",
                        kind="bool",
                        description="Разрешает desktop UI запускать Ollama по требованию, когда она недоступна.",
                        example="true",
                        live=True,
                    ),
                    SettingSpec(
                        "ui.ollama.start_mode",
                        "Ollama start mode",
                        kind="select",
                        options=("serve", "ui"),
                        description=(
                            "serve запускает Ollama API через '<ollama.exe> serve'. "
                            "ui запускает через 'ollama list'."
                        ),
                        example="serve",
                        live=True,
                    ),
                    SettingSpec(
                        "ui.ollama.serve_exe",
                        "Ollama serve exe",
                        description=(
                            "Путь к ollama.exe для режима serve. "
                            "Если оставить пустым, будет использоваться команда ollama из PATH."
                        ),
                        example=r"C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe",
                        live=True,
                    ),
                    SettingSpec(
                        "ui.ollama.models_dir",
                        "Ollama models dir",
                        description=(
                            "Папка моделей Ollama. Передаётся в окружение как OLLAMA_MODELS."
                        ),
                        example=r"D:\.ollama\models",
                        live=True,
                    ),
                    SettingSpec("llm.providers.ollama.base_url", "Ollama URL", description="Базовый URL API Ollama.", example="http://127.0.0.1:11434", restart_required=True),
                    SettingSpec("llm.providers.ollama.timeout_sec", "Ollama timeout", kind="float", description="Таймаут запроса к Ollama в секундах.", example="120.0"),
                    SettingSpec("llm.providers.ollama.retries", "Ollama retries", kind="int", description="Количество попыток при ошибке Ollama.", example="1"),
                    SettingSpec("llm.providers.ollama.keep_alive", "Unload after", description="Сколько держать модель в памяти после ответа. Формат Ollama: 30s, 5m, 1h, 0.", example="5m", live=True),
                ),
            ),
            SettingCard(
                title="OpenAI",
                tag="providers.openai",
                settings=(
                    SettingSpec("llm.providers.openai.api_url", "API URL", description="URL API OpenAI (или совместимого).", example="https://api.openai.com/v1", restart_required=True),
                    SettingSpec("llm.providers.openai.api_key", "API key", description="Сохраняется в конфиге при редактировании здесь.", example="sk-...", dangerous=True),
                    SettingSpec("llm.providers.openai.timeout_sec", "Timeout sec", kind="float", description="Таймаут запроса в секундах.", example="120.0"),
                    SettingSpec("llm.providers.openai.max_retries", "Max retries", kind="int", description="Количество повторных попыток.", example="2"),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="memory",
        title="Memory Core",
        group="Core",
        description="Пути хранения памяти, настройки воркера и поиска в чате.",
        cards=(
            SettingCard(
                title="Storage",
                tag="memory",
                settings=(
                    SettingSpec("memory.enabled", "Enabled", kind="bool", description="Включить систему памяти.", example="true", restart_required=True),
                    SettingSpec("memory.memory_dir", "Memory dir", description="Папка для хранения данных памяти.", example="{dir_path}\\data\\memory_core", restart_required=True),
                    SettingSpec("memory.cache_dir", "Cache dir", description="Папка для кэширования.", example="{dir_path}\\data\\cache", restart_required=True),
                    SettingSpec("memory.db_path", "DB path", description="Путь к базе данных памяти.", example="{dir_path}\\data\\memory_core\\memory.db", restart_required=True),
                ),
            ),
            SettingCard(
                title="Recall",
                tag="memory",
                settings=(
                    SettingSpec("memory.chat_recall_results", "Chat recall results", kind="int", description="Количество результатов поиска памяти для чата.", example="3"),
                    SettingSpec("memory.chat_events_limit", "Chat events limit", kind="int", description="Лимит событий чата для анализа.", example="10"),
                    SettingSpec("memory.chat_proofread", "Proofread memory", kind="bool", description="Проверка фактов по памяти.", example="false"),
                    SettingSpec("memory.chat_proofread_strict", "Strict proofread", kind="bool", description="Строгая проверка фактов.", example="false"),
                ),
            ),
            SettingCard(
                title="Worker",
                tag="memory_core",
                settings=(
                    SettingSpec(
                        "memory_core.enabled",
                        "Memory Core",
                        kind="bool",
                        description="Включает новый Memory Core runtime.",
                        example="true",
                        restart_required=True,
                    ),
                    SettingSpec(
                        "memory_core.enable_background_worker",
                        "Background worker",
                        kind="bool",
                        description="Запускает фоновый worker Memory Core для обработки очереди событий памяти.",
                        example="true",
                        restart_required=True,
                    ),
                    SettingSpec(
                        "memory_core.worker_poll_interval",
                        "Worker poll interval",
                        kind="float",
                        description="Интервал проверки очереди Memory Core worker в секундах.",
                        example="2.0",
                    ),
                ),
            ),
            SettingCard(
                title="Memory LLM",
                tag="memory_llm",
                settings=(
                    SettingSpec(
                        "memory_core.memory_llm.keep_alive",
                        "Unload after",
                        description="Сколько держать Memory LLM в памяти после обработки задач. Формат Ollama: 30s, 5m, 30m, 1h, 0.",
                        example="30m",
                        live=True,
                    ),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="web",
        title="Web / Internet",
        group="Modules",
        description="Настройки поиска в интернете, загрузки страниц и политики сбора данных.",
        cards=(
            SettingCard(
                title="Поиск",
                tag="internet",
                settings=(
                    SettingSpec("internet.search.provider", "Provider", kind="select", options=("searxng", "auto"), description="Провайдер поиска.", example="searxng"),
                    SettingSpec("internet.search.api_url", "Search API URL", description="URL API поиска.", example="http://127.0.0.1:8080/search?format=json"),
                    SettingSpec("internet.search.timeout_sec", "Search timeout", kind="float", description="Таймаут поиска в секундах.", example="12.0"),
                ),
            ),
            SettingCard(
                title="Загрузка страниц",
                tag="fetch",
                settings=(
                    SettingSpec("internet.fetch.timeout_sec", "Fetch timeout", kind="int", description="Таймаут загрузки страницы.", example="12"),
                    SettingSpec("internet.fetch.retries", "Fetch retries", kind="int", description="Количество попыток загрузки.", example="1"),
                    SettingSpec("internet.fetch.clean_max_chars", "Clean max chars", kind="int", description="Макс. символов после очистки HTML.", example="4000"),
                    SettingSpec("internet.fetch.clean_min_chars", "Clean min chars", kind="int", description="Мин. символов для валидного контента.", example="200"),
                ),
            ),
            SettingCard(
                title="Web v2",
                tag="web_v2",
                settings=(
                    SettingSpec("internet.web_v2", "Advanced web config", kind="json", description="Объект вложенной политики веб-доступа.", example='{"continuation": {"ttl_minutes": 20}}'),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="voice",
        title="Voice",
        group="Modules",
        description="Настройки распознавания речи (STT), синтеза (TTS) и голосового режима.",
        cards=(
            SettingCard(
                title="Режим работы",
                tag="voice",
                settings=(
                    SettingSpec("voice.enabled", "Enabled", kind="bool", description="Включить голосовые функции.", example="true", restart_required=True),
                    SettingSpec("voice.mode", "Mode", kind="select", options=("push_to_talk", "toggle"), description="Режим активации микрофона.", example="push_to_talk"),
                    SettingSpec("voice.open_mode_from_rail", "Open from rail", kind="bool", description="Открывать голосовой режим из боковой панели.", example="true"),
                    SettingSpec("voice.auto_speak_replies", "Auto speak replies", kind="bool", description="Автоматически озвучивать ответы.", example="true", live=True),
                    SettingSpec("voice.barge_in", "Barge-in", kind="bool", description="Разрешить перебивать бота голосом.", example="true"),
                ),
            ),
            SettingCard(
                title="STT",
                tag="stt",
                settings=(
                    SettingSpec("voice.stt_engine", "STT engine", description="Движок распознавания речи.", example="faster_whisper", restart_required=True),
                    SettingSpec("voice.stt_model", "STT model", description="Модель распознавания.", example="small", restart_required=True),
                    SettingSpec("voice.stt_device", "STT device", kind="select", options=("cuda", "cpu", "auto"), description="Устройство для вычислений STT.", example="cuda", restart_required=True),
                    SettingSpec("voice.stt_compute_type", "Compute type", description="Тип вычислений (напр. float16).", example="int8_float16", restart_required=True),
                    SettingSpec("voice.stt_language_hint", "Language hint", description="Подсказка языка для STT.", example="ru"),
                ),
            ),
            SettingCard(
                title="TTS",
                tag="tts",
                settings=(
                    SettingSpec("voice.tts_engine", "TTS engine", description="Движок синтеза речи.", example="qwen", restart_required=True),
                    SettingSpec("voice.tts_model", "TTS model", description="Модель синтеза.", example="Qwen3-TTS-0.6B", restart_required=True),
                    SettingSpec("voice.tts_device", "TTS device", kind="select", options=("cuda", "cpu", "auto"), description="Устройство для вычислений TTS.", example="cuda", restart_required=True),
                    SettingSpec("voice.tts.voice", "Voice", description="Голос для озвучки.", example="ru-RU-DmitryNeural"),
                    SettingSpec("voice.tts.rate", "Rate", description="Скорость речи.", example="+0%"),
                    SettingSpec("voice.tts.volume", "Volume", description="Громкость речи.", example="+0%"),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="dialog_ui",
        title="Dialog / UI",
        group="Modules",
        description="Настройки приветствий, таймаутов консоли и визуальных эффектов.",
        cards=(
            SettingCard(
                title="Диалог",
                tag="dialog",
                settings=(
                    SettingSpec("dialog.new_session_after_min", "New session after min", kind="int", description="Время до сброса контекста (мин).", example="360"),
                    SettingSpec("dialog.greeting_max_words", "Greeting max words", kind="int", description="Макс. слов в приветствии.", example="6"),
                    SettingSpec("dialog.greeting_max_chars", "Greeting max chars", kind="int", description="Макс. символов в приветствии.", example="35"),
                    SettingSpec("dialog.greetings", "Greetings", kind="json", description="Список случайных приветствий.", example='["Привет"]'),
                    SettingSpec("dialog.greeting_exclusions", "Greeting exclusions", kind="json", description="Исключения для приветствий.", example='["ошибка"]'),
                ),
            ),
            SettingCard(
                title="Консоль",
                tag="ui.console",
                settings=(
                    SettingSpec("ui.console.timeout_sec", "Timeout sec", kind="float", description="Таймаут запроса к API.", example="2.5"),
                    SettingSpec("ui.console.stream_timeout_sec", "Stream timeout sec", kind="float", description="Таймаут стриминга в консоли.", example="600.0"),
                    SettingSpec("ui.console.store_turn", "Store turns", kind="bool", description="Сохранять ходы диалога.", example="true"),
                    SettingSpec("ui.console.show_thinking", "Show thinking", kind="bool", description="Показывать процесс размышления (thinking).", example="true", live=True),
                    SettingSpec("ui.console.thinking_first", "Thinking first", kind="bool", description="Выводить размышления перед основным ответом.", example="false", live=True),
                    SettingSpec("ui.console.auto_start_api", "Auto start API", kind="bool", description="Автоматически запускать API при старте UI.", example="true", restart_required=True),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="debug",
        title="Debug / Inspector",
        group="System",
        description="Панели инспектора и диагностика системных промптов.",
        cards=(
            SettingCard(
                title="Inspector",
                tag="debug",
                settings=(
                    SettingSpec("debug.memory_inspector_enabled", "Memory inspector", kind="bool", description="Включает панель инспектора памяти.", example="true", live=True),
                    SettingSpec("debug.show_raw_scores", "Show raw scores", kind="bool", description="Показывать сырые баллы релевантности.", example="false", live=True),
                    SettingSpec("debug.show_filtered_items", "Show filtered items", kind="bool", description="Показывать отфильтрованные элементы.", example="false", live=True),
                    SettingSpec("debug.show_prompt_blocks", "Show prompt blocks", kind="bool", description="Показывать блоки системного промпта.", example="false", live=True),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="logging",
        title="Logging",
        group="System",
        description="Настройки лог-файлов, ротации и каналов трассировки.",
        cards=(
            SettingCard(
                title="Files",
                tag="logging",
                settings=(
                    SettingSpec("logging.level", "Level", kind="select", options=("DEBUG", "INFO", "WARNING", "ERROR"), description="Уровень логирования.", example="INFO", live=True),
                    SettingSpec("logging.file", "File", description="Путь к файлу лога.", example="{dir_path}\\data\\logs\\mmis.log", restart_required=True),
                    SettingSpec("logging.colors", "Colors", kind="bool", description="Цветной вывод в консоль.", example="true"),
                    SettingSpec("logging.max_bytes", "Max bytes", kind="int", description="Максимальный размер файла лога.", example="10485760"),
                    SettingSpec("logging.backup_count", "Backup count", kind="int", description="Количество хранимых старых логов.", example="5"),
                    SettingSpec("logging.format", "Format", description="Формат строки лога.", example="%(asctime)s %(levelname)s %(message)s"),
                ),
            ),
            SettingCard(
                title="Trace",
                tag="web_trace",
                settings=(
                    SettingSpec("logging.web_trace_enabled", "Web trace", kind="bool", description="Включает детальную трассировку веб-запросов.", example="true", live=True),
                    SettingSpec("logging.web_trace_logger", "Web trace logger", description="Имя логгера для трассировки.", example="mmis.web.trace"),
                    SettingSpec("logging.channels", "Channels", kind="json", description="Настройка каналов логирования.", example='{"api": true}'),
                ),
            ),
        ),
    ),
    SettingCategory(
        key="safety",
        title="Safety / Tools",
        group="System",
        description="Переключатели опасной автоматизации и фильтры безопасности промптов.",
        cards=(
            SettingCard(
                title="Опасные действия",
                tag="safety",
                dangerous=True,
                settings=(
                    SettingSpec("startup.safety_mode", "Safety mode", kind="select", options=("read_only_tools", "allow_os_actions"), description="Режим безопасности для инструментов.", example="read_only_tools", dangerous=True, live=True),
                    SettingSpec("modules.automation_enabled", "Automation", kind="bool", description="Разрешает автоматизацию действий.", example="false", dangerous=True, restart_required=True),
                    SettingSpec("modules.screen_enabled", "Screen tools", kind="bool", description="Разрешает инструменты работы с экраном.", example="false", dangerous=True, live=True),
                    SettingSpec("prompt.response_safety_filter_enabled", "Response safety filter", kind="bool", description="Фильтр безопасности для ответов.", example="true", dangerous=True, live=True),
                    SettingSpec("prompt.response_formatting_enabled", "Response formatting", kind="bool", description="Автоматическое форматирование ответов.", example="true", live=True),
                ),
            ),
        ),
    ),
)

_CORE_CATEGORY_ORDER = {
    "main": 0,
    "llm": 1,
    "memory": 2,
    "characters": 3,
}
SETTINGS_CATEGORIES = tuple(
    sorted(
        SETTINGS_CATEGORIES,
        key=lambda category: (
            0 if category.group == "Core" else 1,
            _CORE_CATEGORY_ORDER.get(category.key, 100),
        ),
    )
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
