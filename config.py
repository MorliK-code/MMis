import os
from pathlib import Path

basedir = Path(__file__).parent
MemoryStorageDir = basedir / "memory_storage"

MODEL_NAME = "qwen2.5:7b-instruct"
EMBED_MODEL = "nomic-embed-text"

SHORT_MEMORY_LIMIT = 10
RESPONSE_NUM_PREDICT = int(os.getenv("MMIS_RESPONSE_NUM_PREDICT", "180"))


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


MMIS_CHAT_FAST = _get_env_bool("MMIS_CHAT_FAST", False)
MMIS_CHAT_RECALL_RESULTS = _get_env_int("MMIS_CHAT_RECALL_RESULTS", 5 if not MMIS_CHAT_FAST else 0)
MMIS_CHAT_EVENTS_LIMIT = _get_env_int("MMIS_CHAT_EVENTS_LIMIT", 20 if not MMIS_CHAT_FAST else 6)
MMIS_CHAT_ALLOW_REWRITE = _get_env_bool("MMIS_CHAT_ALLOW_REWRITE", True if not MMIS_CHAT_FAST else False)


# Базовые профили Ollama. Можно выбрать через MMIS_PROFILE=QUALITY|BALANCED|FAST.
# Рекомендации по железу:
# - CPU-only: FAST (num_ctx=2048..4096, num_thread ~= количеству физических ядер,
#   num_gpu=0, num_batch=32..64).
# - GPU (8+ GB VRAM): BALANCED/QUALITY (num_gpu=1, num_ctx=4096..8192,
#   num_batch=128..256, keep_alive>=600 для тёплой модели между запросами).
OLLAMA_PROFILES = {
    "QUALITY": {
        "num_thread": 10,
        "num_ctx": 4096,
        "num_gpu": 35,
        "num_batch": 256,
        "repeat_penalty": 1.12,
        "temperature": 0.55,
        "top_p": 0.9,
        "keep_alive": "30m",
    },
    "BALANCED": {
        "num_thread": 8,
        "num_ctx": 4096,
        "num_gpu": 30,
        "num_batch": 128,
        "repeat_penalty": 1.15,
        "temperature": 0.6,
        "top_p": 0.9,
        "keep_alive": "10m",
    },
    "FAST": {
        "num_thread": 6,
        "num_ctx": 2048,
        "num_gpu": 20,
        "num_batch": 64,
        "repeat_penalty": 1.1,
        "temperature": 0.65,
        "top_p": 0.92,
        "keep_alive": "5m",
    },
    # Hybrid profile: VRAM + RAM.
    # Uses most layers on GPU, but intentionally leaves part for system RAM spill.
    "HYBRID_RAM": {
        "num_thread": 10,
        "num_ctx": 8192,
        "num_gpu": 24,
        "num_batch": 128,
        "repeat_penalty": 1.12,
        "temperature": 0.58,
        "top_p": 0.9,
        "keep_alive": "15m",
    },
}

OLLAMA_PROFILE = os.getenv("MMIS_PROFILE", "HYBRID_RAM").upper()
if OLLAMA_PROFILE not in OLLAMA_PROFILES:
    OLLAMA_PROFILE = "HYBRID_RAM"

# Общие (активные) опции с учётом профиля и env override'ов.
OLLAMA_OPTIONS = {
    **OLLAMA_PROFILES[OLLAMA_PROFILE],
    "num_thread": _get_env_int("MMIS_NUM_THREAD", OLLAMA_PROFILES[OLLAMA_PROFILE]["num_thread"]),
    "num_ctx": _get_env_int("MMIS_NUM_CTX", OLLAMA_PROFILES[OLLAMA_PROFILE]["num_ctx"]),
    "num_gpu": _get_env_int("MMIS_NUM_GPU", OLLAMA_PROFILES[OLLAMA_PROFILE]["num_gpu"]),
    "num_batch": _get_env_int("MMIS_NUM_BATCH", OLLAMA_PROFILES[OLLAMA_PROFILE]["num_batch"]),
    "repeat_penalty": _get_env_float(
        "MMIS_REPEAT_PENALTY", OLLAMA_PROFILES[OLLAMA_PROFILE]["repeat_penalty"]
    ),
    "temperature": _get_env_float("MMIS_TEMPERATURE", OLLAMA_PROFILES[OLLAMA_PROFILE]["temperature"]),
    "top_p": _get_env_float("MMIS_TOP_P", OLLAMA_PROFILES[OLLAMA_PROFILE]["top_p"]),
    "keep_alive": os.getenv("MMIS_KEEP_ALIVE", OLLAMA_PROFILES[OLLAMA_PROFILE]["keep_alive"]),
}


def build_ollama_options(task_type: str) -> dict:
    """Собирает опции Ollama под конкретный тип задачи."""
    options = dict(OLLAMA_OPTIONS)

    # Быстрые «детерминированные» извлечения для памяти.
    if task_type in {"fact_extraction", "event_extraction", "assistant_fact_extraction"}:
        options.update(
            {
                "temperature": _get_env_float("MMIS_EXTRACT_TEMPERATURE", 0.0),
                "top_p": _get_env_float("MMIS_EXTRACT_TOP_P", 1.0),
                "repeat_penalty": _get_env_float("MMIS_EXTRACT_REPEAT_PENALTY", 1.0),
                "num_predict": _get_env_int("MMIS_EXTRACT_NUM_PREDICT", 256),
            }
        )
    else:
        options["num_predict"] = _get_env_int("MMIS_NUM_PREDICT", RESPONSE_NUM_PREDICT)

    return options
