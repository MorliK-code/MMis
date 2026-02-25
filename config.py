import os
from pathlib import Path

basedir = Path(__file__).parent
MemoryStorageDir = basedir / "memory_storage"

MODEL_NAME = "mistral:7b"
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


# Базовые профили Ollama. Можно выбрать через MMIS_PROFILE=QUALITY|BALANCED|FAST.
# Рекомендации по железу:
# - CPU-only: FAST (num_ctx=2048..4096, num_thread ~= количеству физических ядер,
#   num_gpu=0, num_batch=32..64).
# - GPU (8+ GB VRAM): BALANCED/QUALITY (num_gpu=1, num_ctx=4096..8192,
#   num_batch=128..256, keep_alive>=600 для тёплой модели между запросами).
OLLAMA_PROFILES = {
    "QUALITY": {
        "num_thread": 8,
        "num_ctx": 4096,
        "num_gpu": 1,
        "num_batch": 256,
        "repeat_penalty": 1.12,
        "temperature": 0.55,
        "top_p": 0.9,
        "keep_alive": "30m",
    },
    "BALANCED": {
        "num_thread": 6,
        "num_ctx": 3072,
        "num_gpu": 1,
        "num_batch": 128,
        "repeat_penalty": 1.15,
        "temperature": 0.6,
        "top_p": 0.9,
        "keep_alive": "10m",
    },
    "FAST": {
        "num_thread": 6,
        "num_ctx": 2048,
        "num_gpu": 1,
        "num_batch": 64,
        "repeat_penalty": 1.1,
        "temperature": 0.65,
        "top_p": 0.92,
        "keep_alive": "5m",
    },
}

OLLAMA_PROFILE = os.getenv("MMIS_PROFILE", "BALANCED").upper()
if OLLAMA_PROFILE not in OLLAMA_PROFILES:
    OLLAMA_PROFILE = "BALANCED"

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
