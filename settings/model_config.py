from __future__ import annotations

import os
from pathlib import Path

from config.settings import load_config


_cfg = load_config()

MemoryStorageDir = _cfg.memory_dir
MMIS_MEMORY_DB_PATH = _cfg.db_path
MODEL_NAME = _cfg.model_name
SHORT_MEMORY_LIMIT = int(os.getenv("MMIS_SHORT_MEMORY_LIMIT", "10"))

MMIS_VOICE_TTS_VOICE = str(os.getenv("MMIS_VOICE_TTS_VOICE", "ru-RU-DmitryNeural")).strip()
MMIS_VOICE_TTS_RATE = str(os.getenv("MMIS_VOICE_TTS_RATE", "+0%")).strip()
MMIS_VOICE_TTS_VOLUME = str(os.getenv("MMIS_VOICE_TTS_VOLUME", "+0%")).strip()
MMIS_VOICE_INPUT_DIR = Path(os.getenv("MMIS_VOICE_INPUT_DIR", str(MemoryStorageDir / "voice" / "input"))).expanduser()
MMIS_VOICE_OUTPUT_DIR = Path(os.getenv("MMIS_VOICE_OUTPUT_DIR", str(MemoryStorageDir / "voice" / "output"))).expanduser()
MMIS_VOICE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
MMIS_VOICE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MMIS_CHAT_RECALL_RESULTS = int(os.getenv("MMIS_CHAT_RECALL_RESULTS", "3"))
MMIS_CHAT_EVENTS_LIMIT = int(os.getenv("MMIS_CHAT_EVENTS_LIMIT", "10"))
MMIS_CHAT_PROOFREAD = str(os.getenv("MMIS_CHAT_PROOFREAD", "false")).lower() in {"1", "true", "yes", "on"}
MMIS_CHAT_PROOFREAD_STRICT = str(os.getenv("MMIS_CHAT_PROOFREAD_STRICT", "false")).lower() in {"1", "true", "yes", "on"}
MODEL_FALLBACKS = [x.strip() for x in str(os.getenv("MMIS_MODEL_FALLBACKS", "")).split(",") if x.strip()]


def build_ollama_options(task_type: str) -> dict:
    _ = task_type
    return {"temperature": 0.7, "num_predict": 768}
