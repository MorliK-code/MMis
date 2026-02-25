import os
from pathlib import Path

basedir = Path(__file__).parent
MemoryStorageDir = basedir / "memory_storage"

MODEL_NAME = "mistral:7b"
EMBED_MODEL = "nomic-embed-text"

SHORT_MEMORY_LIMIT = 10
RESPONSE_NUM_PREDICT = 60

FAST_MODE = os.getenv("FAST_MODE", "0").lower() in {"1", "true", "yes", "on"}
EXTRACTOR_TIMEOUT_SEC = float(os.getenv("EXTRACTOR_TIMEOUT_SEC", "1.8"))
