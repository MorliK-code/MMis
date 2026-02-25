from pathlib import Path

basedir = Path(__file__).parent
MemoryStorageDir = basedir / "memory_storage"

MODEL_NAME = "mistral:7b"
EMBED_MODEL = "nomic-embed-text"
SHORT_MEMORY_LIMIT = 30