from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"
DEFAULT_MEMORY_DIR = DATA_DIR / "memory_storage"
LEGACY_MEMORY_DIR = BASE_DIR / "memory_storage"


def _from_env_path(name: str) -> Path | None:
    raw = str(os.getenv(name, "")).strip()
    return Path(raw).expanduser() if raw else None


def resolve_memory_dir() -> Path:
    env_path = _from_env_path("MMIS_MEMORY_DIR")
    if env_path is not None:
        return env_path

    use_legacy = str(os.getenv("MMIS_USE_LEGACY_MEMORY_DIR", "")).strip().lower() in {"1", "true", "yes", "on"}
    if use_legacy:
        return LEGACY_MEMORY_DIR
    return DEFAULT_MEMORY_DIR


MEMORY_DIR = resolve_memory_dir()
CHROMA_DIR = MEMORY_DIR / "chroma_db"


def ensure_dirs(memory_dir: str | Path | None = None) -> dict[str, Path]:
    mem_dir = Path(memory_dir).expanduser() if memory_dir is not None else resolve_memory_dir()
    chroma_dir = mem_dir / "chroma_db"
    dirs = {
        "base": BASE_DIR,
        "config": CONFIG_DIR,
        "data": DATA_DIR,
        "models": MODELS_DIR,
        "memory": mem_dir,
        "logs": LOG_DIR,
        "cache": CACHE_DIR,
        "chroma": chroma_dir,
        "profiles": mem_dir / "profiles",
        "summaries": mem_dir / "summaries",
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


# Backward compatibility aliases.
ROOT_DIR = BASE_DIR
LOGS_DIR = LOG_DIR
NEW_MEMORY_DIR = DEFAULT_MEMORY_DIR


def ensure_data_dirs() -> None:
    ensure_dirs()
