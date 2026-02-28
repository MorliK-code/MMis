from __future__ import annotations

import os


def _env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


def _env_list(name: str, default_csv: str) -> list[str]:
    raw = os.getenv(name, default_csv)
    return [x.strip() for x in str(raw).split(",") if x.strip()]


METADATA_MODEL = _env_str("MMIS_METADATA_MODEL", "qwen3:1.7b")
METADATA_MODEL_FALLBACKS = _env_list("MMIS_METADATA_MODEL_FALLBACKS", "phi3:mini,llama3.2:1b")

