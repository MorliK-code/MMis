from __future__ import annotations

import os


from config.settings import load_config

_cfg = load_config()

METADATA_MODEL = _cfg.metadata_model
METADATA_MODEL_FALLBACKS = _cfg.metadata_model_fallbacks

