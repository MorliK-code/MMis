"""Run MMis API server."""

from __future__ import annotations

import uvicorn

from config.logging_config import setup_logging
from config.settings import load_config


if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg)
    uvicorn.run("api.app:app", host=cfg.host, port=cfg.port, reload=False)
