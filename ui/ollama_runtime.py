from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any
from urllib import request as urllib_request


_OWNED_OLLAMA_PROCESS: subprocess.Popen | None = None


def _creationflags() -> int:
    if not sys.platform.startswith("win"):
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _startupinfo() -> subprocess.STARTUPINFO | None:
    if not sys.platform.startswith("win"):
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return info


def is_ollama_alive(base_url: str = "http://127.0.0.1:11434", timeout: float = 0.5) -> bool:
    base_url = str(base_url or "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib_request.urlopen(base_url + "/api/tags", timeout=timeout) as resp:
            return 200 <= int(resp.status) < 500
    except Exception:
        return False


def start_ollama_serve() -> tuple[bool, str]:
    global _OWNED_OLLAMA_PROCESS

    if _OWNED_OLLAMA_PROCESS is not None and _OWNED_OLLAMA_PROCESS.poll() is None:
        return True, "Ollama already starting"

    try:
        _OWNED_OLLAMA_PROCESS = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=_creationflags(),
            startupinfo=_startupinfo(),
            close_fds=not sys.platform.startswith("win"),
            env=dict(os.environ),
        )
        return True, "Ollama start requested"
    except FileNotFoundError:
        return False, "Команда `ollama` не найдена в PATH. Установи Ollama или добавь её в PATH."
    except Exception as exc:
        return False, str(exc)


def ensure_ollama_started(base_url: str = "http://127.0.0.1:11434", wait_sec: float = 8.0) -> tuple[bool, str]:
    if is_ollama_alive(base_url):
        return True, "Ollama already running"

    ok, message = start_ollama_serve()
    if not ok:
        return False, message

    deadline = time.time() + max(1.0, float(wait_sec))
    while time.time() < deadline:
        if is_ollama_alive(base_url, timeout=0.5):
            return True, "Ollama started"
        time.sleep(0.35)

    return False, "Ollama process started, but API is still unavailable"
