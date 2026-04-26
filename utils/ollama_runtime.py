from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib import request as urllib_request


_OWNED_OLLAMA_PROCESS: subprocess.Popen | None = None
_START_LOCK = threading.Lock()
_LAST_START_ATTEMPT_AT = 0.0
_LAST_START_MODE = ""
_START_COOLDOWN_SEC = 6.0


def _creationflags() -> int:
    if not sys.platform.startswith("win"):
        return 0

    flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return flags


def _startupinfo() -> subprocess.STARTUPINFO | None:
    if not sys.platform.startswith("win"):
        return None

    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return info


def _clean_base_url(base_url: str) -> str:
    return str(base_url or "http://127.0.0.1:11434").strip().rstrip("/") or "http://127.0.0.1:11434"


def is_ollama_alive(base_url: str = "http://127.0.0.1:11434", timeout: float = 0.5) -> bool:
    base_url = _clean_base_url(base_url)
    try:
        with urllib_request.urlopen(base_url + "/api/tags", timeout=timeout) as resp:
            return 200 <= int(resp.status) < 500
    except Exception:
        return False


def _build_env(models_dir: str = "") -> dict[str, str]:
    env = dict(os.environ)
    models_dir = str(models_dir or "").strip().strip('"')
    if models_dir:
        env["OLLAMA_MODELS"] = str(Path(models_dir).expanduser())
    return env


def _serve_exe_or_default(serve_exe: str = "") -> str:
    raw = str(serve_exe or "").strip().strip('"')
    if raw:
        return raw
    return shutil.which("ollama") or "ollama"


def _popen_serve(serve_exe: str, models_dir: str) -> tuple[bool, str]:
    global _OWNED_OLLAMA_PROCESS

    if _OWNED_OLLAMA_PROCESS is not None and _OWNED_OLLAMA_PROCESS.poll() is None:
        return True, "Ollama serve process already owned by MMis"

    exe = _serve_exe_or_default(serve_exe)

    try:
        _OWNED_OLLAMA_PROCESS = subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=_creationflags(),
            startupinfo=_startupinfo(),
            close_fds=not sys.platform.startswith("win"),
            env=_build_env(models_dir),
        )
        return True, "Ollama serve start requested"
    except FileNotFoundError:
        return False, f"ollama exe не найден: {exe}"
    except Exception as exc:
        return False, str(exc)


def _run_ollama_list(models_dir: str, timeout: float = 8.0) -> tuple[bool, str]:
    exe = shutil.which("ollama") or "ollama"

    try:
        result = subprocess.run(
            [exe, "list"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=_creationflags(),
            startupinfo=_startupinfo(),
            env=_build_env(models_dir),
            timeout=max(2.0, float(timeout)),
            check=False,
        )
        if result.returncode == 0:
            return True, "ollama list completed"
        return False, f"ollama list exited with code {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "ollama list timeout"
    except FileNotFoundError:
        return False, "Команда `ollama` не найдена в PATH"
    except Exception as exc:
        return False, str(exc)


def ensure_ollama_started(
    *,
    base_url: str = "http://127.0.0.1:11434",
    enabled: bool = True,
    start_mode: str = "serve",
    serve_exe: str = "",
    models_dir: str = "",
    wait_sec: float = 8.0,
    force: bool = False,
) -> tuple[bool, str]:
    global _LAST_START_ATTEMPT_AT, _LAST_START_MODE

    base_url = _clean_base_url(base_url)

    if is_ollama_alive(base_url, timeout=0.5):
        return True, "Ollama API already running"

    if not bool(enabled):
        return False, "Ollama autostart disabled"

    mode = str(start_mode or "serve").strip().lower()
    if mode not in {"serve", "ui"}:
        mode = "serve"

    now = time.monotonic()

    with _START_LOCK:
        if is_ollama_alive(base_url, timeout=0.5):
            return True, "Ollama API already running"

        recently_tried = (now - _LAST_START_ATTEMPT_AT) < _START_COOLDOWN_SEC
        same_mode = _LAST_START_MODE == mode

        if recently_tried and same_mode and not force:
            return False, "Ollama start already attempted recently"

        _LAST_START_ATTEMPT_AT = now
        _LAST_START_MODE = mode

        if mode == "ui":
            ok, message = _run_ollama_list(models_dir=models_dir, timeout=wait_sec)
        else:
            ok, message = _popen_serve(serve_exe=serve_exe, models_dir=models_dir)

        if not ok:
            return False, message

    deadline = time.monotonic() + max(1.0, float(wait_sec))
    while time.monotonic() < deadline:
        if is_ollama_alive(base_url, timeout=0.5):
            return True, "Ollama API started"
        time.sleep(0.35)

    return False, f"{message}; но /api/tags всё ещё недоступен"
