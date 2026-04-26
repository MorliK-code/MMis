# MMis — фикс автозапуска Ollama: только `serve` и `ui`, плюс защита от повторного запуска

## Что надо сделать

Нужно убрать режим `server` и оставить только:

```text
serve — запуск API через <ollama.exe> serve
ui    — запуск API через ollama list
```

То есть в селекторе должно быть только:

```python
options=("serve", "ui")
```

Логика:

```text
нужна Ollama
→ проверить /api/tags
→ если уже работает: ничего не запускать
→ если не работает и mode=serve: запустить <ollama.exe> serve
→ если не работает и mode=ui: выполнить ollama list
→ дождаться /api/tags
```

---

## 1. Файл `ui/settings_schema.py`

Найти карточку:

```python
SettingCard(
    title="Ollama",
    tag="providers.ollama",
    settings=(
        ...
    ),
),
```

В ней заменить настройку `ui.ollama.start_mode`.

### Было

```python
SettingSpec(
    "ui.ollama.start_mode",
    "Ollama start mode",
    kind="select",
    options=("off", "server", "ui"),
    ...
),
```

или:

```python
options=("server", "ui")
```

### Должно быть

```python
SettingSpec(
    "ui.ollama.start_mode",
    "Ollama start mode",
    kind="select",
    options=("serve", "ui"),
    description="serve = запуск Ollama API через <ollama.exe> serve; ui = запуск через команду ollama list.",
    example="serve",
    live=True,
),
```

---

## 2. Файл `ui/settings_schema.py`

Переименовать поле:

```python
ui.ollama.server_exe
```

в:

```python
ui.ollama.serve_exe
```

### Было

```python
SettingSpec(
    "ui.ollama.server_exe",
    "Ollama server exe",
    ...
),
```

### Должно быть

```python
SettingSpec(
    "ui.ollama.serve_exe",
    "Ollama serve exe",
    description="Путь к ollama.exe для serve-режима. Например: C:\\Users\\user\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
    example="C:\\Users\\user\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
    live=True,
),
```

---

## 3. Файл `ui/settings_schema.py`

Убедиться, что в карточке `Ollama` есть только такие UI-настройки автозапуска:

```python
SettingSpec(
    "ui.console.auto_start_ollama",
    "Автозапуск Ollama",
    kind="bool",
    description="Разрешает desktop UI запускать Ollama по требованию, когда она недоступна.",
    example="true",
    live=True,
),
SettingSpec(
    "ui.ollama.start_mode",
    "Ollama start mode",
    kind="select",
    options=("serve", "ui"),
    description="serve = запуск Ollama API через <ollama.exe> serve; ui = запуск через команду ollama list.",
    example="serve",
    live=True,
),
SettingSpec(
    "ui.ollama.serve_exe",
    "Ollama serve exe",
    description="Путь к ollama.exe для serve-режима.",
    example="C:\\Users\\user\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
    live=True,
),
SettingSpec(
    "ui.ollama.models_dir",
    "Ollama models dir",
    description="Папка моделей Ollama. Передаётся в env как OLLAMA_MODELS.",
    example="D:\\ollama\\models",
    live=True,
),
SettingSpec("llm.providers.ollama.base_url", "Ollama URL", example="http://127.0.0.1:11434", restart_required=True),
SettingSpec("llm.providers.ollama.timeout_sec", "Ollama timeout", kind="float", example="120.0"),
SettingSpec("llm.providers.ollama.retries", "Ollama retries", kind="int", example="1"),
```

---

## 4. Файл `config/settings.py`

В `AppSettings` заменить:

```python
ui_ollama_start_mode: str = "server"
ui_ollama_server_exe: str = ""
ui_ollama_models_dir: str = ""
```

на:

```python
ui_ollama_start_mode: str = "serve"
ui_ollama_serve_exe: str = ""
ui_ollama_models_dir: str = ""
```

---

## 5. Файл `config/settings.py`

В `_default_config_tree()` заменить блок:

```python
"ollama": {
    "start_mode": "server",
    "server_exe": "",
    "models_dir": "",
},
```

на:

```python
"ollama": {
    "start_mode": "serve",
    "serve_exe": "",
    "models_dir": "",
},
```

---

## 6. Файл `config/settings.py`

В `_settings_from_payload()` заменить:

```python
ui_ollama_start_mode=_norm_lower(_get_dotted(row, "ui.ollama.start_mode") or "server"),
ui_ollama_server_exe=_norm_str(_get_dotted(row, "ui.ollama.server_exe") or ""),
ui_ollama_models_dir=_norm_str(_get_dotted(row, "ui.ollama.models_dir") or ""),
```

на:

```python
ui_ollama_start_mode=_norm_lower(_get_dotted(row, "ui.ollama.start_mode") or "serve"),
ui_ollama_serve_exe=_norm_str(_get_dotted(row, "ui.ollama.serve_exe") or ""),
ui_ollama_models_dir=_norm_str(_get_dotted(row, "ui.ollama.models_dir") or ""),
```

---

## 7. Файл `ui/settings_sync_service.py`

В `default_value_for_path()` заменить старые дефолты:

```python
if path == "ui.ollama.start_mode":
    return "server"
if path == "ui.ollama.server_exe":
    return ""
```

на:

```python
if path == "ui.ollama.start_mode":
    return "serve"
if path == "ui.ollama.serve_exe":
    return ""
if path == "ui.ollama.models_dir":
    return ""
```

---

## 8. Файл `ui/settings_sync_service.py`

В `save_settings_updates()` сделать `ui.ollama.*` локальными настройками.

Найти:

```python
remote_updates = {
    path: value
    for path, value in updates.items()
    if not str(path).startswith("ui.api.")
}
```

Заменить на:

```python
remote_updates = {
    path: value
    for path, value in updates.items()
    if not (
        str(path).startswith("ui.api.")
        or str(path).startswith("ui.ollama.")
    )
}
```

---

## 9. Файл `ui/ollama_runtime.py`

Полностью заменить файл на:

```python
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
```

---

## 10. Файл `ui/settings_window.py`

Убрать запуск Ollama при сохранении настроек.

Найти в `_save()`:

```python
self._maybe_start_ollama_after_save(updates)
```

Удалить строку.

Метод `_maybe_start_ollama_after_save()` тоже лучше удалить полностью.

---

## 11. Файл `ui/settings_widgets.py`

В `ModelListWorker.run()` заменить чтение старого `server_exe`.

### Найти

```python
server_exe = str(dotted_get(settings_payload, "ui.ollama.server_exe", "") or "")
```

### Заменить на

```python
serve_exe = str(dotted_get(settings_payload, "ui.ollama.serve_exe", "") or "")
```

### И вызов заменить

Было:

```python
ensure_ollama_started(
    base_url=base_url,
    enabled=auto_start,
    start_mode=start_mode,
    server_exe=server_exe,
    models_dir=models_dir,
    wait_sec=8.0,
)
```

Должно быть:

```python
ensure_ollama_started(
    base_url=base_url,
    enabled=auto_start,
    start_mode=start_mode,
    serve_exe=serve_exe,
    models_dir=models_dir,
    wait_sec=8.0,
)
```

---

## 12. Файл `ui/workers.py`

В `ReplyWorker.run()` перед `self.api.stream_chat(...)` использовать такие настройки:

```python
from ui.settings_sync_service import load_settings_payload, dotted_get
from ui.ollama_runtime import ensure_ollama_started

settings_payload, _meta = load_settings_payload(allow_remote=False)

provider = str(dotted_get(settings_payload, "llm.provider", "ollama") or "ollama").lower()
auto_start = bool(dotted_get(settings_payload, "ui.console.auto_start_ollama", True))

if provider in {"ollama", "auto"}:
    ensure_ollama_started(
        base_url=str(dotted_get(settings_payload, "llm.providers.ollama.base_url", "http://127.0.0.1:11434") or "http://127.0.0.1:11434"),
        enabled=auto_start,
        start_mode=str(dotted_get(settings_payload, "ui.ollama.start_mode", "serve") or "serve"),
        serve_exe=str(dotted_get(settings_payload, "ui.ollama.serve_exe", "") or ""),
        models_dir=str(dotted_get(settings_payload, "ui.ollama.models_dir", "") or ""),
        wait_sec=8.0,
    )
```

---

## 13. Важно для Ctrl+C

Не вызывать `ensure_ollama_started()` из:

```text
api_main.py
api/app.py
health endpoint
ping endpoint
shutdown hooks
memory worker shutdown
```

Автозапуск Ollama должен быть только в desktop UI:

```text
ui/settings_widgets.py — перед списком моделей
ui/workers.py          — перед отправкой сообщения
```

Иначе API может при завершении снова пытаться поднять Ollama и мешать Ctrl+C.

---

## Проверка

### Serve mode

Настройки:

```text
Автозапуск Ollama: on
Ollama start mode: serve
Ollama serve exe: C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe
Ollama models dir: D:\ollama\models
Ollama URL: http://127.0.0.1:11434
```

Ожидаемо:

```text
MMis запускает <ollama.exe> serve
OLLAMA_MODELS передаётся в env
/api/tags начинает отвечать
повторно процесс не создаётся
```

### UI mode

Настройки:

```text
Автозапуск Ollama: on
Ollama start mode: ui
Ollama models dir: D:\ollama\models
```

Ожидаемо:

```text
MMis выполняет ollama list
Ollama сама поднимает API
/api/tags начинает отвечать
повторно команда не спамится из-за cooldown
```

### Ctrl+C

Запустить:

```cmd
python api_main.py
```

Нажать:

```cmd
Ctrl+C
```

Ожидаемо:

```text
API завершается
новый запуск Ollama не стартует во время shutdown
Python-процесс не висит
```
