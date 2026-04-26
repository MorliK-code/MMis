# MMis — исправить автозапуск Ollama и уведомляющие окна в настройках провайдера

Проверено по последнему архиву `MMis.zip`.

Проблемы сейчас:

1. `Auto start Ollama` есть как настройка, но в desktop UI она только сохраняет значение. Она **не запускает Ollama**.
2. Сам автозапуск Ollama реализован только в `ui_console.py`, а не в PySide6 UI.
3. В `SettingsWindow._save()` используется стандартный `QMessageBox`, поэтому окно белое и выбивается из стиля MMis.
4. Настройка `ui.console.auto_start_ollama` сейчас находится в секции console/system, а тебе она нужна в **настройках провайдера Ollama**.

---

## 1. Перенести выключатель автозапуска Ollama в карточку провайдера

Файл:

```text
ui/settings_schema.py
```

### Найди карточку `Ollama`

Сейчас там примерно так:

```python
SettingCard(
    title="Ollama",
    tag="providers.ollama",
    settings=(
        SettingSpec("llm.providers.ollama.base_url", "Base URL", example="http://127.0.0.1:11434", restart_required=True),
        SettingSpec("llm.providers.ollama.timeout_sec", "Timeout sec", kind="float", example="120.0"),
        SettingSpec("llm.providers.ollama.retries", "Retries", kind="int", example="1"),
    ),
),
```

### Замени на:

```python
SettingCard(
    title="Ollama",
    tag="providers.ollama",
    settings=(
        SettingSpec(
            "ui.console.auto_start_ollama",
            "Auto start Ollama",
            kind="bool",
            description="Automatically start `ollama serve` from the desktop UI when Ollama is unavailable.",
            example="true",
            live=True,
        ),
        SettingSpec("llm.providers.ollama.base_url", "Base URL", example="http://127.0.0.1:11434", restart_required=True),
        SettingSpec("llm.providers.ollama.timeout_sec", "Timeout sec", kind="float", example="120.0"),
        SettingSpec("llm.providers.ollama.retries", "Retries", kind="int", example="1"),
    ),
),
```

### Потом убери этот же пункт из карточки Console

В этом же файле найди:

```python
SettingSpec("ui.console.auto_start_ollama", "Auto start Ollama", kind="bool", example="false", restart_required=True),
```

и удали его из старой секции, чтобы настройка не дублировалась.

---

## 2. Исправить названия и подсказки

Файл:

```text
ui/settings_window.py
```

### В `_TITLE_BY_PATH` оставь/добавь:

```python
"ui.console.auto_start_ollama": "Автозапуск Ollama",
```

### В `_DESCRIPTION_BY_PATH` замени описание на:

```python
"ui.console.auto_start_ollama": "Если включено, desktop UI попробует запустить `ollama serve`, когда Ollama недоступна.",
```

---

## 3. Добавить запуск Ollama для desktop UI

Создай новый файл:

```text
ui/ollama_runtime.py
```

Вставь туда:

```python
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
```

---

## 4. При сохранении настройки реально запускать Ollama

Файл:

```text
ui/settings_window.py
```

### Вверху добавь импорт

Рядом с импортами `ui.*` добавь:

```python
from ui.ollama_runtime import ensure_ollama_started
```

### В методе `_save()` найди этот блок

```python
self._refresh_preview()
self.saved.emit(copy.deepcopy(updates))

if not online:
    QMessageBox.information(
        self,
        "Offline settings",
        f"API сейчас недоступен ({message}).\n"
        "Изменения сохранены локально и будут отправлены после подключения."
    )
```

### Замени на:

```python
self._refresh_preview()
self.saved.emit(copy.deepcopy(updates))

self._maybe_start_ollama_after_save(updates)

if not online:
    SettingsMessageBox.information(
        self,
        "Offline settings",
        f"API сейчас недоступен ({message}).\n"
        "Изменения сохранены локально и будут отправлены после подключения.",
    )
```

### В класс `SettingsWindow` добавь метод

Вставь ниже `_save()`:

```python
def _maybe_start_ollama_after_save(self, updates: dict[str, Any]) -> None:
    enabled = updates.get("ui.console.auto_start_ollama")
    if enabled is None:
        return
    if not bool(enabled):
        return

    base_url = str(dotted_get(self._payload, "llm.providers.ollama.base_url", "http://127.0.0.1:11434") or "http://127.0.0.1:11434")
    ok, message = ensure_ollama_started(base_url=base_url, wait_sec=8.0)

    if ok:
        SettingsMessageBox.information(
            self,
            "Ollama",
            "Ollama запущена или уже была активна.",
        )
    else:
        SettingsMessageBox.warning(
            self,
            "Ollama",
            f"Не удалось запустить Ollama.\n{message}",
        )
```

Теперь выключатель будет не просто сохраняться, а при включении будет пробовать поднять `ollama serve`.

---

## 5. Добавить автозапуск Ollama при старте desktop UI

Файл:

```text
ui/chat_window.py
```

### Вверху добавь импорт

```python
from config.settings import load_config
from ui.ollama_runtime import ensure_ollama_started
```

Если `load_config` уже где-то импортирован — второй раз не добавляй.

### В `ChatWindow.__init__()` после создания `self.api = ApiClient()` добавь

Найди:

```python
if self.api is None:
    self.api = ApiClient()
```

Сразу после него добавь:

```python
self._ensure_ollama_autostart_on_ui_boot()
```

### В класс `ChatWindow` добавь метод

```python
def _ensure_ollama_autostart_on_ui_boot(self) -> None:
    try:
        cfg = load_config()
        if not bool(getattr(cfg, "console_auto_start_ollama", False)):
            return
        if str(getattr(cfg, "llm_default_provider", "ollama") or "ollama").lower() not in {"ollama", "auto"}:
            return
        base_url = str(getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434") or "http://127.0.0.1:11434")
        QTimer.singleShot(300, lambda: ensure_ollama_started(base_url=base_url, wait_sec=1.0))
    except Exception:
        pass
```

Важно: тут используется `QTimer.singleShot`, чтобы UI не подвисал при запуске окна.

---

## 6. Сделать уведомляющие окна в стиле MMis

Файл:

```text
ui/settings_window.py
```

Сейчас используется стандартный `QMessageBox`, из-за этого окно белое.

### Добавь класс `SettingsMessageBox`

Вставь после класса `SettingsHintPopup` или перед `SettingsWindow`:

```python
class SettingsMessageBox(QDialog):
    def __init__(self, parent: QWidget | None, title: str, text: str, *, kind: str = "info"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setObjectName("settings_message_box")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)

        icon = QLabel("i" if kind == "info" else "!")
        icon.setObjectName("settings_message_icon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedSize(30, 30)
        header.addWidget(icon)

        title_label = QLabel(title)
        title_label.setObjectName("settings_message_title")
        header.addWidget(title_label, 1)
        root.addLayout(header)

        body = QLabel(text)
        body.setObjectName("settings_message_body")
        body.setWordWrap(True)
        root.addWidget(body)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("primary_button")
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(ok_btn)
        root.addLayout(buttons)

    @staticmethod
    def information(parent: QWidget | None, title: str, text: str) -> None:
        SettingsMessageBox(parent, title, text, kind="info").exec()

    @staticmethod
    def warning(parent: QWidget | None, title: str, text: str) -> None:
        SettingsMessageBox(parent, title, text, kind="warning").exec()
```

### Замени стандартные `QMessageBox`

В `_save()` замени:

```python
QMessageBox.warning(self, "Settings validation", "\n".join(errors))
```

на:

```python
SettingsMessageBox.warning(self, "Settings validation", "\n".join(errors))
```

В `_open_settings_window()` в `chat_window.py` можно оставить стандартный `QMessageBox`, но лучше тоже заменить позже отдельным общим классом. Главная проблема сейчас именно в `SettingsWindow._save()`.

---

## 7. Добавить стили для нового окна уведомлений

Файл:

```text
ui/settings_styles.py
```

В конец `SETTINGS_STYLE` добавь:

```css
QDialog#settings_message_box {
    background: rgba(15, 16, 24, 248);
    border: 1px solid rgba(139, 92, 246, 82);
    border-radius: 14px;
}

QLabel#settings_message_icon {
    color: #ede9fe;
    background: rgba(139, 92, 246, 58);
    border: 1px solid rgba(139, 92, 246, 92);
    border-radius: 15px;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 17px;
    font-weight: 800;
}

QLabel#settings_message_title {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 13px;
    font-weight: 800;
}

QLabel#settings_message_body {
    color: #cbd5e1;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
    line-height: 1.35;
}
```

Если `primary_button` уже стилизован, отдельный стиль для OK не нужен.

---

## 8. Почему раньше не работало

`ui.console.auto_start_ollama` — это только значение в конфиге. В desktop UI не было кода, который при включении этого значения реально выполнял бы:

```text
ollama serve
```

В `ui_console.py` такой код есть, но PySide6 UI его не использует.

Поэтому выключатель визуально менялся, но Ollama не стартовала.

---

## 9. Быстрая проверка

1. Закрой Ollama полностью.
2. Открой MMis UI.
3. Перейди:

```text
Настройки -> LLM / Models -> Ollama
```

4. Включи:

```text
Автозапуск Ollama
```

5. Нажми `Сохранить`.
6. Должно появиться тёмное уведомление в стиле MMis.
7. Через несколько секунд проверь:

```powershell
ollama list
```

или открой:

```text
http://127.0.0.1:11434/api/tags
```

Если Ollama установлена и доступна в PATH, сервер должен подняться.

---

## 10. Важное уточнение

Этот фикс запускает **сервер Ollama**.

Он не обязан сразу загружать модель в VRAM. Модель загрузится только когда приложение отправит первый запрос или когда ты выберешь модель и начнёшь чат.
