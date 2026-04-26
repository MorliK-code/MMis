# MMis — единое временное хранилище UI: чат, персонаж, настройки

## Цель

UI должен работать как переносимая папка. Если API хоть раз был доступен, UI должен сохранить локально:

- последний чат;
- активную тему;
- последнее имя персонажа;
- настройки/кеш настроек;
- адрес подключения к API.

При следующем запуске без API приложение не должно показывать пустоту. Оно должно брать последние известные данные из:

```text
ui/.mmis_client/
```

---

# 1. Основной принцип

Нужно считать `ui/.mmis_client` локальным временным хранилищем клиента.

Структура должна быть такой:

```text
ui/.mmis_client/
  client_config.json       # настройки, кеш настроек, pending_updates
  ui_state.json            # имя персонажа, активная тема, runtime-флаги
  ui_chats/
    index.json             # список локальных чатов / активный чат
    visible-main-chat/
      chat.json            # история видимого чата
```

Не надо хранить это в `data/`, потому что при переносе только папки `ui` всё потеряется.

---

# 2. Файл: `ui/client_config_store.py`

## 2.1. Добавить пути для UI-state

После:

```python
CLIENT_CONFIG_PATH = CLIENT_DATA_DIR / "client_config.json"
```

добавить:

```python
UI_STATE_PATH = CLIENT_DATA_DIR / "ui_state.json"
```

---

## 2.2. Добавить дефолтный UI-state

После `DEFAULT_CONFIG` добавить:

```python
DEFAULT_UI_STATE = {
    "think_enabled": True,
    "verbose_enabled": False,
    "json_mode_enabled": False,
    "screen_enabled": False,
    "web_mode": "auto",
    "active_topic_title": "",
    "last_persona_name": "Default",
}
```

---

## 2.3. Добавить функции загрузки/сохранения UI-state

В конец файла добавить:

```python
def load_ui_state() -> dict[str, Any]:
    """Loads local portable UI state from ui/.mmis_client/ui_state.json."""
    if not UI_STATE_PATH.exists():
        return dict(DEFAULT_UI_STATE)

    try:
        with UI_STATE_PATH.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULT_UI_STATE)
        out = dict(DEFAULT_UI_STATE)
        out.update(data)
        return out
    except Exception:
        return dict(DEFAULT_UI_STATE)


def save_ui_state(data: dict[str, Any]) -> dict[str, Any]:
    """Saves local portable UI state to ui/.mmis_client/ui_state.json."""
    out = dict(DEFAULT_UI_STATE)
    if isinstance(data, dict):
        out.update(data)

    try:
        CLIENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with UI_STATE_PATH.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return out


def merge_ui_state(updates: dict[str, Any]) -> dict[str, Any]:
    state = load_ui_state()
    if isinstance(updates, dict):
        state.update(updates)
    return save_ui_state(state)


def get_last_persona_name() -> str:
    state = load_ui_state()
    name = str(state.get("last_persona_name") or "").strip()
    return name or "Default"


def set_last_persona_name(name: str) -> None:
    clean = str(name or "").strip()
    if not clean:
        return
    merge_ui_state({"last_persona_name": clean})
```

---

# 3. Файл: `ui/chat_window.py`

## 3.1. Импортировать UI-state helpers

Найти импорт из `ui.client_config_store`.

Сделать так, чтобы там были:

```python
from ui.client_config_store import (
    CLIENT_DATA_DIR,
    get_connection_config,
    load_ui_state,
    save_ui_state,
    set_last_persona_name,
)
```

Если `get_connection_config` уже импортирован отдельно — просто добавь `load_ui_state`, `save_ui_state`, `set_last_persona_name`.

---

## 3.2. Убрать отдельный путь `_ui_state_path`

В `__init__` можно оставить:

```python
self._ui_state_path = CLIENT_DATA_DIR / "ui_state.json"
```

Но лучше дальше его уже не использовать напрямую, а читать через `load_ui_state()` и писать через `save_ui_state()`.

---

## 3.3. Исправить `_load_ui_state()`

Полностью заменить метод:

```python
def _load_ui_state(self) -> None:
    try:
        if not self._ui_state_path.exists():
            return
        payload = json.loads(self._ui_state_path.read_text(encoding="utf-8-sig") or "{}")
    except Exception:
        return
    self._thinking_enabled = bool(payload.get("think_enabled", self._thinking_enabled))
    self._verbose_enabled = bool(payload.get("verbose_enabled", self._verbose_enabled))
    self._json_mode_enabled = bool(payload.get("json_mode_enabled", self._json_mode_enabled))
    self._screen_enabled = bool(payload.get("screen_enabled", self._screen_enabled))
    self._web_mode = str(payload.get("web_mode") or self._web_mode)
    self._last_active_topic_title = str(payload.get("active_topic_title") or "")
    saved_persona = str(payload.get("last_persona_name") or "").strip()
    if saved_persona in {"", "Asya", "Ася", "asya"}:
        self._last_persona_name = "Default"
    else:
        self._last_persona_name = saved_persona
```

на:

```python
def _load_ui_state(self) -> None:
    payload = load_ui_state()

    self._thinking_enabled = bool(payload.get("think_enabled", self._thinking_enabled))
    self._verbose_enabled = bool(payload.get("verbose_enabled", self._verbose_enabled))
    self._json_mode_enabled = bool(payload.get("json_mode_enabled", self._json_mode_enabled))
    self._screen_enabled = bool(payload.get("screen_enabled", self._screen_enabled))
    self._web_mode = str(payload.get("web_mode") or self._web_mode)
    self._last_active_topic_title = str(payload.get("active_topic_title") or "")

    saved_persona = str(payload.get("last_persona_name") or "").strip()
    self._last_persona_name = saved_persona or "Default"
```

Важно: больше нельзя превращать `Ася` в `Default`. Если API когда-то отдал `Ася`, значит это и есть последнее известное имя.

---

## 3.4. Исправить `_save_ui_state()`

Полностью заменить метод:

```python
def _save_ui_state(self) -> None:
    payload = {
        "think_enabled": bool(self._thinking_enabled),
        "verbose_enabled": bool(self._verbose_enabled),
        "json_mode_enabled": bool(self._json_mode_enabled),
        "screen_enabled": bool(self._screen_enabled),
        "web_mode": str(self._web_mode),
        "active_topic_title": str(getattr(self, "_last_active_topic_title", "")),
        "last_persona_name": str(getattr(self, "_last_persona_name", "Default") or "Default"),
    }
    try:
        self._ui_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._ui_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
```

на:

```python
def _save_ui_state(self) -> None:
    payload = {
        "think_enabled": bool(self._thinking_enabled),
        "verbose_enabled": bool(self._verbose_enabled),
        "json_mode_enabled": bool(self._json_mode_enabled),
        "screen_enabled": bool(self._screen_enabled),
        "web_mode": str(self._web_mode),
        "active_topic_title": str(getattr(self, "_last_active_topic_title", "")),
        "last_persona_name": str(getattr(self, "_last_persona_name", "Default") or "Default"),
    }
    save_ui_state(payload)
```

---

## 3.5. Добавить безопасное обновление имени персонажа

Внутри класса `ChatWindow`, рядом с `_refresh_persona_label()`, добавить:

```python
def _cache_persona_name(self, name: str) -> None:
    clean = str(name or "").strip()
    if not clean:
        return

    # Не перетираем уже известное имя пустым/служебным Default,
    # если раньше уже было нормальное имя.
    if clean == "Default" and str(getattr(self, "_last_persona_name", "") or "").strip() not in {"", "Default"}:
        return

    self._api_persona_name = clean
    self._last_persona_name = clean
    set_last_persona_name(clean)
    self._refresh_persona_label()
```

---

## 3.6. Исправить `_sync_runtime_controls()`

Найти:

```python
persona_name = str(payload.get("persona_name") or "Default").strip() or "Default"
self._api_persona_name = persona_name
self._last_persona_name = persona_name
self._refresh_persona_label()
```

заменить на:

```python
persona_name = str(payload.get("persona_name") or "").strip()
if persona_name:
    self._cache_persona_name(persona_name)
```

---

## 3.7. Исправить `_on_backend_status_result()`

Найти:

```python
persona_name = str(row.get("persona_name") or "Default").strip() or "Default"
self._api_persona_name = persona_name
self._last_persona_name = persona_name
self._refresh_persona_label()
```

заменить на:

```python
persona_name = str(row.get("persona_name") or "").strip()
if persona_name:
    self._cache_persona_name(persona_name)
```

Так UI не будет перетирать сохранённую `Ася` на `Default`, если `/health` временно не отдал имя персонажа.

---

## 3.8. Исправить `_resolve_persona_display_name()`

Сейчас метод пытается читать персонажа из серверных путей:

```python
MemoryStorageDir / "characters_runtime" / character_id / "character.json"
DATA_DIR / "specs" / "characters" / character_id / "character.json"
```

Для portable UI это плохой источник. Его можно оставить как дополнительный fallback, но порядок должен быть такой:

1. `_api_persona_name`, если API сейчас отдал имя;
2. `_last_persona_name`, если оно есть в `ui/.mmis_client/ui_state.json`;
3. попытка прочитать локальные файлы полного проекта;
4. `Default`.

В начало `_resolve_persona_display_name()` после проверки `_api_persona_name` добавить:

```python
cached = str(getattr(self, "_last_persona_name", "") or "").strip()
if cached and cached != "Default":
    return cached
```

И в конце оставить:

```python
return self._last_persona_name or "Default"
```

---

# 4. Файл: `ui/workers.py`

## 4.1. Не отправлять `Default`, если API не дал имя

В `StatusPollWorker.run()` найти:

```python
payload["persona_name"] = str(health.get("persona_name") or "").strip()
```

Это нормально, оставь так.

Но если где-то стоит:

```python
payload["persona_name"] = str(health.get("persona_name") or "Default").strip()
```

заменить на:

```python
payload["persona_name"] = str(health.get("persona_name") or "").strip()
```

То же самое проверить в `ReplyWorker.run()`.

---

# 5. Файл: `ui/chat_sessions.py`

Этот файл уже сохраняет чат в `ui/.mmis_client/ui_chats/`.

Проверить, что `ChatWindow.__init__` использует именно:

```python
self._sessions_dir = CLIENT_DATA_DIR / "ui_chats"
self._sessions_index_path = self._sessions_dir / "index.json"
self._legacy_sessions_path = CLIENT_DATA_DIR / "ui_chats.json"
```

Это правильно.

---

# 6. Файл: `ui/chat_window.py` — чат должен сохраняться после каждого изменения

Проверить, что `_save_chat_sessions()` вызывается после:

- отправки сообщения пользователя;
- получения ответа ассистента;
- очистки чата;
- регенерации ответа;
- удаления/изменения сообщений;
- смены темы, если тема хранится в UI-state.

Минимально важно, чтобы после добавления сообщения пользователя было:

```python
self._history.append(("user", text, None, None, None))
self._save_chat_sessions()
```

И после завершения ответа ассистента было:

```python
self._history.append(("ai", answer, stat_line, None, thinking))
self._save_chat_sessions()
```

---

# 7. Важное поведение после исправления

## Первый запуск без API

```text
имя персонажа: Default
чат: пустой
настройки: из schema/default
адрес API: http://127.0.0.1:8027
```

## Было подключение к API, API отдал имя `Ася`

Сохраняется:

```json
{
  "last_persona_name": "Ася"
}
```

## Второй запуск без API

UI должен показать:

```text
имя персонажа: Ася
чат: последний локальный чат
настройки: последний кеш + pending_updates
статистика API: --
статус API: offline
```

## API снова появился

UI должен:

```text
1. обновить имя персонажа из API, если API реально отдал имя;
2. обновить server_snapshot настроек;
3. отправить pending_updates;
4. не затирать локальный кеш пустыми значениями.
```

---

# 8. Главная ошибка, которую нужно убрать

Нельзя делать так:

```python
str(payload.get("persona_name") or "Default")
```

потому что это перетирает локальный кеш, когда сервер временно не вернул имя.

Правильно:

```python
persona_name = str(payload.get("persona_name") or "").strip()
if persona_name:
    self._cache_persona_name(persona_name)
```

---

# 9. Итог

`ui/.mmis_client` должен стать локальной памятью клиента:

```text
settings → client_config.json
chat     → ui_chats/
persona  → ui_state.json
runtime  → ui_state.json
```

Тогда UI сможет запускаться отдельно и не будет терять данные без API.
