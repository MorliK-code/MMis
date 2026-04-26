# План: standalone UI-клиент MMis с локальным кешем настроек и синхронизацией с API

## Что сейчас надо исправить

Сейчас `ui/` не является чистым независимым клиентом, потому что UI напрямую импортирует серверные/общие модули проекта:

- `ui/api_client.py` импортирует `config.settings.load_config`, `utils.logger`.
- `ui/settings_window.py` импортирует `config.settings.get_config_payload`, `update_config_values`.
- `ui/chat_window.py` импортирует `config.settings`, `llm.tokenizer`, `modules.voice.voice_manager`.
- `ui/voice_adapter.py` читает `get_config_payload()` напрямую из локального конфига проекта.
- Окно настроек сейчас сохраняет параметры прямо в локальный `config/config.json`, а не через API.

Из-за этого папку `ui/` нельзя просто вынести отдельно: ей нужны `config/`, `llm/`, `modules/`, `utils/` и часть серверной структуры.

---

## Цель

Сделать UI отдельным клиентом:

```text
MMis_UI/
  main.py
  ui/
  client/
  local/
  assets/
  requirements-ui.txt
```

UI должен уметь:

1. Запускаться без сервера API.
2. Хранить локально адрес подключения к серверу.
3. Показывать/активировать серверные настройки только после успешного подключения к API или при наличии последнего кеша схемы/значений.
4. Если API недоступен — не падать, а работать в offline-режиме.
5. Если пользователь изменил серверные параметры offline — складывать их в очередь pending-изменений.
6. После восстановления API — отправлять pending-изменения на сервер и обновлять локальный кеш.
7. Всегда оставлять активными настройки подключения: `host`, `port`, `base_url`, `api_key/session token` в будущем.

---

## Архитектура после исправления

```text
ui/
  app.py
  chat_window.py
  settings_window.py
  settings_schema.py
  settings_widgets.py
  ...

client/
  api_client.py              # только HTTP, без config.settings
  connection_store.py        # локальный адрес API
  settings_cache.py          # кеш последней серверной конфигурации
  settings_sync.py           # pending queue + flush на API
  client_logger.py           # локальный логгер UI
  tokenizer_stub.py          # estimate_tokens без llm.tokenizer
```

Локальные файлы клиента:

```text
%APPDATA%/MMis/ui_client/
  connection.json
  settings_cache.json
  pending_settings.json
  ui_state.json
  ui_chats/index.json
  logs/ui.log
```

На Windows можно получить путь через:

```python
Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming")) / "MMis" / "ui_client"
```

---

## Новые API endpoints на сервере

В `api/app.py` надо добавить endpoints для работы с конфигом:

```python
@app.get("/config")
def get_config() -> dict:
    return get_config_payload(force_reload=True)

@app.post("/config")
def update_config(req: ConfigUpdateRequest) -> dict:
    update_config_values(req.updates)
    return get_config_payload(force_reload=True)

@app.get("/config/schema")
def get_config_schema() -> dict:
    return build_settings_schema_payload()
```

Минимальная схема запроса:

```python
class ConfigUpdateRequest(BaseModel):
    updates: dict[str, Any]
```

Важно: `SETTINGS_CATEGORIES` можно пока оставить в UI локально, но лучше позже отдавать схему с сервера, чтобы разные версии UI/API не расходились.

---

## Что конкретно поменять в UI

### 1. `ui/api_client.py`

Убрать:

```python
from config.settings import load_config
from utils.logger import get_logger, log_json
```

Заменить на:

```python
from client.connection_store import ConnectionStore
from client.client_logger import get_logger, log_json
```

`ApiClient.__init__()` должен брать URL не из серверного `config.settings`, а из локального `connection.json`:

```python
class ApiClient:
    def __init__(self, base_url: str | None = None, timeout_sec: float = 2.5, stream_timeout_sec: float = 600.0):
        store = ConnectionStore()
        self.base_url = (base_url or store.get_base_url() or "http://127.0.0.1:8000").rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.stream_timeout_sec = float(stream_timeout_sec)
        self._runtime_model_cache = ""
```

Добавить методы:

```python
def get_config(self) -> dict:
    return self._request_json("GET", "/config")

def update_config(self, updates: dict) -> dict:
    return self._request_json("POST", "/config", {"updates": updates})

def get_config_schema(self) -> dict:
    return self._request_json("GET", "/config/schema")
```

---

### 2. Новый `client/connection_store.py`

Назначение: хранить только параметры подключения. Эти поля всегда доступны в UI, даже без API.

```python
class ConnectionStore:
    def __init__(self):
        self.path = client_data_dir() / "connection.json"

    def load(self) -> dict:
        ...

    def save(self, payload: dict) -> None:
        ...

    def get_base_url(self) -> str:
        data = self.load()
        return str(data.get("base_url") or "http://127.0.0.1:8000")
```

Формат:

```json
{
  "base_url": "http://127.0.0.1:8000",
  "last_connected_at": null,
  "profile": "default"
}
```

---

### 3. Новый `client/settings_cache.py`

Назначение: хранить последний полученный от API конфиг.

```python
class SettingsCache:
    def load(self) -> dict:
        ...

    def save(self, payload: dict) -> None:
        ...

    def has_cache(self) -> bool:
        return self.path.exists()
```

Формат:

```json
{
  "updated_at": "2026-04-26T14:00:00",
  "api_base_url": "http://127.0.0.1:8000",
  "payload": {
    "app": {},
    "llm": {},
    "memory": {},
    "voice": {}
  }
}
```

---

### 4. Новый `client/settings_sync.py`

Назначение: очередь изменений, сделанных без API.

```python
class SettingsSyncQueue:
    def load_pending(self) -> dict:
        ...

    def add_pending(self, updates: dict) -> None:
        ...

    def clear(self) -> None:
        ...

    def flush(self, api: ApiClient) -> dict:
        pending = self.load_pending()
        if not pending:
            return {}
        payload = api.update_config(pending)
        self.clear()
        return payload
```

Формат:

```json
{
  "updated_at": "2026-04-26T14:10:00",
  "updates": {
    "llm.temperature": 0.6,
    "memory.enabled": true
  }
}
```

Если один и тот же параметр менялся несколько раз offline — хранить только последнее значение.

---

### 5. `ui/settings_window.py`

Убрать прямую запись в серверный конфиг:

```python
from config.settings import get_config_payload, update_config_values
```

Заменить на фасад:

```python
from client.settings_facade import SettingsFacade
```

Внутри окна:

```python
self.settings = SettingsFacade(api=self.api)
```

`reload()`:

```python
def reload(self) -> None:
    result = self.settings.load()
    self._payload = result.payload
    self._api_online = result.api_online
    self._has_cache = result.has_cache
    self._pending = result.pending
    self._changed.clear()
    self._invalid.clear()
    self._render_category()
    self._refresh_preview()
```

`_save()`:

```python
result = self.settings.save(updates)
self.saved.emit(copy.deepcopy(updates))
self.reload()
```

Логика отображения:

- Вкладка `Подключение` всегда активна.
- Если API offline и кеша нет — все серверные категории скрыть или заменить заглушкой:

```text
Серверные настройки появятся после подключения к MMis API.
Пока можно настроить только адрес подключения.
```

- Если API offline, но кеш есть — показывать настройки из кеша с пометкой `offline cache`.
- Если offline и пользователь сохраняет изменения — писать в `pending_settings.json`, а не показывать ошибку.
- Вверху окна настроек добавить статус:

```text
API: online / offline
Cache: yes / no
Pending changes: N
```

---

### 6. `ui/chat_window.py`

Убрать прямые зависимости:

```python
from config.settings import DATA_DIR, load_config
from llm.tokenizer import estimate_tokens
from modules.voice.voice_manager import VoiceState
```

Заменить:

```python
from client.paths import client_data_dir
from client.tokenizer_stub import estimate_tokens
```

Голосовой режим сделать опциональным:

```python
try:
    from ui.voice_adapter import build_stt_config, build_stt_engine, build_tts_config, build_tts_engine, build_voice_manager
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False
```

Если `VOICE_AVAILABLE == False`, кнопка голоса остаётся видимой, но disabled или показывает:

```text
Голосовой модуль доступен только при подключенном сервере/полной сборке.
```

`DATA_DIR` заменить на локальный UI путь:

```python
UI_DATA_DIR = client_data_dir()
```

---

### 7. Синхронизация при восстановлении API

В `ChatWindow` или отдельном `ConnectionController` сделать периодический health-check:

```python
self._api_timer = QTimer(self)
self._api_timer.timeout.connect(self._check_api_status)
self._api_timer.start(5000)
```

Алгоритм:

```python
def _check_api_status(self):
    try:
        health = self.api.health(timeout=1.5)
        if not self._api_online:
            self._on_api_restored()
        self._api_online = True
    except ApiClientError:
        self._api_online = False
```

При восстановлении:

```python
def _on_api_restored(self):
    pending_result = self.settings_sync.flush(self.api)
    fresh_config = self.api.get_config()
    self.settings_cache.save(fresh_config)
    self._sync_runtime_controls()
    self._apply_context_chips()
```

---

## Как должны вести себя настройки

### Первый запуск, API нет

- UI запускается.
- Чат показывает статус `API offline`.
- В настройках доступна только категория `Подключение`.
- Серверные параметры не показываются или disabled.
- Отправка сообщения disabled/показывает понятную ошибку.

### API был раньше, сейчас недоступен

- UI запускается.
- Настройки показываются из `settings_cache.json`.
- Вверху пометка `offline cache`.
- Изменения сохраняются в `pending_settings.json`.
- После подключения изменения уходят на сервер.

### API доступен

- UI делает `GET /config`.
- Обновляет `settings_cache.json`.
- Если есть pending — сначала применяет `POST /config`, потом снова делает `GET /config`.
- Серверные настройки работают как обычно.

---

## Важное правило по конфликтам

Если параметр был изменён offline, а на сервере он тоже изменился до синхронизации:

MVP-вариант:

```text
pending побеждает сервер
```

То есть UI отправляет последнее локальное значение.

Позже можно добавить конфликтное окно:

```text
Параметр изменён и локально, и на сервере. Что оставить?
```

---

## Минимальный порядок внедрения

1. Добавить `client/paths.py`, `connection_store.py`, `settings_cache.py`, `settings_sync.py`, `client_logger.py`.
2. Переписать `ui/api_client.py`, чтобы он не импортировал `config.settings` и `utils.logger`.
3. Добавить `/config` и `/config/schema` endpoints в `api/app.py`.
4. Переписать `ui/settings_window.py` через `SettingsFacade`.
5. Добавить категорию `Подключение` в настройки UI.
6. Убрать из `chat_window.py` прямые импорты `config`, `llm`, `modules.voice`.
7. Добавить health-check и авто-flush pending изменений.
8. Проверить сценарии:
   - UI без API и без кеша.
   - UI без API, но с кешем.
   - UI с API.
   - UI изменил настройки offline → API появился → настройки применились.

---

## Что не надо делать

- Не тянуть весь `config/` в чистую папку UI.
- Не давать UI напрямую менять серверный `config/config.json`.
- Не делать настройки подключения зависимыми от API.
- Не падать при отсутствии `llm/`, `memory_core/`, `modules/`.
- Не хранить pending-изменения только в памяти процесса — их надо писать в файл.

---

## Итоговая логика

UI должен стать не частью сервера, а клиентом:

```text
UI local state
  ↓
ConnectionStore
  ↓
ApiClient
  ↓
SettingsCache ← GET /config
  ↓
SettingsSyncQueue → POST /config when online
```

Это даст нужное поведение: чистая папка UI, offline-режим, кеш настроек, активные параметры подключения и отложенная синхронизация изменений на сервер.
