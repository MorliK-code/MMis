# Исправление: имя персонажа и статистика использования

## 1. Сервер отдаёт не то имя персонажа

### Файл: `api/app.py`

Найти функцию:

```python
def _build_health_response() -> HealthResponse:
```

Внутри неё заменить этот блок:

```python
character_id = str(_runtime.brain.state_manager.get("active_character_id") or "default")
char_info = _runtime.brain.state_manager.storage.load_character(character_id) or {}
persona_name = str(char_info.get("name") or "Default").strip() or "Default"
```

на:

```python
state_mgr = _runtime.brain.state_manager

character_id = ""
try:
    if hasattr(state_mgr, "get_active_character_id"):
        character_id = str(state_mgr.get_active_character_id() or "").strip().lower()
except Exception:
    character_id = ""

if not character_id:
    try:
        character_id = str(
            state_mgr.get("active_character_id")
            or state_mgr.get("active_personality_id")
            or "default"
        ).strip().lower()
    except Exception:
        character_id = "default"

character_id = character_id or "default"

char_info = {}
try:
    if hasattr(state_mgr, "storage") and hasattr(state_mgr.storage, "load_character"):
        char_info = state_mgr.storage.load_character(character_id) or {}
except Exception:
    char_info = {}

persona_name = str(
    char_info.get("name")
    or char_info.get("display_name")
    or char_info.get("title")
    or "Default"
).strip() or "Default"
```

---

## 2. UI не должен игнорировать `Default`

### Файл: `ui/chat_window.py`

Найти в `_on_backend_status_result()`:

```python
persona_name = str(row.get("persona_name") or "").strip()
if persona_name:
    self._api_persona_name = persona_name
    self._last_persona_name = persona_name
    self._refresh_persona_label()
```

Заменить на:

```python
persona_name = str(row.get("persona_name") or "Default").strip() or "Default"
self._api_persona_name = persona_name
self._last_persona_name = persona_name
self._refresh_persona_label()
```

Найти такой же блок в `_sync_runtime_controls()` и заменить аналогично.

---

## 3. Не сохранять старую `Ася`, если сервер уже говорит `Default`

### Файл: `ui/chat_window.py`

В `_save_ui_state()` заменить:

```python
"last_persona_name": str(getattr(self, "_last_persona_name", "Ассистент")),
```

на:

```python
"last_persona_name": str(getattr(self, "_last_persona_name", "Default") or "Default"),
```

---

## 4. Статистика ресурсов сервера не отображается

### Файл: `ui/workers.py`

В `StatusPollWorker.run()` сейчас после `/ping` делается тяжёлый `/health`. Если `/health` не успевает, `api_ok=True`, но `server_resources={}`.

В блоке:

```python
if payload["api_ok"]:
    try:
        if health is None:
            health = self.api.health(timeout=4.0)
```

увеличить timeout:

```python
if payload["api_ok"]:
    try:
        if health is None:
            health = self.api.health(timeout=8.0)
```

---

## 5. Чтобы топбар не очищал CPU/RAM/GPU при временной ошибке `/health`

### Файл: `ui/chat_shell.py`

Найти в `_on_backend_status_result()`:

```python
self._apply_server_resources(dict(row.get("server_resources") or {}))
```

Заменить на:

```python
resources = dict(row.get("server_resources") or {})
if resources:
    self._apply_server_resources(resources)
```

И в ветке ошибки найти:

```python
self._apply_server_resources({})
```

Лучше временно убрать или закомментировать:

```python
# self._apply_server_resources({})
```

---

## 6. Если не отображается статистика под сообщением

### Файл: `ui/chat_window.py`

В `_perf_from_stats()` статистика `tok/s`, `prompt`, `gen` полностью показывается только если `verbose_enabled=True`.

Проверь, что при отправке сообщения verbose реально включён. В `_on_send()` / месте вызова `chat_stream()` должно быть:

```python
verbose=self._verbose_enabled,
```

Если хочешь показывать минимальную статистику всегда, в `_perf_from_stats()` можно оставить как есть. Если хочешь полную всегда — заменить:

```python
verbose_enabled = bool(stats.get("verbose_enabled"))
```

на:

```python
verbose_enabled = True
```

---

## Результат

После правок:

- имя персонажа будет приходить из активного character state;
- если данных нет, будет отображаться `Default`;
- UI не будет залипать на старой `Ася`;
- CPU/RAM/GPU/VRAM не будут пропадать из-за временного сбоя `/health`;
- статистика под сообщением будет зависеть от `verbose`, либо всегда полной, если включить последний фикс.
