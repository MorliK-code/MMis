# MMis: фикс пропадающего API-статуса и Default persona

## Проблема 1: API подключается и сразу пропадает

### Причина
Сейчас UI считает API offline, если тяжёлый `/health` не успел ответить или вернул ошибку во время проверки статуса. `/health` у сервера собирает не только факт доступности API, но ещё модель, memory status и ресурсы. Из-за этого статус может мигать: API живой, но health-проверка тормозит.

Нужно разделить:

- `ping` — сервер доступен;
- `health` — подробное состояние модели/памяти/ресурсов.

---

## 1. Добавить лёгкий endpoint `/ping`

### Файл: `api/app.py`

Рядом с `/health` добавить:

```python
@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}
```

---

## 2. Добавить метод `ping()` в API client

### Файл: `ui/api_client.py`

После метода `health()` добавить:

```python
def ping(self, timeout: float | None = None) -> dict:
    return self._request_json("GET", "/ping", timeout=timeout or 1.2)
```

---

## 3. Переделать `StatusPollWorker.run()`

### Файл: `ui/workers.py`

Найти:

```python
health = self.api.health(timeout=5.0)
payload["api_ok"] = str(health.get("status") or "").strip().lower() == "ok"
```

Заменить логику на такую:

```python
try:
    ping = self.api.ping(timeout=1.2)
    payload["api_ok"] = str(ping.get("status") or "").strip().lower() == "ok"
except Exception:
    # fallback для старого сервера без /ping
    health = self.api.health(timeout=5.0)
    payload["api_ok"] = str(health.get("status") or "").strip().lower() == "ok"
    ping = None

if payload["api_ok"]:
    try:
        health = self.api.health(timeout=4.0)
        payload["model"] = str(health.get("model") or self.api.get_runtime_model() or "")
        payload["thinking_enabled"] = bool(health.get("thinking_enabled", False))
        payload["verbose_enabled"] = bool(health.get("verbose_enabled", False))
        payload["json_mode_enabled"] = bool(health.get("json_mode_enabled", False))
        payload["web_mode"] = str(health.get("web_mode") or "")
        payload["model_status"] = dict(health.get("model_status") or {})
        payload["memory_status"] = dict(health.get("memory_status") or {})
        payload["server_resources"] = dict(health.get("server_resources") or {})
    except Exception as exc:
        payload["error"] = f"health details unavailable: {exc}"
```

Идея: если `/ping` успешен, API не должен становиться offline только из-за долгой подробной проверки.

---

## 4. Сделать offline-статус менее резким

### Файл: `ui/chat_shell.py`

В `_on_backend_status_result()` найти блок:

```python
if not api_ok:
    self._backend_status_failures = int(getattr(self, "_backend_status_failures", 0)) + 1
    if bool(getattr(self, "_backend_status_seen_ok", False)) and self._backend_status_failures < 3:
        return
    self._apply_backend_status(api_ok=False, error=str(row.get("error") or ""))
    self._apply_server_resources({})
    return
```

Заменить на:

```python
if not api_ok:
    self._backend_status_failures = int(getattr(self, "_backend_status_failures", 0)) + 1
    if self._backend_status_failures < 3:
        return
    self._apply_backend_status(api_ok=False, error=str(row.get("error") or ""))
    self._apply_server_resources({})
    return
```

Так UI не будет мигать offline от одного случайного timeout.

---

# Проблема 2: имя персонажа должно быть Default, когда данных нет

## 5. Исправить fallback в API

### Файл: `api/app.py`

Найти:

```python
character_id = str(_runtime.brain.state_manager.get("active_character_id") or "asya")
char_info = _runtime.brain.state_manager.storage.load_character(character_id)
persona_name = str(char_info.get("name") or "Assistant")
```

Заменить на:

```python
character_id = str(_runtime.brain.state_manager.get("active_character_id") or "default")
char_info = _runtime.brain.state_manager.storage.load_character(character_id) or {}
persona_name = str(char_info.get("name") or "Default").strip() or "Default"
```

---

## 6. Исправить fallback в UI

### Файл: `ui/chat_window.py`

### 6.1. В `_fallback_character_name()`

Сделать так:

```python
@staticmethod
def _fallback_character_name(character_id: str) -> str:
    cleaned = str(character_id or "").strip().lower()
    if cleaned in {"", "default", "assistant", "none", "null"}:
        return "Default"
    if cleaned in {"asya", "асья", "ася"}:
        return "Ася"
    return str(character_id or "Default").strip() or "Default"
```

### 6.2. В `_resolve_active_character_id()`

В конце заменить:

```python
return "asya"
```

на:

```python
return "default"
```

### 6.3. В `_resolve_persona_display_name()`

В самом конце добавить безопасный fallback:

```python
return self._last_persona_name or "Default"
```

---

## 7. Не сохранять старую Асю как default из ui_state

### Файл: `ui/chat_window.py`

В `_load_ui_state()` заменить:

```python
self._last_persona_name = str(payload.get("last_persona_name") or self._last_persona_name)
```

на:

```python
saved_persona = str(payload.get("last_persona_name") or "").strip()
if saved_persona:
    self._last_persona_name = saved_persona
else:
    self._last_persona_name = "Default"
```

Если хочешь прямо сбросить старый кеш, вручную поправь:

```text
ui/.mmis_client/ui_state.json
```

и замени:

```json
"last_persona_name": "Ася"
```

на:

```json
"last_persona_name": "Default"
```

---

# Итог

После правок:

- API-статус не должен мигать из-за долгого `/health`;
- UI будет считать API online по лёгкому `/ping`;
- подробные данные модели/памяти будут подтягиваться отдельно;
- если персонажа нет, сверху будет отображаться `Default`, а не `Ася` или `Assistant`.
