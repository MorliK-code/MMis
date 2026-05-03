# MMis: глобальный замок API по длинному хешу

## Цель

Сделать так, чтобы при включённой защите API не работал вообще без правильного ключа/хеша.

То есть внешний пользователь может знать публичный IP и порт, но без ключа сервер должен отвечать отказом ещё до аккаунтов, чата, моделей, настроек и debug-эндпоинтов.

Целевая схема:

```text
HTTP request
→ X-MMis-Access-Key
→ глобальная проверка API-ключа
→ auth Bearer token
→ account scope
→ endpoint
```

## Важная логика

1. `Bearer token` отвечает за аккаунт.
2. `X-MMis-Access-Key` отвечает за право вообще достучаться до API.
3. Без `X-MMis-Access-Key` не должен работать даже `/auth/login`.
4. Для входа/регистрации клиент должен отправлять access key вместе с запросом.
5. В настройках обычных пользователей этот ключ не показывать.
6. Управлять серверным ключом может только `admin`.

---

# 1. Серверная настройка в `config/settings.py`

## Файл

```text
config/settings.py
```

## Что добавить в `AppSettings`

Найди dataclass `AppSettings` и добавь поля рядом с `host` / `port`:

```python
    api_access_lock_enabled: bool = False
    api_access_key_hash: str = ""
    api_access_deny_status: int = 404
    api_access_allow_ping_without_key: bool = False
    admin_login: str = "admin"
```

## Что добавить в `_default_config_tree()`

Найди блок:

```python
        "api": {
            "host": "127.0.0.1",
            "port": 8027,
        },
```

Замени на:

```python
        "api": {
            "host": "127.0.0.1",
            "port": 8027,
            "access_lock": {
                "enabled": False,
                "key_hash": "",
                "deny_status": 404,
                "allow_ping_without_key": False,
            },
        },
        "security": {
            "admin_login": "admin",
        },
```

## Что добавить в `_settings_from_payload()`

Найди место, где создаётся `AppSettings(...)` и рядом с `host=` / `port=` добавь:

```python
        api_access_lock_enabled=_to_bool(_get_dotted(row, "api.access_lock.enabled")),
        api_access_key_hash=_norm_str(_get_dotted(row, "api.access_lock.key_hash")),
        api_access_deny_status=max(401, min(404, _to_int(_get_dotted(row, "api.access_lock.deny_status"), default=404))),
        api_access_allow_ping_without_key=_to_bool(_get_dotted(row, "api.access_lock.allow_ping_without_key")),
        admin_login=_norm_str(_get_dotted(row, "security.admin_login") or "admin"),
```

---

# 2. Глобальная проверка ключа в `api/app.py`

## Файл

```text
api/app.py
```

## Добавить импорты

Вверху файла уже есть `hashlib`, `hmac` может быть не импортирован. Добавь:

```python
import hashlib
import hmac
```

Если `hashlib` уже есть, добавь только `hmac`.

## Добавить функции проверки

Перед `_bearer_token()` добавь:

```python
def _hash_api_access_key(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _api_access_key_from_request(request) -> str:
    # Основной вариант — заголовок. Query-параметр лучше не использовать,
    # потому что он светится в логах и истории URL.
    return str(request.headers.get("x-mmis-access-key") or "").strip()


def _api_access_allowed(request) -> bool:
    cfg = load_config(force_reload=True)
    if not bool(getattr(cfg, "api_access_lock_enabled", False)):
        return True

    path = str(getattr(getattr(request, "url", None), "path", "") or "")
    if path == "/ping" and bool(getattr(cfg, "api_access_allow_ping_without_key", False)):
        return True

    expected_hash = str(getattr(cfg, "api_access_key_hash", "") or "").strip().lower()
    if not expected_hash:
        # Если замок включён, но ключ не задан — лучше закрыть API полностью,
        # иначе можно случайно открыть сервер наружу.
        return False

    raw_key = _api_access_key_from_request(request)
    if not raw_key:
        return False

    actual_hash = _hash_api_access_key(raw_key).lower()
    return hmac.compare_digest(expected_hash, actual_hash)


def _api_access_denied_response():
    cfg = load_config(force_reload=True)
    status = int(getattr(cfg, "api_access_deny_status", 404) or 404)
    if status == 401:
        return JSONResponse(status_code=401, content={"detail": "api_access_required"})
    if status == 403:
        return JSONResponse(status_code=403, content={"detail": "api_access_denied"})
    return JSONResponse(status_code=404, content={"detail": "not_found"})
```

## Изменить middleware

Сейчас есть middleware:

```python
@app.middleware("http")
async def _account_scoped_runtime_middleware(request, call_next):
    route_path = str(getattr(getattr(request, "url", None), "path", "") or "")
    if route_path.startswith("/characters"):
        ...
    return await call_next(request)
```

Замени начало функции на:

```python
@app.middleware("http")
async def _account_scoped_runtime_middleware(request, call_next):
    if not _api_access_allowed(request):
        return _api_access_denied_response()

    route_path = str(getattr(getattr(request, "url", None), "path", "") or "")
    if route_path.startswith("/characters"):
        try:
            account = _require_account(request.headers.get("authorization"))
        except HTTPException as exc:
            return JSONResponse(status_code=int(exc.status_code), content={"detail": exc.detail})
        with _runtime.lock:
            _ensure_runtime_account(account)
    return await call_next(request)
```

Результат: без ключа будет закрыто всё, включая `/health`, `/models`, `/auth/login`, `/config`, `/metadata`, `/docs`.

---

# 3. Админ-роль в аккаунтах

Сейчас в `accounts` нет роли, поэтому нужно добавить минимальную роль.

## Файл

```text
api/auth_store.py
```

## В `_init_db()` добавить колонку role

После создания таблицы `accounts` добавь миграцию:

```python
        try:
            self._execute("ALTER TABLE accounts ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        except Exception:
            pass
```

## В `create_account()` сделать `admin` админом

Найди INSERT в `create_account()`:

```python
INSERT INTO accounts(account_id, login, display_name, password_salt, password_hash, created_at, updated_at)
VALUES(?, ?, ?, ?, ?, ?, ?)
```

Замени на:

```python
INSERT INTO accounts(account_id, login, display_name, password_salt, password_hash, created_at, updated_at, role)
VALUES(?, ?, ?, ?, ?, ?, ?, ?)
```

И в параметры добавь роль:

```python
role = "admin" if clean_login == "admin" else "user"
```

Полный кусок должен быть таким:

```python
        role = "admin" if clean_login == "admin" else "user"
        self._execute(
            """
            INSERT INTO accounts(account_id, login, display_name, password_salt, password_hash, created_at, updated_at, role)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (account_id, clean_login, clean_display, salt_hex, password_hash, ts, ts, role),
        )
```

## В `verify_login()` возвращать роль

Замени return на:

```python
        return {
            "account_id": str(row["account_id"]),
            "login": str(row["login"]),
            "display_name": str(row["display_name"]),
            "role": str(row["role"] or "user"),
        }
```

## В `get_account_by_token()` возвращать role

В SQL замени:

```sql
SELECT a.account_id, a.login, a.display_name
```

на:

```sql
SELECT a.account_id, a.login, a.display_name, a.role
```

И в return добавь:

```python
            "role": str(row["role"] or "user"),
```

---

# 4. Схемы авторизации

## Файл

```text
api/schemas.py
```

## Изменить `AuthResponse` и `AuthMeResponse`

Добавь поле:

```python
    role: str = "user"
```

Должно быть так:

```python
class AuthResponse(BaseModel):
    token: str
    account_id: str
    login: str
    display_name: str
    role: str = "user"


class AuthMeResponse(BaseModel):
    account_id: str
    login: str
    display_name: str
    role: str = "user"
```

---

# 5. Запрет обычным пользователям менять защищённые настройки

## Файл

```text
api/app.py
```

## Добавить helpers

Рядом с `_require_account()` добавь:

```python
def _is_admin_account(account: dict) -> bool:
    login = str((account or {}).get("login") or "").strip().lower()
    role = str((account or {}).get("role") or "").strip().lower()
    cfg = load_config(force_reload=True)
    admin_login = str(getattr(cfg, "admin_login", "admin") or "admin").strip().lower()
    return role == "admin" or bool(admin_login and login == admin_login)


def _assert_admin(account: dict) -> None:
    if not _is_admin_account(account):
        raise HTTPException(status_code=403, detail="admin_required")


def _strip_admin_only_config(payload: dict) -> dict:
    import copy
    out = copy.deepcopy(payload if isinstance(payload, dict) else {})
    try:
        out.get("api", {}).pop("access_lock", None)
    except Exception:
        pass
    try:
        out.pop("security", None)
    except Exception:
        pass
    return out


def _contains_admin_only_update(updates: dict[str, Any]) -> bool:
    for path in dict(updates or {}).keys():
        clean = str(path or "").strip().lower()
        if clean.startswith("api.access_lock") or clean.startswith("security."):
            return True
    return False
```

## Изменить `/config` GET

Сейчас:

```python
return get_config_payload(force_reload=True)
```

Замени на:

```python
payload = get_config_payload(force_reload=True)
if not _is_admin_account(account):
    payload = _strip_admin_only_config(payload)
return payload
```

## Изменить `/config` PATCH

После:

```python
account = _require_account(authorization)
```

добавь:

```python
    if _contains_admin_only_update(updates):
        _assert_admin(account)
```

Так обычный пользователь не сможет отправить PATCH руками.

---

# 6. Локальная настройка ключа в Desktop UI

Клиент должен отправлять ключ на каждый запрос, включая `/auth/login`.

## Файл

```text
ui/client_config_store.py
```

## Добавить поле в `DEFAULT_CONFIG`

Замени блок connection на:

```python
    "connection": {
        "active_endpoint": "local",
        "local_base_url": "http://127.0.0.1:8027",
        "public_base_url": "",
        "api_access_key": "",
    },
```

## Добавить функцию

```python
def get_api_access_key() -> str:
    conn = get_connection_config()
    return str(conn.get("api_access_key") or "").strip()
```

---

# 7. Отправлять ключ во всех API-запросах

## Файл

```text
ui/api_client.py
```

## Изменить импорт

Сейчас:

```python
from ui.client_config_store import get_selected_base_url
```

Замени на:

```python
from ui.client_config_store import get_selected_base_url, get_api_access_key
```

## В `_request_json()` добавить header

После:

```python
headers.update(self._account_headers())
```

добавь:

```python
        api_access_key = get_api_access_key()
        if api_access_key:
            headers["X-MMis-Access-Key"] = api_access_key
```

## В `stream_chat()` добавить header

После:

```python
headers.update(self._account_headers())
```

добавь:

```python
        api_access_key = get_api_access_key()
        if api_access_key:
            headers["X-MMis-Access-Key"] = api_access_key
```

---

# 8. Настройка видна только admin в UI

## Файл

```text
ui/settings_schema.py
```

## Расширить `SettingSpec`

Добавь поле:

```python
    admin_only: bool = False
```

Итог:

```python
@dataclass(frozen=True)
class SettingSpec:
    path: str
    title: str
    kind: SettingKind = "text"
    description: str = ""
    example: str = ""
    options: tuple[str, ...] = ()
    restart_required: bool = False
    dangerous: bool = False
    live: bool = False
    placeholder: str = ""
    admin_only: bool = False
```

## Добавить настройки в Main config → API

В карточку `API` добавь:

```python
                    SettingSpec(
                        "api.access_lock.enabled",
                        "API lock",
                        kind="bool",
                        description="Полностью закрывает API без правильного X-MMis-Access-Key.",
                        example="true",
                        restart_required=True,
                        dangerous=True,
                        admin_only=True,
                    ),
                    SettingSpec(
                        "api.access_lock.key_hash",
                        "API access key SHA256",
                        description="SHA256 от длинного ключа доступа. Сам ключ здесь не хранить.",
                        example="sha256...",
                        restart_required=True,
                        dangerous=True,
                        admin_only=True,
                    ),
```

## Добавить настройку в `Подключение к API`

В карточку `Подключение к API` добавь:

```python
                    SettingSpec(
                        "ui.api.api_access_key",
                        "API access key",
                        description="Длинный ключ, который Desktop UI отправляет в X-MMis-Access-Key.",
                        example="mmis_...",
                        dangerous=True,
                        live=True,
                        admin_only=True,
                    ),
```

Важно: это локальная клиентская настройка. Её лучше сохранять в `ui/client_config_store.py`, а не отправлять в серверный `/config`.

---

# 9. Фильтр admin-only настроек в SettingsWindow

## Файл

```text
ui/settings_window.py
```

## Добавить импорт

Вверху рядом с auth imports или внутри файла добавь:

```python
from ui.auth_client_store import load_auth_state
```

Если импорт конфликтует, используй локальный импорт внутри функции.

## Добавить функцию

В класс окна настроек добавь:

```python
    def _is_admin_user(self) -> bool:
        try:
            from ui.auth_client_store import load_auth_state
            state = load_auth_state()
            login = str(state.get("login") or "").strip().lower()
            role = str(state.get("role") or "").strip().lower()
            return role == "admin" or login == "admin"
        except Exception:
            return False
```

## Изменить `_filtered_cards()`

Сейчас логика берёт `category.cards`. Нужно отфильтровать `spec.admin_only`.

Замени функцию на:

```python
    def _filtered_cards(self, category: SettingCategory):
        is_admin = self._is_admin_user()
        cards = []
        for card in category.cards:
            visible_settings = tuple(
                spec for spec in card.settings
                if is_admin or not getattr(spec, "admin_only", False)
            )
            if not visible_settings:
                continue
            card = SettingCard(
                title=card.title,
                tag=card.tag,
                settings=visible_settings,
                dangerous=card.dangerous,
            )
            if not getattr(self, "_search_text", ""):
                cards.append(card)
            elif any(self._matches_search(spec, category.title, card.title) for spec in card.settings):
                cards.append(card)
        return tuple(cards)
```

Если `SettingCard` не импортирован в `settings_window.py`, добавь в импорт:

```python
from ui.settings_schema import SETTINGS_CATEGORIES, SettingCategory, SettingCard, SettingSpec, dotted_get, get_category
```

---

# 10. Сохранение role на клиенте

## Файл

```text
ui/auth_client_store.py
```

## Добавить role в состояние

В `DEFAULT_AUTH_STATE` добавь:

```python
    "role": "",
```

В `_session_payload()` добавь:

```python
        "role": str(data.get("role") or ""),
```

Теперь UI сможет понять, admin это или нет.

---

# 11. Как генерировать hash для настройки проекта

В PowerShell:

```powershell
python -c "import secrets,hashlib; k='mmis_'+secrets.token_urlsafe(48); print('KEY=',k); print('SHA256=',hashlib.sha256(k.encode()).hexdigest())"
```

В `config/config.json` / настройках проекта ставишь:

```json
"api": {
  "host": "0.0.0.0",
  "port": 8027,
  "access_lock": {
    "enabled": true,
    "key_hash": "сюда_SHA256",
    "deny_status": 404,
    "allow_ping_without_key": false
  }
}
```

В приложении ставишь именно `KEY`, а не `SHA256`:

```json
"connection": {
  "api_access_key": "mmis_..."
}
```

---

# 12. Проверка

## Без ключа

```powershell
curl http://127.0.0.1:8027/health
```

Должно вернуть:

```text
404
```

или выбранный тобой статус.

## С ключом

```powershell
curl -H "X-MMis-Access-Key: mmis_ТВОЙ_КЛЮЧ" http://127.0.0.1:8027/health
```

Должен быть нормальный ответ API.

## Login тоже должен требовать ключ

Без ключа:

```powershell
curl -X POST http://127.0.0.1:8027/auth/login -H "Content-Type: application/json" -d '{"login":"admin","password":"1234"}'
```

Должен быть отказ.

С ключом:

```powershell
curl -X POST http://127.0.0.1:8027/auth/login -H "Content-Type: application/json" -H "X-MMis-Access-Key: mmis_ТВОЙ_КЛЮЧ" -d '{"login":"admin","password":"1234"}'
```

Должен вернуться token.

---

# Итоговая логика

- `api.access_lock.key_hash` — серверный SHA256, видит и меняет только admin.
- `ui.api.api_access_key` — клиентский сырой ключ, отправляется в `X-MMis-Access-Key`, в настройках видит только admin.
- Остальные пользователи не видят эту настройку.
- Без ключа API не работает вообще.
- Даже если пользователь знает публичный IP, без ключа он не сможет войти, получить health, список моделей или открыть docs.
