# MMis: локальные аккаунты без API

## Цель
Сделать локальные аккаунты внутри desktop-приложения MMis, чтобы у каждого пользователя были свои:

- чаты;
- память;
- настройки;
- персонажи/выбранная персона;
- вложения;
- кэш;
- runtime-состояние.

Без сетевой авторизации, API-токенов и внешнего сервера.

---

## Главная идея

Не делать пока «настоящую серверную авторизацию».  
Сделать **локальный профильный вход**:

```text
Запуск MMis
  -> окно выбора аккаунта
  -> ввод PIN/пароля, если включён
  -> загрузка account_id
  -> все пути, память и чаты работают внутри data/accounts/<account_id>/
```

---

## Структура данных

Добавить глобальную базу аккаунтов:

```text
data/accounts/accounts.db
```

Пример таблицы `accounts`:

```sql
CREATE TABLE accounts (
    account_id TEXT PRIMARY KEY,
    login TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    password_hash TEXT,
    created_at REAL NOT NULL,
    last_login_at REAL,
    is_active INTEGER NOT NULL DEFAULT 1,
    data_path TEXT NOT NULL
);
```

Данные каждого аккаунта хранить отдельно:

```text
data/accounts/<account_id>/
  config/config.json
  memory_core/memory.db
  memory_core/vector/
  chats/chats.db
  attachments/
  cache/
  ui_state.json
```

Глобальными оставить:

```text
models/
data/specs/
config/base/defaults
```

---

## Что важно в текущем коде MMis

Сейчас в `core/brain.py` память пишется с:

```python
workspace_id=str(self.state_manager.get("active_character_id") or "global")
```

Это нельзя оставить для аккаунтов.

Иначе получится ошибка архитектуры:

```text
workspace_id = персонаж
```

А должно быть:

```text
workspace_id = аккаунт пользователя
session_id = id конкретного чата
character_id/persona_id = метаданные
```

---

## Правильная схема идентификаторов

Использовать так:

```text
account_id      = владелец данных
chat_id         = конкретный чат
persona_id      = активная персона, например default/asya
workspace_id    = account_id
session_id      = chat_id
conversation_id = chat_id
```

Для `MemoryEnvelope`:

```python
MemoryEnvelope(
    workspace_id=account_id,
    session_id=chat_id,
    metadata={
        "account_id": account_id,
        "chat_id": chat_id,
        "persona_id": persona_id,
    }
)
```

---

## Новые файлы

### 1. `core/account_manager.py`

Создать менеджер аккаунтов:

```python
class AccountManager:
    def create_account(self, login: str, display_name: str, password: str | None = None) -> Account: ...
    def login(self, login: str, password: str | None = None) -> Account: ...
    def list_accounts(self) -> list[Account]: ...
    def get_current_account(self) -> Account | None: ...
    def set_current_account(self, account_id: str) -> None: ...
    def resolve_account_path(self, *parts: str) -> Path: ...
```

---

### 2. `core/account_context.py`

Создать runtime-контекст текущего аккаунта:

```python
@dataclass
class AccountContext:
    account_id: str
    login: str
    display_name: str
    data_dir: Path
    config_path: Path
    memory_db_path: Path
    memory_vector_path: Path
    chats_db_path: Path
```

---

### 3. `memory_core/account_paths.py`

Добавить функцию резолва путей памяти:

```python
def memory_paths_for_account(account: AccountContext) -> dict[str, str]:
    return {
        "db_path": str(account.memory_db_path),
        "vector_path": str(account.memory_vector_path),
    }
```

---

### 4. `chat/chat_store.py` или `core/chat_store.py`

Сделать локальное хранилище чатов:

```sql
CREATE TABLE chats (
    chat_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    persona_id TEXT NOT NULL DEFAULT 'default',
    archived INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE messages (
    message_id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
```

---

## Что изменить в существующих файлах

### `config/settings.py`

Добавить поля:

```python
account_id: str = ""
accounts_dir: Path = field(default_factory=lambda: DATA_DIR / "accounts")
```

Но настройки конкретного пользователя лучше хранить не в общем `config/config.json`, а в:

```text
data/accounts/<account_id>/config/config.json
```

---

### `memory_core/adapter.py`

Добавить возможность пересоздавать адаптер под аккаунт:

```python
def switch_account(self, account: AccountContext) -> None:
    self.config.db_path = str(account.memory_db_path)
    self.config.vector_path = str(account.memory_vector_path)
    self.service = build_memory_service(self.config)
    self._current_workspace = account.account_id
    self._current_session = "default"
```

---

### `core/brain.py`

Заменить запись в память.

Было:

```python
workspace_id=str(self.state_manager.get("active_character_id") or "global")
```

Должно быть:

```python
workspace_id=str(self.state_manager.get("account_id") or "global")
```

И в `metadata` добавить:

```python
"account_id": account_id,
"persona_id": personality_id,
```

---

### `core/response_pipeline.py`

Заменить вычисление `workspace_id`.

Было логически:

```python
workspace_id = active_character_id or global
```

Должно быть:

```python
workspace_id = account_id or global
```

А `active_character_id` оставить только как метаданные/персонаж.

---

### `core/character_runtime.py`

Состояние `brain_state.json` сейчас завязано на `cfg.memory_dir`.  
Надо перенести его в аккаунт:

```text
data/accounts/<account_id>/memory_core/brain_state.json
```

Или лучше:

```text
data/accounts/<account_id>/state/brain_state.json
```

---

## Логика запуска приложения

```text
main.py
  -> AccountManager.load()
  -> если аккаунтов нет: создать первый аккаунт
  -> показать окно выбора аккаунта
  -> после логина собрать AccountContext
  -> загрузить пользовательский config
  -> пересоздать MemoryCoreAdapter под аккаунт
  -> загрузить список чатов аккаунта
  -> открыть последний чат
```

---

## Минимальный MVP

1. Создать аккаунт при первом запуске.
2. Сделать выбор аккаунта при запуске.
3. Хранить память отдельно по папкам аккаунтов.
4. Хранить чаты отдельно по аккаунтам.
5. Заменить `workspace_id=active_character_id` на `workspace_id=account_id`.
6. `conversation_id` использовать как `chat_id`.
7. Сохранять выбранную персону в метаданных, а не как корневой workspace.

---

## Пароль и безопасность

Для MVP можно сделать вход без пароля или с PIN.

Для нормального варианта:

- хранить не пароль, а `password_hash`;
- использовать `argon2id` или `bcrypt`;
- позже добавить шифрование данных аккаунта.

Важно: локальный пароль без шифрования защищает только интерфейс, но не файлы на диске.

---

## Что не делать сейчас

Пока не надо:

- API-токены;
- JWT;
- refresh-token;
- OAuth;
- серверные сессии;
- публичную регистрацию;
- синхронизацию между устройствами.

Это понадобится только когда появится сетевой режим.
