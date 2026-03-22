# Memory Core - Документация

## Обзор

`memory_core` — новый внутренний слой памяти MMis, реализующий единую систему хранения и обработки памяти.

## Архитектура

```
memory_core/
├── schemas.py           # Модели данных
├── facade.py            # MemoryService (единый фасад)
├── storage/             # Хранилища
│   ├── sqlite_db.py     # SQLite база
│   ├── event_store.py   # Raw events (канон)
│   ├── artifact_store.py# Артефакты
│   ├── workspace_store.py # Workspace
│   └── state_store.py   # Состояния
├── processors/          # Процессоры
│   ├── ingest_analyzer.py   # Анализ событий
│   ├── fact_processor.py    # Факты
│   ├── profile_processor.py # Профиль
│   ├── episode_processor.py # Эпизоды
│   ├── task_processor.py    # Задачи
│   └── document_processor.py# Документы
├── retrieval/           # Поиск
│   ├── retrieval_service.py
│   ├── context_builder.py
│   └── reranker.py
├── indexing/            # Индексация
│   ├── embeddings.py
│   └── vector_index.py
└── inspect/             # Диагностика
    └── inspector.py
```

## Быстрый старт

### Базовое использование

```python
from memory_core_adapter import MemoryCoreAdapter

# Инициализация
memory = MemoryCoreAdapter(
    db_path="data/memory_core/memory.db",
    default_workspace="global",
)

# Ingest события
result = memory.ingest_event(
    text="Меня зовут Паша. Я работаю в VS Code на Windows.",
    source_kind="user",
)
print(f"Создано артефактов: {result['artifacts_created']}")

# Query к памяти
result = memory.query("Кто работает в VS Code?")
print(f"Контекст: {result['context_blocks']}")

# Статистика
stats = memory.get_stats()
print(f"Событий: {stats['events_count']}, Артефактов: {stats['artifacts_count']}")
```

### Прямое использование MemoryService

```python
from memory_core.bootstrap.service_factory import build_memory_service, MemoryServiceConfig
from memory_core.schemas import MemoryEnvelope, MemoryQuery

# Создание сервиса
config = MemoryServiceConfig(
    db_path="data/memory_core/memory.db",
    vector_path="data/memory_core/vector",
    top_k=8,
)
memory_service = build_memory_service(config)

# Ingest
envelope = MemoryEnvelope(
    source_kind="user",
    payload_type="message",
    text="Я использую Python для разработки",
    workspace_id="my_project",
)
result = memory_service.ingest_event(envelope)

# Query
query = MemoryQuery(
    text="Какой язык программирования используется?",
    workspace_id="my_project",
    top_k=5,
)
query_result = memory_service.query(query)

# Доступ к контексту
for block in query_result.context_blocks:
    print(block)
```

## API

### MemoryEnvelope

Конверт для входящего события:

```python
@dataclass
class MemoryEnvelope:
    event_id: str              # Уникальный ID
    source_kind: str           # user|assistant|tool|system|document
    payload_type: str          # message|tool_result|doc_text
    text: str                  # Текст события
    metadata: dict             # Метаданные
    namespace: str             # Пространство имён
    workspace_id: str          # ID workspace
    session_id: str            # ID сессии
    ts: float                  # Timestamp
```

### MemoryArtifact

Артефакт памяти (результат обработки):

```python
@dataclass
class MemoryArtifact:
    artifact_id: str           # Уникальный ID
    artifact_type: str         # fact|profile_fact|episode|task|document_chunk
    source_event_id: str       # ID исходного события
    text: str                  # Текст
    summary: str               # Краткое содержание
    metadata: dict             # Метаданные
    status: str                # active|archived|superseded
    created_at: float          # Время создания
```

### MemoryQuery

Запрос к памяти:

```python
@dataclass
class MemoryQuery:
    text: str                  # Текст запроса
    workspace_id: str          # ID workspace
    artifact_types: list[str]  # Фильтр по типам
    top_k: int                 # Количество результатов
    include_citations: bool    # Включать цитаты
```

## Процессоры

### Fact Processor
Извлекает проверяемые факты:
- "Паша использует VS Code"
- "Проект называется MMis"

### Profile Processor
Извлекает профильные данные:
- Имя пользователя
- Предпочтения (editor, OS, language)
- Железо (GPU, RAM)

### Episode Processor
Собирает смысловые эпизоды диалога.

### Task Processor
Выделяет задачи и открытые петли.

### Document Processor
Обрабатывает документы (chunking, summary).

## Retrieval

### Контекст для LLM

```python
result = memory.query("Что мы обсуждали про память?")

# context_blocks содержит готовые блоки:
# - ## Profile Facts
# - ## Active Tasks
# - ## Recent Episodes
# - ## Relevant Facts
# - ## Document Context
```

## Inspector API

### Debug endpoints

```python
# Список событий
events = memory.inspect(kind="events", limit=50)

# Список артефактов
artifacts = memory.inspect(kind="artifacts", limit=50)

# Трассировка события
trace = memory.inspect(kind="trace", event_id="...")

# Профильные факты
profile = memory.inspect(kind="profile", limit=20)

# Статистика
stats = memory.inspect(kind="stats")
```

## REST API

### Регистрация в FastAPI

```python
from fastapi import FastAPI
from api.memory_core_api import register_memory_core_api

app = FastAPI()
register_memory_core_api(app)
```

### Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| POST | `/memory/ingest` | Добавить событие |
| POST | `/memory/query` | Запрос к памяти |
| GET | `/memory/context` | Получить контекст |
| POST | `/memory/inspect` | Инспекция |
| GET | `/memory/stats` | Статистика |
| GET | `/memory/workspaces` | Список workspace |
| POST | `/memory/workspace` | Установить workspace |
| GET | `/memory/workspace/current` | Текущий workspace |

## Интеграция с response_pipeline

Используйте адаптеры из `memory_core_integration.py`:

```python
from memory_core_integration import create_memory_core_adapters

adapters = create_memory_core_adapters(memory_service)

# Retrieval
context = adapters["retrieve"].build_context_pack(
    query_text="Что мы обсуждали?",
    workspace_id="global",
)

# Profile
profile = adapters["profile"].flatten_profile_snapshot()

# Hints
hints = adapters["hints"].build_hints(query_text="...")
```

## Тесты

Запуск тестов:

```bash
python -m pytest tests/test_memory_core.py -v
```

## Миграция со старой памяти

1. Удалить старые данные:
```powershell
Remove-Item -Recurse -Force .\data\memory_storage\memory_v2
Remove-Item -Force .\data\exports\memory_export.json
```

2. Использовать `memory_core_adapter` вместо `MemoryManager`

3. Постепенно заменять импорты:
```python
# Было
from memory.memory_manager import MemoryManager

# Стало
from memory_core_adapter import MemoryCoreAdapter
```

## Принципы работы

### Raw Events — канон
Сырые события — единственный источник истины. Всё остальное (артефакты, индекс, summary) — производные.

### Vector Index — только индекс
Векторный индекс можно удалить и пересобрать. Истина хранится в SQLite.

### Единый фасад
Весь проект работает только через `MemoryService.ingest_event()` и `MemoryService.query()`.

### Процессоры
События проходят через процессоры, которые создают артефакты.

## Конфигурация

```python
@dataclass
class MemoryServiceConfig:
    db_path: str = "data/memory_core/memory.db"
    vector_path: str = "data/memory_core/vector"
    vector_backend: str = "memory"  # memory | chroma
    default_namespace: str = "default"
    default_workspace: str = "global"
    top_k: int = 8
    enable_profile_memory: bool = True
    enable_task_memory: bool = True
    enable_document_memory: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
```

## Структура БД

### Таблицы
- `events` — сырые события
- `artifacts` — артефакты
- `artifact_links` — связи между артефактами
- `workspaces` — workspace
- `workspace_sources` — источники документов
- `runtime_state` — краткосрочное состояние

## Лицензия

Часть проекта MMis.
