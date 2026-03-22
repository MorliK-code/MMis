# MMis — подробный план избавления от старого `memory/` и полного перехода на `memory_core`

## Что это за документ

Это **не новый архитектурный brainstorming** и не ещё один общий план про память.

Это документ под текущую задачу:

> **добить старый `memory/`, перестать жить на `MemoryManager`, полностью переключить runtime на `memory_core`, а legacy-память оставить только как временную прокладку до полного удаления.**

Документ написан под текущий курс:
- **отдельный memory-сервис сейчас не делаем**;
- **новую память держим внутри MMis**;
- **основная цель — избавиться от старой memory-архитектуры и завершить переход**.

---

# 1. Короткий диагноз

Сейчас в проекте одновременно живут **две памяти**:

## 1.1. Старая память
Старый пакет `memory/`, завязанный на:
- `MemoryManager`
- `MemoryEvent`
- `MemoryScope`
- `MemoryType`
- `build_context`
- `build_memory_debug_snapshot`
- `build_memory_native_state`
- `tool_bridge`
- `recall_policy`
- `project_terms`
- `debug_snapshot`
- `event_store`
- `auto_migration`

## 1.2. Новая память
Новый пакет `memory_core/`, плюс:
- `memory_core_adapter.py`
- `memory_core_integration.py`
- `api/memory_core_api.py`
- `MEMORY_CORE_README.md`
- `MMis_memory_core_blueprint.md`
- `IMPLEMENTATION_REPORT.md`

## 1.3. Главная проблема
Проблема не в том, что `memory_core` не сделан.

Проблема в том, что:
- **runtime всё ещё сидит на старом `memory/`;**
- новая память уже лежит рядом;
- старая и новая системы дублируют ответственность;
- это мешает и запуску, и отладке, и развитию характера/когнитивного слоя.

---

# 2. Главная цель

## Цель в одной фразе

**Полностью снять основной runtime MMis со старого `memory/` и перевести его на `memory_core` через один адаптер/фасад, после чего старый `memory/` оставить только как временный compatibility layer и затем удалить.**

---

# 3. Что НЕ надо делать сейчас

Чтобы не утонуть в распиле, вот чего **не надо** делать сейчас:

- не переписывать заново `memory_core`;
- не проектировать третью систему памяти;
- не делать отдельный сервер;
- не делать красивую веб-админку;
- не пытаться чинить `MemoryManager`;
- не смешивать “удаление старого” с новой когнитивной архитектурой;
- не тянуть новые фичи прямо в legacy memory.

## Что надо делать сейчас

- переключить bootstrap на `memory_core`;
- переключить `Brain` на новый ingest flow;
- переключить `ResponsePipeline` на новый query flow;
- отвязать debug/API от старого `memory_manager`;
- заморозить старый `memory/`;
- поэтапно вырезать импорты legacy memory;
- удалить старую базу и старые скрипты после переключения runtime.

---

# 4. Базовый принцип перехода

## Было

```python
main.py -> MemoryManager
          -> old memory event store
          -> old build_context
          -> old retrieval flow
          -> old debug snapshot
```

## Должно стать

```python
main.py -> memory_core adapter / facade
          -> MemoryService
          -> new ingest/query API
          -> memory_core inspect/debug
```

## Очень важный принцип

Переход делаем **не через “переписать всё сразу”**, а через:

1. новый слой уже есть;
2. runtime переключаем на него;
3. старый слой временно остаётся только как legacy;
4. после стабилизации удаляем старое.

---

# 5. Что именно считать legacy memory

Ниже то, что надо считать **устаревшим слоем**, который больше не является точкой развития.

## Legacy runtime pieces

- `memory/memory_manager.py`
- `memory/memory_models.py`
- `memory/auto_migration.py`
- `memory/event_store.py`
- `memory/debug_snapshot.py`
- `memory/native_state.py`
- `memory/tool_bridge.py`
- `memory/recall_policy.py`
- `memory/project_terms.py`
- `memory/retrieval.py`
- `memory/governor.py`
- `memory/profile_evolution.py`
- `memory/claim_*`
- `memory/dialog_episode_*`
- `memory/memory_lifecycle.py`
- `memory/memory_scoring.py`
- `memory/memory_policy.py`
- `memory/storage_profile.py`
- `memory/retrieval_projection.py`

## Что важно

Это **не значит**, что все алгоритмы надо потерять.
Это значит:

- старые **публичные интерфейсы** больше не трогаем;
- полезные алгоритмы можно переносить в `memory_core`;
- но legacy-память больше не участвует в главном runtime.

---

# 6. Что уже можно использовать как доноров

Из старого слоя можно брать не интерфейсы, а только **внутренние полезные куски**.

## Доноры идей/алгоритмов

- `document_chunker.py`
- `document_ingest.py`
- `document_memory.py`
- `document_models.py`
- `document_retrieval.py`
- `embedding_provider.py`
- `entity_resolver.py`
- `event_store.py`
- `fact_extractor.py`
- `identity_core.py`
- `ingest_analyzer.py`
- `memory_inspector.py`
- `memory_models.py`
- `reranker.py`
- `summary_quality.py`
- `text_sanitizer.py`
- `tool_bridge.py`
- `vector_store.py`

## Но правило простое

Не переносить старый API.
Переносить только:
- алгоритм;
- эвристику;
- полезную обработку текста;
- формулу скоринга;
- chunking;
- rerank-логику.

---

# 7. Правильный порядок избавления от старой memory

## Этап 1. Переключить bootstrap проекта
Сначала runtime должен **создавать только новый memory_core**.

## Этап 2. Переключить запись событий
`Brain` и всё, что пишет в память, должно писать только в `memory_core`.

## Этап 3. Переключить retrieval
`ResponsePipeline` должен читать только из нового query/retrieval flow.

## Этап 4. Переключить debug/API
Все debug endpoint и inspection flow должны смотреть только в `memory_core`.

## Этап 5. Очистить импорты и конфиг
После этого legacy memory остаётся только как висячий код, который уже не нужен runtime.

## Этап 6. Удалить старую БД и старые скрипты
Когда проект реально работает на `memory_core`, можно снести старую базу и всё сопровождение вокруг неё.

## Этап 7. Удалить старый пакет `memory/`
Только после стабилизации и зелёных тестов.

---

# 8. Что менять в `main.py`

Это **самая первая точка**.

## Что убрать

Убрать всё, что связано со старой памятью:

```python
from memory.auto_migration import run_auto_migration
from memory.event_store import EventStore
from memory.memory_manager import MemoryManager
```

Инициализацию тоже убрать:

```python
migration_result = run_auto_migration(...)
event_store = EventStore()
memory_manager = MemoryManager(root_dir=settings.memory_dir)
```

## Что должно появиться

Вместо этого `main.py` должен использовать **новую точку инициализации**:

```python
from memory_core_adapter import MemoryCoreAdapter, init_memory_core
```

или, если уже есть фабрика в новом слое:

```python
from memory_core.bootstrap.service_factory import build_memory_service
```

## Как должен выглядеть новый bootstrap

Пример под твой проект:

```python
memory_core = init_memory_core(
    db_path="data/memory_core/memory.db",
    vector_path="data/memory_core/vector",
    default_workspace="global",
    default_namespace="default",
    top_k=8,
)
```

## Что поменять в `AppContainer`

### Было

```python
memory_manager: MemoryManager
```

### Должно стать

```python
memory_core: MemoryCoreAdapter
```

или:

```python
memory_service: MemoryService
```

Выбери **одно имя** и держи его везде одинаковым.
Я бы советовал `memory_core`, если у тебя реально есть адаптер.
Если хочешь более чистое имя — `memory_service`.

## Что поменять в `shutdown()`

### Было

```python
_safe_call(self.memory_manager, "close")
```

### Должно стать

```python
_safe_call(self.memory_core, "close")
```

---

# 9. Что менять в `core/brain.py`

`Brain` — вторая критичная точка.

Сейчас он знает слишком много про legacy memory.

## Что убрать из импортов

Удалить всё подобное:

```python
from memory.debug_snapshot import build_memory_debug_snapshot
from memory.identity_core import PROTECTED_IDENTITY_CORE_KEYS
from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryEvent, MemoryScope, MemoryType
```

## Что должно остаться

`Brain` должен знать только:
- `memory_core adapter` или `MemoryService`;
- текст события;
- source kind;
- payload type;
- namespace/session/workspace;
- metadata.

Пример нового импорта:

```python
from memory_core_adapter import MemoryCoreAdapter
```

или:

```python
from memory_core.schemas import MemoryEnvelope
from memory_core.facade import MemoryService
```

---

## Что менять в `__init__`

### Было

```python
self.memory_manager = memory_manager or MemoryManager()
```

### Должно стать

```python
self.memory_core = memory_core
```

`Brain` не должен сам создавать память.
Память должна приходить снаружи из `main.py`.

---

## Как должен выглядеть новый ingest user turn

### Было

```python
self.memory_manager.ingest_event(
    MemoryEvent(
        role="user",
        text=user_payload,
        namespace=(conversation_id or "default"),
        scope=MemoryScope.CONVERSATION,
        memory_type=MemoryType.MESSAGE,
        metadata={...},
    )
)
```

### Должно стать

```python
self.memory_core.ingest_event(
    source_kind="user",
    payload_type="message",
    text=user_payload,
    namespace=(conversation_id or "default"),
    workspace_id=str(meta.get("workspace_id") or "global"),
    session_id=(conversation_id or "default"),
    metadata={
        "source": source,
        "trace_id": trace_id,
        "request_id": request_id,
        "turn_id": turn_id,
        "quality_profile": quality_profile,
        "character_id": str(meta.get("active_character_id") or "default"),
    },
)
```

## Аналогично assistant turn

```python
self.memory_core.ingest_event(
    source_kind="assistant",
    payload_type="message",
    text=assistant_payload,
    namespace=(conversation_id or "default"),
    workspace_id=str(meta.get("workspace_id") or "global"),
    session_id=(conversation_id or "default"),
    metadata={
        "source": source,
        "trace_id": trace_id,
        "request_id": request_id,
        "turn_id": turn_id,
        "quality_profile": quality_profile,
        "model": model_name,
        "character_id": str(meta.get("active_character_id") or "default"),
    },
)
```

## Tool result тоже через новый ingest

```python
self.memory_core.ingest_event(
    source_kind="tool",
    payload_type="tool_result",
    text=tool_summary_text,
    namespace=(conversation_id or "default"),
    workspace_id=str(meta.get("workspace_id") or "global"),
    session_id=(conversation_id or "default"),
    metadata={
        "tool_name": tool_name,
        "tool_payload": tool_payload,
        "trace_id": trace_id,
        "request_id": request_id,
        "turn_id": turn_id,
    },
)
```

---

# 10. Что убрать из `Brain` логически

Из `Brain` надо убрать:
- знание про `MemoryScope`;
- знание про `MemoryType`;
- знание про старые debug snapshot;
- знание про identity protected keys из legacy memory;
- знание про старую event model.

## Правильное правило

`Brain` не должен знать, как память устроена внутри.

Он знает только:
- **какое событие пришло**;
- **откуда оно**;
- **в каком namespace/workspace/session это произошло**;
- **какой текст и meta надо передать**.

Всё остальное решает память.

---

# 11. Что менять в `core/response_pipeline.py`

Это **самый грязный слой**, потому что именно там legacy memory сильно протекла в orchestration.

## Что убрать

Надо убрать/вырезать из pipeline любые прямые зависимости на старый memory-flow:

- `ContextBuildRequest`
- `build_context`
- `build_memory_tool_context_pack`
- `normalize_memory_retrieval_plan`
- `build_memory_native_state`
- `classify_query_recall_profile`
- `build_retrieval_hints`
- `flatten_governor_profile_snapshot`
- `format_memory_result_for_llm`
- `build_memory_debug_snapshot`
- `EpisodePlanner` из старого memory-flow

## Почему это критично

Сейчас `response_pipeline` не просто использует память — он частично содержит её логику.

Это надо ломать.

---

## Новый подход для pipeline

`ResponsePipeline` должен делать **один нормальный запрос** к новой памяти.

Пример:

```python
query_result = self.memory_core.query(
    query_text=base_query,
    namespace=conversation_id,
    workspace_id=workspace_id,
    session_id=conversation_id,
    include_profile=True,
    include_tasks=True,
    include_documents=True,
    include_messages=True,
    top_k=8,
)
```

И получать уже готовую структуру:
- `profile_items`
- `task_items`
- `document_items`
- `message_items`
- `citations`
- `debug_info`

---

## Что делать вместо старого `build_context`

В новом слое должен быть **один builder prompt-блоков**.

Например:

```python
memory_blocks = build_prompt_memory_blocks(query_result)
```

И дальше pipeline использует только готовые блоки:

```python
ctx.state["memory_state"] = {
    "profile_items": query_result.profile_items,
    "task_items": query_result.task_items,
    "document_items": query_result.document_items,
}
```

## Очень важное ограничение

`ResponsePipeline` не должен сам:
- ранжировать память;
- строить retrieval-plan старого формата;
- знать про governor snapshot;
- знать про native_state legacy memory.

Это всё должно уже приходить из `memory_core`.

---

# 12. Что делать с retrieval hints и recall policy

Старые:
- `project_terms`
- `recall_policy`
- `classify_query_recall_profile`
- `build_retrieval_hints`

надо либо:
1. перенести как **внутренние utilities** в `memory_core`,
2. либо временно завернуть в адаптер,
3. но **не держать их как внешние зависимости pipeline**.

## Жёсткое правило
Если retrieval/recall эвристика относится к памяти, она должна жить:
- либо в `memory_core/query/*`,
- либо в `memory_core/retrieval/*`,
- либо в `memory_core/utils/*`.

Но не в `core/response_pipeline.py`.

---

# 13. Что менять в `api/app.py`

Сейчас debug endpoint смотрит в старый memory-manager snapshot.

Это надо убрать.

## Было
Что-то в духе:

```python
memory_manager.debug_snapshot(...)
```

## Должно стать

```python
memory_core.inspect(...)
```

или

```python
from api.memory_core_api import router as memory_core_router
```

если новый API уже есть в архиве.

---

## Что должен уметь новый debug API

Новый memory debug/inspect должен уметь:

- список raw events;
- список artifacts;
- trace event -> artifacts;
- список workspace sources;
- profile facts;
- task state;
- document chunks;
- debug what was retrieved;
- runtime query preview.

## Чего он не должен делать

- не должен тянуть старые `DebugRequest`;
- не должен зависеть от старого `memory_manager`;
- не должен лазить в legacy memory storage.

---

# 14. Что делать с `memory_core_api`

Если он уже есть в архиве — это плюс.

## Что надо сделать
- подключить роутер в общее приложение;
- проверить, что он реально использует `memory_core`, а не старую память через обходной путь;
- сделать его единственным API для memory inspection.

---

# 15. Что делать с `memory_core_adapter.py`

Это сейчас **самый полезный мост**, если он уже лежит в проекте.

## Роль адаптера

Адаптер нужен, чтобы:
- не ломать весь проект за один коммит;
- дать runtime-слоям простой интерфейс;
- спрятать внутреннюю структуру `memory_core`.

## Интерфейс, который должен быть у адаптера

Минимум:

```python
class MemoryCoreAdapter:
    def ingest_event(...): ...
    def ingest_document(...): ...
    def query(...): ...
    def inspect(...): ...
    def close(...): ...
```

## Что нельзя делать

Не надо через адаптер тянуть наружу все внутренности нового memory_core.
Адаптер должен быть **тонким фасадом**, а не новым `MemoryManager 2.0`.

---

# 16. Что делать с конфигом

Сейчас конфиг, скорее всего, всё ещё содержит legacy memory keys.

## Что удалить или перестать использовать

После переключения runtime должны уйти из активного использования:
- `memory_migration_auto_on_start`
- `memory_migration_schema_version`
- `memory_facts_scope`
- `memory_version`
- все старые веса retrieval/scoring/promotion/lifecycle,
  если они относятся к старому memory runtime.

## Что оставить
Оставить можно только то, что реально нужно новому `memory_core`:

- `memory_dir` или новый `memory_core_dir`
- `memory_embedding_model`
- `memory_embedding_backend`
- `memory_embedding_dim`
- `memory_chunk_size`
- `memory_chunk_overlap`
- `memory_query_top_k`
- `memory_rerank_top_k`
- `memory_context_budget_*`
- `debug_memory_inspector_enabled`

## Что лучше сделать

Я бы постепенно переименовал всё новое в отдельный нейминг:

- `memory_core_dir`
- `memory_core_db_path`
- `memory_core_vector_path`
- `memory_core_top_k`

Чтобы новое не путалось со старым.

---

# 17. Что делать с данными

После реального переключения runtime старая memory storage уже не должна быть нужна.

## Удалить после переключения

- `data/memory_storage/memory_v2/`
- `data/exports/memory_export.json`

## Удалить желательно

- `data/memory_storage/metadata/`
- `data/memory_storage/cache/metadata_extractor/`

## Оставить
- character files
- project logs
- runtime data, не относящиеся к старой memory
- всё, что не является legacy memory storage

## Команды под PowerShell

```powershell
Remove-Item -Recurse -Force .\data\memory_storage\memory_v2
Remove-Item -Force .\data\exports\memory_export.json
Remove-Item -Recurse -Force .\data\memory_storage\metadata
Remove-Item -Recurse -Force .\data\memory_storage\cache\metadata_extractor
```

---

# 18. Что делать со старыми скриптами

## Удалить или вывести из использования

- `scripts/debug_memory_event.py`
- `scripts/debug_memory_message.py`
- `scripts/export_memory.py`
- `scripts/export_memory_json.py`
- `scripts/finalize_memory_backfill.py`
- `scripts/inspect_memory.py`
- `scripts/memory_backfill_answer_only.py`
- `scripts/migrate_memory_compact.py`
- `scripts/reindex_memory_v2.py`
- `scripts/rename_memory_namespace.py`
- `scripts/reset_memory_phase1.py`
- `scripts/restore_memory_compact.py`

## Заменить на новые

Оставить/создать:
- `scripts/memory_core_reset.py`
- `scripts/memory_core_inspect.py`
- `scripts/memory_core_reindex.py`
- `scripts/memory_core_export.py`

---

# 19. Что делать с тестами

## Старые тесты legacy memory
Их не надо тащить как обязательные для новой системы.

Если они проверяют:
- claim promotion
- governor snapshot
- lifecycle
- backfill
- old retrieval projection
- old storage profile
- old write policy

то их надо:
- либо удалить,
- либо пометить как `legacy`,
- либо вынести из основного test run.

## Новые тесты, которые должны стать основными

- `test_memory_core_bootstrap.py`
- `test_memory_core_adapter.py`
- `test_memory_core_ingest.py`
- `test_memory_core_query.py`
- `test_memory_core_documents.py`
- `test_memory_core_inspect.py`
- `test_brain_memory_core_integration.py`
- `test_response_pipeline_memory_core_query.py`
- `test_api_memory_core_endpoints.py`

---

# 20. Что делать со старым `memory/__init__.py`

Пока legacy-пакет ещё жив, его надо **заморозить**.

## Что это значит

- не добавлять новых экспортов;
- не тянуть новый runtime через старые exports;
- не превращать `memory/__init__.py` в адаптер к новой памяти;
- не делать “гибридную магию”.

## Лучше всего
Оставить его как есть только на короткий переходный период,
а потом удалить вместе со старым `memory/`.

---

# 21. Какой должен быть финальный минимальный flow

После перехода нормальный runtime должен выглядеть так:

```python
main.py
  -> init memory_core
  -> передать memory_core в Brain
  -> передать memory_core в ResponsePipeline
  -> подключить memory_core API router

Brain
  -> ingest_event(user/assistant/tool)

ResponsePipeline
  -> query(memory_core)
  -> получает готовые memory blocks

API
  -> inspect(memory_core)
```

И всё.

Без:
- `MemoryManager`
- `MemoryEvent`
- `MemoryScope`
- `MemoryType`
- `build_context`
- `build_memory_native_state`
- `debug_snapshot`
- `legacy EventStore`

---

# 22. Что делать с automation/event log

Если старый `EventStore` нужен не для памяти, а для automation/logging:
- его надо **вынести из memory/**;
- назвать честно, например:
  - `automation_event_log.py`
  - `runtime_event_log.py`

## Очень важное правило
Если что-то нужно не для памяти, не надо оставлять это в `memory/`.

Иначе потом снова начнётся:
“а тут у нас память, но не совсем память”.

---

# 23. Пошаговый план коммитов

## Коммит 1 — зафиксировать цель
- добавить этот план в репозиторий;
- ничего не чинить в legacy memory;
- пометить старую память как frozen.

## Коммит 2 — bootstrap switch
- переписать `main.py` на инициализацию `memory_core`;
- заменить `memory_manager` в контейнере на `memory_core`.

## Коммит 3 — Brain switch
- переписать `core/brain.py`;
- убрать `MemoryEvent`, `MemoryScope`, `MemoryType`;
- перевести user/assistant/tool ingest на новый адаптер.

## Коммит 4 — ResponsePipeline switch
- переписать `core/response_pipeline.py`;
- вырезать legacy context build flow;
- оставить один `query(...)` в `memory_core`.

## Коммит 5 — API switch
- подключить новый memory_core API;
- убрать старый memory inspector flow.

## Коммит 6 — config cleanup
- выкинуть legacy settings из активного использования;
- завести новые `memory_core_*` ключи, если надо.

## Коммит 7 — data cleanup
- удалить старую память из `data/`;
- проверить запуск на пустой новой БД.

## Коммит 8 — legacy freeze complete
- убрать старые скрипты и legacy tests из основного раннера;
- старый `memory/` оставить только как временную папку.

## Коммит 9 — final delete
- удалить старый `memory/`, если ничего от него не зависит;
- удалить legacy imports окончательно.

---

# 24. Признаки, что переход завершён

Считать задачу завершённой можно только когда:

## Runtime
- `main.py` больше не импортирует `MemoryManager`;
- `Brain` не знает про `MemoryEvent`/`MemoryScope`/`MemoryType`;
- `ResponsePipeline` не использует `build_context` legacy memory;
- `api/app.py` не читает старый debug snapshot.

## Data
- `data/memory_storage/memory_v2/` удалена;
- проект стартует на новой memory DB.

## Tests
- основные тесты проходят без старого memory runtime;
- legacy tests либо удалены, либо отключены.

## Codebase
- поиск по проекту не находит активного использования:
  - `MemoryManager(`
  - `MemoryEvent(`
  - `MemoryScope`
  - `MemoryType`
  - `build_context(`
  - `build_memory_native_state(`
  - `build_memory_debug_snapshot(`
  - `normalize_memory_retrieval_plan(`
  - `build_memory_tool_context_pack(`

---

# 25. Поиск по проекту: что надо вычистить

Вот прям список строк, по которым пройтись поиском по проекту:

```text
MemoryManager
MemoryEvent
MemoryScope
MemoryType
build_context(
build_memory_debug_snapshot(
build_memory_native_state(
normalize_memory_retrieval_plan(
build_memory_tool_context_pack(
classify_query_recall_profile(
flatten_governor_profile_snapshot(
build_retrieval_hints(
EventStore(
run_auto_migration(
memory_manager=
self.memory_manager
```

## Что делать с совпадениями

- если это runtime-код — переписать;
- если это tests legacy memory — изолировать/удалить;
- если это scripts — заменить на `memory_core` версии;
- если это docs — обновить текст.

---

# 26. Анти-правила на переходе

## Не делай вот это

### 1. Не лечи `MemoryManager`
Это тупик.
Он legacy.

### 2. Не делай гибридный runtime на полгода
Иначе ты так и останешься с двумя памятьми.

### 3. Не тяни старые enum/model классы в новый слой
Новый memory_core должен остаться чистым.

### 4. Не делай “временный костыль”, который потом станет постоянным
Особенно в `main.py` и `response_pipeline.py`.

### 5. Не смешивай migration, retrieval, debug и persona в одном файле
Иначе снова получишь старый `memory_manager`.

---

# 27. Мой жёсткий вывод

Сейчас тебе **не надо развивать memory дальше как feature**.

Тебе надо:
1. **остановить рост legacy memory**;
2. **переключить runtime на `memory_core`**;
3. **удалить старый memory-слой как основу проекта**.

То есть задача звучит не:
> “улучшить память”

А так:
> **“ликвидировать старую память как runtime-зависимость проекта”**

И это сейчас самый правильный ход.

---

# 28. Очень короткое резюме

## Было
- старая память управляет runtime;
- новая память уже лежит рядом;
- проект живёт на двух системах сразу.

## Надо сделать
- runtime полностью перевести на `memory_core`;
- старую память оставить только как временный legacy-слой;
- потом удалить старый `memory/`, старую БД, старые скрипты и старые тесты.

## Главный критерий успеха
Проект должен **запускаться, писать, читать, дебажить и отвечать через `memory_core`**, вообще не используя старый `memory_manager`.
