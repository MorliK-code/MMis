# Исправление бага: нормальный ответ + `Я затупила...` / утечка thinking в MMis

## Что сейчас происходит

По симптомам из последнего архива у проекта одновременно проявляются **две отдельные проблемы**:

1. **Ответ модели успевает сгенерироваться, но потом заменяется fallback-ответом**:
   - `Я затупила. Повтори, пожалуйста, еще раз.`
   - иногда этот fallback идёт **вместо** ответа;
   - иногда он **дописывается после** уже нормального ответа.

2. При `Thinking: on` у `qwen3:8b` в обычный ответ попадает сырой reasoning / chain-of-thought:
   - `Okay, let's see...`
   - `The user said...`
   - и т.д.

Из-за этого создаётся ощущение, что модель ломается, перегенерирует ответ и потом зацикливается. Но по коду видно, что главная причина — не сама модель, а **архитектурный сбой после генерации ответа**.

---

## Главный корень проблемы

### 1) `Brain.handle_message()` заменяет уже готовый ответ на fallback, если ломается пост-обработка

В `core/brain.py` сейчас логика выглядит так:

```python
pipeline_result = self.pipeline.run(...)
result = self._to_brain_result(...)
memory_apply_summary = self._apply_memory_ops(...)
self._update_state_after_success(...)
persisted_summary = self._persist_turns(...)
self._append_turn_summaries(...)
self._harvest_response_stats(...)
```

И всё это находится **в одном большом `try`**.

Если после успешной генерации ответа падает **любой** из шагов:
- запись памяти;
- debug snapshot;
- state update;
- summary append;
- stats harvest;

то срабатывает:

```python
result = self._build_error_result(route=route, error=exc)
```

А `_build_error_result()` возвращает:

```python
fallback = "Я затупила. Повтори, пожалуйста, еще раз."
```

### Что это значит на практике

Модель могла ответить нормально, но потом сломался не ответ, а **хвост обработки** — и весь удачный ответ уничтожается fallback-ом.

---

### 2) Падает debug/inspect memory-хвост, потому что `MemoryInspector` живёт на старом API store-слоя

В `memory_core/inspect/memory_inspector.py` используются методы, которых в текущих store уже нет:

#### Сейчас вызываются:
- `event_store.get_recent(...)`
- `event_store.get_by_workspace(...)`
- `artifact_store.get_all(...)`
- `artifact_store.get_by_workspace(...)`
- `artifact_store.get_by_type(...)`
- `artifact_store.get_by_event(...)`
- `workspace_store.get_all(...)`

#### А реально в store есть:
- `event_store.list_events(...)`
- `event_store.list_by_workspace(...)`
- `artifact_store.list_artifacts(...)`
- `artifact_store.get_by_source_event(...)`
- `workspace_store.list_workspaces()`

То есть инспектор отстал от текущей storage API.

Из-за этого после генерации ответа может падать примерно такая цепочка:

```text
Brain.handle_message()
 -> _persist_turns()
 -> _update_debug_trace_after_persist()
 -> memory_core.debug_snapshot()
 -> memory_core.inspect.memory_inspector._inspect_events()
 -> AttributeError: 'EventStore' object has no attribute 'get_recent'
```

И после этого `Brain` заменяет готовый ответ на fallback.

---

## Второй корень: утечка thinking в ответ при `qwen3`

В `core/response_pipeline.py` `_ThinkStreamParser` умеет прятать thinking только в двух случаях:

1. когда провайдер отдаёт отдельное поле `thinking_delta`;
2. когда reasoning приходит в тегах:
   - `<think>...</think>`
   - `<thinking>...</thinking>`
   - `<reasoning>...</reasoning>`

Фрагмент текущей логики:

```python
self._open_tags = ("<think>", "<thinking>", "<reasoning>")
self._close_tags = ("</think>", "</thinking>", "</reasoning>")
```

Проблема в том, что `qwen3:8b` в твоём сетапе иногда отдаёт reasoning **просто как обычный текст**, без отдельного `thinking_delta` и без тегов.

Тогда parser считает его обычным ответом и спокойно пропускает в UI.

Отсюда и такое поведение:

```text
assistant> Okay, let's see. The user said...
assistant> I should respond...
assistant> Нормально, спасибо. А у тебя как?
assistant> Я затупила. Повтори, пожалуйста, еще раз.
```

То есть у тебя одновременно:
- visible reasoning leakage;
- и падение пост-обработки памяти.

---

# Что нужно сделать

Ниже — порядок фикса **по приоритету**.

---

## Этап 1. Перестать уничтожать готовый ответ из-за ошибок памяти/debug

### Файл
`core/brain.py`

### Проблема
Слишком широкий `try/except` вокруг всей цепочки ответа.

### Что нужно
Разделить:
- **генерацию ответа**
- и **пост-обработку**

Так, чтобы если память/инспектор/статистика упали, пользователь всё равно получил уже сгенерированный ответ.

### Что поменять

#### Сейчас по сути так

```python
try:
    pipeline_result = self.pipeline.run(...)
    result = self._to_brain_result(...)
    memory_apply_summary = self._apply_memory_ops(...)
    self._update_state_after_success(...)
    persisted_summary = self._persist_turns(...)
    self._append_turn_summaries(...)
    self._harvest_response_stats(...)
except Exception as exc:
    result = self._build_error_result(route=route, error=exc)
```

#### Нужно сделать так

```python
try:
    pipeline_result = self.pipeline.run(
        route=route,
        user_msg=(event_name if route == "system_event" else text),
        state=state_map,
        meta=meta_for_pipeline,
        retrieved_memories=retrieved_memories,
        traits=traits,
        policies=policies,
    )
    result = self._to_brain_result(route=route, pipeline_result=pipeline_result)
except Exception as exc:
    LOGGER.exception(
        "Brain.handle_message failed route=%s trace_id=%s request_id=%s",
        route,
        str(meta_for_pipeline.get("trace_id") or ""),
        str(meta_for_pipeline.get("request_id") or ""),
    )
    result = self._build_error_result(route=route, error=exc)
else:
    memory_apply_summary = self._apply_memory_ops(result.memory_ops)

    try:
        self._update_state_after_success(
            route=route,
            event_name=event_name,
            meta=meta_for_pipeline,
            result=result,
            track_state=state_updates_enabled,
        )
    except Exception as exc:
        LOGGER.exception("post-success state update failed: %s", exc)
        result.logs.append(f"post_success_warning={type(exc).__name__}:{exc}")

    try:
        persisted_summary = self._persist_turns(
            route=route,
            user_text=text,
            result=result,
            meta=meta_for_pipeline,
            non_persistent_turn=non_persistent_turn,
        )
    except Exception as exc:
        LOGGER.exception("persist_turns failed: %s", exc)
        persisted_summary = self._empty_memory_write_summary()
        result.logs.append(f"persist_warning={type(exc).__name__}:{exc}")

    try:
        self._append_turn_summaries(
            result=result,
            route=route,
            user_text=text,
            meta=meta_for_pipeline,
            memory_apply_summary=memory_apply_summary,
            persisted_summary=persisted_summary,
        )
    except Exception as exc:
        LOGGER.exception("append_turn_summaries failed: %s", exc)
        result.logs.append(f"turn_summary_warning={type(exc).__name__}:{exc}")

    try:
        self._harvest_response_stats(
            result=result,
            route=route,
            meta=meta_for_pipeline,
            conversation_id=conversation_id,
        )
    except Exception as exc:
        LOGGER.exception("harvest_response_stats failed: %s", exc)
        result.logs.append(f"stats_warning={type(exc).__name__}:{exc}")
```

### Зачем это нужно
После этой правки:
- модельный ответ не будет уничтожаться;
- fallback останется только для настоящего сбоя генерации;
- memory/debug ошибки будут логироваться как warning, а не ломать UX.

### Критерий готовности
Если `memory_core.inspect(...)` упадёт, пользователь всё равно должен увидеть нормальный ответ без `Я затупила...`.

---

## Этап 2. Починить `MemoryInspector` под текущий store API

### Файл
`memory_core/inspect/memory_inspector.py`

### Проблема
Инспектор вызывает устаревшие методы store-слоя.

---

### 2.1. `_inspect_events()`

#### Сейчас

```python
if workspace_id:
    events = self.event_store.get_by_workspace(workspace_id, limit=limit)
else:
    events = self.event_store.get_recent(limit=limit)
```

#### Нужно

```python
def _inspect_events(
    self,
    limit: int = 50,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    events = self.event_store.list_events(
        workspace_id=workspace_id,
        limit=limit,
    )

    return {
        "items": [e.to_dict() for e in events],
        "count": len(events),
        "limit": limit,
    }
```

> Если `MemoryEnvelope` у тебя не имеет `to_dict()`, тогда делай руками через `dict`-сборку по полям.

Например:

```python
"items": [
    {
        "event_id": e.event_id,
        "source_kind": e.source_kind,
        "payload_type": e.payload_type,
        "text": e.text,
        "metadata": dict(e.metadata or {}),
        "namespace": e.namespace,
        "workspace_id": e.workspace_id,
        "session_id": e.session_id,
        "ts": e.ts,
    }
    for e in events
]
```

---

### 2.2. `_inspect_artifacts()`

#### Сейчас

```python
if workspace_id:
    artifacts = self.artifact_store.get_by_workspace(workspace_id, limit=limit)
else:
    artifacts = self.artifact_store.get_all(limit=limit)
```

#### Нужно

```python
def _inspect_artifacts(
    self,
    limit: int = 50,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    artifacts = self.artifact_store.list_artifacts(
        workspace_id=workspace_id,
        limit=limit,
    )

    return {
        "items": [a.to_dict() for a in artifacts],
        "count": len(artifacts),
        "limit": limit,
    }
```

---

### 2.3. `_inspect_workspaces()`

#### Сейчас

```python
workspaces = self.workspace_store.get_all(limit=limit)
```

#### Нужно

```python
def _inspect_workspaces(self, limit: int = 50) -> dict[str, Any]:
    workspaces = list(self.workspace_store.list_workspaces())[:limit]

    return {
        "workspaces": [dict(w) for w in workspaces],
        "count": len(workspaces),
        "limit": limit,
    }
```

---

### 2.4. `_inspect_profile()`

#### Сейчас

```python
profile_facts = self.artifact_store.get_by_type(
    artifact_type="profile_fact",
    workspace_id=ws_id,
    status="active",
    limit=20,
)

preferences = self.artifact_store.get_by_type(
    artifact_type="preference",
    workspace_id=ws_id,
    status="active",
    limit=20,
)
```

#### Нужно

```python
profile_facts = self.artifact_store.list_artifacts(
    artifact_type="profile_fact",
    workspace_id=ws_id,
    status="active",
    limit=20,
)

preferences = self.artifact_store.list_artifacts(
    artifact_type="preference",
    workspace_id=ws_id,
    status="active",
    limit=20,
)
```

---

### 2.5. `_inspect_trace()`

#### Сейчас

```python
artifacts = self.artifact_store.get_by_event(event_id)
```

#### Нужно

```python
artifacts = self.artifact_store.get_by_source_event(event_id)
```

---

### Критерий готовности
Все вызовы:
- `memory_core.inspect(kind="events")`
- `memory_core.inspect(kind="artifacts")`
- `memory_core.inspect(kind="workspaces")`
- `memory_core.inspect(kind="profile")`
- `memory_core.inspect(kind="trace", event_id=...)`

должны отрабатывать **без `AttributeError`**.

---

## Этап 3. Временно отключить thinking для `qwen3`, чтобы перестал течь reasoning в ответ

### Файл
`api/app.py`

### Почему это нужно
Пока parser умеет скрывать только:
- `thinking_delta`
- или `<think>...</think>`

а `qwen3` у тебя иногда отдаёт размышления просто plain text.

Значит быстрый и безопасный hotfix сейчас — не давать `think=True` для `qwen3`.

---

### Что менять

#### Сейчас

```python
meta: dict[str, Any] = {
    "model": _runtime.model,
    "think": _runtime.thinking_enabled if req.think is None else bool(req.think),
    ...
}
```

#### Нужно

```python
def _build_chat_meta(req: ChatRequest, *, source: str, **extra: Any) -> dict[str, Any]:
    active_profile, quality_profile = _resolve_effective_profiles()
    profile = get_profile(active_profile)

    requested_think = _runtime.thinking_enabled if req.think is None else bool(req.think)
    model_lower = str(_runtime.model or "").strip().lower()

    if requested_think and "qwen3" in model_lower:
        requested_think = False

    meta: dict[str, Any] = {
        "model": _runtime.model,
        "think": requested_think,
        "verbose": _runtime.verbose_enabled if req.verbose is None else bool(req.verbose),
        "web_mode": str(_runtime.web_mode),
        "json_mode": _runtime.json_mode_enabled if req.json_mode is None else bool(req.json_mode),
        "store_turn": bool(req.store_turn),
        "source": str(source or "api"),
        "quality_profile": str(quality_profile or "BALANCED"),
        "temperature": float(profile.generation.temperature),
        "top_p": float(profile.generation.top_p),
        "repeat_penalty": float(profile.generation.repeat_penalty),
    }
    ...
```

### Что это даст
Сразу пропадут случаи, где в ответ уезжает:
- `Okay, let's see...`
- `The user said...`
- и прочий сырой reasoning.

### Критерий готовности
При `Thinking: on` и модели `qwen3:8b` в UI больше не должен появляться raw reasoning как часть финального ответа.

---

## Этап 4. Потом уже можно делать нормальный thinking-filter для `qwen3`

### Файл
`core/response_pipeline.py`

### Проблема
`_ThinkStreamParser` не умеет ловить untagged reasoning.

### Важно
Это **не срочный фикс**, а улучшение после стабилизации.

### Почему не стоит делать это первым
Если сделать слишком агрессивный эвристический фильтр, можно случайно:
- откусить нормальный ответ;
- сломать стриминг;
- скрыть полезный текст.

### Что можно сделать позже
Сделать отдельный “soft parser”, который:
- смотрит только на начало ответа;
- если первые N символов выглядят как англоязычный CoT (`Okay, let's see`, `The user said`, `I should`, `Need to`),
- буферизует это как hidden thinking;
- и только после перехода на нормальный ответ начинает стримить пользователю.

Но это уже отдельная задача, не hotfix.

---

# Дополнительный техдолг, который лучше не оставлять

Это не то, что прямо вызывает текущую повторяющуюся фразу каждый раз, но оно очень рядом и может бить дальше.

---

## 5. `memory_core/identity/identity_core.py` тоже живёт на старом API `ArtifactStore`

### Проблема
Там используются методы:
- `artifact_store.get_by_type(...)`
- `artifact_store.save(...)`

А в текущем `ArtifactStore` у тебя есть:
- `list_artifacts(...)`
- `create(...)`
- `update(...)`
- `create_many(...)`
- `update_status(...)`

### Что это значит
Даже если инспектор починить, identity-слой всё равно может в другом сценарии упасть на несовместимости API.

### Что делать
Минимум пройтись по `identity_core.py` и заменить:

#### Было

```python
artifacts = self.artifact_store.get_by_type(...)
```

#### Надо

```python
artifacts = self.artifact_store.list_artifacts(...)
```

И вместо `save(...)` сделать разведение:
- если артефакт новый → `create(...)`
- если уже существует → `update(...)`

### Приоритет
Не ниже среднего. Лучше исправить сразу после `MemoryInspector`.

---

# Порядок внедрения

## Минимальный hotfix, чтобы проект перестал ломать ответы

1. `core/brain.py` — разделить генерацию и пост-обработку.
2. `memory_core/inspect/memory_inspector.py` — перевести на актуальные store методы.
3. `api/app.py` — временно выключить think для `qwen3`.

Этого уже должно хватить, чтобы:
- исчезла бесконечная `Я затупила...` после нормального ответа;
- исчезли большинство случаев с raw reasoning в обычном тексте;
- память/debug перестали ломать чат.

---

## Следующий слой стабилизации

4. `memory_core/identity/identity_core.py` — убрать вызовы старого API.
5. Проверить все остальные места по проекту на:
   - `get_recent`
   - `get_all`
   - `get_by_type`
   - `get_by_event`
   - `save`

Команда для поиска по проекту:

```bash
rg -n "get_recent|get_all|get_by_type|get_by_event|\.save\(" core memory_core api
```

---

# Как проверять после фикса

## Тест-кейсы руками

### Кейc 1 — обычный чат

Ввод:

```text
как дела?
```

Ожидание:
- ответ нормальный;
- в конце **нет** `Я затупила...`.

---

### Кейc 2 — приветствие

Ввод:

```text
привет
```

Ожидание:
- обычный короткий ответ;
- без fallback;
- без английского внутреннего reasoning.

---

### Кейc 3 — `Thinking: on` на `qwen3:8b`

Ввод:

```text
как дела?
```

Ожидание после hotfix:
- либо normal answer без thinking;
- либо hidden thinking не показывается;
- но в финальный текст **не должны** попадать куски вида `Okay, let's see...`.

---

### Кейc 4 — искусственно сломать inspector

Временно вставить исключение в `memory_core.inspect(...)` или `_update_debug_trace_after_persist()`.

Ожидание:
- основной ответ остаётся пользователю;
- в логах warning/exception;
- fallback **не заменяет** готовый ответ.

---

# Что считать успешным фиксом

Фикс можно считать готовым, если одновременно выполнены все пункты:

- [ ] `Brain.handle_message()` не заменяет удачный ответ на fallback из-за memory/debug ошибок.
- [ ] `MemoryInspector` больше не падает на старых store методах.
- [ ] `qwen3:8b` перестал сливать reasoning в обычный текст хотя бы в режиме hotfix.
- [ ] В обычном чате больше не появляется `Я затупила...` после уже сгенерированного ответа.
- [ ] `/debug/memory-inspector` и внутренний `debug_snapshot()` отрабатывают без `AttributeError`.

---

# Коротко: что сломано и что чинит проблему

## Сломано
- post-processing слишком хрупкий;
- inspector не совпадает с текущим storage API;
- qwen3 даёт untagged thinking, а parser умеет ловить только tagged/structured thinking.

## Чинит ситуацию
- разделение try/except в `Brain`;
- актуализация `MemoryInspector`;
- временное отключение `think` для `qwen3`.

---

# Рекомендуемый итоговый порядок коммитов

## Commit 1
`fix(brain): preserve generated answer when memory/debug post-processing fails`

## Commit 2
`fix(memory-inspector): migrate inspector to current store api`

## Commit 3
`fix(api): disable think mode for qwen3 pending safe parser`

## Commit 4
`refactor(identity-core): replace legacy artifact store api usage`

---

# Итог

Это не выглядит как деградация модели или поломка всей памяти. Это выглядит как:

1. **нормальный ответ генерируется успешно**;
2. **падает хвост памяти/debug**;
3. `Brain` ошибочно затирает успешный результат fallback-ом;
4. параллельно `qwen3` местами сливает внутреннее thinking в ответ.

То есть баг неприятный, но локальный и вполне нормально чинится.
