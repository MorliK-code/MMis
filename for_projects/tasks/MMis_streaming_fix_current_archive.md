# MMis — исправление live-streaming текста без отключения tools (по текущему архиву)

## Что ты хочешь

Ты хочешь, чтобы:

- текст **показывался по ходу генерации**;
- **tools остались включены**;
- допустимо, если иногда увидишь **преждевременный ответ** до tool call;
- не ломать agent loop, memory tools и history tools.

Это правильная цель.

---

# 1. Главный вывод по текущему архиву

По текущему архиву проблема с "текст приходит сразу целиком" сейчас **не в UI** и уже **не в `probe_*` буферизации**.

## Почему

В `core/response_pipeline.py` в `_generate_with_agent_loop_streaming(...)` у тебя уже стоит:

```python
# В streaming-режиме не буферизуем первые видимые куски ответа.
probe_tool_calls = False
probe_released = True
```

И `_handle_visible_answer(...)` уже шлёт куски сразу в callback.

То есть старая причина “UI молчит, потому что первые чанки буферятся” для этого архива **уже почти снята**.

---

# 2. Что реально ломает live-stream прямо сейчас

## Основная причина
В этом архиве generate-stage всё ещё падает **до старта stream provider**, поэтому UI не получает ни одного chunk.

### Симптом из логов
В `data/logs/app.log` видно:

- `api_chat_stream_done`
- `answer_chars > 0`
- но `streamed_answer_chars = 0`
- и `streamed_thinking_chars = 0`

Это означает:
- финальный ответ пришёл;
- но **callbacks `_on_answer/_on_thinking` вообще не вызывались**;
- следовательно, provider.stream не дошёл до стадии отдачи видимых кусочков.

## Почему это происходит
В `core/response_pipeline.py` у тебя всё ещё есть:

```python
"enum": [scope.value for scope in _default_memory_scopes()],
```

внутри `_memory_retrieve_tool_spec()`.

Но `_default_memory_scopes()` в архиве отсутствует.
Из-за этого на chat-path, где включён agent loop и собираются tools, происходит `NameError` **ещё до реального stream**.

### Итог
Вместо живого стрима ты получаешь:
- fallback текст,
- или обычный финальный ответ целиком,
- но без `chunk` событий.

---

# 3. Что важно понять

## Сейчас проблема двухслойная

### Слой 1 — критичный
Generate-stage падает раньше, чем начинается live-stream.

### Слой 2 — желаемое поведение
Даже после исправления generate-stage тебе нужно сохранить такое поведение:

- tools **не выключаем**;
- visible text chunks отправляем **сразу**;
- tool calls продолжают собираться через `tool_calls_delta`;
- если модель сначала начала отвечать, а потом решила вызвать tool — **мы не скрываем уже показанный текст**.

Это как раз соответствует твоему требованию.

---

# 4. Что менять обязательно

## 4.1. Починить `_memory_retrieve_tool_spec()`

### Файл
`core/response_pipeline.py`

### Найти
```python
"enum": [scope.value for scope in _default_memory_scopes()],
```

### Заменить
На literal enum list:

```python
"enum": [
    "conversation",
    "session",
    "project",
    "global_user",
    "character",
    "temporary",
],
```

### Готовый блок целиком

```python
def _memory_retrieve_tool_spec() -> ToolSpec:
    return ToolSpec(
        name=_MEMORY_TOOL_NAME,
        description=(
            "Retrieve memory candidates and context blocks for the current conversation. "
            "Call it when you need past facts, episodes, tasks, or profile context and answer cannot be grounded from the current prompt alone."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Retrieval mode such as profile, fact, episode, task, document, or context.",
                },
                "topic_hints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Short topic anchors for deterministic memory fan-out.",
                },
                "time_hint": {
                    "type": "string",
                    "description": "Temporal bias such as recent, session, historical, persistent, or any.",
                },
                "scopes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "conversation",
                            "session",
                            "project",
                            "global_user",
                            "character",
                            "temporary",
                        ],
                    },
                    "description": "Optional memory scopes to search.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred memory sources such as facts, episodes, tasks, profile, documents, or messages.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 12,
                    "description": "How many memory candidates to retrieve.",
                },
            },
            "required": ["mode", "topic_hints", "time_hint", "sources", "top_k"],
            "additionalProperties": False,
        },
    )
```

---

## 4.2. Упростить `_generate_with_agent_loop_streaming()`

Сейчас там уже отключена probe-буферизация, но остались мёртвые переменные:

- `probe_answer_buffer`
- `probe_thinking_buffer`
- `probe_visible_pieces`
- `probe_released`

Они уже не нужны.

### Почему это стоит убрать
Чтобы:
- не путаться в логике;
- не было ощущения, что стрим ещё где-то буферится;
- future refactor не “вернул” старое поведение случайно.

### Было
```python
probe_tool_calls = False
probe_answer_buffer: list[str] = []
probe_thinking_buffer: list[str] = []
probe_visible_pieces = 0
probe_released = True
```

### Должно стать
Вообще просто удалить этот блок.

---

## 4.3. Оставить tools работающими

Это важно: **не трогай** вот это поведение:

```python
tool_calls_delta = list(getattr(chunk, "tool_calls_delta", []) or [])
if tool_calls_delta:
    for call in tool_calls_delta:
        ...
        tool_calls.append(call)
```

И не трогай логику:

- `_generate_with_agent_loop(...)`
- `_execute_agent_tool_call(...)`
- `_execute_memory_retrieve_tool(...)`
- `_execute_history_tool(...)`

Потому что tools у тебя уже идут по отдельному каналу через `tool_calls_delta`, а не через визуальный текст.

### Очень важная мысль
Live answer streaming и tool execution — это **разные каналы**.

- Видимый текст → `on_answer(piece)`
- Thinking → `on_thinking(piece)`
- Tool invocation → `tool_calls_delta`

Поэтому **не надо скрывать текст ради tools**.
Tools и так продолжают работать.

---

# 5. Что менять желательно

## 5.1. Сделать более честный лог stream-fallback

Сейчас, если provider.stream падает, у тебя идёт:

```python
ctx.logs.append(f"stage=generate agent_loop_stream_error={type(exc).__name__}")
ctx.errors.append(f"agent_loop_stream:{type(exc).__name__}:{exc}")
ctx.logs.append("stage=generate agent_loop_stream_fallback=generate")
return self.provider.generate(req)
```

Это допустимо, но сейчас fallback полностью скрывает причину, почему исчезли live-chunks.

### Лучше так
Оставить fallback, но логировать подробнее:

```python
except Exception as exc:
    ctx.logs.append(
        f"stage=generate agent_loop_stream_error={type(exc).__name__}:{exc}"
    )
    ctx.errors.append(f"agent_loop_stream:{type(exc).__name__}:{exc}")
    ctx.logs.append("stage=generate agent_loop_stream_fallback=generate")
    return self.provider.generate(req)
```

### Зачем
Если live-stream снова исчезнет, ты сразу увидишь:
- это падение stream provider,
- NameError,
- tool schema issue,
- parser issue,
- или сетевой rollback.

---

## 5.2. Добавить явный лог, что callbacks реально срабатывают

В `_generate_with_agent_loop_streaming(...)` можно временно добавить диагностические счётчики.

### Пример
```python
visible_answer_chunks = 0
visible_thinking_chunks = 0
```

В `_handle_visible_answer(...)`:

```python
nonlocal visible_answer_chunks
visible_answer_chunks += 1
```

В `_handle_visible_thinking(...)`:

```python
nonlocal visible_thinking_chunks
visible_thinking_chunks += 1
```

После цикла:

```python
ctx.logs.append(
    f"stage=generate agent_loop_stream_visible answer_chunks={visible_answer_chunks} "
    f"thinking_chunks={visible_thinking_chunks} tool_calls={len(tool_calls)}"
)
```

### Зачем
Чтобы не гадать:
- дошли ли chunks до callbacks;
- были ли tool calls;
- был ли реально text_delta.

---

# 6. Что НЕ надо менять

## Не надо отключать agent loop
Иначе потеряешь tools.

## Не надо выкидывать `tool_calls_delta`
Иначе tools сломаешь.

## Не надо возвращать probe-буферизацию
Иначе UI снова станет “мёртвым”.

## Не надо скрывать текст до завершения tool pass
Ты сам сказал, что ок видеть преждевременный ответ — значит это допустимо.

---

# 7. Что можно добавить как fallback UX

Это не обязательный фикс live-streaming, но полезно.

## Идея
Если за весь `/chat/stream`:

- `streamed_answer_chars == 0`
- но финальный `answer` не пустой

можно эмитить **один поздний chunk перед final**.

### Где
`api/app.py`, в `chat_stream()` перед `yield _ndjson("final", payload)`.

### Пример
```python
if not sent_answer and result.answer:
    yield _ndjson("chunk", str(result.answer or ""))
```

### Зачем
Это не сделает стрим живым, но:
- уберёт ситуацию, где UI не получает ни одного chunk вообще;
- упростит диагностику;
- не сломает tools.

### Но
Это **не замена настоящему streaming**.
Это только safety fallback.

---

# 8. Почему UI, скорее всего, уже ок

По текущему архиву у тебя UI-часть выглядит нормальной:

- `ReplyWorker` вызывает `api.stream_chat(...)`
- `ui/api_client.py` читает NDJSON по строкам
- `kind == "chunk"` вызывает `on_chunk(piece)`
- `ui/app.py` принимает `worker.chunk.connect(self._on_reply_chunk)`
- `_on_reply_chunk()` сразу вызывает `_flush_stream_chunks()`
- `_flush_stream_chunks()` обновляет текущий message row без полного rerender

То есть после починки generate-stage и tool schema у тебя уже должно заработать “печатание по ходу”.

---

# 9. Что проверить после исправления

## 9.1. Простой запрос
`привет`

### Ожидаемо
- нет fallback-ответа;
- в `app.log` появляются реальные streaming events;
- `streamed_answer_chars > 0`.

---

## 9.2. Длинный ответ без tool call
Например:
`расскажи подробно про event loop в python`

### Ожидаемо
- текст идёт кусками;
- UI обновляется на ходу;
- `chunk_count > 1`.

---

## 9.3. Запрос с memory tool
Например:
`о чём мы говорили вчера?`

### Ожидаемо
- tool path остаётся живой;
- `memory_retrieve` может вызваться;
- текст может пойти до tool вызова или после него — и это ок;
- tools не ломаются.

---

## 9.4. Запрос с history tool
Например:
`покажи последние сообщения`

### Ожидаемо
- exact-read tool работает;
- текст по-прежнему стримится, если модель начинает отвечать до конца цикла.

---

# 10. Минимальный рабочий план

## Шаг 1
Исправить `_default_memory_scopes()` bug.

## Шаг 2
Удалить мёртвые `probe_*` переменные из `_generate_with_agent_loop_streaming()`.

## Шаг 3
Добавить подробный лог stream-visible callbacks.

## Шаг 4
Прогнать:
- простой chat,
- memory tool,
- history tool.

## Шаг 5
Если chunks всё ещё 0 — смотреть уже:
- `llm_stream_start`
- `llm_stream_done`
- `tool_calls_delta`
- `agent_loop_stream_error`

---

# 11. Самый важный итог

По **этому** архиву главный фикс не “починить UI-streaming с нуля”.

Главный фикс такой:

1. **убрать runtime-ошибку в tool schema**, из-за которой stream вообще не стартует;
2. **оставить текущую политику без буферизации первых кусков**;
3. **не отключать tools**, потому что они уже идут отдельным каналом и не мешают live-text.

Именно это даст тебе желаемое поведение:

- tools живы,
- visible text идёт сразу,
- преждевременный ответ допускается,
- UI не ждёт финала, чтобы показать весь текст целиком.
