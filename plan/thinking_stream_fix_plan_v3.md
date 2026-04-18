# План фикса thinking-stream в MMis

## Цель
Сделать `thinking` таким же live-stream, как `answer`: без искусственной нарезки, без псевдо-стриминга, без накопления нескольких предложений в один UI-апдейт.

---

## 1. Убрать искусственную нарезку thinking в API

### Файл
`api/app.py`

### Было
```python
def _on_thinking(piece: str) -> None:
    chunk = str(piece or "")
    if chunk:
        for subchunk in _split_stream_display_piece(chunk, max_chars=12):
            if subchunk:
                events.put(("thinking", subchunk), block=False)
```

### Нужно
```python
def _on_thinking(piece: str) -> None:
    chunk = str(piece or "")
    if chunk:
        events.put(("thinking", chunk), block=False)
```

### Зачем
Сейчас `thinking` дробится отдельно от `answer`, поэтому UI получает не реальные чанки от провайдера, а искусственно переделанные куски.

---

## 2. Исправить вычисление delta для thinking

### Файл
`llm/ollama_provider.py`

### Было
```python
def _stitch_thinking_delta(prev_full: str, current_full: str) -> tuple[str, str]:
    current = str(current_full or "")
    prev = str(prev_full or "")

    if not current:
        return "", prev

    if current.startswith(prev):
        delta = current[len(prev):]
        return delta, current

    return current, current
```

### Нужно
```python
def _stitch_thinking_delta(prev_full: str, current_piece: str) -> tuple[str, str]:
    prev = str(prev_full or "")
    current = str(current_piece or "")

    if not current:
        return "", prev

    # cumulative mode
    if prev and current.startswith(prev):
        return current[len(prev):], current

    # first chunk
    if not prev:
        return current, current

    # delta mode
    return current, prev + current
```

### Зачем
Текущая версия нормально работает только в cumulative-режиме. Если Ollama начинает отдавать thinking как обычные delta-чанки, код воспринимает их как «новый полный thinking» и отправляет в UI слишком крупные куски.

---

## 3. Таймер thinking считать от первого реального thinking-чанка

### Файл
`ui/chat_window.py`

### Было
```python
@Slot(str)
def _on_thinking_chunk(self, piece: str) -> None:
    if not self._pending:
        return
    now = time.perf_counter()
    if self._pending.first_thinking_at is None:
        self._pending.first_thinking_at = now
    self._pending.last_chunk_at = now
    self._pending.thinking_text += piece or ""
    elapsed_ms = int((time.perf_counter() - self._pending.started_at) * 1000)
    self._pending.bubble.update_thinking(self._pending.thinking_text, self._format_duration_label(elapsed_ms))
    self._schedule_scroll_bottom(follow_stream_only=True)
```

### Нужно
```python
@Slot(str)
def _on_thinking_chunk(self, piece: str) -> None:
    if not self._pending:
        return

    piece_text = str(piece or "")
    if not piece_text:
        return

    now = time.perf_counter()
    self._pending.last_chunk_at = now
    self._pending.thinking_text += piece_text

    if self._pending.first_thinking_at is None and piece_text.strip():
        self._pending.first_thinking_at = now

    started = self._pending.first_thinking_at or now
    elapsed_ms = max(1, int((now - started) * 1000))

    self._pending.bubble.update_thinking(
        self._pending.thinking_text,
        self._format_duration_label(elapsed_ms),
    )
    self._schedule_scroll_bottom(follow_stream_only=True)
```

### Зачем
Кнопка/счётчик не должны жить от старта запроса. Они должны стартовать только с первого реально видимого thinking-текста.

---

## 4. Исправить финальный thinking_ms

### Файл
`ui/chat_window.py`

### Было
```python
if self._pending.first_answer_at is not None and thinking_generated:
    local_thinking_ms = max(1, int((self._pending.first_answer_at - self._pending.started_at) * 1000))
    local_answer_ms = max(1, int((finished_at - self._pending.first_answer_at) * 1000))
```

### Нужно
```python
if (
    self._pending.first_answer_at is not None
    and self._pending.first_thinking_at is not None
    and thinking_generated
):
    local_thinking_ms = max(1, int((self._pending.first_answer_at - self._pending.first_thinking_at) * 1000))
    local_answer_ms = max(1, int((finished_at - self._pending.first_answer_at) * 1000))
elif self._pending.first_answer_at is not None:
    local_answer_ms = max(1, int((finished_at - self._pending.first_answer_at) * 1000))
```

### Зачем
Иначе после завершения ответа в плашку снова попадает завышенное время thinking.

---

## 5. Если используешь старое окно — исправить и там

### Файл
`ui/chat_shell.py`

### Что исправить
Та же логика, что и в `ui/chat_window.py`:
- считать live-thinking от `first_thinking_at`, а не от `started_at`;
- не ставить финальный `thinking_ms = elapsed` для всего запроса.

---

## 6. Обновить тесты

### Файл
`tests/test_api_streaming_behavior.py`

### Что убрать
Тест, который ожидает искусственное дробление одного thinking-чанка на несколько live-events.

### Что добавить
Проверку:
- один `stream_on_thinking_chunk(thinking)` -> один `thinking` event в stream.

---

## 7. Добавить тест на два режима thinking от провайдера

### Файл
`tests/test_ollama_thinking_delta.py`

### Добавить кейсы
1. **cumulative mode**
```python
prev = "Hello"
current = "Hello world"
# delta == " world"
```

2. **delta mode**
```python
prev = "Hello"
current = " world"
# delta == " world"
# new_full == "Hello world"
```

---

## Ожидаемый результат после фикса
- `answer` и `thinking` идут одним и тем же принципом: как пришли, так и ушли в UI.
- Кнопка `thinking` появляется с первым реальным thinking-текстом.
- Таймер `thinking` не стартует заранее.
- После первых 10–20 слов поток не начинает склеиваться искусственно из-за старой delta-логики.

---

## Что проверить руками
1. Запустить модель с thinking.
2. Убедиться, что в логике `api/app.py` больше нет `_split_stream_display_piece(..., max_chars=12)` для thinking.
3. Проверить, что `thinking` приходит в UI теми же chunk-ивентами, как отдаёт backend.
4. Проверить длинный reasoning-ответ: начало, середина и конец должны идти одинаково live.
5. Проверить кейс без thinking: кнопка и таймер не должны появляться заранее.
