# План фикса thinking-stream в MMis

## Что исправить

### 1. Убрать искусственную нарезку thinking в API
Файл: `api/app.py`

Заменить:
```python
        def _on_thinking(piece: str) -> None:
            chunk = str(piece or "")
            if chunk:
                for subchunk in _split_stream_display_piece(chunk, max_chars=12):
                    if subchunk:
                        events.put(("thinking", subchunk), block=False)
```

На:
```python
        def _on_thinking(piece: str) -> None:
            chunk = str(piece or "")
            if chunk:
                events.put(("thinking", chunk), block=False)
```

И удалить неиспользуемый `_split_stream_display_piece()` из `api/app.py`.

---

### 2. Сделать корректную сборку delta/cumulative thinking
Файл: `llm/ollama_provider.py`

Заменить функцию `_stitch_thinking_delta` на:
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

---

### 3. Убрать ложный старт таймера thinking
Файл: `ui/chat_window.py`

Заменить `_on_thinking_chunk` на:
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

---

### 4. Исправить финальный расчёт thinking_ms
Файл: `ui/chat_window.py`

Заменить блок расчёта `local_thinking_ms` на:
```python
            if (
                self._pending.first_answer_at is not None
                and self._pending.first_thinking_at is not None
                and thinking_generated
            ):
                local_thinking_ms = max(
                    1,
                    int((self._pending.first_answer_at - self._pending.first_thinking_at) * 1000),
                )
                local_answer_ms = max(
                    1,
                    int((finished_at - self._pending.first_answer_at) * 1000),
                )
```

---

### 5. Если используешь `ui/chat_shell.py`, там те же 2 правки
- старт таймера от `first_thinking_at`
- финальный `thinking_ms` не от `started_at`

---

### 6. Что проверить после фикса
1. `thinking` кнопка появляется только после первого непустого thinking-chunk.
2. Счётчик thinking не стартует до первого thinking-куска.
3. Один chunk от backend -> один event в UI.
4. В логах нет повторной нарезки thinking по 12 символов.

---

## Важное
Если после этих правок thinking всё равно иногда приходит крупными кусками, значит это уже нативная гранулярность chunk-ов со стороны Ollama/модели, а не буферизация MMis. Тогда сделать thinking визуально "буквально как answer" можно только искусственной имитацией посимвольного/помельче стрима в UI.
