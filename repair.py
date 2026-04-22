import sys

with open('ui/chat_window.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = 'self._start_reply_worker_for_text(text, store_turn=store_turn, attachments=attachments)'
end_marker = '@Slot(object)\n    def _on_memory_debug_event'

idx1 = content.find(start_marker)
if idx1 == -1: print('Start not found'); sys.exit(1)
idx1 += len(start_marker)

idx2 = content.find(end_marker)
if idx2 == -1: print('End not found'); sys.exit(1)

new_code = """

    @Slot(str)
    def _on_answer_chunk(self, piece: str) -> None:
        if not self._pending:
            return
        now = time.perf_counter()
        if self._pending.first_answer_at is None:
            self._pending.first_answer_at = now
        self._pending.last_chunk_at = now
        self._pending.answer_text += piece or ""
        self._pending.bubble.update_text(self._pending.answer_text)
        if not self._pending.bubble.isVisible():
            self._pending.bubble.show()
        self._schedule_scroll_bottom(follow_stream_only=True)

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
        if not self._pending.bubble.isVisible():
            self._pending.bubble.show()
        self._schedule_scroll_bottom(follow_stream_only=True)

    """

new_content = content[:idx1] + new_code + content[idx2:]

with open('ui/chat_window.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('Success')
