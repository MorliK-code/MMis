from __future__ import annotations

import time

from ui.api_client import ApiReply
from ui.workers import ReplyWorker, _split_stream_display_piece


class _FakeApi:
    def __init__(self, answer_piece: str, thinking_piece: str = "") -> None:
        self.answer_piece = answer_piece
        self.thinking_piece = thinking_piece

    def stream_chat(
        self,
        *,
        text: str,
        store_turn: bool = True,
        think: bool | None = None,
        on_chunk=None,
        on_thinking_chunk=None,
        on_debug_event=None,
        cancel_requested=None,
        **_kwargs,
    ) -> ApiReply:
        if on_chunk:
            on_chunk(self.answer_piece)
        if on_thinking_chunk and self.thinking_piece:
            on_thinking_chunk(self.thinking_piece)
        return ApiReply(
            answer=self.answer_piece,
            thinking=self.thinking_piece,
            thinking_generated=bool(self.thinking_piece),
            stats={},
            model="fake-model",
        )


def test_split_stream_display_piece_breaks_large_sentence_into_small_parts() -> None:
    text = "Привет! 😊 Как ты? Рада, что ты снова заговорил."
    parts = _split_stream_display_piece(text, max_chars=12)
    assert len(parts) > 1
    assert "".join(parts) == text
    assert all(len(part) <= 12 for part in parts)


def test_reply_worker_emits_large_answer_piece_without_forced_ui_splitting() -> None:
    answer = "Привет! 😊 Как ты? Рада, что ты снова заговорил — чувствую, что наша беседа становится всё интереснее. "
    worker = ReplyWorker(_FakeApi(answer), user_text="hello", store_turn=True, think=False)
    worker._stream_emit_pause_sec = 0.0
    worker._stream_emit_chunk_chars = 12

    events: list[tuple[str, str]] = []
    worker.chunk.connect(lambda piece: events.append(("chunk", piece)))
    worker.finished.connect(lambda res: events.append(("finished", res.text)))

    worker.run()

    chunk_events = [value for kind, value in events if kind == "chunk"]
    assert len(chunk_events) == 1
    assert "".join(chunk_events) == answer
    assert events[-1] == ("finished", answer)
    assert chunk_events[0] == answer


def test_reply_worker_default_answer_stream_does_not_use_artificial_pause(monkeypatch) -> None:
    answer = "Привет! Как ты? Рада, что ты снова заговорил и продолжаешь разговор."
    worker = ReplyWorker(_FakeApi(answer), user_text="hello", store_turn=True, think=False)
    worker._stream_emit_chunk_chars = 12

    sleep_calls: list[float] = []
    real_sleep = time.sleep

    def _fake_sleep(value: float) -> None:
        sleep_calls.append(float(value))
        real_sleep(0)

    monkeypatch.setattr(time, "sleep", _fake_sleep)

    worker.run()

    assert sleep_calls == []


def test_reply_worker_passes_thinking_chunks_through_without_forced_ui_splitting() -> None:
    thinking = (
        "Okay, let's think this through carefully. "
        "The user wants the reasoning text to appear in smaller live chunks."
    )
    worker = ReplyWorker(_FakeApi(answer_piece="ok", thinking_piece=thinking), user_text="hello", store_turn=True, think=True)

    events: list[tuple[str, str]] = []
    worker.thinking_chunk.connect(lambda piece: events.append(("thinking", piece)))
    worker.finished.connect(lambda res: events.append(("finished", res.text)))

    worker.run()

    thinking_events = [value for kind, value in events if kind == "thinking"]
    assert len(thinking_events) == 1
    assert "".join(thinking_events) == thinking
