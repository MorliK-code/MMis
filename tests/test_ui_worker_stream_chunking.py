from __future__ import annotations

import time

from ui.api_client import ApiReply
from ui.workers import ReplyWorker


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


def test_reply_worker_emits_large_answer_piece_without_forced_ui_splitting() -> None:
    answer = (
        "Hello! This answer is intentionally long enough to prove the worker "
        "passes backend chunks through without splitting them for display."
    )
    worker = ReplyWorker(_FakeApi(answer), user_text="hello", store_turn=True, think=False)

    events: list[tuple[str, str]] = []
    worker.chunk.connect(lambda piece: events.append(("chunk", piece)))
    worker.finished.connect(lambda res: events.append(("finished", res.text)))

    worker.run()

    chunk_events = [value for kind, value in events if kind == "chunk"]
    assert chunk_events == [answer]
    assert events[-1] == ("finished", answer)


def test_reply_worker_stream_does_not_use_artificial_pause(monkeypatch) -> None:
    answer = (
        "Hello! This answer is intentionally long enough to catch any old "
        "forced streaming pause logic if it comes back."
    )
    worker = ReplyWorker(_FakeApi(answer), user_text="hello", store_turn=True, think=False)

    sleep_calls: list[float] = []
    real_sleep = time.sleep

    def _fake_sleep(value: float) -> None:
        sleep_calls.append(float(value))
        real_sleep(0)

    monkeypatch.setattr(time, "sleep", _fake_sleep)

    worker.run()

    assert sleep_calls == []


def test_reply_worker_passes_thinking_chunks_like_answer_chunks() -> None:
    thinking = (
        "Okay, let's think this through carefully. "
        "The worker should pass this backend chunk through unchanged."
    )
    worker = ReplyWorker(_FakeApi(answer_piece="ok", thinking_piece=thinking), user_text="hello", store_turn=True, think=True)

    events: list[tuple[str, str]] = []
    worker.thinking_chunk.connect(lambda piece: events.append(("thinking", piece)))
    worker.finished.connect(lambda res: events.append(("finished", res.text)))

    worker.run()

    thinking_events = [value for kind, value in events if kind == "thinking"]
    assert thinking_events == [thinking]
