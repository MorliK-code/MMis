from __future__ import annotations

import io
import sys

from ui_console import _ConsoleChunkRenderer, _StreamRealtimePrinter


def test_hidden_thinking_finalize_prints_single_prefix_and_final_answer(monkeypatch) -> None:
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)

    printer = _StreamRealtimePrinter(
        _ConsoleChunkRenderer(),
        prefer_thinking_first=True,
        show_thinking=False,
        debug_memory=False,
    )
    printer._hidden_hint_active = True
    printer._current_channel = "thinking_hint"
    printer._hidden_hint_last_width = len("assistant> думает...")

    printer.finalize_with_final(answer_final="Привет")

    out = sink.getvalue()
    assert "assistant> Привет" in out
    assert "assistant> assistant>" not in out
    assert printer.rendered_answer() == "Привет"


def test_hidden_thinking_answer_chunks_do_not_duplicate_prefix(monkeypatch) -> None:
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)

    printer = _StreamRealtimePrinter(
        _ConsoleChunkRenderer(),
        prefer_thinking_first=True,
        show_thinking=False,
        debug_memory=False,
    )
    printer._hidden_hint_active = True
    printer._current_channel = "thinking_hint"
    printer._hidden_hint_last_width = len("assistant> думает...")

    printer.on_answer("При")
    printer.on_answer("вет")

    out = sink.getvalue()
    assert out.count("assistant> ") == 0
    assert printer.rendered_answer() == "Привет"


def test_hidden_thinking_stream_prefers_final_answer_over_partial_chunks(monkeypatch) -> None:
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)

    printer = _StreamRealtimePrinter(
        _ConsoleChunkRenderer(),
        prefer_thinking_first=True,
        show_thinking=False,
        debug_memory=False,
    )
    printer._hidden_hint_active = True
    printer._current_channel = "thinking_hint"
    printer._hidden_hint_last_width = len("assistant> думает...")

    printer.on_answer("Привет! Как дела?")
    printer.finalize_with_final(answer_final="Я затупила. Повтори, пожалуйста, еще раз.")

    out = sink.getvalue()
    assert "Привет! Как дела?" not in out
    assert out.count("assistant> ") == 1
    assert "Я затупила. Повтори, пожалуйста, еще раз." in out
    assert printer.rendered_answer() == "Я затупила. Повтори, пожалуйста, еще раз."
