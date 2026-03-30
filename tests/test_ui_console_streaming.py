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


def test_visible_thinking_backfills_from_final_payload(monkeypatch) -> None:
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)

    printer = _StreamRealtimePrinter(
        _ConsoleChunkRenderer(),
        prefer_thinking_first=True,
        show_thinking=True,
        debug_memory=False,
    )

    printer.finalize_with_final(answer_final="Привет", thinking_final="Сначала подумаю")

    out = sink.getvalue()
    assert "thinking> Сначала подумаю" in out
    assert "assistant> Привет" in out
    assert printer.rendered_thinking() == "Сначала подумаю"
    assert printer.rendered_answer() == "Привет"


def test_visible_thinking_does_not_reprint_final_subset_after_answer(monkeypatch) -> None:
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)

    printer = _StreamRealtimePrinter(
        _ConsoleChunkRenderer(),
        prefer_thinking_first=True,
        show_thinking=True,
        debug_memory=False,
    )

    printer.on_thinking("first-pass ")
    printer.on_thinking("last-pass")
    printer.on_answer("answer")
    printer.finalize_with_final(answer_final="answer", thinking_final="last-pass")

    out = sink.getvalue()
    assert out.count("thinking> ") == 1
    assert out.count("last-pass") == 1
    assert out.count("assistant> answer") == 1
    assert printer.rendered_thinking() == "first-pass last-pass"
    assert printer.rendered_answer() == "answer"


def test_visible_answer_does_not_reprint_final_subset(monkeypatch) -> None:
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)

    printer = _StreamRealtimePrinter(
        _ConsoleChunkRenderer(),
        prefer_thinking_first=True,
        show_thinking=True,
        debug_memory=False,
    )

    printer.on_answer("draft ")
    printer.on_answer("final answer")
    printer.finalize_with_final(answer_final="final answer")

    out = sink.getvalue()
    assert out.count("assistant> ") == 1
    assert out.count("final answer") == 1
    assert printer.rendered_answer() == "draft final answer"


def test_hidden_thinking_uses_static_hint_when_stdout_is_not_tty(monkeypatch) -> None:
    sink = io.StringIO()
    monkeypatch.setattr(sys, "stdout", sink)

    printer = _StreamRealtimePrinter(
        _ConsoleChunkRenderer(),
        prefer_thinking_first=True,
        show_thinking=False,
        debug_memory=False,
    )

    printer.start_hidden_thinking_hint()

    out = sink.getvalue()
    assert "assistant> думает..." in out
    assert printer._hidden_hint_thread is None
