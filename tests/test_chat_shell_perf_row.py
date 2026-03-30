from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.chat_shell import MessageBubble


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_message_bubble_perf_row_keeps_all_stats_visible() -> None:
    app = _app()
    perf = ["23123 ms", "write 9437 ms", "7.2 tok/s", "prompt 2600", "gen 68"]
    bubble = MessageBubble("assistant", "text", "", "13552 ms", perf)
    bubble.show()
    app.processEvents()

    assert bubble.perf_wrap.layout().count() == 5
    assert bubble.perf_wrap.sizeHint().width() >= 300


def test_message_bubble_set_perf_recomputes_geometry() -> None:
    app = _app()
    perf = ["23123 ms", "write 9437 ms", "7.2 tok/s", "prompt 2600", "gen 68"]
    bubble = MessageBubble("assistant", "text", "", "13552 ms", [])
    bubble.set_perf(perf)
    bubble.show()
    app.processEvents()

    assert bubble.perf_wrap.layout().count() == 5
    assert bubble.perf_wrap.sizeHint().width() >= 300


def test_message_bubble_perf_row_stays_on_one_line_when_width_exactly_fits() -> None:
    app = _app()
    perf = ["19390 ms", "write 5242 ms", "7.1 tok/s", "prompt 2616", "gen 37"]
    bubble = MessageBubble("assistant", "text", "", "14012 ms", perf)
    bubble.show()
    app.processEvents()

    layout = bubble.perf_wrap.layout()
    ys = []
    for index in range(layout.count()):
        chip = layout.itemAt(index).widget()
        ys.append(chip.geometry().y())

    assert len(set(ys)) == 1


def test_message_bubble_does_not_create_extra_toplevel_perf_window() -> None:
    app = _app()
    before = list(app.topLevelWidgets())
    bubble = MessageBubble("assistant", "text", "", "14.9 s", ["2.1 s"])
    bubble.show()
    app.processEvents()

    after = list(app.topLevelWidgets())
    created = [widget for widget in after if widget not in before]

    assert created == [bubble]


def test_message_bubble_hides_thinking_header_without_reasoning() -> None:
    app = _app()
    bubble = MessageBubble("assistant", "text", "", "", [])
    bubble.show()
    app.processEvents()

    assert bubble.thinking_head is not None
    assert not bubble.thinking_head.isVisible()
    assert bubble.thinking_label is not None
    assert not bubble.thinking_label.isVisible()


def test_message_bubble_reveals_thinking_header_and_keeps_counter_near_label() -> None:
    app = _app()
    bubble = MessageBubble("assistant", "text", "", "", [])
    bubble.update_thinking("reasoning", "14.9 s")
    bubble.show()
    app.processEvents()

    assert bubble.thinking_head is not None
    assert bubble.thinking_head.isVisible()
    assert bubble.thinking_ms_chip is not None
    layout = bubble.thinking_head.layout()
    assert layout is not None
    assert layout.itemAt(0).widget() is bubble.thinking_toggle
    assert layout.itemAt(1).widget() is bubble.thinking_ms_chip
    assert abs(bubble.thinking_ms_chip.geometry().y() - bubble.thinking_toggle.geometry().y()) <= 1
    assert bubble.thinking_ms_chip.geometry().x() > bubble.thinking_toggle.geometry().x()
    assert bubble.thinking_head.width() <= bubble.thinking_toggle.width() + bubble.thinking_ms_chip.width() + 8


def test_message_bubble_keeps_thinking_hidden_when_only_unconfirmed_timer_exists() -> None:
    app = _app()
    bubble = MessageBubble("assistant", "text", "", "", [])
    bubble.update_thinking("", "14.9 s")
    bubble.show()
    app.processEvents()

    assert bubble.thinking_head is not None
    assert not bubble.thinking_head.isVisible()


def test_message_bubble_hides_thinking_header_when_only_timer_exists() -> None:
    app = _app()
    bubble = MessageBubble("assistant", "text", "", "14.9 s", [], show_thinking_header=True)
    bubble.show()
    app.processEvents()

    assert bubble.thinking_head is not None
    assert not bubble.thinking_head.isVisible()


def test_message_bubble_toggle_shows_inline_reasoning() -> None:
    app = _app()
    bubble = MessageBubble("assistant", "answer text", "reasoning block", "14.9 s", [])
    bubble.show()
    app.processEvents()

    assert bubble.thinking_toggle is not None
    bubble.thinking_toggle.setChecked(True)
    app.processEvents()

    assert bubble.thinking_label is not None
    assert bubble.thinking_label.isVisible()
