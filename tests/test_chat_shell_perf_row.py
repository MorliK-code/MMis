from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent, QTextCursor
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from ui.chat_shell import ComposerEdit, ExactChatWindow, MessageBubble, PendingAssistant, _format_message_html


class _FakeBubble:
    def __init__(self) -> None:
        self.text_updates: list[str] = []
        self.thinking_updates: list[tuple[str, str | None]] = []
        self.perf_updates: list[list[str]] = []

    def update_text(self, text: str) -> None:
        self.text_updates.append(text)

    def update_thinking(self, text: str, ms: str | None = None) -> None:
        self.thinking_updates.append((text, ms))

    def set_perf(self, perf: list[str]) -> None:
        self.perf_updates.append(list(perf))


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_composer_enter_requests_submit_without_inserting_newline() -> None:
    app = _app()
    edit = ComposerEdit()
    submitted: list[bool] = []
    edit.submitRequested.connect(lambda: submitted.append(True))
    edit.setPlainText("hello")
    edit.show()
    app.processEvents()

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    edit.keyPressEvent(event)

    assert submitted == [True]
    assert edit.toPlainText() == "hello"
    assert event.isAccepted()


def test_composer_ctrl_enter_inserts_newline_without_submit() -> None:
    app = _app()
    edit = ComposerEdit()
    submitted: list[bool] = []
    edit.submitRequested.connect(lambda: submitted.append(True))
    edit.setPlainText("hello")
    edit.moveCursor(QTextCursor.MoveOperation.End)
    edit.show()
    app.processEvents()

    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    edit.keyPressEvent(event)

    assert submitted == []
    assert edit.toPlainText() == "hello\n"
    assert event.isAccepted()


def test_composer_overlay_text_origin_matches_native_selection_origin() -> None:
    app = _app()
    edit = ComposerEdit()
    edit.resize(400, 76)
    edit.setPlainText("asdfasdfasdf")
    edit.show()
    app.processEvents()

    cursor = QTextCursor(edit.document())
    cursor.movePosition(QTextCursor.MoveOperation.Start)

    assert edit._overlay_text_rect().topLeft() == edit.cursorRect(cursor).topLeft()


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


def test_message_text_formats_bold_and_inline_code_as_rich_text() -> None:
    html = _format_message_html("1. **Duplication** text `visible layer` (plain note)")

    assert "<b>Duplication</b>" in html
    assert "**" not in html
    assert "`" not in html
    assert "background-color" in html
    assert "Cascadia Code" in html
    assert "font-size:12px" in html
    assert "visible layer" in html
    assert "(plain note)" in html


def test_message_text_does_not_highlight_plain_parentheses() -> None:
    html = _format_message_html("text (plain note)")

    assert "background-color" not in html
    assert "(plain note)" in html


def test_message_bubble_keeps_plain_text_value_while_rendering_inline_markup() -> None:
    app = _app()
    text = "**Problem**: answer `detail` (plain note)"
    bubble = MessageBubble("assistant", text, "", "", [])
    bubble.show()
    app.processEvents()

    assert bubble.text_label.text() == text
    assert "<b>Problem</b>" in bubble.text_label.rendered_html()
    assert "background-color" in bubble.text_label.rendered_html()
    assert "`" not in bubble.text_label.rendered_html()
    assert "(plain note)" in bubble.text_label.rendered_html()


def test_user_message_regenerate_button_is_hover_revealed_and_emits_request() -> None:
    app = _app()
    bubble = MessageBubble("user", "question", "", "", [])
    requested: list[object] = []
    bubble.regenerateRequested.connect(lambda item: requested.append(item))
    bubble.show()
    app.processEvents()

    assert bubble.regenerate_btn is not None
    bubble._set_regenerate_button_visible(False)
    app.processEvents()
    assert not bubble.regenerate_btn.isVisible()

    bubble._set_regenerate_button_visible(True)
    app.processEvents()

    assert bubble.regenerate_btn.isVisible()
    bubble.regenerate_btn.click()

    assert requested == [bubble]

    bubble._set_regenerate_button_visible(False)
    app.processEvents()

    assert not bubble.regenerate_btn.isVisible()


def test_short_messages_keep_content_height_when_chat_has_free_space() -> None:
    app = _app()
    window = ExactChatWindow()
    window.resize(1120, 520)
    window.show()
    app.processEvents()
    window._clear_messages()

    bubbles = [
        window._append_message("user", "short"),
        window._append_message("assistant", "short answer"),
        window._append_message("user", "another short"),
    ]
    app.processEvents()

    for bubble in bubbles:
        panel = bubble._message_panel
        assert bubble.height() <= bubble.sizeHint().height() + 2
        assert panel.height() <= panel.sizeHint().height() + 2

    content_height = window.messages_layout.sizeHint().height()
    assert abs(window.scroll.viewport().height() - content_height) <= 2
    assert abs(window.messages_host.height() - content_height) <= 2
    bottom_gap = window.messages_host.height() - bubbles[-1].geometry().bottom()
    assert bottom_gap <= window.messages_layout.contentsMargins().bottom() + 2


def test_message_bubbles_resist_extra_vertical_layout_space() -> None:
    app = _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    bubbles = [
        MessageBubble("user", "short question"),
        MessageBubble("assistant", "short answer"),
        MessageBubble("user", "another short question"),
    ]
    for bubble in bubbles:
        layout.addWidget(bubble)

    host.resize(900, 500)
    host.show()
    app.processEvents()

    for bubble in bubbles:
        panel = bubble._message_panel
        assert bubble.height() <= bubble.sizeHint().height() + 2
        assert panel.height() <= panel.sizeHint().height() + 2


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
    assert bubble.thinking_label.text() == "reasoning block"


def test_message_bubble_thinking_body_keeps_subtle_panel_style() -> None:
    app = _app()
    bubble = MessageBubble("assistant", "answer text", "reasoning block", "14.9 s", [])
    bubble.show()
    app.processEvents()

    assert bubble.thinking_toggle is not None
    bubble.thinking_toggle.setChecked(True)
    app.processEvents()

    assert bubble.thinking_label is not None
    assert bubble.thinking_label._background_color.alpha() > 0
    assert bubble.thinking_label._border_color.alpha() > 0
    assert bubble.thinking_label._border_style == Qt.PenStyle.DashLine


def test_message_bubble_defers_hidden_thinking_body_text_until_toggle() -> None:
    app = _app()
    bubble = MessageBubble("assistant", "answer text", "", "", [])
    long_thinking = "thinking " * 2000

    bubble.update_thinking(long_thinking, "1.2 s")
    bubble.show()
    app.processEvents()

    assert bubble.thinking_label is not None
    assert not bubble.thinking_label.isVisible()
    assert bubble.thinking_label.text() == ""

    assert bubble.thinking_toggle is not None
    bubble.thinking_toggle.setChecked(True)
    app.processEvents()

    assert bubble.thinking_label.isVisible()
    assert bubble.thinking_label.text() == long_thinking
    assert bubble.thinking_label.height() > 100


def test_message_bubble_appends_visible_thinking_delta_instead_of_resetting_body(monkeypatch) -> None:
    app = _app()
    bubble = MessageBubble("assistant", "answer text", "", "", [])
    bubble.update_thinking("first", "0.1 s")

    assert bubble.thinking_toggle is not None
    bubble.thinking_toggle.setChecked(True)
    app.processEvents()

    assert bubble.thinking_label is not None
    original_append = bubble.thinking_label.append_stream_text
    appended: list[str] = []

    def _append_spy(text: str) -> None:
        appended.append(text)
        original_append(text)

    monkeypatch.setattr(bubble.thinking_label, "append_stream_text", _append_spy)
    bubble.update_thinking("first second", "0.2 s")
    app.processEvents()

    assert appended == [" second"]
    assert bubble.thinking_label.text() == "first second"


def test_message_bubble_expanded_long_thinking_height_fits_full_text() -> None:
    app = _app()
    long_thinking = "thinking text wraps across many lines " * 300
    bubble = MessageBubble("assistant", "answer text", long_thinking, "1.2 s", [])
    bubble.show()
    app.processEvents()

    assert bubble.thinking_toggle is not None
    bubble.thinking_toggle.setChecked(True)
    app.processEvents()

    assert bubble.thinking_label is not None
    assert bubble.thinking_label.text() == long_thinking
    assert bubble.thinking_label.height() > 300


def test_message_bubble_thinking_body_aligns_to_message_right_edge() -> None:
    app = _app()
    bubble = MessageBubble(
        "assistant",
        "answer text that keeps the assistant message body at its normal width",
        "reasoning block",
        "1.2 s",
        [],
    )
    bubble.show()
    app.processEvents()

    assert bubble.thinking_toggle is not None
    bubble.thinking_toggle.setChecked(True)
    app.processEvents()

    assert bubble.thinking_label is not None
    panel = bubble.thinking_label.parentWidget()
    left_gap = bubble.thinking_label.geometry().x()
    right_gap = panel.width() - bubble.thinking_label.geometry().x() - bubble.thinking_label.width()

    assert abs(left_gap - right_gap) <= 1
    assert abs(bubble.thinking_label.width() - bubble.text_label.width()) <= 1


def test_message_bubble_thinking_width_stays_stable_as_wrapped_text_grows() -> None:
    app = _app()
    window = ExactChatWindow()
    window.resize(900, 600)
    window.show()
    app.processEvents()
    for index in range(window.messages_layout.count() - 1, -1, -1):
        item = window.messages_layout.itemAt(index)
        widget = item.widget() if item is not None else None
        if widget is None:
            continue
        window.messages_layout.takeAt(index)
        widget.deleteLater()

    bubble = MessageBubble("assistant", "short answer", "", "", [])
    window.messages_layout.addWidget(bubble)
    app.processEvents()

    bubble.update_thinking("short", "0.1 s")
    assert bubble.thinking_toggle is not None
    bubble.thinking_toggle.setChecked(True)
    app.processEvents()

    assert bubble.thinking_label is not None
    initial_panel_width = bubble.thinking_label.parentWidget().width()
    initial_thinking_width = bubble.thinking_label.width()

    heights: list[int] = []
    for index in range(1, 8):
        bubble.update_thinking("word " * (40 * index), f"{index}.0 s")
        app.processEvents()
        heights.append(bubble.thinking_label.height())
        assert bubble.thinking_label.parentWidget().width() == initial_panel_width
        assert bubble.thinking_label.width() == initial_thinking_width

    assert heights[-1] > heights[0]


def test_message_bubble_reflows_answer_below_growing_visible_thinking() -> None:
    app = _app()
    bubble = MessageBubble("assistant", "answer text", "", "", [])
    bubble.resize(760, 100)
    bubble.show()
    app.processEvents()

    bubble.update_thinking("short", "0.1 s")
    assert bubble.thinking_toggle is not None
    bubble.thinking_toggle.setChecked(True)
    app.processEvents()

    long_thinking = "thinking text wraps across many lines and should push answer down. " * 150
    bubble.update_thinking(long_thinking, "2.0 s")
    app.processEvents()

    assert bubble.thinking_label is not None
    assert bubble.text_label.geometry().top() >= bubble.thinking_label.geometry().bottom()


def test_message_bubble_reflows_when_answer_grows_after_large_visible_thinking() -> None:
    app = _app()
    window = ExactChatWindow()
    window.resize(640, 480)
    window.show()
    app.processEvents()
    for index in range(window.messages_layout.count() - 1, -1, -1):
        item = window.messages_layout.itemAt(index)
        widget = item.widget() if item is not None else None
        if widget is None:
            continue
        window.messages_layout.takeAt(index)
        widget.deleteLater()

    bubble = MessageBubble("assistant", "", "", "", [])
    window.messages_layout.addWidget(bubble)
    app.processEvents()

    bubble.update_thinking("short", "0.1 s")
    assert bubble.thinking_toggle is not None
    bubble.thinking_toggle.setChecked(True)
    app.processEvents()

    long_thinking = "thinking text wraps across many lines and should push answer down. " * 120
    bubble.update_thinking(long_thinking, "2.0 s")
    app.processEvents()

    long_answer = "Now the answer grows too. " * 80
    bubble.update_text(long_answer)
    app.processEvents()

    assert bubble.thinking_label is not None
    assert not bubble.thinking_label.geometry().intersects(bubble.text_label.geometry())
    assert bubble.thinking_label.parentWidget().height() >= bubble.thinking_label.parentWidget().layout().sizeHint().height()
    panel = bubble.thinking_label.parentWidget()
    left_gap = bubble.thinking_label.geometry().x()
    right_gap = panel.width() - bubble.thinking_label.geometry().x() - bubble.thinking_label.width()
    assert abs(left_gap - right_gap) <= 1


def test_exact_chat_window_final_thinking_ms_uses_visible_thinking_span() -> None:
    window = ExactChatWindow.__new__(ExactChatWindow)
    bubble = _FakeBubble()
    window._pending = PendingAssistant(
        bubble=bubble,
        started_at=100.0,
        thinking_text="thinking",
        answer_text="answer",
        first_thinking_at=130.0,
        first_answer_at=150.0,
    )
    window._scroll_bottom = lambda: None

    class _Result:
        text = "answer"
        thinking = "thinking"
        stats = {"elapsed_ms": 70000}

    window._on_reply_finished(_Result())

    assert bubble.thinking_updates[-1] == ("thinking", "20000 ms")
