from __future__ import annotations

import os
from types import MethodType

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from ui.chat_window import ChatWindow, HISTORY_INITIAL_RENDER_LIMIT, HISTORY_LAZY_BATCH_SIZE


class _LazyHistoryHarness:
    pass


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _lazy_window() -> ChatWindow:
    window = _LazyHistoryHarness()
    window._history = []
    window._lazy_history_start_index = 0
    window._lazy_history_button = None
    window._lazy_history_loading = False
    window._verbose_enabled = False
    window.messages_host = QWidget()
    window.messages_layout = QVBoxLayout(window.messages_host)
    window.messages_layout.setContentsMargins(0, 0, 0, 0)
    window.messages_layout.setSpacing(0)
    window._schedule_messages_view_height_sync = lambda: None
    window._sync_messages_view_height = lambda: None
    window._scroll_bottom = lambda: None
    window._apply_context_chips = lambda: None
    window._save_chat_sessions = lambda: None
    for name in (
        "_clear_message_widgets",
        "_insert_message_bubble",
        "_last_user_text_before",
        "_history_bubble_for_index",
        "_render_history_range",
        "_remove_lazy_history_button",
        "_sync_lazy_history_button",
        "_load_older_history_batch",
        "_render_history",
        "_upgrade_legacy_verbose_stat_line",
    ):
        setattr(window, name, MethodType(getattr(ChatWindow, name), window))
    for name in (
        "_split_stat_line",
        "_extract_thinking_ms_from_stat_line",
        "_normalize_stat_line_time_units",
    ):
        setattr(window, name, getattr(ChatWindow, name))
    return window


def _message_bubble_count(window: ChatWindow) -> int:
    count = 0
    for index in range(window.messages_layout.count()):
        item = window.messages_layout.itemAt(index)
        widget = item.widget() if item is not None else None
        if widget is not None and widget.objectName() == "message_bubble_host":
            count += 1
    return count


def test_chat_window_renders_recent_history_first_and_loads_older_batches() -> None:
    app = _app()
    window = _lazy_window()
    window._history = [
        ("user" if index % 2 == 0 else "ai", f"message {index}", None, None, None)
        for index in range(30)
    ]

    window._render_history()
    app.processEvents()

    assert _message_bubble_count(window) == HISTORY_INITIAL_RENDER_LIMIT
    assert window._lazy_history_start_index == 30 - HISTORY_INITIAL_RENDER_LIMIT
    assert window._lazy_history_button is not None

    window._load_older_history_batch()
    app.processEvents()

    assert _message_bubble_count(window) == HISTORY_INITIAL_RENDER_LIMIT + HISTORY_LAZY_BATCH_SIZE
    assert window._lazy_history_start_index == 30 - HISTORY_INITIAL_RENDER_LIMIT - HISTORY_LAZY_BATCH_SIZE
    assert window._lazy_history_button is not None

    window._load_older_history_batch()
    app.processEvents()

    assert _message_bubble_count(window) == len(window._history)
    assert window._lazy_history_start_index == 0
    assert window._lazy_history_button is None

    window._clear_message_widgets()
    window.messages_host.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
