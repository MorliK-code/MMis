from __future__ import annotations

import os
from types import MethodType

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

import ui.chat_shell as proto
import ui.chat_window as chat_window_module
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
        "_remove_message_widgets_after",
        "_history_index_for_user_bubble",
        "_assistant_bubble_after_user",
        "_start_reply_worker_for_text",
        "_on_regenerate_requested",
        "_insert_message_bubble_after",
    ):
        setattr(window, name, MethodType(getattr(ChatWindow, name), window))
    for name in (
        "_wire_regenerate_bubble",
        "_message_layout_index",
        "_remove_message_bubble",
    ):
        setattr(window, name, MethodType(getattr(proto.ExactChatWindow, name), window))
    for name in (
        "_split_stat_line",
        "_extract_thinking_ms_from_stat_line",
        "_normalize_stat_line_time_units",
        "_attachment_payloads_from_value",
    ):
        setattr(window, name, getattr(ChatWindow, name))
    return window


class _FakeSignal:
    def __init__(self) -> None:
        self.connected: list[object] = []

    def connect(self, callback) -> None:
        self.connected.append(callback)


class _FakeReplyWorker:
    instances: list["_FakeReplyWorker"] = []

    def __init__(self, api, user_text: str, store_turn: bool = True, think=None, verbose=None, attachments=None) -> None:
        self.api = api
        self.user_text = user_text
        self.store_turn = store_turn
        self.think = think
        self.verbose = verbose
        self.attachments = list(attachments or [])
        self.chunk = _FakeSignal()
        self.thinking_chunk = _FakeSignal()
        self.debug_event = _FakeSignal()
        self.finished = _FakeSignal()
        self.errored = _FakeSignal()
        self.started = False
        self.deleted = False
        self.instances.append(self)

    def isRunning(self) -> bool:
        return False

    def start(self) -> None:
        self.started = True

    def deleteLater(self) -> None:
        self.deleted = True


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


def test_chat_window_regenerate_removes_old_answer_and_does_not_store_duplicate_user(monkeypatch) -> None:
    app = _app()
    window = _lazy_window()
    window._history = [
        ("user", "question", None, None, None),
        ("ai", "old answer", "1.0 s", None, None),
    ]
    window.api = object()
    window._worker = None
    window._pending = None
    window._pending_history_index = None
    window._pending_user_text = ""
    window._stream_follow_scroll = False
    window._force_stream_follow_scroll = False
    window._thinking_enabled = True
    window._verbose_enabled = False
    window._capture_stream_scroll_mode = lambda: None
    window._schedule_scroll_bottom = lambda **_kwargs: None
    window._set_busy_state = lambda _busy: None
    window._on_answer_chunk = lambda _piece: None
    window._on_thinking_chunk = lambda _piece: None
    window._on_memory_debug_event = lambda _payload: None
    window._on_reply_finished = lambda _result: None
    window._on_reply_error = lambda _error: None
    window._cleanup_request = lambda: None
    saved: list[list[tuple]] = []
    window._save_chat_sessions = lambda: saved.append(list(window._history))

    user_bubble = proto.MessageBubble("user", "question", "", "", [])
    user_bubble.setProperty("history_index", 0)
    assistant_bubble = proto.MessageBubble("assistant", "old answer", "", "", [])
    window._insert_message_bubble(user_bubble)
    window._insert_message_bubble(assistant_bubble)
    app.processEvents()

    _FakeReplyWorker.instances = []
    monkeypatch.setattr(chat_window_module, "ReplyWorker", _FakeReplyWorker)

    window._on_regenerate_requested(user_bubble)
    app.processEvents()

    assert len(_FakeReplyWorker.instances) == 1
    worker = _FakeReplyWorker.instances[0]
    assert worker.user_text == "question"
    assert worker.store_turn is False
    assert worker.started
    assert window._pending_user_text == "question"
    assert window._pending_history_index == 1
    assert window._force_stream_follow_scroll is True
    assert window._stream_follow_scroll is True
    assert window._history == [
        ("user", "question", None, None, None),
        ("ai", "", "…", None, None),
    ]
    assert window.messages_layout.count() == 2
    assert window.messages_layout.itemAt(0).widget() is user_bubble
    assert window.messages_layout.itemAt(1).widget() is assistant_bubble
    assert saved[-1] == window._history
