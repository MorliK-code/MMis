from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from ui.app import MainWindow


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_stream_placeholder_and_first_chunk_update_immediately() -> None:
    _app()
    window = MainWindow()
    try:
        window._history = [("ai", "", "...", None, None)]
        window._stream_ai_index = 0
        window._render_chat(scroll_to_bottom=False)

        window._update_stream_waiting_visual()
        row = window._message_row_widgets.get(0)
        assert row is not None
        bubble = getattr(row, "_bubble_label", None)
        assert isinstance(bubble, QLabel)
        assert bubble.text().startswith(("Печатаю", "Думаю"))

        window._on_reply_chunk("При")
        assert bubble.text() == "При"

        window._on_reply_chunk("вет")
        assert bubble.text() == "Привет"
    finally:
        window.close()
