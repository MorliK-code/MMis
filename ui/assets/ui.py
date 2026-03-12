# ui_mock.py
# PySide6 UI mock: Sidebar | Chat | Inspector (Memory/Metadata/Tools/Logs)
# Run: python ui_mock.py

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, List

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QFrame,
    QTabWidget,
    QTextEdit,
    QSizePolicy,
)


# ----------------------------
# Data
# ----------------------------
@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system" | "tool"
    text: str
    ts: float = time.time()


# ----------------------------
# Fake API that can "stream"
# ----------------------------
class FakeApi(QObject):
    token = Signal(str)         # streaming token
    done = Signal(dict)         # final result (including metadata, memory items, etc.)
    error = Signal(str)

    def send(self, user_text: str, streaming: bool = True) -> None:
        """
        Replace this with your real API client.
        For streaming: emit token(...) many times, then done({...}).
        """
        reply = (
            "����  � ������. "
            "Сейчас я сгенерирую ответ и параллельно отмечу метаданные: intent, emotion, topic. "
            "Если хочешь — могу вынести факты в память."
        )

        meta = {
            "intent": "chat",
            "emotion": "playful",
            "lang": "ru",
            "topic": "assistant_ui",
            "timings_ms": {"metadata": 18, "retrieval": 0, "llm": 240, "tts": 0},
        }
        memory_candidates = [
            {"type": "preference", "key": "ui_stack", "value": "PySide6", "confidence": 0.74},
            {"type": "project", "key": "architecture", "value": "API-driven UI", "confidence": 0.81},
        ]
        tool_requests = [
            {"name": "none", "args": {}, "status": "skipped"}
        ]

        if not streaming:
            self.done.emit({"text": reply, "meta": meta, "memory": memory_candidates, "tools": tool_requests})
            return

        tokens = reply.split(" ")
        self._i = 0

        def tick():
            if self._i >= len(tokens):
                self._timer.stop()
                self.done.emit({"text": reply, "meta": meta, "memory": memory_candidates, "tools": tool_requests})
                return
            self.token.emit(tokens[self._i] + (" " if self._i < len(tokens) - 1 else ""))
            self._i += 1

        self._timer = QTimer(self)
        self._timer.timeout.connect(tick)
        self._timer.start(45)  # speed of "stream"


# ----------------------------
# UI Widgets
# ----------------------------
class MessageBubble(QFrame):
    def __init__(self, role: str, text: str) -> None:
        super().__init__()
        self.setObjectName("bubble")
        self.role = role

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        header = QLabel("You" if role == "user" else ("MMis" if role == "assistant" else role))
        header.setObjectName("bubbleHeader")

        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setObjectName("bubbleText")

        root.addWidget(header)
        root.addWidget(body)

        self._header = header
        self._body = body

        if role == "user":
            self.setProperty("bubbleRole", "user")
        elif role == "assistant":
            self.setProperty("bubbleRole", "assistant")
        elif role == "tool":
            self.setProperty("bubbleRole", "tool")
        else:
            self.setProperty("bubbleRole", "system")

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def append_text(self, more: str) -> None:
        self._body.setText(self._body.text() + more)


class ChatView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._bubbles: List[MessageBubble] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.inner = QWidget()
        self.vbox = QVBoxLayout(self.inner)
        self.vbox.setContentsMargins(14, 14, 14, 14)
        self.vbox.setSpacing(10)
        self.vbox.addStretch(1)

        self.scroll.setWidget(self.inner)
        layout.addWidget(self.scroll)

    def add_message(self, msg: ChatMessage) -> MessageBubble:
        bubble = MessageBubble(msg.role, msg.text)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        if msg.role == "user":
            row.addStretch(1)
            row.addWidget(bubble, 0)
        else:
            row.addWidget(bubble, 0)
            row.addStretch(1)

        # insert before stretch
        self.vbox.insertLayout(self.vbox.count() - 1, row)
        self._bubbles.append(bubble)
        self._scroll_to_bottom()
        return bubble

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def last_bubble(self) -> Optional[MessageBubble]:
        return self._bubbles[-1] if self._bubbles else None


class Sidebar(QWidget):
    chat_selected = Signal(str)
    new_chat = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("sidebar")
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        top = QHBoxLayout()
        title = QLabel("Chats")
        title.setObjectName("sidebarTitle")

        btn_new = QPushButton("New")
        btn_new.setObjectName("primaryBtn")
        btn_new.clicked.connect(self.new_chat.emit)

        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(btn_new)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search…")
        self.search.setObjectName("searchBox")

        self.list = QListWidget()
        self.list.setObjectName("chatList")
        self.list.itemClicked.connect(lambda it: self.chat_selected.emit(it.text()))

        # demo chats
        for name in ["Main", "Dev", "Voice", "Ideas"]:
            self.list.addItem(QListWidgetItem(name))
        self.list.setCurrentRow(0)

        self.model_badge = QLabel("mistral:7b • FAST")
        self.model_badge.setObjectName("badge")

        root.addLayout(top)
        root.addWidget(self.search)
        root.addWidget(self.list, 1)
        root.addWidget(self.model_badge)


class Inspector(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("inspector")
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QLabel("Inspector")
        header.setObjectName("rightTitle")

        self.tabs = QTabWidget()
        self.tabs.setObjectName("tabs")

        self.tab_memory = QTextEdit(); self.tab_memory.setReadOnly(True)
        self.tab_meta = QTextEdit(); self.tab_meta.setReadOnly(True)
        self.tab_tools = QTextEdit(); self.tab_tools.setReadOnly(True)
        self.tab_logs = QTextEdit(); self.tab_logs.setReadOnly(True)

        self.tabs.addTab(self.tab_memory, "Memory")
        self.tabs.addTab(self.tab_meta, "Metadata")
        self.tabs.addTab(self.tab_tools, "Tools")
        self.tabs.addTab(self.tab_logs, "Logs")

        root.addWidget(header)
        root.addWidget(self.tabs, 1)

    def set_meta(self, meta: dict) -> None:
        lines = []
        for k, v in meta.items():
            lines.append(f"{k}: {v}")
        self.tab_meta.setPlainText("\n".join(lines))

    def set_memory(self, items: list) -> None:
        out = []
        for it in items:
            out.append(f"- {it['type']} | {it['key']} = {it['value']}  (conf={it['confidence']})")
        self.tab_memory.setPlainText("\n".join(out) if out else "—")

    def set_tools(self, items: list) -> None:
        out = []
        for it in items:
            out.append(f"- {it['name']} | status={it.get('status','?')} | args={it.get('args',{})}")
        self.tab_tools.setPlainText("\n".join(out) if out else "—")

    def log(self, text: str) -> None:
        self.tab_logs.append(text)


class Composer(QWidget):
    send_clicked = Signal(str)
    mic_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("composer")
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Write a message…")
        self.input.setObjectName("composerInput")
        self.input.setFixedHeight(74)

        right = QVBoxLayout()
        right.setSpacing(8)

        self.btn_send = QPushButton("Send")
        self.btn_send.setObjectName("primaryBtn")
        self.btn_send.clicked.connect(self._on_send)

        self.btn_mic = QPushButton("Mic")
        self.btn_mic.setObjectName("ghostBtn")
        self.btn_mic.clicked.connect(self.mic_clicked.emit)

        right.addWidget(self.btn_send)
        right.addWidget(self.btn_mic)
        right.addStretch(1)

        root.addWidget(self.input, 1)
        root.addLayout(right)

    def _on_send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self.send_clicked.emit(text)


# ----------------------------
# Main Window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MMis UI (mock)")
        self.resize(1280, 760)

        self.api = FakeApi()
        self.api.token.connect(self._on_token)
        self.api.done.connect(self._on_done)
        self.api.error.connect(self._on_error)

        self.sidebar = Sidebar()
        self.chat = ChatView()
        self.inspector = Inspector()
        self.composer = Composer()

        self.sidebar.new_chat.connect(self._on_new_chat)
        self.sidebar.chat_selected.connect(self._on_chat_selected)
        self.composer.send_clicked.connect(self._send_message)
        self.composer.mic_clicked.connect(lambda: self.inspector.log("[ui] mic clicked"))

        # Center area: chat + composer
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        topbar = QWidget()
        topbar.setObjectName("topbar")
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(12, 10, 12, 10)
        self.title = QLabel("Main")
        self.title.setObjectName("chatTitle")
        self.status = QLabel("idle")
        self.status.setObjectName("statusPill")
        tb.addWidget(self.title)
        tb.addStretch(1)
        tb.addWidget(self.status)

        center_layout.addWidget(topbar)
        center_layout.addWidget(self.chat, 1)
        center_layout.addWidget(self.composer, 0)

        # Splitters
        root_split = QSplitter(Qt.Horizontal)
        root_split.addWidget(self.sidebar)
        root_split.addWidget(center)
        root_split.addWidget(self.inspector)
        root_split.setStretchFactor(0, 0)
        root_split.setStretchFactor(1, 1)
        root_split.setStretchFactor(2, 0)
        root_split.setSizes([260, 740, 320])

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(root_split)
        self.setCentralWidget(container)

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+L"), self, activated=lambda: self.composer.input.setFocus())
        QShortcut(QKeySequence("Ctrl+K"), self, activated=lambda: self.sidebar.search.setFocus())

        self._stream_bubble: Optional[MessageBubble] = None

        # Demo welcome
        self.chat.add_message(ChatMessage("assistant", "������  ��� ��� ����������. ������ ���-������ �����."))

        self._apply_qss()

    def _apply_qss(self) -> None:
        # Simple QSS theme. Replace with your theme.qss / style.css approach.
        self.setStyleSheet("""
        QMainWindow { background: #0f1115; }
        QLabel { color: #e7e7e7; font-size: 13px; }
        #sidebar, #inspector { background: #121521; }
        #topbar { background: #10131a; border-bottom: 1px solid rgba(255,255,255,0.06); }
        #chatTitle { font-size: 16px; font-weight: 600; }
        #statusPill {
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: #d7d7d7;
        }

        #searchBox {
            padding: 8px 10px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            color: #e7e7e7;
        }

        #chatList {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            padding: 6px;
        }
        QListWidget::item { padding: 10px 10px; border-radius: 10px; }
        QListWidget::item:selected { background: rgba(255,255,255,0.10); }

        #badge {
            padding: 7px 10px;
            border-radius: 10px;
            background: rgba(120,180,255,0.12);
            border: 1px solid rgba(120,180,255,0.18);
        }

        #composer { background: #10131a; border-top: 1px solid rgba(255,255,255,0.06); }
        #composerInput {
            padding: 10px 10px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            color: #f2f2f2;
            font-size: 13px;
        }

        QPushButton { padding: 10px 12px; border-radius: 12px; }
        #primaryBtn {
            background: rgba(120,180,255,0.18);
            border: 1px solid rgba(120,180,255,0.22);
            color: #f2f2f2;
        }
        #primaryBtn:hover { background: rgba(120,180,255,0.25); }
        #ghostBtn {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.07);
            color: #f2f2f2;
        }
        #ghostBtn:hover { background: rgba(255,255,255,0.08); }

        #bubble {
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
        }
        #bubble[ bubbleRole="user" ] { background: rgba(120,180,255,0.14); border: 1px solid rgba(120,180,255,0.18); }
        #bubble[ bubbleRole="assistant" ] { background: rgba(255,255,255,0.04); }
        #bubble[ bubbleRole="tool" ] { background: rgba(255,180,120,0.12); border: 1px solid rgba(255,180,120,0.16); }
        #bubbleHeader { color: rgba(255,255,255,0.70); font-size: 12px; }
        #bubbleText { font-size: 13px; }

        #rightTitle { font-size: 15px; font-weight: 600; }
        QTabWidget::pane { border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }
        QTabBar::tab { padding: 8px 12px; border-top-left-radius: 10px; border-top-right-radius: 10px; }
        QTabBar::tab:selected { background: rgba(255,255,255,0.08); }
        QTextEdit {
            border: none;
            background: transparent;
            color: #e7e7e7;
            font-family: Consolas, monospace;
            font-size: 12px;
        }
        """)

    # ----------------------------
    # Actions
    # ----------------------------
    def _on_new_chat(self) -> None:
        self.inspector.log("[ui] new chat created")
        name = f"Chat {self.sidebar.list.count() + 1}"
        self.sidebar.list.addItem(QListWidgetItem(name))
        self.sidebar.list.setCurrentRow(self.sidebar.list.count() - 1)
        self._on_chat_selected(name)

    def _on_chat_selected(self, name: str) -> None:
        self.title.setText(name)
        self.inspector.log(f"[ui] selected chat: {name}")

    def _send_message(self, text: str) -> None:
        self.chat.add_message(ChatMessage("user", text))
        self.status.setText("thinking…")
        self.inspector.log("[api] send -> streaming")

        # Prepare streaming bubble
        self._stream_bubble = self.chat.add_message(ChatMessage("assistant", ""))

        # Call API (replace with your real client)
        self.api.send(text, streaming=True)

    # ----------------------------
    # API events
    # ----------------------------
    def _on_token(self, tok: str) -> None:
        if self._stream_bubble:
            self._stream_bubble.append_text(tok)
        self.status.setText("streaming…")

    def _on_done(self, payload: dict) -> None:
        self.status.setText("idle")
        self.inspector.log("[api] done")

        # If you want: set final text strictly (sometimes токены могут отличаться)
        if self._stream_bubble:
            # Keep what streamed; or enforce payload["text"]
            # self._stream_bubble._body.setText(payload["text"])
            self._stream_bubble = None

        self.inspector.set_meta(payload.get("meta", {}))
        self.inspector.set_memory(payload.get("memory", []))
        self.inspector.set_tools(payload.get("tools", []))

    def _on_error(self, msg: str) -> None:
        self.status.setText("error")
        self.inspector.log(f"[api] error: {msg}")
        self.chat.add_message(ChatMessage("system", f"Error: {msg}"))


def main() -> None:
    app = QApplication(sys.argv)

    # better default font
    f = QFont()
    f.setPointSize(10)
    app.setFont(f)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()