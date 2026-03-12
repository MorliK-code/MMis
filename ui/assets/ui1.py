# tests/ui_messenger.py
# PySide6 "Messenger-like" UI: fullscreen chat + slide drawers (Memory/Inspector/Logs)
# Run: python tests/ui_messenger.py

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt, QTimer, QObject, Signal, QEasingCurve, QPropertyAnimation
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QScrollArea, QFrame,
    QTextEdit, QSizePolicy
)


# ----------------------------
# Data
# ----------------------------
@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    text: str
    ts: float = time.time()


# ----------------------------
# Fake streaming API
# ----------------------------
class FakeApi(QObject):
    token = Signal(str)
    done = Signal(dict)
    error = Signal(str)

    def send(self, text: str, streaming: bool = True) -> None:
        reply = (
            "�������  ����� ����� ��� ����������-����. "
            "Параллельно помечу intent/emotion/topic и предложу факты в память. "
            "Если хочешь — добавим реакцию, голос и кнопки под сообщением."
        )

        payload = {
            "text": reply,
            "meta": {"intent": "chat", "emotion": "playful", "topic": "ui_messenger", "lang": "ru"},
            "memory": [
                {"key": "ui_style", "value": "messenger", "confidence": 0.86},
                {"key": "project", "value": "API-driven UI", "confidence": 0.79},
            ],
            "logs": [
                "metadata: 14ms",
                "retrieval: 0ms",
                "llm: 210ms",
                "render: 3ms",
            ]
        }

        if not streaming:
            self.done.emit(payload)
            return

        parts = reply.split(" ")
        self._i = 0

        def tick():
            if self._i >= len(parts):
                self._timer.stop()
                self.done.emit(payload)
                return
            self.token.emit(parts[self._i] + (" " if self._i < len(parts) - 1 else ""))
            self._i += 1

        self._timer = QTimer(self)
        self._timer.timeout.connect(tick)
        self._timer.start(40)


# ----------------------------
# UI pieces
# ----------------------------
class Bubble(QFrame):
    def __init__(self, role: str, text: str):
        super().__init__()
        self.setObjectName("bubble")
        self.setProperty("role", role)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        header = QLabel("Ты" if role == "user" else ("MMis" if role == "assistant" else role))
        header.setObjectName("bubbleHeader")

        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setObjectName("bubbleText")

        lay.addWidget(header)
        lay.addWidget(body)
        self._body = body

        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

    def append(self, t: str):
        self._body.setText(self._body.text() + t)

    def set_text(self, t: str):
        self._body.setText(t)


class ChatFeed(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.inner = QWidget()
        self.vbox = QVBoxLayout(self.inner)
        self.vbox.setContentsMargins(16, 16, 16, 16)
        self.vbox.setSpacing(10)
        self.vbox.addStretch(1)

        self.scroll.setWidget(self.inner)
        root.addWidget(self.scroll)

        self._last_assistant_bubble: Optional[Bubble] = None

    def add(self, msg: ChatMessage) -> Bubble:
        bubble = Bubble(msg.role, msg.text)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        if msg.role == "user":
            row.addStretch(1)
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch(1)

        self.vbox.insertLayout(self.vbox.count() - 1, row)
        self._scroll_bottom()

        if msg.role == "assistant":
            self._last_assistant_bubble = bubble

        return bubble

    def last_assistant(self) -> Optional[Bubble]:
        return self._last_assistant_bubble

    def _scroll_bottom(self):
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))


class Drawer(QFrame):
    """A slide-in panel (right side)."""
    def __init__(self, title: str, width: int = 380):
        super().__init__()
        self.setObjectName("drawer")
        self._open = False
        self._w = width
        self.setFixedWidth(width)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        top = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setObjectName("drawerTitle")
        btn = QPushButton("✕")
        btn.setObjectName("ghostBtn")
        btn.setFixedWidth(44)
        btn.clicked.connect(self.close)
        top.addWidget(lbl)
        top.addStretch(1)
        top.addWidget(btn)

        self.body = QTextEdit()
        self.body.setReadOnly(True)
        self.body.setObjectName("drawerBody")

        lay.addLayout(top)
        lay.addWidget(self.body, 1)

        self.anim = QPropertyAnimation(self, b"maximumWidth")
        self.anim.setDuration(180)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

        self.setMaximumWidth(0)  # start closed

    def toggle(self):
        if self._open:
            self.close()
        else:
            self.open()

    def open(self):
        self._open = True
        self.anim.stop()
        self.anim.setStartValue(self.maximumWidth())
        self.anim.setEndValue(self._w)
        self.anim.start()

    def close(self):
        self._open = False
        self.anim.stop()
        self.anim.setStartValue(self.maximumWidth())
        self.anim.setEndValue(0)
        self.anim.start()

    def set_text(self, text: str):
        self.body.setPlainText(text)


class Composer(QFrame):
    send = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("composer")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Напиши сообщение…")
        self.input.setFixedHeight(78)
        self.input.setObjectName("composerInput")

        btn_send = QPushButton("Отправить")
        btn_send.setObjectName("primaryBtn")
        btn_send.clicked.connect(self._on_send)

        lay.addWidget(self.input, 1)
        lay.addWidget(btn_send)

    def _on_send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self.send.emit(text)


# ----------------------------
# Main Window
# ----------------------------
class MessengerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MMis — Messenger UI (mock)")
        self.resize(1100, 760)

        self.api = FakeApi()
        self.api.token.connect(self.on_token)
        self.api.done.connect(self.on_done)
        self.api.error.connect(self.on_error)

        self.feed = ChatFeed()
        self.composer = Composer()
        self.composer.send.connect(self.send_message)

        # top bar (like a messenger header)
        header = QFrame()
        header.setObjectName("header")
        hb = QHBoxLayout(header)
        hb.setContentsMargins(14, 10, 14, 10)

        self.name = QLabel("Ася")
        self.name.setObjectName("nameLabel")

        self.status = QLabel("idle")
        self.status.setObjectName("statusLabel")

        btn_memory = QPushButton("Memory")
        btn_memory.setObjectName("ghostBtn")
        btn_memory.clicked.connect(lambda: self.drawer_memory.toggle())

        btn_inspector = QPushButton("Inspector")
        btn_inspector.setObjectName("ghostBtn")
        btn_inspector.clicked.connect(lambda: self.drawer_inspector.toggle())

        btn_logs = QPushButton("Logs")
        btn_logs.setObjectName("ghostBtn")
        btn_logs.clicked.connect(lambda: self.drawer_logs.toggle())

        hb.addWidget(self.name)
        hb.addSpacing(10)
        hb.addWidget(self.status)
        hb.addStretch(1)
        hb.addWidget(btn_memory)
        hb.addWidget(btn_inspector)
        hb.addWidget(btn_logs)

        # drawers (slide from right)
        self.drawer_memory = Drawer("Memory", 420)
        self.drawer_inspector = Drawer("Metadata / Inspector", 420)
        self.drawer_logs = Drawer("Logs", 420)

        # main layout: content + drawers in overlay row
        root = QWidget()
        rl = QHBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(header)
        cl.addWidget(self.feed, 1)
        cl.addWidget(self.composer, 0)

        rl.addWidget(content, 1)

        # stack drawers (only one typically open, но можно и несколько)
        drawers_col = QWidget()
        dl = QVBoxLayout(drawers_col)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(0)
        dl.addWidget(self.drawer_memory)
        dl.addWidget(self.drawer_inspector)
        dl.addWidget(self.drawer_logs)
        dl.addStretch(1)
        rl.addWidget(drawers_col, 0)

        self.setCentralWidget(root)

        # shortcuts
        QShortcut(QKeySequence("Ctrl+M"), self, activated=self.drawer_memory.toggle)
        QShortcut(QKeySequence("Ctrl+I"), self, activated=self.drawer_inspector.toggle)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.drawer_logs.toggle)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=lambda: self.composer.input.setFocus())

        self._stream_bubble: Optional[Bubble] = None

        self.feed.add(ChatMessage("assistant", "������  � � ��� ����������-����������. ������ ���-������ �����."))
        self.apply_qss()

    def apply_qss(self):
        self.setStyleSheet("""
        QMainWindow { background: #0f1115; }
        QLabel { color: #e9e9e9; font-size: 13px; }
        #header { background: #10131a; border-bottom: 1px solid rgba(255,255,255,0.06); }
        #nameLabel { font-size: 16px; font-weight: 700; }
        #statusLabel {
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: rgba(255,255,255,0.85);
        }

        #composer { background: #10131a; border-top: 1px solid rgba(255,255,255,0.06); }
        #composerInput {
            padding: 10px 12px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            color: #f1f1f1;
            font-size: 13px;
        }

        QPushButton { padding: 9px 12px; border-radius: 12px; }
        #primaryBtn {
            background: rgba(120,180,255,0.18);
            border: 1px solid rgba(120,180,255,0.22);
            color: #f2f2f2;
        }
        #primaryBtn:hover { background: rgba(120,180,255,0.26); }
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
        #bubble[role="user"] { background: rgba(120,180,255,0.14); border: 1px solid rgba(120,180,255,0.18); }
        #bubbleHeader { color: rgba(255,255,255,0.68); font-size: 12px; }
        #bubbleText { font-size: 13px; }

        #drawer {
            background: #121521;
            border-left: 1px solid rgba(255,255,255,0.08);
        }
        #drawerTitle { font-size: 14px; font-weight: 700; }
        #drawerBody {
            border: none;
            background: transparent;
            color: #e9e9e9;
            font-family: Consolas, monospace;
            font-size: 12px;
        }
        """)

    # --- chat actions ---
    def send_message(self, text: str):
        self.feed.add(ChatMessage("user", text))
        self.status.setText("thinking…")

        self._stream_bubble = self.feed.add(ChatMessage("assistant", ""))
        self.api.send(text, streaming=True)

    # --- API events ---
    def on_token(self, t: str):
        if self._stream_bubble:
            self._stream_bubble.append(t)
        self.status.setText("typing…")

    def on_done(self, payload: Dict[str, Any]):
        self.status.setText("idle")

        # fill drawers
        meta = payload.get("meta", {})
        mem = payload.get("memory", [])
        logs = payload.get("logs", [])

        self.drawer_inspector.set_text("\n".join([f"{k}: {v}" for k, v in meta.items()]) or "—")
        self.drawer_memory.set_text("\n".join([f"- {x['key']} = {x['value']} (conf={x['confidence']})" for x in mem]) or "—")
        self.drawer_logs.set_text("\n".join(logs) or "—")

        self._stream_bubble = None

    def on_error(self, msg: str):
        self.status.setText("error")
        self.feed.add(ChatMessage("system", f"Error: {msg}"))


def main():
    app = QApplication(sys.argv)
    f = QFont()
    f.setPointSize(10)
    app.setFont(f)

    w = MessengerWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()