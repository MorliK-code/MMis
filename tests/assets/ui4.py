# tests/ui_minimal.py
# Minimal Chat <-> Voice modes + separate Dev screen (PySide6)
# Run: python tests/ui_minimal.py

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, QTimer, QObject, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QPlainTextEdit,
    QScrollArea, QFrame, QTextEdit,
    QStackedWidget, QTabWidget
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
# Fake API (stream-like)
# Replace with your real API client later.
# ----------------------------
class FakeApi(QObject):
    token = Signal(str)
    done = Signal(dict)
    error = Signal(str)

    def send(self, user_text: str, streaming: bool = True) -> None:
        reply = (
            "Окей 🙂 Лаконично — это правильно. "
            "В чате оставляем только переписку и одну кнопку перехода в голос. "
            "В голосовом режиме — один большой контрол и минимум статуса."
        )
        payload = {
            "text": reply,
            "meta": {"intent": "chat", "emotion": "warm", "topic": "minimal_ui", "lang": "ru"},
            "memory": [{"key": "ui_style", "value": "minimal_chat_voice", "confidence": 0.88}],
            "logs": ["metadata: 11ms", "llm: 190ms", "render: 3ms"],
            "tools": [],
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
        self._timer.start(38)


# ----------------------------
# Chat UI (minimal)
# ----------------------------
class Bubble(QFrame):
    def __init__(self, role: str, text: str):
        super().__init__()
        self.setObjectName("bubble")
        self.setProperty("role", role)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setObjectName("bubbleText")

        lay.addWidget(body)
        self._body = body

    def append(self, t: str) -> None:
        self._body.setText(self._body.text() + t)

    def set_text(self, t: str) -> None:
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

        self._last_assistant: Optional[Bubble] = None

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
            self._last_assistant = bubble

        return bubble

    def last_assistant(self) -> Optional[Bubble]:
        return self._last_assistant

    def _scroll_bottom(self) -> None:
        QTimer.singleShot(
            0,
            lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())
        )


class ChatScreen(QWidget):
    go_voice = Signal()
    go_dev = Signal()
    send_text = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("chatScreen")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Minimal top bar: title + one tiny "…" for dev + one mic button
        top = QFrame()
        top.setObjectName("topBar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(12, 10, 12, 10)

        self.title = QLabel("Ася")
        self.title.setObjectName("title")

        btn_dev = QPushButton("⋯")
        btn_dev.setObjectName("ghostTiny")
        btn_dev.setFixedWidth(42)
        btn_dev.clicked.connect(self.go_dev.emit)

        btn_voice = QPushButton("🎙")
        btn_voice.setObjectName("ghostTiny")
        btn_voice.setFixedWidth(42)
        btn_voice.clicked.connect(self.go_voice.emit)

        tl.addWidget(self.title)
        tl.addStretch(1)
        tl.addWidget(btn_dev)
        tl.addWidget(btn_voice)

        self.feed = ChatFeed()

        # Minimal composer
        composer = QFrame()
        composer.setObjectName("composer")
        cl = QHBoxLayout(composer)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(10)

        self.input = QPlainTextEdit()
        self.input.setObjectName("input")
        self.input.setFixedHeight(76)
        self.input.setPlaceholderText("Напиши…")

        btn_send = QPushButton("↵")
        btn_send.setObjectName("primaryTiny")
        btn_send.setFixedWidth(52)
        btn_send.clicked.connect(self._on_send)

        cl.addWidget(self.input, 1)
        cl.addWidget(btn_send)

        root.addWidget(top)
        root.addWidget(self.feed, 1)
        root.addWidget(composer)

    def _on_send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self.send_text.emit(text)


# ----------------------------
# Voice UI (minimal)
# ----------------------------
class VoiceScreen(QWidget):
    back_to_chat = Signal()
    toggle_record = Signal(bool)  # True start, False stop

    def __init__(self):
        super().__init__()
        self.setObjectName("voiceScreen")
        self._recording = False

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        # Top: only back button
        top = QHBoxLayout()
        btn_back = QPushButton("⌨")
        btn_back.setObjectName("ghostTiny")
        btn_back.setFixedWidth(48)
        btn_back.clicked.connect(self.back_to_chat.emit)

        self.state = QLabel("Готова")
        self.state.setObjectName("voiceState")

        top.addWidget(btn_back)
        top.addStretch(1)
        top.addWidget(self.state)
        top.addStretch(1)

        # Center: big button
        center = QVBoxLayout()
        center.addStretch(1)

        self.big = QPushButton("ГОВОРИТЬ")
        self.big.setObjectName("voiceBig")
        self.big.setMinimumHeight(140)
        self.big.clicked.connect(self._toggle)

        center.addWidget(self.big)
        center.addStretch(1)

        # Bottom: tiny hint
        hint = QLabel("Нажми и говори. Никаких лишних кнопок.")
        hint.setObjectName("hint")

        root.addLayout(top)
        root.addLayout(center, 1)
        root.addWidget(hint, 0, Qt.AlignHCenter)

    def _toggle(self) -> None:
        self._recording = not self._recording
        self.toggle_record.emit(self._recording)

        if self._recording:
            self.big.setText("СТОП")
            self.state.setText("Слушаю…")
        else:
            self.big.setText("ГОВОРИТЬ")
            self.state.setText("Думаю…")  # типично: после записи -> обработка


# ----------------------------
# Dev/Studio screen (separate)
# ----------------------------
class DevScreen(QWidget):
    back = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("devScreen")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        top = QHBoxLayout()
        title = QLabel("Dev / Studio")
        title.setObjectName("devTitle")

        btn_back = QPushButton("← Назад")
        btn_back.setObjectName("ghostBtn")
        btn_back.clicked.connect(self.back.emit)

        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(btn_back)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("tabs")

        self.logs = QTextEdit(); self.logs.setReadOnly(True)
        self.meta = QTextEdit(); self.meta.setReadOnly(True)
        self.memory = QTextEdit(); self.memory.setReadOnly(True)
        self.tools = QTextEdit(); self.tools.setReadOnly(True)

        self.tabs.addTab(self.logs, "Logs")
        self.tabs.addTab(self.meta, "Metadata")
        self.tabs.addTab(self.memory, "Memory")
        self.tabs.addTab(self.tools, "Tools")

        root.addLayout(top)
        root.addWidget(self.tabs, 1)

    def set_payload(self, payload: Dict[str, Any]) -> None:
        meta = payload.get("meta", {})
        memory = payload.get("memory", [])
        logs = payload.get("logs", [])
        tools = payload.get("tools", [])

        self.meta.setPlainText("\n".join([f"{k}: {v}" for k, v in meta.items()]) or "—")
        self.memory.setPlainText("\n".join([f"- {x['key']} = {x['value']} (conf={x.get('confidence')})" for x in memory]) or "—")
        self.logs.setPlainText("\n".join(logs) or "—")
        self.tools.setPlainText("\n".join([str(t) for t in tools]) or "—")


# ----------------------------
# Main window with mode switching
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MMis — Minimal Chat/Voice + Dev")
        self.resize(1100, 760)

        self.api = FakeApi()
        self.api.token.connect(self._on_token)
        self.api.done.connect(self._on_done)
        self.api.error.connect(self._on_error)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.chat = ChatScreen()
        self.voice = VoiceScreen()
        self.dev = DevScreen()

        self.stack.addWidget(self.chat)   # index 0
        self.stack.addWidget(self.voice)  # index 1
        self.stack.addWidget(self.dev)    # index 2

        self.chat.go_voice.connect(lambda: self.stack.setCurrentIndex(1))
        self.chat.go_dev.connect(lambda: self.stack.setCurrentIndex(2))
        self.chat.send_text.connect(self._send_chat)

        self.voice.back_to_chat.connect(lambda: self.stack.setCurrentIndex(0))
        self.voice.toggle_record.connect(self._voice_toggle)

        self.dev.back.connect(lambda: self.stack.setCurrentIndex(0))

        # keyboard shortcuts (не нагромождают UI)
        QShortcut(QKeySequence("Ctrl+D"), self, activated=lambda: self.stack.setCurrentIndex(2))
        QShortcut(QKeySequence("Escape"), self, activated=self._escape_logic)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=lambda: self.chat.input.setFocus())

        # Enter to send in chat (Shift+Enter newline)
        self.chat.input.keyPressEvent = self._wrap_enter_send(self.chat.input.keyPressEvent)

        self._stream_bubble: Optional[Bubble] = None
        self._last_payload: Dict[str, Any] = {}

        self.chat.feed.add(ChatMessage("assistant", "Привет 🙂 Тут минималистичный чат. 🎙 — голос. ⋯ — Dev."))

        self._apply_qss()

    def _escape_logic(self) -> None:
        # Escape: from dev -> chat; from voice -> chat; from chat -> do nothing
        idx = self.stack.currentIndex()
        if idx in (1, 2):
            self.stack.setCurrentIndex(0)

    def _wrap_enter_send(self, original):
        def handler(e):
            if e.key() in (Qt.Key_Return, Qt.Key_Enter) and not (e.modifiers() & Qt.ShiftModifier):
                self.chat._on_send()
                return
            original(e)
        return handler

    # ---- Chat flow ----
    def _send_chat(self, text: str) -> None:
        self.chat.feed.add(ChatMessage("user", text))
        self._stream_bubble = self.chat.feed.add(ChatMessage("assistant", ""))
        self.api.send(text, streaming=True)

    def _on_token(self, tok: str) -> None:
        if self._stream_bubble:
            self._stream_bubble.append(tok)

    def _on_done(self, payload: Dict[str, Any]) -> None:
        self._last_payload = payload
        self.dev.set_payload(payload)
        self._stream_bubble = None

    def _on_error(self, msg: str) -> None:
        self.chat.feed.add(ChatMessage("system", f"Error: {msg}"))
        self._stream_bubble = None

    # ---- Voice flow (mock) ----
    def _voice_toggle(self, recording: bool) -> None:
        # Here you would start/stop STT recording.
        # After stop -> send transcript to API -> speak with TTS, etc.
        if recording:
            # start recording
            return
        # stop recording -> pretend transcript produced
        transcript = "Привет, давай поговорим голосом"
        # show that we go "think", then return "ready"
        self.voice.state.setText("Думаю…")
        self.api.send(transcript, streaming=False)
        # In a real app, you might keep voice mode and play TTS.
        QTimer.singleShot(400, lambda: self.voice.state.setText("Готова"))

    def _apply_qss(self) -> None:
        # Neutral minimal theme (you can swap to your style.css tokens)
        self.setStyleSheet("""
        QMainWindow { background: #0f1115; }
        QLabel { color: #eaeaea; font-size: 13px; }

        #topBar { background: #10131a; border-bottom: 1px solid rgba(255,255,255,0.06); }
        #title { font-size: 16px; font-weight: 700; }

        #composer { background: #10131a; border-top: 1px solid rgba(255,255,255,0.06); }
        #input {
            padding: 10px 12px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            color: #f2f2f2;
            font-size: 13px;
        }

        QPushButton { padding: 9px 12px; border-radius: 12px; }
        #primaryTiny {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.14);
            color: #f2f2f2;
        }
        #primaryTiny:hover { background: rgba(255,255,255,0.14); }

        #ghostTiny {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.07);
            color: #f2f2f2;
        }
        #ghostTiny:hover { background: rgba(255,255,255,0.08); }

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
        #bubble[role="user"] {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
        }
        #bubbleText { font-size: 13px; }

        #voiceState {
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            font-size: 14px;
            font-weight: 600;
        }
        #voiceBig {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.14);
            font-size: 18px;
            font-weight: 800;
            letter-spacing: 1px;
        }
        #voiceBig:hover { background: rgba(255,255,255,0.12); }
        #hint { color: rgba(255,255,255,0.55); }

        #devTitle { font-size: 16px; font-weight: 800; }
        QTabWidget::pane { border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }
        QTabBar::tab { padding: 8px 12px; border-top-left-radius: 10px; border-top-right-radius: 10px; }
        QTabBar::tab:selected { background: rgba(255,255,255,0.08); }
        QTextEdit {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            background: rgba(255,255,255,0.03);
            color: #eaeaea;
            font-family: Consolas, monospace;
            font-size: 12px;
        }
        """)


def main():
    app = QApplication(sys.argv)
    f = QFont()
    f.setPointSize(10)
    app.setFont(f)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()