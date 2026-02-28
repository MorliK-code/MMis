# tests/ui_room.py
# PySide6 "Character Room" UI: center stage + avatar pulse + bottom chat dock + HUD buttons
# Run: python tests/ui_room.py

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Qt, QTimer, QObject, Signal, QEasingCurve, QPropertyAnimation
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QFrame, QDialog, QTextEdit, QSizePolicy
)


# ----------------------------
# Data
# ----------------------------
@dataclass
class ChatMessage:
    role: str
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
            "Я тут 🙂 "
            "Сделала интерфейс как 'комната персонажа': центр — сцена, "
            "чат — нижняя панель, а сервисные штуки — в модалках. "
            "Можно добавить голос и анимацию 'слушает/думает/говорит'."
        )
        payload = {
            "text": reply,
            "meta": {"intent": "chat", "emotion": "warm", "topic": "character_room", "lang": "ru"},
            "memory": [
                {"key": "ui_concept", "value": "character_room", "confidence": 0.90},
            ],
            "logs": ["metadata: 12ms", "llm: 185ms", "render: 4ms"],
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
        self._timer.start(42)


# ----------------------------
# Modal viewer
# ----------------------------
class CardDialog(QDialog):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setObjectName("cardDialog")
        self.resize(560, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setObjectName("cardTitle")
        btn = QPushButton("Закрыть")
        btn.setObjectName("ghostBtn")
        btn.clicked.connect(self.close)

        top.addWidget(lbl)
        top.addStretch(1)
        top.addWidget(btn)

        self.body = QTextEdit()
        self.body.setReadOnly(True)
        self.body.setObjectName("cardBody")

        root.addLayout(top)
        root.addWidget(self.body, 1)

    def set_text(self, t: str) -> None:
        self.body.setPlainText(t)


# ----------------------------
# Avatar widget (simple animated orb)
# ----------------------------
class AvatarOrb(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pulse = 0.0  # 0..1
        self._state = "idle"  # idle/listening/thinking/speaking

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)  # ~60fps

        self._t = 0.0

    def set_state(self, s: str) -> None:
        self._state = s

    def _tick(self):
        self._t += 0.016
        # different pulse depending on state
        speed = {"idle": 1.0, "listening": 1.6, "thinking": 2.2, "speaking": 3.0}.get(self._state, 1.0)
        self._pulse = (1.0 + __import__("math").sin(self._t * speed * 2.0)) * 0.5
        self.update()

    def paintEvent(self, event):
        w = self.width()
        h = self.height()
        size = min(w, h)
        cx, cy = w / 2, h / 2

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # background fade
        p.fillRect(self.rect(), QColor(0, 0, 0, 0))

        # core orb
        base_r = size * 0.18
        glow_r = base_r * (1.4 + self._pulse * 0.9)

        # state tint (no hard colors, but slightly different alpha)
        alpha = {"idle": 60, "listening": 95, "thinking": 110, "speaking": 140}.get(self._state, 70)

        # glow ring
        pen = QPen(QColor(255, 255, 255, alpha))
        pen.setWidthF(max(2.0, size * 0.008))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(int(cx - glow_r), int(cy - glow_r), int(glow_r * 2), int(glow_r * 2))

        # inner orb
        p.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
        p.setBrush(QColor(255, 255, 255, 30))
        p.drawEllipse(int(cx - base_r), int(cy - base_r), int(base_r * 2), int(base_r * 2))

        # small highlights
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 45))
        p.drawEllipse(int(cx - base_r * 0.55), int(cy - base_r * 0.75), int(base_r * 0.35), int(base_r * 0.35))


# ----------------------------
# Bottom chat dock (collapsible)
# ----------------------------
class ChatDock(QFrame):
    send = Signal(str)
    toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("chatDock")
        self._open = True

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        top = QHBoxLayout()
        self.btn_toggle = QPushButton("▾ Свернуть чат")
        self.btn_toggle.setObjectName("ghostBtn")
        self.btn_toggle.clicked.connect(self.toggle)

        self.hint = QLabel("Enter — отправить • Shift+Enter — новая строка")
        self.hint.setObjectName("hint")

        top.addWidget(self.btn_toggle)
        top.addStretch(1)
        top.addWidget(self.hint)

        self.feed = QTextEdit()
        self.feed.setReadOnly(True)
        self.feed.setObjectName("chatFeed")
        self.feed.setFixedHeight(220)

        row = QHBoxLayout()
        self.input = QPlainTextEdit()
        self.input.setObjectName("chatInput")
        self.input.setFixedHeight(76)
        self.input.setPlaceholderText("Напиши…")

        btn_send = QPushButton("Отправить")
        btn_send.setObjectName("primaryBtn")
        btn_send.clicked.connect(self._on_send)

        row.addWidget(self.input, 1)
        row.addWidget(btn_send)

        root.addLayout(top)
        root.addWidget(self.feed)
        root.addLayout(row)

        # animate height (simple)
        self.anim = QPropertyAnimation(self.feed, b"maximumHeight")
        self.anim.setDuration(180)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

    def append(self, who: str, text: str):
        self.feed.append(f"<b>{who}:</b> {text}")

    def toggle(self):
        self._open = not self._open
        self.btn_toggle.setText("▾ Свернуть чат" if self._open else "▸ Развернуть чат")

        self.anim.stop()
        start = self.feed.maximumHeight()
        end = 220 if self._open else 0
        self.anim.setStartValue(start)
        self.anim.setEndValue(end)
        self.anim.start()
        self.toggled.emit(self._open)

    def _on_send(self):
        txt = self.input.toPlainText().strip()
        if not txt:
            return
        self.input.clear()
        self.send.emit(txt)


# ----------------------------
# Main window
# ----------------------------
class RoomWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MMis — Character Room (mock)")
        self.resize(1200, 780)

        self.api = FakeApi()
        self.api.token.connect(self._on_token)
        self.api.done.connect(self._on_done)
        self.api.error.connect(self._on_error)

        # modals
        self.modal_memory = CardDialog("Память", self)
        self.modal_meta = CardDialog("Метаданные", self)
        self.modal_logs = CardDialog("Логи", self)

        # stage
        stage = QWidget()
        stage.setObjectName("stage")
        s = QVBoxLayout(stage)
        s.setContentsMargins(18, 18, 18, 18)
        s.setSpacing(14)

        # top HUD
        hud = QHBoxLayout()
        self.name = QLabel("Ася")
        self.name.setObjectName("hudName")
        self.state = QLabel("idle")
        self.state.setObjectName("hudState")

        hud.addWidget(self.name)
        hud.addSpacing(10)
        hud.addWidget(self.state)
        hud.addStretch(1)

        btn_mem = QPushButton("Память")
        btn_mem.setObjectName("ghostBtn")
        btn_mem.clicked.connect(lambda: self.modal_memory.show())

        btn_meta = QPushButton("Метаданные")
        btn_meta.setObjectName("ghostBtn")
        btn_meta.clicked.connect(lambda: self.modal_meta.show())

        btn_logs = QPushButton("Логи")
        btn_logs.setObjectName("ghostBtn")
        btn_logs.clicked.connect(lambda: self.modal_logs.show())

        hud.addWidget(btn_mem)
        hud.addWidget(btn_meta)
        hud.addWidget(btn_logs)

        # center avatar
        self.orb = AvatarOrb()
        self.orb.setObjectName("orb")

        center = QVBoxLayout()
        center.addStretch(1)
        center.addWidget(self.orb, 0, Qt.AlignHCenter)
        center.addStretch(1)

        # bottom chat dock
        self.dock = ChatDock()
        self.dock.send.connect(self._send)
        self.dock.append("MMis", "Привет 🙂 Тут интерфейс 'комната персонажа'. Напиши что-нибудь.")
        self._streaming = False
        self._buffer = ""

        s.addLayout(hud)
        s.addLayout(center, 1)
        s.addWidget(self.dock, 0)

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(stage)
        self.setCentralWidget(root)

        # shortcuts
        QShortcut(QKeySequence("Ctrl+1"), self, activated=lambda: self.modal_memory.show())
        QShortcut(QKeySequence("Ctrl+2"), self, activated=lambda: self.modal_meta.show())
        QShortcut(QKeySequence("Ctrl+3"), self, activated=lambda: self.modal_logs.show())
        QShortcut(QKeySequence("Ctrl+T"), self, activated=self.dock.toggle)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=lambda: self.dock.input.setFocus())

        # Enter to send (unless Shift+Enter)
        self.dock.input.keyPressEvent = self._input_keypress_wrapper(self.dock.input.keyPressEvent)

        self._apply_qss()

    def _input_keypress_wrapper(self, original):
        def handler(e):
            if e.key() in (Qt.Key_Return, Qt.Key_Enter) and not (e.modifiers() & Qt.ShiftModifier):
                self.dock._on_send()
                return
            original(e)
        return handler

    def _apply_qss(self):
        self.setStyleSheet("""
        QMainWindow { background: #0f1115; }
        #stage {
            background: qradialgradient(cx:0.5, cy:0.38, radius: 1.2,
                        fx:0.5, fy:0.38,
                        stop:0 rgba(255,255,255,0.06),
                        stop:0.35 rgba(255,255,255,0.03),
                        stop:1 rgba(0,0,0,0.00));
        }
        QLabel { color: #eaeaea; font-size: 13px; }
        #hudName { font-size: 18px; font-weight: 800; }
        #hudState {
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: rgba(255,255,255,0.86);
        }

        QPushButton { padding: 9px 12px; border-radius: 12px; }
        #primaryBtn {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.14);
            color: #f2f2f2;
        }
        #primaryBtn:hover { background: rgba(255,255,255,0.14); }
        #ghostBtn {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.07);
            color: #f2f2f2;
        }
        #ghostBtn:hover { background: rgba(255,255,255,0.08); }

        #chatDock {
            background: rgba(16,19,26,0.90);
            border-top: 1px solid rgba(255,255,255,0.08);
        }
        #hint { color: rgba(255,255,255,0.55); font-size: 12px; }

        #chatFeed {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            color: #eaeaea;
            font-size: 13px;
        }
        #chatInput {
            padding: 10px 12px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.04);
            color: #f2f2f2;
            font-size: 13px;
        }

        #cardDialog {
            background: #121521;
        }
        #cardTitle { font-size: 15px; font-weight: 800; }
        #cardBody {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            color: #eaeaea;
            font-family: Consolas, monospace;
            font-size: 12px;
        }
        """)

    # --------------- chat flow ---------------
    def _set_state(self, s: str):
        self.state.setText(s)
        self.orb.set_state(s)

    def _send(self, user_text: str):
        self.dock.append("Ты", user_text)
        self._buffer = ""
        self._streaming = True
        self._set_state("thinking")
        self.api.send(user_text, streaming=True)

    def _on_token(self, tok: str):
        if not self._streaming:
            return
        self._set_state("speaking")
        self._buffer += tok

        # update last assistant line (simple: rewrite by appending; in real UI use a bubble model)
        # we'll just keep appending as separate lines for the mock:
        # better: show as one live line — do it by clearing last line (hard in QTextEdit).
        # For now: add every ~6 tokens as one line to avoid spam.
        if self._buffer.count(" ") % 6 == 0:
            self.dock.append("MMis", self._buffer.strip())

    def _on_done(self, payload: Dict[str, Any]):
        self._streaming = False
        self._set_state("idle")

        final_text = payload.get("text", "").strip()
        if final_text:
            self.dock.append("MMis", final_text)

        meta = payload.get("meta", {})
        mem = payload.get("memory", [])
        logs = payload.get("logs", [])

        self.modal_meta.set_text("\n".join([f"{k}: {v}" for k, v in meta.items()]) or "—")
        self.modal_memory.set_text("\n".join([f"- {x['key']} = {x['value']} (conf={x['confidence']})" for x in mem]) or "—")
        self.modal_logs.set_text("\n".join(logs) or "—")

    def _on_error(self, msg: str):
        self._streaming = False
        self._set_state("error")
        self.dock.append("SYSTEM", f"Error: {msg}")


def main():
    app = QApplication(sys.argv)
    f = QFont()
    f.setPointSize(10)
    app.setFont(f)

    w = RoomWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()