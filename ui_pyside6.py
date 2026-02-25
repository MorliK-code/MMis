"""MMis Desktop UI (PySide6)

Это только UI поверх существующего "мозга" MMis.
Никакой логики ответа тут не переизобретается: мы импортируем Brain/MemoryManager
и вызываем brain.think(...).

Запуск:
  pip install PySide6
  python ui_pyside6.py

Горячие клавиши:
  Ctrl+Enter — отправить сообщение
"""

from __future__ import annotations

import sys
import traceback
import html
from dataclasses import dataclass

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

# --- MMis core imports ---
from memory.short_memory import ShortMemory
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.user_profile import UserProfile
from memory.event_store import EventStore
from memory.chat_log import ChatLog
from memory.assistant_profile import AssistantProfile
from brain import Brain
from config import SHORT_MEMORY_LIMIT, MemoryStorageDir, MODEL_NAME


# --- (optional) TFLOPs estimation (very approximate) ---
FLOPS_PER_TOKEN = 14e9
SHOW_TFLOPS_EST = True


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def est_tflops(tokens_per_sec: float) -> float:
    return (FLOPS_PER_TOKEN * tokens_per_sec) / 1e12


def build_brain() -> Brain:
    """Создаёт те же объекты, что и main.py (CLI), но для UI."""
    short = ShortMemory(limit=SHORT_MEMORY_LIMIT)
    longm = LongMemory(path=MemoryStorageDir / "chroma_db")
    profile_user = UserProfile(MemoryStorageDir / "user_profile.json")
    profile_assistant = AssistantProfile(MemoryStorageDir / "assistant_profile.json")
    events = EventStore(MemoryStorageDir / "events.json")
    log = ChatLog(MemoryStorageDir / "chat_log.jsonl")

    mm = MemoryManager(
        short,
        longm,
        profile_user,
        profile_assistant,
        events,
        log,
        distance_threshold=0.65,
    )
    return Brain(mm)


@dataclass
class ReplyResult:
    text: str
    stats: dict


class ReplyWorker(QObject):
    finished = Signal(object)  # ReplyResult
    errored = Signal(str)

    def __init__(self, brain: Brain, user_text: str, audience: str):
        super().__init__()
        self.brain = brain
        self.user_text = user_text
        self.audience = audience
        self._cancel_requested = False

    def request_cancel(self):
        # Мы не можем мгновенно оборвать blocking-вызов ollama.chat,
        # но можем игнорировать результат.
        self._cancel_requested = True

    @Slot()
    def run(self):
        try:
            answer = self.brain.think(self.user_text, audience=self.audience)
            stats = getattr(self.brain, "last_stats", {}) or {}
            if self._cancel_requested:
                return
            self.finished.emit(ReplyResult(text=answer, stats=stats))
        except Exception:
            self.errored.emit(traceback.format_exc())


CHAT_CSS = """
<style>
/* общие */
.msg { margin: 10px 0 18px 0; }
.name { font-weight: 600; margin-bottom: 6px; }
.bubble {
  display: inline-block;
  padding: 10px 12px;
  border-radius: 12px;
  max-width: 820px;
  line-height: 1.35;
  white-space: pre-wrap;
}

/* твои сообщения */
.msg.user .bubble { background: rgba(120,120,120,0.14); }

/* её сообщения */
.msg.ai .bubble { background: rgba(120,120,120,0.09); }

/* статистика под её ответом */
.stats {
  margin-top: 6px;
  font-size: 11px;
  color: rgba(0,0,0,0.45);  /* светлая тема */
}

/* если вдруг используешь тёмную тему — раскомментируй:
.stats { color: rgba(255,255,255,0.55); }
*/
.sep { height: 10px; }
</style>
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MMis — Desktop")
        self.resize(980, 720)

        self.brain = build_brain()

        self._thread: QThread | None = None
        self._worker: ReplyWorker | None = None
        self._last_user_text: str | None = None

        # агрегаты для средних значений
        self.n_answers = 0
        self.sum_ms = 0.0
        self.sum_eval = 0
        self.sum_prompt = 0

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- Top bar ---
        top = QHBoxLayout()
        layout.addLayout(top)

        self.model_label = QLabel(f"Model: <b>{MODEL_NAME}</b>")
        top.addWidget(self.model_label)

        top.addSpacing(14)
        top.addWidget(QLabel("Кому отвечает:"))
        self.audience_box = QComboBox()
        self.audience_box.addItem("Я один (на ты)", userData="single")
        self.audience_box.addItem("Мы компанией (на вы)", userData="group")
        top.addWidget(self.audience_box)

        top.addStretch(1)

        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        top.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self.on_clear)
        top.addWidget(self.btn_clear)

        # --- Splitter: chat + stats ---
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # Chat view
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.chat.setFont(QFont("Segoe UI", 11))
        self.chat.setHtml(CHAT_CSS)  # важно: CSS
        splitter.addWidget(self.chat)

        # Stats panel (справа)
        stats_wrap = QWidget()
        stats_layout = QVBoxLayout(stats_wrap)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(8)

        self.status_label = QLabel("Статус: <b>Готово</b>")

        # last ответ
        self.last_ms_label = QLabel("Последний ответ: время —")
        self.last_tokens_label = QLabel("Последний ответ: gen — / prompt —")
        self.last_tps_label = QLabel("Последний ответ: tok/s —")
        self.last_tflops_label = QLabel("Последний ответ: TFLOPs —")

        # average
        self.avg_ms_label = QLabel("Среднее: время —")
        self.avg_tokens_label = QLabel("Среднее: gen — / prompt —")
        self.avg_tps_label = QLabel("Среднее: tok/s —")
        self.avg_tflops_label = QLabel("Среднее: TFLOPs —")

        for w in (
            self.status_label,
            self.last_ms_label,
            self.last_tokens_label,
            self.last_tps_label,
            self.last_tflops_label,
            self.avg_ms_label,
            self.avg_tokens_label,
            self.avg_tps_label,
            self.avg_tflops_label,
        ):
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            stats_layout.addWidget(w)

        stats_layout.addStretch(1)
        splitter.addWidget(stats_wrap)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # --- Input area ---
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Напиши сообщение…")
        self.input.setFont(QFont("Segoe UI", 11))
        self.input.setFixedHeight(110)
        layout.addWidget(self.input)

        # Buttons
        bottom = QHBoxLayout()
        layout.addLayout(bottom)

        self.btn_regen = QPushButton("Регенерировать")
        self.btn_regen.clicked.connect(self.on_regen)
        self.btn_regen.setEnabled(False)
        bottom.addWidget(self.btn_regen)

        bottom.addStretch(1)

        self.btn_send = QPushButton("Отправить")
        self.btn_send.clicked.connect(self.on_send)
        self.btn_send.setDefault(True)
        bottom.addWidget(self.btn_send)

        # Ctrl+Enter to send
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.on_send)
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self.on_send)

        self._append_system("MMis UI запущен. Ctrl+Enter — отправить.")

    # --- Chat rendering (HTML bubbles) ---
    def _append_system(self, text: str):
        safe = html.escape(text).replace("\n", "<br>")
        self.chat.append(
            f"<div class='msg'>"
            f"<div class='name'>SYSTEM</div>"
            f"<div class='bubble'>{safe}</div>"
            f"</div><div class='sep'></div>"
        )
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def _append_user(self, text: str):
        safe = html.escape(text).replace("\n", "<br>")
        self.chat.append(
            f"<div class='msg user'>"
            f"<div class='name'>Ты</div>"
            f"<div class='bubble'>{safe}</div>"
            f"</div><div class='sep'></div>"
        )
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def _append_ai(self, text: str, stat_line: str):
        safe = html.escape(text).replace("\n", "<br>")
        stat_safe = html.escape(stat_line)
        self.chat.append(
            f"<div class='msg ai'>"
            f"<div class='name'>Она</div>"
            f"<div class='bubble'>{safe}</div>"
            f"<div class='stats'>{stat_safe}</div>"
            f"</div><div class='sep'></div>"
        )
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def _set_status(self, status: str):
        self.status_label.setText(f"Статус: <b>{status}</b>")

    def _format_stats_line(self, stats: dict) -> tuple[str, float, int, int]:
        """
        Возвращает:
          (строка_под_ответом, ms, gen, prompt)
        """
        ms = stats.get("answer_ms") or stats.get("ms")
        gen = stats.get("eval_count")
        prompt = stats.get("prompt_eval_count")

        ms_f = float(ms) if ms is not None else 0.0
        gen_i = int(gen) if gen is not None else 0
        prompt_i = int(prompt) if prompt is not None else 0

        sec = ms_f / 1000.0 if ms_f else 0.0
        tps = safe_div(gen_i, sec)

        parts = []
        if ms is not None:
            parts.append(f"{int(ms_f)} ms")
        if prompt is not None:
            parts.append(f"prompt {prompt_i}")
        if gen is not None:
            parts.append(f"gen {gen_i}")
        if tps:
            parts.append(f"{tps:.1f} tok/s")
        if SHOW_TFLOPS_EST and tps:
            parts.append(f"~{est_tflops(tps):.2f} TFLOPs (est)")

        return (" • ".join(parts) if parts else "—", ms_f, gen_i, prompt_i)

    def _update_side_stats(self, last_ms: float, last_gen: int, last_prompt: int):
        # last
        sec = last_ms / 1000.0 if last_ms else 0.0
        last_tps = safe_div(last_gen, sec)
        self.last_ms_label.setText(f"Последний ответ: время {int(last_ms) if last_ms else '—'}")
        self.last_tokens_label.setText(f"Последний ответ: gen {last_gen or '—'} / prompt {last_prompt or '—'}")
        self.last_tps_label.setText(f"Последний ответ: tok/s {last_tps:.1f}" if last_tps else "Последний ответ: tok/s —")
        self.last_tflops_label.setText(
            f"Последний ответ: TFLOPs ~{est_tflops(last_tps):.2f}" if (SHOW_TFLOPS_EST and last_tps) else "Последний ответ: TFLOPs —"
        )

        # avg
        n = max(self.n_answers, 1)
        avg_ms = self.sum_ms / n if self.sum_ms else 0.0
        avg_gen = self.sum_eval / n if self.sum_eval else 0.0
        avg_prompt = self.sum_prompt / n if self.sum_prompt else 0.0

        total_sec = (self.sum_ms / 1000.0) if self.sum_ms else 0.0
        avg_tps = safe_div(self.sum_eval, total_sec)

        self.avg_ms_label.setText(f"Среднее: время {int(avg_ms) if avg_ms else '—'}")
        self.avg_tokens_label.setText(f"Среднее: gen {avg_gen:.0f} / prompt {avg_prompt:.0f}")
        self.avg_tps_label.setText(f"Среднее: tok/s {avg_tps:.1f}" if avg_tps else "Среднее: tok/s —")
        self.avg_tflops_label.setText(
            f"Среднее: TFLOPs ~{est_tflops(avg_tps):.2f}" if (SHOW_TFLOPS_EST and avg_tps) else "Среднее: TFLOPs —"
        )

    # --- Actions ---
    @Slot()
    def on_clear(self):
        self.chat.setHtml(CHAT_CSS)
        self._append_system("Чат очищен.")
        self._last_user_text = None
        self.btn_regen.setEnabled(False)

        # reset averages
        self.n_answers = 0
        self.sum_ms = 0.0
        self.sum_eval = 0
        self.sum_prompt = 0

        # reset panel
        self._set_status("Готово")
        self.last_ms_label.setText("Последний ответ: время —")
        self.last_tokens_label.setText("Последний ответ: gen — / prompt —")
        self.last_tps_label.setText("Последний ответ: tok/s —")
        self.last_tflops_label.setText("Последний ответ: TFLOPs —")
        self.avg_ms_label.setText("Среднее: время —")
        self.avg_tokens_label.setText("Среднее: gen — / prompt —")
        self.avg_tps_label.setText("Среднее: tok/s —")
        self.avg_tflops_label.setText("Среднее: TFLOPs —")

    @Slot()
    def on_stop(self):
        if self._worker:
            self._worker.request_cancel()
        self._set_status("Остановка… (результат будет проигнорирован)")

    @Slot()
    def on_regen(self):
        if not self._last_user_text:
            return
        self._start_request(self._last_user_text, show_user=False)

    @Slot()
    def on_send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._last_user_text = text
        self.btn_regen.setEnabled(True)
        self._start_request(text, show_user=True)

    def _start_request(self, user_text: str, show_user: bool):
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "Подожди", "Сейчас уже идёт генерация. Нажми 'Стоп' или дождись ответа.")
            return

        if show_user:
            self._append_user(user_text)

        audience = self.audience_box.currentData() or "single"

        self._set_status("Генерация…")
        self.btn_send.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._thread = QThread()
        self._worker = ReplyWorker(self.brain, user_text=user_text, audience=audience)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_reply)
        self._worker.errored.connect(self._on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.errored.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    @Slot(object)
    def _on_reply(self, res: ReplyResult):
        stats = res.stats or {}
        stat_line, ms, gen, prompt = self._format_stats_line(stats)

        # вывод в чат: ответ + статистика под ним
        self._append_ai(res.text, stat_line)

        # обновить агрегаты
        if ms:
            self.sum_ms += ms
        self.sum_eval += gen
        self.sum_prompt += prompt
        self.n_answers += 1

        self._update_side_stats(ms, gen, prompt)
        self._set_status("Готово")

    @Slot(str)
    def _on_error(self, tb: str):
        self._set_status("Ошибка")
        QMessageBox.critical(self, "Ошибка", tb)

    @Slot()
    def _cleanup_thread(self):
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._worker:
            self._worker.deleteLater()
        if self._thread:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()