"""MMis Desktop UI (PySide6)."""

from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QCursor, QFont, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from brain import Brain
from config import MODEL_NAME, MemoryStorageDir, SHORT_MEMORY_LIMIT
from memory.assistant_profile import AssistantProfile
from memory.chat_log import ChatLog
from memory.event_store import EventStore
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.short_memory import ShortMemory
from memory.user_profile import UserProfile
from ui import config as ui_config
from ui.livecss import LiveCss

DEFAULT_TEXT_SIZE = getattr(ui_config, "DEFAULT_TEXT_SIZE", 14)
DEFAULT_BUBBLE_OPACITY = getattr(ui_config, "DEFAULT_BUBBLE_OPACITY", 0.1)
FLOPS_PER_TOKEN = getattr(ui_config, "FLOPS_PER_TOKEN", 14e9)
SHOW_TFLOPS_EST = getattr(ui_config, "SHOW_TFLOPS_EST", True)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def est_tflops(tokens_per_sec: float) -> float:
    return (FLOPS_PER_TOKEN * tokens_per_sec) / 1e12


def ms_to_s_text(ms_value: float) -> str:
    return f"{(float(ms_value) / 1000.0):.1f} с"


def build_brain() -> Brain:
    short = ShortMemory(limit=SHORT_MEMORY_LIMIT)
    longm = LongMemory(path=MemoryStorageDir / "chroma_db")
    profile_user = UserProfile(MemoryStorageDir / "user_profile.json")
    profile_assistant = AssistantProfile(MemoryStorageDir / "assistant_profile.json")
    events = EventStore(MemoryStorageDir / "events.json")
    log = ChatLog(MemoryStorageDir / "chat_log.jsonl")

    mm = MemoryManager(short, longm, profile_user, profile_assistant, events, log, distance_threshold=0.65)
    return Brain(mm)


@dataclass
class ReplyResult:
    text: str
    stats: dict


class ReplyWorker(QObject):
    finished = Signal(object)
    errored = Signal(str)
    chunk = Signal(str)

    def __init__(self, brain: Brain, user_text: str):
        super().__init__()
        self.brain = brain
        self.user_text = user_text
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    @Slot()
    def run(self):
        try:
            answer = self.brain.think_stream(self.user_text, on_chunk=self.chunk.emit)
            stats = getattr(self.brain, "last_stats", {}) or {}
            if self._cancel_requested:
                return
            self.finished.emit(ReplyResult(text=answer, stats=stats))
        except Exception:
            self.errored.emit(traceback.format_exc())


class _HoverRevealFilter(QObject):
    def __init__(self, owner: QWidget, target: QWidget, show_ms: int = 170, hide_ms: int = 130):
        super().__init__(owner)
        self.owner = owner
        self.target = target
        self.show_ms = max(0, int(show_ms))
        self.hide_ms = max(40, int(hide_ms))
        self._pressed_inside_target = False
        self._outside_streak = 0
        target.setVisible(False)
        self._sync_timer = QTimer(target)
        self._sync_timer.setInterval(60)
        self._sync_timer.timeout.connect(self._sync_visibility)
        self._sync_timer.start()

    def _contains_global(self, w: QWidget, global_pos, pad: int = 2) -> bool:
        if not isValid(w) or not w.isVisible():
            return False
        local = w.mapFromGlobal(global_pos)
        rect = w.rect().adjusted(-pad, -pad, pad, pad)
        return rect.contains(local)

    def _is_cursor_inside_owner_or_target(self) -> bool:
        if not isValid(self.owner) or not isValid(self.target):
            return False
        gpos = QCursor.pos()
        hovered = QApplication.widgetAt(gpos)
        if hovered is None:
            hovered_hit = False
        else:
            hovered_hit = (
                hovered is self.owner
                or hovered is self.target
                or (isinstance(hovered, QWidget) and self.owner.isAncestorOf(hovered))
                or (isinstance(hovered, QWidget) and self.target.isAncestorOf(hovered))
            )
        if hovered_hit:
            return True

        # Fallback: geometry-based check with a tiny tolerance to avoid flicker at edges.
        if self._contains_global(self.owner, gpos, pad=3):
            return True
        if self._contains_global(self.target, gpos, pad=3):
            return True
        for child in self.target.findChildren(QWidget):
            if self._contains_global(child, gpos, pad=3):
                return True
        return False

    def _show_if_inside(self) -> None:
        if not isValid(self.target):
            return
        if self._is_cursor_inside_owner_or_target() or self._pressed_inside_target:
            self._outside_streak = 0
            self.target.setVisible(True)

    def _show(self) -> None:
        if not isValid(self.target):
            return
        self.target.setVisible(True)

    def _sync_visibility(self) -> None:
        if not isValid(self.target):
            return
        if self._pressed_inside_target:
            self._outside_streak = 0
            self.target.setVisible(True)
            return
        if self._is_cursor_inside_owner_or_target():
            self._outside_streak = 0
            self.target.setVisible(True)
            return
        self._outside_streak += 1
        if self._outside_streak >= 4:
            self.target.setVisible(False)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        et = event.type()
        if et in (QEvent.Enter, QEvent.HoverEnter, QEvent.MouseMove, QEvent.Show):
            self._sync_visibility()
        elif et == QEvent.MouseButtonPress:
            if watched is self.target or (isinstance(watched, QWidget) and self.target.isAncestorOf(watched)):
                self._pressed_inside_target = True
                self._show()
        elif et == QEvent.MouseButtonRelease:
            self._pressed_inside_target = False
            self._sync_visibility()
        return False


class _ButtonAnimFilter(QObject):
    def __init__(self, button: QPushButton, fade_ms: int = 170, hover_ms: int = 140, press_ms: int = 90):
        super().__init__(button)
        self.button = button
        self.fade_ms = max(50, int(fade_ms))
        self.hover_ms = max(50, int(hover_ms))
        self.press_ms = max(40, int(press_ms))
        self.idle_opacity = 0.94

        self.opacity = QGraphicsOpacityEffect(button)
        self.opacity.setOpacity(0.0)
        button.setGraphicsEffect(self.opacity)

        self.anim = QPropertyAnimation(self.opacity, b"opacity", button)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self._animate_to(self.idle_opacity, self.fade_ms)

    def set_durations(self, fade_ms: int, hover_ms: int, press_ms: int) -> None:
        self.fade_ms = max(50, int(fade_ms))
        self.hover_ms = max(50, int(hover_ms))
        self.press_ms = max(40, int(press_ms))

    def _animate_to(self, value: float, duration_ms: int) -> None:
        if not isValid(self.button):
            return
        self.anim.stop()
        self.anim.setDuration(int(duration_ms))
        self.anim.setStartValue(self.opacity.opacity())
        self.anim.setEndValue(float(value))
        self.anim.start()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if not isValid(self.button):
            return False
        et = event.type()
        if et == QEvent.Show:
            self._animate_to(self.idle_opacity, self.fade_ms)
        elif et == QEvent.Enter:
            self._animate_to(1.0, self.hover_ms)
        elif et == QEvent.Leave:
            self._animate_to(self.idle_opacity, self.hover_ms)
        elif et == QEvent.MouseButtonPress:
            self._animate_to(0.78, self.press_ms)
        elif et == QEvent.MouseButtonRelease:
            self._animate_to(1.0 if self.button.underMouse() else self.idle_opacity, self.press_ms)
        return False


class _FeedbackVisualFilter(QObject):
    def __init__(
        self,
        button: QPushButton,
        base_bg: str,
        hover_bg: str,
        active_bg: str,
        text_color: str,
        border_color: str,
        radius_px: int,
        font_size_px: int,
        font_weight: int,
        pad_y_px: int,
        pad_x_px: int,
    ):
        super().__init__(button)
        self.button = button
        self.base_bg = base_bg
        self.hover_bg = hover_bg
        self.active_bg = active_bg
        self.text_color = text_color
        self.border_color = border_color
        self.radius_px = int(radius_px)
        self.font_size_px = int(font_size_px)
        self.font_weight = int(font_weight)
        self.pad_y_px = int(pad_y_px)
        self.pad_x_px = int(pad_x_px)
        self.button.installEventFilter(self)
        self.refresh()

    def _compose_qss(self, bg: str) -> str:
        return (
            "QPushButton {"
            f"background: {bg};"
            f"color: {self.text_color};"
            f"border: 1px solid {self.border_color};"
            f"border-radius: {self.radius_px}px;"
            f"font-size: {self.font_size_px}px;"
            f"font-weight: {self.font_weight};"
            f"padding: {self.pad_y_px}px {self.pad_x_px}px;"
            "}"
        )

    def refresh(self) -> None:
        if not isValid(self.button):
            return
        selected = bool(self.button.property("selected"))
        if selected:
            bg = self.active_bg
        elif self.button.underMouse():
            bg = self.hover_bg
        else:
            bg = self.base_bg
        self.button.setStyleSheet(self._compose_qss(bg))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        et = event.type()
        if et in (
            QEvent.Enter,
            QEvent.Leave,
            QEvent.MouseMove,
            QEvent.MouseButtonPress,
            QEvent.MouseButtonRelease,
            QEvent.Show,
            QEvent.DynamicPropertyChange,
        ):
            self.refresh()
        return False


class _MessageCard(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._overlay: QWidget | None = None
        self._overlay_bottom = 6
        self._overlay_right = 8

    def set_overlay(self, widget: QWidget, bottom: int, right: int) -> None:
        self._overlay = widget
        self._overlay_bottom = int(bottom)
        self._overlay_right = int(right)
        widget.setParent(self)
        widget.raise_()
        widget.adjustSize()
        self._reposition_overlay()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        if not self._overlay or not isValid(self._overlay):
            return
        hint = self._overlay.sizeHint()
        max_x = max(0, self.width() - hint.width())
        max_y = max(0, self.height() - hint.height())
        x = self.width() - hint.width() - self._overlay_right
        y = self.height() - hint.height() - self._overlay_bottom
        x = max(0, min(max_x, x))
        y = max(0, min(max_y, y))
        self._overlay.move(x, y)


class _OverlayHost(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._overlay: QWidget | None = None
        self._overlay_bottom = 6
        self._overlay_right = 8

    def set_overlay(self, widget: QWidget, bottom: int, right: int) -> None:
        self._overlay = widget
        self._overlay_bottom = int(bottom)
        self._overlay_right = int(right)
        widget.setParent(self)
        widget.raise_()
        widget.adjustSize()
        self._reposition_overlay()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        if not self._overlay or not isValid(self._overlay):
            return
        hint = self._overlay.sizeHint()
        x = max(0, self.width() - hint.width() - self._overlay_right)
        y = max(0, self.height() - hint.height() - self._overlay_bottom)
        self._overlay.move(x, y)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MMis \u2014 Desktop")
        self.resize(1080, 760)

        self.brain = build_brain()

        self._thread: QThread | None = None
        self._worker: ReplyWorker | None = None
        self._last_user_text: str | None = None
        self._history: list[tuple[str, str, str | None, int | None]] = []
        self._stream_ai_index: int | None = None
        self._stream_chunk_buffer: str = ""
        self._stream_flush_timer = QTimer(self)
        self._stream_flush_timer.setSingleShot(True)
        self._stream_flush_timer.setInterval(70)
        self._stream_flush_timer.timeout.connect(self._flush_stream_chunks)

        self.n_answers = 0
        self.sum_ms = 0.0
        self.sum_decode_ms = 0.0
        self.sum_eval = 0
        self.sum_prompt = 0
        self.sum_tps = 0.0

        self._text_size = DEFAULT_TEXT_SIZE
        self._bubble_opacity = DEFAULT_BUBBLE_OPACITY
        self._live_css = LiveCss(Path(__file__).with_name("style.css"))
        self._feedback_reveal_show_ms = 170
        self._feedback_reveal_hide_ms = 130

        self._build_ui()

        self._css_timer = QTimer(self)
        self._css_timer.setInterval(1000)
        self._css_timer.timeout.connect(self._on_live_css_tick)
        self._css_timer.start()

    @staticmethod
    def _apply_panel_shadow(
        widget: QWidget,
        blur: float = 18.0,
        x: float = 0.0,
        y: float = 8.0,
        color: QColor | None = None,
    ) -> None:
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur)
        shadow.setOffset(x, y)
        shadow.setColor(color or QColor(0, 0, 0, 120))
        widget.setGraphicsEffect(shadow)

    @staticmethod
    def _apply_text_shadow(widget: QWidget, shadow_spec: tuple[float, float, float, QColor]) -> None:
        sx, sy, blur, color = shadow_spec
        if blur <= 0.0 or color.alpha() <= 0:
            widget.setGraphicsEffect(None)
            return
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur)
        shadow.setOffset(sx, sy)
        shadow.setColor(color)
        widget.setGraphicsEffect(shadow)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("app_root")
        self.root_widget = root
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        top = QHBoxLayout()
        layout.addLayout(top)
        self.model_label = QLabel(f"Model: <b>{MODEL_NAME}</b>")
        self.model_label.setObjectName("model_label")
        top.addWidget(self.model_label)
        top.addStretch(1)

        self.btn_stop = QPushButton("\u0421\u0442\u043e\u043f")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        top.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("\u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.clicked.connect(self.on_clear)
        top.addWidget(self.btn_clear)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        self.chat_panel = QFrame()
        self.chat_panel.setObjectName("chat_panel")

        chat_panel_layout = QVBoxLayout(self.chat_panel)
        chat_panel_layout.setContentsMargins(0, 0, 0, 0)
        chat_panel_layout.setSpacing(0)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chat_scroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)
        self.chat_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chat_root = QWidget()
        self.chat_root.setObjectName("chat_inner")
        self.chat_layout = QVBoxLayout(self.chat_root)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.setSpacing(0)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_scroll.setWidget(self.chat_root)
        chat_panel_layout.addWidget(self.chat_scroll)
        splitter.addWidget(self.chat_panel)

        self.stats_panel = QFrame()
        self.stats_panel.setObjectName("stats_panel")
        side_layout = QVBoxLayout(self.stats_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        self.status_label = QLabel("\u0421\u0442\u0430\u0442\u0443\u0441: <b>\u0413\u043e\u0442\u043e\u0432\u043e</b>")
        self.status_label.setObjectName("stats_label")
        self.avg_ms_label = QLabel("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: \u0432\u0440\u0435\u043c\u044f \u2014")
        self.avg_ms_label.setObjectName("stats_label")
        self.avg_decode_label = QLabel("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: decode \u2014")
        self.avg_decode_label.setObjectName("stats_label")
        self.avg_tokens_label = QLabel("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: gen \u2014 / prompt \u2014")
        self.avg_tokens_label.setObjectName("stats_label")
        self.avg_tps_label = QLabel("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: tok/s \u2014")
        self.avg_tps_label.setObjectName("stats_label")
        self.avg_tflops_label = QLabel("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: TFLOPs \u2014")
        self.avg_tflops_label.setObjectName("stats_label")

        for w in (self.status_label, self.avg_ms_label, self.avg_decode_label, self.avg_tokens_label, self.avg_tps_label, self.avg_tflops_label):
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            side_layout.addWidget(w)

        side_layout.addStretch(1)
        splitter.addWidget(self.stats_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        self.chat_scroll.setMinimumWidth(560)
        self.stats_panel.setMinimumWidth(260)
        splitter.setSizes([780, 300])

        self.input = QPlainTextEdit()
        self.input.setObjectName("chat_input")
        self.input.setPlaceholderText("\u041d\u0430\u043f\u0438\u0448\u0438 \u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0435\u2026")
        self.input.setFont(QFont("Segoe UI", self._text_size))
        self.input.setFixedHeight(120)
        layout.addWidget(self.input)

        bottom = QHBoxLayout()
        layout.addLayout(bottom)

        self.btn_regen = QPushButton("\u0420\u0435\u0433\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u0442\u044c")
        self.btn_regen.setObjectName("btn_regen")
        self.btn_regen.clicked.connect(self.on_regen)
        self.btn_regen.setEnabled(False)
        bottom.addWidget(self.btn_regen)

        bottom.addStretch(1)

        self.btn_send = QPushButton("\u041e\u0442\u043f\u0440\u0430\u0432\u0438\u0442\u044c")
        self.btn_send.setObjectName("btn_send")
        self.btn_send.clicked.connect(self.on_send)
        self.btn_send.setDefault(True)
        bottom.addWidget(self.btn_send)

        self.input.installEventFilter(self)

        self._apply_panel_styles(self._resolve_style())
        self._render_chat()
        self._append_system("MMis UI \u0437\u0430\u043f\u0443\u0449\u0435\u043d. Ctrl+Enter \u2014 \u043e\u0442\u043f\u0440\u0430\u0432\u0438\u0442\u044c.")

    @Slot()
    def _on_live_css_tick(self):
        if self._live_css.load_if_changed():
            self._apply_panel_styles(self._resolve_style())
            self._render_chat()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.input and event.type() == QEvent.KeyPress:
            key_event = event  # type: ignore[assignment]
            if isinstance(key_event, QKeyEvent) and key_event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if key_event.modifiers() & Qt.ControlModifier:
                    self.input.insertPlainText("\n")
                    return True
                if key_event.modifiers() == Qt.NoModifier:
                    self.on_send()
                    return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _px(value: str | None, default: int) -> int:
        if not value:
            return default
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return default
        return int(round(float(match.group(0))))

    @staticmethod
    def _float(value: str | None, default: float) -> float:
        if not value:
            return default
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            return default
        return float(match.group(0))

    @staticmethod
    def _str(value: str | None, default: str) -> str:
        raw = (value or "").strip()
        return raw if raw else default

    @staticmethod
    def _color(value: str | None, fallback: str) -> QColor:
        raw = (value or fallback).strip()
        color = QColor(raw)
        if color.isValid():
            return color

        match = re.match(r"rgba?\(([^)]+)\)", raw, flags=re.IGNORECASE)
        if not match:
            return QColor(fallback)

        parts = [x.strip() for x in match.group(1).split(",")]
        if len(parts) < 3:
            return QColor(fallback)

        try:
            r = int(float(parts[0]))
            g = int(float(parts[1]))
            b = int(float(parts[2]))
            a = 255
            if len(parts) >= 4:
                alpha = float(parts[3])
                a = int(round(alpha * 255)) if alpha <= 1.0 else int(round(alpha))
            return QColor(r, g, b, max(0, min(255, a)))
        except ValueError:
            return QColor(fallback)

    @staticmethod
    def _qss_rgba(color: QColor) -> str:
        return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alphaF():.3f})"

    @staticmethod
    def _font_weight(value: int) -> QFont.Weight:
        if value >= 700:
            return QFont.Weight.Bold
        if value >= 600:
            return QFont.Weight.DemiBold
        if value >= 500:
            return QFont.Weight.Medium
        return QFont.Weight.Normal

    @staticmethod
    def _shadow_from_css(value: str | None) -> tuple[float, float, float, QColor]:
        default = (0.0, 10.0, 18.0, QColor(0, 0, 0, 90))
        if not value:
            return default

        match = re.match(
            r"^\s*(-?\d+(?:\.\d+)?)(?:px)?\s+(-?\d+(?:\.\d+)?)(?:px)?\s+(-?\d+(?:\.\d+)?)(?:px)?(?:\s+-?\d+(?:\.\d+)?(?:px)?)?\s+(.+?)\s*$",
            value.strip(),
            flags=re.IGNORECASE,
        )
        if not match:
            return default

        sx = float(match.group(1))
        sy = float(match.group(2))
        blur = float(match.group(3))
        color = MainWindow._color(match.group(4), "rgba(0, 0, 0, 0.35)")
        return (sx, sy, blur, color)

    def _clear_chat_widgets(self):
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _install_hover_reveal(self, owner: QWidget, target: QWidget) -> None:
        target.setVisible(False)
        filt = _HoverRevealFilter(owner, target)
        owner._hover_reveal_filter = filt  # keep reference alive
        for w in [owner, target, *target.findChildren(QWidget)]:
            w.setMouseTracking(True)
            w.installEventFilter(filt)
        QTimer.singleShot(0, filt._sync_visibility)

    def _install_button_animation(self, button: QPushButton, style: dict | None = None) -> None:
        cfg = style or {}
        fade_ms = int(cfg.get("anim_btn_fade_ms", 170))
        hover_ms = int(cfg.get("anim_btn_hover_ms", 140))
        press_ms = int(cfg.get("anim_btn_press_ms", 90))
        if hasattr(button, "_button_anim_filter"):
            button._button_anim_filter.set_durations(fade_ms, hover_ms, press_ms)
            return
        filt = _ButtonAnimFilter(button, fade_ms=fade_ms, hover_ms=hover_ms, press_ms=press_ms)
        button._button_anim_filter = filt
        button.installEventFilter(filt)

    def _install_base_button_animations(self, style: dict | None = None) -> None:
        for btn in (self.btn_stop, self.btn_clear, self.btn_regen, self.btn_send):
            self._install_button_animation(btn, style=style)

    def _resolve_style(self) -> dict:
        vars_map = self._live_css.get_vars(self._text_size, self._bubble_opacity)

        name_weight_css = self._px(vars_map.get("--name-weight"), 600)
        text_weight_css = self._px(vars_map.get("--text-weight"), 400)

        sx, sy, blur, shadow_color = self._shadow_from_css(vars_map.get("--row-shadow"))

        return {
            "msg_gap_top": self._px(vars_map.get("--msg-gap-top"), 10),
            "msg_gap_bottom": self._px(vars_map.get("--msg-gap-bottom"), 18),
            "msg_radius": self._px(vars_map.get("--msg-radius"), 10),
            "name_size": self._px(vars_map.get("--name-size"), 12),
            "name_weight": self._font_weight(name_weight_css),
            "name_pad_top": self._px(vars_map.get("--name-pad-top"), 8),
            "name_pad_x": self._px(vars_map.get("--name-pad-x"), 10),
            "name_pad_bottom": self._px(vars_map.get("--name-pad-bottom"), 4),
            "text_size": self._px(vars_map.get("--text-size"), 14),
            "text_weight": self._font_weight(text_weight_css),
            "text_weight_css": text_weight_css,
            "text_pad_bottom": self._px(vars_map.get("--text-pad-bottom"), 8),
            "name_color": self._color(vars_map.get("--name-color"), "rgb(245, 245, 245)"),
            "msg_text_color": self._color(vars_map.get("--msg-text-color"), "rgb(245, 245, 245)"),
            "stats_color": self._color(vars_map.get("--stats-color"), "rgba(255, 255, 255, 0.60)"),
            "user_bg": self._color(vars_map.get("--user-bg"), "rgba(70, 131, 180, 0.40)"),
            "ai_bg": self._color(vars_map.get("--ai-bg"), "rgba(220, 80, 80, 0.34)"),
            "shadow_x": sx,
            "shadow_y": sy,
            "shadow_blur": blur,
            "shadow_color": shadow_color,
            "panel_bg": self._color(vars_map.get("--panel-bg"), "rgba(33, 35, 40, 0.85)"),
            "panel_border": self._color(vars_map.get("--panel-border"), "rgba(255, 255, 255, 0.08)"),
            "panel_radius": self._px(vars_map.get("--panel-radius"), 12),
            "panel_padding": self._px(vars_map.get("--panel-padding"), 8),
            "panel_shadow": self._shadow_from_css(vars_map.get("--panel-shadow")),
            "chat_inner_bg": self._color(vars_map.get("--chat-inner-bg"), "rgba(24, 25, 29, 0.92)"),
            "chat_inner_border": self._color(vars_map.get("--chat-inner-border"), "rgba(255, 255, 255, 0.06)"),
            "chat_inner_radius": self._px(vars_map.get("--chat-inner-radius"), 10),
            "chat_inner_padding": self._px(vars_map.get("--chat-inner-padding"), 8),
            "stats_padding": self._px(vars_map.get("--stats-padding"), 12),
            "stats_shadow": self._shadow_from_css(vars_map.get("--stats-shadow")),
            "input_bg": self._color(vars_map.get("--input-bg"), "rgba(33, 35, 40, 0.90)"),
            "input_border": self._color(vars_map.get("--input-border"), "rgba(255, 255, 255, 0.12)"),
            "input_radius": self._px(vars_map.get("--input-radius"), 12),
            "input_padding_y": self._px(vars_map.get("--input-padding-y"), 8),
            "input_padding_x": self._px(vars_map.get("--input-padding-x"), 10),
            "input_shadow": self._shadow_from_css(vars_map.get("--input-shadow")),
            "input_text_size": self._px(vars_map.get("--input-text-size"), self._text_size),
            "input_text_weight": self._px(vars_map.get("--input-text-weight"), 400),
            "app_bg": self._color(vars_map.get("--app-bg"), "rgb(26, 27, 31)"),
            "ui_text": self._color(vars_map.get("--ui-text-color"), "rgb(240, 240, 240)"),
            "muted_text": self._color(vars_map.get("--muted-text-color"), "rgba(255, 255, 255, 0.65)"),
            "model_color": self._color(vars_map.get("--model-color"), "rgb(240, 240, 240)"),
            "model_weight": self._px(vars_map.get("--model-weight"), 700),
            "btn_bg": self._color(vars_map.get("--btn-bg"), "rgba(40, 42, 47, 0.95)"),
            "btn_fg": self._color(vars_map.get("--btn-fg"), "rgb(240, 240, 240)"),
            "btn_border": self._color(vars_map.get("--btn-border"), "rgba(255, 255, 255, 0.15)"),
            "btn_weight": self._px(vars_map.get("--btn-weight"), 400),
            "btn_hover_bg": self._color(vars_map.get("--btn-hover-bg"), "rgba(55, 58, 66, 0.95)"),
            "btn_disabled_bg": self._color(vars_map.get("--btn-disabled-bg"), "rgba(55, 55, 55, 0.40)"),
            "btn_disabled_fg": self._color(vars_map.get("--btn-disabled-fg"), "rgba(210, 210, 210, 0.45)"),
            "send_btn_bg": self._color(vars_map.get("--send-btn-bg"), "rgba(58, 102, 163, 0.95)"),
            "send_btn_hover_bg": self._color(vars_map.get("--send-btn-hover-bg"), "rgba(70, 119, 188, 0.95)"),
            "input_text_color": self._color(vars_map.get("--input-text-color"), "rgb(245, 245, 245)"),
            "input_placeholder_color": self._color(vars_map.get("--input-placeholder-color"), "rgba(255, 255, 255, 0.55)"),
            "stats_text_color": self._color(vars_map.get("--stats-text-color"), "rgb(235, 235, 235)"),
            "side_stats_weight": self._px(vars_map.get("--side-stats-weight"), 400),
            "model_text_shadow": self._shadow_from_css(vars_map.get("--model-text-shadow")),
            "side_text_shadow": self._shadow_from_css(vars_map.get("--side-text-shadow")),
            "name_text_shadow": self._shadow_from_css(vars_map.get("--name-text-shadow")),
            "msg_text_shadow": self._shadow_from_css(vars_map.get("--msg-text-shadow")),
            "msg_stats_text_shadow": self._shadow_from_css(vars_map.get("--msg-stats-text-shadow")),
            "msg_stats_size": self._px(vars_map.get("--msg-stats-size"), 11),
            "msg_stats_pad_top": self._px(vars_map.get("--msg-stats-pad-top"), 0),
            "msg_stats_pad_right": self._px(vars_map.get("--msg-stats-pad-right"), 0),
            "msg_stats_pad_bottom": self._px(vars_map.get("--msg-stats-pad-bottom"), 0),
            "msg_stats_pad_left": self._px(vars_map.get("--msg-stats-pad-left"), 0),
            "msg_stats_align": self._str(vars_map.get("--msg-stats-align"), "left").lower(),
            "feedback_like_bg": self._color(vars_map.get("--feedback-like-bg"), "rgba(46, 125, 50, 0.42)"),
            "feedback_dislike_bg": self._color(vars_map.get("--feedback-dislike-bg"), "rgba(166, 47, 47, 0.42)"),
            "feedback_btn_text": self._color(vars_map.get("--feedback-btn-text"), "rgb(230, 230, 230)"),
            "feedback_btn_border": self._color(vars_map.get("--feedback-btn-border"), "rgba(255, 255, 255, 0.22)"),
            "feedback_btn_bg": self._color(vars_map.get("--feedback-btn-bg"), "rgba(40, 42, 47, 0.90)"),
            "feedback_like_hover_bg": self._color(vars_map.get("--feedback-like-hover-bg"), "rgba(46, 125, 50, 0.30)"),
            "feedback_dislike_hover_bg": self._color(vars_map.get("--feedback-dislike-hover-bg"), "rgba(166, 47, 47, 0.30)"),
            "feedback_btn_radius": self._px(vars_map.get("--feedback-btn-radius"), 6),
            "feedback_btn_pad_y": self._px(vars_map.get("--feedback-btn-pad-y"), 2),
            "feedback_btn_pad_x": self._px(vars_map.get("--feedback-btn-pad-x"), 10),
            "feedback_btn_size": self._px(vars_map.get("--feedback-btn-size"), 11),
            "feedback_btn_weight": self._px(vars_map.get("--feedback-btn-weight"), 500),
            "feedback_row_gap": self._px(vars_map.get("--feedback-row-gap"), 8),
            "feedback_row_margin_top": self._px(vars_map.get("--feedback-row-margin-top"), 0),
            "feedback_row_margin_right": self._px(vars_map.get("--feedback-row-margin-right"), 0),
            "feedback_row_margin_bottom": self._px(vars_map.get("--feedback-row-margin-bottom"), 0),
            "feedback_row_margin_left": self._px(vars_map.get("--feedback-row-margin-left"), 0),
            "feedback_row_shift_y": self._px(vars_map.get("--feedback-row-shift-y"), 0),
            "feedback_overlay_bottom": self._px(vars_map.get("--feedback-overlay-bottom"), 4),
            "feedback_overlay_right": self._px(vars_map.get("--feedback-overlay-right"), 8),
            "feedback_align": self._str(vars_map.get("--feedback-align"), "right").lower(),
            "feedback_order": self._str(vars_map.get("--feedback-order"), "like-first").lower(),
            "anim_btn_fade_ms": self._px(vars_map.get("--anim-btn-fade-ms"), 170),
            "anim_btn_hover_ms": self._px(vars_map.get("--anim-btn-hover-ms"), 140),
            "anim_btn_press_ms": self._px(vars_map.get("--anim-btn-press-ms"), 90),
            "anim_feedback_reveal_show_ms": self._px(vars_map.get("--anim-feedback-reveal-show-ms"), 170),
            "anim_feedback_reveal_hide_ms": self._px(vars_map.get("--anim-feedback-reveal-hide-ms"), 130),
        }

    def _apply_panel_styles(self, style: dict) -> None:
        self._feedback_reveal_show_ms = int(style.get("anim_feedback_reveal_show_ms", 170))
        self._feedback_reveal_hide_ms = int(style.get("anim_feedback_reveal_hide_ms", 130))
        panel_bg = self._qss_rgba(style["panel_bg"])
        panel_border = self._qss_rgba(style["panel_border"])
        panel_radius = style["panel_radius"]

        self.chat_panel.setStyleSheet(
            "QFrame#chat_panel {"
            f"background-color: {panel_bg};"
            f"border: 1px solid {panel_border};"
            f"border-radius: {panel_radius}px;"
            "}"
        )
        self.stats_panel.setStyleSheet(
            "QFrame#stats_panel {"
            f"background-color: {panel_bg};"
            f"border: 1px solid {panel_border};"
            f"border-radius: {panel_radius}px;"
            "}"
        )

        p = style["panel_padding"]
        self.chat_panel.layout().setContentsMargins(p, p, p, p)
        cip = style["chat_inner_padding"]
        self.chat_layout.setContentsMargins(cip, cip, cip, cip)

        s = style["stats_padding"]
        self.stats_panel.layout().setContentsMargins(s, s, s, s)

        self.chat_scroll.setStyleSheet(
            "QScrollArea#chat_scroll { background: transparent; border: none; }"
            f"QScrollArea#chat_scroll QWidget#qt_scrollarea_viewport {{ background: {self._qss_rgba(style['chat_inner_bg'])};"
            f" border: 1px solid {self._qss_rgba(style['chat_inner_border'])};"
            f" border-radius: {style['chat_inner_radius']}px; }}"
            f"QWidget#chat_inner {{ background: transparent; }}"
        )

        self.input.setStyleSheet(
            "QPlainTextEdit {"
            f"background-color: {self._qss_rgba(style['input_bg'])};"
            f"border: 1px solid {self._qss_rgba(style['input_border'])};"
            f"border-radius: {style['input_radius']}px;"
            f"padding: {style['input_padding_y']}px {style['input_padding_x']}px;"
            f"color: {self._qss_rgba(style['input_text_color'])};"
            f"font-size: {style['input_text_size']}px;"
            f"font-weight: {style['input_text_weight']};"
            "}"
            f"QPlainTextEdit#chat_input::placeholder {{ color: {self._qss_rgba(style['input_placeholder_color'])}; }}"
        )

        self.root_widget.setStyleSheet(
            "QWidget#app_root {"
            f"background: {self._qss_rgba(style['app_bg'])};"
            f"color: {self._qss_rgba(style['ui_text'])};"
            "}"
            f"QLabel#model_label {{ color: {self._qss_rgba(style['model_color'])}; font-weight: {style['model_weight']}; }}"
            f"QLabel#stats_label {{ color: {self._qss_rgba(style['stats_text_color'])}; font-weight: {style['side_stats_weight']}; }}"
            "QPushButton {"
            f"background: {self._qss_rgba(style['btn_bg'])};"
            f"color: {self._qss_rgba(style['btn_fg'])};"
            f"border: 1px solid {self._qss_rgba(style['btn_border'])};"
            f"font-weight: {style['btn_weight']};"
            "border-radius: 6px;"
            "padding: 5px 12px;"
            "}"
            f"QPushButton:hover {{ background: {self._qss_rgba(style['btn_hover_bg'])}; }}"
            "QPushButton:disabled {"
            f"background: {self._qss_rgba(style['btn_disabled_bg'])};"
            f"color: {self._qss_rgba(style['btn_disabled_fg'])};"
            f"border: 1px solid {self._qss_rgba(style['btn_border'])};"
            "}"
            f"QPushButton#btn_send {{ background: {self._qss_rgba(style['send_btn_bg'])}; }}"
            f"QPushButton#btn_send:hover {{ background: {self._qss_rgba(style['send_btn_hover_bg'])}; }}"
            "QPushButton#feedback_like, QPushButton#feedback_dislike {"
            f"background: {self._qss_rgba(style['feedback_btn_bg'])};"
            f"color: {self._qss_rgba(style['feedback_btn_text'])};"
            f"border: 1px solid {self._qss_rgba(style['feedback_btn_border'])};"
            f"border-radius: {style['feedback_btn_radius']}px;"
            f"font-size: {style['feedback_btn_size']}px;"
            f"font-weight: {style['feedback_btn_weight']};"
            f"padding: {style['feedback_btn_pad_y']}px {style['feedback_btn_pad_x']}px;"
            "}"
        )
        self._apply_text_shadow(self.model_label, style["model_text_shadow"])
        for w in (self.status_label, self.avg_ms_label, self.avg_decode_label, self.avg_tokens_label, self.avg_tps_label, self.avg_tflops_label):
            self._apply_text_shadow(w, style["side_text_shadow"])
        self._install_base_button_animations(style)

        psx, psy, pblur, pcolor = style["panel_shadow"]
        self._apply_panel_shadow(self.chat_panel, blur=pblur, x=psx, y=psy, color=pcolor)
        ssx, ssy, sblur, scolor = style["stats_shadow"]
        self._apply_panel_shadow(self.stats_panel, blur=sblur, x=ssx, y=ssy, color=scolor)
        isx, isy, iblur, icolor = style["input_shadow"]
        self._apply_panel_shadow(self.input, blur=iblur, x=isx, y=isy, color=icolor)

    def _message_widget(
        self,
        msg_index: int,
        role: str,
        text: str,
        stat_line: str | None,
        feedback: int | None,
        style: dict,
    ) -> QWidget:
        if role == "user":
            name_text = "\u0422\u044b"
            bg_color = style["user_bg"]
        elif role == "ai":
            name_text = "\u041e\u043d\u0430"
            bg_color = style["ai_bg"]
        else:
            name_text = "SYSTEM"
            bg_color = QColor(0, 0, 0, 0)

        outer = _OverlayHost()
        outer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, style["msg_gap_top"], 0, style["msg_gap_bottom"])
        outer_layout.setSpacing(6)

        card = QFrame()
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        card.setObjectName("msg_card")
        card.setStyleSheet(
            f"QFrame#msg_card {{ background-color: {self._qss_rgba(bg_color)}; border-radius: {style['msg_radius']}px; }}"
        )

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(style["name_pad_x"], style["name_pad_top"], style["name_pad_x"], style["text_pad_bottom"])
        card_layout.setSpacing(style["name_pad_bottom"])

        name_font = QFont("Segoe UI", style["name_size"])
        name_font.setWeight(style["name_weight"])

        bubble_font = QFont("Segoe UI", style["text_size"])
        bubble_font.setWeight(style["text_weight"])

        name_label = QLabel(name_text)
        name_label.setTextFormat(Qt.PlainText)
        name_label.setFont(name_font)
        name_label.setStyleSheet(f"color: {self._qss_rgba(style['name_color'])}; background: transparent;")
        name_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._apply_text_shadow(name_label, style["name_text_shadow"])
        card_layout.addWidget(name_label)

        bubble_label = QLabel(text)
        bubble_label.setObjectName("msg_bubble_label")
        bubble_label.setTextFormat(Qt.PlainText)
        bubble_label.setWordWrap(True)
        bubble_label.setFont(bubble_font)
        bubble_label.setStyleSheet(f"color: {self._qss_rgba(style['msg_text_color'])}; background: transparent;")
        bubble_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._apply_text_shadow(bubble_label, style["msg_text_shadow"])
        card_layout.addWidget(bubble_label)

        show_ai_meta = role == "ai" and bool(stat_line and stat_line != "…")

        if role in {"user", "ai"}:
            shadow = QGraphicsDropShadowEffect(card)
            shadow.setBlurRadius(style["shadow_blur"])
            shadow.setOffset(style["shadow_x"], style["shadow_y"])
            shadow.setColor(style["shadow_color"])
            card.setGraphicsEffect(shadow)

        outer_layout.addWidget(card)

        if show_ai_meta:
            stats_label = QLabel(stat_line or "—")
            stats_label.setTextFormat(Qt.PlainText)
            stats_label.setWordWrap(True)
            stats_label.setStyleSheet(
                f"color: {self._qss_rgba(style['stats_color'])};"
                "background: transparent;"
                f"font-size: {style['msg_stats_size']}px;"
                f"padding: {style['msg_stats_pad_top']}px {style['msg_stats_pad_right']}px "
                f"{style['msg_stats_pad_bottom']}px {style['msg_stats_pad_left']}px;"
            )
            align_map = {"left": Qt.AlignLeft, "center": Qt.AlignHCenter, "right": Qt.AlignRight}
            stats_label.setAlignment(align_map.get(style["msg_stats_align"], Qt.AlignLeft))
            stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._apply_text_shadow(stats_label, style["msg_stats_text_shadow"])
            outer_layout.addWidget(stats_label)

            actions = QWidget()
            actions_layout = QHBoxLayout(actions)
            actions_layout.setContentsMargins(
                style["feedback_row_margin_left"],
                style["feedback_row_margin_top"],
                style["feedback_row_margin_right"],
                style["feedback_row_margin_bottom"],
            )
            actions_layout.setSpacing(style["feedback_row_gap"])

            like_btn = QPushButton("👍")
            dislike_btn = QPushButton("👎")
            like_btn.setObjectName("feedback_like")
            dislike_btn.setObjectName("feedback_dislike")
            like_btn.setAttribute(Qt.WA_Hover, True)
            dislike_btn.setAttribute(Qt.WA_Hover, True)
            self._install_button_animation(like_btn, style=style)
            self._install_button_animation(dislike_btn, style=style)
            like_btn._feedback_visual_filter = _FeedbackVisualFilter(
                button=like_btn,
                base_bg=self._qss_rgba(style["feedback_btn_bg"]),
                hover_bg=self._qss_rgba(style["feedback_like_hover_bg"]),
                active_bg=self._qss_rgba(style["feedback_like_bg"]),
                text_color=self._qss_rgba(style["feedback_btn_text"]),
                border_color=self._qss_rgba(style["feedback_btn_border"]),
                radius_px=style["feedback_btn_radius"],
                font_size_px=style["feedback_btn_size"],
                font_weight=style["feedback_btn_weight"],
                pad_y_px=style["feedback_btn_pad_y"],
                pad_x_px=style["feedback_btn_pad_x"],
            )
            dislike_btn._feedback_visual_filter = _FeedbackVisualFilter(
                button=dislike_btn,
                base_bg=self._qss_rgba(style["feedback_btn_bg"]),
                hover_bg=self._qss_rgba(style["feedback_dislike_hover_bg"]),
                active_bg=self._qss_rgba(style["feedback_dislike_bg"]),
                text_color=self._qss_rgba(style["feedback_btn_text"]),
                border_color=self._qss_rgba(style["feedback_btn_border"]),
                radius_px=style["feedback_btn_radius"],
                font_size_px=style["feedback_btn_size"],
                font_weight=style["feedback_btn_weight"],
                pad_y_px=style["feedback_btn_pad_y"],
                pad_x_px=style["feedback_btn_pad_x"],
            )
            like_btn.clicked.connect(lambda _, i=msg_index: self._set_feedback(i, 1))
            dislike_btn.clicked.connect(lambda _, i=msg_index: self._set_feedback(i, -1))

            if feedback == 1:
                like_btn.setText("👍✓")
                like_btn.setProperty("selected", True)
                dislike_btn.setProperty("selected", False)
                like_btn._feedback_visual_filter.refresh()
                dislike_btn._feedback_visual_filter.refresh()
            elif feedback == -1:
                dislike_btn.setText("👎✓")
                like_btn.setProperty("selected", False)
                dislike_btn.setProperty("selected", True)
                like_btn._feedback_visual_filter.refresh()
                dislike_btn._feedback_visual_filter.refresh()
            else:
                like_btn.setProperty("selected", False)
                dislike_btn.setProperty("selected", False)
                like_btn._feedback_visual_filter.refresh()
                dislike_btn._feedback_visual_filter.refresh()

            if style["feedback_order"] == "dislike-first":
                ordered_buttons = [dislike_btn, like_btn]
            else:
                ordered_buttons = [like_btn, dislike_btn]

            align = style["feedback_align"]
            if align == "left":
                for b in ordered_buttons:
                    actions_layout.addWidget(b)
                actions_layout.addStretch(1)
            elif align == "center":
                actions_layout.addStretch(1)
                for b in ordered_buttons:
                    actions_layout.addWidget(b)
                actions_layout.addStretch(1)
            else:
                actions_layout.addStretch(1)
                for b in ordered_buttons:
                    actions_layout.addWidget(b)
            shift_y = max(-6, min(24, int(style.get("feedback_row_shift_y", 0))))
            actions.setStyleSheet("background: transparent;")
            actions.adjustSize()
            outer.set_overlay(
                actions,
                bottom=int(style.get("feedback_overlay_bottom", 4)) - shift_y,
                right=int(style.get("feedback_overlay_right", 8)),
            )
            self._install_hover_reveal(outer, actions)

        return outer

    def _render_chat(self, scroll_to_bottom: bool = False):
        style = self._resolve_style()
        bar = self.chat_scroll.verticalScrollBar()
        prev_value = bar.value()
        was_at_bottom = prev_value >= max(0, bar.maximum() - 4)
        self._clear_chat_widgets()
        for i, (role, text, stat_line, feedback) in enumerate(self._history):
            self.chat_layout.addWidget(self._message_widget(i, role, text, stat_line, feedback, style))
        if scroll_to_bottom or was_at_bottom:
            QTimer.singleShot(
                0,
                lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()),
            )
        else:
            QTimer.singleShot(
                0,
                lambda v=prev_value: self.chat_scroll.verticalScrollBar().setValue(
                    min(v, self.chat_scroll.verticalScrollBar().maximum())
                ),
            )

    def _append_system(self, text: str):
        self._history.append(("system", text, None, None))
        self._render_chat(scroll_to_bottom=True)

    def _append_user(self, text: str):
        self._history.append(("user", text, None, None))
        self._render_chat(scroll_to_bottom=True)

    def _append_ai(self, text: str, stat_line: str):
        self._history.append(("ai", text, stat_line, None))
        self._render_chat(scroll_to_bottom=True)

    def _nearest_user_text_before(self, idx: int) -> str:
        i = idx - 1
        while i >= 0:
            role, text, _, _ = self._history[i]
            if role == "user":
                return text
            i -= 1
        return self._last_user_text or ""

    def _set_feedback(self, msg_index: int, rating: int) -> None:
        if msg_index < 0 or msg_index >= len(self._history):
            return
        role, text, stat_line, current = self._history[msg_index]
        if role != "ai":
            return
        norm_rating = 1 if int(rating) > 0 else -1
        if current == norm_rating:
            return
        self._history[msg_index] = (role, text, stat_line, norm_rating)
        try:
            self.brain.mm.register_assistant_feedback(
                user_text=self._nearest_user_text_before(msg_index),
                assistant_text=text,
                feedback=norm_rating,
                penalty=0.20,
            )
        except Exception:
            pass
        if not self._apply_feedback_to_existing_widget(msg_index, norm_rating):
            self._render_chat(scroll_to_bottom=False)

    def _apply_feedback_to_existing_widget(self, msg_index: int, rating: int) -> bool:
        item = self.chat_layout.itemAt(msg_index)
        row = item.widget() if item else None
        if row is None:
            return False
        buttons = row.findChildren(QPushButton)
        like_btn = None
        dislike_btn = None
        for b in buttons:
            if b.objectName() == "feedback_like":
                like_btn = b
            elif b.objectName() == "feedback_dislike":
                dislike_btn = b
        if like_btn is None or dislike_btn is None:
            return False

        like_btn.setText("👍✓" if rating > 0 else "👍")
        dislike_btn.setText("👎✓" if rating < 0 else "👎")
        like_btn.setProperty("selected", bool(rating > 0))
        dislike_btn.setProperty("selected", bool(rating < 0))
        for b in (like_btn, dislike_btn):
            if hasattr(b, "_feedback_visual_filter"):
                b._feedback_visual_filter.refresh()
        return True

    @Slot(int)
    def _on_text_size_changed(self, value: int):
        self._text_size = int(value)
        if hasattr(self, "text_size_label"):
            self.text_size_label.setText(f"{self._text_size}px")
        self.input.setFont(QFont("Segoe UI", self._text_size))
        self._render_chat()

    @Slot(int)
    def _on_opacity_changed(self, value: int):
        self._bubble_opacity = value / 100.0
        if hasattr(self, "opacity_label"):
            self.opacity_label.setText(f"{value}%")
        self._render_chat()

    def _set_status(self, status: str):
        self.status_label.setText(f"\u0421\u0442\u0430\u0442\u0443\u0441: <b>{status}</b>")

    def _format_stats_line(self, stats: dict) -> tuple[str, float, float, int, int, float]:
        ms = stats.get("answer_ms") or stats.get("ms")
        gen = stats.get("eval_count")
        prompt = stats.get("prompt_eval_count")
        eval_ms = stats.get("eval_duration_ms")

        ms_f = float(ms) if ms is not None else 0.0
        gen_i = int(gen) if gen is not None else 0
        prompt_i = int(prompt) if prompt is not None else 0
        eval_ms_f = float(eval_ms) if eval_ms is not None else 0.0

        sec = eval_ms_f / 1000.0 if eval_ms_f else (ms_f / 1000.0 if ms_f else 0.0)
        tps = safe_div(gen_i, sec)

        parts = []
        if ms is not None:
            parts.append(ms_to_s_text(ms_f))
        if eval_ms is not None:
            parts.append(f"decode {ms_to_s_text(eval_ms_f)}")
        if prompt is not None:
            parts.append(f"prompt {prompt_i}")
        if gen is not None:
            parts.append(f"gen {gen_i}")
        if tps:
            parts.append(f"{tps:.1f} tok/s")
        if SHOW_TFLOPS_EST and tps:
            parts.append(f"~{est_tflops(tps):.2f} TFLOPs (est)")

        return (" \u2022 ".join(parts) if parts else "\u2014", ms_f, eval_ms_f, gen_i, prompt_i, tps)

    def _update_side_stats(self):
        n = max(self.n_answers, 1)
        avg_ms = self.sum_ms / n if self.sum_ms else 0.0
        avg_decode_ms = self.sum_decode_ms / n if self.sum_decode_ms else 0.0
        avg_gen = self.sum_eval / n if self.sum_eval else 0.0
        avg_prompt = self.sum_prompt / n if self.sum_prompt else 0.0
        avg_tps = self.sum_tps / n if self.sum_tps else 0.0
        dash = "\u2014"

        self.avg_ms_label.setText(
            f"\u0421\u0440\u0435\u0434\u043d\u0435\u0435: \u0432\u0440\u0435\u043c\u044f {ms_to_s_text(avg_ms) if avg_ms else dash}"
        )
        self.avg_decode_label.setText(
            f"\u0421\u0440\u0435\u0434\u043d\u0435\u0435: decode {ms_to_s_text(avg_decode_ms) if avg_decode_ms else dash}"
        )
        self.avg_tokens_label.setText(f"\u0421\u0440\u0435\u0434\u043d\u0435\u0435: gen {avg_gen:.0f} / prompt {avg_prompt:.0f}")
        self.avg_tps_label.setText(f"\u0421\u0440\u0435\u0434\u043d\u0435\u0435: tok/s {avg_tps:.1f}" if avg_tps else "\u0421\u0440\u0435\u0434\u043d\u0435\u0435: tok/s \u2014")
        self.avg_tflops_label.setText(
            f"\u0421\u0440\u0435\u0434\u043d\u0435\u0435: TFLOPs ~{est_tflops(avg_tps):.2f}" if (SHOW_TFLOPS_EST and avg_tps) else "\u0421\u0440\u0435\u0434\u043d\u0435\u0435: TFLOPs \u2014"
        )

    @Slot()
    def on_clear(self):
        self._history = []
        self._append_system("\u0427\u0430\u0442 \u043e\u0447\u0438\u0449\u0435\u043d.")
        self._last_user_text = None
        self.btn_regen.setEnabled(False)

        self.n_answers = 0
        self.sum_ms = 0.0
        self.sum_decode_ms = 0.0
        self.sum_eval = 0
        self.sum_prompt = 0
        self.sum_tps = 0.0

        self._set_status("\u0413\u043e\u0442\u043e\u0432\u043e")
        self.avg_ms_label.setText("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: \u0432\u0440\u0435\u043c\u044f \u2014")
        self.avg_decode_label.setText("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: decode \u2014")
        self.avg_tokens_label.setText("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: gen \u2014 / prompt \u2014")
        self.avg_tps_label.setText("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: tok/s \u2014")
        self.avg_tflops_label.setText("\u0421\u0440\u0435\u0434\u043d\u0435\u0435: TFLOPs \u2014")

    @Slot()
    def on_stop(self):
        if self._worker:
            self._worker.request_cancel()
        self._set_status("\u041e\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430\u2026 (\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0431\u0443\u0434\u0435\u0442 \u043f\u0440\u043e\u0438\u0433\u043d\u043e\u0440\u0438\u0440\u043e\u0432\u0430\u043d)")

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

        started = self._start_request(text, show_user=True)
        if not started:
            return

        self.input.clear()
        self._last_user_text = text

    def _start_request(self, user_text: str, show_user: bool) -> bool:
        if self._thread and self._thread.isRunning():
            QMessageBox.information(
                self,
                "\u041f\u043e\u0434\u043e\u0436\u0434\u0438",
                "\u0421\u0435\u0439\u0447\u0430\u0441 \u0443\u0436\u0435 \u0438\u0434\u0451\u0442 \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f. \u041d\u0430\u0436\u043c\u0438 '\u0421\u0442\u043e\u043f' \u0438\u043b\u0438 \u0434\u043e\u0436\u0434\u0438\u0441\u044c \u043e\u0442\u0432\u0435\u0442\u0430.",
            )
            return False

        if show_user:
            self._append_user(user_text)
        self._append_ai("", "…")
        self._stream_ai_index = len(self._history) - 1
        self._stream_chunk_buffer = ""
        self._stream_flush_timer.stop()

        self._set_status("\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f\u2026")
        self.btn_send.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_regen.setEnabled(False)

        self._thread = QThread()
        self._worker = ReplyWorker(self.brain, user_text=user_text)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.chunk.connect(self._on_reply_chunk)
        self._worker.finished.connect(self._on_reply)
        self._worker.errored.connect(self._on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.errored.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()
        return True

    @Slot(str)
    def _on_reply_chunk(self, piece: str):
        self._stream_chunk_buffer += piece or ""
        if not self._stream_flush_timer.isActive():
            self._stream_flush_timer.start()

    def _flush_stream_chunks(self):
        if not self._stream_chunk_buffer or self._stream_ai_index is None:
            self._stream_chunk_buffer = ""
            return
        idx = self._stream_ai_index
        if idx < 0 or idx >= len(self._history):
            self._stream_chunk_buffer = ""
            return
        role, text, stat_line, feedback = self._history[idx]
        if role != "ai":
            self._stream_chunk_buffer = ""
            return
        new_text = (text or "") + self._stream_chunk_buffer
        self._history[idx] = (role, new_text, stat_line, feedback)
        self._stream_chunk_buffer = ""

        # Fast path: update only current streaming label to avoid full rerender flicker.
        try:
            item = self.chat_layout.itemAt(idx)
            w = item.widget() if item else None
            bubble = w.findChild(QLabel, "msg_bubble_label") if w else None
            if bubble is not None:
                bubble.setText(new_text)
                self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum())
                return
        except Exception:
            pass

        self._render_chat(scroll_to_bottom=True)

    @Slot(object)
    def _on_reply(self, res: ReplyResult):
        self._flush_stream_chunks()
        stats = res.stats or {}
        stat_line, ms, decode_ms, gen, prompt, tps = self._format_stats_line(stats)
        if self._stream_ai_index is not None and 0 <= self._stream_ai_index < len(self._history):
            i = self._stream_ai_index
            role, text, _, feedback = self._history[i]
            final_text = res.text or text
            self._history[i] = (role, final_text, stat_line, feedback)
            self._render_chat(scroll_to_bottom=True)
            self._stream_ai_index = None
        else:
            self._append_ai(res.text, stat_line)

        if ms:
            self.sum_ms += ms
        if decode_ms:
            self.sum_decode_ms += decode_ms
        self.sum_eval += gen
        self.sum_prompt += prompt
        self.sum_tps += tps
        self.n_answers += 1

        self._update_side_stats()
        self._set_status("\u0413\u043e\u0442\u043e\u0432\u043e")
        self.btn_regen.setEnabled(bool(self._last_user_text))

    @Slot(str)
    def _on_error(self, tb: str):
        self._set_status("\u041e\u0448\u0438\u0431\u043a\u0430")
        self._stream_ai_index = None
        self._stream_chunk_buffer = ""
        self._stream_flush_timer.stop()
        QMessageBox.critical(self, "\u041e\u0448\u0438\u0431\u043a\u0430", tb)

    @Slot()
    def _cleanup_thread(self):
        self._stream_flush_timer.stop()
        self._stream_chunk_buffer = ""
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_regen.setEnabled(bool(self._last_user_text))
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
