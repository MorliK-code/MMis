"""MMis Desktop UI (PySide6)."""

from __future__ import annotations

import re
import sys
import traceback
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QEvent, QPropertyAnimation, QObject, QSize, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QCursor, QFont, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
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

    def __init__(self, brain: Brain, user_text: str, store_turn: bool = True):
        super().__init__()
        self.brain = brain
        self.user_text = user_text
        self.store_turn = bool(store_turn)
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    @Slot()
    def run(self):
        try:
            answer = self.brain.think_stream(self.user_text, on_chunk=self.chunk.emit, store_turn=self.store_turn)
            stats = getattr(self.brain, "last_stats", {}) or {}
            if self._cancel_requested:
                return
            self.finished.emit(ReplyResult(text=answer, stats=stats))
        except Exception:
            self.errored.emit(traceback.format_exc())


class _HoverRevealFilter(QObject):
    def __init__(
        self,
        owner: QWidget,
        target: QWidget,
        show_ms: int = 170,
        hide_ms: int = 130,
        require_reenter: bool = False,
    ):
        super().__init__(owner)
        self.owner = owner
        self.target = target
        self.show_ms = max(0, int(show_ms))
        self.hide_ms = max(40, int(hide_ms))
        self._must_leave_once = bool(require_reenter)
        self._pressed_inside_target = False
        self._outside_streak = 0
        target.setVisible(False)
        self._sync_timer = QTimer(target)
        self._sync_timer.setInterval(60)
        self._sync_timer.timeout.connect(self._sync_visibility)
        self._sync_timer.start()

    def _contains_global(self, w: QWidget, global_pos, pad: int = 2) -> bool:
        if not isValid(w):
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
        inside = self._is_cursor_inside_owner_or_target()
        if self._must_leave_once:
            if inside:
                self._outside_streak = 0
                self.target.setVisible(False)
                return
            self._must_leave_once = False
        if inside:
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


class _ToggleSwitch(QCheckBox):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self._track_off = QColor(95, 95, 105, 180)
        self._track_on = QColor(58, 102, 163, 240)
        self._track_border = QColor(255, 255, 255, 45)
        self._knob = QColor(240, 240, 240)
        self._text_color = QColor(240, 240, 240)
        self._offset = 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(24)
        self.toggled.connect(self._on_toggled)

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text())
        return QSize(38 + 8 + text_w + 6, 24)

    def set_colors(self, track_off: QColor, track_on: QColor, track_border: QColor, text_color: QColor) -> None:
        self._track_off = QColor(track_off)
        self._track_on = QColor(track_on)
        self._track_border = QColor(track_border)
        self._text_color = QColor(text_color)
        self.update()

    def _on_toggled(self, checked: bool) -> None:
        self._offset = 1.0 if checked else 0.0
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        h = max(18, self.height() - 4)
        w = 38
        x = 0
        y = int((self.height() - h) / 2)
        r = h / 2.0

        track_rect = self.rect().adjusted(x, y, -(self.width() - (x + w)), -(self.height() - (y + h)))
        track_color = self._track_on if self.isChecked() else self._track_off
        p.setPen(QPen(self._track_border, 1))
        p.setBrush(track_color)
        p.drawRoundedRect(track_rect, r, r)

        knob_d = h - 4
        knob_y = y + 2
        knob_min_x = x + 2
        knob_max_x = x + w - knob_d - 2
        knob_x = knob_max_x if self.isChecked() else knob_min_x
        p.setPen(Qt.NoPen)
        p.setBrush(self._knob)
        p.drawEllipse(knob_x, knob_y, knob_d, knob_d)

        p.setPen(self._text_color)
        p.drawText(w + 8, 0, self.width() - (w + 8), self.height(), Qt.AlignVCenter | Qt.AlignLeft, self.text())

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
        self._sessions_dir = MemoryStorageDir / "ui_chats"
        self._sessions_index_path = self._sessions_dir / "index.json"
        self._legacy_sessions_path = MemoryStorageDir / "ui_chats.json"
        self._chat_sessions: list[dict] = []
        self._active_chat_id: str | None = None
        self._updating_chat_controls = False
        self._chats_open_width = 240
        self._stats_open_width = 300
        self._chat_min_width = 560
        self._chats_drawer_open = False
        self._stats_drawer_open = False
        self._chats_drawer_anim: QPropertyAnimation | None = None
        self._stats_drawer_anim: QPropertyAnimation | None = None
        self._settings_chat_id: str | None = None

        self._build_ui()
        self._load_or_init_chat_sessions()

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

        self.btn_toggle_chats = QPushButton("Чаты")
        self.btn_toggle_chats.setObjectName("btn_toggle_chats")
        self.btn_toggle_chats.clicked.connect(self._toggle_chats_drawer)
        top.addWidget(self.btn_toggle_chats)

        self.btn_toggle_stats = QPushButton("Статистика")
        self.btn_toggle_stats.setObjectName("btn_toggle_stats")
        self.btn_toggle_stats.clicked.connect(self._toggle_stats_drawer)
        top.addWidget(self.btn_toggle_stats)

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
        splitter.setHandleWidth(1)
        layout.addWidget(splitter, 1)

        self.chats_panel = QFrame()
        self.chats_panel.setObjectName("chats_panel")
        chats_layout = QVBoxLayout(self.chats_panel)
        chats_layout.setContentsMargins(8, 8, 8, 8)
        chats_layout.setSpacing(8)

        chats_title = QLabel("Чаты")
        chats_title.setObjectName("stats_label")
        chats_layout.addWidget(chats_title)

        self.chat_list = QListWidget()
        self.chat_list.setObjectName("chat_list")
        self.chat_list.currentRowChanged.connect(self._on_chat_selected)
        chats_layout.addWidget(self.chat_list, 1)

        self.chat_settings_panel = QFrame()
        self.chat_settings_panel.setObjectName("chat_settings_panel")
        settings_layout = QVBoxLayout(self.chat_settings_panel)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(8)
        self.chat_settings_title = QLabel("Настройки чата")
        self.chat_settings_title.setObjectName("chat_settings_title")
        settings_layout.addWidget(self.chat_settings_title)

        self.chat_settings_incognito = _ToggleSwitch("Инкогнито")
        self.chat_settings_incognito.setObjectName("chat_settings_incognito")
        self.chat_settings_incognito.toggled.connect(self._on_settings_incognito_toggled)
        settings_layout.addWidget(self.chat_settings_incognito)

        settings_actions = QHBoxLayout()
        settings_actions.setContentsMargins(0, 0, 0, 0)
        settings_actions.setSpacing(6)
        self.chat_settings_export = QPushButton("Экспорт")
        self.chat_settings_export.setObjectName("chat_settings_export")
        self.chat_settings_export.clicked.connect(self._on_settings_export_clicked)
        settings_actions.addWidget(self.chat_settings_export)
        self.chat_settings_delete = QPushButton("Удалить")
        self.chat_settings_delete.setObjectName("chat_settings_delete")
        self.chat_settings_delete.clicked.connect(self._on_settings_delete_clicked)
        settings_actions.addWidget(self.chat_settings_delete)
        settings_layout.addLayout(settings_actions)

        self.chat_settings_close = QPushButton("Закрыть")
        self.chat_settings_close.setObjectName("chat_settings_close")
        self.chat_settings_close.clicked.connect(self._close_chat_settings_panel)
        settings_layout.addWidget(self.chat_settings_close)
        self.chat_settings_panel.setVisible(False)
        chats_layout.addWidget(self.chat_settings_panel, 0)

        chats_actions = QHBoxLayout()
        self.btn_new_chat = QPushButton("Новый")
        self.btn_new_chat.setObjectName("btn_new_chat")
        self.btn_new_chat.clicked.connect(self._on_new_chat)
        chats_actions.addWidget(self.btn_new_chat)
        chats_layout.addLayout(chats_actions)
        splitter.addWidget(self.chats_panel)

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
        self.chat_layout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)
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
        self.splitter = splitter
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        self.chats_panel.setMinimumWidth(0)
        self.chats_panel.setMaximumWidth(self._chats_open_width)
        self.chat_scroll.setMinimumWidth(self._chat_min_width)
        self.stats_panel.setMinimumWidth(0)
        self.stats_panel.setMaximumWidth(self._stats_open_width)
        splitter.setSizes([0, 980, 0])
        for i in (1, 2):
            handle = splitter.handle(i)
            if handle:
                handle.setEnabled(False)
                handle.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._set_chats_drawer_open(False, animated=False)
        self._set_stats_drawer_open(False, animated=False)

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

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _history_to_serializable(history: list[tuple[str, str, str | None, int | None]]) -> list[dict]:
        out: list[dict] = []
        for role, text, stat_line, feedback in history:
            out.append(
                {
                    "role": str(role),
                    "text": str(text or ""),
                    "stat_line": None if stat_line is None else str(stat_line),
                    "feedback": None if feedback is None else int(feedback),
                }
            )
        return out

    @staticmethod
    def _history_from_serializable(rows: list[dict] | None) -> list[tuple[str, str, str | None, int | None]]:
        out: list[tuple[str, str, str | None, int | None]] = []
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "")
            if role not in {"system", "user", "ai"}:
                continue
            text = str(row.get("text") or "")
            stat_line_raw = row.get("stat_line")
            stat_line = None if stat_line_raw is None else str(stat_line_raw)
            feedback_raw = row.get("feedback")
            feedback = None
            if feedback_raw is not None:
                try:
                    feedback = int(feedback_raw)
                except Exception:
                    feedback = None
            out.append((role, text, stat_line, feedback))
        return out

    def _new_chat_payload(self, title: str | None = None, incognito: bool = False) -> dict:
        ts = self._now_iso()
        next_num = len(self._chat_sessions) + 1
        return {
            "id": f"chat-{int(datetime.now().timestamp() * 1000)}-{next_num}",
            "title": title or f"Чат {next_num}",
            "incognito": bool(incognito),
            "created_at": ts,
            "updated_at": ts,
            "history": [("system", "MMis UI запущен. Ctrl+Enter — отправить.", None, None)],
        }

    def _chat_dir(self, chat_id: str) -> Path:
        return self._sessions_dir / str(chat_id)

    def _chat_payload_path(self, chat_id: str) -> Path:
        return self._chat_dir(chat_id) / "chat.json"

    def _load_legacy_sessions(self) -> tuple[list[dict], str | None]:
        loaded: list[dict] = []
        active_id: str | None = None
        if not self._legacy_sessions_path.exists():
            return loaded, active_id
        try:
            payload = json.loads(self._legacy_sessions_path.read_text(encoding="utf-8-sig"))
            rows = payload.get("chats", []) if isinstance(payload, dict) else []
            active_id_raw = payload.get("active_chat_id") if isinstance(payload, dict) else None
            active_id = str(active_id_raw) if active_id_raw else None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    chat_id = str(row.get("id") or "").strip()
                    if not chat_id:
                        continue
                    loaded.append(
                        {
                            "id": chat_id,
                            "title": str(row.get("title") or "Чат"),
                            "incognito": bool(row.get("incognito", False)),
                            "created_at": str(row.get("created_at") or self._now_iso()),
                            "updated_at": str(row.get("updated_at") or self._now_iso()),
                            "history": self._history_from_serializable(row.get("history")),
                        }
                    )
            loaded = [c for c in loaded if not bool(c.get("incognito", False))]
            if active_id and not any(str(c.get("id")) == active_id for c in loaded):
                active_id = None
        except Exception:
            loaded = []
            active_id = None
        return loaded, active_id

    def _save_chat_sessions(self) -> None:
        self._sync_active_session_from_history()
        try:
            self._sessions_dir.mkdir(parents=True, exist_ok=True)
            persisted_ids: set[str] = set()
            persisted_meta: list[dict] = []

            for chat in self._chat_sessions:
                if bool(chat.get("incognito", False)):
                    continue
                chat_id = str(chat.get("id") or "").strip()
                if not chat_id:
                    continue
                persisted_ids.add(chat_id)
                payload = {
                    "id": chat_id,
                    "title": str(chat.get("title") or "Чат"),
                    "incognito": False,
                    "created_at": str(chat.get("created_at") or self._now_iso()),
                    "updated_at": str(chat.get("updated_at") or self._now_iso()),
                    "history": self._history_to_serializable(chat.get("history") or []),
                }
                chat_path = self._chat_payload_path(chat_id)
                chat_path.parent.mkdir(parents=True, exist_ok=True)
                chat_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                persisted_meta.append(
                    {
                        "id": payload["id"],
                        "title": payload["title"],
                        "created_at": payload["created_at"],
                        "updated_at": payload["updated_at"],
                    }
                )

            for child in self._sessions_dir.iterdir():
                if not child.is_dir():
                    continue
                if child.name in persisted_ids:
                    continue
                shutil.rmtree(child, ignore_errors=True)

            active_chat = self._active_chat()
            active_id = None
            if active_chat and not bool(active_chat.get("incognito", False)):
                active_id = str(active_chat.get("id") or "") or None
            index_payload = {"version": 2, "active_chat_id": active_id, "chats": persisted_meta}
            self._sessions_index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_or_init_chat_sessions(self) -> None:
        loaded: list[dict] = []
        active_id: str | None = None
        if self._sessions_index_path.exists():
            try:
                payload = json.loads(self._sessions_index_path.read_text(encoding="utf-8-sig"))
                rows = payload.get("chats", []) if isinstance(payload, dict) else []
                active_id_raw = payload.get("active_chat_id") if isinstance(payload, dict) else None
                active_id = str(active_id_raw) if active_id_raw else None
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        chat_id = str(row.get("id") or "").strip()
                        if not chat_id:
                            continue
                        chat_payload_path = self._chat_payload_path(chat_id)
                        if not chat_payload_path.exists():
                            continue
                        chat_payload = json.loads(chat_payload_path.read_text(encoding="utf-8-sig"))
                        loaded.append(
                            {
                                "id": chat_id,
                                "title": str(chat_payload.get("title") or row.get("title") or "Чат"),
                                "incognito": False,
                                "created_at": str(chat_payload.get("created_at") or row.get("created_at") or self._now_iso()),
                                "updated_at": str(chat_payload.get("updated_at") or row.get("updated_at") or self._now_iso()),
                                "history": self._history_from_serializable(chat_payload.get("history")),
                            }
                        )
            except Exception:
                loaded = []

        if not loaded:
            legacy_loaded, legacy_active = self._load_legacy_sessions()
            loaded = legacy_loaded
            active_id = legacy_active

        self._chat_sessions = loaded
        if loaded:
            self._active_chat_id = active_id if any(c.get("id") == active_id for c in loaded) else str(loaded[0]["id"])
        else:
            self._active_chat_id = None
        self._refresh_chat_selector()
        self._apply_active_chat_to_ui(scroll_to_bottom=True)
        self._save_chat_sessions()

    def _active_chat(self) -> dict | None:
        if not self._active_chat_id:
            return None
        return self._chat_by_id(self._active_chat_id)

    def _chat_by_id(self, chat_id: str | None) -> dict | None:
        if not chat_id:
            return None
        cid = str(chat_id)
        for chat in self._chat_sessions:
            if str(chat.get("id")) == cid:
                return chat
        return None

    def _sync_active_session_from_history(self) -> None:
        chat = self._active_chat()
        if not chat:
            return
        chat["history"] = list(self._history)
        chat["updated_at"] = self._now_iso()

    def _refresh_chat_selector(self) -> None:
        if not hasattr(self, "chat_list"):
            return
        self._updating_chat_controls = True
        try:
            self.chat_list.clear()
            for chat in self._chat_sessions:
                title = str(chat.get("title") or "Чат")
                if bool(chat.get("incognito", False)):
                    title = f"{title} (инкогнито)"
                item = QListWidgetItem(self.chat_list)
                item.setSizeHint(QSize(180, 30))
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(6, 2, 2, 2)
                row_layout.setSpacing(6)

                title_label = QLabel(title)
                title_label.setObjectName("chat_title")
                title_label.setTextInteractionFlags(Qt.NoTextInteraction)
                row_layout.addWidget(title_label, 1)

                dots_btn = QToolButton()
                dots_btn.setObjectName("chat_item_menu_btn")
                dots_btn.setText("⋯")
                chat_id = str(chat.get("id") or "")
                dots_btn.clicked.connect(lambda _=False, cid=chat_id, b=dots_btn: self._show_chat_item_menu(cid, b))
                row_layout.addWidget(dots_btn, 0)
                self.chat_list.setItemWidget(item, row)

            idx = 0
            for i, chat in enumerate(self._chat_sessions):
                if str(chat.get("id")) == self._active_chat_id:
                    idx = i
                    break
            if self._chat_sessions:
                self.chat_list.setCurrentRow(idx)
            else:
                self.chat_list.setCurrentRow(-1)
        finally:
            self._updating_chat_controls = False
        self._update_chat_controls_state()

    def _apply_active_chat_to_ui(self, scroll_to_bottom: bool = False) -> None:
        chat = self._active_chat()
        if not chat:
            self._close_chat_settings_panel()
            self._history = []
            self._stream_ai_index = None
            self._stream_chunk_buffer = ""
            self._last_user_text = None
            self._set_status("Готово")
            self._render_chat(scroll_to_bottom=scroll_to_bottom)
            self._update_chat_controls_state()
            return
        self._history = list(chat.get("history") or [])
        if not self._history:
            self._history = [("system", "MMis UI запущен. Ctrl+Enter — отправить.", None, None)]
            chat["history"] = list(self._history)
        self._stream_ai_index = None
        self._stream_chunk_buffer = ""
        self._last_user_text = None
        if self._settings_chat_id and self._settings_chat_id != str(chat.get("id") or ""):
            self._close_chat_settings_panel()
        for role, text, _, _ in reversed(self._history):
            if role == "user":
                self._last_user_text = text
                break
        self._set_status("Готово")
        self._render_chat(scroll_to_bottom=scroll_to_bottom)
        self._update_chat_controls_state()

    def _update_chat_controls_state(self) -> None:
        has_active = self._active_chat() is not None
        self.btn_new_chat.setEnabled(True)
        self.btn_send.setEnabled(has_active and not (self._thread and self._thread.isRunning()))
        self.chat_settings_export.setEnabled(has_active)
        self.chat_settings_delete.setEnabled(has_active)
        self.chat_settings_incognito.setEnabled(has_active)

    def _close_chat_settings_panel(self) -> None:
        self._settings_chat_id = None
        self.chat_settings_panel.setVisible(False)

    def _show_chat_item_menu(self, chat_id: str, anchor: QWidget) -> None:
        chat = self._chat_by_id(chat_id)
        if not chat:
            return
        if chat_id != self._active_chat_id:
            self._sync_active_session_from_history()
            self._active_chat_id = chat_id
            self._apply_active_chat_to_ui(scroll_to_bottom=True)
            self._save_chat_sessions()
            self._refresh_chat_selector()

        menu = QMenu(self)
        settings_action = menu.addAction("Настройки")
        chosen = menu.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        if chosen == settings_action:
            self._open_chat_settings(chat_id)

    def _open_chat_settings(self, chat_id: str) -> None:
        chat = self._chat_by_id(chat_id)
        if not chat:
            return
        self._settings_chat_id = chat_id
        self.chat_settings_title.setText(f"Настройки: {str(chat.get('title') or 'Чат')}")
        self.chat_settings_incognito.blockSignals(True)
        self.chat_settings_incognito.setChecked(bool(chat.get("incognito", False)))
        self.chat_settings_incognito.blockSignals(False)
        s = self._resolve_style()
        self.chat_settings_incognito.set_colors(
            track_off=s["btn_disabled_bg"],
            track_on=s["send_btn_bg"],
            track_border=s["btn_border"],
            text_color=s["ui_text"],
        )
        self.chat_settings_panel.setVisible(True)

    @Slot(bool)
    def _on_settings_incognito_toggled(self, checked: bool) -> None:
        if not self._settings_chat_id:
            return
        self._set_chat_incognito(self._settings_chat_id, checked)

    @Slot()
    def _on_settings_export_clicked(self) -> None:
        if not self._settings_chat_id:
            return
        self._export_chat_by_id(self._settings_chat_id)

    @Slot()
    def _on_settings_delete_clicked(self) -> None:
        if not self._settings_chat_id:
            return
        self._delete_chat_by_id(self._settings_chat_id)

    @Slot(int)
    def _on_chat_selected(self, index: int) -> None:
        if self._updating_chat_controls:
            return
        if index < 0 or index >= len(self._chat_sessions):
            return
        if self._thread and self._thread.isRunning():
            self._updating_chat_controls = True
            try:
                self._refresh_chat_selector()
            finally:
                self._updating_chat_controls = False
            QMessageBox.information(self, "Подожди", "Сначала останови или дождись завершения генерации.")
            return
        new_id = str(self._chat_sessions[index].get("id") or "")
        if not new_id or new_id == self._active_chat_id:
            return
        self._sync_active_session_from_history()
        self._active_chat_id = new_id
        self._apply_active_chat_to_ui(scroll_to_bottom=True)
        self._save_chat_sessions()
        if self._chats_drawer_open:
            self._set_chats_drawer_open(False, animated=True)

    @Slot()
    def _on_new_chat(self) -> None:
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "Подожди", "Сначала останови или дождись завершения генерации.")
            return
        self._sync_active_session_from_history()
        chat = self._new_chat_payload(incognito=False)
        self._chat_sessions.append(chat)
        self._active_chat_id = str(chat["id"])
        self._refresh_chat_selector()
        self._apply_active_chat_to_ui(scroll_to_bottom=True)
        self._save_chat_sessions()

    def _set_chat_incognito(self, chat_id: str, checked: bool) -> None:
        chat = self._chat_by_id(chat_id)
        if not chat:
            return
        chat["incognito"] = bool(checked)
        chat["updated_at"] = self._now_iso()
        if self._settings_chat_id == chat_id and self.chat_settings_incognito.isChecked() != bool(checked):
            self.chat_settings_incognito.blockSignals(True)
            self.chat_settings_incognito.setChecked(bool(checked))
            self.chat_settings_incognito.blockSignals(False)
        self._refresh_chat_selector()
        if str(chat.get("id") or "") == self._active_chat_id:
            self._apply_active_chat_to_ui(scroll_to_bottom=False)
        self._save_chat_sessions()

    def _delete_chat_by_id(self, chat_id: str) -> None:
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "Подожди", "Сначала останови или дождись завершения генерации.")
            return
        chat = self._chat_by_id(chat_id)
        if not chat:
            return
        title = str(chat.get("title") or "Чат")
        answer = QMessageBox.question(
            self,
            "Удалить чат",
            f"Удалить чат «{title}»?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self._chat_sessions = [c for c in self._chat_sessions if str(c.get("id") or "") != chat_id]
        shutil.rmtree(self._chat_dir(chat_id), ignore_errors=True)

        if self._chat_sessions and self._active_chat_id == chat_id:
            self._active_chat_id = str(self._chat_sessions[0].get("id") or "")
        elif not self._chat_sessions:
            self._active_chat_id = None
        self._refresh_chat_selector()
        self._apply_active_chat_to_ui(scroll_to_bottom=True)
        self._save_chat_sessions()
        if self._settings_chat_id == chat_id:
            self._close_chat_settings_panel()

    def _export_chat_by_id(self, chat_id: str) -> None:
        chat = self._chat_by_id(chat_id)
        if not chat:
            return
        self._sync_active_session_from_history()
        export_dir = MemoryStorageDir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{str(chat.get('id') or 'chat')}_{stamp}.json"
        path = export_dir / filename
        payload = {
            "id": chat.get("id"),
            "title": chat.get("title"),
            "incognito": bool(chat.get("incognito", False)),
            "created_at": chat.get("created_at"),
            "updated_at": chat.get("updated_at"),
            "history": self._history_to_serializable(chat.get("history") or []),
        }
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            QMessageBox.information(self, "Экспорт", f"Чат экспортирован:\n{path}")
        except Exception:
            QMessageBox.warning(self, "Экспорт", "Не удалось экспортировать чат.")

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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_chat_content_geometry)

    @Slot()
    def _toggle_chats_drawer(self) -> None:
        self._set_chats_drawer_open(not self._chats_drawer_open, animated=True)

    def _set_chats_drawer_open(self, is_open: bool, animated: bool = True) -> None:
        target = int(self._chats_open_width if is_open else 0)
        self._chats_drawer_open = bool(is_open)
        self.btn_toggle_chats.setText("Скрыть чаты" if is_open else "Чаты")
        if not animated:
            self.chats_panel.setMinimumWidth(target)
            self.chats_panel.setMaximumWidth(target)
            self.chats_panel.setVisible(target > 0)
            return

        if self._chats_drawer_anim and self._chats_drawer_anim.state() == QAbstractAnimation.Running:
            self._chats_drawer_anim.stop()
        if target > 0:
            self.chats_panel.setVisible(True)
        anim = QPropertyAnimation(self.chats_panel, b"maximumWidth", self)
        anim.setDuration(180)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(int(self.chats_panel.maximumWidth()))
        anim.setEndValue(target)

        def _finish():
            self.chats_panel.setMinimumWidth(target)
            self.chats_panel.setVisible(target > 0)

        anim.finished.connect(_finish)
        self._chats_drawer_anim = anim
        anim.start()

    @Slot()
    def _toggle_stats_drawer(self) -> None:
        self._set_stats_drawer_open(not self._stats_drawer_open, animated=True)

    def _set_stats_drawer_open(self, is_open: bool, animated: bool = True) -> None:
        target = int(self._stats_open_width if is_open else 0)
        self._stats_drawer_open = bool(is_open)
        self.btn_toggle_stats.setText("Скрыть статистику" if is_open else "Статистика")
        if not animated:
            self.stats_panel.setMinimumWidth(target)
            self.stats_panel.setMaximumWidth(target)
            self.stats_panel.setVisible(target > 0)
            return

        if self._stats_drawer_anim and self._stats_drawer_anim.state() == QAbstractAnimation.Running:
            self._stats_drawer_anim.stop()
        if target > 0:
            self.stats_panel.setVisible(True)
        anim = QPropertyAnimation(self.stats_panel, b"maximumWidth", self)
        anim.setDuration(180)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(int(self.stats_panel.maximumWidth()))
        anim.setEndValue(target)

        def _finish():
            self.stats_panel.setMinimumWidth(target)
            self.stats_panel.setVisible(target > 0)

        anim.finished.connect(_finish)
        self._stats_drawer_anim = anim
        anim.start()

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
    def _normalize_stream_text(text: str) -> str:
        s = str(text or "")
        if not s:
            return ""
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # Prevent stream artifacts from inflating bubble height.
        s = re.sub(r"\n{2,}", "\n", s)
        s = re.sub(r"[ \t\f\v]+", " ", s)
        s = re.sub(r"\s*\n\s*", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    @staticmethod
    def _is_pending_stat_line(stat_line: str | None) -> bool:
        s = str(stat_line or "").strip()
        if not s:
            return True
        if re.fullmatch(r"[.\?…]{1,3}", s):
            return True
        return False

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

    def _sync_chat_content_geometry(self) -> None:
        layout = self.chat_layout
        if layout is None:
            return
        layout.invalidate()
        layout.activate()
        content_h = max(1, int(layout.sizeHint().height()))
        # Keep content height strictly equal to rendered rows to avoid
        # phantom scroll space caused by stale minimum-height states.
        self.chat_root.setMinimumHeight(0)
        self.chat_root.setMaximumHeight(16777215)
        self.chat_root.setFixedHeight(content_h)
        self.chat_root.updateGeometry()
        self.chat_root.adjustSize()
        bar = self.chat_scroll.verticalScrollBar()
        if bar.value() > bar.maximum():
            bar.setValue(bar.maximum())

    def _install_hover_reveal(self, owner: QWidget, target: QWidget, require_reenter: bool = False) -> None:
        target.setVisible(False)
        filt = _HoverRevealFilter(
            owner,
            target,
            show_ms=self._feedback_reveal_show_ms,
            hide_ms=self._feedback_reveal_hide_ms,
            require_reenter=require_reenter,
        )
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
        for btn in (
            self.btn_toggle_chats,
            self.btn_toggle_stats,
            self.btn_stop,
            self.btn_clear,
            self.btn_new_chat,
            self.chat_settings_export,
            self.chat_settings_delete,
            self.chat_settings_close,
            self.btn_regen,
            self.btn_send,
        ):
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
            "chats_drawer_width": self._px(vars_map.get("--chats-drawer-width"), 240),
            "stats_drawer_width": self._px(vars_map.get("--stats-drawer-width"), 300),
            "chat_min_width": self._px(vars_map.get("--chat-min-width"), 560),
            "chat_settings_bg": self._color(vars_map.get("--chat-settings-bg"), "rgba(20, 21, 25, 0.75)"),
            "chat_settings_border": self._color(vars_map.get("--chat-settings-border"), "rgba(255, 255, 255, 0.10)"),
            "chat_settings_radius": self._px(vars_map.get("--chat-settings-radius"), 10),
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
        self._chats_open_width = max(170, int(style.get("chats_drawer_width", 240)))
        self._stats_open_width = max(220, int(style.get("stats_drawer_width", 300)))
        self._chat_min_width = max(360, int(style.get("chat_min_width", 560)))
        self.chat_scroll.setMinimumWidth(self._chat_min_width)
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
        self.chats_panel.setStyleSheet(
            "QFrame#chats_panel {"
            f"background-color: {panel_bg};"
            f"border: 1px solid {panel_border};"
            f"border-radius: {panel_radius}px;"
            "}"
        )
        self.chat_settings_panel.setStyleSheet(
            "QFrame#chat_settings_panel {"
            f"background-color: {self._qss_rgba(style['chat_settings_bg'])};"
            f"border: 1px solid {self._qss_rgba(style['chat_settings_border'])};"
            f"border-radius: {int(style['chat_settings_radius'])}px;"
            "}"
        )

        p = style["panel_padding"]
        self.chat_panel.layout().setContentsMargins(p, p, p, p)
        cip = style["chat_inner_padding"]
        self.chat_layout.setContentsMargins(cip, cip, cip, cip)

        s = style["stats_padding"]
        self.stats_panel.layout().setContentsMargins(s, s, s, s)
        self.chats_panel.layout().setContentsMargins(s // 2, s // 2, s // 2, s // 2)

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
            "QListWidget#chat_list {"
            f"background: {self._qss_rgba(style['chat_inner_bg'])};"
            f"border: 1px solid {self._qss_rgba(style['chat_inner_border'])};"
            f"border-radius: {style['chat_inner_radius']}px;"
            "padding: 4px;"
            "}"
            "QListWidget#chat_list::item {"
            "padding: 7px 8px;"
            "border-radius: 6px;"
            "}"
            "QListWidget#chat_list::item:selected {"
            f"background: {self._qss_rgba(style['btn_hover_bg'])};"
            "}"
            "QLabel#chat_title {"
            f"color: {self._qss_rgba(style['ui_text'])};"
            "}"
            "QLabel#chat_settings_title {"
            f"color: {self._qss_rgba(style['ui_text'])};"
            "font-weight: 700;"
            "}"
            "QToolButton#chat_item_menu_btn {"
            f"background: {self._qss_rgba(style['btn_bg'])};"
            f"color: {self._qss_rgba(style['btn_fg'])};"
            f"border: 1px solid {self._qss_rgba(style['btn_border'])};"
            "border-radius: 5px;"
            "padding: 0px 6px;"
            "}"
            f"QToolButton#chat_item_menu_btn:hover {{ background: {self._qss_rgba(style['btn_hover_bg'])}; }}"
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
        self.chat_settings_incognito.set_colors(
            track_off=style["btn_disabled_bg"],
            track_on=style["send_btn_bg"],
            track_border=style["btn_border"],
            text_color=style["ui_text"],
        )
        self._install_base_button_animations(style)
        self._set_chats_drawer_open(self._chats_drawer_open, animated=False)
        self._set_stats_drawer_open(self._stats_drawer_open, animated=False)

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
        outer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, style["msg_gap_top"], 0, style["msg_gap_bottom"])
        outer_layout.setSpacing(6)

        card = QFrame()
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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

        show_ai_meta = role == "ai" and (not self._is_pending_stat_line(stat_line))

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
                like_btn.setText("👍")
                like_btn.setProperty("selected", True)
                dislike_btn.setProperty("selected", False)
            elif feedback == -1:
                dislike_btn.setText("👎")
                like_btn.setProperty("selected", False)
                dislike_btn.setProperty("selected", True)
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
            self._install_hover_reveal(card, actions, require_reenter=feedback is not None)

        return outer

    def _render_chat(self, scroll_to_bottom: bool = False):
        self._prune_empty_ai_placeholders()
        style = self._resolve_style()
        bar = self.chat_scroll.verticalScrollBar()
        prev_value = bar.value()
        was_at_bottom = prev_value >= max(0, bar.maximum() - 4)
        self._clear_chat_widgets()
        for i, (role, text, stat_line, feedback) in enumerate(self._history):
            self.chat_layout.addWidget(self._message_widget(i, role, text, stat_line, feedback, style))
        self._sync_chat_content_geometry()
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
        QTimer.singleShot(0, self._sync_chat_content_geometry)

    @staticmethod
    def _is_empty_ai_placeholder(row: tuple[str, str, str | None, int | None]) -> bool:
        role, text, stat_line, _ = row
        if role != "ai":
            return False
        if str(text or "").strip():
            return False
        return MainWindow._is_pending_stat_line(stat_line)

    def _prune_empty_ai_placeholders(self) -> None:
        if not self._history:
            return
        keep: list[tuple[str, str, str | None, int | None]] = []
        active_idx = self._stream_ai_index
        for i, row in enumerate(self._history):
            if active_idx is not None and i == active_idx:
                keep.append(row)
                continue
            if self._is_empty_ai_placeholder(row):
                continue
            keep.append(row)
        if len(keep) != len(self._history):
            if active_idx is not None:
                try:
                    prefix_kept = 0
                    for i, row in enumerate(self._history):
                        if i == active_idx:
                            break
                        if not self._is_empty_ai_placeholder(row):
                            prefix_kept += 1
                    self._stream_ai_index = prefix_kept
                except Exception:
                    self._stream_ai_index = None
            self._history = keep

    def _append_system(self, text: str):
        self._history.append(("system", text, None, None))
        self._render_chat(scroll_to_bottom=True)
        self._save_chat_sessions()

    def _append_user(self, text: str):
        self._history.append(("user", text, None, None))
        self._render_chat(scroll_to_bottom=True)
        self._save_chat_sessions()

    def _append_ai(self, text: str, stat_line: str):
        self._history.append(("ai", text, stat_line, None))
        self._render_chat(scroll_to_bottom=True)
        self._save_chat_sessions()

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
        chat = self._active_chat()
        is_incognito = bool(chat.get("incognito", False)) if chat else False
        if not is_incognito:
            try:
                self.brain.mm.register_assistant_feedback(
                    user_text=self._nearest_user_text_before(msg_index),
                    assistant_text=text,
                    feedback=norm_rating,
                    penalty=0.20,
                )
            except Exception:
                pass
        self._render_chat(scroll_to_bottom=False)
        self._save_chat_sessions()

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
            b.adjustSize()

        actions = like_btn.parentWidget()
        if actions is not None:
            actions.adjustSize()

        # After text/state changes (e.g. adding/removing checkmark), realign overlay
        # so hover hit-testing still matches the visible feedback buttons.
        if isinstance(row, _OverlayHost):
            row._reposition_overlay()
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
        if not self._active_chat():
            QMessageBox.information(self, "Чаты", "Сначала создай чат.")
            return
        if not self._last_user_text:
            return
        self._start_request(self._last_user_text, show_user=False)

    @Slot()
    def on_send(self):
        if not self._active_chat():
            QMessageBox.information(self, "Чаты", "Сначала создай чат кнопкой «Новый».")
            return
        text = self.input.toPlainText().strip()
        if not text:
            return

        started = self._start_request(text, show_user=True)
        if not started:
            return

        self.input.clear()
        self._last_user_text = text

    def _start_request(self, user_text: str, show_user: bool) -> bool:
        if not self._active_chat():
            QMessageBox.information(self, "Чаты", "Сначала создай чат.")
            return False
        if self._thread and self._thread.isRunning():
            QMessageBox.information(
                self,
                "\u041f\u043e\u0434\u043e\u0436\u0434\u0438",
                "\u0421\u0435\u0439\u0447\u0430\u0441 \u0443\u0436\u0435 \u0438\u0434\u0451\u0442 \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f. \u041d\u0430\u0436\u043c\u0438 '\u0421\u0442\u043e\u043f' \u0438\u043b\u0438 \u0434\u043e\u0436\u0434\u0438\u0441\u044c \u043e\u0442\u0432\u0435\u0442\u0430.",
            )
            return False

        if show_user:
            self._append_user(user_text)
        # Append streaming AI placeholder first, set index, then render.
        # This prevents pruning logic from removing placeholder before index bind.
        self._history.append(("ai", "", "…", None))
        self._stream_ai_index = len(self._history) - 1
        self._render_chat(scroll_to_bottom=True)
        self._save_chat_sessions()
        self._stream_chunk_buffer = ""
        self._stream_flush_timer.stop()

        self._set_status("\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f\u2026")
        self.btn_send.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_regen.setEnabled(False)

        self._thread = QThread()
        chat = self._active_chat()
        store_turn = not bool(chat.get("incognito", False)) if chat else True
        self._worker = ReplyWorker(self.brain, user_text=user_text, store_turn=store_turn)
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
            self._stream_ai_index = None
            return
        new_text = self._normalize_stream_text((text or "") + self._stream_chunk_buffer)
        self._history[idx] = (role, new_text, stat_line, feedback)
        self._stream_chunk_buffer = ""

        # Fast path: update only current streaming label to avoid full rerender flicker.
        try:
            item = self.chat_layout.itemAt(idx)
            w = item.widget() if item else None
            bubble = w.findChild(QLabel, "msg_bubble_label") if w else None
            if bubble is not None:
                bubble.setText(new_text)
                bubble.updateGeometry()
                bubble.adjustSize()
                if w is not None:
                    w.updateGeometry()
                    w.adjustSize()
                self._sync_chat_content_geometry()
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
            if role == "ai":
                final_text = res.text or text
                self._history[i] = (role, final_text, stat_line, feedback)
                self._render_chat(scroll_to_bottom=True)
            else:
                # Defensive fallback: never overwrite user/system rows with AI output.
                self._append_ai(res.text, stat_line)
            self._stream_ai_index = None
        else:
            self._append_ai(res.text, stat_line)
        self._save_chat_sessions()

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
        self._stream_ai_index = None
        self._prune_empty_ai_placeholders()
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_regen.setEnabled(bool(self._last_user_text))
        if self._worker:
            self._worker.deleteLater()
        if self._thread:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._render_chat(scroll_to_bottom=False)
        self._save_chat_sessions()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
