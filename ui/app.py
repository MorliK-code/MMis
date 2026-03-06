"""MMis Desktop UI (PySide6)."""

from __future__ import annotations

import re
import sys
import json
import shutil
import time
import os
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QLockFile,
    QObject,
    QPropertyAnimation,
    QSize,
    QStandardPaths,
    QThread,
    QTimer,
    Qt,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent, QColor, QFont, QKeyEvent
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid

from config.settings import load_config
from ui.chat_sessions import (
    history_from_serializable as chat_history_from_serializable,
    history_to_serializable as chat_history_to_serializable,
    load_sessions as load_chat_sessions,
    make_new_chat_payload,
    now_iso as chat_now_iso,
    save_sessions as save_chat_sessions,
    session_dir as chat_session_dir,
)
from ui.livecss import LiveCss
from ui.voice_adapter import build_stt_engine, build_tts_engine

from ui.api_client import ApiClient, ApiClientError
from ui.constants import DEFAULT_BUBBLE_OPACITY, DEFAULT_TEXT_SIZE, FEMALE_TONE_PRESETS, SHOW_TFLOPS_EST
from ui.metrics import est_tflops, ms_to_s_text, safe_div
from ui.widgets import (
    _ButtonAnimFilter,
    _FeedbackVisualFilter,
    _HoverRevealFilter,
    _MessageCard,
    _MiniSparkline,
    _OverlayHost,
    _ToggleSwitch,
)
from ui.workers import ReplyResult, ReplyWorker


_cfg = load_config()
MemoryStorageDir = Path(_cfg.memory_dir).expanduser().resolve()
MMIS_VOICE_INPUT_DIR = Path(_cfg.voice_input_dir or (MemoryStorageDir / "voice" / "input")).expanduser().resolve()
MMIS_VOICE_OUTPUT_DIR = Path(_cfg.voice_output_dir or (MemoryStorageDir / "voice" / "output")).expanduser().resolve()
MMIS_VOICE_TTS_VOICE = str(_cfg.voice_tts_voice or "ru-RU-DmitryNeural")
MMIS_VOICE_TTS_RATE = str(_cfg.voice_tts_rate or "+0%")
MMIS_VOICE_TTS_VOLUME = str(_cfg.voice_tts_volume or "+0%")


class MainWindow(QMainWindow):
    _models_refresh_done = Signal(object, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MMis \u2014 Desktop")
        self.resize(1080, 760)

        self.api = ApiClient()
        self._models_refresh_inflight = False
        self._models_refresh_done.connect(self._apply_models_refresh_result)

        self._thread: QThread | None = None
        self._worker: ReplyWorker | None = None
        self._last_user_text: str | None = None
        self._history: list[tuple[str, str, str | None, int | None, str | None]] = []
        self._thinking_open_by_msg: set[int] = set()
        self._stream_ai_index: int | None = None
        self._stream_chunk_buffer: str = ""
        self._stream_thinking_buffer: str = ""
        self._stream_flush_timer = QTimer(self)
        self._stream_flush_timer.setSingleShot(True)
        self._stream_flush_timer.setInterval(110)
        self._stream_flush_timer.timeout.connect(self._flush_stream_chunks)
        self._message_row_widgets: dict[int, QWidget] = {}
        self._chat_sync_pending_scroll_bottom = False
        self._chat_sync_timer = QTimer(self)
        self._chat_sync_timer.setSingleShot(True)
        self._chat_sync_timer.setInterval(130)
        self._chat_sync_timer.timeout.connect(self._on_deferred_chat_sync)

        self._request_started_perf: float | None = None
        self._rt_chunk_count = 0
        self._rt_output_chars = 0
        self._rt_thinking_chars = 0
        self._rt_last_flush_ms = 0.0
        self._rt_timer = QTimer(self)
        self._rt_timer.setInterval(250)
        self._rt_timer.timeout.connect(self._tick_realtime_stats)

        self.n_answers = 0
        self.sum_ms = 0.0
        self.sum_decode_ms = 0.0
        self.sum_eval = 0
        self.sum_prompt = 0
        self.sum_tps = 0.0
        self._request_primary_model: str = ""

        self._text_size = DEFAULT_TEXT_SIZE
        self._bubble_opacity = DEFAULT_BUBBLE_OPACITY
        self._live_css = LiveCss(Path(__file__).with_name("style.css"))
        self._feedback_reveal_show_ms = 170
        self._feedback_reveal_hide_ms = 130
        self._sessions_dir = MemoryStorageDir / "ui_chats"
        self._sessions_index_path = self._sessions_dir / "index.json"
        self._legacy_sessions_path = MemoryStorageDir / "ui_chats.json"
        self._ui_state_path = MemoryStorageDir / "ui_state.json"
        self._chat_sessions: list[dict] = []
        self._active_chat_id: str | None = None
        self._saved_think_enabled: bool | None = None
        self._pending_think_sync_from_saved = False
        self._updating_chat_controls = False
        self._chats_open_width = 240
        self._stats_open_width = 300
        self._chat_min_width = 560
        self._chats_drawer_open = False
        self._stats_drawer_open = False
        self._quick_settings_open = False
        self._quick_settings_open_height = 96
        self._chats_drawer_anim: QPropertyAnimation | None = None
        self._stats_drawer_anim: QPropertyAnimation | None = None
        self._quick_settings_anim: QPropertyAnimation | None = None
        self._settings_chat_id: str | None = None
        self._voice_stt = None
        default_voice = MMIS_VOICE_TTS_VOICE
        self._voice_tts_voice = default_voice
        self._voice_rate_percent = self._percent_to_int(MMIS_VOICE_TTS_RATE, default=0)
        self._voice_volume_percent = self._percent_to_int(MMIS_VOICE_TTS_VOLUME, default=0)
        self._voice_cache_dir = MMIS_VOICE_OUTPUT_DIR / ".cache"
        self._voice_cache_dir.mkdir(parents=True, exist_ok=True)
        self._voice_reply_cache_path = self._voice_cache_dir / "reply_live.wav"
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(1.0)
        self._media_player = QMediaPlayer(self)
        self._media_player.setAudioOutput(self._audio_output)
        self._media_player.errorOccurred.connect(self._on_media_error)
        self._load_ui_state()

        self._build_ui()
        self._load_or_init_chat_sessions()
        self._rt_timer.start()

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
        layout.setSpacing(8)

        top_bar = QFrame(root)
        top_bar.setObjectName("top_bar")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(8, 6, 8, 6)
        top_bar_layout.setSpacing(8)

        self.model_label = QLabel("Model:")
        self.model_label.setObjectName("model_label")
        top_bar_layout.addWidget(self.model_label)

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("model_combo")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.model_combo.setMinimumWidth(190)
        top_bar_layout.addWidget(self.model_combo, 1)

        self.btn_model_refresh = QPushButton("Обновить модели")
        self.btn_model_refresh.setObjectName("btn_model_refresh")
        self.btn_model_refresh.clicked.connect(self._on_model_refresh_clicked)
        top_bar_layout.addWidget(self.btn_model_refresh)

        self.btn_new_chat = QPushButton("Новый чат")
        self.btn_new_chat.setObjectName("btn_new_chat")
        self.btn_new_chat.clicked.connect(self._on_new_chat)
        top_bar_layout.addWidget(self.btn_new_chat)

        top_bar_layout.addStretch(1)

        self.btn_toggle_chats = QPushButton("Скрыть чаты")
        self.btn_toggle_chats.setObjectName("btn_toggle_chats")
        self.btn_toggle_chats.clicked.connect(self._toggle_chats_drawer)
        top_bar_layout.addWidget(self.btn_toggle_chats)

        self.btn_toggle_stats = QPushButton("Статистика")
        self.btn_toggle_stats.setObjectName("btn_toggle_stats")
        self.btn_toggle_stats.clicked.connect(self._toggle_stats_drawer)
        top_bar_layout.addWidget(self.btn_toggle_stats)

        self.btn_toggle_quick_settings = QPushButton("Настройки")
        self.btn_toggle_quick_settings.setObjectName("btn_toggle_quick_settings")
        self.btn_toggle_quick_settings.clicked.connect(self._toggle_quick_settings)
        top_bar_layout.addWidget(self.btn_toggle_quick_settings)

        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        top_bar_layout.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("Очистить чат")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.clicked.connect(self.on_clear)
        top_bar_layout.addWidget(self.btn_clear)

        layout.addWidget(top_bar, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        layout.addWidget(splitter, 1)

        # Left: chat list + chat settings
        self.chats_panel = QFrame()
        self.chats_panel.setObjectName("chats_panel")
        chats_layout = QVBoxLayout(self.chats_panel)
        chats_layout.setContentsMargins(10, 10, 10, 10)
        chats_layout.setSpacing(8)

        chats_header = QHBoxLayout()
        chats_header.setContentsMargins(0, 0, 0, 0)
        chats_header.setSpacing(6)
        chats_title = QLabel("Чаты")
        chats_title.setObjectName("stats_label")
        chats_header.addWidget(chats_title)
        chats_header.addStretch(1)
        self.chat_count_label = QLabel("0")
        self.chat_count_label.setObjectName("stats_label")
        chats_header.addWidget(self.chat_count_label)
        chats_layout.addLayout(chats_header)

        self.chat_list = QListWidget()
        self.chat_list.setObjectName("chat_list")
        self.chat_list.currentRowChanged.connect(self._on_chat_selected)
        chats_layout.addWidget(self.chat_list, 1)

        self.btn_chat_settings_open = QPushButton("Настройки выбранного")
        self.btn_chat_settings_open.setObjectName("btn_chat_settings_open")
        self.btn_chat_settings_open.clicked.connect(self._open_active_chat_settings)
        chats_layout.addWidget(self.btn_chat_settings_open, 0)

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
        splitter.addWidget(self.chats_panel)

        # Center: chat stream + compact composer + quick settings drawer
        self.chat_panel = QFrame()
        self.chat_panel.setObjectName("chat_panel")
        chat_panel_layout = QVBoxLayout(self.chat_panel)
        chat_panel_layout.setContentsMargins(0, 0, 0, 0)
        chat_panel_layout.setSpacing(8)

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
        chat_panel_layout.addWidget(self.chat_scroll, 1)

        composer = QFrame(self.chat_panel)
        composer.setObjectName("composer_panel")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(0, 0, 0, 0)
        composer_layout.setSpacing(6)

        self.input = QPlainTextEdit()
        self.input.setObjectName("chat_input")
        self.input.setPlaceholderText("Напиши сообщение…")
        self.input.setFont(QFont("Segoe UI", self._text_size))
        self.input.setFixedHeight(120)
        composer_layout.addWidget(self.input)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)

        controls.addStretch(1)

        self.btn_send = QPushButton("Отправить")
        self.btn_send.setObjectName("btn_send")
        self.btn_send.clicked.connect(self.on_send)
        self.btn_send.setDefault(True)
        self.btn_send.setMinimumWidth(120)
        controls.addWidget(self.btn_send)

        composer_layout.addLayout(controls)
        chat_panel_layout.addWidget(composer, 0)

        self.quick_settings_panel = QFrame(self.chat_panel)
        self.quick_settings_panel.setObjectName("quick_settings_panel")
        quick_layout = QVBoxLayout(self.quick_settings_panel)
        quick_layout.setContentsMargins(6, 4, 6, 6)
        quick_layout.setSpacing(6)

        quick_row_1 = QHBoxLayout()
        quick_row_1.setContentsMargins(0, 0, 0, 0)
        quick_row_1.setSpacing(8)

        self.btn_voice_input = QPushButton("Голос файл")
        self.btn_voice_input.setObjectName("btn_voice_input")
        self.btn_voice_input.clicked.connect(self.on_voice_input_file)
        quick_row_1.addWidget(self.btn_voice_input)

        self.btn_voice_speak = QPushButton("Озвучить")
        self.btn_voice_speak.setObjectName("btn_voice_speak")
        self.btn_voice_speak.clicked.connect(self.on_voice_speak_last_ai)
        quick_row_1.addWidget(self.btn_voice_speak)

        self.voice_tone_label = QLabel("Тон")
        self.voice_tone_label.setObjectName("stats_label")
        quick_row_1.addWidget(self.voice_tone_label)

        self.voice_tone_combo = QComboBox()
        self.voice_tone_combo.setObjectName("voice_tone_combo")
        selected_idx = 0
        for i, (label, voice_id) in enumerate(FEMALE_TONE_PRESETS):
            self.voice_tone_combo.addItem(label, userData=voice_id)
            if voice_id == self._voice_tts_voice:
                selected_idx = i
        self.voice_tone_combo.setCurrentIndex(selected_idx)
        self._voice_tts_voice = str(self.voice_tone_combo.currentData() or self._voice_tts_voice)
        self.voice_tone_combo.currentIndexChanged.connect(self._on_voice_tone_changed)
        quick_row_1.addWidget(self.voice_tone_combo)

        self.voice_rate_label = QLabel("Скорость")
        self.voice_rate_label.setObjectName("stats_label")
        quick_row_1.addWidget(self.voice_rate_label)

        self.voice_rate_spin = QSpinBox()
        self.voice_rate_spin.setObjectName("voice_rate_spin")
        self.voice_rate_spin.setRange(-60, 60)
        self.voice_rate_spin.setSingleStep(5)
        self.voice_rate_spin.setSuffix("%")
        self.voice_rate_spin.setValue(self._voice_rate_percent)
        quick_row_1.addWidget(self.voice_rate_spin)

        self.voice_volume_label = QLabel("Громкость")
        self.voice_volume_label.setObjectName("stats_label")
        quick_row_1.addWidget(self.voice_volume_label)

        self.voice_volume_spin = QSpinBox()
        self.voice_volume_spin.setObjectName("voice_volume_spin")
        self.voice_volume_spin.setRange(-90, 100)
        self.voice_volume_spin.setSingleStep(5)
        self.voice_volume_spin.setSuffix("%")
        self.voice_volume_spin.setValue(self._voice_volume_percent)
        quick_row_1.addWidget(self.voice_volume_spin)

        quick_row_1.addStretch(1)
        quick_layout.addLayout(quick_row_1)

        quick_row_2 = QHBoxLayout()
        quick_row_2.setContentsMargins(0, 0, 0, 0)
        quick_row_2.setSpacing(8)

        self.voice_auto_tts = _ToggleSwitch("Авто-озвучка")
        self.voice_auto_tts.setObjectName("voice_auto_tts")
        self.voice_auto_tts.setChecked(False)
        quick_row_2.addWidget(self.voice_auto_tts)

        self.think_toggle = _ToggleSwitch("Думать")
        self.think_toggle.setObjectName("think_toggle")
        self.think_toggle.setChecked(True if self._saved_think_enabled is None else bool(self._saved_think_enabled))
        self.think_toggle.toggled.connect(self._on_think_toggled)
        quick_row_2.addWidget(self.think_toggle)

        quick_row_2.addStretch(1)
        quick_layout.addLayout(quick_row_2)

        self._quick_settings_open_height = max(86, self.quick_settings_panel.sizeHint().height())
        self.quick_settings_panel.setVisible(False)
        self.quick_settings_panel.setMinimumHeight(0)
        self.quick_settings_panel.setMaximumHeight(0)
        chat_panel_layout.addWidget(self.quick_settings_panel, 0)

        splitter.addWidget(self.chat_panel)

        # Right: concise stats panel
        self.stats_panel = QFrame()
        self.stats_panel.setObjectName("stats_panel")
        side_layout = QVBoxLayout(self.stats_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        self.status_label = QLabel("Статус: <b>Готово</b>")
        self.status_label.setObjectName("stats_label")
        side_layout.addWidget(self.status_label)

        rt_block = QFrame(self.stats_panel)
        rt_block.setObjectName("stats_rt_block")
        rt_layout = QVBoxLayout(rt_block)
        rt_layout.setContentsMargins(10, 10, 10, 10)
        rt_layout.setSpacing(6)

        self.rt_title_label = QLabel("Realtime")
        self.rt_title_label.setObjectName("stats_label")
        rt_layout.addWidget(self.rt_title_label)

        self.rt_elapsed_label = QLabel("Время: —")
        self.rt_elapsed_label.setObjectName("stats_label")
        self.rt_tokens_out_label = QLabel("Токены out (оценка): —")
        self.rt_tokens_out_label.setObjectName("stats_label")
        self.rt_tokens_think_label = QLabel("Токены think (оценка): —")
        self.rt_tokens_think_label.setObjectName("stats_label")
        self.rt_tps_label = QLabel("Tok/s (оценка): —")
        self.rt_tps_label.setObjectName("stats_label")
        self.rt_chunks_label = QLabel("Chunk/s: —")
        self.rt_chunks_label.setObjectName("stats_label")
        self.rt_flush_label = QLabel("UI flush: —")
        self.rt_flush_label.setObjectName("stats_label")
        self.rt_buffer_label = QLabel("Буфер: out 0 / think 0")
        self.rt_buffer_label.setObjectName("stats_label")

        self.rt_tps_graph = _MiniSparkline("tok/s")
        self.rt_chunk_graph = _MiniSparkline("chunk/s")
        self.rt_ui_graph = _MiniSparkline("ui flush ms")

        for w in (
            self.rt_elapsed_label,
            self.rt_tokens_out_label,
            self.rt_tokens_think_label,
            self.rt_tps_label,
            self.rt_chunks_label,
            self.rt_flush_label,
            self.rt_buffer_label,
            self.rt_tps_graph,
            self.rt_chunk_graph,
            self.rt_ui_graph,
        ):
            rt_layout.addWidget(w)
        side_layout.addWidget(rt_block)

        avg_block = QFrame(self.stats_panel)
        avg_block.setObjectName("stats_avg_block")
        avg_layout = QVBoxLayout(avg_block)
        avg_layout.setContentsMargins(10, 10, 10, 10)
        avg_layout.setSpacing(6)

        self.avg_title_label = QLabel("Средние значения")
        self.avg_title_label.setObjectName("stats_label")
        avg_layout.addWidget(self.avg_title_label)

        self.avg_ms_label = QLabel("Время —")
        self.avg_ms_label.setObjectName("stats_label")
        self.avg_decode_label = QLabel("Decode —")
        self.avg_decode_label.setObjectName("stats_label")
        self.avg_tokens_label = QLabel("Gen — / Prompt —")
        self.avg_tokens_label.setObjectName("stats_label")
        self.avg_tps_label = QLabel("Tok/s —")
        self.avg_tps_label.setObjectName("stats_label")
        self.avg_tflops_label = QLabel("TFLOPs —")
        self.avg_tflops_label.setObjectName("stats_label")

        for w in (
            self.status_label,
            self.rt_title_label,
            self.rt_elapsed_label,
            self.rt_tokens_out_label,
            self.rt_tokens_think_label,
            self.rt_tps_label,
            self.rt_chunks_label,
            self.rt_flush_label,
            self.rt_buffer_label,
            self.avg_title_label,
            self.avg_ms_label,
            self.avg_decode_label,
            self.avg_tokens_label,
            self.avg_tps_label,
            self.avg_tflops_label,
            self.chat_count_label,
        ):
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)

        for w in (self.avg_ms_label, self.avg_decode_label, self.avg_tokens_label, self.avg_tps_label, self.avg_tflops_label):
            avg_layout.addWidget(w)
        side_layout.addWidget(avg_block)

        side_layout.addStretch(1)
        splitter.addWidget(self.stats_panel)

        self.splitter = splitter
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        splitter.setStretchFactor(2, 3)

        self.chats_panel.setMinimumWidth(0)
        self.chats_panel.setMaximumWidth(self._chats_open_width)
        self.chat_scroll.setMinimumWidth(self._chat_min_width)
        self.stats_panel.setMinimumWidth(0)
        self.stats_panel.setMaximumWidth(self._stats_open_width)

        splitter.setSizes([self._chats_open_width, 980, 0])
        for i in (1, 2):
            handle = splitter.handle(i)
            if handle:
                handle.setEnabled(False)
                handle.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self._set_chats_drawer_open(True, animated=False)
        self._set_stats_drawer_open(False, animated=False)
        self._set_quick_settings_open(False, animated=False)

        self.input.installEventFilter(self)

        QTimer.singleShot(0, lambda: self._refresh_models_in_ui(show_popup=False))
        self._apply_panel_styles(self._resolve_style())
        self._render_chat()

    @staticmethod
    def _now_iso() -> str:
        return chat_now_iso()

    @staticmethod
    def _percent_to_int(value: str, default: int = 0) -> int:
        s = str(value or "").strip()
        if not s.endswith("%"):
            return int(default)
        try:
            return int(float(s[:-1]))
        except Exception:
            return int(default)

    @staticmethod
    def _int_to_percent(value: int) -> str:
        v = int(value)
        return f"{v:+d}%"

    def _sync_model_combo_to(self, model_name: str | None) -> None:
        target = str(model_name or "").strip()
        if not target or not hasattr(self, "model_combo") or self.model_combo.count() <= 0:
            return
        idx = self.model_combo.findText(target)
        if idx < 0 or idx == self.model_combo.currentIndex():
            return
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _update_model_label(self, current_model: str | None = None) -> None:
        current = str(current_model or "").strip()
        if not current:
            current = str(self.api.get_runtime_model() or "").strip()
        if not current and hasattr(self, "model_combo") and self.model_combo.count() > 0:
            current = str(self.model_combo.currentText() or "").strip()
        self.model_label.setText(f"Model: <b>{current}</b>" if current else "Model: <b>—</b>")

    @Slot()
    def _on_model_refresh_clicked(self) -> None:
        self._refresh_models_in_ui(show_popup=True)

    def _refresh_models_in_ui(self, show_popup: bool = True) -> None:
        if self._models_refresh_inflight:
            return
        self._models_refresh_inflight = True
        self.btn_model_refresh.setEnabled(False)
        self._set_status("Обновляю модели…")

        def _worker():
            payload: dict = {"ok": False}
            try:
                models_payload = self.api.list_models()
                health_payload = self.api.health()
                payload = {
                    "ok": True,
                    "models_payload": models_payload,
                    "health_payload": health_payload,
                }
            except ApiClientError as exc:
                payload = {"ok": False, "error": str(exc)}
            except Exception as exc:
                payload = {"ok": False, "error": str(exc)}
            self._models_refresh_done.emit(payload, bool(show_popup))

        threading.Thread(target=_worker, name="ui-models-refresh", daemon=True).start()

    @Slot(object, bool)
    def _apply_models_refresh_result(self, payload: object, show_popup: bool) -> None:
        self._models_refresh_inflight = False
        self.btn_model_refresh.setEnabled(not bool(self._thread and self._thread.isRunning()))
        data = payload if isinstance(payload, dict) else {}
        if not bool(data.get("ok")):
            self._set_status("API недоступен")
            cached = str(self.api.get_runtime_model() or "").strip()
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            if cached:
                self.model_combo.addItem(cached)
                self.model_combo.setCurrentIndex(0)
            self.model_combo.blockSignals(False)
            self._update_model_label(cached)
            if show_popup:
                QMessageBox.warning(self, "API", str(data.get("error") or "Не удалось обновить список моделей"))
            self._update_chat_controls_state()
            return

        models_payload = data.get("models_payload") if isinstance(data.get("models_payload"), dict) else {}
        health_payload = data.get("health_payload") if isinstance(data.get("health_payload"), dict) else {}
        models = [str(x) for x in (models_payload.get("models") or []) if str(x).strip()]
        current = str(models_payload.get("runtime_model") or health_payload.get("model") or "")
        thinking_enabled = bool(health_payload.get("thinking_enabled", True))

        if not models and current:
            models = [current]
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for m in models:
            self.model_combo.addItem(m)
        if current and current in models:
            self.model_combo.setCurrentIndex(models.index(current))
        elif models:
            chosen = models[0]
            self.model_combo.setCurrentIndex(0)
            try:
                self.api.set_model(chosen)
            except ApiClientError:
                pass
        self.model_combo.blockSignals(False)
        self.think_toggle.blockSignals(True)
        self.think_toggle.setChecked(thinking_enabled)
        self.think_toggle.blockSignals(False)
        if self._pending_think_sync_from_saved and self._saved_think_enabled is not None:
            try:
                actual = self.api.set_thinking_enabled(bool(self._saved_think_enabled))
            except ApiClientError:
                actual = bool(thinking_enabled)
            self._pending_think_sync_from_saved = False
            self.think_toggle.blockSignals(True)
            self.think_toggle.setChecked(bool(actual))
            self.think_toggle.blockSignals(False)
            self._save_ui_state()
        self._update_model_label(current)
        self._set_status("Готово")
        self._update_chat_controls_state()

    @Slot(int)
    def _on_model_changed(self, _index: int) -> None:
        name = str(self.model_combo.currentText() or "").strip()
        if not name:
            return
        try:
            self.api.set_model(name)
            actual = self.api.get_runtime_model()
            self._sync_model_combo_to(actual)
            self._update_model_label(actual)
        except ApiClientError as exc:
            QMessageBox.warning(self, "Модель", str(exc))

    @Slot(bool)
    def _on_think_toggled(self, checked: bool) -> None:
        try:
            actual = self.api.set_thinking_enabled(bool(checked))
            if actual != bool(checked):
                self.think_toggle.blockSignals(True)
                self.think_toggle.setChecked(actual)
                self.think_toggle.blockSignals(False)
            self._save_ui_state()
        except ApiClientError as exc:
            self.think_toggle.blockSignals(True)
            self.think_toggle.setChecked(not bool(checked))
            self.think_toggle.blockSignals(False)
            QMessageBox.warning(self, "API", str(exc))

    @staticmethod
    def _history_to_serializable(history: list[tuple[str, str, str | None, int | None, str | None]]) -> list[dict]:
        return chat_history_to_serializable(history)

    @staticmethod
    def _history_from_serializable(rows: list[dict] | None) -> list[tuple[str, str, str | None, int | None, str | None]]:
        return chat_history_from_serializable(rows)

    def _new_chat_payload(self, title: str | None = None, incognito: bool = False) -> dict:
        return make_new_chat_payload(existing_count=len(self._chat_sessions), title=title, incognito=incognito)

    def _chat_dir(self, chat_id: str) -> Path:
        return chat_session_dir(self._sessions_dir, chat_id)

    def _load_ui_state(self) -> None:
        self._saved_think_enabled = None
        self._pending_think_sync_from_saved = False
        try:
            if not self._ui_state_path.exists():
                return
            payload = json.loads(self._ui_state_path.read_text(encoding="utf-8-sig") or "{}")
            if not isinstance(payload, dict):
                return
            if "think_enabled" in payload:
                self._saved_think_enabled = bool(payload.get("think_enabled"))
                self._pending_think_sync_from_saved = True
        except Exception:
            self._saved_think_enabled = None
            self._pending_think_sync_from_saved = False

    def _save_ui_state(self) -> None:
        try:
            self._ui_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"think_enabled": bool(self.think_toggle.isChecked())}
            self._ui_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _save_chat_sessions(self) -> None:
        self._sync_active_session_from_history()
        try:
            active_chat = self._active_chat()
            active_id = None
            if active_chat and not bool(active_chat.get("incognito", False)):
                active_id = str(active_chat.get("id") or "") or None
            save_chat_sessions(self._sessions_dir, self._sessions_index_path, self._chat_sessions, active_id)
        except Exception:
            pass

    def _load_or_init_chat_sessions(self) -> None:
        loaded, _active_id = load_chat_sessions(self._sessions_dir, self._sessions_index_path, self._legacy_sessions_path)
        self._chat_sessions = loaded
        # Always start with no active chat selected.
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
                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, str(chat.get("id") or ""))
                item.setSizeHint(QSize(0, 32))
                self.chat_list.addItem(item)

            idx = -1
            for i, chat in enumerate(self._chat_sessions):
                if self._active_chat_id and str(chat.get("id")) == self._active_chat_id:
                    idx = i
                    break
            self.chat_list.setCurrentRow(idx if self._chat_sessions else -1)
            if hasattr(self, "chat_count_label"):
                self.chat_count_label.setText(str(len(self._chat_sessions)))
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
            self._stream_thinking_buffer = ""
            self._last_user_text = None
            self._set_status("Готово")
            self._render_chat(scroll_to_bottom=scroll_to_bottom)
            self._update_chat_controls_state()
            return
        self._history = list(chat.get("history") or [])
        self._stream_ai_index = None
        self._stream_chunk_buffer = ""
        self._stream_thinking_buffer = ""
        self._last_user_text = None
        if self._settings_chat_id and self._settings_chat_id != str(chat.get("id") or ""):
            self._close_chat_settings_panel()
        for role, text, _, _, _thinking in reversed(self._history):
            if role == "user":
                self._last_user_text = text
                break
        self._set_status("Готово")
        self._render_chat(scroll_to_bottom=scroll_to_bottom)
        self._update_chat_controls_state()

    def _update_chat_controls_state(self) -> None:
        has_active = self._active_chat() is not None
        is_busy = bool(self._thread and self._thread.isRunning())
        self.btn_new_chat.setEnabled(True)
        self.btn_model_refresh.setEnabled(not is_busy)
        self.model_combo.setEnabled((self.model_combo.count() > 0) and (not is_busy))
        self.think_toggle.setEnabled(not is_busy)
        self.btn_send.setEnabled(not is_busy)
        self.btn_chat_settings_open.setEnabled(has_active and not is_busy)
        self.btn_voice_input.setEnabled(has_active and not is_busy)
        self.btn_voice_speak.setEnabled(has_active and not is_busy)
        self.voice_tone_combo.setEnabled(has_active and not is_busy)
        self.voice_rate_spin.setEnabled(has_active and not is_busy)
        self.voice_volume_spin.setEnabled(has_active and not is_busy)
        self.voice_auto_tts.setEnabled(has_active)
        self.chat_settings_export.setEnabled(has_active)
        self.chat_settings_delete.setEnabled(has_active)
        self.chat_settings_incognito.setEnabled(has_active)

    @Slot()
    def _open_active_chat_settings(self) -> None:
        chat = self._active_chat()
        if not chat:
            return
        self._open_chat_settings(str(chat.get("id") or ""))

    def _close_chat_settings_panel(self) -> None:
        self._settings_chat_id = None
        self.chat_settings_title.setText("Настройки чата")
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
        self._set_chats_drawer_open(True, animated=True)
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

    @Slot()
    def _on_new_chat(self) -> None:
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "Подожди", "Сначала останови или дождись завершения генерации.")
            return
        self._close_chat_settings_panel()
        self._sync_active_session_from_history()
        chat = self._new_chat_payload(incognito=False)
        self._chat_sessions.append(chat)
        self._active_chat_id = str(chat["id"])
        self._refresh_chat_selector()
        self._apply_active_chat_to_ui(scroll_to_bottom=True)
        self._close_chat_settings_panel()
        self._save_chat_sessions()
        if self._chats_drawer_open:
            self._set_chats_drawer_open(False, animated=True)

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

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self._css_timer.stop()
        except Exception:
            pass
        try:
            self._rt_timer.stop()
        except Exception:
            pass
        try:
            self._stream_flush_timer.stop()
            self._chat_sync_timer.stop()
        except Exception:
            pass
        try:
            self._media_player.stop()
        except Exception:
            pass

        if self._worker:
            try:
                self._worker.request_cancel()
            except Exception:
                pass
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            if not self._thread.wait(650):
                self._thread.terminate()
                self._thread.wait(180)
        self._save_chat_sessions()
        super().closeEvent(event)
        app = QApplication.instance()
        if app:
            app.quit()
        try:
            # Safety net: force process exit if non-daemon background threads keep it alive.
            killer = threading.Timer(1.2, lambda: os._exit(0))
            killer.daemon = True
            killer.start()
        except Exception:
            pass

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
        self.chats_panel.setMinimumWidth(0)
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
        self.stats_panel.setMinimumWidth(0)
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

    @Slot()
    def _toggle_quick_settings(self) -> None:
        self._set_quick_settings_open(not self._quick_settings_open, animated=True)

    def _set_quick_settings_open(self, is_open: bool, animated: bool = True) -> None:
        target = int(self._quick_settings_open_height if is_open else 0)
        self._quick_settings_open = bool(is_open)
        self.btn_toggle_quick_settings.setText("Скрыть настройки" if is_open else "Настройки")
        if not animated:
            self.quick_settings_panel.setMinimumHeight(target)
            self.quick_settings_panel.setMaximumHeight(target)
            self.quick_settings_panel.setVisible(target > 0)
            return

        if self._quick_settings_anim and self._quick_settings_anim.state() == QAbstractAnimation.Running:
            self._quick_settings_anim.stop()
        self.quick_settings_panel.setMinimumHeight(0)
        if target > 0:
            self.quick_settings_panel.setVisible(True)
        anim = QPropertyAnimation(self.quick_settings_panel, b"maximumHeight", self)
        anim.setDuration(180)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(int(self.quick_settings_panel.maximumHeight()))
        anim.setEndValue(target)

        def _finish():
            self.quick_settings_panel.setMinimumHeight(target)
            self.quick_settings_panel.setVisible(target > 0)

        anim.finished.connect(_finish)
        self._quick_settings_anim = anim
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
    def _css_text(value: str | None, default: str) -> str:
        raw = MainWindow._str(value, default)
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            return raw[1:-1]
        return raw

    @staticmethod
    def _normalize_stream_text(text: str) -> str:
        s = str(text or "")
        if not s:
            return ""
        s = re.sub(r"</?think>", "", s, flags=re.I)
        s = re.sub(r"</?thinking>", "", s, flags=re.I)
        s = re.sub(r"</?reasoning>", "", s, flags=re.I)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # Prevent stream artifacts from inflating bubble height.
        s = re.sub(r"\n{2,}", "\n", s)
        s = re.sub(r"[ \t\f\v]+", " ", s)
        s = re.sub(r"\s*\n\s*", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    @staticmethod
    def _preview_first_words(text: str, max_words: int = 10) -> str:
        s = re.sub(r"\s+", " ", str(text or "").strip())
        if not s:
            return ""
        words = s.split(" ")
        if len(words) <= max_words:
            return s
        return " ".join(words[:max_words]) + "..."

    @staticmethod
    def _preview_last_words(text: str, max_words: int = 7) -> str:
        s = re.sub(r"\s+", " ", str(text or "").strip())
        if not s:
            return ""
        words = s.split(" ")
        if len(words) <= max_words:
            return s
        return "..." + " ".join(words[-max_words:])

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
        self._message_row_widgets = {}

    def _schedule_chat_sync(self, scroll_to_bottom: bool) -> None:
        if scroll_to_bottom:
            self._chat_sync_pending_scroll_bottom = True
        if not self._chat_sync_timer.isActive():
            self._chat_sync_timer.start()

    @Slot()
    def _on_deferred_chat_sync(self) -> None:
        if self._thread and self._thread.isRunning():
            self.chat_layout.invalidate()
            self.chat_layout.activate()
            self.chat_root.adjustSize()
        else:
            self._sync_chat_content_geometry()
        if self._chat_sync_pending_scroll_bottom:
            bar = self.chat_scroll.verticalScrollBar()
            bar.setValue(bar.maximum())
        self._chat_sync_pending_scroll_bottom = False

    def _sync_chat_content_geometry(self) -> None:
        layout = self.chat_layout
        if layout is None:
            return
        layout.invalidate()
        layout.activate()
        content_h = max(1, int(layout.sizeHint().height()))
        if not self._history:
            viewport_h = max(1, int(self.chat_scroll.viewport().height()))
            content_h = max(content_h, viewport_h)
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
            self.btn_model_refresh,
            self.btn_toggle_chats,
            self.btn_toggle_stats,
            self.btn_toggle_quick_settings,
            self.btn_stop,
            self.btn_clear,
            self.btn_new_chat,
            self.btn_chat_settings_open,
            self.chat_settings_export,
            self.chat_settings_delete,
            self.chat_settings_close,
            self.btn_voice_input,
            self.btn_voice_speak,
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
            "msg_card_margin_x": self._px(vars_map.get("--msg-card-margin-x"), 8),
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
            "chats_drawer_width": self._px(vars_map.get("--chats-drawer-width"), 250),
            "stats_drawer_width": self._px(vars_map.get("--stats-drawer-width"), 320),
            "chat_min_width": self._px(vars_map.get("--chat-min-width"), 560),
            "chat_settings_bg": self._color(vars_map.get("--chat-settings-bg"), "rgba(20, 21, 25, 0.75)"),
            "chat_settings_border": self._color(vars_map.get("--chat-settings-border"), "rgba(255, 255, 255, 0.10)"),
            "chat_settings_radius": self._px(vars_map.get("--chat-settings-radius"), 10),
            "chat_empty_text": self._css_text(vars_map.get("--chat-empty-text"), "Привет. Напиши сюда что угодно."),
            "chat_empty_color": self._color(vars_map.get("--chat-empty-color"), "rgba(163, 163, 163, 0.75)"),
            "chat_empty_size": self._px(vars_map.get("--chat-empty-size"), 14),
            "chat_empty_weight": self._px(vars_map.get("--chat-empty-weight"), 600),
            "chat_empty_max_width": self._px(vars_map.get("--chat-empty-max-width"), 560),
            "chat_empty_pad_y": self._px(vars_map.get("--chat-empty-pad-y"), 24),
            "chat_empty_pad_x": self._px(vars_map.get("--chat-empty-pad-x"), 10),
            "chat_empty_align": self._str(vars_map.get("--chat-empty-align"), "center").lower(),
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
        self._chats_open_width = max(190, int(style.get("chats_drawer_width", 250)) + 16)
        self._stats_open_width = max(250, int(style.get("stats_drawer_width", 320)) + 20)
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
            "QFrame#composer_panel {"
            f"background: {self._qss_rgba(style['panel_bg'])};"
            f"border: 1px solid {self._qss_rgba(style['panel_border'])};"
            f"border-radius: {max(8, int(style['panel_radius']) - 2)}px;"
            "padding: 6px;"
            "}"
            "QFrame#quick_settings_panel {"
            f"background: {self._qss_rgba(style['panel_bg'])};"
            f"border: 1px solid {self._qss_rgba(style['panel_border'])};"
            f"border-radius: {max(8, int(style['panel_radius']) - 2)}px;"
            "padding: 4px;"
            "}"
            "QFrame#stats_rt_block, QFrame#stats_avg_block {"
            f"background: {self._qss_rgba(style['chat_inner_bg'])};"
            f"border: 1px solid {self._qss_rgba(style['chat_inner_border'])};"
            f"border-radius: {max(8, int(style['chat_inner_radius']) - 1)}px;"
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
        for w in (
            self.status_label,
            self.rt_title_label,
            self.rt_elapsed_label,
            self.rt_tokens_out_label,
            self.rt_tokens_think_label,
            self.rt_tps_label,
            self.rt_chunks_label,
            self.rt_flush_label,
            self.rt_buffer_label,
            self.avg_title_label,
            self.avg_ms_label,
            self.avg_decode_label,
            self.avg_tokens_label,
            self.avg_tps_label,
            self.avg_tflops_label,
            self.chat_count_label,
        ):
            self._apply_text_shadow(w, style["side_text_shadow"])
        self.chat_settings_incognito.set_colors(
            track_off=style["btn_disabled_bg"],
            track_on=style["send_btn_bg"],
            track_border=style["btn_border"],
            text_color=style["ui_text"],
        )
        self._quick_settings_open_height = max(86, self.quick_settings_panel.sizeHint().height())
        self._install_base_button_animations(style)
        self._set_chats_drawer_open(self._chats_drawer_open, animated=False)
        self._set_stats_drawer_open(self._stats_drawer_open, animated=False)
        self._set_quick_settings_open(self._quick_settings_open, animated=False)

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
        thinking: str | None,
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

        outer = _OverlayHost(self.chat_root)
        outer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer_layout = QVBoxLayout(outer)
        msg_margin_x = max(0, int(style.get("msg_card_margin_x", 0)))
        outer_layout.setContentsMargins(msg_margin_x, style["msg_gap_top"], msg_margin_x, style["msg_gap_bottom"])
        outer_layout.setSpacing(6)

        card = QFrame(outer)
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

        name_label = QLabel(name_text, card)
        name_label.setTextFormat(Qt.PlainText)
        name_label.setFont(name_font)
        name_label.setStyleSheet(f"color: {self._qss_rgba(style['name_color'])}; background: transparent;")
        name_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._apply_text_shadow(name_label, style["name_text_shadow"])
        card_layout.addWidget(name_label)

        think_toggle: QPushButton | None = None
        think_preview_label: QLabel | None = None
        think_full_label: QLabel | None = None
        think_row: QFrame | None = None
        if role == "ai":
            think_text = str(thinking or "").strip()
            has_thinking = bool(think_text)
            is_open = msg_index in self._thinking_open_by_msg
            is_streaming_thinking = bool(
                self._stream_ai_index == msg_index and self._thread and self._thread.isRunning()
            )
            compact_text = self._preview_last_words(think_text, max_words=7) if is_streaming_thinking else ""

            think_row = QFrame(card)
            think_row.setObjectName("think_row")
            think_row_layout = QVBoxLayout(think_row)
            think_row_layout.setContentsMargins(0, 0, 0, 0)
            think_row_layout.setSpacing(2)

            header_wrap = QWidget(think_row)
            header_layout = QHBoxLayout(header_wrap)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(6)

            think_toggle = QPushButton("Мысли ▾" if is_open else "Мысли ▸", header_wrap)
            think_toggle.setStyleSheet(
                "QPushButton {"
                "background: transparent;"
                "border: none;"
                f"color: {self._qss_rgba(QColor(220, 220, 220, 150))};"
                "padding: 0px;"
                "text-align: left;"
                "}"
                "QPushButton:hover { text-decoration: underline; }"
            )
            think_toggle.setCursor(Qt.PointingHandCursor)
            think_toggle.clicked.connect(lambda _, i=msg_index: self._toggle_thinking(i))
            think_toggle.setVisible(has_thinking)
            think_toggle.setEnabled(has_thinking)
            header_layout.addWidget(think_toggle, 0)

            think_preview_label = QLabel(compact_text, header_wrap)
            think_preview_label.setTextFormat(Qt.PlainText)
            think_preview_label.setWordWrap(False)
            think_preview_label.setStyleSheet(
                f"color: {self._qss_rgba(QColor(220, 220, 220, 150))};"
                "background: transparent;"
            )
            think_preview_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            think_preview_label.setVisible(bool(has_thinking and is_streaming_thinking))
            header_layout.addWidget(think_preview_label, 1)
            think_row_layout.addWidget(header_wrap)

            think_full_label = QLabel(think_text, think_row)
            think_full_label.setTextFormat(Qt.PlainText)
            think_full_label.setWordWrap(True)
            think_full_label.setStyleSheet(
                f"color: {self._qss_rgba(QColor(220, 220, 220, 175))};"
                "background: transparent;"
            )
            think_full_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            think_full_label.setVisible(bool(has_thinking and is_open))
            think_row_layout.addWidget(think_full_label)
            think_row.setVisible(has_thinking)
            card_layout.addWidget(think_row)

        bubble_label = QLabel(text, card)
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

        stats_label: QLabel | None = None
        if show_ai_meta:
            stats_label = QLabel(stat_line or "—", outer)
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

            actions = QWidget(outer)
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

        outer._bubble_label = bubble_label
        outer._think_toggle = think_toggle
        outer._think_row = think_row
        outer._think_preview_label = think_preview_label
        outer._think_full_label = think_full_label
        outer._stats_label = stats_label
        return outer

    def _render_chat(self, scroll_to_bottom: bool = False):
        self._prune_empty_ai_placeholders()
        style = self._resolve_style()
        bar = self.chat_scroll.verticalScrollBar()
        prev_value = bar.value()
        was_at_bottom = prev_value >= max(0, bar.maximum() - 4)
        self._clear_chat_widgets()
        self.chat_layout.setAlignment(Qt.AlignTop)
        if not self._history:
            placeholder = QLabel(style["chat_empty_text"], self.chat_root)
            placeholder.setObjectName("chat_empty_placeholder")
            placeholder.setWordWrap(True)
            align_map = {"left": Qt.AlignLeft, "center": Qt.AlignHCenter, "right": Qt.AlignRight}
            h_align = align_map.get(style["chat_empty_align"], Qt.AlignHCenter)
            placeholder.setAlignment(Qt.AlignVCenter | h_align)
            placeholder.setTextInteractionFlags(Qt.NoTextInteraction)
            placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            max_w = int(style.get("chat_empty_max_width", 560))
            if max_w > 0:
                placeholder.setMaximumWidth(max_w)
            placeholder.setStyleSheet(
                f"color: {self._qss_rgba(style['chat_empty_color'])};"
                "background: transparent;"
                f"padding: {int(style['chat_empty_pad_y'])}px {int(style['chat_empty_pad_x'])}px;"
                f"font-size: {int(style['chat_empty_size'])}px;"
                f"font-weight: {int(style['chat_empty_weight'])};"
            )
            self.chat_layout.setAlignment(Qt.AlignCenter)
            self.chat_layout.addWidget(placeholder, 0, Qt.AlignCenter | h_align)
        for i, (role, text, stat_line, feedback, thinking) in enumerate(self._history):
            row = self._message_widget(i, role, text, stat_line, feedback, thinking, style)
            self.chat_layout.addWidget(row)
            self._message_row_widgets[i] = row
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
    def _is_empty_ai_placeholder(row: tuple[str, str, str | None, int | None, str | None]) -> bool:
        role, text, stat_line, _, _thinking = row
        if role != "ai":
            return False
        if str(text or "").strip():
            return False
        return MainWindow._is_pending_stat_line(stat_line)

    def _prune_empty_ai_placeholders(self) -> None:
        if not self._history:
            return
        keep: list[tuple[str, str, str | None, int | None, str | None]] = []
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

    def _toggle_thinking(self, msg_index: int) -> None:
        if msg_index in self._thinking_open_by_msg:
            self._thinking_open_by_msg.remove(msg_index)
        else:
            self._thinking_open_by_msg.add(msg_index)
        self._render_chat(scroll_to_bottom=False)

    def _update_stream_row_widgets(self, msg_index: int, text: str, thinking: str | None) -> bool:
        row = self._message_row_widgets.get(msg_index)
        if row is None or not isValid(row):
            return False
        bar = self.chat_scroll.verticalScrollBar()
        was_near_bottom = bar.value() >= max(0, bar.maximum() - 10)
        bubble_label = getattr(row, "_bubble_label", None)
        if not isinstance(bubble_label, QLabel) or not isValid(bubble_label):
            return False
        if bubble_label.text() != text:
            bubble_label.setText(text)

        think_toggle = getattr(row, "_think_toggle", None)
        think_row = getattr(row, "_think_row", None)
        think_preview_label = getattr(row, "_think_preview_label", None)
        think_full_label = getattr(row, "_think_full_label", None)
        if isinstance(think_toggle, QPushButton) and isValid(think_toggle):
            think_text = str(thinking or "").strip()
            has_thinking = bool(think_text)
            is_open = msg_index in self._thinking_open_by_msg
            is_streaming_thinking = bool(
                self._stream_ai_index == msg_index and self._thread and self._thread.isRunning()
            )
            think_toggle.setVisible(has_thinking)
            think_toggle.setEnabled(has_thinking)
            think_toggle.setText("Мысли ▾" if is_open else "Мысли ▸")
            compact_text = self._preview_last_words(think_text, max_words=7) if is_streaming_thinking else ""
            if isinstance(think_preview_label, QLabel) and isValid(think_preview_label):
                if think_preview_label.text() != compact_text:
                    think_preview_label.setText(compact_text)
                think_preview_label.setVisible(bool(has_thinking and is_streaming_thinking))
            if isinstance(think_full_label, QLabel) and isValid(think_full_label):
                if think_full_label.text() != think_text:
                    think_full_label.setText(think_text)
                think_full_label.setVisible(bool(has_thinking and is_open))
            if isinstance(think_row, QFrame) and isValid(think_row):
                think_row.setVisible(has_thinking)

        self._schedule_chat_sync(scroll_to_bottom=was_near_bottom)
        return True

    def _append_system(self, text: str):
        self._history.append(("system", text, None, None, None))
        self._render_chat(scroll_to_bottom=True)
        self._save_chat_sessions()

    def _append_user(self, text: str):
        self._history.append(("user", text, None, None, None))
        self._render_chat(scroll_to_bottom=True)
        self._save_chat_sessions()

    def _append_ai(self, text: str, stat_line: str, thinking: str | None = None):
        self._history.append(("ai", text, stat_line, None, thinking))
        self._render_chat(scroll_to_bottom=True)
        self._save_chat_sessions()

    def _nearest_user_text_before(self, idx: int) -> str:
        i = idx - 1
        while i >= 0:
            role, text, _, _, _thinking = self._history[i]
            if role == "user":
                return text
            i -= 1
        return self._last_user_text or ""

    def _set_feedback(self, msg_index: int, rating: int) -> None:
        if msg_index < 0 or msg_index >= len(self._history):
            return
        role, text, stat_line, current, thinking = self._history[msg_index]
        if role != "ai":
            return
        norm_rating = 1 if int(rating) > 0 else -1
        if current == norm_rating:
            return
        self._history[msg_index] = (role, text, stat_line, norm_rating, thinking)
        chat = self._active_chat()
        is_incognito = bool(chat.get("incognito", False)) if chat else False
        if not is_incognito:
            try:
                self.api.register_feedback(
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

    @staticmethod
    def _set_label_text_if_changed(label: QLabel, text: str) -> None:
        if label.text() != text:
            label.setText(text)

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

        self._set_label_text_if_changed(self.avg_ms_label, f"Время {ms_to_s_text(avg_ms) if avg_ms else dash}")
        self._set_label_text_if_changed(self.avg_decode_label, f"Decode {ms_to_s_text(avg_decode_ms) if avg_decode_ms else dash}")
        self._set_label_text_if_changed(self.avg_tokens_label, f"Gen {avg_gen:.0f} / Prompt {avg_prompt:.0f}")
        self._set_label_text_if_changed(self.avg_tps_label, f"Tok/s {avg_tps:.1f}" if avg_tps else "Tok/s —")
        self._set_label_text_if_changed(
            self.avg_tflops_label,
            f"TFLOPs ~{est_tflops(avg_tps):.2f}" if (SHOW_TFLOPS_EST and avg_tps) else "TFLOPs —",
        )

    def _tick_realtime_stats(self) -> None:
        running = bool(self._thread and self._thread.isRunning() and self._request_started_perf is not None)
        if not running:
            self._set_label_text_if_changed(
                self.rt_buffer_label,
                f"Буфер: out {len(self._stream_chunk_buffer)} / think {len(self._stream_thinking_buffer)}"
            )
            return

        elapsed = max(0.001, time.perf_counter() - float(self._request_started_perf or 0.0))
        out_chars = self._rt_output_chars
        approx_tokens = max(1, int(out_chars / 4.0)) if out_chars else 0
        tps_est = safe_div(float(approx_tokens), elapsed)
        chunk_rate = safe_div(float(self._rt_chunk_count), elapsed)

        self._set_label_text_if_changed(self.rt_elapsed_label, f"Время: {elapsed:.1f} с")
        self._set_label_text_if_changed(self.rt_tokens_out_label, f"Токены out (оценка): ~{approx_tokens}")
        self._set_label_text_if_changed(
            self.rt_tokens_think_label,
            f"Токены think (оценка): ~{int(self._rt_thinking_chars / 4.0)}",
        )
        self._set_label_text_if_changed(self.rt_tps_label, f"Tok/s (оценка): {tps_est:.1f}")
        self._set_label_text_if_changed(self.rt_chunks_label, f"Chunk/s: {chunk_rate:.1f}")
        self._set_label_text_if_changed(self.rt_flush_label, f"UI flush: {self._rt_last_flush_ms:.1f} мс")
        self._set_label_text_if_changed(
            self.rt_buffer_label,
            f"Буфер: out {len(self._stream_chunk_buffer)} / think {len(self._stream_thinking_buffer)}"
        )

        self.rt_tps_graph.push_value(tps_est)
        self.rt_chunk_graph.push_value(chunk_rate)
        self.rt_ui_graph.push_value(self._rt_last_flush_ms)

    @Slot()
    def on_clear(self):
        self._history = []
        self._append_system("\u0427\u0430\u0442 \u043e\u0447\u0438\u0449\u0435\u043d.")
        self._last_user_text = None

        self.n_answers = 0
        self.sum_ms = 0.0
        self.sum_decode_ms = 0.0
        self.sum_eval = 0
        self.sum_prompt = 0
        self.sum_tps = 0.0

        self._set_status("\u0413\u043e\u0442\u043e\u0432\u043e")
        self.avg_ms_label.setText("Время —")
        self.avg_decode_label.setText("Decode —")
        self.avg_tokens_label.setText("Gen — / Prompt —")
        self.avg_tps_label.setText("Tok/s —")
        self.avg_tflops_label.setText("TFLOPs —")
        self.rt_elapsed_label.setText("Время: —")
        self.rt_tokens_out_label.setText("Токены out (оценка): —")
        self.rt_tokens_think_label.setText("Токены think (оценка): —")
        self.rt_tps_label.setText("Tok/s (оценка): —")
        self.rt_chunks_label.setText("Chunk/s: —")
        self.rt_flush_label.setText("UI flush: —")
        self.rt_buffer_label.setText("Буфер: out 0 / think 0")
        self.rt_tps_graph.clear_values()
        self.rt_chunk_graph.clear_values()
        self.rt_ui_graph.clear_values()

    @Slot()
    def on_stop(self):
        if self._worker:
            self._worker.request_cancel()
        self._set_status("\u041e\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430\u2026 (\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0431\u0443\u0434\u0435\u0442 \u043f\u0440\u043e\u0438\u0433\u043d\u043e\u0440\u0438\u0440\u043e\u0432\u0430\u043d)")

    @Slot()
    def on_send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return

        cmd = text.lower()
        if cmd in {"/nothink", "/think"}:
            target = cmd == "/think"
            try:
                actual = self.api.set_thinking_enabled(target)
                self.think_toggle.blockSignals(True)
                self.think_toggle.setChecked(bool(actual))
                self.think_toggle.blockSignals(False)
                self._save_ui_state()
                self._append_system("Мысли: включены." if actual else "Мысли: выключены.")
                self.input.clear()
                self._set_status("Готово")
            except ApiClientError as exc:
                QMessageBox.warning(self, "API", str(exc))
            return

        if not self._active_chat():
            self._sync_active_session_from_history()
            chat = self._new_chat_payload(incognito=False)
            self._chat_sessions.append(chat)
            self._active_chat_id = str(chat["id"])
            self._refresh_chat_selector()
            self._apply_active_chat_to_ui(scroll_to_bottom=True)
            self._save_chat_sessions()

        started = self._start_request(text, show_user=True)
        if not started:
            return

        self.input.clear()
        self._last_user_text = text

    def _get_voice_stt(self):
        if self._voice_stt is None:
            self._voice_stt = build_stt_engine()
        return self._voice_stt

    def _copy_audio_into_input_dir(self, source_path: Path) -> Path:
        MMIS_VOICE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        src = source_path.expanduser().resolve()
        input_dir = MMIS_VOICE_INPUT_DIR.resolve()
        if src.parent == input_dir:
            return src

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = MMIS_VOICE_INPUT_DIR / f"{stamp}_{src.name}"
        shutil.copy2(src, dst)
        return dst

    def _latest_ai_text(self) -> str:
        for role, text, _, _, _thinking in reversed(self._history):
            if role == "ai" and str(text or "").strip():
                return str(text).strip()
        return ""

    @Slot(int)
    def _on_voice_tone_changed(self, _index: int):
        self._voice_tts_voice = str(self.voice_tone_combo.currentData() or self._voice_tts_voice)

    @Slot(object, str)
    def _on_media_error(self, _error, error_text: str):
        if error_text:
            self._set_status("Ошибка")
            QMessageBox.warning(self, "Плеер", f"Ошибка воспроизведения:\n{error_text}")

    def _speak_text_in_app(self, text: str) -> bool:
        rate = self._int_to_percent(self.voice_rate_spin.value())
        volume = self._int_to_percent(self.voice_volume_spin.value())
        tts = build_tts_engine(tts_voice=self._voice_tts_voice, tts_rate=rate, tts_volume=volume)
        saved = tts.synthesize_to_file(text, self._voice_reply_cache_path)
        self._media_player.stop()
        self._media_player.setSource(QUrl.fromLocalFile(str(saved.resolve())))
        self._media_player.play()
        return True

    @Slot()
    def on_voice_input_file(self):
        if not self._active_chat():
            QMessageBox.information(self, "Чаты", "Сначала создай чат кнопкой «Новый».")
            return
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "Подожди", "Сначала дождись завершения текущей генерации.")
            return

        selected, _flt = QFileDialog.getOpenFileName(
            self,
            "Выбери голосовой файл",
            str(MMIS_VOICE_INPUT_DIR),
            "Audio (*.wav *.mp3 *.m4a *.ogg *.flac);;All files (*.*)",
        )
        if not selected:
            return

        try:
            source_path = self._copy_audio_into_input_dir(Path(selected))
            self._set_status("Распознавание голоса…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            recognized = (self._get_voice_stt().transcribe_file(source_path) or "").strip()
        except Exception as exc:
            QMessageBox.critical(self, "Голос", f"Не удалось распознать файл:\n{exc}")
            self._set_status("Ошибка")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not recognized:
            QMessageBox.warning(self, "Голос", "Распознавание вернуло пустой текст.")
            self._set_status("Готово")
            return

        self.input.setPlainText(recognized)
        self.input.setFocus()
        self._set_status("Готово")

    @Slot()
    def on_voice_speak_last_ai(self):
        text = self._latest_ai_text()
        if not text:
            QMessageBox.information(self, "Озвучка", "Пока нет ответа AI для озвучки.")
            return

        try:
            self._set_status("Озвучка ответа…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            self._speak_text_in_app(text)
        except Exception as exc:
            QMessageBox.critical(self, "Озвучка", f"Не удалось озвучить ответ:\n{exc}")
            self._set_status("Ошибка")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._set_status("Готово")

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
        self._history.append(("ai", "", "…", None, None))
        self._stream_ai_index = len(self._history) - 1
        self._render_chat(scroll_to_bottom=True)
        self._save_chat_sessions()
        self._stream_chunk_buffer = ""
        self._stream_thinking_buffer = ""
        self._stream_flush_timer.stop()
        self._request_started_perf = time.perf_counter()
        self._rt_chunk_count = 0
        self._rt_output_chars = 0
        self._rt_thinking_chars = 0
        self._rt_last_flush_ms = 0.0

        self._set_status("\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f\u2026")
        self.btn_model_refresh.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.think_toggle.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_voice_input.setEnabled(False)
        self.btn_voice_speak.setEnabled(False)
        self.voice_tone_combo.setEnabled(False)
        self.voice_rate_spin.setEnabled(False)
        self.voice_volume_spin.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._thread = QThread()
        chat = self._active_chat()
        store_turn = not bool(chat.get("incognito", False)) if chat else True
        self._request_primary_model = str(self.model_combo.currentText() or self.api.get_runtime_model() or "").strip()
        self._worker = ReplyWorker(
            self.api,
            user_text=user_text,
            store_turn=store_turn,
            think=bool(self.think_toggle.isChecked()),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.chunk.connect(self._on_reply_chunk)
        self._worker.thinking_chunk.connect(self._on_reply_thinking_chunk)
        self._worker.finished.connect(self._on_reply)
        self._worker.errored.connect(self._on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.errored.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()
        return True

    @Slot(str)
    def _on_reply_chunk(self, piece: str):
        piece_text = piece or ""
        self._stream_chunk_buffer += piece_text
        self._rt_output_chars += len(piece_text)
        if piece_text:
            self._rt_chunk_count += 1
        if not self._stream_flush_timer.isActive():
            self._stream_flush_timer.start()

    @Slot(str)
    def _on_reply_thinking_chunk(self, piece: str):
        piece_text = piece or ""
        self._stream_thinking_buffer += piece_text
        self._rt_thinking_chars += len(piece_text)
        if piece_text:
            self._rt_chunk_count += 1
        if not self._stream_flush_timer.isActive():
            self._stream_flush_timer.start()

    def _flush_stream_chunks(self):
        t0 = time.perf_counter()
        if (not self._stream_chunk_buffer and not self._stream_thinking_buffer) or self._stream_ai_index is None:
            self._stream_chunk_buffer = ""
            self._stream_thinking_buffer = ""
            return
        idx = self._stream_ai_index
        if idx < 0 or idx >= len(self._history):
            self._stream_chunk_buffer = ""
            self._stream_thinking_buffer = ""
            return
        role, text, stat_line, feedback, thinking = self._history[idx]
        if role != "ai":
            self._stream_chunk_buffer = ""
            self._stream_thinking_buffer = ""
            self._stream_ai_index = None
            return
        new_text = self._normalize_stream_text((text or "") + self._stream_chunk_buffer) if self._stream_chunk_buffer else (text or "")
        new_thinking = ((thinking or "") + self._stream_thinking_buffer).strip()
        self._history[idx] = (role, new_text, stat_line, feedback, new_thinking or None)
        self._stream_chunk_buffer = ""
        self._stream_thinking_buffer = ""

        # Fast path: update only current streaming label to avoid full rerender flicker.
        if not self._update_stream_row_widgets(idx, new_text, new_thinking or None):
            self._render_chat(scroll_to_bottom=True)
        self._rt_last_flush_ms = (time.perf_counter() - t0) * 1000.0

    @Slot(object)
    def _on_reply(self, res: ReplyResult):
        self._flush_stream_chunks()
        stats = res.stats or {}
        stat_line, ms, decode_ms, gen, prompt, tps = self._format_stats_line(stats)
        if self._stream_ai_index is not None and 0 <= self._stream_ai_index < len(self._history):
            i = self._stream_ai_index
            role, text, _, feedback, _old_thinking = self._history[i]
            if role == "ai":
                final_text = res.text or text
                self._history[i] = (role, final_text, stat_line, feedback, (res.thinking or ""))
                self._stream_ai_index = None
                self._render_chat(scroll_to_bottom=True)
            else:
                # Defensive fallback: never overwrite user/system rows with AI output.
                self._append_ai(res.text, stat_line, thinking=res.thinking)
                self._stream_ai_index = None
        else:
            self._append_ai(res.text, stat_line, thinking=res.thinking)
        self._save_chat_sessions()

        served_model = str(res.model or stats.get("served_model") or "").strip()
        primary_model = str(self._request_primary_model or self.model_combo.currentText() or "").strip()
        should_count = (not primary_model) or (not served_model) or (served_model == primary_model)
        if should_count:
            if ms:
                self.sum_ms += ms
            if decode_ms:
                self.sum_decode_ms += decode_ms
            self.sum_eval += gen
            self.sum_prompt += prompt
            self.sum_tps += tps
            self.n_answers += 1

        self._update_side_stats()
        self._tick_realtime_stats()
        actual_model = served_model
        self._sync_model_combo_to(actual_model)
        self._update_model_label(actual_model)
        self._set_status("\u0413\u043e\u0442\u043e\u0432\u043e")
        if self.voice_auto_tts.isChecked():
            try:
                self._set_status("Озвучка ответа…")
                self._speak_text_in_app(res.text or self._latest_ai_text())
                self._set_status("Готово")
            except Exception:
                self._set_status("Ошибка")

    @Slot(str)
    def _on_error(self, tb: str):
        self._set_status("\u041e\u0448\u0438\u0431\u043a\u0430")
        self._stream_ai_index = None
        self._stream_chunk_buffer = ""
        self._stream_thinking_buffer = ""
        self._stream_flush_timer.stop()
        self._request_started_perf = None
        QMessageBox.critical(self, "\u041e\u0448\u0438\u0431\u043a\u0430", tb)

    @Slot()
    def _cleanup_thread(self):
        self._stream_flush_timer.stop()
        self._stream_chunk_buffer = ""
        self._stream_thinking_buffer = ""
        self._stream_ai_index = None
        self._request_started_perf = None
        self._prune_empty_ai_placeholders()
        self.btn_send.setEnabled(True)
        self.btn_model_refresh.setEnabled(True)
        self.model_combo.setEnabled(self.model_combo.count() > 0)
        self.think_toggle.setEnabled(True)
        self.btn_voice_input.setEnabled(True)
        self.btn_voice_speak.setEnabled(True)
        self.voice_tone_combo.setEnabled(True)
        self.voice_rate_spin.setEnabled(True)
        self.voice_volume_spin.setEnabled(True)
        self.btn_stop.setEnabled(False)
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
    lock_dir = QStandardPaths.writableLocation(QStandardPaths.TempLocation) or str(Path.cwd())
    lock = QLockFile(str(Path(lock_dir) / "mmis_desktop.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(1):
        return
    win = MainWindow()
    win.show()
    code = app.exec()
    try:
        if lock.isLocked():
            lock.unlock()
    except Exception:
        pass
    sys.exit(code)


if __name__ == "__main__":
    main()






