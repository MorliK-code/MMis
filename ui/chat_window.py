from __future__ import annotations

import json
import base64
import mimetypes
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent

    def _norm_path(value: str) -> str:
        return os.path.normcase(os.path.abspath(str(value)))

    if all(_norm_path(path) != _norm_path(str(_ROOT)) for path in sys.path):
        sys.path.insert(0, str(_ROOT))
    sys.path = [path for path in sys.path if _norm_path(path) != _norm_path(str(_HERE))]
    _config_mod = sys.modules.get("config")
    if _config_mod is not None and not hasattr(_config_mod, "__path__"):
        sys.modules.pop("config", None)

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, QSignalBlocker, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QCursor
from PySide6.QtMultimedia import QAudioInput, QAudioOutput, QMediaCaptureSession, QMediaFormat, QMediaPlayer, QMediaRecorder
from PySide6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QStackedWidget, QToolButton, QVBoxLayout, QWidget, QDialog
from ui.widgets.message_box import MmisMessageBox
from core.chat_store import ChatStore

from ui.settings_sync_service import load_settings_payload
from ui.client_config_store import (
    get_active_account_data_dir,
    get_client_data_dir,
    get_selected_base_url,
    load_ui_state,
    save_ui_state,
    set_last_persona_name,
    get_ollama_models_cache,
    set_ollama_models_cache,
)
from ui.auth_client_store import (
    activate_auth_session,
    clear_auth_state,
    get_auth_display_name,
    get_auth_token,
    list_auth_sessions,
    save_auth_state,
)
try:
    from llm.tokenizer import estimate_tokens
except ImportError:
    def estimate_tokens(text: str) -> int:
        """Simple fallback token estimator when llm module is missing."""
        if not text:
            return 0
        # Rough estimate: ~4 chars per token for English, ~2 for Russian/Cyrillic
        return len(text) // 3 + 1
import ui.chat_shell as proto
from ui.api_client import ApiClient, ApiClientError
from ui.chat_sessions import SINGLE_VISIBLE_CHAT_ID, SINGLE_VISIBLE_CHAT_TITLE
from ui.chat_sessions import collapse_to_single_visible_chat, history_to_serializable
from ui.chat_sessions import load_sessions as load_chat_sessions
from ui.chat_sessions import make_new_chat_payload, now_iso as chat_now_iso
from ui.chat_sessions import save_sessions as save_chat_sessions
from ui.settings_window import SettingsWindow
from ui.settings_schema import dotted_get
try:
    from modules.voice.voice_manager import VoiceState
except ImportError:
    class VoiceState:
        IDLE = "idle"
        LISTENING = "listening"
        PROCESSING = "processing"
        SPEAKING = "speaking"
        ERROR = "error"
try:
    from ui.voice_adapter import build_stt_config, build_stt_engine, build_tts_config, build_tts_engine, build_voice_manager
except ImportError:
    # Fallback for voice manager if modules.voice is missing
    def build_voice_manager():
        class DummyVoiceManager:
            def set_callbacks(self, **kwargs): pass
            def get_state(self): return "idle"
        return DummyVoiceManager()
from ui.voice_panel import VoicePanel
from ui.widgets.memory_inspector_panel import MemoryInspectorPanel
from ui.workers import ReplyResult, ReplyWorker


def _get_portable_config():
    """Helper to get config values without depending on config.settings."""
    try:
        payload, _ = load_settings_payload()
        return {
            "memory_dir": payload.get("memory", {}).get("memory_dir") or "data/memory_core",
            "voice_input_dir": payload.get("voice", {}).get("paths", {}).get("input_dir"),
            "voice_output_dir": payload.get("voice", {}).get("paths", {}).get("output_dir"),
            "voice_tts_voice": payload.get("voice", {}).get("tts", {}).get("voice") or "ru-RU-DmitryNeural",
            "voice_tts_rate": payload.get("voice", {}).get("tts", {}).get("rate") or "+0%",
            "voice_tts_volume": payload.get("voice", {}).get("tts", {}).get("volume") or "+0%",
            "voice_auto_speak_replies": payload.get("voice", {}).get("auto_speak_replies", True),
            "data_dir": payload.get("paths", {}).get("data_dir") or "data"
        }
    except Exception:
        return {
            "memory_dir": "data/memory_core",
            "voice_tts_voice": "ru-RU-DmitryNeural",
            "voice_auto_speak_replies": True,
            "data_dir": "data"
        }

_cfg_dict = _get_portable_config()
MemoryStorageDir = Path(_cfg_dict["memory_dir"]).expanduser().resolve()
DATA_DIR = Path(_cfg_dict["data_dir"]).expanduser().resolve()
STATE_DIR = MemoryStorageDir
CHARACTERS_RUNTIME_DIR = MemoryStorageDir / "characters_runtime"
CHARACTER_SPECS_DIR = DATA_DIR / "specs" / "characters"
MMIS_VOICE_INPUT_DIR = Path(_cfg_dict["voice_input_dir"] or (MemoryStorageDir / "voice" / "input")).expanduser().resolve()
MMIS_VOICE_OUTPUT_DIR = Path(_cfg_dict["voice_output_dir"] or (MemoryStorageDir / "voice" / "output")).expanduser().resolve()
MMIS_VOICE_TTS_VOICE = str(_cfg_dict["voice_tts_voice"])
MMIS_VOICE_TTS_RATE = str(_cfg_dict["voice_tts_rate"])
MMIS_VOICE_TTS_VOLUME = str(_cfg_dict["voice_tts_volume"])
MMIS_VOICE_AUTO_SPEAK = bool(_cfg_dict["voice_auto_speak_replies"])

HistoryRow = tuple[str, str, str | None, int | None, str | None]
STAT_THINKING_PREFIX = "__thinking_ms__="
TEXT_FILE_SUFFIXES = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".log",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}
IMAGE_FILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_INLINE_FILE_BYTES = 64 * 1024
MAX_PENDING_ATTACHMENTS = 8
ATTACHMENT_EMPTY_PROMPT = "Проанализируй вложения."
HISTORY_INITIAL_RENDER_LIMIT = 12
HISTORY_LAZY_BATCH_SIZE = 12
HISTORY_SCROLL_LOAD_THRESHOLD_PX = 24


class AccountSwitchRow(QLabel):
    clicked = Signal()

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._pressed = False
        self._hovered = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_state()

    def _apply_state(self) -> None:
        if self._pressed:
            color = "#f3f4f6"
        elif self._hovered:
            color = "#a294d2"
        else:
            color = "#8f96a3"
        self.setStyleSheet(
            f"color: {color};"
            "background: transparent;"
            "padding: 0;"
            "font-family: Cascadia Code;"
            "font-size: 12px;"
        )

    def enterEvent(self, event) -> None:
        self._hovered = True
        self._apply_state()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self._pressed = False
        self._apply_state()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self._apply_state()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            was_pressed = self._pressed
            self._pressed = False
            self._apply_state()
            if was_pressed and self.rect().contains(event.position().toPoint()):
                self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ActiveAccountRow(QWidget):
    logoutClicked = Signal()

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._expanded_width = 0
        self._anim = QPropertyAnimation()
        self.setMouseTracking(True)

        self.state = QLabel(text, self)
        self.state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state.setFixedHeight(22)
        self.state.setStyleSheet(
            "color: #d9d0ff;"
            "background: rgba(139,92,246,.10);"
            "border: 1px solid rgba(139,92,246,.18);"
            "border-radius: 6px;"
            "padding: 0 8px;"
            "font-weight: 500;"
            "font-family: Cascadia Code;"
            "font-size: 11px;"
        )

        self.logout_btn = QPushButton("Выйти", self)
        self.logout_btn.setObjectName("dangerButton")
        self.logout_btn.setMaximumWidth(0)
        self.logout_btn.setMinimumWidth(0)
        self.logout_btn.clicked.connect(self.logoutClicked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.state, 1)
        layout.addWidget(self.logout_btn, 0)

    def _set_logout_open(self, open_: bool) -> None:
        if self._expanded_width <= 0:
            self._expanded_width = max(48, self.logout_btn.sizeHint().width())
        target = self._expanded_width if open_ else 0
        if self.logout_btn.maximumWidth() == target:
            return
        self._anim.stop()
        self._anim = QPropertyAnimation(self.logout_btn, b"maximumWidth", self)
        self._anim.setDuration(145)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.setStartValue(self.logout_btn.maximumWidth())
        self._anim.setEndValue(target)
        self._anim.start()

    def enterEvent(self, event) -> None:
        self._set_logout_open(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_logout_open(False)
        super().leaveEvent(event)


class LogoutRevealFilter(QObject):
    def __init__(self, state: QWidget, button: QPushButton, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._button = button
        self._target_width = max(48, button.sizeHint().width())
        self._animation = QPropertyAnimation()
        button.setMaximumWidth(0)
        button.setMinimumWidth(0)
        state.installEventFilter(self)
        button.installEventFilter(self)

    def eventFilter(self, watched: QObject, event) -> bool:
        if event.type() == QEvent.Type.Enter:
            self._set_open(True)
        elif event.type() == QEvent.Type.Leave:
            QTimer.singleShot(25, self._hide_if_outside)
        return super().eventFilter(watched, event)

    def _contains_cursor(self, widget: QWidget) -> bool:
        return widget.isVisible() and widget.rect().contains(widget.mapFromGlobal(QCursor.pos()))

    def _hide_if_outside(self) -> None:
        if not self._contains_cursor(self._state) and not self._contains_cursor(self._button):
            self._set_open(False)

    def _set_open(self, open_: bool) -> None:
        target = self._target_width if open_ else 0
        if self._button.maximumWidth() == target:
            return
        self._animation.stop()
        self._animation = QPropertyAnimation(self._button, b"maximumWidth", self)
        self._animation.setDuration(145)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.setStartValue(self._button.maximumWidth())
        self._animation.setEndValue(target)
        self._animation.start()


def _configure_qt_startup() -> None:
    if not sys.platform.startswith("win"):
        return
    os.environ.setdefault("QT_FONT_DPI", "96")
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor)
    except Exception:
        pass
    try:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi, True)
    except Exception:
        pass


def _display_role(role: str) -> str:
    return "assistant" if str(role or "").strip() == "ai" else "user"


def _trim_title(text: str, limit: int = 40) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return "Чат"
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


class ChatWindow(proto.ExactChatWindow):
    def __init__(self):
        self._project_root = Path(__file__).resolve().parents[1]
        self._legacy_sessions_dirs = [
            self._project_root / "data" / "memory_core" / "ui_chats",
            self._project_root / "data" / "ui_chats",
            self._project_root / "tasks",
        ]
        self._legacy_ui_state_paths = [
            self._project_root / "data" / "memory_core" / "ui_state.json",
            self._project_root / "data" / "ui_state.json",
        ]
        self._refresh_client_paths()
        if self._attachments_dir is not None:
            self._attachments_dir.mkdir(parents=True, exist_ok=True)
        self._chat_sessions: list[dict] = []
        self._active_chat_id: str | None = None
        self._account_scope_id = self._current_account_scope_id()
        self._history: list[HistoryRow] = []
        self._lazy_history_start_index = 0
        self._lazy_history_button: QWidget | None = None
        self._lazy_history_loading = False
        self._lazy_history_scroll_connected = False
        self._active_model: str = ""
        self._available_models: list[str] = []
        self._thinking_enabled = True
        self._verbose_enabled = False
        self._json_mode_enabled = False
        self._screen_enabled = False
        self._web_mode = "auto"
        self._pending_history_index: int | None = None
        self._pending_user_text: str = ""
        self._stream_follow_scroll = False
        self._force_stream_follow_scroll = False
        self._scroll_bottom_queued = False
        self._queued_scroll_follow_only = False
        self._pending_elapsed_timer: QTimer | None = None
        self._regen_scroll_anchor: QWidget | None = None
        self._suppress_lazy_history_until = 0.0
        self._regenerate_scroll_spacer: QWidget | None = None
        self._last_memory_debug_snapshot: dict = {}
        self._runtime_flags_dirty_until = 0.0
        self._last_active_topic_title: str = ""
        self._last_persona_name: str = "Default"
        self._api_persona_name: str = ""
        self._voice_stt = None
        self._voice_tts_voice = MMIS_VOICE_TTS_VOICE
        self._voice_rate_percent = self._percent_to_int(MMIS_VOICE_TTS_RATE, default=0)
        self._voice_volume_percent = self._percent_to_int(MMIS_VOICE_TTS_VOLUME, default=0)
        self._voice_cache_dir = MMIS_VOICE_OUTPUT_DIR / ".cache"
        self._voice_cache_dir.mkdir(parents=True, exist_ok=True)
        self._voice_reply_cache_path = self._voice_cache_dir / "reply_live.wav"
        self._voice_manager = build_voice_manager()
        self._voice_stack: QStackedWidget | None = None
        self._chat_page: QWidget | None = None
        self._voice_page: VoicePanel | None = None
        self._voice_capture_session: QMediaCaptureSession | None = None
        self._voice_recorder: QMediaRecorder | None = None
        self._voice_audio_input: QAudioInput | None = None
        self._voice_recording_path: Path | None = None
        self._voice_send_pending = False
        self._inspector_window: QMainWindow | None = None
        self._inspector_panel: MemoryInspectorPanel | None = None
        self._settings_window: SettingsWindow | None = None
        self._chat_rail_button = None
        self._voice_rail_button = None
        self._file_rail_button = None
        self._memory_rail_button = None
        self._settings_rail_button = None
        self._mode_chip = None
        self._screen_chip = None
        self._web_chip = None
        self._think_chip = None
        self._verbose_chip = None
        self._json_chip = None
        self._pending_attachments: list[dict] = []
        self._attachment_row: QWidget | None = None
        self._attachment_layout: QHBoxLayout | None = None
        self._load_ui_state()
        super().__init__()
        self._pending_elapsed_timer = QTimer(self)
        self._pending_elapsed_timer.setInterval(250)
        self._pending_elapsed_timer.timeout.connect(self._refresh_pending_elapsed_perf)
        self.setWindowTitle("MMis - Chat")
        if hasattr(self, "_metrics_timer"):
            self._metrics_timer.timeout.connect(self._refresh_persona_label)
        if self.api is None:
            self.api = ApiClient()
        self._audio_output = QAudioOutput(self)
        self._media_player = QMediaPlayer(self)
        self._media_player.setAudioOutput(self._audio_output)
        self._media_player.errorOccurred.connect(self._on_media_error)
        self._voice_manager.set_callbacks(
            on_final=lambda text: QTimer.singleShot(0, lambda: self._on_voice_final_text(text)),
            on_state=lambda state: QTimer.singleShot(0, lambda: self._set_voice_state_ui(state)),
        )
        self._voice_state_timer = QTimer(self)
        self._voice_state_timer.setInterval(250)
        self._voice_state_timer.timeout.connect(lambda: self._set_voice_state_ui(self._voice_manager.get_state()))
        self._voice_state_timer.start()
        self._sync_runtime_controls()
        self._apply_context_chips()

    def _build_ui(self):
        super()._build_ui()
        self._install_voice_stack()
        if not self._lazy_history_scroll_connected:
            self.scroll.verticalScrollBar().valueChanged.connect(self._on_history_scroll_value_changed)
            self._lazy_history_scroll_connected = True
        self._refresh_account_button()
        self._refresh_persona_label()
        self.input.setPlaceholderText("Напиши сообщение")
        self.search_btn.setText("Очистить")
        self.clear_btn.setText("Инспектор")
        self.plus_btn.set_button_padding(0, 0, 0, 0)
        self.plus_btn.setFixedSize(24, 24)
        self.mic_btn.set_button_padding(0, 0, 0, 0)
        self.mic_btn.setFixedSize(24, 24)
        self.plus_btn.setToolTip("Прикрепить файл в промпт")
        self.mic_btn.setToolTip("Голосовой файл")
        self.functions_btn.setToolTip("Функции модели")
        self.models_button.setToolTip("Модели")
        self.send_btn.setToolTip("Отправить сообщение")
        self._rebuild_mode_row()
        self._install_attachment_row()
        self._wire_rail_buttons()

    def _install_voice_stack(self) -> None:
        if self._voice_stack is not None:
            return
        root_layout = self.centralWidget().layout()
        body_layout = root_layout.itemAt(1).layout() if root_layout and root_layout.count() > 1 else None
        if body_layout is None or body_layout.count() < 2:
            return
        item = body_layout.takeAt(1)
        chat_page = item.widget() if item is not None else None
        if chat_page is None:
            return
        stack = QStackedWidget(self.centralWidget())
        stack.setObjectName("main_mode_stack")
        self._chat_page = chat_page
        self._voice_page = self._build_voice_page()
        stack.addWidget(chat_page)
        stack.addWidget(self._voice_page)
        body_layout.insertWidget(1, stack, 1)
        self._voice_stack = stack
        self._set_voice_state_ui(VoiceState.IDLE)

    def _build_voice_page(self) -> VoicePanel:
        panel = VoicePanel(self)
        panel.listenPressed.connect(self._start_voice_recording)
        panel.listenReleased.connect(self._stop_voice_recording)
        panel.repeatRequested.connect(self.on_voice_speak_last_ai)
        panel.stopRequested.connect(self._stop_voice_mode_audio)
        panel.closeRequested.connect(self._close_voice_mode)
        panel.fileRequested.connect(self._voice_input_file_to_message)
        return panel

    @staticmethod
    def _safe_character_id(value) -> str:
        raw = str(value or "").strip().lower()
        out = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        return out or ""

    @staticmethod
    def _read_json_payload(path: Path) -> dict:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _payload_character_id(cls, payload: dict) -> str:
        for key in ("active_character_id", "character_id", "id"):
            cid = cls._safe_character_id(payload.get(key))
            if cid:
                return cid
        return ""

    @staticmethod
    def _fallback_character_name(character_id: str) -> str:
        cleaned = str(character_id or "").strip().lower()
        if cleaned in {"asya", "ася", "асья"}:
            return "Ася"
        if cleaned in {"", "default", "assistant", "none", "null"}:
            return "Default"
        if cleaned in {"asya", "асья", "ася"}:
            return "Ася"
        return str(character_id or "Default").strip() or "Default"

    @staticmethod
    def _is_generic_persona_name(name: str) -> bool:
        cleaned = str(name or "").strip().lower()
        return cleaned in {"", "default", "assistant", "ассистент", "помощник", "none", "null"}

    @staticmethod
    def _payload_character_name(payload: dict) -> str:
        for key in ("name", "display_name", "title"):
            name = str(payload.get(key) or "").strip()
            if name:
                return name
        return ""

    def _resolve_active_character_id(self) -> str:
        state = self._read_json_payload(STATE_DIR / "brain_state.json")
        cid = self._payload_character_id(state)
        if cid:
            return cid
        global_state = state.get("global")
        if isinstance(global_state, dict):
            cid = self._payload_character_id(global_state)
            if cid:
                return cid

        manifest = self._read_json_payload(CHARACTERS_RUNTIME_DIR / "manifest.json")
        cid = self._payload_character_id(manifest)
        if cid:
            return cid
        for row in list(manifest.get("characters") or []):
            if not isinstance(row, dict) or row.get("enabled") is False:
                continue
            cid = self._payload_character_id(row)
            if cid:
                return cid
        return "default"

    def _cache_persona_name(self, name: str) -> None:
        clean = str(name or "").strip()
        if not clean:
            return

        # Не перетираем уже известное имя пустым/служебным Default,
        # если раньше уже было нормальное имя.
        current = str(getattr(self, "_last_persona_name", "") or "").strip()
        if self._is_generic_persona_name(clean) and not self._is_generic_persona_name(current):
            return

        self._api_persona_name = clean
        self._last_persona_name = clean
        set_last_persona_name(clean)
        self._refresh_persona_label()

    def _resolve_persona_display_name(self) -> str:
        if self._api_persona_name and not self._is_generic_persona_name(self._api_persona_name):
            return self._api_persona_name

        cached = str(getattr(self, "_last_persona_name", "") or "").strip()
        if cached and not self._is_generic_persona_name(cached):
            return cached

        character_id = self._resolve_active_character_id()
        manifest = self._read_json_payload(CHARACTERS_RUNTIME_DIR / "manifest.json")
        character_paths = [
            CHARACTERS_RUNTIME_DIR / character_id / "character.json",
            CHARACTER_SPECS_DIR / character_id / "character.json",
        ]
        
        name = ""
        for path in character_paths:
            name = self._payload_character_name(self._read_json_payload(path))
            if name:
                break
        
        if not name:
            for row in list(manifest.get("characters") or []):
                if not isinstance(row, dict):
                    continue
                if self._safe_character_id(row.get("id")) != character_id:
                    continue
                name = self._payload_character_name(row)
                if name:
                    break
        
        if not name:
            name = self._fallback_character_name(character_id)

        if name:
            self._last_persona_name = name
        
        return str(getattr(self, "_last_persona_name", "") or "Default").strip() or "Default"

    def _refresh_persona_label(self) -> None:
        label = getattr(self, "persona_label", None)
        if label is None:
            return
        name = self._resolve_persona_display_name()
        if name and label.text() != name:
            label.setText(name)
            label.adjustSize()
            self._save_ui_state()

    def _refresh_account_button(self) -> None:
        button = getattr(self, "account_btn", None)
        if button is None:
            return

        name = get_auth_display_name() or "Войти"
        button.setText(name)
        button.setToolTip("Аккаунт MMis" if name != "Войти" else "Войти в аккаунт MMis")
        button.adjustSize()

    def _reload_account_scope(self) -> None:
        if getattr(self, "_worker", None) is not None and self._worker.isRunning():
            MmisMessageBox.information(self, "Подожди", "Сначала дождись завершения генерации.")
            return
        self._stop_background_threads(include_reply=False)
        self._reset_chat_scope_view()
        self._refresh_client_paths()
        self._account_scope_id = self._current_account_scope_id()
        self._apply_account_environment()
        if self.api is not None:
            self.api.set_base_url(get_selected_base_url())
        self._load_ui_state()
        self._sync_runtime_controls()
        if getattr(self, "_settings_window", None) is not None:
            try:
                self._settings_window.reload()
            except Exception:
                pass
        self._refresh_account_button()
        self._load_or_init_chat_sessions()
        self._refresh_persona_label()
        self._apply_context_chips()

    def _open_account_dialog(self) -> None:
        self._show_account_menu()

    def _show_account_menu(self) -> None:
        from PySide6.QtWidgets import QMenu

        button = getattr(self, "account_btn", None)
        if button is None:
            return
        name = get_auth_display_name() or "Аккаунт"
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: rgba(15,18,22,.96); color: #f3f4f6; border: 1px solid rgba(255,255,255,.08); }"
            "QMenu::item { padding: 7px 18px; }"
            "QMenu::item:selected { background: rgba(139,92,246,.18); }"
        )
        menu.addAction(name).setEnabled(False)
        menu.addSeparator()
        logout_action = menu.addAction("Выйти")
        action = menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        if action != logout_action:
            return
        if self.api:
            try:
                self.api.auth_logout()
            except Exception:
                clear_auth_state()
        else:
            clear_auth_state()
        self._refresh_account_button()

    def _show_login_dialog(self) -> None:
        from PySide6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Вход в MMis")
        dialog.setModal(True)
        dialog.setMinimumWidth(360)
        dialog.setStyleSheet(
            "QDialog { background: rgba(15,18,22,.98); color: #f3f4f6; }"
            "QLineEdit { background: rgba(255,255,255,.04); color: #f3f4f6; border: 1px solid rgba(255,255,255,.08); border-radius: 8px; padding: 7px; }"
            "QPushButton { background: rgba(139,92,246,.14); color: #f3f4f6; border: 1px solid rgba(139,92,246,.22); border-radius: 8px; padding: 7px 12px; }"
        )

        root = QVBoxLayout(dialog)
        title = QLabel("Войти в аккаунт MMis")
        root.addWidget(title)

        form = QFormLayout()
        login_edit = QLineEdit()
        password_edit = QLineEdit()
        password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Логин", login_edit)
        form.addRow("Пароль", password_edit)
        root.addLayout(form)

        error_label = QLabel("")
        error_label.setStyleSheet("color: #fca5a5;")
        root.addWidget(error_label)

        buttons = QHBoxLayout()
        login_btn = QPushButton("Войти")
        register_btn = QPushButton("Создать")
        cancel_btn = QPushButton("Отмена")
        register_btn.setText("Создать пароль")
        buttons.addWidget(login_btn)
        buttons.addWidget(register_btn)
        buttons.addWidget(cancel_btn)
        root.addLayout(buttons)

        def do_login(register: bool = False) -> None:
            login = login_edit.text().strip()
            password = password_edit.text()
            if not self.api:
                error_label.setText("API недоступен")
                return
            try:
                self._save_chat_sessions()
                if register:
                    payload = self.api.auth_register(login, password, display_name=login)
                else:
                    payload = self.api.auth_login(login, password)
                save_auth_state(payload)
                self._reload_account_scope()
                dialog.accept()
            except Exception as exc:
                error_label.setText(str(exc))

        login_btn.clicked.connect(lambda: do_login(False))
        register_btn.clicked.connect(lambda: do_login(True))
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def _show_account_menu(self) -> None:
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QFrame

        button = getattr(self, "account_btn", None)
        if button is None:
            return

        existing = getattr(self, "_account_panel", None)
        if existing is not None and existing.isVisible():
            self._hide_account_panel()
            return
        if existing is not None:
            existing.deleteLater()

        parent = self.centralWidget() or self
        panel = QFrame(parent)
        panel.setObjectName("accountSlidePanel")
        panel.setStyleSheet(
            "QFrame#accountSlidePanel { background: rgba(13,16,19,.97); border: 1px solid rgba(255,255,255,.08); border-radius: 12px; }"
            "QFrame#accountPanelDivider { background: rgba(255,255,255,.08); border: 0; max-height: 1px; min-height: 1px; }"
            "QLabel { color: #f3f4f6; font-family: Cascadia Code; }"
            "QLabel#accountPanelSection { color: #8b949e; font-size: 10px; font-weight: 700; letter-spacing: 0px; }"
            "QPushButton { min-height: 22px; padding: 0 8px; border-radius: 6px; font-size: 11px; font-weight: 600; font-family: Cascadia Code; }"
            "QPushButton#accountCommandButton { border: 1px solid rgba(255,255,255,.07); background: rgba(255,255,255,.035); color: #c4b5fd; }"
            "QPushButton#accountCommandButton:hover { background: rgba(255,255,255,.07); border-color: rgba(139,92,246,.24); color: #ffffff; }"
            "QPushButton#dangerButton { border: 1px solid rgba(248,113,113,.18); background: rgba(248,113,113,.06); color: #fca5a5; }"
            "QPushButton#dangerButton:hover { background: rgba(248,113,113,.12); border-color: rgba(248,113,113,.32); color: #fecaca; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        title = QLabel("Аккаунт MMis")
        title.setText("Аккаунты")
        title.setStyleSheet("font-family: Cascadia Code; font-weight: 700;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        name = get_auth_display_name()
        state = QLabel(name or "В аккаунт не выполнен вход")
        state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        state.setFixedHeight(22)
        state.setStyleSheet(
            "color: #d9d0ff;"
            "background: rgba(139,92,246,.10);"
            "border: 1px solid rgba(139,92,246,.18);"
            "border-radius: 6px;"
            "padding: 0 8px;"
            "font-weight: 500;"
            "font-family: Cascadia Code;"
            "font-size: 11px;"
        )
        state.adjustSize()
        logout_btn = None
        active_row = QHBoxLayout()
        active_row.setContentsMargins(0, 0, 0, 0)
        active_row.setSpacing(6)
        active_row.addWidget(state, 1)
        if get_auth_token():
            logout_btn = QPushButton("Выйти")
            logout_btn.setObjectName("dangerButton")
            self._account_logout_reveal_filter = LogoutRevealFilter(state, logout_btn, panel)
            active_row.addWidget(logout_btn, 0)
        layout.addLayout(active_row)

        sessions = list_auth_sessions()
        current_id = self._current_account_scope_id()
        switch_sessions = [
            session
            for session in sessions
            if str(session.get("account_id") or "").strip()
            and str(session.get("account_id") or "").strip() != current_id
        ]
        for session in switch_sessions:
            account_id = str(session.get("account_id") or "").strip()
            label = str(session.get("display_name") or session.get("login") or account_id).strip()
            switch_btn = AccountSwitchRow(label, panel)
            switch_btn.setToolTip("Перейти на аккаунт")
            layout.addWidget(switch_btn)
            switch_btn.clicked.connect(lambda aid=account_id: self._switch_saved_account(aid))

        add_btn = QPushButton("Добавить")
        divider = QFrame(panel)
        divider.setObjectName("accountPanelDivider")
        layout.addWidget(divider)

        add_btn.setObjectName("accountCommandButton")
        layout.addWidget(add_btn)
        panel_width = 190
        panel_height = 102 + (26 * len(switch_sessions))
        top_left = parent.mapFromGlobal(button.mapToGlobal(button.rect().bottomLeft()))
        x = max(8, min(int(top_left.x()), max(8, parent.width() - panel_width - 8)))
        y = int(top_left.y() + 8)
        panel.setGeometry(x, y, panel_width, 0)
        panel.show()
        panel.raise_()

        animation = QPropertyAnimation(panel, b"geometry", panel)
        animation.setDuration(150)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.setStartValue(QRect(x, y, panel_width, 0))
        animation.setEndValue(QRect(x, y, panel_width, panel_height))
        animation.start()
        self._account_panel = panel
        self._account_panel_animation = animation

        def open_login() -> None:
            self._hide_account_panel()
            self._show_login_dialog()

        def logout() -> None:
            self._hide_account_panel()
            self._save_chat_sessions()
            if self.api:
                try:
                    self.api.auth_logout()
                except Exception:
                    clear_auth_state(forget_current=True)
            else:
                clear_auth_state(forget_current=True)
            self._reload_account_scope()

        add_btn.clicked.connect(open_login)
        if logout_btn is not None:
            logout_btn.clicked.connect(logout)

    def _switch_saved_account(self, account_id: str) -> None:
        self._hide_account_panel()
        self._save_chat_sessions()
        try:
            activate_auth_session(account_id)
        except Exception as exc:
            MmisMessageBox.warning(self, "Аккаунт", f"Не удалось перейти на аккаунт:\n{exc}")
            return
        self._reload_account_scope()

    def _hide_account_panel(self) -> None:
        panel = getattr(self, "_account_panel", None)
        if panel is None:
            return
        panel.hide()
        panel.deleteLater()
        self._account_panel = None

    def _on_account_clicked(self) -> None:
        self._show_account_menu()

    def _show_login_overlay(self) -> None:
        from PySide6.QtWidgets import QFormLayout, QFrame, QGraphicsBlurEffect, QLineEdit

        existing = getattr(self, "_login_overlay", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            return

        blur_target = self.centralWidget()
        if blur_target is not None:
            effect = QGraphicsBlurEffect(blur_target)
            effect.setBlurRadius(16)
            effect.setBlurHints(QGraphicsBlurEffect.BlurHint.QualityHint)
            blur_target.setGraphicsEffect(effect)
            self._login_blur_target = blur_target

        overlay = QFrame(self)
        overlay.setObjectName("loginOverlay")
        central = self.centralWidget()
        overlay.setGeometry(central.geometry() if central is not None else self.rect())
        overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        overlay.setStyleSheet(
            "QFrame#loginOverlay { background: rgba(5,7,10,.46); }"
            "QFrame#loginPanel { background: rgba(13,16,19,.97); border: 1px solid rgba(255,255,255,.08); border-radius: 12px; }"
            "QLabel { color: #f3f4f6; font-family: Cascadia Code; font-size: 11px; }"
            "QLabel#loginError { color: #fca5a5; }"
            "QLineEdit { background: rgba(255,255,255,.04); color: #f3f4f6; border: 1px solid rgba(255,255,255,.08); border-radius: 7px; padding: 4px 7px; font-family: Cascadia Code; font-size: 11px; min-height: 22px; }"
            "QLineEdit:focus { border-color: rgba(139,92,246,.32); background: rgba(139,92,246,.06); }"
            "QPushButton { background: rgba(139,92,246,.14); color: #f3f4f6; border: 1px solid rgba(139,92,246,.22); border-radius: 7px; padding: 5px 9px; font-family: Cascadia Code; font-size: 11px; }"
            "QPushButton:hover { background: rgba(139,92,246,.22); }"
        )
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(18, 18, 18, 18)
        overlay_layout.setSpacing(0)

        panel = QFrame(overlay)
        panel.setObjectName("loginPanel")
        panel.setFixedWidth(360)
        root = QVBoxLayout(panel)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(8)

        title = QLabel("Войти или добавить аккаунт MMis")
        title.setStyleSheet("font-size: 12px; font-weight: 700;")
        root.addWidget(title)
        form = QFormLayout()
        form.setVerticalSpacing(7)
        form.setHorizontalSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        login_edit = QLineEdit()
        login_edit.setPlaceholderText("Например: admin")
        password_edit = QLineEdit()
        password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Логин", login_edit)
        form.addRow("Пароль", password_edit)
        root.addLayout(form)

        error_label = QLabel("")
        error_label.setObjectName("loginError")
        root.addWidget(error_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        login_btn = QPushButton("Войти")
        register_btn = QPushButton("Создать / задать пароль")
        cancel_btn = QPushButton("Отмена")
        register_btn.setText("Создать пароль")
        buttons.addWidget(login_btn)
        buttons.addWidget(register_btn)
        buttons.addWidget(cancel_btn)
        root.addLayout(buttons)
        overlay_layout.addWidget(panel, 0, Qt.AlignmentFlag.AlignCenter)

        def close_overlay() -> None:
            target = getattr(self, "_login_blur_target", None)
            if target is not None:
                target.setGraphicsEffect(None)
                self._login_blur_target = None
            current = getattr(self, "_login_overlay", None)
            if current is not None:
                current.hide()
                current.deleteLater()
                self._login_overlay = None

        def do_login(register: bool = False) -> None:
            login = login_edit.text().strip()
            password = password_edit.text()
            if not self.api:
                error_label.setText("API недоступен")
                return
            try:
                self._save_chat_sessions()
                if register:
                    payload = self.api.auth_register(login, password, display_name=login)
                else:
                    payload = self.api.auth_login(login, password)
                save_auth_state(payload)
                self._reload_account_scope()
                close_overlay()
            except Exception as exc:
                error_label.setText(str(exc))

        login_btn.clicked.connect(lambda: do_login(False))
        register_btn.clicked.connect(lambda: do_login(True))
        cancel_btn.clicked.connect(close_overlay)
        self._login_overlay = overlay
        overlay.show()
        overlay.raise_()
        login_edit.setFocus()

    def _show_login_dialog(self) -> None:
        self._show_login_overlay()
        return
        from PySide6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Вход в MMis")
        dialog.setModal(True)
        dialog.setMinimumWidth(390)
        dialog.setStyleSheet(
            "QDialog { background: rgba(15,18,22,.98); color: #f3f4f6; }"
            "QLabel { color: #f3f4f6; }"
            "QLineEdit { background: rgba(255,255,255,.04); color: #f3f4f6; border: 1px solid rgba(255,255,255,.08); border-radius: 8px; padding: 7px; }"
            "QPushButton { background: rgba(139,92,246,.14); color: #f3f4f6; border: 1px solid rgba(139,92,246,.22); border-radius: 8px; padding: 7px 12px; }"
            "QPushButton:hover { background: rgba(139,92,246,.22); }"
        )

        root = QVBoxLayout(dialog)
        title = QLabel("Войти или добавить аккаунт MMis")
        root.addWidget(title)
        hint = QLabel("Для admin нажми «Создать / задать пароль», если пароль ещё не был задан.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9ca3af;")
        root.addWidget(hint)

        form = QFormLayout()
        login_edit = QLineEdit()
        login_edit.setPlaceholderText("Например: admin")
        password_edit = QLineEdit()
        password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Логин", login_edit)
        form.addRow("Пароль", password_edit)
        root.addLayout(form)

        error_label = QLabel("")
        error_label.setStyleSheet("color: #fca5a5;")
        root.addWidget(error_label)

        buttons = QHBoxLayout()
        login_btn = QPushButton("Войти")
        register_btn = QPushButton("Создать / задать пароль")
        cancel_btn = QPushButton("Отмена")
        buttons.addWidget(login_btn)
        buttons.addWidget(register_btn)
        buttons.addWidget(cancel_btn)
        root.addLayout(buttons)

        def do_login(register: bool = False) -> None:
            login = login_edit.text().strip()
            password = password_edit.text()
            if not self.api:
                error_label.setText("API недоступен")
                return
            try:
                self._save_chat_sessions()
                if register:
                    payload = self.api.auth_register(login, password, display_name=login)
                else:
                    payload = self.api.auth_login(login, password)
                save_auth_state(payload)
                self._reload_account_scope()
                dialog.accept()
            except Exception as exc:
                error_label.setText(str(exc))

        login_btn.clicked.connect(lambda: do_login(False))
        register_btn.clicked.connect(lambda: do_login(True))
        cancel_btn.clicked.connect(dialog.reject)
        dialog.exec()

    def _build_models_popup(self, anchor: QWidget) -> proto.PopupFrame:
        return super()._build_models_popup(anchor)

    def _build_functions_popup(self, anchor: QWidget) -> proto.PopupFrame:
        popup = proto.PopupFrame(anchor, width=230, line_orientation="vertical")
        lay = QVBoxLayout(popup)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        title = proto.CrispLabel("Управление функциями")
        title.setFont(proto._button_font(pixel_size=11, weight=proto.QFont.Weight.Medium))
        title.set_text_color(proto.TEXT)
        lay.addWidget(title)

        commands = proto.HoverSubmenuRow(
            "команды",
            popup,
            icon_text="⌘",
            submenu_title="Команды",
            submenu_rows=[
                ("think", "think", self._thinking_enabled),
                ("verbose", "verbose", self._verbose_enabled),
                ("json", "json", self._json_mode_enabled),
            ],
        )
        lay.addWidget(commands)

        screen_row = proto.function_row("screen", icon_text="▣")
        self.screen_toggle = proto.ToggleSwitch(self._screen_enabled)
        screen_row.layout().addWidget(self.screen_toggle)
        lay.addWidget(screen_row)

        web_row = proto.function_row("web", icon_text="🌐")
        web_row.layout().setContentsMargins(10, 5, 8, 7)
        self.web_mode_selector = proto.WebModeSelector(self._web_mode)
        web_row.layout().addWidget(self.web_mode_selector, 0, proto.Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(web_row)

        self.think_toggle = commands.controls["think"]
        self.verbose_toggle = commands.controls["verbose"]
        self.json_toggle = commands.controls["json"]
        self.think_toggle.toggled.connect(self._on_think_toggled)
        self.verbose_toggle.toggled.connect(self._on_verbose_toggled)
        self.json_toggle.toggled.connect(self._on_json_toggled)
        self.screen_toggle.toggled.connect(self._on_screen_toggled)
        self.web_mode_selector.modeChanged.connect(self._on_web_mode_changed)
        self._commands_row = commands
        return popup

    def _bind_popups(self) -> None:
        super()._bind_popups()
        try:
            self.clear_btn.clicked.disconnect(self._clear_messages)
        except Exception:
            pass
        self.search_btn.clicked.connect(self._clear_messages)
        self.clear_btn.clicked.connect(self._toggle_inspector)
        self.plus_btn.clicked.connect(self._attach_file)
        self.mic_btn.clicked.connect(self._open_voice_mode)

    def _refresh_client_paths(self) -> None:
        account_root = get_active_account_data_dir()
        client_dir = get_client_data_dir()
        self._account_root = account_root
        self._client_data_dir = client_dir
        self._sessions_dir = (account_root / "chats" / "ui_chats") if account_root else client_dir / "ui_chats"
        self._sessions_index_path = self._sessions_dir / "index.json"
        self._legacy_sessions_path = client_dir / "ui_chats.json"
        self._ui_state_path = (account_root / "ui_state.json") if account_root else client_dir / "ui_state.json"
        self._legacy_visible_chat_backup_path = self._sessions_dir / "_legacy_multi_chat_backup.json"
        self._chat_store = ChatStore(account_root / "chats" / "chats.db") if account_root else None
        self._attachments_dir = (account_root / "attachments") if account_root else client_dir / "attachments"
        if self._attachments_dir is not None:
            self._attachments_dir.mkdir(parents=True, exist_ok=True)

    def _is_account_isolated_scope(self) -> bool:
        return bool(get_auth_token())

    def _current_account_scope_id(self) -> str:
        try:
            from ui.auth_client_store import load_auth_state

            state = load_auth_state()
            return str(state.get("account_id") or "guest").strip() or "guest"
        except Exception:
            return "guest"

    def _reset_chat_scope_view(self) -> None:
        self._chat_sessions = []
        self._active_chat_id = None
        self._history = []
        self._pending_history_index = None
        self._pending_user_text = ""
        self._lazy_history_start_index = 0
        self._lazy_history_loading = False
        self._remove_lazy_history_button()
        if hasattr(self, "messages_layout"):
            self._clear_message_widgets()

    def _apply_account_environment(self) -> None:
        account_root = getattr(self, "_account_root", None)
        account_id = self._current_account_scope_id()
        if account_root is None or not account_id or account_id == "guest":
            os.environ.pop("MMIS_ACTIVE_ACCOUNT_ID", None)
            os.environ.pop("MMIS_CONFIG_FILE", None)
            return
        accounts_dir = account_root.parent
        os.environ["MMIS_ACTIVE_ACCOUNT_ID"] = account_id
        os.environ["MMIS_ACCOUNTS_DIR"] = str(accounts_dir)
        os.environ["MMIS_CONFIG_FILE"] = str(account_root / "config" / "config.json")

    def _stop_qthread(self, attr_name: str, *, cancel: bool = False, terminate: bool = False, wait_ms: int = 1500) -> None:
        worker = getattr(self, attr_name, None)
        if worker is None:
            return
        try:
            if cancel and hasattr(worker, "request_cancel"):
                worker.request_cancel()
        except Exception:
            pass
        try:
            worker.disconnect()
        except Exception:
            pass
        try:
            if worker.isRunning():
                if terminate:
                    worker.terminate()
                worker.wait(int(wait_ms))
        except Exception:
            pass
        try:
            if not worker.isRunning():
                worker.deleteLater()
        except Exception:
            pass
        setattr(self, attr_name, None)

    def _stop_background_threads(self, *, include_reply: bool) -> None:
        timer = getattr(self, "_runtime_sync_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        if include_reply:
            self._stop_pending_elapsed_timer()
            self._stop_qthread("_worker", cancel=True, terminate=True, wait_ms=1500)
        self._stop_qthread("_status_worker", terminate=True, wait_ms=1500)
        self._backend_status_inflight = False
        self._stop_qthread("_runtime_sync_worker", terminate=True, wait_ms=1500)

    def _append_demo_messages(self) -> None:
        self._load_or_init_chat_sessions()

    def _clear_message_widgets(self) -> None:
        for index in range(self.messages_layout.count() - 1, -1, -1):
            item = self.messages_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            taken = self.messages_layout.takeAt(index)
            if taken is not None and widget is not None:
                widget.deleteLater()
        self._lazy_history_button = None
        sync = getattr(self, "_sync_messages_view_height", None)
        if callable(sync):
            sync()

    def _insert_message_bubble(self, bubble: QWidget) -> None:
        wire = getattr(self, "_wire_regenerate_bubble", None)
        if callable(wire):
            wire(bubble)
        insert_index = self.messages_layout.count()
        self.messages_layout.insertWidget(insert_index, bubble)
        sync = getattr(self, "_schedule_messages_view_height_sync", None)
        if callable(sync):
            sync()

    def _last_user_text_before(self, index: int) -> str:
        start = max(0, min(int(index), len(self._history)))
        for role, text, _stat_line, _feedback, _thinking in reversed(self._history[:start]):
            if role == "user" and str(text or "").strip():
                return str(text or "")
        return ""

    def _history_bubble_for_index(self, index: int, last_user_text: str) -> tuple[QWidget, bool, str]:
        role, text, stat_line, feedback, thinking = self._history[index]
        upgraded_stat_line = self._upgrade_legacy_verbose_stat_line(
            role=role,
            text=text,
            stat_line=stat_line,
            thinking=thinking,
            user_text=last_user_text,
        )
        if str(role or "").strip() == "ai":
            upgraded_stat_line = self._normalize_stat_line_time_units(upgraded_stat_line)
        history_changed = upgraded_stat_line != stat_line
        if history_changed:
            self._history[index] = (role, text, upgraded_stat_line, feedback, thinking)
            stat_line = upgraded_stat_line
        thinking_ms = self._extract_thinking_ms_from_stat_line(stat_line)
        perf_items = self._split_stat_line(stat_line)
        display_text = str(text or "")
        extracted_attachments = []
        if display_text and "\n\nВложения:\n-" in display_text:
            parts = display_text.split("\n\nВложения:\n")
            display_text = parts[0]
            for line in parts[1].split("\n"):
                if line.startswith("- "):
                    raw_name = line[2:]
                    import re
                    raw_name = re.sub(r"\((?:[^,]+,\s*)*(?=[^,]+$)", "(", raw_name)
                    extracted_attachments.append({"name": raw_name})

        bubble = proto.MessageBubble(
            _display_role(role),
            display_text,
            thinking or "",
            thinking_ms if str(thinking or "").strip() else "",
            perf_items,
            show_thinking_header=bool(str(thinking or "").strip()),
            attachments=extracted_attachments,
        )
        if role == "user":
            bubble.setProperty("history_index", index)
        if role == "user" and str(text or "").strip():
            last_user_text = str(text or "")
        return bubble, history_changed, last_user_text

    def _render_history_range(self, start: int, end: int, *, insert_at: int | None = None) -> bool:
        history_changed = False
        start = max(0, min(int(start), len(self._history)))
        end = max(start, min(int(end), len(self._history)))
        last_user_text = self._last_user_text_before(start)
        inserted = 0
        for index in range(start, end):
            bubble, row_changed, last_user_text = self._history_bubble_for_index(index, last_user_text)
            history_changed = history_changed or row_changed
            if insert_at is None:
                self._insert_message_bubble(bubble)
            else:
                self.messages_layout.insertWidget(insert_at + inserted, bubble)
                inserted += 1
        if insert_at is not None:
            sync = getattr(self, "_schedule_messages_view_height_sync", None)
            if callable(sync):
                sync()
        return history_changed

    def _remove_lazy_history_button(self) -> None:
        button = self._lazy_history_button
        if button is None:
            return
        layout = self.messages_layout
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is not None and item.widget() is button:
                layout.takeAt(index)
                break
        self._lazy_history_button = None
        button.deleteLater()

    def _sync_lazy_history_button(self) -> None:
        if self._lazy_history_start_index <= 0:
            self._remove_lazy_history_button()
            return
        if self._lazy_history_button is not None:
            return
        button = proto.HoverButton("Загрузить ещё историю")
        button.setToolTip("Показать более старые сообщения")
        button.clicked.connect(self._load_older_history_batch)
        self._lazy_history_button = button
        self.messages_layout.insertWidget(0, button, 0, Qt.AlignmentFlag.AlignHCenter)
        sync = getattr(self, "_schedule_messages_view_height_sync", None)
        if callable(sync):
            sync()

    @Slot(int)
    def _on_history_scroll_value_changed(self, value: int) -> None:
        bar = self.scroll.verticalScrollBar()
        
        # Dynamically update whether we should stick to the bottom during generation
        if getattr(self, "_worker", None) is not None and self._worker.isRunning():
            if getattr(self, "_force_stream_follow_scroll", False):
                self._stream_follow_scroll = True
            else:
                self._stream_follow_scroll = self._should_follow_stream_scroll(value, bar.maximum())
            
        if self._lazy_history_loading or self._lazy_history_start_index <= 0:
            return
        if getattr(self, "_regen_scroll_anchor", None) is not None:
            return
        if time.monotonic() < float(getattr(self, "_suppress_lazy_history_until", 0.0) or 0.0):
            return
        if bar.maximum() <= 0:
            return
        if int(value) <= HISTORY_SCROLL_LOAD_THRESHOLD_PX:
            self._schedule_load_older_history()

    def _schedule_load_older_history(self) -> None:
        if self._lazy_history_loading:
            return
        self._lazy_history_loading = True
        QTimer.singleShot(0, self._load_older_history_batch)

    @Slot()
    def _load_older_history_batch(self) -> None:
        if self._lazy_history_start_index <= 0:
            self._lazy_history_loading = False
            self._sync_lazy_history_button()
            return
        old_start = int(self._lazy_history_start_index)
        new_start = max(0, old_start - HISTORY_LAZY_BATCH_SIZE)
        self._remove_lazy_history_button()
        history_changed = self._render_history_range(new_start, old_start, insert_at=0)
        self._lazy_history_start_index = new_start
        self._sync_lazy_history_button()
        if history_changed:
            self._save_chat_sessions()
        sync = getattr(self, "_sync_messages_view_height", None)
        if callable(sync):
            sync()
        QTimer.singleShot(0, lambda: setattr(self, "_lazy_history_loading", False))

    def _clear_messages(self) -> None:
        if self._worker and self._worker.isRunning():
            MmisMessageBox.information(self, "Подожди", "Сначала дождись завершения генерации.")
            return
        chat = self._active_chat()
        if not chat or not self._history:
            return
        answer = MmisMessageBox.question(
            self,
            "Очистить чат",
            "Удалить историю текущего чата?",
            MmisMessageBox.Yes | MmisMessageBox.No,
            MmisMessageBox.No,
        )
        if answer != MmisMessageBox.Yes:
            return
        self._history = []
        chat["history"] = []
        chat["updated_at"] = chat_now_iso()
        self._render_history()
        self._save_chat_sessions()

    def _render_history(self) -> None:
        self._clear_message_widgets()
        self._lazy_history_loading = False
        self._lazy_history_start_index = max(0, len(self._history) - HISTORY_INITIAL_RENDER_LIMIT)
        history_changed = self._render_history_range(self._lazy_history_start_index, len(self._history))
        self._sync_lazy_history_button()
        if history_changed:
            self._save_chat_sessions()
        QTimer.singleShot(0, self._scroll_bottom)
        self._apply_context_chips()

    @staticmethod
    def _split_stat_line(stat_line: str | None) -> list[str]:
        raw = str(stat_line or "").strip()
        if not raw:
            return []
        if raw.startswith(STAT_THINKING_PREFIX):
            _prefix, _sep, raw = raw.partition("||")
            raw = raw.strip()
            if not raw:
                return []
        normalized = raw
        for token in ("\u2022", "•", "·", " • ", " ? ", "?"):
            normalized = normalized.replace(token, "|")
        parts = [part.strip() for part in normalized.split("|")]
        return [part for part in parts if part]

    @staticmethod
    def _extract_thinking_ms_from_stat_line(stat_line: str | None) -> str:
        raw = str(stat_line or "").strip()
        if not raw:
            return ""
        if raw.startswith(STAT_THINKING_PREFIX):
            prefix, _sep, _rest = raw.partition("||")
            value = prefix[len(STAT_THINKING_PREFIX) :].strip()
            return value if ChatWindow._ms_value(value) > 0 else ""
        return ""

    @staticmethod
    def _compose_stat_line(perf: list[str], thinking_ms: str | None) -> str | None:
        pieces = [str(item or "").strip() for item in list(perf or []) if str(item or "").strip()]
        encoded_thinking = str(thinking_ms or "").strip()
        if encoded_thinking:
            body = " | ".join(pieces)
            return f"{STAT_THINKING_PREFIX}{encoded_thinking}||{body}" if body else f"{STAT_THINKING_PREFIX}{encoded_thinking}"
        return " | ".join(pieces) if pieces else None

    @staticmethod
    def _format_duration_label(value_ms: int | float) -> str:
        seconds = max(0.0, float(value_ms or 0.0)) / 1000.0
        return f"{seconds:.1f} s"

    @staticmethod
    def _ms_value(value: str | None) -> int:
        text = str(value or "").strip().lower()
        if text.endswith("ms"):
            try:
                return max(0, int(float(text[:-2].strip()) or 0))
            except Exception:
                return 0
        if not text.endswith("s"):
            return 0
        try:
            return max(0, int(round((float(text[:-1].strip()) or 0.0) * 1000.0)))
        except Exception:
            return 0

    @classmethod
    def _normalize_perf_label(cls, item: str) -> str:
        text = str(item or "").strip()
        if not text:
            return ""
        lower = text.lower()
        if lower.startswith("write "):
            value_ms = cls._ms_value(text[6:].strip())
            return f"write {cls._format_duration_label(value_ms)}" if value_ms > 0 else text
        value_ms = cls._ms_value(text)
        return cls._format_duration_label(value_ms) if value_ms > 0 else text

    @classmethod
    def _normalize_stat_line_time_units(cls, stat_line: str | None) -> str | None:
        if not str(stat_line or "").strip():
            return stat_line
        thinking_label = cls._extract_thinking_ms_from_stat_line(stat_line)
        perf_items = cls._split_stat_line(stat_line)
        normalized_perf = [cls._normalize_perf_label(item) for item in perf_items]
        normalized_thinking = cls._normalize_perf_label(thinking_label) if thinking_label else ""
        normalized = cls._compose_stat_line(normalized_perf, normalized_thinking or None)
        return normalized or stat_line

    def _upgrade_legacy_verbose_stat_line(
        self,
        *,
        role: str,
        text: str,
        stat_line: str | None,
        thinking: str | None,
        user_text: str,
    ) -> str | None:
        if str(role or "").strip() != "ai":
            return stat_line
        if not self._verbose_enabled:
            return stat_line
        perf_items = self._split_stat_line(stat_line)
        if len(perf_items) >= 5:
            return stat_line
        thinking_ms = self._extract_thinking_ms_from_stat_line(stat_line)
        elapsed_ms = 0
        if perf_items:
            elapsed_ms = self._ms_value(perf_items[0])
        if elapsed_ms <= 0:
            elapsed_ms = self._ms_value(thinking_ms)
        if elapsed_ms <= 0:
            return stat_line
        local_thinking_ms = self._ms_value(thinking_ms)
        stats = self._complete_verbose_stats(
            {"answer_ms": elapsed_ms, "elapsed_ms": elapsed_ms, "total_duration_ms": elapsed_ms, "verbose_enabled": True},
            user_text=user_text,
            answer_text=str(text or ""),
            fallback_elapsed_ms=elapsed_ms,
            local_answer_ms=elapsed_ms,
            local_thinking_ms=local_thinking_ms,
            verbose_enabled=True,
        )
        resolved_thinking_ms = self._normalize_perf_label(thinking_ms) if thinking_ms and str(thinking or "").strip() else ""
        if not resolved_thinking_ms and str(thinking or "").strip():
            resolved_thinking_ms = self._thinking_ms_label(stats, fallback_ms=local_thinking_ms)
        perf, _stat_line = self._perf_from_stats(stats, fallback_elapsed_ms=elapsed_ms)
        return self._compose_stat_line(perf, resolved_thinking_ms or None)

    def _load_legacy_project_sessions(self) -> tuple[list[dict], str | None]:
        best_chats: list[dict] = []
        best_active_id: str | None = None
        best_history_len = 0

        for sessions_dir in getattr(self, "_legacy_sessions_dirs", []):
            index_path = sessions_dir / "index.json"
            legacy_path = sessions_dir.parent / "ui_chats.json"
            backup_path = sessions_dir / "_legacy_multi_chat_backup.json"
            standalone_chat_path = sessions_dir / "chat.json"
            
            # Try main paths
            try:
                chats, active_id = load_chat_sessions(sessions_dir, index_path, legacy_path)
                history_len = sum(len(chat.get("history") or []) for chat in chats if isinstance(chat, dict))
                if history_len > best_history_len:
                    best_chats = chats
                    best_active_id = active_id
                    best_history_len = history_len
            except Exception:
                pass

            # Try standalone chat.json (often found in task backups)
            if standalone_chat_path.exists():
                try:
                    chat_payload = json.loads(standalone_chat_path.read_text(encoding="utf-8-sig"))
                    if isinstance(chat_payload, dict):
                        # Convert single chat to list
                        chat_payload["history"] = history_from_serializable(chat_payload.get("history"))
                        chats = [chat_payload]
                        history_len = len(chat_payload["history"])
                        if history_len > best_history_len:
                            best_chats = chats
                            best_active_id = str(chat_payload.get("id"))
                            best_history_len = history_len
                except Exception:
                    pass

            # Try backup path specifically
            if backup_path.exists():
                try:
                    # load_chat_sessions can take any file as legacy_path if others don't exist
                    chats, active_id = load_chat_sessions(sessions_dir, sessions_dir / "non-existent.json", backup_path)
                    history_len = sum(len(chat.get("history") or []) for chat in chats if isinstance(chat, dict))
                    if history_len > best_history_len:
                        best_chats = chats
                        best_active_id = active_id
                        best_history_len = history_len
                except Exception:
                    pass

        return best_chats, best_active_id

    def _load_or_init_chat_sessions(self) -> None:
        if getattr(self, "_chat_store", None) is not None:
            self._load_or_init_chat_sessions_from_store()
            return

        loaded, active_id = load_chat_sessions(
            self._sessions_dir,
            self._sessions_index_path,
            self._legacy_sessions_path,
        )

        current_history_len = sum(len(chat.get("history") or []) for chat in loaded if isinstance(chat, dict))

        if current_history_len <= 0 and not self._is_account_isolated_scope():
            legacy_loaded, legacy_active_id = self._load_legacy_project_sessions()
            legacy_history_len = sum(len(chat.get("history") or []) for chat in legacy_loaded if isinstance(chat, dict))
            if legacy_history_len > 0:
                loaded = legacy_loaded
                active_id = legacy_active_id
        if current_history_len <= 0 and not loaded:
            loaded, active_id = self._load_chat_store_sessions()

        self._backup_legacy_visible_chats(loaded, active_id)
        chosen = collapse_to_single_visible_chat(loaded, active_id)
        self._chat_sessions = [chosen]
        self._active_chat_id = str(chosen.get("id") or SINGLE_VISIBLE_CHAT_ID)
        self._history = list(chosen.get("history") or [])
        self._render_history()
        self._save_chat_sessions()

    def _load_or_init_chat_sessions_from_store(self) -> None:
        json_loaded, json_active_id = load_chat_sessions(
            self._sessions_dir,
            self._sessions_index_path,
            self._legacy_sessions_path,
        )
        json_history_len = self._chat_history_len(json_loaded)
        if json_history_len <= 0 and not self._is_account_isolated_scope():
            legacy_loaded, legacy_active_id = self._load_legacy_project_sessions()
            legacy_history_len = self._chat_history_len(legacy_loaded)
            if legacy_history_len > json_history_len:
                json_loaded = legacy_loaded
                json_active_id = legacy_active_id
                json_history_len = legacy_history_len

        store_loaded, store_active_id = self._load_chat_store_sessions()
        store_history_len = self._chat_history_len(store_loaded)

        if json_history_len > store_history_len:
            self._backup_legacy_visible_chats(json_loaded, json_active_id)
            migrated = collapse_to_single_visible_chat(json_loaded, json_active_id)
            self._chat_sessions = [migrated]
            self._active_chat_id = str(migrated.get("id") or SINGLE_VISIBLE_CHAT_ID)
            self._history = list(migrated.get("history") or [])
            self._mirror_chat_sessions_to_store()
            store_loaded, store_active_id = self._load_chat_store_sessions()

        if store_loaded:
            chosen = collapse_to_single_visible_chat(store_loaded, store_active_id)
        else:
            chosen = self._new_chat_payload()

        self._chat_sessions = [chosen]
        self._active_chat_id = str(chosen.get("id") or SINGLE_VISIBLE_CHAT_ID)
        self._history = list(chosen.get("history") or [])
        self._render_history()
        self._save_chat_sessions()

    def _load_chat_store_sessions(self) -> tuple[list[dict], str | None]:
        store = getattr(self, "_chat_store", None)
        if store is None:
            return [], None
        try:
            chats = []
            for chat in store.list_chats(include_archived=False):
                messages = []
                for message in store.list_messages(chat.chat_id):
                    role = "ai" if message.role == "assistant" else message.role
                    meta = dict(message.metadata or {})
                    messages.append((
                        role,
                        message.text,
                        str(meta.get("stat_line") or "") or None,
                        meta.get("feedback"),
                        str(meta.get("thinking") or "") or None,
                    ))
                chats.append({
                    "id": chat.chat_id,
                    "title": chat.title,
                    "created_at": self._iso_from_epoch(chat.created_at),
                    "updated_at": self._iso_from_epoch(chat.updated_at),
                    "history": messages,
                })
            active = str(chats[0].get("id") or "") if chats else None
            return chats, active
        except Exception:
            return [], None

    @staticmethod
    def _chat_history_len(chats: list[dict]) -> int:
        return sum(len(chat.get("history") or []) for chat in list(chats or []) if isinstance(chat, dict))

    @staticmethod
    def _iso_from_epoch(value: float | int | str | None) -> str:
        try:
            return datetime.fromtimestamp(float(value or 0.0)).isoformat(timespec="seconds")
        except Exception:
            return chat_now_iso()

    def _new_chat_payload(self) -> dict:
        payload = make_new_chat_payload(existing_count=1, title=SINGLE_VISIBLE_CHAT_TITLE)
        payload["id"] = SINGLE_VISIBLE_CHAT_ID
        payload["title"] = SINGLE_VISIBLE_CHAT_TITLE
        payload["history"] = []
        return payload

    def _backup_legacy_visible_chats(self, chats: list[dict], active_id: str | None) -> None:
        needs_backup = len(list(chats or [])) > 1 or any(
            str((chat or {}).get("id") or "").strip() not in {"", SINGLE_VISIBLE_CHAT_ID}
            for chat in list(chats or [])
            if isinstance(chat, dict)
        )
        if not needs_backup or self._legacy_visible_chat_backup_path.exists():
            return
        try:
            payload = {
                "created_at": chat_now_iso(),
                "active_chat_id": str(active_id or ""),
                "chats": [
                    {
                        "id": str(chat.get("id") or ""),
                        "title": str(chat.get("title") or ""),
                        "created_at": str(chat.get("created_at") or ""),
                        "updated_at": str(chat.get("updated_at") or ""),
                        "history": history_to_serializable(list(chat.get("history") or [])),
                    }
                    for chat in list(chats or [])
                    if isinstance(chat, dict)
                ],
            }
            self._sessions_dir.mkdir(parents=True, exist_ok=True)
            self._legacy_visible_chat_backup_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _chat_by_id(self, chat_id: str | None) -> dict | None:
        if not chat_id:
            return None
        cid = str(chat_id)
        for chat in self._chat_sessions:
            if str(chat.get("id") or "") == cid:
                return chat
        return None

    def _active_chat(self) -> dict | None:
        return self._chat_by_id(self._active_chat_id)

    def _sync_active_chat_from_history(self) -> None:
        chat = self._active_chat()
        if chat is None:
            return
        chat["history"] = list(self._history)
        chat["updated_at"] = chat_now_iso()

    def _save_chat_sessions(self) -> None:
        if getattr(self, "_account_scope_id", "") != self._current_account_scope_id():
            return
        self._sync_active_chat_from_history()
        active_chat = self._active_chat()
        active_id = None
        if active_chat and not bool(active_chat.get("incognito", False)):
            active_id = str(active_chat.get("id") or "") or None
        try:
            if getattr(self, "_chat_store", None) is not None:
                self._mirror_chat_sessions_to_store()
                self._write_chat_store_backup(active_id)
            else:
                save_chat_sessions(self._sessions_dir, self._sessions_index_path, self._chat_sessions, active_id)
        except Exception:
            pass

    def _mirror_chat_sessions_to_store(self) -> None:
        store = getattr(self, "_chat_store", None)
        if store is None:
            return
        active_ids: set[str] = set()
        for chat in list(self._chat_sessions or []):
            if not isinstance(chat, dict) or bool(chat.get("incognito", False)):
                continue
            chat_id = str(chat.get("id") or "").strip()
            if not chat_id:
                continue
            active_ids.add(chat_id)
            store.upsert_chat(
                chat_id=chat_id,
                title=str(chat.get("title") or "Чат"),
                persona_id=str(chat.get("persona_id") or "default"),
            )
            rows = []
            for index, row in enumerate(list(chat.get("history") or [])):
                try:
                    role, text, stat_line, feedback, thinking = row
                except Exception:
                    continue
                rows.append({
                    "message_id": f"{chat_id}:{index}",
                    "role": "assistant" if str(role) == "ai" else str(role),
                    "text": str(text or ""),
                    "metadata": {
                        "stat_line": str(stat_line or ""),
                        "feedback": feedback,
                        "thinking": str(thinking or ""),
                    },
                })
            store.replace_messages(chat_id, rows)
        if hasattr(store, "archive_chats_except"):
            store.archive_chats_except(active_ids)

    def _write_chat_store_backup(self, active_id: str | None) -> None:
        try:
            self._sessions_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self._sessions_dir / "_chatstore_backup.json"
            payload = {
                "version": 1,
                "source": "ChatStore",
                "created_at": chat_now_iso(),
                "active_chat_id": active_id,
                "chats": [
                    {
                        "id": str(chat.get("id") or ""),
                        "title": str(chat.get("title") or ""),
                        "created_at": str(chat.get("created_at") or ""),
                        "updated_at": str(chat.get("updated_at") or ""),
                        "history": history_to_serializable(list(chat.get("history") or [])),
                    }
                    for chat in list(self._chat_sessions or [])
                    if isinstance(chat, dict) and not bool(chat.get("incognito", False))
                ],
            }
            backup_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _ensure_active_chat(self) -> dict:
        chat = self._active_chat()
        if chat is not None:
            return chat
        chat = self._new_chat_payload()
        self._chat_sessions = [chat]
        self._active_chat_id = str(chat.get("id") or "")
        self._history = []
        self._render_history()
        self._save_chat_sessions()
        return chat

    def _apply_context_chips(self) -> None:
        chat = self._active_chat()
        title = self._resolve_active_topic_title()
        self._apply_chip_style(self.topic_chip, f"активная тема: {title}", False)
        self._apply_chip_style(self._mode_chip, "режим: chat", False)
        self._apply_chip_style(self._screen_chip, "screen", False)
        self._apply_chip_style(self._web_chip, f"web: {self._web_mode}", False)
        self._apply_chip_style(self._think_chip, "think", False)
        self._apply_chip_style(self._verbose_chip, "verbose", self._verbose_enabled)
        self._apply_chip_style(self._json_chip, "json", self._json_mode_enabled)
        if self._screen_chip is not None:
            self._screen_chip.setVisible(bool(self._screen_enabled))
        if self._web_chip is not None:
            self._web_chip.setVisible(self._web_mode != "off")
        if self._think_chip is not None:
            self._think_chip.setVisible(bool(self._thinking_enabled))
        if self._verbose_chip is not None:
            self._verbose_chip.setVisible(bool(self._verbose_enabled))
        if self._json_chip is not None:
            self._json_chip.setVisible(bool(self._json_mode_enabled))
        self._refresh_persona_label()

    @staticmethod
    def _apply_chip_style(chip, text: str, active: bool) -> None:
        if chip is None:
            return
        chip.setText(text)
        chip.apply_chip_style(active)

    def _rebuild_mode_row(self) -> None:
        modes_layout = self._mode_row_layout()
        if modes_layout is None:
            return
        while modes_layout.count():
            item = modes_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.topic_chip = proto.Chip("активная тема: Чат", active=True)
        self._mode_chip = proto.Chip("режим: chat")
        self._screen_chip = proto.Chip("screen")
        self._web_chip = proto.Chip("web: auto")
        self._think_chip = proto.Chip("think")
        self._verbose_chip = proto.Chip("verbose")
        self._json_chip = proto.Chip("json")
        for chip in (
            self.topic_chip,
            self._mode_chip,
            self._screen_chip,
            self._web_chip,
            self._think_chip,
            self._verbose_chip,
            self._json_chip,
        ):
            modes_layout.addWidget(chip)
        modes_layout.addStretch(1)
        self._apply_context_chips()

    def _mode_row_layout(self):
        chat_frame = self._chat_page
        if chat_frame is None:
            root_layout = self.centralWidget().layout()
            body_layout = root_layout.itemAt(1).layout() if root_layout and root_layout.count() > 1 else None
            chat_frame = body_layout.itemAt(1).widget() if body_layout and body_layout.count() > 1 else None
        chat_layout = chat_frame.layout() if chat_frame is not None else None
        modes_frame = chat_layout.itemAt(1).widget() if chat_layout and chat_layout.count() > 1 else None
        return modes_frame.layout() if modes_frame is not None else None

    def _wire_rail_buttons(self) -> None:
        root_layout = self.centralWidget().layout()
        body_layout = root_layout.itemAt(1).layout() if root_layout and root_layout.count() > 1 else None
        rail = body_layout.itemAt(0).widget() if body_layout and body_layout.count() > 0 else None
        rail_layout = rail.layout() if rail is not None else None
        if rail_layout is None:
            return
        widgets = [rail_layout.itemAt(i).widget() for i in range(rail_layout.count())]
        actual_widgets = [widget for widget in widgets if widget is not None]
        if len(actual_widgets) < 6:
            return
        self._chat_rail_button = actual_widgets[0]
        self._voice_rail_button = actual_widgets[1]
        self._file_rail_button = actual_widgets[2]
        self._memory_rail_button = actual_widgets[3]
        self._settings_rail_button = actual_widgets[5]

        self._chat_rail_button.setToolTip("Чат")
        self._chat_rail_button.clicked.connect(self._close_voice_mode)
        self._voice_rail_button.setToolTip("Голосовое общение")
        self._voice_rail_button.clicked.connect(self._open_voice_mode)
        self._file_rail_button.setToolTip("Прикрепить файл")
        self._file_rail_button.clicked.connect(self._attach_file)
        self._memory_rail_button.setToolTip("Память / Inspector")
        self._memory_rail_button.clicked.connect(self._toggle_inspector)
        self._settings_rail_button.setToolTip("Настройки")
        self._settings_rail_button.clicked.connect(self._open_settings_window)
        self._sync_rail_mode_buttons()

    @Slot()
    def _open_settings_window(self) -> None:
        first_open = self._settings_window is None
        if first_open:
            self._settings_window = SettingsWindow(self)
            self._settings_window.saved.connect(self._on_settings_saved)
            self._settings_window.finished.connect(lambda _code: setattr(self, "_settings_window", None))
        else:
            try:
                self._settings_window.reload()
            except Exception as exc:
                MmisMessageBox.warning(self, "Настройки", f"Не удалось обновить настройки:\n{exc}")

        self._settings_window.show()
        self._settings_window.raise_()
        self._settings_window.activateWindow()

    @Slot(object)
    def _on_settings_saved(self, _updates: object) -> None:
        self.api = ApiClient()
        self._apply_context_chips()

    def _install_attachment_row(self) -> None:
        if self._attachment_row is not None:
            return
        composer = self.input.parentWidget()
        composer_layout = composer.layout() if composer is not None else None
        if composer_layout is None:
            return
        from ui.chat_shell import FlowWrap, FlowLayout
        row = FlowWrap(composer)
        row.expand_width_hint = False
        row.setObjectName("attachment_row")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setStyleSheet("QWidget#attachment_row { background: transparent; }")
        layout = FlowLayout(row, margin=2, hspacing=6, vspacing=6, align_right=False)
        row.setLayout(layout)
        self._attachment_row = row
        self._attachment_layout = layout
        composer_layout.insertWidget(1, row)
        row.setVisible(False)
        self._sync_composer_height()

    def _sync_composer_height(self) -> None:
        def _apply_height():
            composer_wrap = self.findChild(QWidget, "composer_wrap")
            if composer_wrap is not None:
                composer = self.input.parentWidget()
                if composer and composer.layout():
                    composer.layout().invalidate()
                if composer_wrap.layout():
                    composer_wrap.layout().invalidate()
                composer_wrap.setFixedHeight(composer_wrap.sizeHint().height())
            sync = getattr(self, "_sync_messages_view_height", None)
            if callable(sync):
                sync()
        QTimer.singleShot(0, _apply_height)

    def _sync_attachment_row(self) -> None:
        row = self._attachment_row
        layout = self._attachment_layout
        if row is None or layout is None:
            return
            
        # Clean up existing chips completely
        while layout.count():
            layout.takeAt(0)
            
        for child in row.findChildren(QWidget, "attachment_chip"):
            child.hide()
            child.setParent(None)
            child.deleteLater()
            
        # Add new chips
        for index, attachment in enumerate(list(self._pending_attachments or [])):
            layout.addWidget(self._build_attachment_chip(index, attachment))
            
        # Toggle visibility
        is_visible = bool(self._pending_attachments)
        row.setVisible(is_visible)
        
        # Invalidate layouts to guarantee correct sizeHint
        composer = self.input.parentWidget()
        if composer and composer.layout():
            composer.layout().invalidate()
        composer_wrap = self.findChild(QWidget, "composer_wrap")
        if composer_wrap and composer_wrap.layout():
            composer_wrap.layout().invalidate()
            
        if is_visible and hasattr(row, "refresh_height"):
            row.refresh_height()
            
        self._sync_composer_height()

    def _build_attachment_chip(self, index: int, attachment: dict) -> QWidget:
        chip = QWidget(self._attachment_row)
        chip.setFixedHeight(22)
        chip.setObjectName("attachment_chip")
        chip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        chip.setStyleSheet(
            """
            QWidget#attachment_chip {
                background: rgba(31, 36, 44, .82);
                border: 1px solid rgba(148, 163, 184, .22);
                border-radius: 5px;
            }
            QLabel {
                color: rgba(226, 232, 240, .92);
                background: transparent;
                font-size: 11px;
            }
            QToolButton {
                color: rgba(148, 163, 184, .95);
                background: transparent;
                border: none;
                padding: 0;
            }
            QToolButton:hover {
                color: rgba(248, 250, 252, .98);
            }
            """
        )
        layout = QHBoxLayout(chip)
        layout.setContentsMargins(6, 1, 4, 1)
        layout.setSpacing(5)
        label = QLabel(self._attachment_chip_text(attachment), chip)
        label.setToolTip(str(attachment.get("path") or attachment.get("name") or ""))
        layout.addWidget(label)
        close_btn = QToolButton(chip)
        close_btn.setText("x")
        close_btn.setFixedSize(16, 16)
        close_btn.setToolTip("Убрать вложение")
        close_btn.clicked.connect(lambda _checked=False, idx=index: self._remove_pending_attachment(idx))
        layout.addWidget(close_btn)
        chip.show()
        chip.layout().activate()
        chip.adjustSize()
        w = max(10, chip.sizeHint().width())
        chip.setFixedWidth(w)
        return chip

    def _remove_pending_attachment(self, index: int) -> None:
        if 0 <= int(index) < len(self._pending_attachments):
            self._pending_attachments.pop(int(index))
            self._sync_attachment_row()
            self.input.setFocus()

    def _sync_runtime_controls(self) -> None:
        if self.api:
            try:
                payload = self.api.health(timeout=5.0)
                if str(payload.get("status") or "").strip().lower() == "ok":
                    self._thinking_enabled = bool(payload.get("thinking_enabled", self._thinking_enabled))
                    self._verbose_enabled = bool(payload.get("verbose_enabled", self._verbose_enabled))
                    self._json_mode_enabled = bool(payload.get("json_mode_enabled", self._json_mode_enabled))
                    self._web_mode = str(payload.get("web_mode") or self._web_mode)
                    if "persona_name" in payload:
                        self._cache_persona_name(payload.get("persona_name"))
                    active_topic = str(payload.get("active_topic_title") or "").strip()
                    if active_topic:
                        self._last_active_topic_title = active_topic
                    model = str(payload.get("model") or "").strip()
                    if model:
                        self._active_model = model
                        if model not in self._available_models:
                            self._available_models.append(model)
                        self._populate_models(self._active_model, self._available_models)
                    self._backend_status_seen_ok = True
                    self._backend_status_failures = 0
            except Exception:
                pass
        self._sync_controls_to_state()
        self._save_ui_state()

    @Slot(object)
    def _on_backend_status_result(self, payload: object) -> None:
        super()._on_backend_status_result(payload)
        row = dict(payload or {}) if isinstance(payload, dict) else {}
        if not bool(row.get("api_ok")):
            return
        model = str(row.get("model") or "").strip()
        if model:
            self._active_model = model
            if model not in self._available_models:
                self._available_models.append(model)
            self._populate_models(self._active_model, self._available_models)
        
        if row.get("api_ok") and "persona_name" in row:
            self._cache_persona_name(row.get("persona_name"))
        if time.monotonic() >= float(getattr(self, "_runtime_flags_dirty_until", 0.0)):
            for attr, key in (
                ("_thinking_enabled", "thinking_enabled"),
                ("_verbose_enabled", "verbose_enabled"),
                ("_json_mode_enabled", "json_mode_enabled"),
            ):
                if key in row:
                    setattr(self, attr, bool(row.get(key)))
            if str(row.get("web_mode") or "").strip():
                self._web_mode = str(row.get("web_mode") or self._web_mode)
            self._sync_controls_to_state()

    def _sync_controls_to_state(self) -> None:
        if hasattr(self, "think_toggle"):
            self.think_toggle.blockSignals(True)
            self.think_toggle.setChecked(self._thinking_enabled)
            self.think_toggle.blockSignals(False)
        if hasattr(self, "verbose_toggle"):
            self.verbose_toggle.blockSignals(True)
            self.verbose_toggle.setChecked(self._verbose_enabled)
            self.verbose_toggle.blockSignals(False)
        if hasattr(self, "json_toggle"):
            self.json_toggle.blockSignals(True)
            self.json_toggle.setChecked(self._json_mode_enabled)
            self.json_toggle.blockSignals(False)
        if hasattr(self, "screen_toggle"):
            self.screen_toggle.blockSignals(True)
            self.screen_toggle.setChecked(self._screen_enabled)
            self.screen_toggle.blockSignals(False)
        if hasattr(self, "web_mode_selector"):
            self.web_mode_selector.blockSignals(True)
            self.web_mode_selector.set_mode(self._web_mode)
            self.web_mode_selector.blockSignals(False)
        self._apply_context_chips()


    def _check_api_status(self) -> None:
        """Background health check and status synchronization."""
        if hasattr(self, "api") and self.api:
            try:
                # We reuse the sync logic to update UI state from backend health
                self._sync_runtime_controls()
            except Exception:
                pass

    def _load_ui_state(self) -> None:
        payload = load_ui_state()
        self._thinking_enabled = bool(payload.get("think_enabled", self._thinking_enabled))
        self._verbose_enabled = bool(payload.get("verbose_enabled", self._verbose_enabled))
        self._json_mode_enabled = bool(payload.get("json_mode_enabled", self._json_mode_enabled))
        self._screen_enabled = bool(payload.get("screen_enabled", self._screen_enabled))
        self._web_mode = str(payload.get("web_mode") or self._web_mode)
        self._last_active_topic_title = str(payload.get("active_topic_title") or "")
        saved_persona = str(payload.get("last_persona_name") or "").strip()
        self._last_persona_name = saved_persona or "Default"

    def _save_ui_state(self) -> None:
        payload = {
            "think_enabled": bool(self._thinking_enabled),
            "verbose_enabled": bool(self._verbose_enabled),
            "json_mode_enabled": bool(self._json_mode_enabled),
            "screen_enabled": bool(self._screen_enabled),
            "web_mode": str(self._web_mode),
            "active_topic_title": str(getattr(self, "_last_active_topic_title", "")),
            "last_persona_name": str(getattr(self, "_last_persona_name", "Default") or "Default"),
        }
        save_ui_state(payload)

    def _load_models(self) -> None:
        runtime = self._active_model or (self.api.get_runtime_model() if self.api else "")
        models = list(self._available_models) or get_ollama_models_cache()

        if self.api:
            try:
                payload = self.api.list_models(timeout=5.0)
                runtime = str(payload.get("runtime_model") or self.api.get_runtime_model() or runtime)
                available = payload.get("available_models") or payload.get("models") or []
                fresh_models = [str(x).strip() for x in available if str(x).strip()]
                if fresh_models:
                    models = fresh_models
                    set_ollama_models_cache(models)
            except Exception:
                models = models or get_ollama_models_cache()

        if not models and runtime:
            models = [runtime]
        if not models:
            models = ["qwen3:8b"]
        self._available_models = models
        self._active_model = runtime or models[0]
        self._populate_models(self._active_model, self._available_models)
        self._apply_context_chips()

    def _set_model(self, model_name: str) -> None:
        actual = str(model_name or "").strip()
        if self.api:
            try:
                self.api.set_model(actual)
                actual = str(self.api.get_runtime_model() or actual)
            except ApiClientError as exc:
                MmisMessageBox.warning(self, "Модель", str(exc))
                return
        if actual and actual not in self._available_models:
            self._available_models.append(actual)
        self._active_model = actual
        self._populate_models(self._active_model, self._available_models)
        self.models_popup.close_popup()
        self._apply_context_chips()

    @Slot(bool)
    def _on_think_toggled(self, checked: bool) -> None:
        self._runtime_flags_dirty_until = time.monotonic() + 2.0
        self._thinking_enabled = bool(checked)
        self._runtime_flags["think"] = self._thinking_enabled
        self._runtime_flags["verbose"] = self._verbose_enabled
        self._runtime_flags["json"] = self._json_mode_enabled
        self._runtime_flags["web_mode"] = self._web_mode
        self._sync_controls_to_state()
        self._save_ui_state()
        self._runtime_sync_timer.start(150)
        QTimer.singleShot(80, self._render_history)
        return
        previous = self._thinking_enabled
        self._thinking_enabled = bool(checked)
        if self.api:
            try:
                self.api.set_thinking_enabled(self._thinking_enabled)
            except ApiClientError as exc:
                self._thinking_enabled = previous
                self._sync_controls_to_state()
                MmisMessageBox.warning(self, "Функции", str(exc))
                return
        self._sync_controls_to_state()
        self._save_ui_state()
        self._render_history()

    @Slot(bool)
    def _on_verbose_toggled(self, checked: bool) -> None:
        self._runtime_flags_dirty_until = time.monotonic() + 2.0
        self._verbose_enabled = bool(checked)
        self._runtime_flags["think"] = self._thinking_enabled
        self._runtime_flags["verbose"] = self._verbose_enabled
        self._runtime_flags["json"] = self._json_mode_enabled
        self._runtime_flags["web_mode"] = self._web_mode
        self._sync_controls_to_state()
        self._save_ui_state()
        self._runtime_sync_timer.start(150)
        QTimer.singleShot(80, self._render_history)
        return
        previous = self._verbose_enabled
        self._verbose_enabled = bool(checked)
        if self.api:
            try:
                self.api.set_verbose_enabled(self._verbose_enabled)
            except ApiClientError as exc:
                self._verbose_enabled = previous
                self._sync_controls_to_state()
                MmisMessageBox.warning(self, "Функции", str(exc))
                return
        self._sync_controls_to_state()
        self._save_ui_state()
        self._render_history()

    @Slot(bool)
    def _on_json_toggled(self, checked: bool) -> None:
        self._runtime_flags_dirty_until = time.monotonic() + 2.0
        self._json_mode_enabled = bool(checked)
        self._runtime_flags["think"] = self._thinking_enabled
        self._runtime_flags["verbose"] = self._verbose_enabled
        self._runtime_flags["json"] = self._json_mode_enabled
        self._runtime_flags["web_mode"] = self._web_mode
        self._sync_controls_to_state()
        self._save_ui_state()
        self._runtime_sync_timer.start(150)
        return
        previous = self._json_mode_enabled
        self._json_mode_enabled = bool(checked)
        if self.api:
            try:
                self.api.set_json_mode_enabled(self._json_mode_enabled)
            except ApiClientError as exc:
                self._json_mode_enabled = previous
                self._sync_controls_to_state()
                MmisMessageBox.warning(self, "Функции", str(exc))
                return
        self._sync_controls_to_state()
        self._save_ui_state()

    @Slot(bool)
    def _on_screen_toggled(self, checked: bool) -> None:
        self._screen_enabled = bool(checked)
        self._sync_controls_to_state()
        self._save_ui_state()

    @Slot(str)
    def _on_web_mode_changed(self, mode: str) -> None:
        previous = self._web_mode
        self._web_mode = str(mode or "auto")
        if self.api:
            try:
                self._web_mode = str(self.api.set_web_mode(self._web_mode) or self._web_mode)
            except ApiClientError as exc:
                self._web_mode = previous
                self._sync_controls_to_state()
                MmisMessageBox.warning(self, "Функции", str(exc))
                return
        self._sync_controls_to_state()
        self._save_ui_state()

    @Slot()
    def _on_new_chat_clicked(self) -> None:
        if self._worker and self._worker.isRunning():
            MmisMessageBox.information(self, "Подожди", "Сначала дождись завершения генерации.")
            return
        self._sync_active_chat_from_history()
        chat = self._new_chat_payload()
        self._chat_sessions.append(chat)
        self._active_chat_id = str(chat.get("id") or "")
        self._history = []
        self._render_history()
        self._save_chat_sessions()

    def _update_active_chat_title(self, user_text: str) -> None:
        chat = self._active_chat()
        if chat is None:
            return
        current = str(chat.get("title") or "").strip()
        if current and not current.startswith("Чат "):
            return
        chat["title"] = _trim_title(user_text)
        chat["updated_at"] = chat_now_iso()

    def _resolve_active_topic_title(self) -> str:
        snapshot = dict(self._last_memory_debug_snapshot or {})
        final_meta = dict(snapshot.get("final_answer_meta") or {})
        memory_context = dict(snapshot.get("memory_context") or {})
        candidates = [
            snapshot.get("active_topic_title"),
            snapshot.get("topic_thread_title"),
            snapshot.get("topic_title"),
            final_meta.get("active_topic_title"),
            final_meta.get("topic_thread_title"),
            final_meta.get("topic_title"),
            memory_context.get("active_topic_title"),
            memory_context.get("topic_thread_title"),
            memory_context.get("topic_title"),
            snapshot.get("topic_key"),
            final_meta.get("topic_key"),
            memory_context.get("topic_key"),
        ]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                return _trim_title(text, limit=40)
        
        cached = str(getattr(self, "_last_active_topic_title", "") or "").strip()
        if cached:
            return _trim_title(cached, limit=40)
            
        return SINGLE_VISIBLE_CHAT_TITLE

    @staticmethod
    def _should_follow_stream_scroll(scroll_value: int, scroll_max: int, margin_px: int = 24) -> bool:
        try:
            value = int(scroll_value)
            maximum = int(scroll_max)
            margin = max(0, int(margin_px))
        except Exception:
            return False
        if maximum <= 0:
            return False
        return (maximum - value) <= margin

    def _capture_stream_scroll_mode(self) -> None:
        bar = self.scroll.verticalScrollBar()
        self._stream_follow_scroll = self._should_follow_stream_scroll(bar.value(), bar.maximum())

    def _remove_message_widgets_after(self, bubble: QWidget) -> None:
        layout = getattr(self, "messages_layout", None)
        if layout is None:
            return
        start = self._message_layout_index(bubble)
        if start < 0:
            return
        for index in range(layout.count() - 1, start, -1):
            item = layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            taken = layout.takeAt(index)
            if taken is not None:
                widget.deleteLater()
        sync = getattr(self, "_schedule_messages_view_height_sync", None)
        if callable(sync):
            sync()

    def _assistant_bubble_after_user(self, bubble: QWidget) -> proto.MessageBubble | None:
        layout = getattr(self, "messages_layout", None)
        if layout is None:
            return None
        start = self._message_layout_index(bubble)
        if start < 0:
            return None
        for index in range(start + 1, layout.count()):
            item = layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if not isinstance(widget, proto.MessageBubble):
                continue
            role = str(getattr(widget, "role", "") or "")
            if role == "user":
                return None
            if role == "assistant":
                return widget
        return None

    def _history_index_for_user_bubble(self, bubble: QWidget) -> int:
        raw_index = bubble.property("history_index")
        try:
            index = int(raw_index)
        except Exception:
            index = -1
        if 0 <= index < len(self._history) and self._history[index][0] == "user":
            return index

        text = ""
        if isinstance(bubble, proto.MessageBubble):
            text = bubble.text_label.text().strip()
        for candidate in range(len(self._history) - 1, -1, -1):
            role, row_text, _stat_line, _feedback, _thinking = self._history[candidate]
            if role == "user" and str(row_text or "").strip() == text:
                return candidate
        return -1

    def _start_reply_worker_for_text(self, text: str, *, store_turn: bool, attachments: list[dict] | None = None) -> None:
        if not get_auth_token():
            self._show_login_dialog()
            if not get_auth_token():
                return
        if self.api is None:
            self._finalize_pending(
                text="Нативное окно поднялось, но API сейчас недоступен.",
                thinking="Нужен доступный ApiClient, чтобы окно работало как основной чат.",
                thinking_ms=None,
                perf=["offline"],
                stat_line="offline",
            )
            return

        self._set_busy_state(True)
        self._worker = ReplyWorker(
            self.api,
            user_text=text,
            store_turn=store_turn,
            think=self._thinking_enabled,
            verbose=self._verbose_enabled,
            attachments=attachments,
        )
        self._worker.chunk.connect(self._on_answer_chunk)
        self._worker.thinking_chunk.connect(self._on_thinking_chunk)
        self._worker.debug_event.connect(self._on_memory_debug_event)
        self._worker.finished.connect(self._on_reply_finished)
        self._worker.errored.connect(self._on_reply_error)
        self._worker.finished.connect(self._cleanup_request)
        self._worker.errored.connect(self._cleanup_request)
        self._worker.start()

    @Slot(object)
    def _on_regenerate_requested(self, bubble: object) -> None:
        if self._worker and self._worker.isRunning():
            MmisMessageBox.information(self, "Подожди", "Сейчас уже идёт генерация.")
            return
        if not get_auth_token():
            self._show_login_dialog()
            if not get_auth_token():
                return
        if not isinstance(bubble, proto.MessageBubble):
            return
        user_index = self._history_index_for_user_bubble(bubble)
        if user_index < 0:
            return
        role, text, _stat_line, _feedback, _thinking = self._history[user_index]
        if role != "user":
            return
        text = str(bubble.property("raw_user_text") or text or "").strip()
        if not text:
            return
        attachments = self._attachment_payloads_from_value(bubble.property("attachments_payload"))
        existing_assistant = self._assistant_bubble_after_user(bubble)

        scroll_widget = getattr(self, "scroll", None)
        scroll_bar = scroll_widget.verticalScrollBar() if scroll_widget is not None else None
        scroll_blocker = QSignalBlocker(scroll_bar) if scroll_bar is not None else None
        self._suppress_lazy_history_until = time.monotonic() + 2.0
        frozen_widgets: list[QWidget] = []
        for widget in (
            scroll_widget,
            scroll_widget.viewport() if scroll_widget is not None else None,
            getattr(self, "messages_host", None),
        ):
            if isinstance(widget, QWidget) and widget not in frozen_widgets:
                frozen_widgets.append(widget)

        for widget in frozen_widgets:
            widget.setUpdatesEnabled(False)

        try:
            self._history = self._history[: user_index + 1]
            self._pending_user_text = text
            self._pending_history_index = len(self._history)
            self._history.append(("ai", "", "…", None, None))

            if existing_assistant is not None:
                assistant_bubble = existing_assistant
                preserved_height = max(
                    int(assistant_bubble.minimumHeight() or 0),
                    int(assistant_bubble.height() or 0),
                    int(assistant_bubble.sizeHint().height() or 0),
                )
                assistant_bubble.setProperty("regen_previous_min_height", assistant_bubble.minimumHeight())
                if preserved_height > 0:
                    assistant_bubble.setMinimumHeight(preserved_height)
                self._remove_message_widgets_after(assistant_bubble)
            else:
                self._remove_message_widgets_after(bubble)
                assistant_bubble = proto.MessageBubble("assistant", "", "", "", [])
                self._insert_message_bubble_after(bubble, assistant_bubble)

            assistant_bubble.update_text("")
            assistant_bubble.update_thinking("", None)
            assistant_bubble.set_perf([])
            assistant_bubble.show()
            self._regen_scroll_anchor = assistant_bubble
            self._pending = proto.PendingAssistant(bubble=assistant_bubble, started_at=time.perf_counter())
            self._start_pending_elapsed_timer()
            if hasattr(self, "_lazy_history_start_index"):
                self._lazy_history_start_index = min(self._lazy_history_start_index, len(self._history))
            sync_lazy = getattr(self, "_sync_lazy_history_button", None)
            if callable(sync_lazy):
                sync_lazy()
            self._save_chat_sessions()
            self._force_stream_follow_scroll = True
            self._stream_follow_scroll = True
            sync_height = getattr(self, "_sync_messages_view_height", None)
            if callable(sync_height):
                sync_height()
            if scroll_widget and scroll_widget.widget() and scroll_widget.widget().layout():
                scroll_widget.widget().layout().activate()
            self._scroll_to_regenerate_anchor()
        finally:
            del scroll_blocker
            for widget in reversed(frozen_widgets):
                widget.setUpdatesEnabled(True)

        self._schedule_scroll_to_regenerate_anchor()
        self._start_reply_worker_for_text(text, store_turn=False, attachments=attachments)

    def _insert_message_bubble_after(self, previous: QWidget, bubble: QWidget) -> None:
        wire = getattr(self, "_wire_regenerate_bubble", None)
        if callable(wire):
            wire(bubble)
        layout = getattr(self, "messages_layout", None)
        if layout is None:
            return
        previous_index = self._message_layout_index(previous)
        insert_index = layout.count() if previous_index < 0 else previous_index + 1
        layout.insertWidget(insert_index, bubble)
        sync = getattr(self, "_schedule_messages_view_height_sync", None)
        if callable(sync):
            sync()

    def _schedule_scroll_bottom(self, *, follow_stream_only: bool = False) -> None:
        requested_follow_only = bool(follow_stream_only)
        if self._scroll_bottom_queued:
            if not requested_follow_only:
                self._queued_scroll_follow_only = False
            return
        self._scroll_bottom_queued = True
        self._queued_scroll_follow_only = requested_follow_only

        def _flush() -> None:
            follow_only = bool(self._queued_scroll_follow_only)
            self._scroll_bottom_queued = False
            self._queued_scroll_follow_only = False
            if follow_only and not self._stream_follow_scroll:
                return
            if self._regen_scroll_anchor is not None:
                self._scroll_to_regenerate_anchor()
                return
            scroll_widget = getattr(self, "scroll", None)
            if scroll_widget and scroll_widget.widget() and scroll_widget.widget().layout():
                scroll_widget.widget().layout().activate()
            self._scroll_bottom()

        QTimer.singleShot(0, _flush)

    def _scroll_to_regenerate_anchor(self) -> None:
        anchor = self._regen_scroll_anchor
        scroll_widget = getattr(self, "scroll", None)
        if anchor is None or scroll_widget is None:
            return
        try:
            if scroll_widget.widget() is not None and scroll_widget.widget().layout() is not None:
                scroll_widget.widget().layout().activate()
            scroll_widget.ensureWidgetVisible(anchor, 0, 32)
            bar = scroll_widget.verticalScrollBar()
            if bar is not None:
                bar.setValue(bar.maximum())
        except Exception:
            pass

    def _schedule_scroll_to_regenerate_anchor(self) -> None:
        if self._regen_scroll_anchor is None:
            return
        QTimer.singleShot(0, self._scroll_to_regenerate_anchor)
        QTimer.singleShot(80, self._scroll_to_regenerate_anchor)
        QTimer.singleShot(250, self._scroll_to_regenerate_anchor)

    def _set_busy_state(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self.search_btn.setEnabled(not busy)
        self.plus_btn.setEnabled(not busy)
        self.mic_btn.setEnabled(not busy)
        self.functions_btn.setEnabled(not busy)
        self.models_button.setEnabled(not busy)
        self.input.setReadOnly(busy)
        if hasattr(self, "think_toggle"):
            self.think_toggle.setEnabled(not busy)
        if hasattr(self, "verbose_toggle"):
            self.verbose_toggle.setEnabled(not busy)
        if hasattr(self, "json_toggle"):
            self.json_toggle.setEnabled(not busy)
        if hasattr(self, "web_mode_selector"):
            self.web_mode_selector.setEnabled(not busy)

    def _send_message(self) -> None:
        typed_text = self.input.toPlainText().strip()
        attachments = self._pending_attachment_payloads()
        if not typed_text and not attachments:
            return
        text = typed_text or ATTACHMENT_EMPTY_PROMPT
        display_text = self._display_text_for_message(text, attachments)
        if self._worker and self._worker.isRunning():
            MmisMessageBox.information(self, "Подожди", "Сейчас уже идёт генерация.")
            return

        if not get_auth_token():
            self._show_login_dialog()
            if not get_auth_token():
                return

        self._capture_stream_scroll_mode()
        chat = self._ensure_active_chat()
        user_history_index = len(self._history)
        self._history.append(("user", display_text, None, None, None))
        self._pending_user_text = text
        self._update_active_chat_title(display_text)
        ui_attachments = []
        for att in (attachments or []):
            name = str(att.get("name") or "файл")
            size = self._format_attachment_size(att.get("size"))
            ui_attachments.append({"name": f"{name} ({size})"})
        user_bubble = proto.MessageBubble("user", text, "", "", [], attachments=ui_attachments)
        user_bubble.setProperty("history_index", user_history_index)
        user_bubble.setProperty("raw_user_text", text)
        user_bubble.setProperty("attachments_payload", attachments)
        self._insert_message_bubble(user_bubble)
        assistant_bubble = proto.MessageBubble("assistant", "", "", "", [])
        assistant_bubble.hide()
        self._insert_message_bubble(assistant_bubble)
        self._pending = proto.PendingAssistant(bubble=assistant_bubble, started_at=time.perf_counter())
        self._start_pending_elapsed_timer()
        self._pending_history_index = len(self._history)
        self._history.append(("ai", "", "…", None, None))
        self.input.clear()
        self._pending_attachments = []
        self._sync_attachment_row()
        self._schedule_scroll_bottom()
        self._save_chat_sessions()

        if self.api is None:
            self._finalize_pending(
                text="Нативное окно поднялось, но API сейчас недоступен.",
                thinking="Нужен доступный ApiClient, чтобы окно работало как основной чат.",
                thinking_ms=None,
                perf=["offline"],
                stat_line="offline",
            )
            return

        store_turn = not bool(chat.get("incognito", False))
        self._start_reply_worker_for_text(text, store_turn=store_turn, attachments=attachments)

    @Slot(str)
    def _on_answer_chunk(self, piece: str) -> None:
        if not self._pending:
            return
        now = time.perf_counter()
        if self._pending.first_answer_at is None:
            self._pending.first_answer_at = now
        self._pending.last_chunk_at = now
        self._pending.answer_text += piece or ""
        self._pending.bubble.update_text(self._pending.answer_text)
        if not self._pending.bubble.isVisible():
            self._pending.bubble.show()
        self._schedule_scroll_bottom(follow_stream_only=True)

    @Slot(str)
    def _on_thinking_chunk(self, piece: str) -> None:
        if not self._pending:
            return

        piece_text = str(piece or "")
        if not piece_text:
            return

        now = time.perf_counter()
        self._pending.last_chunk_at = now
        self._pending.thinking_text += piece_text

        if self._pending.first_thinking_at is None and piece_text.strip():
            self._pending.first_thinking_at = now

        started = self._pending.first_thinking_at or now
        elapsed_ms = max(1, int((now - started) * 1000))

        self._pending.bubble.update_thinking(
            self._pending.thinking_text,
            self._format_duration_label(elapsed_ms),
        )
        if not self._pending.bubble.isVisible():
            self._pending.bubble.show()
        self._schedule_scroll_bottom(follow_stream_only=True)

    @Slot(object)
    def _on_memory_debug_event(self, snapshot: object) -> None:
        payload = dict(snapshot or {}) if isinstance(snapshot, dict) else {}
        self._last_memory_debug_snapshot = payload
        if self._inspector_panel is not None:
            self._inspector_panel.set_snapshot(payload)

    @staticmethod
    def _merge_streamed_and_final_text(streamed: str, final: str) -> str:
        streamed_text = str(streamed or "")
        final_text = str(final or "")
        if not final_text.strip():
            return streamed_text.strip()
        if not streamed_text.strip():
            return final_text.strip()
        if final_text.startswith(streamed_text):
            return final_text.strip()
        if final_text in streamed_text:
            return streamed_text.strip()
        return streamed_text.strip()

    @classmethod
    def _resolve_thinking_text(cls, result_thinking: str, pending_thinking: str, thinking_generated: bool) -> str:
        if not thinking_generated:
            return str(pending_thinking or "").strip()
        return cls._merge_streamed_and_final_text(pending_thinking, result_thinking)

    @Slot(object)
    def _on_reply_finished(self, result: ReplyResult) -> None:
        self._mark_api_online()
        stats = dict(result.stats or {})
        text = str(result.text or (self._pending.answer_text if self._pending else "")).strip()
        pending_thinking = self._pending.thinking_text if self._pending else ""
        thinking_generated = bool(getattr(result, "thinking_generated", False) or str(pending_thinking or "").strip())
        thinking = self._resolve_thinking_text(str(result.thinking or ""), pending_thinking, thinking_generated)
        if not text:
            text = "Модель не вернула видимый ответ. Повтори запрос, я перегенерирую его без пустого вывода."
            stats["error"] = True
            stats["error_type"] = "empty_visible_answer"
        debug_trace = dict(result.debug_trace or {})
        finished_at = time.perf_counter()
        fallback_elapsed_ms = 0
        local_thinking_ms = 0
        local_answer_ms = 0
        if self._pending is not None:
            fallback_elapsed_ms = max(1, int((finished_at - self._pending.started_at) * 1000))
            stats["client_wall_elapsed_ms"] = fallback_elapsed_ms
            stats["display_elapsed_ms"] = fallback_elapsed_ms
            if (
                self._pending.first_answer_at is not None
                and self._pending.first_thinking_at is not None
                and thinking_generated
            ):
                local_thinking_ms = max(
                    1,
                    int((self._pending.first_answer_at - self._pending.first_thinking_at) * 1000),
                )
                local_answer_ms = max(
                    1,
                    int((finished_at - self._pending.first_answer_at) * 1000),
                )
            elif self._pending.first_answer_at is not None:
                local_answer_ms = max(1, int((finished_at - self._pending.first_answer_at) * 1000))
            elif text:
                local_answer_ms = fallback_elapsed_ms
        stats = self._complete_verbose_stats(
            stats,
            user_text=self._pending_user_text,
            answer_text=text,
            fallback_elapsed_ms=fallback_elapsed_ms,
            local_answer_ms=local_answer_ms,
            local_thinking_ms=local_thinking_ms,
            verbose_enabled=bool(self._verbose_enabled or stats.get("verbose_enabled")),
            debug_trace=debug_trace,
        )
        thinking_ms = self._thinking_ms_label(stats, fallback_ms=local_thinking_ms) if thinking_generated and thinking else None
        perf, _stat_line = self._perf_from_stats(stats, fallback_elapsed_ms=fallback_elapsed_ms)
        stat_line = self._compose_stat_line(perf, thinking_ms)
        served_model = str(result.model or stats.get("served_model") or self.api.get_runtime_model() if self.api else "").strip()
        if served_model:
            self._active_model = served_model
        snapshot = dict(result.memory_debug_snapshot or {})
        if not snapshot:
            snapshot = debug_trace
        if snapshot:
            self._last_memory_debug_snapshot = snapshot
            if self._inspector_panel is not None:
                self._inspector_panel.set_snapshot(snapshot)
        self._finalize_pending(text=text, thinking=thinking, thinking_ms=thinking_ms, perf=perf, stat_line=stat_line)
        models = list(self._available_models)
        if self._active_model and self._active_model not in models:
            models.append(self._active_model)
            self._available_models = models
        self._populate_models(self._active_model, models)
        self._apply_context_chips()

    @Slot(str)
    def _on_reply_error(self, error_text: str) -> None:
        thinking = self._pending.thinking_text if self._pending else ""
        self._finalize_pending(
            text=f"Ошибка: {error_text}",
            thinking=thinking,
            thinking_ms=None,
            perf=[],
            stat_line=None,
        )
        MmisMessageBox.warning(self, "Ошибка", str(error_text or "Не удалось получить ответ"))

    def _finalize_pending(self, *, text: str, thinking: str, thinking_ms: str | None, perf: list[str], stat_line: str | None) -> None:
        self._stop_pending_elapsed_timer()
        pending_bubble = self._pending.bubble if self._pending else None
        if self._pending:
            self._pending.bubble.update_text(text)
            self._pending.bubble.update_thinking(thinking, thinking_ms)
            self._pending.bubble.set_perf(perf)
        if pending_bubble is not None:
            previous_min_height = pending_bubble.property("regen_previous_min_height")
            if previous_min_height is not None:
                try:
                    pending_bubble.setMinimumHeight(max(0, int(previous_min_height)))
                except Exception:
                    pending_bubble.setMinimumHeight(0)
                pending_bubble.setProperty("regen_previous_min_height", None)
        idx = self._pending_history_index
        if idx is not None and 0 <= idx < len(self._history):
            self._history[idx] = ("ai", text, stat_line, None, thinking or None)
        elif text:
            self._history.append(("ai", text, stat_line, None, thinking or None))
        self._pending = None
        self._pending_history_index = None
        self._pending_user_text = ""
        self._save_chat_sessions()
        if self._voice_page is not None and text:
            self._voice_page.set_assistant_text(text)
        if MMIS_VOICE_AUTO_SPEAK and self._is_voice_mode_open() and text:
            self._set_voice_state_ui(VoiceState.SPEAKING)
            self._voice_manager.enqueue_speak(
                text,
                lang="",
                config=build_tts_config(tts_voice=self._voice_tts_voice, tts_rate=self._int_to_percent(self._voice_rate_percent)),
            )
        self._set_busy_state(False)
        self._schedule_scroll_bottom(follow_stream_only=True)

    def _mark_api_online(self) -> None:
        self._backend_status_seen_ok = True
        self._backend_status_failures = 0
        model = self._active_model or (self.api.get_runtime_model() if self.api else "")
        self._apply_backend_status(
            api_ok=True,
            model_ok=bool(model),
            memory_ok=False,
            model=model,
            model_state="active" if model else "",
            memory_state="unknown",
        )

    @Slot()
    def _cleanup_request(self) -> None:
        self._stop_pending_elapsed_timer()
        self._set_busy_state(False)
        worker = self._worker
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.wait(1000)
            except Exception:
                pass
            try:
                if not worker.isRunning():
                    worker.deleteLater()
            except Exception:
                pass
        self._worker = None
        self._force_stream_follow_scroll = False
        self._stream_follow_scroll = False
        self._regen_scroll_anchor = None
        self._apply_context_chips()

    def _start_pending_elapsed_timer(self) -> None:
        if not self._verbose_enabled:
            return
        if self._pending_elapsed_timer is None:
            return
        self._refresh_pending_elapsed_perf()
        self._pending_elapsed_timer.start()

    def _stop_pending_elapsed_timer(self) -> None:
        timer = getattr(self, "_pending_elapsed_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    @Slot()
    def _refresh_pending_elapsed_perf(self) -> None:
        pending = self._pending
        if pending is None:
            self._stop_pending_elapsed_timer()
            return
        if not self._verbose_enabled:
            pending.bubble.set_perf([])
            self._stop_pending_elapsed_timer()
            return
        elapsed_ms = max(1, int((time.perf_counter() - pending.started_at) * 1000))
        pending.bubble.set_perf([self._format_duration_label(elapsed_ms)])
        if not pending.bubble.isVisible():
            pending.bubble.show()
        self._schedule_scroll_bottom(follow_stream_only=True)

    @staticmethod
    def _complete_verbose_stats(
        stats: dict,
        *,
        user_text: str = "",
        answer_text: str = "",
        fallback_elapsed_ms: int = 0,
        local_answer_ms: int = 0,
        local_thinking_ms: int = 0,
        verbose_enabled: bool = False,
        debug_trace: dict | None = None,
    ) -> dict:
        out = dict(stats or {})
        trace = dict(debug_trace or {})
        if verbose_enabled:
            out["verbose_enabled"] = True

        def _int_value(*keys: str) -> int:
            for key in keys:
                try:
                    value = out.get(key)
                    if value not in (None, ""):
                        return int(float(value))
                except Exception:
                    pass
            return 0

        def _float_value(*keys: str) -> float:
            for key in keys:
                try:
                    value = out.get(key)
                    if value not in (None, ""):
                        return float(value)
                except Exception:
                    pass
            return 0.0

        elapsed = _int_value("elapsed_ms", "total_ms", "total_duration_ms", "answer_ms")
        if elapsed <= 0 and fallback_elapsed_ms > 0:
            out["elapsed_ms"] = int(fallback_elapsed_ms)
            elapsed = int(fallback_elapsed_ms)
        if elapsed > 0:
            out.setdefault("elapsed_ms", int(elapsed))
            out.setdefault("answer_ms", int(elapsed))
            out.setdefault("total_duration_ms", int(elapsed))

        backend_prompt_tokens = _int_value("prompt_eval_count", "prompt_tokens")
        backend_gen_tokens = _int_value("eval_count", "gen_tokens")
        backend_total_tokens = _int_value("total_tokens")
        backend_prompt_eval_duration_ms = _float_value("prompt_eval_duration_ms")
        backend_eval_duration_ms = _float_value("eval_duration_ms", "eval_ms", "decode_ms")
        backend_total_duration_ms = _float_value("total_duration_ms", "total_ms", "answer_ms", "elapsed_ms")
        backend_tok_s = _float_value("eval_tokens_per_sec", "tokens_per_second", "tok_s", "tps")

        prompt_eval_duration_ms = backend_prompt_eval_duration_ms or _float_value("thinking_ms")
        if prompt_eval_duration_ms <= 0.0 and local_thinking_ms > 0:
            out["prompt_eval_duration_ms"] = int(local_thinking_ms)
            prompt_eval_duration_ms = float(local_thinking_ms)
        explicit_thinking_ms = _int_value("thinking_ms", "prompt_ms")
        display_thinking_ms = int(local_thinking_ms or explicit_thinking_ms or 0)
        if display_thinking_ms > 0:
            out["display_thinking_ms"] = display_thinking_ms

        if not verbose_enabled:
            return out

        prompt_tokens = backend_prompt_tokens
        if prompt_tokens <= 0:
            trace_prompt_pack = dict(trace.get("prompt_pack") or {})
            try:
                prompt_tokens = int(float(trace_prompt_pack.get("token_estimate") or 0) or 0)
            except Exception:
                prompt_tokens = 0
            if prompt_tokens > 0:
                out["prompt_eval_count"] = prompt_tokens
        if prompt_tokens <= 0 and user_text.strip():
            prompt_tokens = max(1, int(estimate_tokens(user_text)))
            out["prompt_eval_count"] = prompt_tokens
        if prompt_tokens > 0:
            out.setdefault("prompt_tokens", int(prompt_tokens))
            out["display_prompt_tokens"] = int(prompt_tokens)

        gen_tokens = backend_gen_tokens
        visible_gen_tokens = 0
        if answer_text.strip():
            visible_gen_tokens = max(1, int(estimate_tokens(answer_text)))
            if gen_tokens <= 0:
                gen_tokens = visible_gen_tokens
                out["eval_count"] = gen_tokens
        if gen_tokens > 0:
            out.setdefault("gen_tokens", int(gen_tokens))
            out["display_gen_tokens"] = int(gen_tokens)

        total_tokens = backend_total_tokens
        if total_tokens <= 0 and (prompt_tokens > 0 or gen_tokens > 0):
            out["total_tokens"] = int(prompt_tokens + gen_tokens)

        decode_ms = backend_eval_duration_ms
        if decode_ms <= 0.0 and local_answer_ms > 0:
            out["eval_duration_ms"] = int(local_answer_ms)
            decode_ms = float(local_answer_ms)
        if decode_ms <= 0.0:
            total_duration_ms = _float_value("total_duration_ms", "elapsed_ms", "total_ms", "answer_ms")
            load_duration_ms = _float_value("load_duration_ms")
            derived_decode_ms = 0.0
            if total_duration_ms > 0.0 and prompt_eval_duration_ms > 0.0:
                derived_decode_ms = total_duration_ms - prompt_eval_duration_ms - max(0.0, load_duration_ms)
            elif total_duration_ms > 0.0 and gen_tokens > 0:
                derived_decode_ms = total_duration_ms
            if derived_decode_ms > 0.0:
                out["eval_duration_ms"] = round(float(derived_decode_ms), 2)
                decode_ms = float(derived_decode_ms)
        if decode_ms > 0.0:
            out.setdefault("decode_ms", round(float(decode_ms), 2))
        display_write_ms = int(round(float(decode_ms or 0.0)) or 0)
        if display_write_ms > 0:
            out["display_write_ms"] = display_write_ms

        tok_s = backend_tok_s
        if tok_s > 0.0:
            out.setdefault("eval_tokens_per_sec", round(float(tok_s), 2))
        display_tok_s = round(float(tok_s), 2) if tok_s > 0.0 else 0.0
        if display_tok_s > 0.0:
            out["display_tok_s"] = display_tok_s

        # First verbose chip is user-visible wait time: send click -> final reply.
        # Backend prompt/write timings can omit queueing, model loading, and setup.
        display_elapsed_ms = _int_value("client_wall_elapsed_ms")
        if display_elapsed_ms <= 0:
            display_elapsed_ms = int(fallback_elapsed_ms or 0)
        if display_elapsed_ms <= 0:
            display_elapsed_ms = int(out.get("display_thinking_ms") or 0) + int(out.get("display_write_ms") or 0)
        if display_elapsed_ms <= 0:
            display_elapsed_ms = int(round(float(backend_total_duration_ms or 0.0)) or 0)
        if display_elapsed_ms <= 0:
            display_elapsed_ms = int(elapsed or 0)
        if display_elapsed_ms > 0:
            out["display_elapsed_ms"] = display_elapsed_ms

        return out

    @staticmethod
    def _thinking_ms_label(stats: dict, fallback_ms: int = 0) -> str | None:
        for key in ("display_thinking_ms", "thinking_ms", "prompt_ms"):
            try:
                value = int(float(stats.get(key) or 0) or 0)
            except Exception:
                value = 0
            if value > 0:
                return ChatWindow._format_duration_label(value)
        if fallback_ms > 0:
            return ChatWindow._format_duration_label(fallback_ms)
        return None

    @staticmethod
    def _perf_from_stats(stats: dict, fallback_elapsed_ms: int = 0) -> tuple[list[str], str | None]:
        verbose_enabled = True
        elapsed = int(float(stats.get("client_wall_elapsed_ms") or 0) or 0)
        if elapsed <= 0:
            elapsed = int(float(stats.get("display_elapsed_ms") or 0) or 0)
        if elapsed <= 0:
            elapsed = int(float(
                stats.get("elapsed_ms")
                or stats.get("total_ms")
                or stats.get("total_duration_ms")
                or stats.get("answer_ms")
                or fallback_elapsed_ms
                or 0
            ) or 0)
        if elapsed <= 0 and verbose_enabled:
            elapsed = max(1, int(fallback_elapsed_ms or 0))
        decode_ms = int(float(stats.get("display_write_ms") or 0) or 0)
        if decode_ms <= 0:
            decode_ms = int(float(
                stats.get("decode_ms")
                or stats.get("eval_ms")
                or stats.get("eval_duration_ms")
                or 0
            ) or 0)
        if decode_ms <= 0 and verbose_enabled and elapsed > 0:
            decode_ms = elapsed
        tok_s = (
            stats.get("display_tok_s")
            or 0.0
        )
        if tok_s in (None, "", 0, 0.0):
            tok_s = (
            stats.get("eval_tokens_per_sec")
            or stats.get("tokens_per_second")
            or stats.get("tok_s")
            or stats.get("tps")
            )
        prompt_t = stats.get("display_prompt_tokens") or stats.get("prompt_tokens") or stats.get("prompt_eval_count")
        gen_t = stats.get("display_gen_tokens") or stats.get("gen_tokens") or stats.get("eval_count")
        try:
            prompt_t_int = int(float(prompt_t or 0) or 0)
        except Exception:
            prompt_t_int = 0
        try:
            gen_t_int = int(float(gen_t or 0) or 0)
        except Exception:
            gen_t_int = 0
        try:
            tok_s_float = float(tok_s) if tok_s not in (None, "") else 0.0
        except Exception:
            tok_s_float = 0.0
        perf: list[str] = []
        if elapsed:
            perf.append(ChatWindow._format_duration_label(elapsed))
        if decode_ms:
            perf.append(f"write {ChatWindow._format_duration_label(decode_ms)}")
        if verbose_enabled:
            if tok_s_float > 0.0:
                perf.append(f"{tok_s_float:.1f} tok/s")
            perf.append(f"prompt {prompt_t_int}")
            perf.append(f"gen {gen_t_int}")
        else:
            if tok_s not in (None, ""):
                try:
                    perf.append(f"{float(tok_s):.1f} tok/s")
                except Exception:
                    perf.append(f"{tok_s} tok/s")
            if prompt_t not in (None, ""):
                perf.append(f"prompt {prompt_t}")
            if gen_t not in (None, ""):
                perf.append(f"gen {gen_t}")
        return perf, (" | ".join(perf) if perf else None)

    def _toggle_inspector(self) -> None:
        if self._inspector_window is None:
            self._inspector_window = QMainWindow(self, Qt.Window)
            self._inspector_window.setWindowTitle("MMis - Memory Inspector")
            self._inspector_window.resize(720, 760)
            self._inspector_panel = MemoryInspectorPanel(self._inspector_window)
            self._inspector_window.setCentralWidget(self._inspector_panel)
        if self._inspector_panel is not None:
            self._inspector_panel.set_snapshot(self._last_memory_debug_snapshot)
        if self._inspector_window.isVisible():
            self._inspector_window.close()
            return
        self._inspector_window.show()
        self._inspector_window.raise_()
        self._inspector_window.activateWindow()

    def _attach_file(self) -> None:
        if self._worker and self._worker.isRunning():
            MmisMessageBox.information(self, "Подожди", "Сначала дождись завершения генерации.")
            return
        selected, _flt = QFileDialog.getOpenFileNames(
            self,
            "Выбери файл",
            str(Path.cwd()),
            "Files (*.txt *.md *.py *.json *.yaml *.yml *.toml *.log *.csv *.png *.jpg *.jpeg *.webp *.bmp *.gif);;All files (*.*)",
        )
        if not selected:
            return
        errors: list[str] = []
        existing = {str(item.get("path") or "") for item in self._pending_attachments}
        for raw_path in list(selected):
            if len(self._pending_attachments) >= MAX_PENDING_ATTACHMENTS:
                errors.append(f"Максимум вложений: {MAX_PENDING_ATTACHMENTS}")
                break
            try:
                attachment = self._attachment_from_path(Path(raw_path).expanduser().resolve())
            except Exception as exc:
                errors.append(str(exc))
                continue
            path_key = str(attachment.get("path") or "")
            if path_key and path_key in existing:
                continue
            existing.add(path_key)
            self._pending_attachments.append(attachment)
        self._sync_attachment_row()
        self.input.setFocus()
        if errors:
            MmisMessageBox.warning(self, "Вложения", "\n".join(errors[:4]))

    def _attachment_from_path(self, path: Path) -> dict:
        if not path.exists() or not path.is_file():
            raise ValueError(f"Не найден файл: {path}")
        stored_path = self._store_attachment_file(path)
        size = int(path.stat().st_size)
        suffix = path.suffix.lower()
        mime_type = str(mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        attachment = {
            "kind": "file",
            "name": path.name,
            "mime_type": mime_type,
            "size": size,
            "path": str(stored_path or path),
            "source_path": str(path),
        }
        is_image = suffix in IMAGE_FILE_SUFFIXES or mime_type.lower().startswith("image/")
        if is_image:
            attachment["kind"] = "image"
            attachment["data_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
            return attachment

        is_text = suffix in TEXT_FILE_SUFFIXES or mime_type.lower().startswith("text/")
        if is_text and 0 <= size <= MAX_INLINE_FILE_BYTES:
            text = self._read_attachment_text(path)
            if text:
                attachment["kind"] = "text"
                attachment["text"] = text
        return attachment

    def _store_attachment_file(self, path: Path) -> Path | None:
        root = getattr(self, "_attachments_dir", None)
        if root is None:
            return None
        try:
            root.mkdir(parents=True, exist_ok=True)
            target = root / path.name
            if target.exists():
                target = root / f"{path.stem}-{int(time.time() * 1000)}{path.suffix}"
            shutil.copy2(path, target)
            return target
        except Exception:
            return None

    @staticmethod
    def _read_attachment_text(path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "cp1251"):
            try:
                return path.read_text(encoding=encoding).strip()
            except UnicodeDecodeError:
                continue
            except Exception:
                return ""
        return ""

    @classmethod
    def _attachment_payloads_from_value(cls, value) -> list[dict]:
        rows = value if isinstance(value, (list, tuple)) else []
        out: list[dict] = []
        allowed = {"kind", "name", "mime_type", "size", "path", "text", "data_base64"}
        for item in list(rows):
            if not isinstance(item, dict):
                continue
            row = {str(key): item.get(key) for key in allowed if item.get(key) is not None}
            if not str(row.get("name") or "").strip() and not str(row.get("path") or "").strip():
                continue
            row["kind"] = str(row.get("kind") or "file").strip().lower() or "file"
            row["name"] = str(row.get("name") or Path(str(row.get("path") or "")).name or "attachment")
            row["mime_type"] = str(row.get("mime_type") or "application/octet-stream")
            try:
                row["size"] = max(0, int(row.get("size") or 0))
            except Exception:
                row["size"] = 0
            out.append(row)
        return out

    def _pending_attachment_payloads(self) -> list[dict]:
        return self._attachment_payloads_from_value(self._pending_attachments)

    @staticmethod
    def _format_attachment_size(size: int | float | str | None) -> str:
        try:
            value = max(0, int(float(size or 0)))
        except Exception:
            value = 0
        if value >= 1024 * 1024:
            return f"{value / (1024 * 1024):.1f} MB"
        if value >= 1024:
            return f"{value / 1024:.1f} KB"
        return f"{value} B"

    @classmethod
    def _attachment_chip_text(cls, attachment: dict) -> str:
        name = str(attachment.get("name") or "attachment")
        if len(name) > 34:
            name = name[:16].rstrip() + "..." + name[-13:].lstrip()
        return f"{name} ({cls._format_attachment_size(attachment.get('size'))})"

    @classmethod
    def _display_text_for_message(cls, text: str, attachments: list[dict] | None) -> str:
        rows = cls._attachment_payloads_from_value(attachments or [])
        if not rows:
            return str(text or "").strip()
        lines = [str(text or "").strip(), "", "Вложения:"]
        for attachment in rows:
            name = str(attachment.get("name") or "attachment")
            size = cls._format_attachment_size(attachment.get("size"))
            lines.append(f"- {name} ({size})")
        return "\n".join(lines).strip()

    def _latest_ai_text(self) -> str:
        for role, text, _stat_line, _feedback, _thinking in reversed(self._history):
            if role == "ai" and str(text or "").strip():
                return str(text).strip()
        return ""

    @staticmethod
    def _percent_to_int(value: str, default: int = 0) -> int:
        try:
            text = str(value or "").strip().replace("%", "")
            return int(float(text))
        except Exception:
            return int(default)

    @staticmethod
    def _int_to_percent(value: int) -> str:
        return f"{int(value):+d}%"

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
        dst = input_dir / f"{stamp}_{src.name}"
        shutil.copy2(src, dst)
        return dst

    def _is_voice_mode_open(self) -> bool:
        return bool(self._voice_stack is not None and self._voice_page is not None and self._voice_stack.currentWidget() is self._voice_page)

    def _sync_rail_mode_buttons(self) -> None:
        voice_open = self._is_voice_mode_open()
        for button, active in (
            (getattr(self, "_chat_rail_button", None), not voice_open),
            (getattr(self, "_voice_rail_button", None), voice_open),
        ):
            setter = getattr(button, "set_active", None)
            if callable(setter):
                setter(bool(active))

    @Slot()
    def _toggle_voice_mode(self) -> None:
        if self._is_voice_mode_open():
            self._close_voice_mode()
        else:
            self._open_voice_mode()

    @Slot()
    def _open_voice_mode(self) -> None:
        if self._voice_stack is None or self._voice_page is None:
            return
        self._voice_page.set_user_text(self._last_user_text())
        self._voice_page.set_assistant_text(self._latest_ai_text())
        self._voice_stack.setCurrentWidget(self._voice_page)
        self._sync_rail_mode_buttons()
        self._set_voice_state_ui(self._voice_manager.get_state())

    @Slot()
    def _close_voice_mode(self) -> None:
        if self._voice_stack is not None and self._chat_page is not None:
            self._voice_stack.setCurrentWidget(self._chat_page)
        self._sync_rail_mode_buttons()
        self._stop_voice_mode_audio()

    def _set_voice_state_ui(self, state) -> None:
        if self._voice_page is None:
            return
        value = state.value if hasattr(state, "value") else str(state or "idle")
        self._voice_page.set_state(value)

    def _last_user_text(self) -> str:
        for role, text, _stat_line, _feedback, _thinking in reversed(self._history):
            if role == "user" and str(text or "").strip():
                return str(text).strip()
        return ""

    @Slot()
    def _start_voice_recording(self) -> None:
        if self._worker and self._worker.isRunning():
            MmisMessageBox.information(self, "Voice", "Wait until the current reply is finished.")
            return
        if self._voice_manager.get_state() == VoiceState.SPEAKING:
            self._voice_manager.barge_in()
        try:
            MMIS_VOICE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._voice_recording_path = MMIS_VOICE_INPUT_DIR / f"ptt_{stamp}.wav"
            self._voice_capture_session = QMediaCaptureSession(self)
            self._voice_audio_input = QAudioInput(self)
            self._voice_recorder = QMediaRecorder(self)
            fmt = QMediaFormat()
            fmt.setFileFormat(QMediaFormat.FileFormat.Wave)
            self._voice_recorder.setMediaFormat(fmt)
            self._voice_recorder.setOutputLocation(QUrl.fromLocalFile(str(self._voice_recording_path)))
            self._voice_capture_session.setAudioInput(self._voice_audio_input)
            self._voice_capture_session.setRecorder(self._voice_recorder)
            self._voice_manager.start_listening()
            self._voice_recorder.record()
        except Exception as exc:
            self._voice_manager.stop_listening()
            MmisMessageBox.critical(self, "Voice", f"Could not start recording:\n{exc}")

    @Slot()
    def _stop_voice_recording(self) -> None:
        recorder = self._voice_recorder
        if recorder is None:
            self._voice_manager.stop_listening()
            return
        try:
            recorder.stop()
        finally:
            self._voice_manager.stop_listening()
        path = self._voice_recording_path
        if path is not None:
            QTimer.singleShot(450, lambda p=path: self._process_voice_audio_file(p))

    def _process_voice_audio_file(self, path: Path) -> None:
        try:
            if not path.exists() or path.stat().st_size <= 0:
                MmisMessageBox.warning(self, "Voice", "Recording is empty.")
                return
            result = self._voice_manager.transcribe(str(path), config=build_stt_config())
            text = str(result.text or "").strip()
        except Exception as exc:
            MmisMessageBox.critical(self, "Voice", f"Could not recognize audio:\n{exc}")
            return
        if not text:
            MmisMessageBox.warning(self, "Voice", "Recognition returned empty text.")
            return

    def _on_voice_final_text(self, text: str) -> None:
        payload = str(text or "").strip()
        if not payload:
            return
        if self._voice_page is not None:
            self._voice_page.set_user_text(payload)
            self._voice_page.set_state("thinking")
        self.input.setPlainText(payload)
        self._send_message()

    @Slot()
    def _voice_input_file_to_message(self) -> None:
        selected, _flt = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            str(MMIS_VOICE_INPUT_DIR),
            "Audio (*.wav *.mp3 *.m4a *.ogg *.flac);;All files (*.*)",
        )
        if not selected:
            return
        try:
            source_path = self._copy_audio_into_input_dir(Path(selected))
            result = self._voice_manager.transcribe(str(source_path), config=build_stt_config())
            text = str(result.text or "").strip()
        except Exception as exc:
            MmisMessageBox.critical(self, "Voice", f"Could not recognize file:\n{exc}")
            return
        if not text:
            MmisMessageBox.warning(self, "Voice", "Recognition returned empty text.")
            return

    @Slot()
    def _stop_voice_mode_audio(self) -> None:
        try:
            self._media_player.stop()
        except Exception:
            pass
        self._voice_manager.barge_in()
        self._voice_manager.tts.stop()
        self._set_voice_state_ui(VoiceState.IDLE)

    @Slot(object, str)
    def _on_media_error(self, _error, error_text: str) -> None:
        if error_text:
            MmisMessageBox.warning(self, "Плеер", f"Ошибка воспроизведения:\n{error_text}")

    def _speak_text_in_app(self, text: str) -> bool:
        if not text.strip():
            return False
        self._voice_manager.enqueue_speak(
            text,
            lang="",
            config=build_tts_config(tts_voice=self._voice_tts_voice, tts_rate=self._int_to_percent(self._voice_rate_percent)),
        )
        return True

    @Slot()
    def on_voice_input_file(self) -> None:
        if self._worker and self._worker.isRunning():
            MmisMessageBox.information(self, "Подожди", "Сначала дождись завершения генерации.")
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
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            recognized = (self._get_voice_stt().transcribe_file(source_path) or "").strip()
        except Exception as exc:
            MmisMessageBox.critical(self, "Голос", f"Не удалось распознать файл:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not recognized:
            MmisMessageBox.warning(self, "Голос", "Распознавание вернуло пустой текст.")
            return
        current = self.input.toPlainText().rstrip()
        if current:
            current += "\n\n"
        self.input.setPlainText(current + recognized)
        self.input.setFocus()

    @Slot()
    def on_voice_speak_last_ai(self) -> None:
        text = self._latest_ai_text()
        if not text:
            MmisMessageBox.information(self, "Озвучка", "Пока нет ответа AI для озвучки.")
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            self._speak_text_in_app(text)
        except Exception as exc:
            MmisMessageBox.critical(self, "Озвучка", f"Не удалось озвучить ответ:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()


    def closeEvent(self, event) -> None:
        metrics_timer = getattr(self, "_metrics_timer", None)
        if metrics_timer is not None:
            try:
                metrics_timer.stop()
            except Exception:
                pass
        self._save_chat_sessions()
        voice_state_timer = getattr(self, "_voice_state_timer", None)
        if voice_state_timer is not None:
            try:
                voice_state_timer.stop()
            except Exception:
                pass
        try:
            self._voice_manager.shutdown()
        except Exception:
            pass
        self._stop_background_threads(include_reply=True)
        if self._inspector_window is not None:
            self._inspector_window.close()
        super().closeEvent(event)


def main() -> int:
    _configure_qt_startup()
    app = QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
