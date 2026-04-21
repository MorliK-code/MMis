from __future__ import annotations

import json
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

from PySide6.QtCore import QTimer, Qt, QUrl, Slot
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from config.settings import DATA_DIR, load_config
from llm.tokenizer import estimate_tokens
import ui.chat_shell as proto
from ui.api_client import ApiClient, ApiClientError
from ui.chat_sessions import SINGLE_VISIBLE_CHAT_ID, SINGLE_VISIBLE_CHAT_TITLE
from ui.chat_sessions import collapse_to_single_visible_chat, history_to_serializable
from ui.chat_sessions import load_sessions as load_chat_sessions
from ui.chat_sessions import make_new_chat_payload, now_iso as chat_now_iso
from ui.chat_sessions import save_sessions as save_chat_sessions
from ui.voice_adapter import build_stt_engine, build_tts_engine
from ui.widgets.memory_inspector_panel import MemoryInspectorPanel
from ui.workers import ReplyResult, ReplyWorker


_cfg = load_config()
MemoryStorageDir = Path(_cfg.memory_dir).expanduser().resolve()
MMIS_VOICE_INPUT_DIR = Path(_cfg.voice_input_dir or (MemoryStorageDir / "voice" / "input")).expanduser().resolve()
MMIS_VOICE_OUTPUT_DIR = Path(_cfg.voice_output_dir or (MemoryStorageDir / "voice" / "output")).expanduser().resolve()
MMIS_VOICE_TTS_VOICE = str(_cfg.voice_tts_voice or "ru-RU-DmitryNeural")
MMIS_VOICE_TTS_RATE = str(_cfg.voice_tts_rate or "+0%")
MMIS_VOICE_TTS_VOLUME = str(_cfg.voice_tts_volume or "+0%")

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
MAX_INLINE_FILE_BYTES = 64 * 1024
HISTORY_INITIAL_RENDER_LIMIT = 12
HISTORY_LAZY_BATCH_SIZE = 12
HISTORY_SCROLL_LOAD_THRESHOLD_PX = 24


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
        self._sessions_dir = MemoryStorageDir / "ui_chats"
        self._sessions_index_path = self._sessions_dir / "index.json"
        self._legacy_sessions_path = MemoryStorageDir / "ui_chats.json"
        self._ui_state_path = MemoryStorageDir / "ui_state.json"
        self._legacy_visible_chat_backup_path = self._sessions_dir / "_legacy_multi_chat_backup.json"
        self._chat_sessions: list[dict] = []
        self._active_chat_id: str | None = None
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
        self._scroll_bottom_queued = False
        self._queued_scroll_follow_only = False
        self._last_memory_debug_snapshot: dict = {}
        self._voice_stt = None
        self._voice_tts_voice = MMIS_VOICE_TTS_VOICE
        self._voice_rate_percent = self._percent_to_int(MMIS_VOICE_TTS_RATE, default=0)
        self._voice_volume_percent = self._percent_to_int(MMIS_VOICE_TTS_VOLUME, default=0)
        self._voice_cache_dir = MMIS_VOICE_OUTPUT_DIR / ".cache"
        self._voice_cache_dir.mkdir(parents=True, exist_ok=True)
        self._voice_reply_cache_path = self._voice_cache_dir / "reply_live.wav"
        self._inspector_window: QMainWindow | None = None
        self._inspector_panel: MemoryInspectorPanel | None = None
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
        self._load_ui_state()
        super().__init__()
        self.setWindowTitle("MMis - Chat")
        if hasattr(self, "_metrics_timer"):
            self._metrics_timer.timeout.connect(self._refresh_persona_label)
        if self.api is None:
            self.api = ApiClient()
        self._audio_output = QAudioOutput(self)
        self._media_player = QMediaPlayer(self)
        self._media_player.setAudioOutput(self._audio_output)
        self._media_player.errorOccurred.connect(self._on_media_error)
        self._sync_runtime_controls()
        self._apply_context_chips()

    def _build_ui(self):
        super()._build_ui()
        if not self._lazy_history_scroll_connected:
            self.scroll.verticalScrollBar().valueChanged.connect(self._on_history_scroll_value_changed)
            self._lazy_history_scroll_connected = True
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
        self._wire_rail_buttons()

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
        cleaned = str(character_id or "").strip().replace("_", " ").replace("-", " ")
        parts = [part for part in cleaned.split(" ") if part]
        if not parts:
            return ""
        return " ".join(part[:1].upper() + part[1:] for part in parts)

    @staticmethod
    def _payload_character_name(payload: dict) -> str:
        for key in ("name", "display_name", "title"):
            name = str(payload.get(key) or "").strip()
            if name:
                return name
        return ""

    def _resolve_active_character_id(self) -> str:
        state = self._read_json_payload(MemoryStorageDir / "brain_state.json")
        cid = self._payload_character_id(state)
        if cid:
            return cid
        global_state = state.get("global")
        if isinstance(global_state, dict):
            cid = self._payload_character_id(global_state)
            if cid:
                return cid

        manifest = self._read_json_payload(MemoryStorageDir / "characters_runtime" / "manifest.json")
        cid = self._payload_character_id(manifest)
        if cid:
            return cid
        for row in list(manifest.get("characters") or []):
            if not isinstance(row, dict) or row.get("enabled") is False:
                continue
            cid = self._payload_character_id(row)
            if cid:
                return cid
        return "asya"

    def _resolve_persona_display_name(self) -> str:
        character_id = self._resolve_active_character_id()
        manifest = self._read_json_payload(MemoryStorageDir / "characters_runtime" / "manifest.json")
        character_paths = [
            MemoryStorageDir / "characters_runtime" / character_id / "character.json",
            DATA_DIR / "specs" / "characters" / character_id / "character.json",
        ]
        for path in character_paths:
            name = self._payload_character_name(self._read_json_payload(path))
            if name:
                return name
        for row in list(manifest.get("characters") or []):
            if not isinstance(row, dict):
                continue
            if self._safe_character_id(row.get("id")) != character_id:
                continue
            name = self._payload_character_name(row)
            if name:
                return name
        return self._fallback_character_name(character_id)

    def _refresh_persona_label(self) -> None:
        label = getattr(self, "persona_label", None)
        if label is None:
            return
        name = self._resolve_persona_display_name()
        if name and label.text() != name:
            label.setText(name)
            label.adjustSize()

    def _build_models_popup(self, anchor: QWidget) -> proto.PopupFrame:
        return super()._build_models_popup(anchor)

    def _build_functions_popup(self, anchor: QWidget) -> proto.PopupFrame:
        popup = proto.PopupFrame(anchor, width=230, line_orientation="vertical")
        lay = QVBoxLayout(popup)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        title = proto.CrispLabel("Управление функциями")
        title.setFont(proto._ui_font(pixel_size=11, weight=proto.QFont.Weight.Medium))
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
        self.mic_btn.clicked.connect(self.on_voice_input_file)

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
        bubble = proto.MessageBubble(
            _display_role(role),
            text or "",
            thinking or "",
            thinking_ms if str(thinking or "").strip() else "",
            perf_items,
            show_thinking_header=bool(str(thinking or "").strip()),
        )
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
        if self._lazy_history_loading or self._lazy_history_start_index <= 0:
            return
        bar = self.scroll.verticalScrollBar()
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
            QMessageBox.information(self, "Подожди", "Сначала дождись завершения генерации.")
            return
        chat = self._active_chat()
        if not chat or not self._history:
            return
        answer = QMessageBox.question(
            self,
            "Очистить чат",
            "Удалить историю текущего чата?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
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

    def _load_or_init_chat_sessions(self) -> None:
        loaded, active_id = load_chat_sessions(self._sessions_dir, self._sessions_index_path, self._legacy_sessions_path)
        self._backup_legacy_visible_chats(loaded, active_id)
        chosen = collapse_to_single_visible_chat(loaded, active_id)
        self._chat_sessions = [chosen]
        self._active_chat_id = str(chosen.get("id") or SINGLE_VISIBLE_CHAT_ID)
        self._history = list(chosen.get("history") or [])
        self._render_history()
        self._save_chat_sessions()

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
        self._sync_active_chat_from_history()
        active_chat = self._active_chat()
        active_id = None
        if active_chat and not bool(active_chat.get("incognito", False)):
            active_id = str(active_chat.get("id") or "") or None
        try:
            save_chat_sessions(self._sessions_dir, self._sessions_index_path, self._chat_sessions, active_id)
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
        self._voice_rail_button = actual_widgets[1]
        self._file_rail_button = actual_widgets[2]
        self._memory_rail_button = actual_widgets[3]
        self._settings_rail_button = actual_widgets[5]

        self._voice_rail_button.setToolTip("Озвучить последний ответ")
        self._voice_rail_button.clicked.connect(self.on_voice_speak_last_ai)
        self._file_rail_button.setToolTip("Прикрепить файл")
        self._file_rail_button.clicked.connect(self._attach_file)
        self._memory_rail_button.setToolTip("Память / Inspector")
        self._memory_rail_button.clicked.connect(self._toggle_inspector)
        self._settings_rail_button.setToolTip("Функции")
        self._settings_rail_button.clicked.connect(self._toggle_functions)

    def _sync_runtime_controls(self) -> None:
        if not self.api:
            return
        try:
            self._thinking_enabled = bool(self.api.set_thinking_enabled(self._thinking_enabled))
        except Exception:
            pass
        try:
            self._verbose_enabled = bool(self.api.set_verbose_enabled(self._verbose_enabled))
        except Exception:
            pass
        try:
            self._json_mode_enabled = bool(self.api.set_json_mode_enabled(self._json_mode_enabled))
        except Exception:
            pass
        try:
            self._web_mode = str(self.api.set_web_mode(self._web_mode) or self._web_mode)
        except Exception:
            pass
        self._sync_controls_to_state()
        self._save_ui_state()
        self._render_history()

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

    def _load_ui_state(self) -> None:
        try:
            if not self._ui_state_path.exists():
                return
            payload = json.loads(self._ui_state_path.read_text(encoding="utf-8-sig") or "{}")
        except Exception:
            return
        self._thinking_enabled = bool(payload.get("think_enabled", self._thinking_enabled))
        self._verbose_enabled = bool(payload.get("verbose_enabled", self._verbose_enabled))
        self._json_mode_enabled = bool(payload.get("json_mode_enabled", self._json_mode_enabled))
        self._screen_enabled = bool(payload.get("screen_enabled", self._screen_enabled))
        self._web_mode = str(payload.get("web_mode") or self._web_mode)

    def _save_ui_state(self) -> None:
        payload = {
            "think_enabled": bool(self._thinking_enabled),
            "verbose_enabled": bool(self._verbose_enabled),
            "json_mode_enabled": bool(self._json_mode_enabled),
            "screen_enabled": bool(self._screen_enabled),
            "web_mode": str(self._web_mode),
        }
        try:
            self._ui_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._ui_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_models(self) -> None:
        runtime = self._active_model or (self.api.get_runtime_model() if self.api else "")
        models = list(self._available_models)
        if self.api:
            try:
                payload = self.api.list_models()
                runtime = str(payload.get("runtime_model") or self.api.get_runtime_model() or runtime)
                available = payload.get("available_models") or payload.get("models") or []
                models = [str(x) for x in available if str(x).strip()]
            except Exception:
                pass
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
                QMessageBox.warning(self, "Модель", str(exc))
                return
        if actual and actual not in self._available_models:
            self._available_models.append(actual)
        self._active_model = actual
        self._populate_models(self._active_model, self._available_models)
        self.models_popup.close_popup()
        self._apply_context_chips()

    @Slot(bool)
    def _on_think_toggled(self, checked: bool) -> None:
        previous = self._thinking_enabled
        self._thinking_enabled = bool(checked)
        if self.api:
            try:
                self.api.set_thinking_enabled(self._thinking_enabled)
            except ApiClientError as exc:
                self._thinking_enabled = previous
                self._sync_controls_to_state()
                QMessageBox.warning(self, "Функции", str(exc))
                return
        self._sync_controls_to_state()
        self._save_ui_state()

    @Slot(bool)
    def _on_verbose_toggled(self, checked: bool) -> None:
        previous = self._verbose_enabled
        self._verbose_enabled = bool(checked)
        if self.api:
            try:
                self.api.set_verbose_enabled(self._verbose_enabled)
            except ApiClientError as exc:
                self._verbose_enabled = previous
                self._sync_controls_to_state()
                QMessageBox.warning(self, "Функции", str(exc))
                return
        self._sync_controls_to_state()
        self._save_ui_state()
        self._render_history()

    @Slot(bool)
    def _on_json_toggled(self, checked: bool) -> None:
        previous = self._json_mode_enabled
        self._json_mode_enabled = bool(checked)
        if self.api:
            try:
                self.api.set_json_mode_enabled(self._json_mode_enabled)
            except ApiClientError as exc:
                self._json_mode_enabled = previous
                self._sync_controls_to_state()
                QMessageBox.warning(self, "Функции", str(exc))
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
                QMessageBox.warning(self, "Функции", str(exc))
                return
        self._sync_controls_to_state()
        self._save_ui_state()

    @Slot()
    def _on_new_chat_clicked(self) -> None:
        self._clear_messages()
        return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Подожди", "Сначала дождись завершения генерации.")
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
        if chat is not None:
            chat["title"] = SINGLE_VISIBLE_CHAT_TITLE
            chat["updated_at"] = chat_now_iso()
        return
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
            snapshot.get("topic_thread_title"),
            snapshot.get("topic_title"),
            final_meta.get("topic_thread_title"),
            final_meta.get("topic_title"),
            memory_context.get("topic_thread_title"),
            memory_context.get("topic_title"),
        ]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                return _trim_title(text, limit=40)
        for role, text, _stat_line, _feedback, _thinking in reversed(self._history):
            if role == "user" and str(text or "").strip():
                return _trim_title(text, limit=40)
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
            self._scroll_bottom()

        QTimer.singleShot(0, _flush)

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
        text = self.input.toPlainText().strip()
        if not text:
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Подожди", "Сейчас уже идёт генерация.")
            return

        self._capture_stream_scroll_mode()
        chat = self._ensure_active_chat()
        self._history.append(("user", text, None, None, None))
        self._pending_user_text = text
        self._update_active_chat_title(text)
        user_bubble = proto.MessageBubble("user", text, "", "", [])
        self._insert_message_bubble(user_bubble)
        assistant_bubble = proto.MessageBubble("assistant", "", "", "", [])
        self._insert_message_bubble(assistant_bubble)
        self._pending = proto.PendingAssistant(bubble=assistant_bubble, started_at=time.perf_counter())
        self._pending_history_index = len(self._history)
        self._history.append(("ai", "", "…", None, None))
        self.input.clear()
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

        self._set_busy_state(True)
        store_turn = not bool(chat.get("incognito", False))
        self._worker = ReplyWorker(
            self.api,
            user_text=text,
            store_turn=store_turn,
            think=self._thinking_enabled,
            verbose=self._verbose_enabled,
        )
        self._worker.chunk.connect(self._on_answer_chunk)
        self._worker.thinking_chunk.connect(self._on_thinking_chunk)
        self._worker.debug_event.connect(self._on_memory_debug_event)
        self._worker.finished.connect(self._on_reply_finished)
        self._worker.errored.connect(self._on_reply_error)
        self._worker.finished.connect(self._cleanup_request)
        self._worker.errored.connect(self._cleanup_request)
        self._worker.start()

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
        stats = dict(result.stats or {})
        text = str(result.text or (self._pending.answer_text if self._pending else "")).strip()
        pending_thinking = self._pending.thinking_text if self._pending else ""
        thinking_generated = bool(getattr(result, "thinking_generated", False) or str(pending_thinking or "").strip())
        thinking = self._resolve_thinking_text(str(result.thinking or ""), pending_thinking, thinking_generated)
        debug_trace = dict(result.debug_trace or {})
        finished_at = time.perf_counter()
        fallback_elapsed_ms = 0
        local_thinking_ms = 0
        local_answer_ms = 0
        if self._pending is not None:
            fallback_elapsed_ms = max(1, int((finished_at - self._pending.started_at) * 1000))
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
        QMessageBox.warning(self, "Ошибка", str(error_text or "Не удалось получить ответ"))

    def _finalize_pending(self, *, text: str, thinking: str, thinking_ms: str | None, perf: list[str], stat_line: str | None) -> None:
        if self._pending:
            self._pending.bubble.update_text(text)
            self._pending.bubble.update_thinking(thinking, thinking_ms)
            self._pending.bubble.set_perf(perf)
        idx = self._pending_history_index
        if idx is not None and 0 <= idx < len(self._history):
            self._history[idx] = ("ai", text, stat_line, None, thinking or None)
        elif text:
            self._history.append(("ai", text, stat_line, None, thinking or None))
        self._pending = None
        self._pending_history_index = None
        self._pending_user_text = ""
        self._save_chat_sessions()
        self._set_busy_state(False)
        self._schedule_scroll_bottom(follow_stream_only=True)

    @Slot()
    def _cleanup_request(self) -> None:
        self._set_busy_state(False)
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self._stream_follow_scroll = False
        self._apply_context_chips()

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
        if tok_s <= 0.0 and gen_tokens > 0 and decode_ms > 0.0:
            out["eval_tokens_per_sec"] = round(float(gen_tokens) / (decode_ms / 1000.0), 2)
            tok_s = float(out["eval_tokens_per_sec"])
        elif tok_s > 0.0:
            out.setdefault("eval_tokens_per_sec", round(float(tok_s), 2))
        display_tok_s = round(float(tok_s), 2) if tok_s > 0.0 else 0.0
        if display_tok_s > 0.0:
            out["display_tok_s"] = display_tok_s

        display_elapsed_ms = int(round(float(backend_total_duration_ms or 0.0)) or 0)
        if display_elapsed_ms <= 0 and fallback_elapsed_ms > 0:
            display_elapsed_ms = int(fallback_elapsed_ms or 0)
        if display_elapsed_ms <= 0:
            display_elapsed_ms = int(elapsed or 0)
        if display_elapsed_ms <= 0:
            display_elapsed_ms = max(
                int(out.get("display_thinking_ms") or 0),
                int(out.get("display_write_ms") or 0),
            )
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
        verbose_enabled = bool(stats.get("verbose_enabled"))
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
        if verbose_enabled and tok_s_float <= 0.0 and gen_t_int > 0 and decode_ms > 0:
            tok_s_float = round(float(gen_t_int) / (decode_ms / 1000.0), 2)
        perf: list[str] = []
        if elapsed:
            perf.append(ChatWindow._format_duration_label(elapsed))
        if decode_ms:
            perf.append(f"write {ChatWindow._format_duration_label(decode_ms)}")
        if verbose_enabled:
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
            QMessageBox.information(self, "Подожди", "Сначала дождись завершения генерации.")
            return
        selected, _flt = QFileDialog.getOpenFileName(
            self,
            "Выбери файл",
            str(Path.cwd()),
            "All files (*.*)",
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        block = self._build_file_prompt_block(path)
        current = self.input.toPlainText().rstrip()
        if current:
            current += "\n\n"
        self.input.setPlainText(current + block)
        self.input.setFocus()

    def _build_file_prompt_block(self, path: Path) -> str:
        header = f"[Файл: {path.name}]"
        try:
            size = path.stat().st_size
        except Exception:
            size = -1
        if path.suffix.lower() not in TEXT_FILE_SUFFIXES or size < 0 or size > MAX_INLINE_FILE_BYTES:
            return f"{header}\nПуть: {path}"
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except Exception:
                return f"{header}\nПуть: {path}"
        except Exception:
            return f"{header}\nПуть: {path}"
        return f"{header}\n{text.strip()}"

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

    @Slot(object, str)
    def _on_media_error(self, _error, error_text: str) -> None:
        if error_text:
            QMessageBox.warning(self, "Плеер", f"Ошибка воспроизведения:\n{error_text}")

    def _speak_text_in_app(self, text: str) -> bool:
        if not text.strip():
            return False
        rate = self._int_to_percent(self._voice_rate_percent)
        volume = self._int_to_percent(self._voice_volume_percent)
        tts = build_tts_engine(tts_voice=self._voice_tts_voice, tts_rate=rate, tts_volume=volume)
        saved = tts.synthesize_to_file(text, self._voice_reply_cache_path)
        self._media_player.stop()
        self._media_player.setSource(QUrl.fromLocalFile(str(saved.resolve())))
        self._media_player.play()
        return True

    @Slot()
    def on_voice_input_file(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Подожди", "Сначала дождись завершения генерации.")
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
            QMessageBox.critical(self, "Голос", f"Не удалось распознать файл:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if not recognized:
            QMessageBox.warning(self, "Голос", "Распознавание вернуло пустой текст.")
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
            QMessageBox.information(self, "Озвучка", "Пока нет ответа AI для озвучки.")
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            self._speak_text_in_app(text)
        except Exception as exc:
            QMessageBox.critical(self, "Озвучка", f"Не удалось озвучить ответ:\n{exc}")
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
        if self._worker is not None:
            try:
                self._worker.request_cancel()
            except Exception:
                pass
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(1500)
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
