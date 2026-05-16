from __future__ import annotations

import copy
import hashlib
import json
import secrets
from typing import Any
from urllib.parse import urlparse

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics, QPainter, QPen, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsBlurEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.settings_sync_service import dotted_set, load_settings_payload, save_settings_updates
from ui.client_config_store import get_selected_base_url, save_account_client_config_updates
from ui.chat_shell import ChatScrollOverlay, PlainTextScrollOverlay, _to_qcolor, _ui_font
from ui.settings_schema import SETTINGS_CATEGORIES, SettingCard, SettingCategory, SettingSpec, dotted_get, get_category
from ui.settings_styles import SETTINGS_STYLE, apply_settings_tooltip_style
from ui.settings_widgets import SettingEditor, cleanup_active_model_workers
from ui.api_client import ApiClient
from ui.widgets.character_manager import CharacterManager
from ui.widgets.message_box import MmisMessageBox

CARD_TAG_HEIGHT = 18
CARD_TAG_HPAD = 9
CARD_TAG_RADIUS = 9


class CardTagBadge(QWidget):
    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._text = str(text or "")
        self.setFont(_ui_font(pixel_size=10))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedSize(self.sizeHint())

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = str(text or "")
        self.setFixedSize(self.sizeHint())
        self.update()

    def sizeHint(self) -> QSize:
        width = self.fontMetrics().horizontalAdvance(self._text) + CARD_TAG_HPAD * 2
        return QSize(max(width, CARD_TAG_HEIGHT), CARD_TAG_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setPen(QPen(_to_qcolor("rgba(139, 92, 246, 40)"), 1))
        painter.setBrush(_to_qcolor("rgba(139, 92, 246, 20)"))
        painter.drawRoundedRect(rect, CARD_TAG_RADIUS, CARD_TAG_RADIUS)

        painter.setPen(_to_qcolor("#c4b5fd"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)




class SettingsHintPopup(QFrame):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("settings_hint_popup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(286)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)
        self.title_label = QLabel("")
        self.title_label.setObjectName("hint_popup_title")
        self.body_label = QLabel("")
        self.body_label.setObjectName("hint_popup_body")
        self.body_label.setWordWrap(True)
        self.example_label = QLabel("")
        self.example_label.setObjectName("hint_popup_example")
        self.example_label.setWordWrap(True)
        self.restart_badge = QLabel("restart required")
        self.restart_badge.setObjectName("restart_badge")
        self.updated_badge = QLabel("updated")
        self.updated_badge.setObjectName("updated_badge")
        badge_layout = QHBoxLayout()
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.addWidget(self.restart_badge)
        badge_layout.addWidget(self.updated_badge)
        badge_layout.addStretch()

        layout.addWidget(self.title_label)
        layout.addWidget(self.body_label)
        layout.addLayout(badge_layout)
        layout.addWidget(self.example_label)
        self.hide()

    def set_spec(self, spec: SettingSpec, updated: bool = False) -> None:
        for label in (self.title_label, self.body_label, self.example_label):
            label.setMinimumHeight(0)
            label.setMaximumHeight(16777215)
            label.setMinimumWidth(0)
            label.setMaximumWidth(16777215)
        self.title_label.setText(_hint_title(spec))
        self.body_label.setText(_hint_description(spec))
        self.example_label.setText(_hint_example(spec))
        self.example_label.setVisible(bool(self.example_label.text().strip()))
        self.restart_badge.setVisible(bool(spec.restart_required))
        self.updated_badge.setVisible(bool(updated))

    def prepare_for_width(self, width: int) -> int:
        for label in (self.title_label, self.body_label, self.example_label):
            label.setMinimumHeight(0)
            label.setMaximumHeight(16777215)
            label.setMinimumWidth(0)
            label.setMaximumWidth(16777215)
        self.setFixedWidth(width)
        layout = self.layout()
        margins = layout.contentsMargins()
        body_width = max(80, width - margins.left() - margins.right())
        total = margins.top() + margins.bottom()
        spacing = layout.spacing()

        visible_heights: list[int] = []
        for label in (self.title_label, self.body_label, self.example_label):
            if label.isHidden():
                continue
            label.setFixedWidth(body_width)
            height = label.heightForWidth(body_width) if label.wordWrap() else label.sizeHint().height()
            if height < 0:
                height = label.sizeHint().height()
            if label.wordWrap():
                height += 6
            label.setMinimumHeight(height)
            visible_heights.append(height)

        badge_height = 0
        if not self.restart_badge.isHidden() or not self.updated_badge.isHidden():
            badge_height = max(self.restart_badge.sizeHint().height(), self.updated_badge.sizeHint().height())
            # Usually badges go after title (index 0) and body (index 1)
            # So insert at index 2 or just append if others are missing
            idx = min(2, len(visible_heights))
            visible_heights.insert(idx, badge_height)

        total += sum(visible_heights)
        total += spacing * max(0, len(visible_heights) - 1)
        total += 8
        self.setFixedHeight(total)
        self.setFixedWidth(width)
        return total




class HintButton(QToolButton):
    activated = Signal(object)
    hovered = Signal(object)
    unhovered = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("hint_button")

        # 22x22 — это hitbox (v36), сам круг рисуем 18x18.
        self.setFixedSize(22, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setText("")
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setAutoRaise(True)
        self._forced_hover = False

        self.setStyleSheet(
            "QToolButton#hint_button {"
            "border: 0;"
            "background: transparent;"
            "padding: 0px;"
            "margin: 0px;"
            "}"
        )

    def event(self, event) -> bool:
        if event.type() in {QEvent.Type.HoverEnter, QEvent.Type.HoverMove}:
            self.hovered.emit(self)
        return super().event(event)

    def enterEvent(self, event) -> None:
        self.hovered.emit(self)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.unhovered.emit(self)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self)
            event.accept()
            return
        super().mousePressEvent(event)

    def set_forced_hover(self, value: bool) -> None:
        value = bool(value)
        if getattr(self, "_forced_hover", False) == value:
            return
        self._forced_hover = value
        self.setCursor(Qt.CursorShape.PointingHandCursor if value else Qt.CursorShape.ArrowCursor)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        is_hover = self.underMouse() or bool(getattr(self, "_forced_hover", False))

        # Центрируем круг 18x18 внутри 22x22 (2.5, 2.5) (v36)
        rect = QRectF(2.5, 2.5, 17.0, 17.0)
        border = "rgba(139,92,246,.62)" if is_hover else "rgba(139,92,246,.40)"
        bg = "rgba(139,92,246,.24)" if is_hover else "rgba(139,92,246,.16)"
        text_color = "#c4b5fd" if is_hover else "#9f8bff"

        painter.setPen(QPen(_to_qcolor(border), 1))
        painter.setBrush(_to_qcolor(bg))
        painter.drawEllipse(rect)

        painter.setPen(_to_qcolor(text_color))
        font = _ui_font(pixel_size=10)
        font.setFamily("Segoe UI")
        painter.setFont(font)
        # Центрируем текст внутри круга
        painter.drawText(
            QRectF(2.0, 1.5, 18.0, 18.0),
            int(Qt.AlignmentFlag.AlignCenter),
            "?",
        )


class ApiAccessKeyManager(QFrame):
    def __init__(self, api: ApiClient, parent: QWidget | None = None, owner: QWidget | None = None):
        super().__init__(parent)
        self.api = api
        self.owner = owner
        self._backend = "api"
        self._accounts: list[dict[str, Any]] = []
        self._keys: list[dict[str, Any]] = []
        self._account_ids_by_login: dict[str, str] = {}
        self.setObjectName("settings_card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        title = QLabel("меню управления API-ключами")
        title.setObjectName("card_title")
        top.addWidget(title)
        top.addStretch(1)
        reload_btn = QPushButton("Обновить")
        reload_btn.clicked.connect(self.reload)
        top.addWidget(reload_btn)
        root.addLayout(top)

        add = QHBoxLayout()
        self.account_combo = QComboBox()
        self.account_combo.setEditable(True)
        self.account_combo.setMinimumWidth(120)
        self.hash_edit = QLineEdit()
        self.hash_edit.setPlaceholderText("SHA256")
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("или сырой KEY")
        self.key_edit.textChanged.connect(self._sync_hash_from_key)
        gen_btn = QPushButton("+")
        gen_btn.setToolTip("Сгенерировать сырой ключ")
        gen_btn.clicked.connect(self._generate_key)
        add_btn = QPushButton("Добавить")
        add_btn.setObjectName("primary_button")
        add_btn.clicked.connect(lambda: self._submit_key(replace=False))
        replace_btn = QPushButton("Сменить")
        replace_btn.clicked.connect(lambda: self._submit_key(replace=True))
        add.addWidget(self.account_combo)
        add.addWidget(self.hash_edit, 1)
        add.addWidget(self.key_edit, 1)
        add.addWidget(gen_btn)
        add.addWidget(add_btn)
        add.addWidget(replace_btn)
        root.addLayout(add)

        self.rows = QGridLayout()
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setHorizontalSpacing(8)
        self.rows.setVerticalSpacing(5)
        root.addLayout(self.rows)

        self.status = QLabel("")
        self.status.setObjectName("settings_muted")
        root.addWidget(self.status)
        self._syncing_hash = False
        self.reload()

    @staticmethod
    def _hash_key(value: str) -> str:
        return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _short_hash(value: str) -> str:
        text = str(value or "").strip()
        return text if len(text) <= 20 else f"{text[:12]}...{text[-8:]}"

    @staticmethod
    def _short_key(value: str) -> str:
        text = str(value or "").strip()
        return text if len(text) <= 28 else f"{text[:16]}...{text[-8:]}"

    @staticmethod
    def _table_label(text: str, *, muted: bool = False) -> QLabel:
        label = QLabel(str(text or ""))
        label.setObjectName("api_key_table_muted" if muted else "api_key_table_text")
        label.setFont(_ui_font(pixel_size=11, bold=True))
        return label

    def _clear_rows(self) -> None:
        while self.rows.count():
            item = self.rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _local_store(self):
        try:
            from api.auth_store import AuthStore

            return AuthStore()
        except Exception:
            from ui.local_auth_store import AuthStore

            return AuthStore()

    def _refresh_api_base_url(self) -> None:
        try:
            override = getattr(self.owner, "_current_api_base_url_from_fields", None)
            base_url = override() if callable(override) else ""
            self.api.set_base_url(base_url or get_selected_base_url())
        except Exception:
            pass

    def _can_use_local_fallback(self) -> bool:
        try:
            self._refresh_api_base_url()
            parsed = urlparse(str(self.api.base_url or get_selected_base_url() or ""))
            host = str(parsed.hostname or "").strip().lower()
            return host in {"", "127.0.0.1", "localhost", "::1"}
        except Exception:
            return False

    def _load_keys_payload(self) -> dict[str, Any]:
        try:
            self._refresh_api_base_url()
            payload = self.api.admin_list_api_access_keys()
            self._backend = "api"
            return payload
        except Exception as exc:
            if not self._can_use_local_fallback():
                raise exc
            store = self._local_store()
            try:
                self._backend = "local"
                return {
                    "accounts": store.list_accounts(),
                    "keys": store.list_api_access_keys(),
                }
            finally:
                try:
                    store._conn.close()
                except Exception:
                    pass

    def _create_key(self, *, login: str, key: str, key_hash: str, replace: bool) -> None:
        try:
            self._refresh_api_base_url()
            self.status.setText(f"POST -> {self.api.base_url}/auth/api-access-keys")
            self.api.admin_create_api_access_key(
                login=login,
                key=key,
                key_hash=key_hash,
                label="ui",
                replace=replace,
            )
            self._backend = "api"
            return
        except Exception as exc:
            if not self._can_use_local_fallback():
                raise exc
        store = self._local_store()
        try:
            if replace:
                store.delete_api_access_keys_for_account(login)
            store.set_api_access_key_hash(login, key_hash, label="ui", raw_key=key)
            self._backend = "local"
        finally:
            try:
                store._conn.close()
            except Exception:
                pass

    def _set_key_enabled(self, key_hash: str, enabled: bool) -> None:
        try:
            self._refresh_api_base_url()
            self.status.setText(f"PATCH -> {self.api.base_url}/auth/api-access-keys/...")
            self.api.admin_set_api_access_key_enabled(key_hash, enabled)
            self._backend = "api"
            return
        except Exception as exc:
            if not self._can_use_local_fallback():
                raise exc
        store = self._local_store()
        try:
            if not store.set_api_access_key_enabled(key_hash, enabled):
                raise ValueError("key_not_found")
            self._backend = "local"
        finally:
            try:
                store._conn.close()
            except Exception:
                pass

    def _delete_key_hash(self, key_hash: str) -> None:
        try:
            self._refresh_api_base_url()
            self.status.setText(f"DELETE -> {self.api.base_url}/auth/api-access-keys/...")
            self.api.admin_delete_api_access_key(key_hash)
            self._backend = "api"
            return
        except Exception as exc:
            if not self._can_use_local_fallback():
                raise exc
        store = self._local_store()
        try:
            if not store.delete_api_access_key(key_hash):
                raise ValueError("key_not_found")
            self._backend = "local"
        finally:
            try:
                store._conn.close()
            except Exception:
                pass

    def reload(self) -> None:
        try:
            payload = self._load_keys_payload()
            self._accounts = list(payload.get("accounts") or [])
            self._keys = list(payload.get("keys") or [])
            self._account_ids_by_login = {
                str(account.get("login") or "").strip().lower(): str(account.get("account_id") or "").strip()
                for account in self._accounts
                if str(account.get("login") or "").strip() and str(account.get("account_id") or "").strip()
            }
            self.status.setText("" if self._backend == "api" else "Локальный доступ к auth.db.")
        except Exception as exc:
            self._accounts = []
            self._keys = []
            self.status.setText(f"Не удалось загрузить: {exc}")
        self._refresh_accounts()
        self._render_rows()

    def _refresh_accounts(self) -> None:
        current = self.account_combo.currentData()
        self.account_combo.clear()
        for account in self._accounts:
            if bool(account.get("disabled")):
                continue
            login = str(account.get("login") or "").strip()
            if login:
                self.account_combo.addItem(login, login)
        if current:
            idx = self.account_combo.findData(current)
            if idx >= 0:
                self.account_combo.setCurrentIndex(idx)

    def _render_rows(self) -> None:
        self._clear_rows()
        for col, text in enumerate(("аккаунт", "sha256", "ключ", "копия", "вкл/выкл", "удалить")):
            label = self._table_label(text, muted=True)
            self.rows.addWidget(label, 0, col)
        if not self._keys:
            empty = self._table_label("Ключей пока нет.", muted=True)
            self.rows.addWidget(empty, 1, 0, 1, 6)
            return
        for row, item in enumerate(self._keys, start=1):
            login = str(item.get("login") or "")
            key_hash = str(item.get("key_hash") or "")
            raw_key = str(item.get("key") or "")
            revoked = bool(item.get("revoked"))
            login_label = self._table_label(login)
            hash_label = self._table_label(self._short_hash(key_hash))
            hash_label.setToolTip(key_hash)
            hash_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            raw_label = self._table_label(self._short_key(raw_key) if raw_key else "ключ неизвестен", muted=not bool(raw_key))
            raw_label.setToolTip(raw_key or "Этот ключ был добавлен только как SHA256, сырой KEY не сохранён.")
            raw_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            copy_btn = QPushButton("Копировать")
            copy_btn.setEnabled(bool(raw_key))
            copy_btn.clicked.connect(lambda _checked=False, key=raw_key: self._copy_key(key))
            enabled = QCheckBox("")
            enabled.setObjectName("api_key_enabled_checkbox")
            enabled.setFont(_ui_font(pixel_size=11, bold=True))
            enabled.setChecked(not revoked)
            enabled.stateChanged.connect(lambda _state, h=key_hash, login=login, box=enabled: self._set_enabled(h, box.isChecked(), login=login))
            delete_btn = QPushButton("Удалить")
            delete_btn.setObjectName("dangerButton")
            delete_btn.clicked.connect(lambda _checked=False, h=key_hash, login=login: self._delete_key(h, login=login))
            self.rows.addWidget(login_label, row, 0)
            self.rows.addWidget(hash_label, row, 1)
            self.rows.addWidget(raw_label, row, 2)
            self.rows.addWidget(copy_btn, row, 3)
            self.rows.addWidget(enabled, row, 4, Qt.AlignmentFlag.AlignCenter)
            self.rows.addWidget(delete_btn, row, 5)

    def _generate_key(self) -> None:
        key = "mmis_" + secrets.token_urlsafe(48)
        self.key_edit.setText(key)
        self.hash_edit.setText(self._hash_key(key))

    def _sync_hash_from_key(self, value: str) -> None:
        if getattr(self, "_syncing_hash", False):
            return
        key = str(value or "").strip()
        if not key:
            return
        self._syncing_hash = True
        try:
            self.hash_edit.setText(self._hash_key(key))
        finally:
            self._syncing_hash = False

    def _submit_key(self, *, replace: bool) -> None:
        login = str(self.account_combo.currentData() or self.account_combo.currentText() or "").strip().lower()
        key = self.key_edit.text().strip()
        key_hash = self.hash_edit.text().strip().lower()
        if not login:
            self.status.setText("Выбери аккаунт.")
            return
        if not key and not key_hash:
            self.status.setText("Введи сырой KEY или SHA256.")
            return
        try:
            final_key_hash = self._hash_key(key) if key else key_hash
            self._create_key(login=login, key=key, key_hash=final_key_hash, replace=replace)
            self._save_key_for_login_local_profile(login, key, final_key_hash)
            self.key_edit.clear()
            self.hash_edit.clear()
            self.reload()
            sync_key = getattr(self.owner, "_sync_api_access_key_fields", None)
            if callable(sync_key):
                sync_key(login, key, final_key_hash)
            self.status.setText("Ключ сменён." if replace else "Ключ добавлен.")
        except Exception as exc:
            self.status.setText(f"Не удалось добавить: {exc}")

    def _sync_login_from_rows(self, login: str) -> None:
        clean_login = str(login or "").strip().lower()
        sync_key = getattr(self.owner, "_sync_api_access_key_fields", None)
        if not clean_login or not callable(sync_key):
            return
        for item in self._keys:
            if str(item.get("login") or "").strip().lower() != clean_login:
                continue
            if bool(item.get("revoked")):
                continue
            raw_key = str(item.get("key") or "")
            key_hash = str(item.get("key_hash") or "")
            self._save_key_for_login_local_profile(clean_login, raw_key, key_hash)
            sync_key(clean_login, raw_key, key_hash)
            return
        self._save_key_for_login_local_profile(clean_login, "", "", clear=True)
        sync_key(clean_login, "", "")

    def _save_key_for_login_local_profile(self, login: str, raw_key: str, key_hash: str, *, clear: bool = False) -> None:
        clean_login = str(login or "").strip().lower()
        account_id = str(self._account_ids_by_login.get(clean_login) or "").strip()
        if not account_id:
            return
        connection: dict[str, Any] = {}
        values: dict[str, Any] = {}
        clean_key = str(raw_key or "").strip()
        clean_hash = str(key_hash or "").strip().lower()
        if clear:
            connection["api_access_key"] = ""
            values["api.access_lock.key_hash"] = ""
        elif clean_key:
            connection["api_access_key"] = clean_key
            if clean_hash:
                values["api.access_lock.key_hash"] = clean_hash
        elif clean_hash:
            values["api.access_lock.key_hash"] = clean_hash
        if not connection and not values:
            return
        try:
            save_account_client_config_updates(account_id, connection=connection, values=values)
        except Exception:
            pass

    def _copy_key(self, raw_key: str) -> None:
        key = str(raw_key or "").strip()
        if not key:
            self.status.setText("Сырой ключ не сохранён.")
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(key)
            self.status.setText("Ключ скопирован.")

    def _set_enabled(self, key_hash: str, enabled: bool, *, login: str = "") -> None:
        try:
            self._set_key_enabled(key_hash, enabled)
            self.reload()
            self._sync_login_from_rows(login)
            self.status.setText("Состояние обновлено.")
        except Exception as exc:
            self.status.setText(f"Не удалось изменить: {exc}")

    def _delete_key(self, key_hash: str, *, login: str = "") -> None:
        try:
            self._delete_key_hash(key_hash)
            self.reload()
            self._sync_login_from_rows(login)
            self.status.setText("Ключ удалён.")
        except Exception as exc:
            self.status.setText(f"Не удалось удалить: {exc}")


class SettingsWindow(QDialog):
    saved = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("settings_window")
        self.setWindowTitle("Настройки MMis")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if parent is not None:
            self.setWindowFlags(Qt.WindowType.Widget)
            parent.installEventFilter(self)
        else:
            self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.resize(1200, 820)
        self._payload: dict[str, Any] = {}
        self._panel: QFrame | None = None
        self._blur_target: QWidget | None = None
        self._scroll_overlay: ChatScrollOverlay | None = None
        self._hint_popup: SettingsHintPopup | None = None
        self._hint_targets: dict[QWidget, SettingSpec] = {}
        self._hint_rows: dict[QWidget, tuple[QWidget, SettingSpec]] = {}
        self._category_key = SETTINGS_CATEGORIES[0].key
        self._editors: dict[str, SettingEditor] = {}
        self._changed: dict[str, Any] = {}
        self._invalid: dict[str, str] = {}
        self._labels: dict[str, QLabel] = {}
        self._nav_buttons: dict[str, QPushButton] = {}
        self._sync_meta: dict[str, Any] = {}
        self._search_text = ""
        self.api = ApiClient()
        self._build_ui()
        self.reload()
        _validate_hint_coverage()

    def reload(self) -> None:
        try:
            self._payload, self._sync_meta = load_settings_payload(allow_remote=True)
        except Exception as exc:
            # Fallback for "offline" or broken state
            self._payload = {}
            self._sync_meta = {"status": "error", "message": str(exc), "offline": True}
        self._changed.clear()
        self._invalid.clear()
        self._render_category()
        self._refresh_preview()

    def _build_ui(self) -> None:
        self.setStyleSheet(SETTINGS_STYLE)
        apply_settings_tooltip_style()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        overlay = QFrame(self)
        overlay.setObjectName("settings_overlay")
        overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root.addWidget(overlay)
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(0)

        panel = QFrame(overlay)
        self._panel = panel
        panel.setObjectName("settings_panel")
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel.setMaximumSize(1360, 750)
        overlay_layout.addWidget(panel, 0, Qt.AlignmentFlag.AlignCenter)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        top = QFrame(panel)
        top.setObjectName("settings_top")
        top.setFixedHeight(58)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(16, 10, 12, 10)
        top_layout.setSpacing(12)
        gear = QLabel("⚙")
        gear.setObjectName("settings_gear")
        gear.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(gear, 0)
        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(1)

        title = QLabel("Настройки MMis")
        title.setObjectName("settings_title")
        title.setMinimumHeight(22)
        title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        subtitle = QLabel("Полупрозрачное окно поверх основного UI - config/settings.py + config.json")
        subtitle.setObjectName("settings_subtitle")
        subtitle.setMinimumHeight(17)
        subtitle.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        top_layout.addLayout(title_col, 1)
        self.search = QLineEdit(top)
        self.search.setObjectName("settings_search")
        self.search.setPlaceholderText("Поиск: memory top_k, ollama, voice...")
        self.search.setFixedWidth(324)
        self.search.textChanged.connect(self._on_search)
        top_layout.addWidget(self.search, 0)
        close_btn = QToolButton(top)
        close_btn.setText("x")
        close_btn.setObjectName("close_button")
        close_btn.clicked.connect(self.close)
        top_layout.addWidget(close_btn, 0)
        panel_layout.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        panel_layout.addLayout(body, 1)

        self.nav = QFrame(panel)
        self.nav.setObjectName("settings_nav")
        self.nav.setFixedWidth(240)
        nav_root_layout = QVBoxLayout(self.nav)
        nav_root_layout.setContentsMargins(0, 0, 0, 0)
        nav_root_layout.setSpacing(0)

        self.nav_scroll = QScrollArea(self.nav)
        self.nav_scroll.setObjectName("settings_nav_scroll")
        self.nav_scroll.setWidgetResizable(True)
        self.nav_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.nav_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav_scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        nav_root_layout.addWidget(self.nav_scroll)

        self.nav_content = QWidget()
        self.nav_content.setObjectName("settings_nav_content")
        self.nav_content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.nav_layout = QVBoxLayout(self.nav_content)
        self.nav_layout.setContentsMargins(14, 14, 22, 14) # Increased right margin
        self.nav_layout.setSpacing(4)
        self.nav_scroll.setWidget(self.nav_content)

        self._nav_scroll_overlay = ChatScrollOverlay(self.nav_scroll)

        # Создаем виджеты заранее (v46_fix), так как они нужны в _build_nav
        self.state_label = QLabel("")
        self.state_label.setObjectName("settings_muted")
        self.state_label.setWordWrap(True)

        self.diff_box = QPlainTextEdit()
        self.diff_box.setObjectName("json_preview")
        self.diff_box.setReadOnly(True)
        self.diff_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.diff_box.setFixedHeight(180)
        self.diff_box.setViewportMargins(6, 4, 12, 4)
        self._diff_scroll_overlay = PlainTextScrollOverlay(self.diff_box)

        self.warning_label = QLabel("")
        self.warning_label.setObjectName("settings_muted")
        self.warning_label.setWordWrap(True)

        self._build_nav()
        body.addWidget(self.nav)

        center = QWidget(panel)
        center.setObjectName("settings_center")
        center.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(14, 12, 10, 0)
        center_layout.setSpacing(9)
        header = QHBoxLayout()
        self.category_title = QLabel("")
        self.category_title.setObjectName("settings_title")
        self.category_desc = QLabel("")
        self.category_desc.setObjectName("settings_muted")
        self.category_desc.setWordWrap(True)
        header_text = QVBoxLayout()
        header_text.addWidget(self.category_title)
        header_text.addWidget(self.category_desc)
        header.addLayout(header_text, 1)
        reset_btn = QPushButton("Сбросить")
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.clicked.connect(self.reload)

        self.save_btn = QPushButton("Сохранить")
        self.save_btn.setObjectName("primary_button")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.clicked.connect(self._save)
        header.addWidget(reset_btn)
        header.addWidget(self.save_btn)
        center_layout.addLayout(header)

        self.scroll = QScrollArea(center)
        self.scroll.setObjectName("settings_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.scroll.viewport().setAutoFillBackground(False)
        self.content = QWidget()
        self.content.setObjectName("settings_content")
        self.content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.content_layout = QGridLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 18, 20)
        self.content_layout.setHorizontalSpacing(8)
        self.content_layout.setVerticalSpacing(10)
        self.scroll.setWidget(self.content)
        self._scroll_overlay = ChatScrollOverlay(self.scroll)
        self.scroll.verticalScrollBar().valueChanged.connect(lambda _value: self._hide_hint_popup())
        center_layout.addWidget(self.scroll, 1)
        body.addWidget(center, 1)

        self._hint_popup = SettingsHintPopup(self)

    def showEvent(self, event) -> None:
        self._fit_to_parent()
        self._apply_parent_blur()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._clear_parent_blur()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self._dispose_editors()
        cleanup_active_model_workers()
        self._clear_parent_blur()
        super().closeEvent(event)

    def _dispose_editors(self) -> None:
        for editor in list(self._editors.values()):
            control = editor.control()
            dispose = getattr(control, "dispose", None)
            if callable(dispose):
                try:
                    dispose()
                except Exception:
                    pass

        for row_widget in list(self._hint_rows.keys()):
            try:
                row_widget.removeEventFilter(self)
            except Exception:
                pass

        self._hint_rows.clear()
        self._editors.clear()
        self._labels.clear()

    def _fit_to_parent(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            self.resize(1200, 820)
            return
            
        is_dialog = bool(self.windowFlags() & Qt.WindowType.Dialog)
        if not is_dialog:
            self.setGeometry(0, 0, parent.width(), parent.height())
        else:
            top_left = parent.mapToGlobal(parent.rect().topLeft())
            self.setGeometry(top_left.x(), top_left.y(), parent.width(), parent.height())
            
        panel_width = max(1060, min(1360, parent.width() - 56))
        panel_height = max(660, min(728, parent.height() - 96))
        if self._panel is not None:
            self._panel.setFixedSize(panel_width, panel_height)

    def _apply_parent_blur(self) -> None:
        parent = self.parentWidget()
        target = parent.centralWidget() if hasattr(parent, "centralWidget") else parent
        if not isinstance(target, QWidget):
            return
        if self._blur_target is target:
            return
        self._clear_parent_blur()
        effect = QGraphicsBlurEffect(target)
        effect.setBlurRadius(7.0)
        effect.setBlurHints(QGraphicsBlurEffect.BlurHint.QualityHint)
        target.setGraphicsEffect(effect)
        self._blur_target = target

    def _clear_parent_blur(self) -> None:
        if self._blur_target is not None:
            self._blur_target.setGraphicsEffect(None)
            self._blur_target = None

    def _build_nav(self) -> None:
        while self.nav_layout.count():
            item = self.nav_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        current_group = ""
        for category in SETTINGS_CATEGORIES:
            if category.group != current_group:
                current_group = category.group
                label = QLabel(_group_title(current_group))
                label.setObjectName("settings_muted")
                self.nav_layout.addWidget(label)
            count = sum(len(card.settings) for card in self._filtered_cards(category))
            button = QPushButton(f"{_category_icon(category.key)} {category.title}  {count}")
            button.setObjectName("nav_button")
            button.setProperty("count", str(count))
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, key=category.key: self._select_category(key))
            self.nav_layout.addWidget(button)
            self._nav_buttons[category.key] = button
        self.nav_layout.addStretch(1)

        # Секция системной информации в боковой панели (v46)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(255, 255, 255, 12); margin: 10px 0;")
        self.nav_layout.addWidget(sep)

        info_header = QLabel("ИНФОРМАЦИЯ")
        info_header.setObjectName("settings_muted")
        info_header.setStyleSheet("margin-bottom: 4px;")
        self.nav_layout.addWidget(info_header)

        self.state_label.setMinimumWidth(0)
        self.state_label.setMaximumWidth(204)
        self.nav_layout.addWidget(self.state_label)

        # Фрейм для варнингов, если они есть
        self.sidebar_warn_frame = QFrame()
        self.sidebar_warn_frame.setObjectName("danger_card")
        self.sidebar_warn_frame.setStyleSheet("background: rgba(127, 29, 29, 20); border: 1px solid rgba(252, 165, 165, 30); margin: 4px 8px;")
        warn_layout = QVBoxLayout(self.sidebar_warn_frame)
        warn_layout.setContentsMargins(6, 6, 6, 6)
        warn_layout.addWidget(self.warning_label)
        self.sidebar_warn_frame.setMinimumWidth(0)
        self.sidebar_warn_frame.setMaximumWidth(204)
        self.sidebar_warn_frame.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.nav_layout.addWidget(self.sidebar_warn_frame)
        self.sidebar_warn_frame.hide()

        json_label = QLabel("CHANGES (JSON)")
        json_label.setObjectName("settings_muted")
        json_label.setStyleSheet("font-size: 9px; margin-top: 6px;")
        self.nav_layout.addWidget(json_label)

        self.diff_box.setFixedHeight(180) 
        self.diff_box.setMinimumWidth(0)
        self.diff_box.setMaximumWidth(204)
        self.diff_box.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.diff_box.setViewportMargins(8, 4, 16, 4)
        self.nav_layout.addWidget(self.diff_box)

    def _preview_card(self, title: str, widget: QWidget) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("settings_card")
        frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)
        label = QLabel(title)
        label.setStyleSheet("padding: 0 10px;")
        label.setObjectName("card_title")
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        layout.addSpacing(14)
        return frame

    def _select_category(self, key: str) -> None:
        self._category_key = key
        self._render_category()

    def _render_category(self) -> None:
        self._hide_hint_popup()
        self._hint_targets.clear()
        self._hint_rows.clear()
        cleanup_active_model_workers(wait_ms=800)
        self._dispose_editors()

        for button_key, button in self._nav_buttons.items():
            button.setChecked(button_key == self._category_key)

        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for row in range(self.content_layout.rowCount()):
            self.content_layout.setRowStretch(row, 0)
        category = get_category(self._category_key)
        self.category_title.setText(category.title)
        self.category_desc.setText(_category_description(category))

        if self._category_key == "characters":
            self._render_character_manager()
            return
        
        cards = self._filtered_cards(category)
        if not cards:
            empty = QLabel("No settings match the search.")
            empty.setObjectName("settings_muted")
            self.content_layout.addWidget(empty, 0, 0)
            return
        last_row = 0
        for index, card in enumerate(cards):
            two_column = index < 2 and len(cards) > 1
            row = index // 2 if two_column else (index - 1 if len(cards) > 1 else 0)
            col = index % 2 if two_column else 0
            last_row = max(last_row, row)
            frame = QFrame(self.content)
            frame.setObjectName("danger_card" if card.dangerous else "settings_card")
            frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            frame.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(0, 8, 0, 10)
            layout.setSpacing(4)
            header_layout = QHBoxLayout()
            header_layout.setContentsMargins(10, 0, 10, 2)
            title_lbl = QLabel(card.title)
            title_lbl.setObjectName("danger_title" if card.dangerous else "card_title")
            tag_lbl = CardTagBadge(card.tag, frame)
            header_layout.addWidget(title_lbl)
            header_layout.addStretch()
            header_layout.addWidget(tag_lbl, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addLayout(header_layout)
            for spec in card.settings:
                if not self._matches_search(spec, category.title, card.title):
                    continue
                layout.addWidget(self._setting_row(spec))
            layout.addStretch(1)
            colspan = 1 if two_column else 2
            self.content_layout.addWidget(frame, row, col, 1, colspan)
        if self._category_key == "main" and self._can_show_api_key_manager() and self._should_show_api_key_manager():
            manager = ApiAccessKeyManager(self.api, self.content, owner=self)
            api_row = last_row + 1
            self.content_layout.addWidget(manager, api_row, 0, 1, 2)
            last_row = api_row
        self.content_layout.setColumnStretch(0, 1)
        self.content_layout.setColumnStretch(1, 1)
        self.content_layout.setRowStretch(last_row + 1, 1)
        self._refresh_preview()

    def _setting_row(self, spec: SettingSpec) -> QWidget:
        row = QFrame()
        row.setObjectName("setting_row")
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.setFixedHeight(30)

        layout = QHBoxLayout(row)
        # Было 6 слева. Нужно +6px вправо для пары [название + ?] (v34/v36)
        layout.setContentsMargins(12, 4, 10, 2)
        layout.setSpacing(4)

        label = QLabel(_setting_title(spec), row)
        label.setObjectName("setting_label")
        label_font = _ui_font(pixel_size=11)
        label_font.setFamily("Cascadia Code")
        label.setFont(label_font)

        label.setFixedHeight(18)
        label.setWordWrap(False)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Полный текст, без max-width и без elide. (v37_fix2)
        text_width = label.fontMetrics().horizontalAdvance(label.text())
        label.setFixedWidth(text_width + 16) # 8px padding + 8px safety/border

        hint = HintButton(row)
        hint.setObjectName("hint_button")
        hint.setFixedSize(22, 22) # (v36)
        hint.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hint.setMouseTracking(True)
        hint.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        hint.hovered.connect(lambda _button, button=hint: self._show_hint_popup(button, self._hint_targets[button]))
        hint.activated.connect(lambda _button, button=hint: self._show_hint_popup(button, self._hint_targets[button]))
        hint.unhovered.connect(lambda _button: self._hide_hint_popup())
        self._hint_targets[hint] = spec

        # Страховка (v37/v38): ловим движение мыши на уровне всей строки.
        row.setMouseTracking(True)
        row.setCursor(Qt.CursorShape.ArrowCursor)
        row.installEventFilter(self)
        self._hint_rows[row] = (hint, spec)

        layout.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1) # Pushes editor to the right

        value = self._changed.get(spec.path, dotted_get(self._payload, spec.path))
        editor = SettingEditor(spec, value, row)
        editor.valueChanged.connect(lambda value, path=spec.path: self._on_editor_changed(path, value))
        control = editor.control()
        control.setMinimumWidth(44)
        layout.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
        self._editors[spec.path] = editor
        self._labels[spec.path] = label
        self._update_badge(spec.path)
        return row

    def _render_character_manager(self) -> None:
        """Рендерит специальный интерфейс управления персонажами."""
        manager = CharacterManager(self.api, self.content)
        self.content_layout.addWidget(manager, 0, 0, 1, 2)
        self.content_layout.setRowStretch(0, 1)
        self.content_layout.setColumnStretch(0, 1)
        self.content_layout.setColumnStretch(1, 1)
        self._refresh_preview()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self._fit_to_parent()

        if isinstance(watched, QWidget) and watched in self._hint_rows:
            hint, spec = self._hint_rows[watched]

            if event.type() in {
                QEvent.Type.MouseMove,
                QEvent.Type.HoverMove,
                QEvent.Type.Enter,
                QEvent.Type.MouseButtonPress,
            }:
                try:
                    if hasattr(event, "position"):
                        pos = event.position().toPoint()
                    elif hasattr(event, "pos"):
                        pos = event.pos()
                    else:
                        pos = None

                    if pos is not None:
                        # Расширенная зона вокруг видимого `?`.
                        hint_rect = hint.geometry().adjusted(-6, -4, 6, 4)
                        inside = hint_rect.contains(pos)

                        if inside:
                            watched.setCursor(Qt.CursorShape.PointingHandCursor)

                            if hasattr(hint, "set_forced_hover"):
                                hint.set_forced_hover(True)

                            self._show_hint_popup(hint, spec)

                            if event.type() == QEvent.Type.MouseButtonPress:
                                event.accept()
                                return True
                        else:
                            watched.setCursor(Qt.CursorShape.ArrowCursor)

                            if hasattr(hint, "set_forced_hover"):
                                hint.set_forced_hover(False)

                            self._hide_hint_popup()

                except Exception:
                    pass

            elif event.type() in {QEvent.Type.Leave, QEvent.Type.Hide}:
                watched.setCursor(Qt.CursorShape.ArrowCursor)

                if hasattr(hint, "set_forced_hover"):
                    hint.set_forced_hover(False)

                self._hide_hint_popup()

        return super().eventFilter(watched, event)

    def _show_hint_popup(self, anchor: QWidget, spec: SettingSpec) -> None:
        if self._hint_popup is None:
            return
        self._hint_popup.set_spec(spec, updated=spec.path in self._changed)
        width = _hint_popup_width(spec)
        height = self._hint_popup.prepare_for_width(width)
        preferred = anchor.mapTo(self, QPoint(-12, anchor.height() + 7))
        panel_rect = self._panel.geometry() if self._panel is not None else self.rect()
        x = max(panel_rect.left() + 12, min(preferred.x(), panel_rect.right() - width - 12))
        y = preferred.y()
        if y + height > panel_rect.bottom() - 12:
            y = anchor.mapTo(self, QPoint(-12, -height - 7)).y()
        y = max(panel_rect.top() + 12, min(y, panel_rect.bottom() - height - 12))
        self._hint_popup.move(x, y)
        self._hint_popup.show()
        self._hint_popup.raise_()

    def _hide_hint_popup(self) -> None:
        for hint, _spec in list(getattr(self, "_hint_rows", {}).values()):
            if hasattr(hint, "set_forced_hover"):
                hint.set_forced_hover(False)

        if self._hint_popup is not None:
            self._hint_popup.hide()

    def _on_editor_changed(self, path: str, value: Any) -> None:
        editor = self._editors.get(path)
        if editor is not None:
            ok, message = editor.validate_value()
            if ok:
                self._invalid.pop(path, None)
            else:
                self._invalid[path] = message
        original = editor.initial_value() if editor is not None else dotted_get(self._payload, path)
        if value == original:
            self._changed.pop(path, None)
        else:
            self._changed[path] = value
        self._update_badge(path)
        self._refresh_preview()

    def _update_badge(self, path: str) -> None:
        label = self._labels.get(path)
        changed = path in self._changed

        if label:
            label.setObjectName("setting_label_changed" if changed else "setting_label")
            label.style().unpolish(label)
            label.style().polish(label)

            # После смены objectName QSS может поменять padding/border.
            # Возвращаем точную ширину полного текста + запас для padding (v37_fix2).
            text_width = label.fontMetrics().horizontalAdvance(label.text())
            label.setFixedWidth(text_width + 16)
            label.update()
            label.repaint()

    def _refresh_preview(self) -> None:
        api_host = dotted_get(self._payload, "api.host", "")
        api_port = dotted_get(self._payload, "api.port", "")
        profile = dotted_get(self._payload, "startup.active_profile", "")
        model = dotted_get(self._payload, "llm.model_name", "")
        web_mode = dotted_get(self._payload, "internet.web_mode", "")
        active_api = dotted_get(self._payload, "ui.api.active_endpoint", "local")
        local_api = dotted_get(self._payload, "ui.api.local_base_url", "")
        public_api = dotted_get(self._payload, "ui.api.public_base_url", "")
        selected_api = public_api if str(active_api).lower() == "public" else local_api
        online = bool(self._sync_meta.get("online"))
        source = self._sync_meta.get("source", "schema")
        pending = self._sync_meta.get("pending_count", 0)
        
        self.state_label.setText(
            f"API                                      {selected_api or f'{api_host}:{api_port}'}\n"
            f"Profile: {profile}\n"
            f"Memory: enabled\n"
            f"Web: {web_mode}\n"
            f"Sync: {'Online' if online else 'Offline'} ({source})\n"
            f"Pending: {pending}\n"
            f"Changed: {len(self._changed)}"
        )
        self.diff_box.setPlainText(json.dumps(self._changed, ensure_ascii=False, indent=2, sort_keys=True))
        warnings: list[str] = []
        for path, message in self._invalid.items():
            warnings.append(f"{path}: invalid value ({message})")
        for path in self._changed:
            spec = _spec_for(path)
            if spec and spec.restart_required:
                warnings.append(f"{path}: restart required")
            if spec and spec.dangerous:
                warnings.append(f"{path}: dangerous setting")

        warn_text = "\n".join(warnings) if warnings else ""
        self.warning_label.setText(warn_text)
        if hasattr(self, "sidebar_warn_frame"):
            self.sidebar_warn_frame.setVisible(bool(warn_text))
            
        self.save_btn.setEnabled(bool(self._changed) and not bool(self._invalid))

    def _save(self) -> None:
        updates: dict[str, Any] = {}
        errors: list[str] = []
        for path, message in self._invalid.items():
            errors.append(f"{path}: {message}")
        for path, editor in self._editors.items():
            if path not in self._changed:
                continue
            ok, message = editor.validate_value()
            if not ok:
                errors.append(f"{path}: {message}")
                continue
            updates[path] = editor.value()
        for path, value in self._changed.items():
            if path not in updates and path not in self._editors:
                updates[path] = value
        if errors:
            MmisMessageBox.warning(self, "Settings validation", "\n".join(errors))
            return
        if not updates:
            return
        online, message = save_settings_updates(updates, sync_remote=True)
        self._sync_meta["online"] = bool(online)
        self._sync_meta["source"] = "server" if online else "cache"
        
        for path, value in updates.items():
            dotted_set(self._payload, path, value)
        
        self._changed.clear()
        self._invalid.clear()
        
        for path in updates:
            self._update_badge(path)
            
        self._refresh_preview()
        self.saved.emit(copy.deepcopy(updates))
        if online:
            self._apply_live_runtime_updates(updates)
        
        if not online:
            try:
                from ui.api_client import ApiClient
                ApiClient().ping(timeout=1.5)
                api_reachable = True
            except Exception:
                api_reachable = False

            if not api_reachable:
                SettingsMessageBox.warning(
                    self,
                    "Settings sync",
                    f"Настройки сохранены локально, но не отправлены в MMis API.\n\nПричина: {message}"
                )

    def _apply_live_runtime_updates(self, updates: dict) -> None:
        model_name = str(dict(updates or {}).get("llm.model_name") or "").strip()
        if not model_name:
            return
        try:
            self.api.set_model(model_name)
        except Exception as exc:
            MmisMessageBox.warning(self, "Model", f"Настройка сохранена, но runtime-модель не переключилась:\n{exc}")


    def _on_search(self, text: str) -> None:
        self._search_text = str(text or "").strip().lower()
        if self._search_text:
            current = get_category(self._category_key)
            if not self._filtered_cards(current):
                for category in SETTINGS_CATEGORIES:
                    if self._filtered_cards(category):
                        self._category_key = category.key
                        break
        self._render_category()

    def _filtered_cards(self, category: SettingCategory):
        is_admin = self._is_admin_user()
        cards = []
        for card in category.cards:
            visible_settings = tuple(
                spec for spec in card.settings
                if is_admin or not getattr(spec, "admin_only", False)
            )
            if not visible_settings:
                continue
            visible_card = SettingCard(
                title=card.title,
                tag=card.tag,
                settings=visible_settings,
                dangerous=card.dangerous,
            )
            if not getattr(self, "_search_text", ""):
                cards.append(visible_card)
            elif any(self._matches_search(spec, category.title, visible_card.title) for spec in visible_card.settings):
                cards.append(visible_card)
        return tuple(cards)

    def _is_admin_user(self) -> bool:
        try:
            from ui.auth_client_store import load_auth_state

            state = load_auth_state()
            token = str(state.get("token") or "").strip()
            login = str(state.get("login") or "").strip().lower()
            role = str(state.get("role") or "").strip().lower()
            return bool(token) and (role == "admin" or login == "admin")
        except Exception:
            return False

    def _can_show_api_key_manager(self) -> bool:
        try:
            from ui.auth_client_store import load_auth_state

            state = load_auth_state()
            token = str(state.get("token") or "").strip()
            login = str(state.get("login") or "").strip().lower()
            role = str(state.get("role") or "").strip().lower()
            return role == "admin" or login == "admin"
        except Exception:
            return self._is_admin_user()

    def _current_setting_value(self, path: str) -> Any:
        if path in self._changed:
            return self._changed[path]
        editor = self._editors.get(path)
        if editor is not None:
            try:
                return editor.value()
            except Exception:
                pass
        return dotted_get(self._payload, path)

    def _current_api_base_url_from_fields(self) -> str:
        active = str(self._current_setting_value("ui.api.active_endpoint") or "local").strip().lower()
        path = "ui.api.public_base_url" if active == "public" else "ui.api.local_base_url"
        url = str(self._current_setting_value(path) or "").strip()
        if not url:
            return ""
        url = url.rstrip("/")
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "http://" + url
        return url

    def _active_login(self) -> str:
        try:
            from ui.auth_client_store import load_auth_state

            state = load_auth_state()
            return str(state.get("login") or "").strip().lower()
        except Exception:
            return ""

    def _set_setting_value(self, path: str, value: Any, *, persist_local: bool = False) -> None:
        dotted_set(self._payload, path, value)
        editor = self._editors.get(path)
        if editor is not None:
            try:
                editor.blockSignals(True)
                editor.set_value(value)
            finally:
                editor.blockSignals(False)
        self._changed.pop(path, None)
        if persist_local:
            try:
                save_settings_updates({path: value}, sync_remote=False)
            except Exception:
                pass
        self._update_badge(path)
        self._refresh_preview()

    def _sync_api_access_key_fields(self, login: str, raw_key: str, key_hash: str) -> None:
        if str(login or "").strip().lower() != self._active_login():
            return
        clean_key = str(raw_key or "").strip()
        clean_hash = str(key_hash or "").strip().lower()
        self._set_setting_value("ui.api.api_access_key", clean_key, persist_local=True)
        self._set_setting_value("api.access_lock.key_hash", clean_hash, persist_local=True)

    def _should_show_api_key_manager(self) -> bool:
        needle = str(getattr(self, "_search_text", "") or "").strip().lower()
        if not needle:
            return True
        haystack = "api key access sha256 hash ключ аккаунт account управление"
        return needle in haystack

    def _matches_search(self, spec: SettingSpec, category_title: str, card_title: str) -> bool:
        needle = getattr(self, "_search_text", "")
        if not needle:
            return True
        haystack = " ".join(
            [
                category_title,
                card_title,
                spec.path,
                spec.title,
                spec.description,
                spec.example,
                " ".join(spec.options),
            ]
        ).lower()
        return needle in haystack


def _hint_text(spec: SettingSpec) -> str:
    parts = [_setting_title(spec), _hint_title(spec), _hint_description(spec), _hint_example(spec)]
    return "\n".join(part for part in parts if part)


def _hint_title(spec: SettingSpec) -> str:
    return spec.path


def _setting_title(spec: SettingSpec) -> str:
    return _TITLE_BY_PATH.get(spec.path, spec.title or spec.path)


def _hint_description(spec: SettingSpec) -> str:
    text = str(_DESCRIPTION_BY_PATH.get(spec.path) or spec.description or "").strip()

    if not text:
        text = (
            f"Описание для {spec.path} ещё не задано. "
            "Добавь его в _DESCRIPTION_BY_PATH или в SettingSpec.description."
        )

    if spec.dangerous:
        text = f"{text}\n\nОпасный параметр: меняй только если понимаешь последствия."

    return text


def _hint_example(spec: SettingSpec) -> str:
    if spec.example:
        return str(spec.example)
    if spec.options:
        return " / ".join(str(option) for option in spec.options)
    return ""


def _hint_popup_width(spec: SettingSpec) -> int:
    from PySide6.QtGui import QFontMetrics

    font = _ui_font(pixel_size=11)
    font.setFamily("Cascadia Code")
    metrics = QFontMetrics(font)
    description = _hint_description(spec)
    example = _hint_example(spec)
    words = description.replace("\n", " ").split()
    longest_word_width = max((metrics.horizontalAdvance(word) for word in words), default=0)
    example_width = max((metrics.horizontalAdvance(line) for line in example.splitlines()), default=0)
    title_width = metrics.horizontalAdvance(spec.path)

    target = max(236, title_width + 42, longest_word_width + 62, min(example_width + 42, 420))
    if len(description) > 220 or len(example) > 90:
        target = max(target, 360)
    elif len(description) > 150 or len(example) > 60:
        target = max(target, 320)
    return min(440, target)


_TITLE_BY_PATH: dict[str, str] = {
    "app.name": "Название приложения",
    "app.locale": "Локаль приложения",
    "app.default_language": "Язык по умолчанию",
    "app.debug": "Debug mode",
    "startup.mode": "Режим запуска",
    "startup.active_profile": "Активный профиль",
    "startup.safety_mode": "Режим безопасности",
    "api.host": "API host",
    "api.port": "API port",
    "api.access_lock.enabled": "API lock",
    "api.access_lock.key_hash": "API access key SHA256",
    "ui.api.active_endpoint": "Активный API endpoint",
    "ui.api.local_base_url": "Локальный API URL",
    "ui.api.public_base_url": "Публичный API URL",
    "ui.api.api_access_key": "API access key",
    "llm.provider": "Провайдер LLM",
    "llm.model_name": "Основная модель",
    "llm.thinking_enabled": "Thinking",
    "llm.json_mode_enabled": "JSON mode",
    "llm.model_fallbacks": "Fallback модели",
    "llm.providers.ollama.base_url": "Ollama URL",
    "llm.providers.ollama.timeout_sec": "Ollama timeout",
    "llm.providers.ollama.retries": "Ollama retries",
    "llm.providers.ollama.keep_alive": "Unload after",
    "llm.providers.openai.api_key": "OpenAI API key",
    "llm.providers.openai.api_url": "OpenAI API URL",
    "llm.providers.openai.timeout_sec": "OpenAI timeout",
    "llm.providers.openai.max_retries": "OpenAI retries",
    "memory.enabled": "Память",
    "memory.memory_dir": "Папка памяти",
    "memory.cache_dir": "Папка кеша",
    "memory.db_path": "База памяти",
    "memory.chat_recall_results": "Recall results",
    "memory.chat_events_limit": "Events limit",
    "memory.chat_proofread": "Proofread памяти",
    "memory.chat_proofread_strict": "Строгий proofread",
    "memory_core.enabled": "Memory Core",
    "memory_core.enable_background_worker": "Background worker",
    "memory_core.worker_poll_interval": "Worker poll interval",
    "memory_core.memory_llm.keep_alive": "Memory LLM unload after",
    "internet.enabled": "Интернет",
    "internet.web_mode": "Web mode",
    "internet.search.provider": "Search provider",
    "internet.search.api_url": "Search API URL",
    "internet.search.timeout_sec": "Search timeout",
    "internet.fetch.timeout_sec": "Fetch timeout",
    "internet.fetch.retries": "Fetch retries",
    "internet.fetch.clean_max_chars": "Clean max chars",
    "internet.fetch.clean_min_chars": "Clean min chars",
    "internet.web_v2": "Web v2 config",
    "voice.enabled": "Голос",
    "voice.mode": "Режим голоса",
    "voice.open_mode_from_rail": "Открывать voice из rail",
    "voice.auto_speak_replies": "Автоозвучка",
    "voice.barge_in": "Barge-in",
    "voice.stt_engine": "STT engine",
    "voice.stt_model": "STT model",
    "voice.stt_device": "STT device",
    "voice.stt_compute_type": "STT compute type",
    "voice.stt_language_hint": "STT language",
    "voice.tts_engine": "TTS engine",
    "voice.tts_model": "TTS model",
    "voice.tts_device": "TTS device",
    "voice.tts.voice": "TTS voice",
    "voice.tts.rate": "TTS rate",
    "voice.tts.volume": "TTS volume",
    "dialog.new_session_after_min": "Новая сессия",
    "dialog.greeting_max_words": "Greeting max words",
    "dialog.greeting_max_chars": "Greeting max chars",
    "dialog.greetings": "Приветствия",
    "dialog.greeting_exclusions": "Исключения приветствий",
    "ui.console.timeout_sec": "Console timeout",
    "ui.console.stream_timeout_sec": "Stream timeout",
    "ui.console.store_turn": "Store turn",
    "ui.console.show_thinking": "Показывать thinking",
    "ui.console.thinking_first": "Thinking первым",
    "ui.console.auto_start_api": "Автостарт API",
    "ui.console.auto_start_ollama": "Автозапуск Ollama",
    "ui.ollama.start_mode": "Режим запуска Ollama",
    "ui.ollama.serve_exe": "Путь к Ollama serve",
    "ui.ollama.models_dir": "Папка моделей Ollama",
    "debug.memory_inspector_enabled": "Memory Inspector",
    "debug.show_raw_scores": "Raw scores",
    "debug.show_filtered_items": "Filtered items",
    "debug.show_prompt_blocks": "Prompt blocks",
    "logging.level": "Уровень логов",
    "logging.file": "Файл логов",
    "logging.colors": "Цветные логи",
    "logging.max_bytes": "Размер лог-файла",
    "logging.backup_count": "Количество backup",
    "logging.format": "Формат логов",
    "logging.web_trace_enabled": "Web trace",
    "logging.web_trace_logger": "Web trace logger",
    "logging.channels": "Каналы логов",
    "modules.automation_enabled": "Automation",
    "modules.screen_enabled": "Screen tools",
    "prompt.response_safety_filter_enabled": "Safety filter",
    "prompt.response_formatting_enabled": "Response formatting",
}


_DESCRIPTION_BY_PATH: dict[str, str] = {
    "app.name": "Имя, которое показывается в UI, логах и служебных сообщениях приложения.",
    "app.locale": "Локаль интерфейса и форматирования. Влияет на языковые подсказки и региональные значения.",
    "app.default_language": "Основной язык ответов и внутренних подсказок, если пользователь явно не выбрал другой.",
    "app.debug": "Включает расширенную диагностику и больше служебной информации в логах.",
    "startup.mode": "Определяет, как стартует приложение: API, UI или комбинированный режим.",
    "startup.active_profile": "Выбирает профиль производительности из performance_profiles.json для основного LLM runtime.",
    "startup.safety_mode": "Ограничивает или разрешает потенциально опасные действия инструментов и автоматизации.",
    "api.host": "Адрес, на котором API-сервер принимает подключения. 127.0.0.1 только локально, 0.0.0.0 для сети.",
    "api.port": "Порт FastAPI/uvicorn сервера. UI и внешние клиенты должны ходить на этот порт.",
    "ui.api.active_endpoint": "Выбирает, куда десктопный UI будет отправлять запросы: локальный или публичный API endpoint.",
    "ui.api.local_base_url": "URL API для подключения с этой же машины. Обычно это 127.0.0.1 с портом приложения.",
    "ui.api.public_base_url": "URL API для подключения с другого устройства или через внешний адрес.",
    "llm.provider": "Основной backend генерации ответов: локальная Ollama, OpenAI-compatible endpoint или auto.",
    "llm.model_name": "Модель, которой отвечает главный чат. Значение должно совпадать с именем модели у провайдера.",
    "llm.thinking_enabled": "Разрешает reasoning/thinking режим для моделей, которые его поддерживают.",
    "llm.json_mode_enabled": "Включает JSON mode по умолчанию для запросов, которым нужен структурированный ответ.",
    "llm.model_fallbacks": "Список моделей, которые можно пробовать, если основная модель недоступна.",
    "llm.providers.ollama.base_url": "Адрес Ollama API, к которому подключается приложение.",
    "llm.providers.ollama.timeout_sec": "Сколько секунд ждать ответ Ollama до ошибки timeout.",
    "llm.providers.ollama.retries": "Сколько раз повторять запрос к Ollama после временной ошибки.",
    "llm.providers.openai.api_key": "Ключ для OpenAI-compatible API. Если хранить здесь, он попадёт в config.json.",
    "llm.providers.openai.api_url": "Base URL OpenAI-compatible API.",
    "llm.providers.openai.timeout_sec": "Сколько секунд ждать ответ OpenAI-compatible API.",
    "llm.providers.openai.max_retries": "Сколько повторных попыток делать для OpenAI-compatible API.",
    "memory.enabled": "Включает или выключает слой памяти для чата.",
    "memory.memory_dir": "Папка, где лежат файлы памяти, состояния и runtime-артефакты.",
    "memory.cache_dir": "Папка временного кеша для памяти и вспомогательных процессов.",
    "memory.db_path": "Путь к SQLite базе Memory Core.",
    "memory.chat_recall_results": "Сколько найденных воспоминаний подтягивать в контекст ответа.",
    "memory.chat_events_limit": "Сколько последних событий чата учитывать при сборке контекста.",
    "memory.chat_proofread": "Включает дополнительную проверку памяти перед использованием в ответе.",
    "memory.chat_proofread_strict": "Делает проверку памяти строже, снижая риск мусорного контекста.",
    "memory_core.enabled": "Включает новый Memory Core runtime.",
    "memory_core.enable_background_worker": "Разрешает фоновый Memory Core worker, который обрабатывает очередь событий памяти.",
    "memory_core.worker_poll_interval": "Интервал в секундах, с которым Memory Core worker проверяет очередь задач.",
    "memory_core.memory_llm.keep_alive": "Сколько держать Memory LLM в памяти после обработки задач. Поддерживает формат Ollama: 30s, 5m, 30m, 1h, 0.",
    "ui.ollama.start_mode": (
        "Выбирает способ автозапуска Ollama, если её API сейчас недоступен. "
        "serve запускает указанный ollama.exe с аргументом serve. "
        "ui выполняет команду ollama list, чтобы Ollama сама подняла API-контур."
    ),
    "ui.ollama.serve_exe": (
        "Путь к ollama.exe, который будет использоваться в режиме serve. "
        "MMis запустит его как '<ollama.exe> serve'. "
        "Оставь пустым, если ollama доступна из PATH."
    ),
    "ui.ollama.models_dir": (
        "Папка, где Ollama должна искать и хранить модели. "
        "При запуске MMis передаёт этот путь через переменную окружения OLLAMA_MODELS."
    ),
    "internet.enabled": "Разрешает интернет-модуль и web-поиск.",
    "internet.web_mode": "Определяет агрессивность web-поиска: off, auto, on или aggressive.",
    "internet.search.provider": "Выбирает поисковый backend, например локальный SearXNG.",
    "internet.search.api_url": "Endpoint поискового API.",
    "internet.search.timeout_sec": "Сколько секунд ждать ответ поискового сервиса.",
    "internet.fetch.timeout_sec": "Сколько секунд ждать загрузку страницы.",
    "internet.fetch.retries": "Сколько раз повторять fetch после временной ошибки.",
    "internet.fetch.clean_max_chars": "Максимальный размер очищенного текста страницы.",
    "internet.fetch.clean_min_chars": "Минимальный размер текста, при котором страница считается полезной.",
    "internet.web_v2": "Расширенная JSON-конфигурация web policy, бюджетов, trust policy и citations.",
    "voice.enabled": "Включает голосовые функции приложения.",
    "voice.mode": "Определяет, как работает голосовой ввод: push-to-talk или переключатель.",
    "voice.open_mode_from_rail": "Разрешает открывать голосовой экран кнопкой в левой панели.",
    "voice.auto_speak_replies": "Автоматически озвучивает ответы ассистента.",
    "voice.barge_in": "Позволяет перебить озвучку новым голосовым вводом.",
    "voice.stt_engine": "Движок распознавания речи.",
    "voice.stt_model": "Модель распознавания речи.",
    "voice.stt_device": "Устройство для STT: cuda, cpu или auto.",
    "voice.stt_compute_type": "Тип вычислений STT, влияющий на скорость и VRAM/RAM.",
    "voice.stt_language_hint": "Языковая подсказка для распознавания речи.",
    "voice.tts_engine": "Движок озвучки ответов.",
    "voice.tts_model": "Модель синтеза речи.",
    "voice.tts_device": "Устройство для TTS: cuda, cpu или auto.",
    "voice.tts.voice": "Голос или пресет, который используется TTS.",
    "voice.tts.rate": "Скорость речи TTS.",
    "voice.tts.volume": "Громкость речи TTS.",
    "dialog.new_session_after_min": "Через сколько минут простоя начинать новую диалоговую сессию.",
    "dialog.greeting_max_words": "Максимум слов в автоприветствии.",
    "dialog.greeting_max_chars": "Максимум символов в автоприветствии.",
    "dialog.greetings": "Список разрешённых вариантов приветствий.",
    "dialog.greeting_exclusions": "Слова и шаблоны, при которых приветствие не показывается.",
    "ui.console.timeout_sec": "Timeout коротких API-запросов из UI/console.",
    "ui.console.stream_timeout_sec": "Timeout потокового ответа чата.",
    "ui.console.store_turn": "Сохранять реплики в историю и память.",
    "ui.console.show_thinking": "Показывать блок thinking в интерфейсе.",
    "ui.console.thinking_first": "Показывать thinking перед финальным ответом.",
    "ui.console.auto_start_api": "Автоматически стартовать API при запуске UI/console.",
    "ui.console.auto_start_ollama": "Если включено, desktop UI попробует запустить `ollama serve`, когда Ollama недоступна.",
    "debug.memory_inspector_enabled": "Включает Memory Inspector в интерфейсе.",
    "debug.show_raw_scores": "Показывает сырые оценки retrieval/scoring.",
    "debug.show_filtered_items": "Показывает элементы, отфильтрованные из memory retrieval.",
    "debug.show_prompt_blocks": "Показывает блоки промпта для отладки сборки контекста.",
    "logging.level": "Минимальный уровень сообщений, которые пишутся в лог.",
    "logging.file": "Файл, куда пишутся логи приложения.",
    "logging.colors": "Включает цветной вывод логов в консоли.",
    "logging.max_bytes": "Максимальный размер одного лог-файла до ротации.",
    "logging.backup_count": "Количество старых лог-файлов, которые сохраняются при ротации.",
    "logging.format": "Формат строки логов.",
    "logging.web_trace_enabled": "Включает отдельный trace web-пайплайна.",
    "logging.web_trace_logger": "Имя logger для web trace.",
    "logging.channels": "JSON-настройка каналов логирования и их prefix-фильтров.",
    "modules.automation_enabled": "Разрешает модуль автоматизации действий.",
    "modules.screen_enabled": "Разрешает screen/OCR инструменты.",
    "prompt.response_safety_filter_enabled": "Включает фильтр безопасности финального ответа.",
    "prompt.response_formatting_enabled": "Включает постобработку форматирования ответа.",
}


def _group_title(group: str) -> str:
    mapping = {
        "Core": "ОСНОВНОЕ",
        "Modules": "МОДУЛИ",
        "System": "СИСТЕМА",
    }
    return mapping.get(str(group or ""), str(group or "").upper())


def _category_icon(key: str) -> str:
    return {
        "main": "⌂",
        "memory": "●",
        "llm": "◆",
        "web": "◉",
        "voice": "⌕",
        "dialog_ui": "▣",
        "debug": "⚙",
        "logging": "□",
        "safety": "!",
        "characters": "👤",
    }.get(str(key or ""), "•")


def _category_description(category: SettingCategory) -> str:
    descriptions = {
        "main": "Базовые параметры приложения, API, режим запуска и безопасные флаги. Нужные подсказки появляются при наведении на знак вопроса.",
        "llm": "Основная модель, провайдеры и runtime-флаги. Лимитов max_tokens здесь нет: длину держат num_ctx и num_batch.",
        "memory": "Пути, воркер памяти и параметры retrieval для контекста чата.",
        "web": "Поиск, fetch, web mode и политика внешних источников.",
        "voice": "STT, TTS, голосовой режим и устройства.",
        "dialog_ui": "Поведение диалога, консольные флаги и отображение thinking.",
        "debug": "Inspector, prompt blocks и диагностические панели.",
        "logging": "Логи, ротация, каналы и web trace.",
        "safety": "Опасные переключатели автоматизации и safety-фильтров.",
        "characters": "Управление списком персонажей, их именами, ролями и активным состоянием.",
    }
    return descriptions.get(category.key, category.description)


def _spec_for(path: str) -> SettingSpec | None:
    for category in SETTINGS_CATEGORIES:
        for card in category.cards:
            for spec in card.settings:
                if spec.path == path:
                    return spec
    return None


def _validate_hint_coverage() -> None:
    missing_titles: list[str] = []
    missing_descriptions: list[str] = []

    for category in SETTINGS_CATEGORIES:
        for card in category.cards:
            for spec in card.settings:
                title = str(_TITLE_BY_PATH.get(spec.path) or spec.title or "").strip()
                description = str(_DESCRIPTION_BY_PATH.get(spec.path) or spec.description or "").strip()

                if not title:
                    missing_titles.append(spec.path)

                if not description:
                    missing_descriptions.append(spec.path)

    if missing_titles or missing_descriptions:
        lines: list[str] = []
        if missing_titles:
            lines.append("Нет title для: " + ", ".join(missing_titles))
        if missing_descriptions:
            lines.append("Нет description для: " + ", ".join(missing_descriptions))

        print("[settings hints] " + " | ".join(lines))
