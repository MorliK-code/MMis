from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal, Qt, QSize, QRect, QRectF, QTimer, QEvent, QThread
from PySide6.QtGui import QCursor, QFont, QFontMetricsF, QIcon, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.settings_schema import SettingSpec
from ui.chat_shell import (
    ToggleSwitch,
    PaintedButton,
    _ui_font,
    _to_qcolor,
    TEXT,
    MUTED,
    LINE,
    PlainTextScrollOverlay,
)
from ui.api_client import ApiClient
from ui.client_config_store import get_ollama_models_cache, set_ollama_models_cache
from ui.settings_sync_service import load_settings_payload, dotted_get
from utils.ollama_runtime import ensure_ollama_started

_ACTIVE_MODEL_LIST_WORKERS: set[QThread] = set()

_PENCIL_ICON = Path(__file__).resolve().parent / "assets" / "settings_pencil.svg"
_CHEVRON_ICON = Path(__file__).resolve().parent / "assets" / "settings_chevron_down.svg"

_TEXT_EDITOR_POPUP_STYLE = """
QFrame#settings_text_editor_popup {
    background: rgba(15, 16, 24, 248);
    border: 1px solid rgba(139, 92, 246, 82);
    border-radius: 12px;
}
QLabel#settings_text_editor_title {
    color: #c4b5fd;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
    font-weight: 700;
}
QLabel#settings_text_editor_description {
    color: #eef0f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
}
QLabel#restart_badge {
    color: #fbbf24;
    background: rgba(251, 191, 36, 22);
    border: 1px solid rgba(251, 191, 36, 48);
    border-radius: 5px;
    padding: 1px 5px;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 9px;
}
QPlainTextEdit#settings_text_editor_body {
    border-radius: 6px;
    border: 1px solid rgba(139, 92, 246, 62);
    background: #21193a;
    color: #f3f4f6;
    selection-background-color: rgba(139, 92, 246, 96);
    padding: 7px 9px;
    font-size: 11px;
}
QPlainTextEdit#settings_text_editor_body:focus {
    border-color: rgba(139, 92, 246, 118);
    background: #231a3f;
}
"""


class SegmentedSelector(QWidget):
    valueChanged = Signal(str)
    _segment_padding = 12.0
    _frame_padding = 6.0
    _segment_gap = 1.0
    _min_segment_width = 24.0

    def __init__(self, choices: list[str], current: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._font = _ui_font(pixel_size=10)
        self._choices = choices
        self._current = current if current in choices else (choices[0] if choices else "")
        self._hovered: str | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(self.minimumSizeHint().width())
        self.setFixedHeight(24)

    def _natural_width(self) -> int:
        if not self._choices:
            return 48

        fm = QFontMetricsF(self._font)
        width = self._frame_padding

        for choice in self._choices:
            segment_width = fm.horizontalAdvance(choice) + self._segment_padding
            width += max(self._min_segment_width, segment_width)

        width += max(0, len(self._choices) - 1) * self._segment_gap
        return int(width + 0.999)

    def sizeHint(self) -> QSize:
        return QSize(self._natural_width(), 24)

    def minimumSizeHint(self) -> QSize:
        return QSize(self._natural_width(), 24)

    def _segment_rects(self) -> dict[str, QRectF]:
        from PySide6.QtCore import QRectF
        fm = QFontMetricsF(self._font)
        y = 2.0
        height = self.height() - 4.0
        rects: dict[str, QRectF] = {}
        if not self._choices:
            return rects
        gap_total = max(0, len(self._choices) - 1) * self._segment_gap
        available = max(1.0, self.width() - self._frame_padding - gap_total)
        natural = [
            float(max(self._min_segment_width, fm.horizontalAdvance(choice) + self._segment_padding))
            for choice in self._choices
        ]
        natural_total = sum(natural)
        if natural_total <= available:
            widths = natural
            content_width = natural_total + gap_total
            x = max(3.0, (self.width() - content_width) / 2.0)
        else:
            natural_total = max(1.0, natural_total)
            widths = [max(self._min_segment_width, width * (available / natural_total)) for width in natural]
            x = 3.0
        for choice, width in zip(self._choices, widths):
            rects[choice] = QRectF(x, y, width, height)
            x += width + self._segment_gap
        return rects

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        frame_rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setPen(QPen(_to_qcolor("rgba(255,255,255,.08)"), 1))
        painter.setBrush(_to_qcolor("rgba(255,255,255,.015)"))
        painter.drawRoundedRect(frame_rect, 12, 12)

        painter.setFont(self._font)
        for choice, rect in self._segment_rects().items():
            is_active = choice == self._current
            is_hover = choice == self._hovered
            if is_active or is_hover:
                fill = "rgba(139,92,246,.18)" if is_active else "rgba(255,255,255,.05)"
                border = "rgba(139,92,246,.34)" if is_active else "rgba(255,255,255,.09)"
                active_rect = rect.adjusted(0.0, 1.0, -1.0, -1.0)
                painter.setPen(QPen(_to_qcolor(border), 1))
                painter.setBrush(_to_qcolor(fill))
                painter.drawRoundedRect(active_rect, 10, 10)

            text_rect = rect
            painter.setPen(_to_qcolor(TEXT if (is_active or is_hover) else MUTED))
            elided = painter.fontMetrics().elidedText(choice, Qt.TextElideMode.ElideRight, max(8, int(text_rect.width()) - 2))
            painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter), elided)

    def mouseMoveEvent(self, event) -> None:
        hovered = None
        pos = event.position()
        for choice, rect in self._segment_rects().items():
            if rect.contains(pos):
                hovered = choice
                break
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            for choice, rect in self._segment_rects().items():
                if rect.contains(pos):
                    if choice != self._current:
                        self._current = choice
                        self.update()
                        self.valueChanged.emit(choice)
                    break
        super().mouseReleaseEvent(event)

    def setCurrentText(self, text: str) -> None:
        if text in self._choices and text != self._current:
            self._current = text
            self.update()


class RightAlignedSelectorContainer(QWidget):
    def __init__(self, selector: SegmentedSelector, parent: QWidget | None = None):
        super().__init__(parent)
        self.selector = selector
        self.selector.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(24)

    def sizeHint(self) -> QSize:
        return QSize(self.selector.sizeHint().width(), 24)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.selector.minimumSizeHint().width(), 24)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        width = max(self.selector.minimumSizeHint().width(), min(self.selector.sizeHint().width(), self.width()))
        self.selector.setGeometry(QRect(max(0, self.width() - width), 0, width, 24))


class SettingsResizeGrip(QSizeGrip):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(_to_qcolor("rgba(196,181,253,.85)"), 1.25))
        for offset in (3, 6, 9):
            painter.drawLine(self.width() - offset, self.height() - 2, self.width() - 2, self.height() - offset)


class RestartBadge(QLabel):
    def __init__(self, parent: QWidget | None = None):
        super().__init__("restart required", parent)
        self.setObjectName("restart_badge")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        font = _ui_font(pixel_size=9, weight=QFont.Weight.DemiBold)
        font.setFamily("Cascadia Code")
        self.setFont(font)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(90, 18)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(90, 18)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class TextEditorPopup(QFrame):
    textChanged = Signal(str)

    def __init__(self, spec: SettingSpec, as_json: bool, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.spec = spec
        self.as_json = as_json
        self.setObjectName("settings_text_editor_popup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_TEXT_EDITOR_POPUP_STYLE)
        self.setMinimumSize(190, 112)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 14)
        layout.setSpacing(6)

        self.title_label = QLabel(spec.path)
        self.title_label.setObjectName("settings_text_editor_title")
        self.body_label = QLabel(_editor_description(spec))
        self.body_label.setObjectName("settings_text_editor_description")
        self.body_label.setWordWrap(True)

        badge_layout = QHBoxLayout()
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.setSpacing(5)
        self.restart_badge = RestartBadge(self)
        self.restart_badge.setVisible(bool(spec.restart_required))
        badge_layout.addWidget(self.restart_badge)
        badge_layout.addStretch()

        self.editor = QPlainTextEdit(self)
        self.editor.setObjectName("settings_text_editor_body")
        self.editor.setMinimumHeight(44 if not as_json else 112)
        self.editor.setViewportMargins(0, 0, 0, 0)
        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.editor.textChanged.connect(lambda: self.textChanged.emit(self.editor.toPlainText()))
        self.scroll_overlay = PlainTextScrollOverlay(self.editor)
        self.resize_grip = SettingsResizeGrip(self)

        layout.addWidget(self.title_label)
        layout.addWidget(self.body_label)
        layout.addLayout(badge_layout)
        layout.addWidget(self.editor, 1)

        target_width = 420 if not as_json else 560
        self.setFixedWidth(target_width)
        layout.activate()
        min_height = layout.minimumSize().height()
        self.setMinimumWidth(190)
        self.setMaximumWidth(16777215)
        self.setMinimumHeight(min_height)
        self.resize(target_width, max(146 if not as_json else 246, min_height))

    def set_text(self, text: str) -> None:
        if self.editor.toPlainText() == text:
            return
        blocked = self.editor.blockSignals(True)
        self.editor.setPlainText(text)
        self.editor.blockSignals(blocked)

    def content_min_height(self, width: int) -> int:
        layout = self.layout()
        margins = layout.contentsMargins() if layout is not None else None
        left = margins.left() if margins is not None else 12
        top = margins.top() if margins is not None else 10
        right = margins.right() if margins is not None else 12
        bottom = margins.bottom() if margins is not None else 14
        spacing = layout.spacing() if layout is not None else 6
        body_width = max(80, width - left - right)

        self.body_label.setFixedWidth(body_width)
        description_height = self.body_label.heightForWidth(body_width)
        if description_height < 0:
            description_height = self.body_label.sizeHint().height()
        self.body_label.setMinimumHeight(description_height)

        badge_height = self.restart_badge.sizeHint().height() if self.restart_badge.isVisible() else 0
        parts = [
            self.title_label.sizeHint().height(),
            description_height,
            badge_height,
            self.editor.minimumHeight(),
        ]
        visible_parts = [height for height in parts if height > 0]
        return top + bottom + sum(visible_parts) + spacing * max(0, len(visible_parts) - 1) + 6

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 12.0, 12.0)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))
        self.resize_grip.move(self.width() - self.resize_grip.width() - 2, self.height() - self.resize_grip.height() - 2)
        self.resize_grip.raise_()


class TextValueEditor(QWidget):
    valueChanged = Signal()

    def __init__(self, spec: SettingSpec, value: Any, as_json: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self.as_json = as_json
        self._text = _json_text(value) if as_json else ("" if value is None else str(value))
        self._popup: TextEditorPopup | None = None
        self.setProperty("compact_value_control", True)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.preview = QPushButton(self)
        self.preview.setObjectName("settings_value_preview")
        self.preview.setCursor(Qt.CursorShape.ArrowCursor)
        self.preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preview.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.preview.setFixedHeight(24)
        layout.addWidget(self.preview)

        self.edit_button = QToolButton(self)
        self.edit_button.setObjectName("settings_value_edit_button")
        self.edit_button.setIcon(QIcon(str(_PENCIL_ICON)))
        self.edit_button.setFixedSize(28, 24)
        self.edit_button.setIconSize(QSize(13, 13))
        self.edit_button.setToolTip("Edit")
        self.edit_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.edit_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preview.pressed.connect(self._ensure_popup)
        self.edit_button.pressed.connect(self._ensure_popup)
        self.preview.clicked.connect(self._request_editor)
        self.edit_button.clicked.connect(self._request_editor)
        layout.addWidget(self.edit_button)
        self._refresh_preview()

    def _ensure_popup(self) -> None:
        if self._popup is not None:
            return

        self._popup = TextEditorPopup(self.spec, self.as_json, self.window())
        self._popup.textChanged.connect(self.set_text)
        self._popup.set_text(self._text)
        self._popup.ensurePolished()
        self._prepare_popup_geometry(*self._initial_popup_size())


    def _request_editor(self) -> None:
        self._ensure_popup()
        QTimer.singleShot(0, self._show_editor)

    def eventFilter(self, wathched, event) -> bool:
        if wathched in (self.preview, self.edit_button):
            if event.type() in (
                QEvent.Type.Enter,
                QEvent.Type.HoverEnter,
                QEvent.Type.MouseButtonPress,
            ):
                self._ensure_popup()

            return super().eventFilter(wathched, event)

    def sizeHint(self) -> QSize:
        text_width = self.preview.fontMetrics().horizontalAdvance(self.preview.text())
        max_width = 420 if self.as_json or len(self._text) > 34 else 260
        width = min(max_width, max(92, text_width + 48))
        return QSize(width, 22)

    def minimumSizeHint(self) -> QSize:
        return QSize(86, 22)

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        if text == self._text:
            return
        self._text = text
        self._refresh_preview()
        if self._popup is not None:
            self._popup.set_text(text)
        self.valueChanged.emit()

    def set_value(self, value: Any) -> None:
        self.set_text(_json_text(value) if self.as_json else ("" if value is None else str(value)))

    def parsed_value(self) -> Any:
        if not self.as_json:
            return self._text
        text = self._text.strip()
        if not text:
            return None
        return json.loads(text)

    def _show_editor(self) -> None:
        self._ensure_popup()

        if self._popup is None:
            return

        if self._popup.isVisible():
            self._popup.raise_()
            self._popup.activateWindow()
            self._popup.editor.setFocus(Qt.FocusReason.MouseFocusReason)
            return

        self._popup.set_text(self._text)
        width, height = self._initial_popup_size()
        width, height = self._prepare_popup_geometry(width, height)

        pos = self.mapToGlobal(self.rect().bottomLeft())
        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            if pos.x() + width > available.right():
                pos.setX(max(available.left(), available.right() - width))
            if pos.y() + height > available.bottom():
                pos.setY(max(available.top(), self.mapToGlobal(self.rect().topLeft()).y() - height - 6))
        self._popup.move(pos)
        self._popup.show()
        self._popup.raise_()
        self._popup.editor.setFocus(Qt.FocusReason.MouseFocusReason)

    def _prepare_popup_geometry(self, width: int, height: int) -> tuple[int, int]:
        if self._popup is None:
            return width, height

        self._popup.setFixedWidth(width)
        content_min_height = self._popup.content_min_height(width)
        layout = self._popup.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
            min_height = max(layout.minimumSize().height(), content_min_height)
        else:
            min_height = max(self._popup.minimumSizeHint().height(), content_min_height)

        height = max(height, min_height + (12 if not self.as_json else 8))
        self._popup.setMinimumWidth(190)
        self._popup.setMaximumWidth(16777215)
        self._popup.setMinimumHeight(min_height)
        self._popup.resize(width, height)
        self._popup.setMinimumWidth(190)
        return width, height

    def _initial_popup_size(self) -> tuple[int, int]:
        text = self._text or ""
        lines = max(1, text.count("\n") + 1)
        longest = max((len(line) for line in text.splitlines()), default=len(text))
        description_len = len(_editor_description(self.spec))
        fm = self.preview.fontMetrics()
        text_width = max((fm.horizontalAdvance(line) for line in text.splitlines()), default=fm.horizontalAdvance(text))
        title_width = fm.horizontalAdvance(self.spec.path)
        description_width = fm.horizontalAdvance(_editor_description(self.spec))

        if self.as_json:
            width = max(260, text_width + 92, title_width + 44, min(description_width + 34, 220))
            if longest > 42:
                width = max(width, 420)
            if longest > 72:
                width = max(width, 560)
            width = min(640, width)
            height = 170
            if lines > 5:
                height = 230
            if lines > 12:
                height = 300
            return width, height

        width = max(190, text_width + 86, title_width + 44, min(description_width + 34, 190))
        if longest > 28 or description_len > 86:
            width = max(width, 320)
        if longest > 54:
            width = max(width, 440)
        width = min(560, width)

        height = 132 if width < 240 else 126
        if lines > 1 or description_len > 92:
            height = 154
        if lines > 3:
            height = 188
        return width, height

    def _refresh_preview(self) -> None:
        preview = _preview_text(self._text)
        self.preview.setText(preview or "empty")
        self.preview.setToolTip(self._text)
        self.setMaximumWidth(self.sizeHint().width())
        self.updateGeometry()


def _extract_model_names(raw_models: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for item in list(raw_models or []):
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("model") or item.get("id") or "").strip()
        else:
            name = str(
                getattr(item, "name", "")
                or getattr(item, "model", "")
                or getattr(item, "id", "")
                or ""
            ).strip()

        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)

    return out


class ModelListWorker(QThread):
    loaded = Signal(str, list)
    failed = Signal(str)

    def __init__(self, current: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.current = str(current or "").strip()

    def run(self) -> None:
        try:
            from ui.api_client import ApiClient
            from ui.settings_sync_service import load_settings_payload
            from ui.settings_schema import dotted_get
            from utils.ollama_runtime import ensure_ollama_started

            try:
                cfg, _ = load_settings_payload()
            except Exception:
                cfg = {}

            base_url = str(dotted_get(cfg, "llm.providers.ollama.base_url", "http://127.0.0.1:11434") or "http://127.0.0.1:11434")
            enabled = bool(dotted_get(cfg, "ui.console.auto_start_ollama", True))
            start_mode = str(dotted_get(cfg, "ui.ollama.start_mode", "serve") or "serve")
            serve_exe = str(dotted_get(cfg, "ui.ollama.serve_exe", "") or "")
            models_dir = str(dotted_get(cfg, "ui.ollama.models_dir", "") or "")

            ensure_ollama_started(
                base_url=base_url,
                wait_sec=8.0,
                enabled=enabled,
                start_mode=start_mode,
                serve_exe=serve_exe,
                models_dir=models_dir,
            )

            payload = ApiClient().list_models(timeout=8.0)
            runtime = str(payload.get("runtime_model") or self.current or "").strip()
            models = _extract_model_names(payload.get("models") or payload.get("available_models"))

            if runtime and runtime not in models:
                models.insert(0, runtime)
            if self.current and self.current not in models:
                models.insert(0, self.current)

            if models:
                set_ollama_models_cache(models)

            self.loaded.emit(runtime or self.current, models)
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelPickerPopup(QFrame):
    modelSelected = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("settings_text_editor_popup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_TEXT_EDITOR_POPUP_STYLE)
        self.setFixedWidth(240)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(8)

        self.title_label = QLabel("Модели")
        self.title_label.setObjectName("settings_text_editor_title")
        self._layout.addWidget(self.title_label)

        self.status_label = QLabel("")
        self.status_label.setObjectName("settings_text_editor_description")
        self.status_label.setVisible(False)
        self._layout.addWidget(self.status_label)

        self.list_wrap = QWidget(self)
        self.list_layout = QVBoxLayout(self.list_wrap)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(8)
        self._layout.addWidget(self.list_wrap)

    def set_status(self, text: str) -> None:
        text = str(text or "").strip()
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))

    def set_models(self, current: str, models: list[str]) -> None:
        current = str(current or "").strip()
        models = _extract_model_names(models)
        if current and current not in models:
            models.insert(0, current)
        
        self._populate(current, models)

    def _populate(self, active: str, models: list[str]) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not models:
            models = [active or "qwen3:8b"]

        for model in models:
            button = PaintedButton()
            button.set_button_font(_ui_font(pixel_size=12))
            button.set_button_padding(8, 6, 8, 6)
            button.set_button_radius(8)
            button.set_text_alignment(Qt.AlignmentFlag.AlignLeft)
            button.configure_colors(
                normal_bg="rgba(16,18,22,.28)",
                normal_border=LINE,
                normal_text=MUTED,
                hover_bg="rgba(139,92,246,.10)",
                hover_border="rgba(139,92,246,.18)",
                hover_text=TEXT,
                active_bg="rgba(139,92,246,.14)",
                active_border="rgba(139,92,246,.22)",
                active_text=TEXT,
            )
            button.set_active(model == active)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setText(f"{model}{'    ✓' if model == active else ''}")
            button.clicked.connect(lambda _checked=False, name=model: self._select_model(name))
            self.list_layout.addWidget(button)

        self.adjustSize()

    def _select_model(self, model: str) -> None:
        self.modelSelected.emit(str(model or ""))
        self.close()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12.0, 12.0)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))


class ModelNameEditor(QWidget):
    valueChanged = Signal()

    def __init__(self, value: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self._value = str(value or "").strip()
        cached_models = get_ollama_models_cache()
        self._models_cache: list[str] = cached_models or ([self._value] if self._value else [])
        if self._value and self._value not in self._models_cache:
            self._models_cache.insert(0, self._value)
        self._popup: ModelPickerPopup | None = None
        self._worker: ModelListWorker | None = None
        self._loading = False
        self.setProperty("compact_value_control", True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.preview = QPushButton(self)
        self.preview.setObjectName("settings_value_preview")
        self.preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preview.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.preview.setFixedHeight(24)
        self.preview.clicked.connect(self._request_popup)
        layout.addWidget(self.preview)

        self.arrow_button = QToolButton(self)
        self.arrow_button.setObjectName("settings_value_edit_button")
        self.arrow_button.setIcon(QIcon(str(_CHEVRON_ICON)))
        self.arrow_button.setFixedSize(28, 24)
        self.arrow_button.setIconSize(QSize(13, 13))
        self.arrow_button.setToolTip("Выбрать модель")
        self.arrow_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.arrow_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.arrow_button.clicked.connect(self._request_popup)
        layout.addWidget(self.arrow_button)

        self._refresh_preview()
        self.destroyed.connect(lambda *_: self.dispose())

    def value(self) -> str:
        return self._value

    def set_value(self, value: Any) -> None:
        value = str(value or "").strip()
        if value == self._value:
            return
        self._value = value
        if value and value not in self._models_cache:
            self._models_cache.insert(0, value)
        self._refresh_preview()
        if self._popup is not None:
            self._popup.set_models(self._value, self._models_cache)
        self.valueChanged.emit()

    def sizeHint(self) -> QSize:
        text_width = self.preview.fontMetrics().horizontalAdvance(self.preview.text())
        width = min(420, max(120, text_width + 52))
        return QSize(width, 22)

    def minimumSizeHint(self) -> QSize:
        return QSize(110, 22)

    def refresh_models_async(self) -> None:
        if self._loading:
            return
        self._loading = True
        if self._popup is not None:
            self._popup.set_status("обновляю список...")

        worker = ModelListWorker(self._value, None)
        self._worker = worker
        _ACTIVE_MODEL_LIST_WORKERS.add(worker)

        worker.loaded.connect(self._on_models_loaded)
        worker.failed.connect(self._on_models_failed)
        worker.finished.connect(lambda w=worker: self._finish_model_worker(w))
        worker.start()

    def _ensure_popup(self) -> None:
        if self._popup is not None:
            return
        self._popup = ModelPickerPopup(self.window())
        self._popup.modelSelected.connect(self.set_value)
        self._popup.set_models(self._value, self._models_cache)

    def _request_popup(self) -> None:
        self._ensure_popup()
        self._show_popup()

        if not self._loading:
            self.refresh_models_async()

    def _show_popup(self) -> None:
        self._ensure_popup()
        if self._popup is None:
            return

        if self._popup.isVisible():
            self._popup.raise_()
            self._popup.activateWindow()
            return

        self._popup.set_models(self._value, self._models_cache)
        self._popup.adjustSize()

        width = max(240, self.width())
        height = min(360, max(96, self._popup.sizeHint().height()))
        self._popup.setFixedWidth(width)
        self._popup.resize(width, height)

        pos = self.mapToGlobal(self.rect().bottomLeft())
        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            if pos.x() + width > available.right():
                pos.setX(max(available.left(), available.right() - width))
            if pos.y() + height > available.bottom():
                pos.setY(max(available.top(), self.mapToGlobal(self.rect().topLeft()).y() - height - 6))

        self._popup.move(pos)
        self._popup.show()
        self._popup.raise_()

    def _on_models_loaded(self, runtime: str, models: list[str]) -> None:
        runtime = str(runtime or "").strip()
        models = _extract_model_names(models)

        if runtime and runtime not in models:
            models.insert(0, runtime)
        if self._value and self._value not in models:
            models.insert(0, self._value)

        self._models_cache = models
        if models:
            set_ollama_models_cache(models)

        self._refresh_preview()
        if self._popup is not None:
            self._popup.set_status("")
            self._popup.set_models(self._value, self._models_cache)

    def _on_models_failed(self, error_text: str) -> None:
        if self._value and self._value not in self._models_cache:
            self._models_cache.insert(0, self._value)
        if self._popup is not None:
            self._popup.set_status("API недоступно, показан кеш")
            self._popup.set_models(self._value, self._models_cache)

    def _finish_model_worker(self, worker: ModelListWorker) -> None:
        _ACTIVE_MODEL_LIST_WORKERS.discard(worker)

        if self._worker is worker:
            self._loading = False
            self._worker = None

        worker.deleteLater()

    def dispose(self) -> None:
        popup = self._popup
        self._popup = None
        if popup is not None:
            popup.close()
            popup.deleteLater()

        worker = self._worker
        self._worker = None
        self._loading = False

        if worker is not None:
            try:
                worker.loaded.disconnect(self._on_models_loaded)
            except Exception:
                pass
            try:
                worker.failed.disconnect(self._on_models_failed)
            except Exception:
                pass

    def _refresh_preview(self) -> None:
        text = self._value or "empty"
        self.preview.setText(text)
        self.preview.setToolTip(self._value)
        self.setMaximumWidth(self.sizeHint().width())
        self.updateGeometry()


class NumericValueEditor(QWidget):
    valueChanged = Signal()

    def __init__(self, value: Any, *, is_float: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.is_float = is_float
        self._minimum = -1_000_000_000.0 if is_float else -2_147_483_648
        self._maximum = 1_000_000_000.0 if is_float else 2_147_483_647
        self.setProperty("compact_value_control", True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.input = QLineEdit(self)
        self.input.setObjectName("settings_numeric_input")
        self.input.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.input.editingFinished.connect(self._commit_text)
        self.input.returnPressed.connect(self._commit_text)
        self.input.setMinimumWidth(20)

        buttons = QWidget(self)
        buttons.setObjectName("settings_numeric_stepper")
        buttons.setFixedSize(24, 24)
        button_layout = QVBoxLayout(buttons)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)

        self.plus_button = QToolButton(self)
        self.plus_button.setObjectName("settings_numeric_step_button_up")
        self.plus_button.setText("▲")
        self.plus_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plus_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.plus_button.clicked.connect(lambda: self._step(1))

        self.minus_button = QToolButton(self)
        self.minus_button.setObjectName("settings_numeric_step_button_down")
        self.minus_button.setText("▼")
        self.minus_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.minus_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.minus_button.clicked.connect(lambda: self._step(-1))

        button_layout.addWidget(self.plus_button)
        button_layout.addWidget(self.minus_button)
        layout.addWidget(self.input)
        layout.addWidget(buttons)
        self.set_value(value)

    def value(self) -> int | float:
        text = self.input.text().strip()
        return float(text or 0.0) if self.is_float else int(float(text or 0))

    def set_value(self, value: Any) -> None:
        number = float(value or 0.0) if self.is_float else int(value or 0)
        number = max(self._minimum, min(self._maximum, number))
        text = self._format(number)
        if self.input.text() != text:
            self.input.setText(text)
        self.setMaximumWidth(self.sizeHint().width())
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        width = self.input.fontMetrics().horizontalAdvance(self.input.text() or "0") + 46
        return QSize(min(190, max(72, width)), 24)

    def minimumSizeHint(self) -> QSize:
        return QSize(72, 24)

    def _step(self, direction: int) -> None:
        step = 0.1 if self.is_float else 1
        self.set_value(self.value() + direction * step)
        self.valueChanged.emit()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self._step(1 if delta > 0 else -1)
        event.accept()

    def _commit_text(self) -> None:
        try:
            value = self.value()
        except ValueError:
            value = 0.0 if self.is_float else 0
        self.set_value(value)
        self.valueChanged.emit()

    def _format(self, value: int | float) -> str:
        if not self.is_float:
            return str(int(value))
        text = f"{float(value):.3f}".rstrip("0").rstrip(".")
        return text or "0"


class SettingEditor(QWidget):
    valueChanged = Signal(object)

    def __init__(self, spec: SettingSpec, value: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self._widget = self._build_widget(value)
        try:
            self._initial = self.value()
        except Exception:
            self._initial = value

    def control(self) -> QWidget:
        return self._widget

    def initial_value(self) -> Any:
        return self._initial

    def value(self) -> Any:
        widget = self._widget
        if isinstance(widget, QWidget) and widget.property("is_switch_container"):
            switch = widget.findChild(ToggleSwitch)
            return bool(switch.isChecked())
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, NumericValueEditor):
            return widget.value()
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QWidget) and widget.property("is_segmented_container"):
            selector = widget.findChild(SegmentedSelector)
            return str(selector._current)
        if isinstance(widget, SegmentedSelector):
            return str(widget._current)
        if isinstance(widget, QComboBox):
            return str(widget.currentText())
        if isinstance(widget, ModelNameEditor):
            return widget.value()
        if isinstance(widget, TextValueEditor):
            return widget.parsed_value()
        if isinstance(widget, QPlainTextEdit):
            text = widget.toPlainText().strip()
            if not text:
                return None
            return json.loads(text)
        if isinstance(widget, QLineEdit):
            return str(widget.text())
        return None

    def set_value(self, value: Any) -> None:
        widget = self._widget
        if isinstance(widget, QWidget) and widget.property("is_switch_container"):
            switch = widget.findChild(ToggleSwitch)
            switch.setChecked(bool(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, NumericValueEditor):
            widget.set_value(value)
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value or 0))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value or 0.0))
        elif isinstance(widget, QWidget) and widget.property("is_segmented_container"):
            selector = widget.findChild(SegmentedSelector)
            selector.setCurrentText(str(value or ""))
        elif isinstance(widget, SegmentedSelector):
            widget.setCurrentText(str(value or ""))
        elif isinstance(widget, QComboBox):
            text = str(value or "")
            idx = widget.findText(text)
            if idx < 0 and text:
                widget.addItem(text)
                idx = widget.findText(text)
            widget.setCurrentIndex(max(0, idx))
        elif isinstance(widget, ModelNameEditor):
            widget.set_value(value)
        elif isinstance(widget, TextValueEditor):
            widget.set_value(value)
        elif isinstance(widget, QPlainTextEdit):
            widget.setPlainText(_json_text(value))
        elif isinstance(widget, QLineEdit):
            widget.setText("" if value is None else str(value))

    def validate_value(self) -> tuple[bool, str]:
        try:
            self.value()
        except Exception as exc:
            return False, str(exc)
        return True, ""

    def _emit_changed(self) -> None:
        try:
            self.valueChanged.emit(self.value())
        except Exception:
            self.valueChanged.emit(None)

    def _build_widget(self, value: Any) -> QWidget:
        if self.spec.path == "llm.model_name":
            widget = ModelNameEditor(value)
            widget.valueChanged.connect(self._emit_changed)
            return widget

        kind = self.spec.kind
        if kind == "bool":
            container = QWidget()
            container.setProperty("is_switch_container", True)
            container.setFixedHeight(28)
            container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            switch = ToggleSwitch(bool(value))
            switch.toggled.connect(lambda _state: self._emit_changed())
            layout.addStretch()
            layout.addWidget(switch, 0, Qt.AlignmentFlag.AlignVCenter)
            return container
        if kind == "int":
            widget = NumericValueEditor(value, is_float=False)
            widget.valueChanged.connect(self._emit_changed)
            return widget
        if kind == "float":
            widget = NumericValueEditor(value, is_float=True)
            widget.valueChanged.connect(self._emit_changed)
            return widget
        if kind == "select":
            choices = [str(option) for option in self.spec.options]
            text = str(value or "")
            if text and text not in choices:
                choices.append(text)
            
            total_len = sum(len(c) for c in choices)
            if len(choices) <= 4 and total_len <= 25:
                selector = SegmentedSelector(choices, text)
                container = RightAlignedSelectorContainer(selector)
                container.setProperty("is_segmented_container", True)
                container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                selector.valueChanged.connect(lambda _text: self._emit_changed())
                return container
            
            widget = QComboBox()
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            for option in choices:
                widget.addItem(option)
            idx = widget.findText(text)
            widget.setCurrentIndex(max(0, idx))
            widget.currentTextChanged.connect(lambda _text: self._emit_changed())
            return widget
        if kind == "json":
            widget = TextValueEditor(self.spec, value, as_json=True)
            widget.valueChanged.connect(self._emit_changed)
            return widget
        widget = TextValueEditor(self.spec, value, as_json=False)
        widget.valueChanged.connect(self._emit_changed)
        return widget


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _preview_text(text: str) -> str:
    compact = " ".join(str(text or "").replace("\r", "\n").split())
    if len(compact) <= 54:
        return compact
    return f"{compact[:54]}..."


def _editor_description(spec: SettingSpec) -> str:
    text = str(spec.description or "").strip()
    if text:
        return text
    return f"Edit {spec.path}. The value is applied when settings are saved."
