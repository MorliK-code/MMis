from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal, Qt, QSize, QRect, QRectF
from PySide6.QtGui import QCursor, QFontMetricsF, QIcon, QPainter, QPainterPath, QPen, QRegion
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
from ui.chat_shell import ToggleSwitch, _ui_font, _to_qcolor, TEXT, MUTED, PlainTextScrollOverlay

_PENCIL_ICON = Path(__file__).resolve().parent / "assets" / "settings_pencil.svg"

_TEXT_EDITOR_POPUP_STYLE = """
QFrame#settings_text_editor_popup {
    background: rgba(15, 16, 24, 248);
    border: 1px solid rgba(139, 92, 246, 82);
    border-radius: 12px;
}
QLabel#settings_text_editor_title {
    color: #c4b5fd;
    font-size: 11px;
    font-weight: 700;
}
QLabel#settings_text_editor_description {
    color: #eef0f6;
    font-size: 11px;
}
QLabel#restart_badge {
    color: #fbbf24;
    background: rgba(251, 191, 36, 22);
    border: 1px solid rgba(251, 191, 36, 48);
    border-radius: 5px;
    padding: 1px 5px;
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
    _segment_padding = 20.0

    def __init__(self, choices: list[str], current: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._font = _ui_font(pixel_size=10)
        self._choices = choices
        self._current = current if current in choices else (choices[0] if choices else "")
        self._hovered: str | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(84)
        self.setFixedHeight(24)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        width = 0
        for choice in self._choices:
            width += int(fm.horizontalAdvance(choice) + self._segment_padding + 0.999)
        width += 6 + max(0, len(self._choices) - 1)
        return QSize(max(width, 84), 24)

    def minimumSizeHint(self) -> QSize:
        return QSize(84, 24)

    def _segment_rects(self) -> dict[str, QRectF]:
        from PySide6.QtCore import QRectF
        fm = QFontMetricsF(self._font)
        y = 2.0
        height = self.height() - 4.0
        rects: dict[str, QRectF] = {}
        if not self._choices:
            return rects
        available = max(1.0, self.width() - 6.0 - max(0, len(self._choices) - 1))
        natural = [float(fm.horizontalAdvance(choice) + self._segment_padding) for choice in self._choices]
        natural_total = sum(natural)
        if natural_total <= available:
            widths = natural
            content_width = natural_total + max(0, len(self._choices) - 1)
            x = max(3.0, (self.width() - content_width) / 2.0)
        else:
            natural_total = max(1.0, natural_total)
            widths = [max(22.0, width * (available / natural_total)) for width in natural]
            x = 3.0
        for choice, width in zip(self._choices, widths):
            rects[choice] = QRectF(x, y, width, height)
            x += width + 1.0
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
        self.setFixedSize(16, 16)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(_to_qcolor("rgba(196,181,253,.85)"), 1.25))
        for offset in (4, 8, 12):
            painter.drawLine(self.width() - offset, self.height() - 2, self.width() - 2, self.height() - offset)


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
        self.restart_badge = QLabel("restart required")
        self.restart_badge.setObjectName("restart_badge")
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 12.0, 12.0)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))
        self.resize_grip.move(self.width() - self.resize_grip.width() - 4, self.height() - self.resize_grip.height() - 4)
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
        self.edit_button.clicked.connect(self._show_editor)
        layout.addWidget(self.edit_button)
        self._refresh_preview()

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
        if self._popup is None:
            self._popup = TextEditorPopup(self.spec, self.as_json, self.window())
            self._popup.textChanged.connect(self.set_text)
        self._popup.set_text(self._text)
        width, height = self._initial_popup_size()
        editor = self._popup.editor
        old_policy = editor.sizePolicy()
        old_min = editor.minimumHeight()
        editor.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        editor.setFixedHeight(old_min)
        
        layout = self._popup.layout()
        min_height = layout.heightForWidth(width) if layout.hasHeightForWidth() else layout.minimumSize().height()
        
        editor.setSizePolicy(old_policy)
        editor.setMinimumHeight(old_min)
        editor.setMaximumHeight(16777215)
        
        self._popup.setMinimumWidth(190)
        self._popup.setMaximumWidth(16777215)
        self._popup.setMinimumHeight(min_height)
        
        self._popup.resize(width, max(height, min_height))

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
            widget = QSpinBox()
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            widget.setRange(-2_147_483_648, 2_147_483_647)
            widget.setValue(int(value or 0))
            widget.valueChanged.connect(lambda _value: self._emit_changed())
            return widget
        if kind == "float":
            widget = QDoubleSpinBox()
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            widget.setRange(-1_000_000_000.0, 1_000_000_000.0)
            widget.setDecimals(3)
            widget.setValue(float(value or 0.0))
            widget.valueChanged.connect(lambda _value: self._emit_changed())
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
