from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Signal, Qt, QSize, QRectF
from PySide6.QtGui import QCursor, QPainter, QPen
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QSizePolicy, QSpinBox, QWidget, QHBoxLayout, QLabel

from ui.settings_schema import SettingSpec
from ui.chat_shell import ToggleSwitch, _ui_font, _to_qcolor, TEXT, MUTED


class SegmentedSelector(QWidget):
    valueChanged = Signal(str)

    def __init__(self, choices: list[str], current: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._font = _ui_font(pixel_size=10)
        self._choices = choices
        self._current = current if current in choices else (choices[0] if choices else "")
        self._hovered: str | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)

    def sizeHint(self) -> QSize:
        from PySide6.QtGui import QFontMetricsF
        fm = QFontMetricsF(self._font)
        width = 0
        for choice in self._choices:
            width += int(fm.horizontalAdvance(choice) + 10 * 2)
        width += 4 + max(0, len(self._choices) - 1) * 2
        return QSize(width, 24)

    def _segment_rects(self) -> dict[str, QRectF]:
        from PySide6.QtGui import QFontMetricsF
        from PySide6.QtCore import QRectF
        fm = QFontMetricsF(self._font)
        x = 3.0
        y = 2.0
        height = self.height() - 4.0
        rects: dict[str, QRectF] = {}
        for choice in self._choices:
            width = fm.horizontalAdvance(choice) + 20
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
            painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter), choice)

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


class SettingEditor(QWidget):
    valueChanged = Signal(object)

    def __init__(self, spec: SettingSpec, value: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self._initial = value
        self._widget = self._build_widget(value)

    def control(self) -> QWidget:
        return self._widget

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
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            switch = ToggleSwitch(bool(value))
            switch.toggled.connect(lambda _state: self._emit_changed())
            layout.addStretch()
            layout.addWidget(switch)
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
                container = QWidget()
                container.setProperty("is_segmented_container", True)
                layout = QHBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addStretch()
                selector = SegmentedSelector(choices, text)
                selector.valueChanged.connect(lambda _text: self._emit_changed())
                layout.addWidget(selector)
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
            widget = QPlainTextEdit()
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            widget.setMinimumHeight(72)
            widget.setPlainText(_json_text(value))
            widget.textChanged.connect(self._emit_changed)
            return widget
        widget = QLineEdit()
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        widget.setPlaceholderText(self.spec.placeholder or self.spec.example)
        widget.setText("" if value is None else str(value))
        widget.textChanged.connect(lambda _text: self._emit_changed())
        return widget


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(value)
