"""Reusable custom widgets used by MMis desktop UI."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QCheckBox, QFrame, QWidget
from shiboken6 import isValid


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


class _MiniSparkline(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._title = str(title)
        self._values: list[float] = []
        self._max_points = 80
        self._line_color = QColor(110, 175, 255, 230)
        self._fill_color = QColor(110, 175, 255, 40)
        self._grid_color = QColor(255, 255, 255, 28)
        self._title_color = QColor(220, 220, 220, 190)
        self.setMinimumHeight(96)
        self.setMaximumHeight(122)

    def clear_values(self) -> None:
        self._values.clear()
        self.update()

    def push_value(self, value: float) -> None:
        v = max(0.0, float(value))
        self._values.append(v)
        if len(self._values) > self._max_points:
            self._values = self._values[-self._max_points :]
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        base_font = p.font()
        base_size = base_font.pointSizeF() if base_font.pointSizeF() > 0 else 10.0
        label_font = QFont(base_font)
        label_font.setPointSizeF(max(5.0, base_size * 0.5))
        p.setFont(label_font)

        plot = self.rect().adjusted(8, 20, -36, -20)
        if plot.width() <= 8 or plot.height() <= 8:
            return

        p.setPen(self._title_color)
        p.drawText(6, 13, self._title)

        p.setPen(self._grid_color)
        p.drawLine(plot.left(), plot.top(), plot.right(), plot.top())
        p.drawLine(plot.left(), plot.top() + plot.height() // 2, plot.right(), plot.top() + plot.height() // 2)
        p.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())

        if self._values:
            vmax = max(max(self._values), 1e-6)
        else:
            vmax = 1.0
        vmid = vmax / 2.0

        p.setPen(self._title_color)
        rx = plot.right() + 5
        y_top = plot.top()
        y_mid = plot.top() + plot.height() // 2
        y_bot = plot.bottom()
        for yy in (y_top, y_mid, y_bot):
            p.setPen(self._grid_color)
            p.drawLine(plot.right(), yy, plot.right() + 4, yy)
        p.setPen(self._title_color)
        p.drawText(rx, y_top + 4, f"{vmax:.1f}")
        p.drawText(rx, y_mid + 4, f"{vmid:.1f}")
        p.drawText(rx, y_bot + 4, "0.0")

        x_left = plot.left()
        x_mid = plot.left() + plot.width() // 2
        x_right = plot.right()
        for xx in (x_left, x_mid, x_right):
            p.setPen(self._grid_color)
            p.drawLine(xx, plot.bottom(), xx, plot.bottom() + 4)
        p.setPen(self._title_color)
        y_axis_text = self.height() - 5
        p.drawText(x_left - 2, y_axis_text, "0")
        p.drawText(x_mid - 7, y_axis_text, "50")
        p.drawText(x_right - 14, y_axis_text, "100")

        if len(self._values) < 2:
            return

        n = len(self._values) - 1
        points: list[tuple[int, int]] = []
        for i, v in enumerate(self._values):
            x = plot.left() + int((i / n) * plot.width())
            y = plot.bottom() - int((v / vmax) * plot.height())
            points.append((x, y))

        p.setPen(Qt.NoPen)
        p.setBrush(self._fill_color)
        poly = [*points, (plot.right(), plot.bottom()), (plot.left(), plot.bottom())]
        p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in poly]))

        p.setPen(QPen(self._line_color, 2))
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            p.drawLine(x1, y1, x2, y2)
