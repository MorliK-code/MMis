from __future__ import annotations

import os
import re
import sys
import time
import warnings
from math import exp
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QLinearGradient, QMouseEvent, QPainter, QFontMetricsF, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The pynvml package is deprecated\..*",
            category=FutureWarning,
        )
        import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None

try:
    from ui.api_client import ApiClient, ApiClientError
    from ui.workers import ReplyWorker, ReplyResult
except Exception:  # pragma: no cover
    ApiClient = None
    ApiClientError = RuntimeError
    ReplyWorker = None
    ReplyResult = None


BG = "#0a0b0d"
SURFACE = "rgba(16,18,22,0.82)"
SURFACE_SOFT = "rgba(18,21,26,0.78)"
LINE = "rgba(255,255,255,0.06)"
TEXT = "#f3f4f6"
MUTED = "#8f96a3"
MUTED_2 = "#646b76"
USER_BG = "rgba(23,18,42,0.92)"
ASSISTANT_BG = "rgba(18,22,27,0.92)"
ACCENT = "#8b5cf6"
OK_BG = "rgba(34,197,94,0.10)"
OK_LINE = "rgba(34,197,94,0.18)"
OK_TEXT = "#86efac"
THINKING = "#b5a9d4"
THINKING_TIME_TEXT = "#A294D2"
THINKING_TIME_BG = "#1A1D25"
THINKING_TIME_BORDER = "#313141"
TOPBAR_BG = "#0d1013"
STATUS_BG = OK_BG
STATUS_LINE = OK_LINE
RESOURCE_BG = "rgba(255,255,255,.03)"
RESOURCE_LINE = LINE


def _pct_color() -> str:
    return "rgba(139,92,246,0.10)"


def _pct_border() -> str:
    return "rgba(139,92,246,0.18)"


_RGBA_RE = re.compile(r"^(rgba?|RGBA?)\((.+)\)$")


def _to_qcolor(value) -> QColor:
    if isinstance(value, QColor):
        return QColor(value)
    if value is None:
        return QColor(0, 0, 0, 0)
    text = str(value).strip()
    if not text:
        return QColor(0, 0, 0, 0)
    if text.lower() == "transparent":
        return QColor(0, 0, 0, 0)

    match = _RGBA_RE.match(text)
    if match:
        mode = match.group(1).lower()
        parts = [part.strip() for part in match.group(2).split(",")]
        if (mode == "rgb" and len(parts) == 3) or (mode == "rgba" and len(parts) == 4):
            try:
                red = max(0, min(255, int(float(parts[0]))))
                green = max(0, min(255, int(float(parts[1]))))
                blue = max(0, min(255, int(float(parts[2]))))
                alpha = 255
                if mode == "rgba":
                    alpha_value = float(parts[3])
                    if alpha_value <= 1.0:
                        alpha = max(0, min(255, int(round(alpha_value * 255.0))))
                    else:
                        alpha = max(0, min(255, int(round(alpha_value))))
                return QColor(red, green, blue, alpha)
            except Exception:
                pass

    color = QColor(text)
    if color.isValid():
        return color
    return QColor(0, 0, 0, 0)


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


def _ui_font(*, pixel_size: int | None = None, weight: int = QFont.Weight.Medium) -> QFont:
    font = QFont("Segoe UI")
    if pixel_size is not None:
        font.setPixelSize(int(pixel_size))
    font.setWeight(weight)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    font.setHintingPreference(QFont.HintingPreference.PreferVerticalHinting)
    return font


def _topbar_font(*, pixel_size: int | None = None, weight: int = QFont.Weight.Medium) -> QFont:
    font = QFont("Segoe UI")
    if pixel_size is not None:
        font.setPixelSize(int(pixel_size))
    font.setWeight(weight)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    return font


class CrispLabel(QLabel):
    def __init__(self, text: str = "", color: str = TEXT, parent: QWidget | None = None):
        super().__init__(text, parent)
        self._text_color = _to_qcolor(color)
        self._background = _to_qcolor("transparent")
        self._border_color = _to_qcolor("transparent")
        self._border_style = Qt.PenStyle.SolidLine
        self._radius = 0
        self._padding = (0, 0, 0, 0)
        self._preferred_text_width: int | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def set_text_color(self, color: str) -> None:
        self._text_color = _to_qcolor(color)
        self.update()

    def set_box_style(
        self,
        *,
        background: str = "transparent",
        border: str = "transparent",
        radius: int = 0,
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        dashed: bool = False,
    ) -> None:
        self._background = _to_qcolor(background)
        self._border_color = _to_qcolor(border)
        self._border_style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        self._radius = max(0, int(radius))
        self._padding = tuple(int(value) for value in padding)
        self.updateGeometry()
        self.update()

    def set_preferred_text_width(self, width: int | None) -> None:
        self._preferred_text_width = None if width is None else max(0, int(width))
        self.updateGeometry()
        self.update()

    def setText(self, text: str) -> None:  # noqa: N802
        super().setText(text)
        self.updateGeometry()
        self.update()

    def setWordWrap(self, on: bool) -> None:  # noqa: N802
        super().setWordWrap(on)
        self.updateGeometry()
        self.update()

    def hasHeightForWidth(self) -> bool:
        return self.wordWrap()

    def heightForWidth(self, width: int) -> int:
        return self._measure_size(width).height()

    def sizeHint(self) -> QSize:
        if self.wordWrap():
            width = self._preferred_text_width or 320
            return self._measure_size(width)
        return self._measure_size(None)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _text_flags(self) -> int:
        flags = int(self.alignment() or (Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter))
        if self.wordWrap():
            flags |= int(Qt.TextFlag.TextWordWrap)
        return flags

    def _content_rect(self) -> QRect:
        left, top, right, bottom = self._padding
        return self.rect().adjusted(left, top, -right, -bottom)

    def _measure_size(self, width: int | None) -> QSize:
        fm = QFontMetricsF(self.font())
        text = self.text() or " "
        left, top, right, bottom = self._padding
        if self.wordWrap():
            inner_width = max(24, (width or self._preferred_text_width or 320) - left - right)
            rect = fm.boundingRect(QRect(0, 0, inner_width, 10_000), self._text_flags(), text)
            return QSize(int(rect.width()) + left + right + 2, int(rect.height()) + top + bottom + 2)
        width_px = int(fm.horizontalAdvance(text)) + left + right + 2
        height_px = int(fm.height()) + top + bottom + 2
        return QSize(width_px, height_px)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        text = self.text()
        if not text:
            return

        frame_rect = self.rect().adjusted(0, 0, -1, -1)
        if self._background.alpha() > 0 or self._border_color.alpha() > 0:
            painter.setBrush(self._background)
            if self._border_color.alpha() > 0:
                pen = QPen(self._border_color)
                pen.setStyle(self._border_style)
                painter.setPen(pen)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(frame_rect, self._radius, self._radius)

        painter.setPen(self._text_color)
        painter.setFont(self.font())
        painter.drawText(self._content_rect(), self._text_flags(), text)


class PaintedButton(QPushButton):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self._font = _ui_font(pixel_size=13)
        self._padding = (10, 0, 10, 0)
        self._radius = 8
        self._draw_offset_y = 0
        self._hover_draw_offset_y = 0
        self._text_offset_y = 0
        self._hover_text_offset_y = 0
        self._text_alignment = Qt.AlignmentFlag.AlignCenter
        self._active = False
        self._normal_bg = _to_qcolor("transparent")
        self._normal_border = _to_qcolor("transparent")
        self._normal_text = _to_qcolor(TEXT)
        self._hover_bg = _to_qcolor("transparent")
        self._hover_border = _to_qcolor("transparent")
        self._hover_text = _to_qcolor(TEXT)
        self._active_bg = _to_qcolor("transparent")
        self._active_border = _to_qcolor("transparent")
        self._active_text = _to_qcolor(TEXT)
        self._disabled_bg = _to_qcolor("transparent")
        self._disabled_border = _to_qcolor("transparent")
        self._disabled_text = _to_qcolor(MUTED_2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlat(True)

    def set_button_font(self, font: QFont) -> None:
        self._font = QFont(font)
        self.updateGeometry()
        self.update()

    def set_button_padding(self, left: int, top: int, right: int, bottom: int) -> None:
        self._padding = (int(left), int(top), int(right), int(bottom))
        self.updateGeometry()
        self.update()

    def set_button_radius(self, radius: int) -> None:
        self._radius = max(0, int(radius))
        self.update()

    def set_text_alignment(self, alignment: Qt.AlignmentFlag) -> None:
        self._text_alignment = alignment
        self.update()

    def set_draw_offset_y(self, offset: int) -> None:
        self._draw_offset_y = int(offset)
        self.update()

    def set_hover_draw_offset_y(self, offset: int) -> None:
        self._hover_draw_offset_y = int(offset)
        self.update()

    def set_text_offset_y(self, offset: int) -> None:
        self._text_offset_y = int(offset)
        self.update()

    def set_hover_text_offset_y(self, offset: int) -> None:
        self._hover_text_offset_y = int(offset)
        self.update()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self.update()

    def configure_colors(
        self,
        *,
        normal_bg: str = "transparent",
        normal_border: str = "transparent",
        normal_text: str = TEXT,
        hover_bg: str | None = None,
        hover_border: str | None = None,
        hover_text: str | None = None,
        active_bg: str | None = None,
        active_border: str | None = None,
        active_text: str | None = None,
        disabled_bg: str = "transparent",
        disabled_border: str = "transparent",
        disabled_text: str = MUTED_2,
    ) -> None:
        self._normal_bg = _to_qcolor(normal_bg)
        self._normal_border = _to_qcolor(normal_border)
        self._normal_text = _to_qcolor(normal_text)
        self._hover_bg = _to_qcolor(hover_bg if hover_bg is not None else normal_bg)
        self._hover_border = _to_qcolor(hover_border if hover_border is not None else normal_border)
        self._hover_text = _to_qcolor(hover_text if hover_text is not None else normal_text)
        self._active_bg = _to_qcolor(active_bg if active_bg is not None else normal_bg)
        self._active_border = _to_qcolor(active_border if active_border is not None else normal_border)
        self._active_text = _to_qcolor(active_text if active_text is not None else normal_text)
        self._disabled_bg = _to_qcolor(disabled_bg)
        self._disabled_border = _to_qcolor(disabled_border)
        self._disabled_text = _to_qcolor(disabled_text)
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        left, top, right, bottom = self._padding
        width = int(fm.horizontalAdvance(self.text() or " ") + left + right + 2)
        height = int(fm.height() + top + bottom + 2)
        base = super().sizeHint()
        return QSize(max(width, base.width()), max(height, base.height()))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _resolved_colors(self) -> tuple[QColor, QColor, QColor]:
        if not self.isEnabled():
            return self._disabled_bg, self._disabled_border, self._disabled_text
        if self.isDown() or self.isChecked() or self._active:
            if self.underMouse():
                return self._hover_bg, self._hover_border, self._hover_text
            return self._active_bg, self._active_border, self._active_text
        if self.underMouse():
            return self._hover_bg, self._hover_border, self._hover_text
        return self._normal_bg, self._normal_border, self._normal_text

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        bg, border, text_color = self._resolved_colors()
        draw_offset_y = self._draw_offset_y
        text_offset_y = self._text_offset_y
        if self.underMouse() and not (self.isDown() or self.isChecked() or self._active):
            draw_offset_y = self._hover_draw_offset_y
            text_offset_y = self._hover_text_offset_y
        base_rect = self.rect().adjusted(0, 0, -1, -1)
        frame_rect = base_rect.adjusted(0, draw_offset_y, 0, 0)
        if bg.alpha() > 0 or border.alpha() > 0:
            painter.setBrush(bg)
            painter.setPen(border if border.alpha() > 0 else Qt.PenStyle.NoPen)
            painter.drawRoundedRect(frame_rect, self._radius, self._radius)

        left, top, right, bottom = self._padding
        text_rect = base_rect.adjusted(left, top + text_offset_y, -right, -bottom + text_offset_y)
        painter.setPen(text_color)
        painter.setFont(self._font)
        painter.drawText(text_rect, int(self._text_alignment | Qt.AlignmentFlag.AlignVCenter), self.text())


class BrandBadge(QWidget):
    def __init__(self, text: str = "MMis", parent: QWidget | None = None):
        super().__init__(parent)
        self._text = str(text or "")
        self._font = _topbar_font(pixel_size=13, weight=QFont.Weight.DemiBold)
        self.setFixedHeight(20)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        width = int(10 + 8 + fm.horizontalAdvance(self._text) + 2)
        return QSize(width, 20)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        center_y = self.height() / 2.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_to_qcolor(ACCENT))
        painter.drawEllipse(QRect(int(0), int(center_y - 3), 6, 6))

        painter.setFont(self._font)
        painter.setPen(_to_qcolor("#ffffff"))
        painter.drawText(QRect(12, 0, self.width() - 12, self.height()), int(Qt.AlignmentFlag.AlignVCenter), self._text)


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, margin: int = 0, hspacing: int = 6, vspacing: int = 6):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._hspacing = hspacing
        self._vspacing = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        width = 0
        height = 0
        visible_count = 0
        for item in self._items:
            hint = item.sizeHint()
            if hint.width() <= 0 or hint.height() <= 0:
                continue
            width += hint.width()
            height = max(height, hint.height())
            visible_count += 1
        if visible_count > 1:
            width += self._hspacing * (visible_count - 1)
        size = QSize(width, height)
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        x = rect.x()
        y = rect.y()
        line_height = 0
        max_x = rect.x() + rect.width()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._hspacing
            if line_height > 0 and next_x - self._hspacing > max_x and rect.width() > 0:
                x = rect.x()
                y = y + line_height + self._vspacing
                next_x = x + hint.width() + self._hspacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class FlowWrap(QWidget):
    def hasHeightForWidth(self) -> bool:
        layout = self.layout()
        return bool(layout and layout.hasHeightForWidth())

    def heightForWidth(self, width: int) -> int:
        layout = self.layout()
        if layout and layout.hasHeightForWidth():
            return max(0, int(layout.heightForWidth(max(1, int(width)))))
        return super().heightForWidth(width)

    def sizeHint(self) -> QSize:
        layout = self.layout()
        if layout is None:
            return super().sizeHint()
        base = layout.minimumSize()
        width = max(1, self.width() or base.width())
        height = self.heightForWidth(width)
        return QSize(base.width(), max(base.height(), height))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def refresh_height(self) -> None:
        layout = self.layout()
        if layout is None:
            return
        width = max(1, self.width() or layout.minimumSize().width())
        height = max(layout.minimumSize().height(), self.heightForWidth(width))
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.updateGeometry()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.refresh_height()


class HoverButton(PaintedButton):
    def __init__(self, text: str = "", accent: bool = False, parent: QWidget | None = None):
        super().__init__(text, parent)
        base_color = "rgba(196,181,253,0.78)" if accent else MUTED
        self.set_button_font(_ui_font(pixel_size=13, weight=QFont.Weight.Medium))
        self.set_button_padding(9, 0, 9, 0)
        self.set_button_radius(8)
        self.setFixedHeight(28)
        self.configure_colors(
            normal_text=base_color,
            hover_bg=_pct_color(),
            hover_border=_pct_border(),
            hover_text=TEXT,
            active_bg=_pct_color(),
            active_border=_pct_border(),
            active_text=TEXT,
        )


class RailButton(PaintedButton):
    def __init__(self, text: str, active: bool = False):
        super().__init__(text)
        self.set_button_font(_ui_font(pixel_size=16))
        self.set_button_padding(0, 0, 0, 0)
        self.set_button_radius(10)
        self.setFixedSize(38, 38)
        self.configure_colors(
            normal_text=MUTED,
            hover_bg=SURFACE,
            hover_border=LINE,
            hover_text=TEXT,
            active_bg="rgba(20,19,28,.84)",
            active_border="rgba(139,92,246,.16)",
            active_text=TEXT,
        )
        self.set_active(active)


class Chip(CrispLabel):
    def __init__(self, text: str, active: bool = False):
        super().__init__(text)
        self.setFont(_ui_font(pixel_size=10))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_box_style(radius=12, padding=(8, 4, 8, 4))
        self.apply_chip_style(active)

    def apply_chip_style(self, active: bool) -> None:
        self.set_text_color(MUTED)
        self.set_box_style(
            background="transparent",
            border=LINE,
            radius=12,
            padding=(8, 4, 8, 4),
        )


class StatusPill(QWidget):
    def __init__(self, text: str):
        super().__init__()
        self._text = str(text or "")
        self._font = _topbar_font(pixel_size=10)
        self.setFixedHeight(20)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        width = int(14 + 5 + 6 + fm.horizontalAdvance(self._text) + 10)
        return QSize(width, 20)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(_to_qcolor(STATUS_LINE))
        painter.setBrush(_to_qcolor(STATUS_BG))
        painter.drawRoundedRect(rect, 6, 6)

        center_y = self.height() / 2.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_to_qcolor("#22c55e"))
        painter.drawEllipse(QRect(8, int(center_y - 2), 5, 5))

        painter.setPen(_to_qcolor(OK_TEXT))
        painter.setFont(self._font)
        painter.drawText(QRect(19, 0, self.width() - 27, self.height()), int(Qt.AlignmentFlag.AlignVCenter), self._text)


class ResourcePill(QWidget):
    def __init__(self, name: str, value: str):
        super().__init__()
        self._name = str(name or "")
        self._value = str(value or "")
        self._font = _topbar_font(pixel_size=10)
        self.setFixedHeight(22)

    def set_value(self, text: str) -> None:
        self._value = str(text or "")
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        width = int(8 + fm.horizontalAdvance(self._name) + 8 + fm.horizontalAdvance(self._value) + 10)
        return QSize(width, 22)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(_to_qcolor(RESOURCE_LINE))
        painter.setBrush(_to_qcolor(RESOURCE_BG))
        painter.drawRoundedRect(rect, 11, 11)

        painter.setFont(self._font)
        fm = QFontMetricsF(self._font)
        left = 8
        painter.setPen(_to_qcolor(MUTED_2))
        name_width = int(fm.horizontalAdvance(self._name))
        painter.drawText(QRect(left, 0, name_width + 2, self.height()), int(Qt.AlignmentFlag.AlignVCenter), self._name)

        painter.setPen(_to_qcolor(TEXT))
        value_left = left + name_width + 8
        painter.drawText(QRect(value_left, 0, self.width() - value_left - 8, self.height()), int(Qt.AlignmentFlag.AlignVCenter), self._value)


class ToggleSwitch(QCheckBox):
    def __init__(self, checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(34, 20)
        self.setText("")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._pressed_inside = False
        self._toggled_on_press = False

    def sizeHint(self) -> QSize:
        return QSize(34, 20)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._pressed_inside = self.rect().contains(event.position().toPoint())
            self._toggled_on_press = False
            if self._pressed_inside:
                self.toggle()
                self._toggled_on_press = True
            event.accept()
            return
        self._pressed_inside = False
        self._toggled_on_press = False
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            inside = self.rect().contains(event.position().toPoint())
            if self._pressed_inside and inside and not self._toggled_on_press:
                self.toggle()
            self._pressed_inside = False
            self._toggled_on_press = False
            event.accept()
            return
        self._pressed_inside = False
        self._toggled_on_press = False
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)

        if self.isChecked():
            track_bg = _to_qcolor("rgba(139,92,246,.78)")
            track_border = _to_qcolor("rgba(139,92,246,.95)")
            knob_color = _to_qcolor("#f5f3ff")
        else:
            track_bg = _to_qcolor("rgba(255,255,255,.07)")
            track_border = _to_qcolor("rgba(255,255,255,.10)")
            knob_color = _to_qcolor("#d1d5db")

        if not self.isEnabled():
            track_bg = _to_qcolor("rgba(255,255,255,.04)")
            track_border = _to_qcolor("rgba(255,255,255,.06)")
            knob_color = _to_qcolor("#8f96a3")

        painter.setPen(QPen(track_border, 1))
        painter.setBrush(track_bg)
        painter.drawRoundedRect(rect, 10, 10)

        knob_size = 14
        knob_x = rect.right() - knob_size - 2 if self.isChecked() else rect.left() + 2
        knob_y = rect.top() + (rect.height() - knob_size) // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(knob_color)
        painter.drawEllipse(QRect(knob_x, knob_y, knob_size, knob_size))


class InlineToggleButton(PaintedButton):
    def __init__(self, text: str, color: str = THINKING):
        super().__init__(text)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.set_button_font(_ui_font(pixel_size=9, weight=QFont.Weight.Medium))
        self.set_button_padding(0, 0, 0, 0)
        self.set_button_radius(0)
        self.set_text_alignment(Qt.AlignmentFlag.AlignLeft)
        self.configure_colors(
            normal_text=color,
            hover_text=color,
            active_text=color,
        )
        self.setFlat(True)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        width = int(fm.horizontalAdvance(self.text() or " ") + 1)
        height = max(13, int(fm.height() + 1))
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class TinyStatChip(CrispLabel):
    def __init__(self, text: str, *, active: bool = False):
        super().__init__(text, color=THINKING_TIME_TEXT if active else "rgba(148,163,184,.86)")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setFont(_ui_font(pixel_size=8 if active else 9))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_box_style(
            background=THINKING_TIME_BG if active else "rgba(17,20,26,.82)",
            border=THINKING_TIME_BORDER if active else "rgba(255,255,255,.08)",
            radius=7 if active else 8,
            padding=(6, 1, 6, 1) if active else (7, 2, 7, 2),
        )


class ChatBackdrop(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._cache_image: QImage | None = None
        self._cache_size = QSize()

    def _invalidate_cache(self) -> None:
        self._cache_image = None
        self._cache_size = QSize()

    def resizeEvent(self, event) -> None:
        self._invalidate_cache()
        super().resizeEvent(event)

    def _build_cache(self) -> None:
        w = max(1, self.width())
        h = max(1, self.height())
        if w <= 0 or h <= 0:
            return

        base_w = max(280, min(420, w // 4))
        base_h = max(180, min(300, h // 4))
        image = QImage(base_w, base_h, QImage.Format.Format_ARGB32_Premultiplied)

        bg = _to_qcolor(BG)
        base_r = bg.red()
        base_g = bg.green()
        base_b = bg.blue()

        sources = (
            (-0.08 * base_w, -0.10 * base_h, max(base_w * 0.48, base_h * 0.56), (108, 43, 217), 0.19, 2.55),
            (-0.10 * base_w, 1.10 * base_h, max(base_w * 0.52, base_h * 0.64), (108, 43, 217), 0.13, 2.40),
            (1.08 * base_w, -0.08 * base_h, max(base_w * 0.48, base_h * 0.58), (84, 152, 232), 0.13, 2.35),
        )

        for y in range(base_h):
            yf = float(y)
            for x in range(base_w):
                xf = float(x)
                r = float(base_r)
                g = float(base_g)
                b = float(base_b)

                for cx, cy, radius, (sr, sg, sb), strength, curve in sources:
                    dx = (xf - cx) / radius
                    dy = (yf - cy) / radius
                    dist2 = dx * dx + dy * dy
                    glow = exp(-dist2 * curve)
                    if glow < 0.0020:
                        continue
                    amount = glow * strength * 255.0
                    r += (sr / 255.0) * amount
                    g += (sg / 255.0) * amount
                    b += (sb / 255.0) * amount

                # Tiny procedural dither to break remaining dark-banding after upscale.
                noise = ((((x * 928371 + y * 364479) & 255) / 255.0) - 0.5) * 2.2
                r = max(0.0, min(255.0, r + noise))
                g = max(0.0, min(255.0, g + noise))
                b = max(0.0, min(255.0, b + noise))
                image.setPixelColor(x, y, QColor(int(r), int(g), int(b), 255))

        final_image = image.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        veil = QLinearGradient(0, 0, 0, h)
        veil.setColorAt(0.0, _to_qcolor("rgba(9,10,13,.040)"))
        veil.setColorAt(0.34, _to_qcolor("rgba(9,10,13,.010)"))
        veil.setColorAt(0.68, _to_qcolor("rgba(9,10,13,.00)"))
        veil.setColorAt(1.0, _to_qcolor("rgba(9,10,13,.030)"))
        painter = QPainter(final_image)
        painter.fillRect(QRectF(0, 0, w, h), veil)
        center_tone = QLinearGradient(0, 0, w, 0)
        center_tone.setColorAt(0.0, _to_qcolor("rgba(9,10,13,.00)"))
        center_tone.setColorAt(0.42, _to_qcolor("rgba(9,10,13,.020)"))
        center_tone.setColorAt(0.62, _to_qcolor("rgba(9,10,13,.024)"))
        center_tone.setColorAt(1.0, _to_qcolor("rgba(9,10,13,.00)"))
        painter.fillRect(QRectF(0, 0, w, h), center_tone)
        painter.end()

        self._cache_image = final_image
        self._cache_size = QSize(w, h)

    def paintEvent(self, _event) -> None:
        if self._cache_image is None or self._cache_size != self.size():
            self._build_cache()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if self._cache_image is not None:
            painter.drawImage(self.rect(), self._cache_image)
        else:
            painter.fillRect(self.rect(), _to_qcolor(BG))


class ComposerEdit(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._display_font = _ui_font(pixel_size=14)
        self.setFont(self._display_font)
        self.setStyleSheet(
            "QPlainTextEdit{background:transparent;border:none;color:transparent;padding:2px 0 0 5px;"
            "selection-background-color:rgba(139,92,246,.22);}"
        )
        self.textChanged.connect(self._refresh_overlay)
        self.cursorPositionChanged.connect(self._refresh_overlay)
        self.updateRequest.connect(lambda *_args: self._refresh_overlay())

    def _refresh_overlay(self) -> None:
        self.viewport().update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setFont(self._display_font)
        rect = self.viewport().rect().adjusted(5, 2, -4, -2)
        text = self.toPlainText()
        content_offset = self.contentOffset()
        rect.translate(int(content_offset.x()), int(content_offset.y()))
        if text:
            painter.setPen(_to_qcolor(TEXT))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap), text)
        elif self.placeholderText():
            painter.setPen(_to_qcolor(MUTED_2))
            painter.drawText(
                rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
                self.placeholderText(),
            )


class WebModeSelector(QWidget):
    modeChanged = Signal(str)

    def __init__(self, mode: str = "auto"):
        super().__init__()
        self._font = _ui_font(pixel_size=10)
        self._order = ("off", "auto", "aggressive")
        self._button_labels = {"off": "off", "auto": "auto", "aggressive": "on"}
        self._mode = mode if mode in self._order else "auto"
        self._hovered: str | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        width = 0
        for key in self._order:
            text = self._button_labels[key]
            width += int(fm.horizontalAdvance(text) + 10 * 2)
        width += 2 + 2 + (len(self._order) - 1) * 2
        return QSize(width, 24)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _segment_rects(self) -> dict[str, QRectF]:
        fm = QFontMetricsF(self._font)
        x = 3.0
        y = 2.0
        height = self.height() - 4.0
        rects: dict[str, QRectF] = {}
        for key in self._order:
            text = self._button_labels[key]
            horiz_pad = 10
            width = fm.horizontalAdvance(text) + horiz_pad * 2
            rects[key] = QRectF(x, y, width, height)
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
        for key, rect in self._segment_rects().items():
            is_active = key == self._mode
            is_hover = key == self._hovered
            if is_active or is_hover:
                fill = "rgba(139,92,246,.18)" if is_active else "rgba(255,255,255,.05)"
                border = "rgba(139,92,246,.34)" if is_active else "rgba(255,255,255,.09)"
                active_rect = rect.adjusted(0.0, 1.0, -1.0, -1.0)
                painter.setPen(QPen(_to_qcolor(border), 1))
                painter.setBrush(_to_qcolor(fill))
                painter.drawRoundedRect(active_rect, 10, 10)

            text_rect = rect
            painter.setPen(_to_qcolor(TEXT if (is_active or is_hover) else MUTED))
            painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter), self._button_labels[key])

    def set_mode(self, mode: str) -> None:
        resolved = mode if mode in self._order else "auto"
        changed = resolved != self._mode
        self._mode = resolved
        self.update()
        if changed:
            self.modeChanged.emit(resolved)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        hovered = None
        pos = event.position()
        for key, rect in self._segment_rects().items():
            if rect.contains(pos):
                hovered = key
                break
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            for key, rect in self._segment_rects().items():
                if rect.contains(pos):
                    self.set_mode(key)
                    break
        super().mouseReleaseEvent(event)


class PopupFrame(QFrame):
    closed = Signal()

    def __init__(self, anchor: QWidget, width: int = 240, line_orientation: str = "vertical"):
        super().__init__(anchor.window())
        self.anchor = anchor
        self._linked_popups: list["PopupFrame"] = []
        self.anchor.window().installEventFilter(self)
        self.setObjectName("popup_frame")
        self.setStyleSheet(
            f"QFrame#popup_frame {{background:rgba(15,18,22,.94); border:1px solid {LINE}; border-radius:14px;}}"
        )
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.SubWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedWidth(width)
        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._anim = QPropertyAnimation(self._opacity, b"opacity", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(340)
        self._hide_timer.timeout.connect(self.close_popup)
        self._line = QFrame(anchor.window())
        self._line.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._line_orientation = line_orientation
        if line_orientation == "vertical":
            self._line.setStyleSheet("background: rgba(139,92,246,.18); border-radius: 1px;")
            self._line.setFixedSize(3, 12)
        else:
            self._line.setStyleSheet("background: rgba(148,163,184,.28); border-radius: 1px;")
            self._line.setFixedSize(18, 3)
        self.hide()
        self._line.hide()

    def add_linked_popup(self, popup: "PopupFrame") -> None:
        if popup not in self._linked_popups:
            self._linked_popups.append(popup)

    def _has_visible_linked_popup(self) -> bool:
        return any(popup.isVisible() for popup in self._linked_popups)

    def open_above(self, x_offset: int = 0, y_gap: int = 12) -> None:
        anchor_bottom_left = self.anchor.mapTo(self.anchor.window(), QPoint(0, 0))
        x = anchor_bottom_left.x() + x_offset
        y = anchor_bottom_left.y() - self.height() - y_gap
        self.move(x, y)
        self._position_line(y_gap)
        self._hide_timer.stop()
        self.show()
        self.raise_()
        self._line.show()
        self._line.raise_()
        self._anim.stop()
        self._opacity.setOpacity(0.0)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()
        if hasattr(self.anchor, "set_active"):
            self.anchor.set_active(True)

    def _position_line(self, gap: int) -> None:
        anchor_pos = self.anchor.mapTo(self.anchor.window(), QPoint(0, 0))
        if self._line_orientation == "vertical":
            lx = anchor_pos.x() + self.anchor.width() // 2 - self._line.width() // 2
            ly = anchor_pos.y() - gap
            self._line.move(lx, ly)
        else:
            start_x = anchor_pos.x() + self.anchor.width() + 1
            end_x = self.x() + 2
            width = max(12, end_x - start_x)
            self._line.setFixedSize(width, 3)
            lx = start_x
            ly = anchor_pos.y() + self.anchor.height() // 2 - self._line.height() // 2
            self._line.move(lx, ly)

    def delayed_close(self, ms: int = 340) -> None:
        if self._has_visible_linked_popup():
            return
        self._hide_timer.start(ms)

    def cancel_close(self) -> None:
        self._hide_timer.stop()

    def close_popup(self) -> None:
        if self._has_visible_linked_popup():
            return
        self._hide_timer.stop()
        self.hide()
        self._line.hide()
        if hasattr(self.anchor, "set_active"):
            self.anchor.set_active(False)
        self.closed.emit()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress:
            if self.isVisible():
                mouse_event = event  # type: ignore[assignment]
                if isinstance(mouse_event, QMouseEvent):
                    global_pos = mouse_event.globalPosition().toPoint()
                else:
                    global_pos = QCursor.pos()
                local = self.mapFromGlobal(global_pos)
                anchor_local = self.anchor.mapFromGlobal(global_pos)
                linked_hit = False
                for popup in self._linked_popups:
                    if popup.isVisible() and popup.rect().contains(popup.mapFromGlobal(global_pos)):
                        linked_hit = True
                        break
                if not self.rect().contains(local) and not self.anchor.rect().contains(anchor_local) and not linked_hit:
                    self.close_popup()
        elif event.type() == QEvent.Type.KeyPress and self.isVisible():
            if getattr(event, 'key', lambda: None)() == Qt.Key.Key_Escape:
                self.close_popup()
        return False


class ThinkingPopup(QFrame):
    closed = Signal()

    def __init__(self, anchor: QWidget, preferred_width: int = 580):
        super().__init__(anchor.window())
        self.anchor = anchor
        self._preferred_width = max(280, int(preferred_width))
        self.anchor.window().installEventFilter(self)
        self.setObjectName("thinking_popup")
        self.setStyleSheet("QFrame#thinking_popup { background: transparent; border: none; }")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.SubWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.label = CrispLabel("", color=THINKING)
        self.label.setFont(_ui_font(pixel_size=12))
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.label.set_box_style(
            background="rgba(184,168,239,.02)",
            border="rgba(184,168,239,.18)",
            radius=10,
            padding=(10, 8, 10, 8),
            dashed=True,
        )
        lay.addWidget(self.label)
        self.hide()

    def set_text(self, text: str) -> None:
        self.label.setText(str(text or ""))
        self.refresh_geometry()

    def set_preferred_width(self, width: int) -> None:
        self._preferred_width = max(280, int(width))
        self.refresh_geometry()

    def _resolved_width(self) -> int:
        anchor_width = max(0, int(self.anchor.width()))
        return max(320, min(620, max(self._preferred_width, anchor_width)))

    def refresh_geometry(self) -> None:
        width = self._resolved_width()
        self.setFixedWidth(width)
        self.label.set_preferred_text_width(width - 20)
        self.setFixedHeight(self.sizeHint().height())
        if not self.isVisible():
            return
        anchor_pos = self.anchor.mapTo(self.anchor.window(), QPoint(0, 0))
        x = anchor_pos.x()
        y = max(8, anchor_pos.y() - self.height() - 6)
        self.move(x, y)

    def open_popup(self) -> None:
        self.refresh_geometry()
        self.show()
        self.raise_()

    def close_popup(self) -> None:
        if not self.isVisible():
            return
        self.hide()
        self.closed.emit()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if not self.isVisible():
            return False
        if event.type() == QEvent.Type.MouseButtonPress:
            mouse_event = event  # type: ignore[assignment]
            if isinstance(mouse_event, QMouseEvent):
                global_pos = mouse_event.globalPosition().toPoint()
            else:
                global_pos = QCursor.pos()
            local = self.mapFromGlobal(global_pos)
            anchor_local = self.anchor.mapFromGlobal(global_pos)
            if not self.rect().contains(local) and not self.anchor.rect().contains(anchor_local):
                self.close_popup()
        elif event.type() == QEvent.Type.KeyPress:
            if getattr(event, "key", lambda: None)() == Qt.Key.Key_Escape:
                self.close_popup()
        elif event.type() in {QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show}:
            self.refresh_geometry()
        elif event.type() in {QEvent.Type.Hide, QEvent.Type.Close}:
            self.close_popup()
        return False


class HoverSubmenuRow(QFrame):
    def __init__(
        self,
        title: str,
        parent_popup: PopupFrame,
        *,
        icon_text: str = "",
        submenu_title: str = "Команды",
        submenu_rows: list[tuple[str, str, bool]] | None = None,
    ):
        super().__init__(parent_popup)
        self.parent_popup = parent_popup
        self.controls: dict[str, ToggleSwitch] = {}
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"background:rgba(255,255,255,.02); border:1px solid {LINE}; border-radius:10px;")
        self.setFixedHeight(38)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)
        if icon_text:
            icon_label = CrispLabel(icon_text)
            icon_label.setFont(_ui_font(pixel_size=11))
            icon_label.set_text_color("#d1d5db")
            lay.addWidget(icon_label)
        title_label = CrispLabel(title)
        title_label.setFont(_ui_font(pixel_size=11))
        title_label.set_text_color(TEXT)
        lay.addWidget(title_label)
        caret = CrispLabel(">")
        caret.setFont(_ui_font(pixel_size=11))
        caret.set_text_color(MUTED)
        lay.addStretch(1)
        lay.addWidget(caret)
        self._submenu = PopupFrame(self, width=206, line_orientation="horizontal")
        self.parent_popup.add_linked_popup(self._submenu)
        self._submenu.setStyleSheet(f"QFrame#popup_frame {{background:rgba(15,18,22,.96); border:1px solid {LINE}; border-radius:14px;}}")
        v = QVBoxLayout(self._submenu)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)
        title_lbl = CrispLabel(submenu_title)
        title_lbl.setFont(_ui_font(pixel_size=11, weight=QFont.Weight.Medium))
        title_lbl.set_text_color(TEXT)
        v.addWidget(title_lbl)
        rows = submenu_rows or [("think", "think", True), ("verbose", "verbose", True), ("json", "json", False)]
        icons = {"think": "🤔", "verbose": "📝", "json": "{ }"}
        for key, label, checked in rows:
            row = function_row(label, icon_text=icons.get(key, ""))
            toggle = ToggleSwitch(checked)
            row.layout().addWidget(toggle)
            v.addWidget(row)
            self.controls[str(key)] = toggle
        self.setMouseTracking(True)
        self._submenu.setMouseTracking(True)
        self._submenu.enterEvent = lambda e: self._submenu.cancel_close()  # type: ignore[method-assign]
        self._submenu.leaveEvent = lambda e: self._submenu.delayed_close(450)  # type: ignore[method-assign]

    def enterEvent(self, event):
        super().enterEvent(event)
        self._submenu.setFixedHeight(self._submenu.sizeHint().height())
        self._submenu.open_above(x_offset=self.width() + 12, y_gap=-self.height() + 2)
        # manual placement to right side
        anchor_pos = self.mapTo(self.window(), QPoint(0, 0))
        x = anchor_pos.x() + self.width() + 12
        y = anchor_pos.y() - self._submenu.height() + self.height()
        self._submenu.move(x, y)
        self._submenu._position_line(0)
        self._submenu.cancel_close()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._submenu.delayed_close(450)

    def close_submenu(self) -> None:
        self._submenu.close_popup()


class MessageBubble(QFrame):
    def __init__(
        self,
        role: str,
        text: str = "",
        thinking: str = "",
        thinking_ms: str = "",
        perf: list[str] | None = None,
        *,
        show_thinking_header: bool = False,
    ):
        super().__init__()
        self.setObjectName("message_bubble_host")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("QFrame#message_bubble_host { background: transparent; border: none; }")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        bubble = QFrame()
        bubble.setObjectName("message_bubble_panel")
        bubble.setStyleSheet(
            f"QFrame#message_bubble_panel {{ background:{ASSISTANT_BG if role == 'assistant' else USER_BG}; border:1px solid {LINE}; border-radius:14px; }}"
        )
        bubble.setMaximumWidth(760)
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        bubble_lay = QVBoxLayout(bubble)
        bubble_lay.setContentsMargins(13, 11, 13, 11)
        bubble_lay.setSpacing(4)
        self._show_thinking_header = bool(show_thinking_header)
        self._thinking_text = str(thinking or "")
        self.thinking_head = None
        self.thinking_toggle = None
        self.thinking_label = None
        self.thinking_ms_chip = None
        if role == "assistant":
            thinking_head = QWidget()
            self.thinking_head = thinking_head
            thinking_head.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            thinking_head_lay = QHBoxLayout(thinking_head)
            thinking_head_lay.setContentsMargins(0, 0, 0, 0)
            thinking_head_lay.setSpacing(2)
            self.thinking_toggle = InlineToggleButton("▸  thinking")
            self.thinking_toggle.setFixedHeight(13)
            self.thinking_toggle.setCheckable(True)
            self.thinking_toggle.setChecked(False)
            thinking_head_lay.addWidget(self.thinking_toggle, 0, Qt.AlignmentFlag.AlignLeft)
            if thinking_ms:
                self.thinking_ms_chip = TinyStatChip(thinking_ms, active=True)
                thinking_head_lay.addWidget(self.thinking_ms_chip, 0, Qt.AlignmentFlag.AlignLeft)
            bubble_lay.addWidget(thinking_head, 0, Qt.AlignmentFlag.AlignLeft)
            self.thinking_label = CrispLabel(self._thinking_text, color=THINKING)
            self.thinking_label.setFont(_ui_font(pixel_size=12))
            self.thinking_label.setWordWrap(True)
            self.thinking_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self.thinking_label.set_preferred_text_width(560)
            self.thinking_label.setVisible(False)
            self.thinking_label.set_box_style(
                background="rgba(184,168,239,.02)",
                border="rgba(184,168,239,.18)",
                radius=10,
                padding=(10, 8, 10, 8),
                dashed=True,
            )
            bubble_lay.addWidget(self.thinking_label)
            self.thinking_toggle.toggled.connect(self._on_thinking_toggled)
            self.thinking_toggle.toggled.connect(self._sync_thinking_toggle_text)
            self._sync_thinking_toggle_text(False)
            self._sync_thinking_header_visibility()
        self.text_label = CrispLabel(text, color=TEXT)
        self.text_label.setFont(_ui_font(pixel_size=14))
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.text_label.set_preferred_text_width(620)
        bubble_lay.addWidget(self.text_label)
        outer.addWidget(bubble, 0, Qt.AlignmentFlag.AlignRight if role == 'user' else Qt.AlignmentFlag.AlignLeft)
        self.perf_wrap = FlowWrap(self)
        self.perf_wrap.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.perf_wrap.setVisible(bool(perf))
        perf_lay = FlowLayout(self.perf_wrap, margin=0, hspacing=6, vspacing=6)
        self.perf_wrap.setLayout(perf_lay)
        self.set_perf(list(perf or []))
        if role == "assistant":
            outer.addWidget(self.perf_wrap, 0, Qt.AlignmentFlag.AlignLeft)

    def _sync_thinking_toggle_text(self, checked: bool) -> None:
        if self.thinking_toggle is not None:
            self.thinking_toggle.setText(("▾  " if checked else "▸  ") + "thinking")
            self.thinking_toggle.setFixedWidth(self.thinking_toggle.sizeHint().width())
        if self.thinking_ms_chip is not None:
            self.thinking_ms_chip.setFixedWidth(self.thinking_ms_chip.sizeHint().width())

    def _sync_thinking_header_visibility(self) -> None:
        if self.thinking_head is None:
            return
        thinking_text = str(self._thinking_text or "").strip()
        has_thinking_text = bool(thinking_text)
        should_show_header = has_thinking_text
        self.thinking_head.setVisible(should_show_header)
        if not has_thinking_text and self.thinking_label is not None:
            self.thinking_label.setVisible(False)
        if not should_show_header and self.thinking_toggle is not None and self.thinking_toggle.isChecked():
            self.thinking_toggle.setChecked(False)

    def _find_scroll_area(self) -> QScrollArea | None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                return parent
            parent = parent.parentWidget()
        return None

    def _adjust_scroll_after_thinking_toggle(self, previous_height: int) -> None:
        scroll = self._find_scroll_area()
        if scroll is None:
            return
        delta = int(self.sizeHint().height() or self.height()) - int(previous_height or 0)
        if delta == 0:
            return
        bar = scroll.verticalScrollBar()
        bar.setValue(max(0, min(bar.maximum(), bar.value() + delta)))

    def _on_thinking_toggled(self, checked: bool) -> None:
        previous_height = int(self.sizeHint().height() or self.height())
        if checked:
            if not str(self._thinking_text or "").strip():
                if self.thinking_toggle is not None:
                    self.thinking_toggle.setChecked(False)
                return
            if self.thinking_label is not None:
                self.thinking_label.setVisible(True)
        elif self.thinking_label is not None:
            self.thinking_label.setVisible(False)
        self.updateGeometry()
        self.adjustSize()
        QTimer.singleShot(0, lambda prev=previous_height: self._adjust_scroll_after_thinking_toggle(prev))

    def update_thinking(self, text: str, ms: str | None = None) -> None:
        self._thinking_text = str(text or "")
        if self.thinking_label is not None:
            self.thinking_label.setText(self._thinking_text)
        if self.thinking_ms_chip is not None and ms is not None:
            self.thinking_ms_chip.setText(ms)
        elif ms is not None and self.thinking_toggle is not None and self.thinking_ms_chip is None:
            self.thinking_ms_chip = TinyStatChip(ms, active=True)
            head = self.thinking_toggle.parentWidget()
            if head is not None and head.layout() is not None:
                head.layout().insertWidget(1, self.thinking_ms_chip, 0, Qt.AlignmentFlag.AlignLeft)
        self._sync_thinking_header_visibility()
        if self.thinking_toggle is not None and self.thinking_toggle.isChecked() and self.thinking_label is not None:
            self.thinking_label.setVisible(bool(self._thinking_text.strip()))

    def update_text(self, text: str) -> None:
        self.text_label.setText(text)

    def set_perf(self, perf: list[str]) -> None:
        layout = self.perf_wrap.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        for item in perf:
            chip = TinyStatChip(item)
            layout.addWidget(chip)
        layout.invalidate()
        layout.activate()
        self.perf_wrap.adjustSize()
        self.perf_wrap.refresh_height()
        self.updateGeometry()
        self.perf_wrap.setVisible(bool(perf))


@dataclass
class PendingAssistant:
    bubble: MessageBubble
    started_at: float
    thinking_text: str = ""
    answer_text: str = ""
    first_thinking_at: float | None = None
    first_answer_at: float | None = None
    last_chunk_at: float | None = None


class ExactChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MMis — Exact PySide6 Chat")
        self.resize(1200, 820)
        self.setFont(_ui_font())
        self.setStyleSheet(f"QMainWindow {{background:{BG};}} QWidget {{color:{TEXT};}}")
        self.api = ApiClient() if ApiClient else None
        self._thread: QThread | None = None
        self._worker: ReplyWorker | None = None
        self._pending: PendingAssistant | None = None
        self._shutting_down = False
        self._gpu_ok = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._gpu_ok = True
            except Exception:
                self._gpu_ok = False
        self._build_ui()
        self._bind_popups()
        self._load_models()
        self._append_demo_messages()
        self._start_metrics_timer()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_about_to_quit)

    def _build_ui(self):
        root = ChatBackdrop()
        root.setObjectName("chat_root")
        root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)
        root.setStyleSheet("QWidget#chat_root { background: transparent; }")

        topbar = QFrame()
        topbar.setObjectName("chat_topbar")
        topbar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        topbar.setStyleSheet(f"QFrame#chat_topbar {{ background:rgba(13,16,19,.74); border-bottom:1px solid {LINE}; }}")
        topbar.setFixedHeight(64)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(12, 6, 12, 6)
        tb.setSpacing(10)

        left = QHBoxLayout()
        left.setSpacing(10)
        brand = BrandBadge("MMis")
        left.addWidget(brand)
        status_wrap = QHBoxLayout(); status_wrap.setSpacing(5)
        for name in ("api", "model", "memory"):
            status_wrap.addWidget(StatusPill(name))
        sw = QWidget(); sw.setLayout(status_wrap)
        left.addWidget(sw)
        left_w = QWidget(); left_w.setLayout(left)
        tb.addWidget(left_w, 0, Qt.AlignmentFlag.AlignLeft)
        tb.addStretch(1)

        self.cpu_pill = ResourcePill("CPU", "--")
        self.ram_pill = ResourcePill("RAM", "--")
        self.gpu_pill = ResourcePill("GPU", "--")
        self.vram_pill = ResourcePill("VRAM", "--")
        rr = QHBoxLayout(); rr.setSpacing(6)
        for pill in (self.cpu_pill, self.ram_pill, self.gpu_pill, self.vram_pill):
            rr.addWidget(pill)
        rw = QWidget(); rw.setLayout(rr)
        tb.addWidget(rw, 0, Qt.AlignmentFlag.AlignRight)
        root_lay.addWidget(topbar)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        root_lay.addLayout(layout, 1)

        rail = QFrame()
        rail.setObjectName("chat_rail")
        rail.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        rail.setFixedWidth(52)
        rail.setStyleSheet(f"QFrame#chat_rail {{ background:rgba(10,11,13,.10); border-right:1px solid {LINE}; }}")
        rail_lay = QVBoxLayout(rail)
        rail_lay.setContentsMargins(5, 7, 5, 7)
        rail_lay.setSpacing(5)
        rail_lay.addWidget(RailButton("💬", active=True), 0, Qt.AlignmentFlag.AlignTop)
        rail_lay.addWidget(RailButton("🎙"), 0, Qt.AlignmentFlag.AlignTop)
        rail_lay.addWidget(RailButton("📎"), 0, Qt.AlignmentFlag.AlignTop)
        rail_lay.addStretch(1)
        rail_lay.addWidget(RailButton("🧠"), 0, Qt.AlignmentFlag.AlignBottom)
        self.models_button = RailButton("🧬")
        rail_lay.addWidget(self.models_button, 0, Qt.AlignmentFlag.AlignBottom)
        rail_lay.addWidget(RailButton("⚙"), 0, Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(rail)

        chat = QFrame()
        chat.setObjectName("chat_surface")
        chat.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        chat.setStyleSheet("QFrame#chat_surface { background:transparent; }")
        chat_lay = QVBoxLayout(chat)
        chat_lay.setContentsMargins(0, 0, 0, 0)
        chat_lay.setSpacing(0)
        layout.addWidget(chat, 1)

        head = QFrame()
        head.setObjectName("chat_head")
        head.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        head.setStyleSheet(f"QFrame#chat_head {{ background:rgba(11,13,16,.04); border-bottom:1px solid {LINE}; }}")
        head_lay = QHBoxLayout(head)
        head_lay.setContentsMargins(14, 8, 14, 8)
        self.persona_label = CrispLabel("Ася")
        persona_font = QFont("Segoe Script")
        persona_font.setPixelSize(18)
        persona_font.setWeight(QFont.Weight.Black)
        self.persona_label.setFont(persona_font)
        self.persona_label.set_text_color("#c4b5fd")
        head_lay.addWidget(self.persona_label)
        head_lay.addStretch(1)
        self.search_btn = HoverButton("Поиск", accent=True)
        self.clear_btn = HoverButton("Очистить", accent=True)
        self.clear_btn.clicked.connect(self._clear_messages)
        head_lay.addWidget(self.search_btn)
        head_lay.addWidget(self.clear_btn)
        chat_lay.addWidget(head)

        modes = QFrame()
        modes.setObjectName("chat_modes")
        modes.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        modes.setStyleSheet(f"QFrame#chat_modes {{ background:rgba(11,13,16,.03); border-bottom:1px solid {LINE}; }}")
        modes_lay = QHBoxLayout(modes)
        modes_lay.setContentsMargins(14, 8, 14, 8)
        modes_lay.setSpacing(6)
        self.topic_chip = Chip("активная тема: вечер / самочувствие", active=True)
        modes_lay.addWidget(self.topic_chip)
        for t in ("режим: chat", "web: auto", "think", "verbose", "json"):
            modes_lay.addWidget(Chip(t))
        modes_lay.addStretch(1)
        chat_lay.addWidget(modes)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea{background:transparent;border:none;} QScrollBar:vertical{width:10px;background:transparent;} QScrollBar::handle:vertical{background:rgba(255,255,255,.16);border-radius:5px;}")
        self.messages_host = QWidget()
        self.messages_host.setObjectName("messages_host")
        self.messages_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.messages_host.setStyleSheet("QWidget#messages_host { background: transparent; }")
        self.messages_layout = QVBoxLayout(self.messages_host)
        self.messages_layout.setContentsMargins(18, 12, 18, 12)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch(1)
        self.scroll.setWidget(self.messages_host)
        chat_lay.addWidget(self.scroll, 1)

        composer_wrap = QFrame()
        composer_wrap.setObjectName("composer_wrap")
        composer_wrap.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        composer_wrap.setStyleSheet(f"QFrame#composer_wrap {{ background:rgba(11,13,16,.03); border-top:1px solid {LINE}; }}")
        composer_wrap_lay = QVBoxLayout(composer_wrap)
        composer_wrap_lay.setContentsMargins(14, 10, 14, 12)
        composer = QFrame()
        composer.setObjectName("composer_panel")
        composer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        composer.setStyleSheet(f"QFrame#composer_panel {{ background:rgba(15,18,22,.72); border:1px solid {LINE}; border-radius:14px; }}")
        composer_lay = QVBoxLayout(composer)
        composer_lay.setContentsMargins(8, 8, 8, 8)
        composer_lay.setSpacing(6)
        self.input = ComposerEdit()
        self.input.setPlaceholderText("Напиши сообщение")
        self.input.setFixedHeight(76)
        self.input.setFont(_ui_font(pixel_size=14))
        self.input.setStyleSheet(
            "QPlainTextEdit{background:transparent;border:none;color:transparent;padding:2px 0 0 5px;"
            "selection-background-color:rgba(139,92,246,.22);}"
        )
        composer_lay.addWidget(self.input)
        actions = QHBoxLayout(); actions.setSpacing(8)
        left_actions = QHBoxLayout(); left_actions.setSpacing(6)
        self.plus_btn = HoverButton("＋")
        self.mic_btn = HoverButton("🎙")
        self.functions_btn = HoverButton("Функции", accent=True)
        self.send_btn = HoverButton("Отправить", accent=True)
        self.send_btn.clicked.connect(self._send_message)
        left_actions.addWidget(self.plus_btn)
        left_actions.addWidget(self.mic_btn)
        left_actions.addWidget(self.functions_btn)
        actions.addLayout(left_actions)
        actions.addStretch(1)
        actions.addWidget(self.send_btn)
        composer_lay.addLayout(actions)
        composer_wrap_lay.addWidget(composer)
        chat_lay.addWidget(composer_wrap)

        self.functions_popup = self._build_functions_popup(self.functions_btn)
        self.models_popup = self._build_models_popup(self.models_button)

    def _build_models_popup(self, anchor: QWidget) -> PopupFrame:
        popup = PopupFrame(anchor, width=230, line_orientation="vertical")
        lay = QVBoxLayout(popup)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        title = CrispLabel("Модели")
        title.setFont(_ui_font(pixel_size=11, weight=QFont.Weight.Medium))
        title.set_text_color(TEXT)
        lay.addWidget(title)
        self.models_list_wrap = QWidget()
        self.models_list_layout = QVBoxLayout(self.models_list_wrap)
        self.models_list_layout.setContentsMargins(0, 0, 0, 0)
        self.models_list_layout.setSpacing(8)
        lay.addWidget(self.models_list_wrap)
        return popup

    def _populate_models(self, runtime: str, models: list[str]) -> None:
        while self.models_list_layout.count():
            item = self.models_list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if not models:
            models = [runtime or "qwen3:8b", "mistral:7b", "qwen coder"]
        active = runtime or (models[0] if models else "")
        for model in models:
            btn = PaintedButton()
            btn.set_button_font(_ui_font(pixel_size=12))
            btn.set_button_padding(8, 6, 8, 6)
            btn.set_button_radius(8)
            btn.set_text_alignment(Qt.AlignmentFlag.AlignLeft)
            btn.configure_colors(
                normal_bg="rgba(16,18,22,.28)",
                normal_border=LINE,
                normal_text=MUTED,
                hover_bg=_pct_color(),
                hover_border=_pct_border(),
                hover_text=TEXT,
                active_bg="rgba(16,18,22,.28)",
                active_border=LINE,
                active_text=MUTED,
            )
            btn.set_active(False)
            btn.setText(f"{model}{'    ✓' if model == active else ''}")
            btn.clicked.connect(lambda _=False, name=model: self._set_model(name))
            self.models_list_layout.addWidget(btn)

    def _build_functions_popup(self, anchor: QWidget) -> PopupFrame:
        popup = PopupFrame(anchor, width=240, line_orientation="vertical")
        lay = QVBoxLayout(popup)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        title = CrispLabel("Управление функциями")
        title.setFont(_ui_font(pixel_size=11, weight=QFont.Weight.Medium))
        title.set_text_color(TEXT)
        lay.addWidget(title)
        commands = HoverSubmenuRow(
            "команды",
            popup,
            icon_text="⌘",
            submenu_title="Команды",
            submenu_rows=[("think", "think", True), ("verbose", "verbose", True), ("json", "json", False)],
        )
        lay.addWidget(commands)
        screen = function_row("screen", icon_text="▣")
        screen.layout().addWidget(ToggleSwitch(False))
        lay.addWidget(screen)
        web = function_row("web", icon_text="🌐")
        web.layout().setContentsMargins(10, 5, 8, 7)
        web.layout().addWidget(WebModeSelector("auto"), 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(web)
        self._commands_row = commands
        return popup

    def _bind_popups(self) -> None:
        self.functions_btn.clicked.connect(self._toggle_functions)
        self.models_button.clicked.connect(self._toggle_models)

    def _toggle_functions(self) -> None:
        self._toggle_popup(self.functions_popup, self.functions_btn, x_offset=0, y_gap=12)

    def _toggle_models(self) -> None:
        self._toggle_popup(self.models_popup, self.models_button, x_offset=0, y_gap=12)

    def _toggle_popup(self, popup: PopupFrame, anchor: QWidget, x_offset: int, y_gap: int) -> None:
        if popup.isVisible():
            popup.close_popup()
            return
        for other in (self.functions_popup, self.models_popup):
            if other is not popup:
                other.close_popup()
        popup.setFixedHeight(popup.sizeHint().height())
        popup.open_above(x_offset=x_offset, y_gap=y_gap)

    def _append_demo_messages(self) -> None:
        self._append_message("user", "Я сегодня вообще вымотался. Надо ещё немного позаниматься проектом, но голова уже кипит. Можешь помочь спокойно разложить вечер без душных советов?")
        self._append_message(
            "assistant",
            "Да. Давай без насилия над собой: сначала 15–20 минут просто выдохни, попей воды или чая и вообще не трогай код. Потом можно взять одну маленькую задачу по проекту минут на 30–40, без попытки «сейчас всё наверстаю». А после этого — уже спокойно выключаться на отдых.",
            thinking="Пользователь устал и просит не жёсткий план, а спокойную помощь. Лучше ответить мягко, дать короткий реалистичный сценарий на вечер и не перегружать его пунктами.",
            thinking_ms="412 ms",
            perf=["2986 ms", "write 2647 ms", "17.0 tok/s", "prompt 697", "gen 45"],
        )
        self._append_message("user", "Вот это уже ближе. И давай ещё так, чтобы я не залип до трёх ночи. Мне нужен какой-то мягкий стоп, а то я опять скажу себе «ещё пять минут» и пропаду.")
        self._append_message(
            "assistant",
            "Тогда ставим себе честный стоп, например через час. До него — только одна конкретная задача, без новых идей и без «заодно ещё это». А когда время выйдет, закрываешь проект, умываешься, откладываешь телефон хотя бы на 15 минут и уже потом ложишься. Так шанс сорваться в ночной марафон будет сильно меньше.",
            thinking="Нужно поддержать его и дать ощущение контроля. Подойдёт мягкий ритуал завершения: конкретное время стопа, один последний шаг по проекту и короткое переключение перед сном.",
            thinking_ms="287 ms",
            perf=["2142 ms", "write 1880 ms", "16.5 tok/s", "prompt 512", "gen 31"],
        )

    def _append_message(self, role: str, text: str, thinking: str = "", thinking_ms: str = "", perf: list[str] | None = None) -> MessageBubble:
        bubble = MessageBubble(role, text, thinking, thinking_ms, perf)
        self.messages_layout.addWidget(bubble)
        QTimer.singleShot(0, self._scroll_bottom)
        return bubble

    def _clear_messages(self) -> None:
        for index in range(self.messages_layout.count() - 1, -1, -1):
            item = self.messages_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            taken = self.messages_layout.takeAt(index)
            if taken is not None and widget is not None:
                widget.deleteLater()

    def _scroll_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _send_message(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        self._append_message("user", text)
        self.input.clear()
        if self.api is None or ReplyWorker is None:
            self._append_message(
                "assistant",
                "Я накинула тебе нативную PySide6-оболочку. Чтобы сделать её полностью живой, нужно подключить твой текущий ApiClient/ReplyWorker прямо в проекте.",
                thinking="Сейчас это режим локального превью без реального API-ответа.",
                thinking_ms="96 ms",
                perf=["612 ms", "write 410 ms", "14.2 tok/s", "prompt 143", "gen 20"],
            )
            return
        self._pending = PendingAssistant(
            bubble=self._append_message("assistant", "", thinking="", thinking_ms="0 ms", perf=[]),
            started_at=time.perf_counter(),
        )
        self._worker = ReplyWorker(self.api, text, store_turn=True, think=True)
        self._worker.chunk.connect(self._on_answer_chunk)
        self._worker.thinking_chunk.connect(self._on_thinking_chunk)
        self._worker.finished.connect(self._on_reply_finished)
        self._worker.errored.connect(self._on_reply_error)
        self._worker.start()

    @Slot(str)
    def _on_answer_chunk(self, piece: str) -> None:
        if not self._pending:
            return
        self._pending.answer_text += piece
        self._pending.bubble.update_text(self._pending.answer_text)
        self._scroll_bottom()

    @Slot(str)
    def _on_thinking_chunk(self, piece: str) -> None:
        if not self._pending:
            return
        self._pending.thinking_text += piece
        elapsed_ms = int((time.perf_counter() - self._pending.started_at) * 1000)
        self._pending.bubble.update_thinking(self._pending.thinking_text, f"{elapsed_ms} ms")
        self._scroll_bottom()

    @Slot(object)
    def _on_reply_finished(self, result) -> None:
        if not self._pending:
            return
        text = getattr(result, "text", "") or self._pending.answer_text
        thinking = getattr(result, "thinking", "") or self._pending.thinking_text
        stats = getattr(result, "stats", {}) or {}
        elapsed = int(float(stats.get("elapsed_ms") or stats.get("total_ms") or max(1, (time.perf_counter() - self._pending.started_at) * 1000)))
        write_ms = int(float(stats.get("decode_ms") or stats.get("eval_ms") or 0))
        tok_s = stats.get("tokens_per_second") or stats.get("tok_s") or stats.get("tps")
        prompt_t = stats.get("prompt_tokens") or stats.get("prompt_eval_count")
        gen_t = stats.get("gen_tokens") or stats.get("eval_count")
        perf = [f"{elapsed} ms"]
        if write_ms:
            perf.append(f"write {write_ms} ms")
        if tok_s not in (None, ""):
            try:
                perf.append(f"{float(tok_s):.1f} tok/s")
            except Exception:
                perf.append(f"{tok_s} tok/s")
        if prompt_t not in (None, ""):
            perf.append(f"prompt {prompt_t}")
        if gen_t not in (None, ""):
            perf.append(f"gen {gen_t}")
        self._pending.bubble.update_text(text)
        self._pending.bubble.update_thinking(thinking, f"{elapsed} ms")
        self._pending.bubble.set_perf(perf)
        self._pending = None
        self._scroll_bottom()

    @Slot(str)
    def _on_reply_error(self, error_text: str) -> None:
        if self._pending:
            self._pending.bubble.update_text(f"Ошибка: {error_text}")
            self._pending.bubble.set_perf([])
            self._pending = None

    def _start_metrics_timer(self) -> None:
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._refresh_metrics)
        self._metrics_timer.start(1500)
        self._refresh_metrics()

    def _on_about_to_quit(self) -> None:
        self._shutting_down = True
        timer = getattr(self, "_metrics_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass

    def _refresh_metrics(self) -> None:
        if self._shutting_down:
            return
        if psutil is not None:
            try:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                self.cpu_pill.set_value(f"{cpu:.0f}%")
                self.ram_pill.set_value(f"{mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB")
            except KeyboardInterrupt:
                return
            except Exception:
                pass
        if self._gpu_ok and pynvml is not None:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_pill.set_value(f"{util.gpu}%")
                self.vram_pill.set_value(f"{meminfo.used / (1024**3):.1f} / {meminfo.total / (1024**3):.1f} GB")
            except KeyboardInterrupt:
                return
            except Exception:
                self.gpu_pill.set_value("--")
                self.vram_pill.set_value("--")

    def closeEvent(self, event) -> None:
        self._on_about_to_quit()
        super().closeEvent(event)

    def _load_models(self) -> None:
        if not self.api:
            self._populate_models("qwen3:8b", ["qwen3:8b", "mistral:7b", "qwen coder"])
            return
        try:
            payload = self.api.list_models()
            runtime = str(payload.get("runtime_model") or self.api.get_runtime_model() or "")
            available = payload.get("available_models") or payload.get("models") or []
            models = [str(x) for x in available if str(x).strip()]
            self._populate_models(runtime, models)
        except Exception:
            self._populate_models(self.api.get_runtime_model() or "qwen3:8b", ["qwen3:8b", "mistral:7b", "qwen coder"])

    def _set_model(self, model_name: str) -> None:
        if self.api:
            try:
                self.api.set_model(model_name)
            except Exception:
                pass
        self._populate_models(model_name, [model_name, "mistral:7b", "qwen coder"])
        self.models_popup.close_popup()


def function_row(label_text: str, *, icon_text: str = "") -> QFrame:
    row = QFrame()
    row.setStyleSheet(f"background:rgba(255,255,255,.02); border:1px solid {LINE}; border-radius:10px;")
    row.setFixedHeight(38)
    lay = QHBoxLayout(row)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(8)
    if icon_text:
        icon = CrispLabel(icon_text)
        icon.setFont(_ui_font(pixel_size=11))
        icon.set_text_color("#d1d5db")
        lay.addWidget(icon)
    label = CrispLabel(label_text)
    label.setFont(_ui_font(pixel_size=11))
    label.set_text_color(TEXT)
    lay.addWidget(label)
    lay.addStretch(1)
    return row


def main() -> int:
    _configure_qt_startup()
    app = QApplication(sys.argv)
    win = ExactChatWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
