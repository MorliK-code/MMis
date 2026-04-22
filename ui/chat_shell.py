from __future__ import annotations

import os
import re
import sys
import time
import warnings
from html import escape
from math import exp
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPoint, QPointF, Property, QPropertyAnimation, QRect, QRectF, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QLinearGradient, QMouseEvent, QPainter, QPainterPath, QFontMetricsF, QPen, QRadialGradient, QTextCursor, QTextDocument
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
    QScrollBar,
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
    from ui.workers import ReplyWorker, ReplyResult, StatusPollWorker
except Exception:  # pragma: no cover
    ApiClient = None
    ApiClientError = RuntimeError
    ReplyWorker = None
    ReplyResult = None
    StatusPollWorker = None


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
BAD_BG = "rgba(239,68,68,0.12)"
BAD_LINE = "rgba(239,68,68,0.24)"
BAD_TEXT = "#fca5a5"
BAD_DOT = "#ef4444"
OK_DOT = "#22c55e"
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

def _button_font(*, pixel_size: int | None = None, weight: int = QFont.Weight.Medium) -> QFont:
    font = QFont("Cascadia Code")
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


INLINE_CODE_TEXT = "#d8ccff"
INLINE_CODE_BG = "#211a2f"


def _paired_marker_positions(text: str, marker: str) -> set[int]:
    positions: set[int] = set()
    start_at = 0
    marker_len = len(marker)
    while True:
        start = text.find(marker, start_at)
        if start < 0:
            break
        end = text.find(marker, start + marker_len)
        if end < 0:
            break
        positions.add(start)
        positions.add(end)
        start_at = end + marker_len
    return positions


def _format_message_html(text: str, *, text_color: str = TEXT) -> str:
    raw = str(text or "")
    if not raw:
        return ""

    bold_markers = _paired_marker_positions(raw, "**")
    inline_code_markers = _paired_marker_positions(raw, "`")
    chars: list[tuple[str, bool, bool]] = []
    bold = False
    inline_code = False
    index = 0
    while index < len(raw):
        if index in inline_code_markers and raw.startswith("`", index):
            inline_code = not inline_code
            index += 1
            continue
        if not inline_code and index in bold_markers and raw.startswith("**", index):
            bold = not bold
            index += 2
            continue
        ch = raw[index]
        chars.append((ch, bold, inline_code))
        index += 1

    parts: list[str] = []
    run: list[str] = []
    run_style: tuple[bool, bool] | None = None

    def _flush() -> None:
        nonlocal run, run_style
        if not run:
            return
        bold_on, inline_code_on = run_style or (False, False)
        body = "".join(run)
        if bold_on:
            body = f"<b>{body}</b>"
        if inline_code_on:
            body = (
                f'<span style="color:{INLINE_CODE_TEXT}; background-color:{INLINE_CODE_BG}; '
                f"font-family:'Cascadia Code','Consolas','monospace'; "
                f"font-size:12px;\">{body}</span>"
            )
        parts.append(body)
        run = []

    for ch, bold_on, inline_code_on in chars:
        style = (bold_on, inline_code_on)
        if run_style is not None and style != run_style:
            _flush()
        run_style = style
        run.append("<br>" if ch == "\n" else escape(ch))
    _flush()
    return f'<span style="color:{text_color};">{"".join(parts)}</span>'


class StyledMessageLabel(QLabel):
    def __init__(self, text: str = "", color: str = TEXT, parent: QWidget | None = None):
        super().__init__("", parent)
        self._raw_text = ""
        self._text_color = str(color or TEXT)
        self._preferred_text_width: int | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setStyleSheet("QLabel { background: transparent; }")
        self.setText(text)

    def text(self) -> str:  # noqa: N802
        return self._raw_text

    def rendered_html(self) -> str:
        return super().text()

    def setText(self, text: str) -> None:  # noqa: N802
        self._raw_text = str(text or "")
        super().setText(_format_message_html(self._raw_text, text_color=self._text_color))
        self.updateGeometry()
        self.update()

    def set_text_color(self, color: str) -> None:
        self._text_color = str(color or TEXT)
        super().setText(_format_message_html(self._raw_text, text_color=self._text_color))
        self.update()

    def set_preferred_text_width(self, width: int | None) -> None:
        self._preferred_text_width = None if width is None else max(0, int(width))
        self.updateGeometry()

    def setWordWrap(self, on: bool) -> None:  # noqa: N802
        super().setWordWrap(on)
        self.updateGeometry()

    def hasHeightForWidth(self) -> bool:
        return self.wordWrap()

    def heightForWidth(self, width: int) -> int:
        return self._measure_size(width).height()

    def sizeHint(self) -> QSize:
        if self.wordWrap():
            return self._measure_size(self._preferred_text_width or 320)
        return self._measure_size(None)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _document_for_width(self, width: int | None) -> QTextDocument:
        doc = QTextDocument()
        doc.setDefaultFont(self.font())
        doc.setDocumentMargin(0)
        doc.setHtml(self.rendered_html() or " ")
        if width is not None:
            doc.setTextWidth(max(24, int(width)))
        return doc

    def _measure_size(self, width: int | None) -> QSize:
        if not self.wordWrap() or width is None:
            doc = self._document_for_width(None)
            return QSize(int(doc.idealWidth()), int(doc.size().height()) + 2)

        target_width = max(24, int(width or self._preferred_text_width or 320))
        ideal_doc = self._document_for_width(None)
        text_width = min(target_width, max(24, int(ideal_doc.idealWidth()) + 1))
        doc = self._document_for_width(text_width)
        return QSize(text_width, int(doc.size().height()) + 2)


class StreamingTextBox(QFrame):
    def __init__(self, text: str = "", color: str = THINKING, parent: QWidget | None = None):
        super().__init__(parent)
        self._text_color = str(color or THINKING)
        self._background_color = _to_qcolor("transparent")
        self._border_color = _to_qcolor("transparent")
        self._border_style = Qt.PenStyle.SolidLine
        self._radius = 0
        self._padding = (0, 0, 0, 0)
        self._preferred_text_width = 320
        self._editor = QPlainTextEdit(self)
        self._editor.setReadOnly(True)
        self._editor.setFrameShape(QFrame.Shape.NoFrame)
        self._editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._editor.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._editor.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._editor.document().setDocumentMargin(0)
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._editor.setStyleSheet(
            "QPlainTextEdit {"
            "background:transparent;"
            "border:none;"
            f"color:{self._text_color};"
            "selection-background-color:rgba(184,168,239,.25);"
            "}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setText(text)

    def setFont(self, font: QFont) -> None:  # noqa: N802
        super().setFont(font)
        self._editor.setFont(font)
        self._refresh_height()

    def text(self) -> str:
        return self._editor.toPlainText()

    def setText(self, text: str) -> None:  # noqa: N802
        value = str(text or "")
        if self._editor.toPlainText() == value:
            return
        self._editor.setPlainText(value)
        self._editor.moveCursor(QTextCursor.MoveOperation.End)
        self._refresh_height()

    def append_stream_text(self, text: str) -> None:
        value = str(text or "")
        if not value:
            return
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(value)
        self._editor.setTextCursor(cursor)
        self._refresh_height()

    def setWordWrap(self, on: bool) -> None:  # noqa: N802
        self._editor.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth
            if bool(on)
            else QPlainTextEdit.LineWrapMode.NoWrap
        )
        self._refresh_height()

    def setAlignment(self, _alignment) -> None:  # noqa: N802
        return

    def set_preferred_text_width(self, width: int | None) -> None:
        value = max(24, int(width or 320))
        if self._preferred_text_width == value:
            return
        self._preferred_text_width = value
        self._refresh_height()

    def set_box_style(
        self,
        *,
        background: str = "transparent",
        border: str = "transparent",
        radius: int = 0,
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        dashed: bool = False,
    ) -> None:
        self._background_color = _to_qcolor(background)
        self._border_color = _to_qcolor(border)
        self._border_style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        self._radius = max(0, int(radius))
        self._padding = tuple(int(value) for value in padding)
        self._refresh_height()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self._preferred_text_width, max(1, self.height()))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _refresh_height(self) -> None:
        left, top, right, bottom = self._padding
        width = max(24, int(self._preferred_text_width))
        inner_width = max(24, width - left - right - 2)
        self._editor.document().setTextWidth(inner_width)
        fm = QFontMetricsF(self._editor.font())
        text = self._editor.toPlainText() or " "
        rect = fm.boundingRect(
            QRect(0, 0, inner_width, 10_000_000),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
            text,
        )
        doc_height = max(int(rect.height()), int(fm.height()))
        self.setFixedWidth(width)
        self.setFixedHeight(max(1, doc_height + top + bottom + 8))
        self._sync_editor_geometry()
        self.updateGeometry()
        self.update()

    def _content_rect(self) -> QRect:
        left, top, right, bottom = self._padding
        return self.rect().adjusted(left, top, -right, -bottom)

    def _sync_editor_geometry(self) -> None:
        self._editor.setGeometry(self._content_rect())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_editor_geometry()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        frame_rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setBrush(self._background_color)
        if self._border_color.alpha() > 0:
            pen = QPen(self._border_color)
            pen.setStyle(self._border_style)
            painter.setPen(pen)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(frame_rect, self._radius, self._radius)


class PaintedButton(QPushButton):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        # self._font = _ui_font(pixel_size=13)
        self._font = QFont("Cascadia Code")
        self._font.setPixelSize(13)
        self._font.setWeight(QFont.Weight.Medium)
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
        self._font = QFont("Cascadia Code")
        self._font.setPixelSize(14)
        self._font.setWeight(QFont.Weight.DemiBold)
        self._font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
        self._font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
        self._dot_opacity = 0.42
        self.setFixedHeight(24)
        self._dot_animation = QPropertyAnimation(self, b"dotOpacity", self)
        self._dot_animation.setDuration(3200)
        self._dot_animation.setStartValue(0.42)
        self._dot_animation.setKeyValueAt(0.5, 1.0)
        self._dot_animation.setEndValue(0.42)
        self._dot_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._dot_animation.setLoopCount(-1)
        self._dot_animation.start()

    def get_dot_opacity(self) -> float:
        return float(self._dot_opacity)

    def set_dot_opacity(self, value: float) -> None:
        self._dot_opacity = max(0.0, min(1.0, float(value)))
        self.update()

    dotOpacity = Property(float, get_dot_opacity, set_dot_opacity)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        width = int(24 + 5 + fm.horizontalAdvance(self._text) + 2)
        return QSize(width, 24)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        center_y = self.height() / 2.0
        painter.setPen(Qt.PenStyle.NoPen)
        dot_center_x = 9.5
        dot_radius = 3.0
        glow_radius = 9.5
        glow_power = self._dot_opacity * self._dot_opacity
        glow_center = QColor(_to_qcolor(ACCENT))
        glow_center.setAlphaF(0.025 + (0.18 * glow_power))
        glow_mid = QColor(_to_qcolor(ACCENT))
        glow_mid.setAlphaF(0.006 + (0.055 * glow_power))
        glow_edge = QColor(_to_qcolor(ACCENT))
        glow_edge.setAlphaF(0.0)
        glow = QRadialGradient(QPointF(dot_center_x, center_y), glow_radius)
        glow.setColorAt(0.0, glow_center)
        glow.setColorAt(0.64, glow_mid)
        glow.setColorAt(1.0, glow_edge)
        painter.setBrush(glow)
        painter.drawEllipse(QRectF(dot_center_x - glow_radius, center_y - glow_radius, glow_radius * 2.0, glow_radius * 2.0))

        dot_color = _to_qcolor(ACCENT)
        dot_color.setAlphaF(self._dot_opacity)
        painter.setBrush(dot_color)
        painter.drawEllipse(QRectF(dot_center_x - dot_radius, center_y - dot_radius, dot_radius * 2.0, dot_radius * 2.0))

        painter.setFont(self._font)
        painter.setPen(_to_qcolor("#d9d0ff"))
        painter.drawText(QRect(24, 0, self.width() - 24, self.height()), int(Qt.AlignmentFlag.AlignVCenter), self._text)


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
        self.set_button_font(_button_font(pixel_size=13, weight=QFont.Weight.DemiBold))
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
        self.set_button_font(_button_font(pixel_size=16, weight=QFont.Weight.DemiBold))
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
    def __init__(self, text: str, active: bool = False):
        super().__init__()
        self._text = str(text or "")
        self._font = _topbar_font(pixel_size=10)
        self._active = bool(active)
        self.setFixedHeight(20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_status(self, active: bool, *, tooltip: str = "", text: str | None = None) -> None:
        if text is not None:
            self._text = str(text or "")
        self._active = bool(active)
        if tooltip:
            self.setToolTip(tooltip)
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font)
        width = int(14 + 5 + 6 + fm.horizontalAdvance(self._text) + 10)
        return QSize(width, 20)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(0, 0, -1, -1)
        bg = STATUS_BG if self._active else BAD_BG
        line = STATUS_LINE if self._active else BAD_LINE
        dot = OK_DOT if self._active else BAD_DOT
        text = OK_TEXT if self._active else BAD_TEXT
        painter.setPen(_to_qcolor(line))
        painter.setBrush(_to_qcolor(bg))
        painter.drawRoundedRect(rect, 6, 6)

        center_y = self.height() / 2.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_to_qcolor(dot))
        painter.drawEllipse(QRect(8, int(center_y - 2), 5, 5))

        painter.setPen(_to_qcolor(text))
        painter.setFont(self._font)
        painter.drawText(QRect(19, 0, self.width() - 27, self.height()), int(Qt.AlignmentFlag.AlignVCenter), self._text)


class ChatScrollOverlay(QWidget):
    def __init__(self, scroll_area: QScrollArea):
        super().__init__(scroll_area.viewport())
        self._scroll_area = scroll_area
        self._bar = scroll_area.verticalScrollBar()
        self._dragging = False
        self._drag_offset = 0.0
        self._last_handle_rect = QRectF()
        self._trail: list[tuple[QRectF, float, int]] = []
        self._trail_start_alpha = 0.42
        self._last_scroll_direction = 0
        self._trail_fade_delay = QTimer(self)
        self._trail_fade_delay.setSingleShot(True)
        self._trail_fade_delay.setInterval(35)
        self._trail_fade_delay.timeout.connect(self._start_trail_fade)
        self._trail_fade_timer = QTimer(self)
        self._trail_fade_timer.setInterval(24)
        self._trail_fade_timer.timeout.connect(self._fade_trail)
        self.setFixedWidth(12)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        scroll_area.viewport().installEventFilter(self)
        self._bar.valueChanged.connect(self._on_value_changed)
        self._bar.rangeChanged.connect(self._on_range_changed)
        QTimer.singleShot(0, self._sync_geometry)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._scroll_area.viewport() and event.type() in {
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.LayoutRequest,
        }:
            self._sync_geometry()
        return super().eventFilter(watched, event)

    def _sync_geometry(self) -> None:
        viewport = self._scroll_area.viewport()
        self.setGeometry(max(0, viewport.width() - 14), 0, 12, viewport.height())
        visible = self._bar.maximum() > self._bar.minimum()
        self.setVisible(visible)
        if visible:
            self.raise_()
        self.update()

    def _track_rect(self) -> QRectF:
        return QRectF(self.rect()).adjusted(2.0, 10.0, -2.0, -10.0)

    def _handle_rect(self) -> QRectF:
        track = self._track_rect()
        if track.width() <= 0.0 or track.height() <= 0.0:
            return QRectF()
        minimum = int(self._bar.minimum())
        maximum = int(self._bar.maximum())
        if maximum <= minimum:
            return QRectF()

        page_step = max(1, int(self._bar.pageStep()))
        visible_ratio = page_step / max(1, (maximum - minimum) + page_step)
        natural_height = track.height() * visible_ratio
        max_height = min(92.0, track.height() * 0.34)
        handle_height = max(28.0, min(natural_height, max_height))
        handle_height = min(handle_height, track.height())
        travel = max(0.0, track.height() - handle_height)
        value_ratio = (int(self._bar.value()) - minimum) / max(1, maximum - minimum)
        top = track.top() + travel * value_ratio
        return QRectF(track.left(), top, track.width(), handle_height)

    def _set_value_from_y(self, y: float) -> None:
        track = self._track_rect()
        handle = self._handle_rect()
        travel = max(1.0, track.height() - handle.height())
        top = max(track.top(), min(float(y) - self._drag_offset, track.bottom() - handle.height()))
        ratio = (top - track.top()) / travel
        minimum = int(self._bar.minimum())
        maximum = int(self._bar.maximum())
        self._bar.setValue(round(minimum + (maximum - minimum) * ratio))

    def _on_value_changed(self, _value: int) -> None:
        current = self._handle_rect()
        delta = current.top() - self._last_handle_rect.top()
        if self._last_handle_rect.width() > 0.0 and abs(delta) > 1.0:
            direction = 1 if delta > 0.0 else -1
            if self._last_scroll_direction and self._last_scroll_direction != direction:
                self._trail.clear()
            self._last_scroll_direction = direction
            self._trail.append((QRectF(self._last_handle_rect), self._trail_start_alpha, direction))
            self._trail = self._trail[-8:]
            self._trail_fade_timer.stop()
            self._trail_fade_delay.start()
        self._last_handle_rect = QRectF(current)
        self.update()

    def _on_range_changed(self, _minimum: int, _maximum: int) -> None:
        self._trail.clear()
        self._last_scroll_direction = 0
        self._trail_fade_delay.stop()
        self._trail_fade_timer.stop()
        self._last_handle_rect = self._handle_rect()
        self._sync_geometry()

    def _start_trail_fade(self) -> None:
        if self._trail:
            self._trail_fade_timer.start()

    def _fade_trail(self) -> None:
        if not self._trail:
            self._trail_fade_timer.stop()
            return

        handle = self._handle_rect()
        if handle.width() <= 0.0 or handle.height() <= 0.0:
            self._trail.clear()
            self._trail_fade_timer.stop()
            self.update()
            return

        handle_center = handle.center().y()
        distances = [abs(rect.center().y() - handle_center) for rect, _alpha, _direction in self._trail]
        max_distance = max(distances) if distances else 0.0
        track_height = max(1.0, self._track_rect().height())
        # The longer the trail, the stronger the overall geometric fade.
        length_factor = 1.0 + min(1.6, max_distance / (track_height * 0.45))

        faded_trail: list[tuple[QRectF, float, int]] = []
        for (rect, alpha, direction), distance in zip(self._trail, distances):
            distance_ratio = (distance / max_distance) if max_distance > 0.0 else 0.0
            # Near the thumb: slower decay. Far from the thumb: faster decay.
            geo_base = self._lerp(0.90, 0.56, distance_ratio)
            next_alpha = float(alpha) * (geo_base**length_factor)
            if next_alpha > 0.006:
                faded_trail.append((rect, next_alpha, direction))

        self._trail = faded_trail
        if not self._trail:
            self._trail_fade_timer.stop()
        self.update()

    @staticmethod
    def _scroll_color(alpha: float) -> QColor:
        color = QColor(165, 139, 255)
        color.setAlphaF(max(0.0, min(1.0, float(alpha))))
        return color

    @staticmethod
    def _lerp(start: float, end: float, amount: float) -> float:
        t = max(0.0, min(1.0, float(amount)))
        return float(start) + ((float(end) - float(start)) * t)

    def _paint_trail_smear(self, painter: QPainter, track: QRectF, handle: QRectF) -> None:
        if not self._trail:
            return
        direction = self._last_scroll_direction
        relevant_trail = [
            (rect, alpha, rect_direction)
            for rect, alpha, rect_direction in self._trail
            if alpha > 0.006
            and rect.width() > 0.0
            and rect.height() > 0.0
            and (direction == 0 or rect_direction == direction)
        ]
        if not relevant_trail:
            return
        fade_progress = 1.0 - (relevant_trail[0][1] / max(0.001, self._trail_start_alpha))
        if direction > 0:
            relevant_trail = [
                (rect, alpha, rect_direction)
                for rect, alpha, rect_direction in relevant_trail
                if rect.center().y() <= handle.center().y()
            ]
            if not relevant_trail:
                return
            far_rect = relevant_trail[0][0]
            next_rect = relevant_trail[1][0] if len(relevant_trail) > 1 else handle
            top = self._lerp(far_rect.top(), next_rect.top(), fade_progress)
            bottom = handle.top() + min(5.0, handle.height() * 0.18)
        elif direction < 0:
            relevant_trail = [
                (rect, alpha, rect_direction)
                for rect, alpha, rect_direction in relevant_trail
                if rect.center().y() >= handle.center().y()
            ]
            if not relevant_trail:
                return
            far_rect = relevant_trail[0][0]
            next_rect = relevant_trail[1][0] if len(relevant_trail) > 1 else handle
            top = handle.bottom() - min(5.0, handle.height() * 0.18)
            bottom = self._lerp(far_rect.bottom(), next_rect.bottom(), fade_progress)
        else:
            rects = [rect for rect, _alpha, _rect_direction in relevant_trail]
            top = min([rect.top() for rect in rects] + [handle.top()])
            bottom = max([rect.bottom() for rect in rects] + [handle.bottom()])
        if bottom <= top:
            return
        max_alpha = max(alpha for _rect, alpha, _rect_direction in relevant_trail)
        bounds = QRectF(track)
        if direction > 0:
            # Keep a subtle overlap with the handle so the smear reads as continuous.
            bounds.setBottom(min(track.bottom(), handle.top() + 8.0))
        elif direction < 0:
            # Keep a subtle overlap with the handle so the smear reads as continuous.
            bounds.setTop(max(track.top(), handle.bottom() - 8.0))
        smear = QRectF(
            handle.left(),
            top - 9.0,
            handle.width(),
            (bottom - top) + 18.0,
        ).intersected(bounds)
        if smear.width() <= 0.0 or smear.height() <= 0.0:
            return

        for rect, alpha_scale in (
            (smear.adjusted(0.0, -3.0, 0.0, 3.0).intersected(bounds), 0.30),
            (smear, 0.82),
        ):
            if rect.width() <= 0.0 or rect.height() <= 0.0:
                continue
            mid_alpha = max_alpha * alpha_scale
            gradient = QLinearGradient(0.0, rect.top(), 0.0, rect.bottom())
            if direction < 0:
                gradient.setColorAt(0.0, self._scroll_color(mid_alpha * 0.88))
                gradient.setColorAt(0.18, self._scroll_color(mid_alpha))
                gradient.setColorAt(0.72, self._scroll_color(mid_alpha * 0.20))
                gradient.setColorAt(1.0, self._scroll_color(0.0))
            else:
                gradient.setColorAt(0.0, self._scroll_color(0.0))
                gradient.setColorAt(0.24, self._scroll_color(mid_alpha * 0.20))
                gradient.setColorAt(0.82, self._scroll_color(mid_alpha))
                gradient.setColorAt(1.0, self._scroll_color(mid_alpha * 0.88))
            painter.setBrush(gradient)
            painter.drawRect(rect)

    def paintEvent(self, _event) -> None:
        track = self._track_rect()
        handle = self._handle_rect()
        if handle.width() <= 0.0 or handle.height() <= 0.0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)

        track_color = QColor(165, 139, 255)
        track_color.setAlphaF(0.035)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, track.width() / 2.0, track.width() / 2.0)

        # Clip trail by the exact rounded thumb shape to avoid any visual gap.
        painter.save()
        full_path = QPainterPath()
        full_path.addRect(QRectF(self.rect()))
        handle_radius = min(handle.width(), handle.height()) / 2.0
        handle_path = QPainterPath()
        handle_path.addRoundedRect(handle, handle_radius, handle_radius)
        painter.setClipPath(full_path.subtracted(handle_path))
        self._paint_trail_smear(painter, track, handle)
        painter.restore()

        alpha = 0.76 if self._dragging else (0.66 if self.underMouse() else 0.56)
        painter.setBrush(self._scroll_color(alpha))
        painter.drawRoundedRect(handle, handle_radius, handle_radius)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        handle = self._handle_rect()
        y = event.position().y()
        self._dragging = True
        self._drag_offset = y - handle.top() if handle.contains(event.position()) else handle.height() / 2.0
        self._set_value_from_y(y)
        event.accept()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._set_value_from_y(event.position().y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
            self.update()
            return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.update()


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
        self.set_button_font(_button_font(pixel_size=9, weight=QFont.Weight.Medium))
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
    submitRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._display_font = _ui_font(pixel_size=14)
        self._native_text_visible: bool | None = None
        self.setFont(self._display_font)
        self._apply_native_text_style(False)
        self.textChanged.connect(self._refresh_overlay)
        self.cursorPositionChanged.connect(self._refresh_overlay)
        self.selectionChanged.connect(self._refresh_overlay)
        self.updateRequest.connect(lambda *_args: self._refresh_overlay())

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.insertPlainText("\n")
            else:
                self.submitRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _refresh_overlay(self) -> None:
        self._apply_native_text_style(self.textCursor().hasSelection())
        self.viewport().update()

    def _apply_native_text_style(self, visible: bool) -> None:
        native_visible = bool(visible)
        if self._native_text_visible is native_visible:
            return
        self._native_text_visible = native_visible
        text_color = TEXT if native_visible else "transparent"
        self.setStyleSheet(
            "QPlainTextEdit{"
            "background:transparent;"
            "border:none;"
            f"color:{text_color};"
            "padding:2px 0 0 5px;"
            "selection-background-color:rgba(139,92,246,.22);"
            f"selection-color:{TEXT};"
            "}"
        )

    def _overlay_text_rect(self) -> QRect:
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        origin = self.cursorRect(cursor)
        left = max(0, int(origin.left()))
        top = max(-self.viewport().height(), int(origin.top()))
        return QRect(
            left,
            top,
            max(1, self.viewport().width() - left - 4),
            max(1, self.viewport().height() - top - 2),
        )

    def paintEvent(self, event) -> None:
        has_selection = self.textCursor().hasSelection()
        self._apply_native_text_style(has_selection)
        super().paintEvent(event)
        if has_selection:
            return
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setFont(self._display_font)
        rect = self._overlay_text_rect()
        text = self.toPlainText()
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
    regenerateRequested = Signal(object)

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
        self.role = str(role or "")
        self.setObjectName("message_bubble_host")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("QFrame#message_bubble_host { background: transparent; border: none; }")
        self.setMouseTracking(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        bubble = QFrame()
        self._message_panel = bubble
        bubble.setMouseTracking(True)
        bubble.setObjectName("message_bubble_panel")
        bubble.setStyleSheet(
            f"QFrame#message_bubble_panel {{ background:{ASSISTANT_BG if role == 'assistant' else USER_BG}; border:1px solid {LINE}; border-radius:14px; }}"
        )
        bubble.setMaximumWidth(760)
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        bubble_lay = QVBoxLayout(bubble)
        self._message_layout = bubble_lay
        bubble_lay.setContentsMargins(13, 11, 13, 11)
        bubble_lay.setSpacing(4)
        self._body_text_width = 620
        self._show_thinking_header = bool(show_thinking_header)
        self._thinking_text = str(thinking or "")
        self._thinking_label_text = ""
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
            self.thinking_label = StreamingTextBox("", color=THINKING)
            self.thinking_label.setFont(_ui_font(pixel_size=12))
            self.thinking_label.setWordWrap(True)
            self.thinking_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            self.thinking_label.set_preferred_text_width(self._body_text_width)
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
        self.text_label = StyledMessageLabel(text, color=TEXT)
        self.text_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.text_label.setFont(_ui_font(pixel_size=14))
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.text_label.set_preferred_text_width(self._body_text_width)
        bubble_lay.addWidget(self.text_label)
        self._sync_body_widths()
        self.regenerate_btn = None
        if role == "user":
            row = QWidget()
            row.setMouseTracking(True)
            row.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(6)
            self.regenerate_btn = PaintedButton("↻")
            self.regenerate_btn.setToolTip("Перегенерировать ответ")
            self.regenerate_btn.setFixedSize(24, 24)
            self.regenerate_btn.set_button_font(_button_font(pixel_size=14, weight=QFont.Weight.DemiBold))
            self.regenerate_btn.set_button_padding(0, 0, 0, 1)
            self.regenerate_btn.set_button_radius(7)
            self.regenerate_btn.configure_colors(
                normal_bg="rgba(16,18,22,.70)",
                normal_border="rgba(255,255,255,.08)",
                normal_text=MUTED,
                hover_bg=_pct_color(),
                hover_border=_pct_border(),
                hover_text=TEXT,
                active_bg=_pct_color(),
                active_border=_pct_border(),
                active_text=TEXT,
            )
            self.regenerate_btn.clicked.connect(lambda _checked=False: self.regenerateRequested.emit(self))
            self.regenerate_btn.setVisible(False)
            row_lay.addWidget(self.regenerate_btn, 0, Qt.AlignmentFlag.AlignVCenter)
            row_lay.addWidget(bubble, 0, Qt.AlignmentFlag.AlignVCenter)
            outer.addWidget(row, 0, Qt.AlignmentFlag.AlignRight)
        else:
            outer.addWidget(bubble, 0, Qt.AlignmentFlag.AlignLeft)
        self.perf_wrap = FlowWrap(self)
        self.perf_wrap.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.perf_wrap.setVisible(bool(perf))
        perf_lay = FlowLayout(self.perf_wrap, margin=0, hspacing=6, vspacing=6)
        self.perf_wrap.setLayout(perf_lay)
        self.set_perf(list(perf or []))
        if role == "assistant":
            outer.addWidget(self.perf_wrap, 0, Qt.AlignmentFlag.AlignLeft)

    def _set_regenerate_button_visible(self, visible: bool) -> None:
        if self.regenerate_btn is None:
            return
        self.regenerate_btn.setVisible(bool(visible) and self.isEnabled())

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._set_regenerate_button_visible(True)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._set_regenerate_button_visible(False)

    def _sync_thinking_toggle_text(self, checked: bool) -> None:
        if self.thinking_toggle is not None:
            self.thinking_toggle.setText(("▾  " if checked else "▸  ") + "thinking")
            self.thinking_toggle.setFixedWidth(self.thinking_toggle.sizeHint().width())
        if self.thinking_ms_chip is not None:
            self.thinking_ms_chip.setFixedWidth(self.thinking_ms_chip.sizeHint().width())

    def _sync_thinking_header_visibility(self) -> None:
        if self.thinking_head is None:
            return
        has_thinking_text = bool(self._thinking_text)
        should_show_header = has_thinking_text
        self.thinking_head.setVisible(should_show_header)
        if not has_thinking_text and self.thinking_label is not None:
            self.thinking_label.setVisible(False)
        if not should_show_header and self.thinking_toggle is not None and self.thinking_toggle.isChecked():
            self.thinking_toggle.setChecked(False)

    def _sync_thinking_label_text(self) -> bool:
        if self.thinking_label is None:
            return False
        if self._thinking_label_text == self._thinking_text:
            return False
        self._thinking_label_text = self._thinking_text
        self.thinking_label.setText(self._thinking_text)
        return True

    def _append_thinking_label_text(self) -> bool:
        if self.thinking_label is None:
            return False
        if self._thinking_label_text == self._thinking_text:
            return False
        if self._thinking_text.startswith(self._thinking_label_text):
            delta = self._thinking_text[len(self._thinking_label_text):]
            self._thinking_label_text = self._thinking_text
            append = getattr(self.thinking_label, "append_stream_text", None)
            if callable(append):
                append(delta)
                return True
        return self._sync_thinking_label_text()

    def _sync_body_widths(self) -> bool:
        panel = getattr(self, "_message_panel", None)
        panel_layout = getattr(self, "_message_layout", None)
        target = int(getattr(self, "_body_text_width", 620) or 620)
        if panel is not None and panel_layout is not None:
            margins = panel_layout.contentsMargins()
            content_rect = panel.contentsRect()
            if panel.width() > 0:
                target = max(
                    target,
                    int(content_rect.width()) - margins.left() - margins.right(),
                )
            max_panel_width = int(panel.maximumWidth() or 0)
            if 0 < max_panel_width < 16_777_215:
                target = min(target, max(24, max_panel_width - margins.left() - margins.right()))
        target = max(24, target)
        changed = False
        if getattr(self.text_label, "_preferred_text_width", None) != target:
            self.text_label.set_preferred_text_width(target)
            changed = True
        if self.thinking_label is not None and getattr(self.thinking_label, "_preferred_text_width", None) != target:
            self.thinking_label.set_preferred_text_width(target)
            changed = True
        return changed

    def _reflow_message_body(self) -> None:
        panel = getattr(self, "_message_panel", None)
        panel_layout = getattr(self, "_message_layout", None)
        self._sync_body_widths()
        if panel is not None:
            if panel_layout is not None:
                panel_layout.invalidate()
                needed = max(panel_layout.sizeHint().height(), panel_layout.minimumSize().height())
                if needed > 0:
                    panel.setMinimumHeight(needed)
                    panel.resize(panel.width(), needed)
                panel_layout.activate()
            panel.updateGeometry()
        host_layout = self.layout()
        if host_layout is not None:
            host_layout.invalidate()
            host_layout.activate()
        self.updateGeometry()
        self.adjustSize()
        parent = self.parentWidget()
        while parent is not None:
            parent_layout = parent.layout()
            if parent_layout is not None:
                parent_layout.invalidate()
                parent_layout.activate()
            parent.updateGeometry()
            if isinstance(parent, QScrollArea):
                break
            parent = parent.parentWidget()
        window = self.window()
        sync = getattr(window, "_schedule_messages_view_height_sync", None)
        if callable(sync):
            sync()

    def _reflow_after_thinking_body_change(self) -> None:
        self._reflow_message_body()

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
            if not self._thinking_text:
                if self.thinking_toggle is not None:
                    self.thinking_toggle.setChecked(False)
                return
            self._sync_thinking_label_text()
            if self.thinking_label is not None:
                self.thinking_label.setVisible(True)
        elif self.thinking_label is not None:
            self.thinking_label.setVisible(False)
        self._reflow_message_body()
        QTimer.singleShot(0, lambda prev=previous_height: self._adjust_scroll_after_thinking_toggle(prev))

    def update_thinking(self, text: str, ms: str | None = None) -> None:
        self._thinking_text = str(text or "")
        body_changed = False
        if (
            self.thinking_label is not None
            and self.thinking_toggle is not None
            and self.thinking_toggle.isChecked()
        ):
            body_changed = self._append_thinking_label_text()
        if self.thinking_ms_chip is not None and ms is not None:
            self.thinking_ms_chip.setText(ms)
        elif ms is not None and self.thinking_toggle is not None and self.thinking_ms_chip is None:
            self.thinking_ms_chip = TinyStatChip(ms, active=True)
            head = self.thinking_toggle.parentWidget()
            if head is not None and head.layout() is not None:
                head.layout().insertWidget(1, self.thinking_ms_chip, 0, Qt.AlignmentFlag.AlignLeft)
        self._sync_thinking_header_visibility()
        if self.thinking_toggle is not None and self.thinking_toggle.isChecked() and self.thinking_label is not None:
            was_visible = self.thinking_label.isVisible()
            self.thinking_label.setVisible(bool(self._thinking_text))
            if body_changed or was_visible != self.thinking_label.isVisible():
                self._reflow_message_body()

    def update_text(self, text: str) -> None:
        value = str(text or "")
        if self.text_label.text() == value:
            return
        self.text_label.setText(value)
        self._reflow_message_body()

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
        self._status_worker: QThread | None = None
        self._pending: PendingAssistant | None = None
        self._messages_view_height_sync_queued = False
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

    def _resolve_persona_display_name(self) -> str:
        return ""

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
        tb.setContentsMargins(6, 6, 12, 6)
        tb.setSpacing(10)

        left = QHBoxLayout()
        left.setSpacing(10)
        brand = BrandBadge("MMis")
        left.addWidget(brand)
        status_wrap = QHBoxLayout(); status_wrap.setSpacing(5)
        self.status_pills: dict[str, StatusPill] = {}
        for name in ("api", "model", "memory"):
            pill = StatusPill(name, active=False)
            self.status_pills[name] = pill
            status_wrap.addWidget(pill)
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
        self.persona_label = CrispLabel(self._resolve_persona_display_name())
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
        self.scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            """
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget {
                background: transparent;
            }
            """
        )
        self.messages_host = QWidget()
        self.messages_host.setObjectName("messages_host")
        self.messages_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.messages_host.setStyleSheet("QWidget#messages_host { background: transparent; }")
        self.messages_layout = QVBoxLayout(self.messages_host)
        self.messages_layout.setContentsMargins(18, 12, 18, 12)
        self.messages_layout.setSpacing(10)
        self.scroll.setWidget(self.messages_host)
        self.chat_scroll_overlay = ChatScrollOverlay(self.scroll)
        chat_lay.addWidget(self.scroll)

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
        self.input.submitRequested.connect(self._send_message)
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
        chat_lay.addStretch(1)

        self.functions_popup = self._build_functions_popup(self.functions_btn)
        self.models_popup = self._build_models_popup(self.models_button)
        composer_wrap.setFixedHeight(composer_wrap.sizeHint().height())

    def _message_widget_count(self) -> int:
        layout = getattr(self, "messages_layout", None)
        if layout is None:
            return 0
        count = 0
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is not None and item.widget() is not None:
                count += 1
        return count

    def _available_messages_view_height(self) -> int:
        scroll = getattr(self, "scroll", None)
        if scroll is None:
            return 0
        parent = scroll.parentWidget()
        if parent is None:
            return int(scroll.height() or 0)
        layout = parent.layout()
        margins = layout.contentsMargins() if layout is not None else None
        available = int(parent.height() or 0)
        if margins is not None:
            available -= int(margins.top() + margins.bottom())
        for name in ("chat_head", "chat_modes", "composer_wrap"):
            widget = parent.findChild(QWidget, name)
            if widget is not None and widget.isVisible():
                available -= int(widget.height() or widget.sizeHint().height())
        return max(0, available)

    def _sync_messages_view_height(self) -> None:
        self._messages_view_height_sync_queued = False
        layout = getattr(self, "messages_layout", None)
        scroll = getattr(self, "scroll", None)
        if layout is None or scroll is None:
            return
        if self._message_widget_count() <= 0:
            scroll.setVisible(False)
            scroll.setFixedHeight(0)
            return
        scroll.setVisible(True)
        layout.invalidate()
        layout.activate()
        content_height = max(int(layout.sizeHint().height()), int(layout.minimumSize().height()), 1)
        available_height = self._available_messages_view_height()
        target_height = content_height if available_height <= 0 else min(content_height, available_height)
        target_height = max(1, int(target_height))
        if int(scroll.minimumHeight()) != target_height or int(scroll.maximumHeight()) != target_height:
            scroll.setFixedHeight(target_height)
        self.messages_host.updateGeometry()
        scroll.updateGeometry()
        parent = scroll.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().invalidate()
            parent.layout().activate()

    def _schedule_messages_view_height_sync(self) -> None:
        if getattr(self, "_messages_view_height_sync_queued", False):
            return
        self._messages_view_height_sync_queued = True
        QTimer.singleShot(0, self._sync_messages_view_height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule_messages_view_height_sync()

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
        self._wire_regenerate_bubble(bubble)
        insert_index = self.messages_layout.count()
        self.messages_layout.insertWidget(insert_index, bubble)
        self._schedule_messages_view_height_sync()
        QTimer.singleShot(0, self._scroll_bottom)
        return bubble

    def _wire_regenerate_bubble(self, bubble: QWidget) -> None:
        if not isinstance(bubble, MessageBubble) or getattr(bubble, "role", "") != "user":
            return
        if bool(bubble.property("regenerate_wired")):
            return
        bubble.regenerateRequested.connect(self._on_regenerate_requested)
        bubble.setProperty("regenerate_wired", True)

    def _message_layout_index(self, bubble: QWidget) -> int:
        layout = getattr(self, "messages_layout", None)
        if layout is None:
            return -1
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is not None and item.widget() is bubble:
                return index
        return -1

    def _remove_message_bubble(self, bubble: QWidget) -> None:
        layout = getattr(self, "messages_layout", None)
        if layout is None:
            return
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is not None and item.widget() is bubble:
                taken = layout.takeAt(index)
                if taken is not None:
                    bubble.deleteLater()
                self._schedule_messages_view_height_sync()
                return

    def _remove_assistant_after_user_bubble(self, user_bubble: QWidget) -> MessageBubble | None:
        layout = getattr(self, "messages_layout", None)
        if layout is None:
            return None
        start = self._message_layout_index(user_bubble)
        if start < 0:
            return None
        for index in range(start + 1, layout.count()):
            item = layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if not isinstance(widget, MessageBubble):
                continue
            if getattr(widget, "role", "") == "user":
                return None
            if getattr(widget, "role", "") == "assistant":
                self._remove_message_bubble(widget)
                return widget
        return None

    def _start_reply_for_text(self, text: str, *, store_turn: bool, append_user: bool) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        if self._worker is not None and self._worker.isRunning():
            return False
        if append_user:
            self._append_message("user", text)
        if self.api is None or ReplyWorker is None:
            self._append_message(
                "assistant",
                "Я накинула тебе нативную PySide6-оболочку. Чтобы сделать её полностью живой, нужно подключить твой текущий ApiClient/ReplyWorker прямо в проекте.",
                thinking="Сейчас это режим локального превью без реального API-ответа.",
                thinking_ms="96 ms",
                perf=["612 ms", "write 410 ms", "14.2 tok/s", "prompt 143", "gen 20"],
            )
            return True
        self._pending = PendingAssistant(
            bubble=self._append_message("assistant", "", thinking="", thinking_ms="0 ms", perf=[]),
            started_at=time.perf_counter(),
        )
        self._worker = ReplyWorker(self.api, text, store_turn=store_turn, think=True)
        self._worker.chunk.connect(self._on_answer_chunk)
        self._worker.thinking_chunk.connect(self._on_thinking_chunk)
        self._worker.finished.connect(self._on_reply_finished)
        self._worker.errored.connect(self._on_reply_error)
        self._worker.start()
        return True

    @Slot(object)
    def _on_regenerate_requested(self, bubble: object) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if not isinstance(bubble, MessageBubble):
            return
        text = bubble.text_label.text().strip()
        if not text:
            return
        self._remove_assistant_after_user_bubble(bubble)
        self._start_reply_for_text(text, store_turn=False, append_user=False)

    def _clear_messages(self) -> None:
        for index in range(self.messages_layout.count() - 1, -1, -1):
            item = self.messages_layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            taken = self.messages_layout.takeAt(index)
            if taken is not None and widget is not None:
                widget.deleteLater()
        self._sync_messages_view_height()

    def _scroll_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _send_message(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        if self._start_reply_for_text(text, store_turn=True, append_user=True):
            self.input.clear()

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
        self._scroll_bottom()

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

        self._pending.bubble.update_thinking(self._pending.thinking_text, f"{elapsed_ms} ms")
        self._scroll_bottom()

    @Slot(object)
    def _on_reply_finished(self, result) -> None:
        if not self._pending:
            return
        finished_at = time.perf_counter()
        text = getattr(result, "text", "") or self._pending.answer_text
        thinking = _merge_streamed_and_final_text(
            self._pending.thinking_text,
            getattr(result, "thinking", ""),
        )
        stats = getattr(result, "stats", {}) or {}
        elapsed = int(float(stats.get("elapsed_ms") or stats.get("total_ms") or max(1, (finished_at - self._pending.started_at) * 1000)))
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
        thinking_ms = ""
        if str(thinking or "").strip():
            if self._pending.first_answer_at is not None and self._pending.first_thinking_at is not None:
                thinking_ms = f"{max(1, int((self._pending.first_answer_at - self._pending.first_thinking_at) * 1000))} ms"
        self._pending.bubble.update_text(text)
        self._pending.bubble.update_thinking(thinking, thinking_ms)
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
        self._metrics_timer.timeout.connect(self._refresh_backend_status)
        self._metrics_timer.start(1500)
        self._refresh_metrics()
        self._refresh_backend_status()

    def _on_about_to_quit(self) -> None:
        self._shutting_down = True
        timer = getattr(self, "_metrics_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        worker = getattr(self, "_status_worker", None)
        if worker is not None:
            try:
                worker.wait(3000)
            except Exception:
                pass

    def _set_status_pill(self, name: str, active: bool, tooltip: str = "") -> None:
        pill = getattr(self, "status_pills", {}).get(str(name or ""))
        if pill is None:
            return
        pill.set_status(bool(active), tooltip=tooltip)

    def _apply_backend_status(
        self,
        *,
        api_ok: bool,
        model_ok: bool = False,
        memory_ok: bool = False,
        model: str = "",
        model_state: str = "",
        memory_state: str = "",
        error: str = "",
    ) -> None:
        model_text = str(model or "").strip() or "missing"
        model_state_text = str(model_state or ("active" if model_ok else "off")).strip()
        memory_text = str(memory_state or "").strip() or "off"
        api_tip = "API: online" if api_ok else f"API: offline{': ' + error if error else ''}"
        model_tip = f"Model: {model_state_text} ({model_text})"
        memory_tip = f"Memory LLM: {memory_text}"
        self._set_status_pill("api", api_ok, api_tip)
        self._set_status_pill("model", bool(api_ok and model_ok), model_tip)
        self._set_status_pill("memory", bool(api_ok and memory_ok), memory_tip)

    def _refresh_backend_status(self) -> None:
        if self._shutting_down:
            return
        if self.api is None or StatusPollWorker is None:
            self._apply_backend_status(api_ok=False, error="ApiClient unavailable")
            return
        worker = getattr(self, "_status_worker", None)
        if worker is not None and worker.isRunning():
            return
        worker = StatusPollWorker(self.api)
        self._status_worker = worker
        worker.status_ready.connect(self._on_backend_status_result)
        worker.finished.connect(lambda: self._finish_status_worker(worker))
        worker.start()

    def _finish_status_worker(self, worker: QThread) -> None:
        if getattr(self, "_status_worker", None) is worker:
            self._status_worker = None
        try:
            worker.deleteLater()
        except Exception:
            pass

    @Slot(object)
    def _on_backend_status_result(self, payload: object) -> None:
        row = dict(payload or {}) if isinstance(payload, dict) else {}
        api_ok = bool(row.get("api_ok"))
        if not api_ok:
            self._apply_backend_status(api_ok=False, error=str(row.get("error") or ""))
            return
        model = str(row.get("model") or "").strip()
        memory_status = dict(row.get("memory_status") or {})
        model_status = dict(row.get("model_status") or {})
        model_state = str(model_status.get("state") or "")
        memory_state = str(memory_status.get("state") or "off")
        memory_ok = bool(memory_status.get("active"))
        if "active" in model_status:
            model_ok = bool(model_status.get("active"))
        else:
            model_ok = bool(model)
        if not model and self.api is not None:
            model = self.api.get_runtime_model()
        self._apply_backend_status(
            api_ok=True,
            model_ok=model_ok,
            memory_ok=memory_ok,
            model=model,
            model_state=model_state,
            memory_state=memory_state,
        )

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
