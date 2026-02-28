"""Event filters and interactive button visuals."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QGraphicsOpacityEffect, QPushButton, QWidget
from shiboken6 import isValid


class _HoverRevealFilter(QObject):
    def __init__(
        self,
        owner: QWidget,
        target: QWidget,
        show_ms: int = 170,
        hide_ms: int = 130,
        require_reenter: bool = False,
    ):
        super().__init__(owner)
        self.owner = owner
        self.target = target
        self.show_ms = max(0, int(show_ms))
        self.hide_ms = max(40, int(hide_ms))
        self._must_leave_once = bool(require_reenter)
        self._pressed_inside_target = False
        self._outside_streak = 0
        target.setVisible(False)
        self._sync_timer = QTimer(target)
        self._sync_timer.setInterval(60)
        self._sync_timer.timeout.connect(self._sync_visibility)
        self._sync_timer.start()

    def _contains_global(self, w: QWidget, global_pos, pad: int = 2) -> bool:
        if not isValid(w):
            return False
        local = w.mapFromGlobal(global_pos)
        rect = w.rect().adjusted(-pad, -pad, pad, pad)
        return rect.contains(local)

    def _is_cursor_inside_owner_or_target(self) -> bool:
        if not isValid(self.owner) or not isValid(self.target):
            return False
        gpos = QCursor.pos()
        hovered = QApplication.widgetAt(gpos)
        if hovered is None:
            hovered_hit = False
        else:
            hovered_hit = (
                hovered is self.owner
                or hovered is self.target
                or (isinstance(hovered, QWidget) and self.owner.isAncestorOf(hovered))
                or (isinstance(hovered, QWidget) and self.target.isAncestorOf(hovered))
            )
        if hovered_hit:
            return True

        if self._contains_global(self.owner, gpos, pad=3):
            return True
        if self._contains_global(self.target, gpos, pad=3):
            return True
        for child in self.target.findChildren(QWidget):
            if self._contains_global(child, gpos, pad=3):
                return True
        return False

    def _show_if_inside(self) -> None:
        if not isValid(self.target):
            return
        if self._is_cursor_inside_owner_or_target() or self._pressed_inside_target:
            self._outside_streak = 0
            self.target.setVisible(True)

    def _show(self) -> None:
        if not isValid(self.target):
            return
        self.target.setVisible(True)

    def _sync_visibility(self) -> None:
        if not isValid(self.target):
            return
        if self._pressed_inside_target:
            self._outside_streak = 0
            self.target.setVisible(True)
            return
        inside = self._is_cursor_inside_owner_or_target()
        if self._must_leave_once:
            if inside:
                self._outside_streak = 0
                self.target.setVisible(False)
                return
            self._must_leave_once = False
        if inside:
            self._outside_streak = 0
            self.target.setVisible(True)
            return
        self._outside_streak += 1
        if self._outside_streak >= 4:
            self.target.setVisible(False)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        et = event.type()
        if et in (QEvent.Enter, QEvent.HoverEnter, QEvent.MouseMove, QEvent.Show):
            self._sync_visibility()
        elif et == QEvent.MouseButtonPress:
            if watched is self.target or (isinstance(watched, QWidget) and self.target.isAncestorOf(watched)):
                self._pressed_inside_target = True
                self._show()
        elif et == QEvent.MouseButtonRelease:
            self._pressed_inside_target = False
            self._sync_visibility()
        return False


class _ButtonAnimFilter(QObject):
    def __init__(self, button: QPushButton, fade_ms: int = 170, hover_ms: int = 140, press_ms: int = 90):
        super().__init__(button)
        self.button = button
        self.fade_ms = max(50, int(fade_ms))
        self.hover_ms = max(50, int(hover_ms))
        self.press_ms = max(40, int(press_ms))
        self.idle_opacity = 0.94

        self.opacity = QGraphicsOpacityEffect(button)
        self.opacity.setOpacity(0.0)
        button.setGraphicsEffect(self.opacity)

        self.anim = QPropertyAnimation(self.opacity, b"opacity", button)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self._animate_to(self.idle_opacity, self.fade_ms)

    def set_durations(self, fade_ms: int, hover_ms: int, press_ms: int) -> None:
        self.fade_ms = max(50, int(fade_ms))
        self.hover_ms = max(50, int(hover_ms))
        self.press_ms = max(40, int(press_ms))

    def _animate_to(self, value: float, duration_ms: int) -> None:
        if not isValid(self.button):
            return
        self.anim.stop()
        self.anim.setDuration(int(duration_ms))
        self.anim.setStartValue(self.opacity.opacity())
        self.anim.setEndValue(float(value))
        self.anim.start()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if not isValid(self.button):
            return False
        et = event.type()
        if et == QEvent.Show:
            self._animate_to(self.idle_opacity, self.fade_ms)
        elif et == QEvent.Enter:
            self._animate_to(1.0, self.hover_ms)
        elif et == QEvent.Leave:
            self._animate_to(self.idle_opacity, self.hover_ms)
        elif et == QEvent.MouseButtonPress:
            self._animate_to(0.78, self.press_ms)
        elif et == QEvent.MouseButtonRelease:
            self._animate_to(1.0 if self.button.underMouse() else self.idle_opacity, self.press_ms)
        return False


class _FeedbackVisualFilter(QObject):
    def __init__(
        self,
        button: QPushButton,
        base_bg: str,
        hover_bg: str,
        active_bg: str,
        text_color: str,
        border_color: str,
        radius_px: int,
        font_size_px: int,
        font_weight: int,
        pad_y_px: int,
        pad_x_px: int,
    ):
        super().__init__(button)
        self.button = button
        self.base_bg = base_bg
        self.hover_bg = hover_bg
        self.active_bg = active_bg
        self.text_color = text_color
        self.border_color = border_color
        self.radius_px = int(radius_px)
        self.font_size_px = int(font_size_px)
        self.font_weight = int(font_weight)
        self.pad_y_px = int(pad_y_px)
        self.pad_x_px = int(pad_x_px)
        self.button.installEventFilter(self)
        self.refresh()

    def _compose_qss(self, bg: str) -> str:
        return (
            "QPushButton {"
            f"background: {bg};"
            f"color: {self.text_color};"
            f"border: 1px solid {self.border_color};"
            f"border-radius: {self.radius_px}px;"
            f"font-size: {self.font_size_px}px;"
            f"font-weight: {self.font_weight};"
            f"padding: {self.pad_y_px}px {self.pad_x_px}px;"
            "}"
        )

    def refresh(self) -> None:
        if not isValid(self.button):
            return
        selected = bool(self.button.property("selected"))
        if selected:
            bg = self.active_bg
        elif self.button.underMouse():
            bg = self.hover_bg
        else:
            bg = self.base_bg
        self.button.setStyleSheet(self._compose_qss(bg))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        et = event.type()
        if et in (
            QEvent.Enter,
            QEvent.Leave,
            QEvent.MouseMove,
            QEvent.MouseButtonPress,
            QEvent.MouseButtonRelease,
            QEvent.Show,
            QEvent.DynamicPropertyChange,
        ):
            self.refresh()
        return False
