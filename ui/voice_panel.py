from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QRadialGradient
from PySide6.QtWidgets import QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

import ui.chat_shell as proto


class VoicePanel(QWidget):
    listenPressed = Signal()
    listenReleased = Signal()
    repeatRequested = Signal()
    stopRequested = Signal()
    closeRequested = Signal()
    fileRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("voice_panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._pulse_animation: QPropertyAnimation | None = None
        self._opacity_effect: QGraphicsOpacityEffect | None = None
        self._build_ui()
        self.set_state("idle")

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QWidget#voice_panel {
                background: transparent;
                color: #f3f4f6;
            }
            QFrame#voice_head {
                background: rgba(11, 13, 16, 0.22);
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }
            QLabel#voice_persona {
                color: #c4b5fd;
                font-size: 18px;
                font-weight: 900;
            }
            QFrame#voice_modes {
                background: rgba(11, 13, 16, 0.12);
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }
            QLabel#voice_chip {
                color: #8f96a3;
                font-size: 10px;
                padding: 5px 8px;
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                background: transparent;
            }
            QLabel#voice_chip_active {
                color: #f3f4f6;
                font-size: 10px;
                padding: 5px 8px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                background: rgba(18, 21, 26, 0.78);
            }
            QLabel#voice_note {
                color: #646b76;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QPushButton#voice_mic {
                color: #c4b5fd;
                background: transparent;
                border: none;
                font-size: 54px;
                font-weight: 800;
                min-width: 124px;
                min-height: 84px;
            }
            QPushButton#voice_mic:hover {
                color: #ddd6fe;
            }
            QLabel#voice_status {
                color: #f3f4f6;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#voice_status_sub {
                color: #8f96a3;
                font-size: 12px;
                line-height: 1.45;
            }
            QFrame#voice_line {
                background: rgba(255, 255, 255, 0.025);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
            }
            QLabel#voice_line_role {
                color: #f3f4f6;
                font-size: 13px;
                font-weight: 700;
            }
            QLabel#voice_line_text {
                color: #8f96a3;
                font-size: 13px;
            }
            QFrame#voice_composer_wrap {
                background: rgba(11, 13, 16, 0.18);
                border-top: 1px solid rgba(255, 255, 255, 0.06);
            }
            QFrame#voice_composer {
                background: rgba(15, 18, 22, 0.72);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 14px;
            }
            QPushButton {
                color: #8f96a3;
                background: rgba(16, 18, 22, 0.28);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 8px;
                min-height: 28px;
                padding: 0 9px;
                font-size: 11px;
            }
            QPushButton:hover {
                color: #f3f4f6;
                background: rgba(139, 92, 246, 0.10);
                border-color: rgba(139, 92, 246, 0.18);
            }
            QPushButton#voice_accent {
                color: rgba(196, 181, 253, 0.88);
            }
            QPushButton#voice_stop {
                color: #fca5a5;
                background: rgba(239, 68, 68, 0.12);
                border-color: rgba(239, 68, 68, 0.22);
            }
            QLabel#voice_hint {
                color: #646b76;
                font-size: 10px;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_head())
        root.addWidget(self._build_modes())

        voice_page = QWidget(self)
        voice_page.setObjectName("voice_page_body")
        voice_page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body = QVBoxLayout(voice_page)
        body.setContentsMargins(18, 24, 18, 24)
        body.setSpacing(14)
        body.addStretch(1)

        note = QLabel("ГОЛОСОВОЕ ОБЩЕНИЕ", voice_page)
        note.setObjectName("voice_note")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(note)

        mic_wrap = QWidget(voice_page)
        mic_lay = QHBoxLayout(mic_wrap)
        mic_lay.setContentsMargins(0, 0, 0, 0)
        mic_lay.addStretch(1)
        self.mic_button = QPushButton("\U0001f399", mic_wrap)
        self.mic_button.setObjectName("voice_mic")
        self.mic_button.setToolTip("Удерживай, чтобы говорить")
        self.mic_button.pressed.connect(self.listenPressed.emit)
        self.mic_button.released.connect(self.listenReleased.emit)
        self._opacity_effect = QGraphicsOpacityEffect(self.mic_button)
        self.mic_button.setGraphicsEffect(self._opacity_effect)
        mic_lay.addWidget(self.mic_button)
        mic_lay.addStretch(1)
        body.addWidget(mic_wrap)

        self.status_label = QLabel("", voice_page)
        self.status_label.setObjectName("voice_status")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(self.status_label)

        self.status_sub_label = QLabel("", voice_page)
        self.status_sub_label.setObjectName("voice_status_sub")
        self.status_sub_label.setWordWrap(True)
        self.status_sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_sub_label.setMaximumWidth(460)
        body.addWidget(self.status_sub_label, 0, Qt.AlignmentFlag.AlignHCenter)

        history = QWidget(voice_page)
        history.setMaximumWidth(560)
        history.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        history_lay = QVBoxLayout(history)
        history_lay.setContentsMargins(0, 4, 0, 0)
        history_lay.setSpacing(8)
        self.user_text = self._history_line(history_lay, "Ты")
        self.assistant_text = self._history_line(history_lay, "Ася")
        body.addWidget(history, 0, Qt.AlignmentFlag.AlignHCenter)
        body.addStretch(1)
        root.addWidget(voice_page, 1)

        root.addWidget(self._build_composer())

    def _build_head(self) -> QFrame:
        head = QFrame(self)
        head.setObjectName("voice_head")
        lay = QHBoxLayout(head)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(8)

        persona = QLabel("Ася", head)
        persona.setObjectName("voice_persona")
        font = QFont("Segoe Script")
        font.setPixelSize(18)
        font.setWeight(QFont.Weight.Black)
        persona.setFont(font)
        lay.addWidget(persona)
        lay.addStretch(1)

        back = QPushButton("В чат", head)
        back.setObjectName("voice_accent")
        stop = QPushButton("Стоп", head)
        stop.setObjectName("voice_stop")
        back.clicked.connect(self.closeRequested.emit)
        stop.clicked.connect(self.stopRequested.emit)
        lay.addWidget(back)
        lay.addWidget(stop)
        return head

    def _build_modes(self) -> QFrame:
        modes = QFrame(self)
        modes.setObjectName("voice_modes")
        lay = QHBoxLayout(modes)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(6)
        for idx, text in enumerate(("страница: voice", "режим: voice", "stt: on", "tts: on", "interrupt: on")):
            chip = QLabel(text, modes)
            chip.setObjectName("voice_chip_active" if idx == 0 else "voice_chip")
            lay.addWidget(chip)
        lay.addStretch(1)
        return modes

    def _build_composer(self) -> QFrame:
        wrap = QFrame(self)
        wrap.setObjectName("voice_composer_wrap")
        wrap_lay = QVBoxLayout(wrap)
        wrap_lay.setContentsMargins(14, 10, 14, 12)

        composer = QFrame(wrap)
        composer.setObjectName("voice_composer")
        row = QHBoxLayout(composer)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(8)

        back = QPushButton("<- чат", composer)
        back.clicked.connect(self.closeRequested.emit)
        file_button = QPushButton("Файл", composer)
        file_button.clicked.connect(self.fileRequested.emit)
        mic = QPushButton("\U0001f399", composer)
        mic.setObjectName("voice_accent")
        mic.setToolTip("Удерживай, чтобы говорить")
        mic.pressed.connect(self.listenPressed.emit)
        mic.released.connect(self.listenReleased.emit)
        functions = self._build_functions_button(composer)

        row.addWidget(back)
        row.addWidget(file_button)
        row.addWidget(mic)
        row.addWidget(functions)
        row.addStretch(1)

        hint = QLabel("voice page внутри текущего чата", composer)
        hint.setObjectName("voice_hint")
        row.addWidget(hint)

        repeat = QPushButton("Повторить", composer)
        repeat.clicked.connect(self.repeatRequested.emit)
        stop = QPushButton("Стоп", composer)
        stop.setObjectName("voice_stop")
        stop.clicked.connect(self.stopRequested.emit)
        row.addWidget(repeat)
        row.addWidget(stop)

        wrap_lay.addWidget(composer)
        return wrap

    def _build_functions_button(self, parent: QWidget) -> QToolButton:
        button = proto.HoverButton("Функции", accent=True)
        button.setParent(parent)
        button.clicked.connect(self._toggle_functions)
        self.functions_button = button
        self.functions_popup = self._build_functions_popup(button)
        return button

        button = QToolButton(parent)
        button.setObjectName("voice_accent")
        button.setText("Функции")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        menu = QMenu(button)
        for label, checked in (
            ("Авто-слушание", False),
            ("Прерывание TTS", True),
            ("Только текст", False),
            ("Всегда слушать", False),
        ):
            checkbox = QCheckBox(label, menu)
            checkbox.setChecked(checked)
            action = QWidgetAction(menu)
            action.setDefaultWidget(checkbox)
            menu.addAction(action)
        button.setMenu(menu)
        return button

    def _build_functions_popup(self, anchor: QWidget) -> proto.PopupFrame:
        popup = proto.PopupFrame(anchor, width=230, line_orientation="vertical")
        lay = QVBoxLayout(popup)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        title = proto.CrispLabel("Управление функциями")
        title.setFont(proto._ui_font(pixel_size=11, weight=QFont.Weight.Medium))
        title.set_text_color(proto.TEXT)
        lay.addWidget(title)

        commands = proto.HoverSubmenuRow(
            "команды",
            popup,
            icon_text="⌘",
            submenu_title="Команды",
            submenu_rows=[("think", "think", True), ("verbose", "verbose", True), ("json", "json", False)],
        )
        lay.addWidget(commands)

        screen_row = proto.function_row("screen", icon_text="▣")
        screen_row.layout().addWidget(proto.ToggleSwitch(False))
        lay.addWidget(screen_row)

        web_row = proto.function_row("web", icon_text="🌐")
        web_row.layout().setContentsMargins(10, 5, 8, 7)
        web_row.layout().addWidget(proto.WebModeSelector("auto"), 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(web_row)
        return popup

    def _build_functions_button(self, parent: QWidget) -> QWidget:
        button = proto.HoverButton("Функции", accent=True)
        button.setParent(parent)
        button.clicked.connect(self._toggle_functions)
        self.functions_button = button
        self.functions_popup = self._build_functions_popup(button)
        return button

    def _build_functions_popup(self, anchor: QWidget) -> proto.PopupFrame:
        popup = proto.PopupFrame(anchor, width=230, line_orientation="vertical")
        lay = QVBoxLayout(popup)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        code_font = proto._button_font(pixel_size=11, weight=QFont.Weight.Medium)

        title = proto.CrispLabel("Настройки голосового режима")
        title.setFont(code_font)
        title.set_text_color(proto.TEXT)
        lay.addWidget(title)

        self.voice_auto_listen_toggle = proto.ToggleSwitch(False)
        self.voice_barge_in_toggle = proto.ToggleSwitch(True)
        self.voice_text_only_toggle = proto.ToggleSwitch(False)
        self.voice_always_listen_toggle = proto.ToggleSwitch(False)

        for label, icon, toggle in (
            ("Авто-слушание", "🎙", self.voice_auto_listen_toggle),
            ("Прерывание TTS", "↯", self.voice_barge_in_toggle),
            ("Только текст", "T", self.voice_text_only_toggle),
            ("Всегда слушать", "∞", self.voice_always_listen_toggle),
        ):
            row = proto.function_row(label, icon_text=icon)
            self._apply_voice_function_font(row, code_font)
            row.layout().addWidget(toggle)
            lay.addWidget(row)
        return popup

    @staticmethod
    def _apply_voice_function_font(row: QFrame, font: QFont) -> None:
        for label in row.findChildren(proto.CrispLabel):
            label.setFont(font)

    def _toggle_functions(self) -> None:
        popup = getattr(self, "functions_popup", None)
        if popup is None:
            return
        popup.setFixedHeight(popup.sizeHint().height())
        if popup.isVisible():
            popup.close_popup()
        else:
            popup.open_above(x_offset=0, y_gap=12)

    def _history_line(self, parent_layout: QVBoxLayout, role: str) -> QLabel:
        frame = QFrame(self)
        frame.setObjectName("voice_line")
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        role_label = QLabel(f"{role}:", frame)
        role_label.setObjectName("voice_line_role")
        role_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        text_label = QLabel("-", frame)
        text_label.setObjectName("voice_line_text")
        text_label.setWordWrap(True)
        text_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(role_label, 0)
        lay.addWidget(text_label, 1)
        parent_layout.addWidget(frame)
        return text_label

    def set_state(self, state: str) -> None:
        label = str(state or "idle").strip().lower()
        if label == "listening":
            status = "слушаю..."
            sub = "Говори свободно. Это тот же чат, просто в голосовом режиме."
            self._set_mic_pulse(True, fast=False, opacity=1.0)
        elif label in {"processing", "thinking", "recognizing"}:
            status = "думаю..."
            sub = "Распознаю речь и собираю ответ."
            self._set_mic_pulse(False, opacity=0.82)
        elif label == "speaking":
            status = "говорю..."
            sub = "Ася сейчас озвучивает ответ."
            self._set_mic_pulse(True, fast=True, opacity=1.0)
        else:
            status = "ожидание"
            sub = "Нажми на микрофон, чтобы снова начать разговор."
            self._set_mic_pulse(False, opacity=0.62)
        self.status_label.setText(status)
        self.status_sub_label.setText(sub)

    def set_user_text(self, text: str) -> None:
        self.user_text.setText(str(text or "").strip() or "-")

    def set_assistant_text(self, text: str) -> None:
        self.assistant_text.setText(str(text or "").strip() or "-")

    def _set_mic_pulse(self, enabled: bool, *, fast: bool = False, opacity: float = 1.0) -> None:
        if self._opacity_effect is None:
            return
        if self._pulse_animation is not None:
            self._pulse_animation.stop()
            self._pulse_animation.deleteLater()
            self._pulse_animation = None
        self._opacity_effect.setOpacity(float(opacity))
        if not enabled:
            return
        anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        anim.setDuration(900 if fast else 1800)
        anim.setStartValue(0.68)
        anim.setKeyValueAt(0.5, 1.0)
        anim.setEndValue(0.68)
        anim.setLoopCount(-1)
        anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        anim.start()
        self._pulse_animation = anim

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor("#0a0b0d"))

        top_left = QRadialGradient(rect.left(), rect.top(), max(rect.width(), rect.height()) * 0.55)
        top_left.setColorAt(0.0, QColor(76, 29, 149, 86))
        top_left.setColorAt(1.0, QColor(76, 29, 149, 0))
        painter.fillRect(rect, top_left)

        top_right = QRadialGradient(rect.right(), rect.top(), max(rect.width(), rect.height()) * 0.45)
        top_right.setColorAt(0.0, QColor(59, 130, 246, 46))
        top_right.setColorAt(1.0, QColor(59, 130, 246, 0))
        painter.fillRect(rect, top_right)

        vertical = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        vertical.setColorAt(0.0, QColor(18, 15, 31, 82))
        vertical.setColorAt(0.38, QColor(10, 11, 13, 0))
        vertical.setColorAt(1.0, QColor(10, 11, 13, 0))
        painter.fillRect(rect, vertical)
        super().paintEvent(event)
