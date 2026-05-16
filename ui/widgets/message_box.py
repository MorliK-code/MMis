"""
Custom Styled Message Box for MMis.
"""

from __future__ import annotations

from typing import Literal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QToolButton, QWidget, QTextEdit, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from ui.chat_shell import _ui_font


class MmisMessageBox(QDialog):
    """
    Универсальное окно уведомлений в стиле MMis.
    Заменяет стандартный QMessageBox.
    """
    
    Yes = 1
    No = 0
    
    def __init__(
        self, 
        parent: QWidget | None, 
        title: str, 
        text: str, 
        *, 
        kind: Literal["info", "warning", "critical", "question"] = "info",
        buttons: list[str] | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setObjectName("mmis_message_box")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedWidth(430)
        
        # Result mapping for buttons
        self._button_results = {}
        
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        
        panel = QFrame(self)
        panel.setObjectName("mmis_message_panel")
        root.addWidget(panel)
        
        body_root = QVBoxLayout(panel)
        body_root.setContentsMargins(18, 16, 18, 16)
        body_root.setSpacing(14)
        
        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(12)
        
        icon_text = "i"
        if kind == "warning": icon_text = "!"
        elif kind == "critical": icon_text = "!!"
        elif kind == "question": icon_text = "?"
        
        self.icon = QLabel(icon_text)
        self.icon.setObjectName(f"mmis_message_icon_{kind}")
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon.setFixedSize(32, 32)
        header.addWidget(self.icon)
        
        self.title_label = QLabel(title)
        self.title_label.setObjectName("mmis_message_title")
        self.title_label.setFont(_ui_font(pixel_size=13, bold=True))
        header.addWidget(self.title_label, 1)
        
        close_btn = QToolButton()
        close_btn.setObjectName("mmis_message_close")
        close_btn.setText("×")
        close_btn.clicked.connect(self.reject)
        header.addWidget(close_btn)
        
        body_root.addLayout(header)
        
        # Body text
        self.body = QTextEdit()
        self.body.setObjectName("mmis_message_body")
        self.body.setReadOnly(True)
        self.body.setPlainText(str(text or ""))
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setFont(_ui_font(pixel_size=12))
        self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.body.setMinimumHeight(58)
        self.body.setMaximumHeight(170)
        doc_height = int(self.body.document().size().height()) + 12
        self.body.setFixedHeight(max(58, min(170, doc_height)))
        self.body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        body_root.addWidget(self.body)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 4, 0, 0)
        button_layout.addStretch()
        
        if buttons is None:
            if kind == "question":
                buttons = ["Да", "Нет"]
            else:
                buttons = ["OK"]
                
        for i, btn_text in enumerate(buttons):
            btn = QPushButton(btn_text)
            if i == 0:
                btn.setObjectName("primary_button")
            else:
                btn.setObjectName("secondary_button")
            
            # Use a closure to capture btn_text
            def make_callback(val):
                return lambda: self.done_with_result(val)
                
            btn.clicked.connect(make_callback(btn_text))
            button_layout.addWidget(btn)
            
        body_root.addLayout(button_layout)
        
        self._result_text = ""

    def done_with_result(self, text: str):
        self._result_text = text
        is_yes = text.lower() in {"ok", "yes", "да"}
        self.setResult(self.Yes if is_yes else self.No)
        if is_yes:
            self.accept()
        else:
            self.reject()

    def result_text(self) -> str:
        return self._result_text

    @staticmethod
    def information(parent: QWidget | None, title: str, text: str):
        MmisMessageBox(parent, title, text, kind="info").exec()

    @staticmethod
    def warning(parent: QWidget | None, title: str, text: str):
        MmisMessageBox(parent, title, text, kind="warning").exec()

    @staticmethod
    def critical(parent: QWidget | None, title: str, text: str):
        MmisMessageBox(parent, title, text, kind="critical").exec()

    @staticmethod
    def question(parent: QWidget | None, title: str, text: str, buttons: list[str] | None = None) -> int:
        dlg = MmisMessageBox(parent, title, text, kind="question", buttons=buttons)
        return dlg.exec()
