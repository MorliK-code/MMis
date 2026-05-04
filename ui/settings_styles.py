from __future__ import annotations

from pathlib import Path

_CHEVRON_DOWN = (Path(__file__).resolve().parent / "assets" / "settings_chevron_down.svg").as_posix()
_CHECK_ICON = (Path(__file__).resolve().parent / "assets" / "settings_check.svg").as_posix()


SETTINGS_TOOLTIP_STYLE = """
/* MMIS_SETTINGS_TOOLTIP_STYLE */
QToolTip {
    color: #eef0f6;
    background-color: rgba(15, 16, 24, 248);
    border: 1px solid rgba(139, 92, 246, 82);
    border-radius: 8px;
    padding: 6px 8px;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 10px;
}
"""


def apply_settings_tooltip_style() -> None:
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            return

        current = app.styleSheet() or ""
        marker = "/* MMIS_SETTINGS_TOOLTIP_STYLE */"

        if marker in current:
            return

        app.setStyleSheet(current + "\n" + SETTINGS_TOOLTIP_STYLE)
    except Exception:
        pass


SETTINGS_STYLE = """
QDialog#settings_window {
    background: transparent;
}
QFrame#settings_overlay {
    background: rgba(5, 6, 9, 118);
}
QFrame#settings_panel {
    background: rgba(13, 15, 20, 238);
    border: 1px solid rgba(255, 255, 255, 34);
    border-radius: 17px;
}
QFrame#settings_top {
    background: rgba(10, 11, 13, 72);
    border-bottom: 1px solid rgba(255, 255, 255, 18);
}
QLabel#settings_title {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 15px;
    font-weight: 700;
    min-height: 22px;
    padding: 0px;
    margin: 0px;
}
QLabel#settings_gear {
    min-width: 32px;
    min-height: 32px;
    max-width: 32px;
    max-height: 32px;
    border-radius: 10px;
    border: 1px solid rgba(139, 92, 246, 54);
    background: rgba(139, 92, 246, 30);
    color: #c4b5fd;
    font-size: 16px;
    font-weight: 700;
}
QLabel#settings_subtitle {
    color: #8f96a3;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 10px;
    min-height: 17px;
    padding: 0px;
    margin: 0px;
}
QLabel#settings_muted {
    color: #8f96a3;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
}
QLabel#api_key_table_text {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
    font-weight: 600;
}
QLabel#api_key_table_muted {
    color: #8f96a3;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
    font-weight: 600;
}
QWidget#settings_center {
    background: rgba(10, 12, 16, 34);
}
QWidget#settings_content {
    background: transparent;
}
QLineEdit#settings_search {
    min-height: 30px;
    border-radius: 9px;
    border: 1px solid rgba(255, 255, 255, 18);
    background: rgba(18, 21, 27, 225);
    color: #f3f4f6;
    padding: 0 12px;
}
QPushButton, QToolButton {
    min-height: 24px;
    max-height: 24px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 18);
    background: rgba(255, 255, 255, 8);
    color: #c4b5fd;
    padding: 0 10px;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
}
QToolButton#close_button {
    min-width: 24px;
    max-width: 24px;
    min-height: 24px;
    max-height: 24px;
    border-radius: 8px;
    color: #8f96a3;
    font-size: 13px;
    padding: 0;
}
QToolButton#hint_button {
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    border: 0;
    background: transparent;
    padding: 0;
    margin: 0;
}
QFrame#settings_hint_popup {
    background: rgba(15, 16, 24, 242);
    border: 1px solid rgba(139, 92, 246, 70);
    border-radius: 10px;
}
QLabel#hint_popup_title {
    color: #c4b5fd;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 10px;
    font-weight: 700;
}
QLabel#hint_popup_body {
    color: #d8dee9;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 10px;
    line-height: 130%;
}
QLabel#hint_popup_example {
    color: #f3f4f6;
    background: rgba(139, 92, 246, 24);
    border-top: 1px solid rgba(139, 92, 246, 34);
    border-radius: 7px;
    padding: 5px 7px;
    font-family: Consolas, Cascadia Code, monospace;
    font-size: 9px;
}
QPushButton:hover, QToolButton:hover {
    background: rgba(139, 92, 246, 28);
    border-color: rgba(139, 92, 246, 54);
    color: #f3f4f6;
}
QPushButton:pressed, QToolButton:pressed {
    background: rgba(139, 92, 246, 44);
    border-color: rgba(139, 92, 246, 82);
    color: #ffffff;
    padding-top: 1px;
    padding-bottom: 0px;
}
QPushButton#primary_button {
    background: rgba(139, 92, 246, 42);
    border: 1px solid rgba(139, 92, 246, 70);
    color: #ede9fe;
}
QPushButton#primary_button:hover {
    background: rgba(139, 92, 246, 70);
    border: 1px solid rgba(167, 139, 250, 112);
    color: #ffffff;
}
QPushButton#primary_button:pressed {
    background: rgba(109, 40, 217, 95);
    border: 1px solid rgba(196, 181, 253, 140);
    color: #f5f3ff;
    padding-top: 1px;
    padding-bottom: 0px;
}
QPushButton#primary_button:disabled {
    background: rgba(139, 92, 246, 18);
    border: 1px solid rgba(139, 92, 246, 34);
    color: rgba(237, 233, 254, 90);
}
QPushButton#nav_button {
    min-height: 28px;
    max-height: 28px;
    text-align: left;
    padding: 0 9px;
    color: #8f96a3;
    border-color: transparent;
    background: transparent;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 12px;
}
QPushButton#nav_button:checked {
    background: rgba(139, 92, 246, 30);
    border-color: rgba(139, 92, 246, 48);
    color: #f3f4f6;
}
QFrame#settings_nav {
    border-right: 1px solid rgba(255, 255, 255, 18);
    background: rgba(8, 10, 14, 58);
}
QScrollArea#settings_nav_scroll, QWidget#settings_nav_content {
    background: transparent;
}
QFrame#settings_card {
    background: rgba(255, 255, 255, 14);
    border: 1px solid rgba(255, 255, 255, 36);
    border-radius: 12px;
}
QFrame#setting_row {
    border: none;
}
QFrame#danger_card {
    background: rgba(127, 29, 29, 40);
    border: 1px solid rgba(252, 165, 165, 60);
    border-radius: 10px;
}
QLabel#card_title {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
}
QLabel#danger_title {
    color: #fca5a5;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
}
QLabel#setting_label {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
    font-weight: 600;
    padding: 0 4px;
    border: 1px solid transparent;
    border-radius: 6px;
}
QLabel#setting_label_changed {
    color: #c4b5fd;
    background: rgba(139, 92, 246, 28);
    border: 1px solid rgba(139, 92, 246, 52);
    border-radius: 6px;
    padding: 0 4px;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
    font-weight: 600;
}
QFrame#setting_row {
    background: transparent;
    border-top: 1px solid rgba(255, 255, 255, 10);
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
QLabel#updated_badge {
    color: #c4b5fd;
    background: rgba(139, 92, 246, 28);
    border: 1px solid rgba(139, 92, 246, 52);
    border-radius: 5px;
    padding: 1px 5px;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 9px;
}
QLabel#changed_badge {
    color: #c4b5fd;
    background: rgba(139, 92, 246, 28);
    border: 1px solid rgba(139, 92, 246, 52);
    border-radius: 5px;
    padding: 1px 5px;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 9px;
}
QLineEdit, QComboBox, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 26);
    background: rgba(8, 10, 14, 138);
    color: #f3f4f6;
    selection-background-color: rgba(139, 92, 246, 96);
    font-size: 11px;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: rgba(139, 92, 246, 112);
    background: rgba(7, 9, 13, 168);
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 26px;
    padding: 0 8px;
}
QPushButton#settings_value_preview {
    min-height: 22px;
    max-height: 22px;
    border-top-left-radius: 8px;
    border-bottom-left-radius: 8px;
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    border: 1px solid rgba(255, 255, 255, 22);
    border-right: 0;
    background: rgba(8, 10, 14, 108);
    color: #f3f4f6;
    text-align: left;
    padding: 0 7px 1px 7px;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 11px;
}
QPushButton#settings_value_preview:hover {
    background: rgba(8, 10, 14, 108);
    border-color: rgba(255, 255, 255, 22);
    color: #f3f4f6;
}
QToolButton#settings_value_edit_button {
    min-width: 26px;
    max-width: 26px;
    min-height: 22px;
    max-height: 22px;
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    border: 1px solid rgba(139, 92, 246, 46);
    background: rgba(139, 92, 246, 20);
    color: #c4b5fd;
    padding: 0;
    font-size: 12px;
}
QToolButton#settings_value_edit_button:hover {
    background: rgba(139, 92, 246, 36);
    border-color: rgba(139, 92, 246, 76);
    color: #f3f4f6;
}
QLineEdit#settings_numeric_input {
    min-height: 22px;
    max-height: 22px;
    border-top-left-radius: 8px;
    border-bottom-left-radius: 8px;
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    border: 1px solid rgba(255, 255, 255, 22);
    border-right: 0;
    background: rgba(8, 10, 14, 128);
    color: #f3f4f6;
    padding: 0 7px;
    font-size: 11px;
}
QLineEdit#settings_numeric_input:focus {
    border-color: rgba(139, 92, 246, 96);
    background: rgba(8, 10, 14, 168);
}
QLineEdit#settings_search:focus {
    border: 1px solid rgba(139, 92, 246, 82);
    background: rgba(139, 92, 246, 12);
}

/* Character Manager Styles */
QWidget#character_manager {
    background: transparent;
}
QScrollArea#character_list_scroll,
QScrollArea#character_editor_scroll,
QScrollArea#character_list_scroll > QWidget,
QScrollArea#character_editor_scroll > QWidget,
QWidget#character_list_container,
QWidget#character_editor_content {
    background: transparent;
    border: 0;
}
QFrame#character_list_header {
    border-bottom: 1px solid rgba(255, 255, 255, 14);
}
QFrame#character_list_header QLabel {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
}
QFrame#character_item, QFrame#character_item_active {
    background: rgba(255, 255, 255, 6);
    border: 1px solid rgba(255, 255, 255, 10);
    border-radius: 8px;
}
QFrame#character_item:hover {
    background: rgba(255, 255, 255, 12);
    border: 1px solid rgba(139, 92, 246, 40);
}
QFrame#character_item_active {
    background: rgba(139, 92, 246, 20);
    border: 1px solid rgba(139, 92, 246, 60);
}
QFrame#character_item[selected="true"], QFrame#character_item_active[selected="true"] {
    background: rgba(139, 92, 246, 30);
    border: 1px solid rgba(139, 92, 246, 86);
}
QFrame#character_item_active:hover {
    background: rgba(139, 92, 246, 30);
}
QLabel#character_item_name {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
}
QLabel#character_item_id {
    color: #8f96a3;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 10px;
}
QLabel#character_active_badge {
    background: rgba(139, 92, 246, 40);
    color: #c4b5fd;
    border: 1px solid rgba(139, 92, 246, 60);
    border-radius: 5px;
    padding: 1px 6px;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 9px;
    font-weight: 700;
}
QLabel#character_editor_title {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 14px;
    font-weight: 700;
    min-height: 24px;
}
QLabel#character_field_label {
    color: #8f96a3;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
}
QTabWidget#character_tabs {
    background: transparent;
    border: 0;
}
QTabWidget#character_tabs::pane {
    top: -1px;
    background: rgba(255, 255, 255, 8);
    border: 1px solid rgba(255, 255, 255, 24);
    border-radius: 10px;
}
QTabBar#character_tab_bar {
    background: transparent;
}
QTabBar#character_tab_bar::tab {
    min-height: 27px;
    padding: 0 13px;
    margin-right: 3px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 18);
    border-bottom-color: rgba(255, 255, 255, 24);
    background: rgba(255, 255, 255, 8);
    color: #8f96a3;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
}
QTabBar#character_tab_bar::tab:selected {
    background: rgba(139, 92, 246, 34);
    border-color: rgba(139, 92, 246, 70);
    color: #f3f4f6;
}
QTabBar#character_tab_bar::tab:hover {
    background: rgba(139, 92, 246, 22);
    color: #ede9fe;
}
QWidget#character_tab_page {
    background: rgba(10, 12, 16, 72);
    border: 0;
}
QPushButton#danger_button {
    color: #fca5a5;
    border-color: rgba(252, 165, 165, 38);
    background: rgba(127, 29, 29, 28);
}
QPushButton#danger_button:hover {
    color: #fee2e2;
    border-color: rgba(252, 165, 165, 72);
    background: rgba(127, 29, 29, 54);
}
QWidget#settings_numeric_stepper {
    background: rgba(139, 92, 246, 20);
    border: 1px solid rgba(139, 92, 246, 46);
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}
QToolButton#settings_numeric_step_button_up,
QToolButton#settings_numeric_step_button_down {
    min-width: 22px;
    max-width: 22px;
    min-height: 11px;
    max-height: 11px;
    border-radius: 0;
    border: 0;
    background: transparent;
    color: #c4b5fd;
    padding: 0;
    font-size: 8px;
    font-weight: 700;
}
QToolButton#settings_numeric_step_button_up {
    border-bottom: 1px solid rgba(139, 92, 246, 38);
}
QToolButton#settings_numeric_step_button_down {
    border: 0;
}
QToolButton#settings_numeric_step_button_up:hover,
QToolButton#settings_numeric_step_button_down:hover {
    background: rgba(139, 92, 246, 36);
    border-color: rgba(139, 92, 246, 76);
    color: #f3f4f6;
}
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
    line-height: 145%;
}
QPlainTextEdit#settings_text_editor_body {
    border-radius: 7px;
    border: 1px solid rgba(139, 92, 246, 62);
    background: rgba(31, 25, 51, 230);
    color: #f3f4f6;
    selection-background-color: rgba(139, 92, 246, 96);
    padding: 7px 9px;
    font-size: 11px;
}
QPlainTextEdit#settings_text_editor_body:focus {
    border-color: rgba(139, 92, 246, 118);
    background: rgba(33, 25, 58, 238);
}
QPlainTextEdit {
    padding: 7px;
}
QCheckBox {
    color: #f3f4f6;
    spacing: 8px;
    font-size: 11px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 38);
    background: rgba(8, 10, 14, 176);
}
QCheckBox::indicator:checked {
    background: rgba(139, 92, 246, 120);
    border-color: rgba(196, 181, 253, 110);
}
QCheckBox#api_key_enabled_checkbox {
    color: #d8dee9;
    spacing: 6px;
    font-size: 11px;
    font-weight: 600;
}
QCheckBox#api_key_enabled_checkbox::indicator {
    width: 13px;
    height: 13px;
}
QCheckBox#api_key_enabled_checkbox::indicator:checked {
    image: url(__CHECK_ICON__);
    background: rgba(139, 92, 246, 140);
    border-color: rgba(196, 181, 253, 130);
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid rgba(255, 255, 255, 22);
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background: rgba(255, 255, 255, 20);
}
QComboBox::down-arrow {
    image: url(__CHEVRON_DOWN__);
    width: 11px;
    height: 11px;
}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    border: 0;
    background: rgba(255, 255, 255, 20);
    width: 18px;
}
QComboBox QAbstractItemView {
    background: #101216;
    color: #f3f4f6;
    border: 1px solid rgba(139, 92, 246, 58);
    selection-background-color: rgba(139, 92, 246, 70);
}
QScrollArea, QScrollArea#settings_scroll, QScrollArea#settings_scroll > QWidget {
    border: 0;
    background: transparent;
}
QFrame#json_preview {
    color: #c4b5fd;
    background: rgba(139, 92, 246, 16);
    border: 1px solid rgba(139, 92, 246, 45);
    border-radius: 8px;
    font-family: Consolas, Cascadia Code, monospace;
    font-size: 10px;
}

QDialog#settings_message_box {
    background: transparent;
}

QFrame#settings_message_panel, QFrame#mmis_message_panel {
    background: rgba(15, 17, 24, 252);
    border: 1px solid rgba(139, 92, 246, 50);
    border-radius: 14px;
}
QLabel#settings_message_title, QLabel#mmis_message_title {
    color: #f3f4f6;
    font-size: 14px;
    font-weight: 700;
}
QLabel#settings_message_body, QLabel#mmis_message_body {
    color: #d1d5db;
    font-size: 12px;
}
QLabel#settings_message_icon, QLabel#mmis_message_icon_info {
    background: rgba(139, 92, 246, 30);
    color: #c4b5fd;
    border: 1px solid rgba(139, 92, 246, 50);
    border-radius: 15px;
    font-weight: 800;
    font-size: 14px;
}
QLabel#mmis_message_icon_warning {
    background: rgba(245, 158, 11, 20);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 40);
    border-radius: 15px;
    font-weight: 800;
    font-size: 14px;
}
QLabel#mmis_message_icon_critical {
    background: rgba(239, 68, 68, 20);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 40);
    border-radius: 15px;
    font-weight: 800;
    font-size: 14px;
}
QLabel#mmis_message_icon_question {
    background: rgba(16, 185, 129, 20);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 40);
    border-radius: 15px;
    font-weight: 800;
    font-size: 14px;
}
QToolButton#settings_message_close, QToolButton#mmis_message_close {
    background: transparent;
    color: #8f96a3;
}

QLabel#settings_message_title {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 13px;
    font-weight: 800;
}

QLabel#settings_message_body {
    color: #cbd5e1;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
    line-height: 1.35;
}
""".replace("__CHEVRON_DOWN__", _CHEVRON_DOWN).replace("__CHECK_ICON__", _CHECK_ICON)
