"""
Character Manager widget for MMis Settings.
"""

from __future__ import annotations

from typing import Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QScrollArea, QSizePolicy, QLineEdit, QPlainTextEdit, QLayout,
    QComboBox, QCheckBox, QDoubleSpinBox, QTabWidget
)
from ui.widgets.message_box import MmisMessageBox
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPainter, QPen, QColor
import json

from ui.api_client import ApiClient
from ui.workers import CharacterListWorker, CharacterActionWorker
from ui.chat_shell import ChatScrollOverlay, _ui_font, _to_qcolor


class CharacterItemWidget(QFrame):
    clicked = Signal(str)
    
    def __init__(self, char_id: str, name: str, is_active: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.char_id = char_id
        self.setObjectName("character_item_active" if is_active else "character_item")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(50)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #f3f4f6;")
        layout.addWidget(self.name_label)
        
        layout.addStretch()
        
        if is_active:
            self.active_badge = QLabel("АКТИВЕН")
            self.active_badge.setStyleSheet(\
                "background: rgba(139, 92, 246, 40); "
                "color: #c4b5fd; "
                "border-radius: 4px; "
                "padding: 2px 6px; "
                "font-size: 9px; "
                "font-weight: bold;"
            )
            layout.addWidget(self.active_badge)
            
        self.id_label = QLabel(f"@{char_id}")
        self.id_label.setStyleSheet("color: #8f96a3; font-size: 10px;")
        layout.addWidget(self.id_label)

    def mousePressEvent(self, event):
        self.clicked.emit(self.char_id)
        super().mousePressEvent(event)


class CharacterManager(QWidget):
    def __init__(self, api: ApiClient, parent: QWidget | None = None):
        super().__init__(parent)
        self.api = api
        self._chars = []
        self._active_id = ""
        self._selected_id = ""
        self._workers: list[CharacterActionWorker | CharacterListWorker] = []
        
        self.setup_ui()
        self.refresh()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Левая часть - список
        self.left_panel = QFrame()
        self.left_panel.setFixedWidth(280)
        self.left_panel.setObjectName("settings_card")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        list_header = QFrame()
        list_header.setFixedHeight(40)
        list_header.setStyleSheet("border-bottom: 1px solid rgba(255, 255, 255, 10);")
        h_layout = QHBoxLayout(list_header)
        h_layout.addWidget(QLabel("СПИСОК ПЕРСОНАЖЕЙ"))
        left_layout.addWidget(list_header)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background: transparent;")
        
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(4, 4, 4, 4)
        self.list_layout.setSpacing(4)
        self.list_layout.addStretch()
        
        self.scroll.setWidget(self.list_container)
        self._scroll_overlay = ChatScrollOverlay(self.scroll)
        left_layout.addWidget(self.scroll)
        
        self.create_btn = QPushButton("+ Создать персонажа")
        self.create_btn.setObjectName("primary_button")
        self.create_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.create_btn.clicked.connect(self._on_create_clicked)
        left_layout.addWidget(self.create_btn)
        
        layout.addWidget(self.left_panel)
        
        # Правая часть - редактор
        self.editor_panel = QFrame()
        self.editor_panel.setObjectName("settings_card")
        editor_panel_layout = QVBoxLayout(self.editor_panel)
        editor_panel_layout.setContentsMargins(0, 0, 0, 0)
        editor_panel_layout.setSpacing(0)

        self.editor_scroll = QScrollArea(self.editor_panel)
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.editor_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.editor_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.editor_scroll.setStyleSheet("background: transparent;")
        self.editor_scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.editor_content = QWidget()
        self.editor_layout = QVBoxLayout(self.editor_content)
        self.editor_layout.setContentsMargins(12, 12, 12, 12)
        self.editor_layout.setSpacing(10)

        self.empty_label = QLabel("Выберите персонажа для настройки или создайте нового.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #8f96a3;")
        self.editor_layout.addWidget(self.empty_label)

        self.editor_scroll.setWidget(self.editor_content)
        self.editor_overlay = ChatScrollOverlay(self.editor_scroll)
        editor_panel_layout.addWidget(self.editor_scroll)

        layout.addWidget(self.editor_panel, 1)
    def refresh(self):
        self._set_loading(True)
        worker = CharacterListWorker(self.api)
        worker.finished.connect(self._on_list_ready)
        worker.errored.connect(self._on_error)
        self._workers.append(worker)
        worker.start()

    def _set_loading(self, loading: bool):
        if hasattr(self, "left_panel"):
            self.left_panel.setEnabled(not loading)
        if hasattr(self, "editor_panel"):
            self.editor_panel.setEnabled(not loading)

    def _on_error(self, message: str):
        self._set_loading(False)
        MmisMessageBox.critical(self, "Ошибка", f"Не удалось выполнить действие:\n{message}")

    def _on_list_ready(self, data):
        self._set_loading(False)
        manifest = data.get("manifest", {})
        active_data = data.get("active", {})
        self._active_id = active_data.get("id", "")
        self._chars = manifest.get("characters", [])
        
        # Очистка списка
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        for char in self._chars:
            cid = char.get("id")
            name = char.get("name", cid)
            item = CharacterItemWidget(cid, name, is_active=(cid == self._active_id))
            item.clicked.connect(self._on_char_selected)
            self.list_layout.insertWidget(self.list_layout.count() - 1, item)
            
        if self._selected_id:
            self._on_char_selected(self._selected_id)

    def _on_create_clicked(self):
        # Упрощенный диалог создания
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Новый персонаж", "Введите имя персонажа:")
        if ok and name:
            cid = name.lower().replace(" ", "_")
            self._run_action("create", id=cid, name=name)

    def _clear_layout(self, layout: QLayout, *, keep_empty_label: bool = False) -> None:
        while layout.count():
            item = layout.takeAt(0)

            child_layout = item.layout()
            if child_layout is not None:
                self._clear_layout(child_layout, keep_empty_label=False)
                child_layout.deleteLater()
                continue

            widget = item.widget()
            if widget is not None:
                if keep_empty_label and widget is self.empty_label:
                    widget.setParent(None)
                    continue
                widget.deleteLater()

    def _line(self, value: str = "") -> QLineEdit:
        w = QLineEdit()
        w.setText("" if value is None else str(value))
        return w

    def _combo(self, values: list[str], current: str = "") -> QComboBox:
        w = QComboBox()
        w.addItems(values)
        if current and w.findText(current) < 0:
            w.addItem(current)
        idx = w.findText(current)
        w.setCurrentIndex(max(0, idx))
        return w

    def _check(self, value: bool = False) -> QCheckBox:
        w = QCheckBox()
        w.setChecked(bool(value))
        return w

    def _float(self, value: float = 0.0, minimum: float = -1.0, maximum: float = 1.0, step: float = 0.05) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(float(minimum), float(maximum))
        w.setSingleStep(float(step))
        w.setDecimals(3)
        try:
            w.setValue(float(value))
        except Exception:
            w.setValue(0.0)
        return w

    def _json_edit(self, value) -> QPlainTextEdit:
        w = QPlainTextEdit()
        w.setMinimumHeight(80)
        w.setPlainText(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return w

    def _field(self, layout: QVBoxLayout, label_text: str, widget: QWidget) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        lbl = QLabel(label_text)
        lbl.setFixedWidth(190)
        lbl.setStyleSheet("color: #8f96a3;")
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        layout.addLayout(row)
        return widget

    def _section(self, title: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("settings_card")
        box = QVBoxLayout(frame)
        box.setContentsMargins(10, 10, 10, 10)
        box.setSpacing(8)
        lbl = QLabel(title)
        lbl.setObjectName("card_title")
        box.addWidget(lbl)
        frame._box = box
        return frame

    def _parse_json_edit(self, widget: QPlainTextEdit, fallback):
        text = widget.toPlainText().strip()
        if not text:
            return fallback
        return json.loads(text)

    def _on_char_selected(self, char_id):
        self._selected_id = char_id
        # Обновляем визуальное выделение в списке
        for i in range(self.list_layout.count() - 1):
            w = self.list_layout.itemAt(i).widget()
            if isinstance(w, CharacterItemWidget):
                is_sel = w.char_id == char_id
                w.setStyleSheet("background: rgba(139, 92, 246, 30); border: 1px solid #8b5cf6;" if is_sel else "")

        self._show_editor(char_id)

    def _show_editor(self, char_id):
        self._clear_layout(self.editor_layout, keep_empty_label=True)
        self.empty_label.hide()

        self._current_payload = {}
        try:
            self._cleanup_workers()
            worker = CharacterActionWorker(self.api, "get", id=char_id)
            worker.finished.connect(lambda data: self._render_editor(char_id, data))
            worker.errored.connect(self._on_error)
            self._workers.append(worker)
            worker.start()
        except Exception as exc:
            self._on_error(str(exc))

    def _render_editor(self, char_id: str, payload: dict) -> None:
        self._clear_layout(self.editor_layout, keep_empty_label=True)
        self.empty_label.hide()

        self._current_payload = dict(payload or {})
        character = dict(payload.get("character") or {})
        state = dict(payload.get("state") or {})
        persona_state = dict(payload.get("persona_state") or {})
        persona_spec = dict(payload.get("persona_spec") or {})
        emotion_state = dict(payload.get("emotion_state") or {})
        user_addressing = dict(payload.get("user_addressing") or {})
        identity_core = dict(payload.get("identity_core") or {})
        builtin_traits = dict(payload.get("builtin_traits") or {})
        learned_traits = dict(payload.get("learned_traits") or {})
        rules = dict(payload.get("rules") or {})

        title = QLabel(f"Настройка: {character.get('name', char_id)}  @{char_id}")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f3f4f6; margin-bottom: 6px;")
        self.editor_layout.addWidget(title)

        tabs = QTabWidget()
        self.editor_layout.addWidget(tabs)

        self._char_widgets = {}

        # TAB 1 — Основное
        main = QWidget()
        main_l = QVBoxLayout(main)
        tabs.addTab(main, "Основное")

        sec = self._section("character.json")
        main_l.addWidget(sec)
        box = sec._box

        self._char_widgets["character.name"] = self._field(box, "name", self._line(character.get("name", "")))
        self._char_widgets["character.version"] = self._field(box, "version", self._line(character.get("version", "1.0.0")))
        self._char_widgets["character.default_mood"] = self._field(box, "default_mood", self._line(character.get("default_mood", "thoughtful")))
        self._char_widgets["character.llm_profile"] = self._field(box, "llm_profile", self._combo(["FAST", "BALANCED", "QUALITY", "ECONOM"], str(character.get("llm_profile") or "BALANCED").upper()))
        self._char_widgets["character.performance"] = self._field(box, "performance", self._combo(["FAST", "BALANCED", "QUALITY", "ECONOM"], str(character.get("performance") or character.get("llm_profile") or "BALANCED").upper()))
        self._char_widgets["character.default_mode"] = self._field(box, "default_mode", self._combo(["chatting", "helper", "engineer", "debugger", "planner", "voice"], str(character.get("default_mode") or "chatting")))
        self._char_widgets["character.voice_style"] = self._field(box, "voice_style", self._combo(["neutral", "warm", "soft", "strict", "playful"], str(character.get("voice_style") or "neutral")))
        self._char_widgets["character.system_prompt"] = self._field(box, "system_prompt", self._json_edit(character.get("system_prompt", "")))
        self._char_widgets["character.style_prompt"] = self._field(box, "style_prompt", self._json_edit(character.get("style_prompt", "")))
        self._char_widgets["character.rules_prompt"] = self._field(box, "rules_prompt", self._json_edit(character.get("rules_prompt", "")))
        self._char_widgets["character.locks"] = self._field(box, "locks", self._json_edit(character.get("locks", {})))
        main_l.addStretch()

        # TAB 2 — Runtime state
        runtime = QWidget()
        runtime_l = QVBoxLayout(runtime)
        tabs.addTab(runtime, "Runtime")

        sec = self._section("state.json")
        runtime_l.addWidget(sec)
        box = sec._box
        self._char_widgets["state.mood"] = self._field(box, "mood", self._line(state.get("mood", "neutral")))
        self._char_widgets["state.active_traits"] = self._field(box, "active_traits", self._json_edit(state.get("active_traits", [])))
        self._char_widgets["state.disabled_traits"] = self._field(box, "disabled_traits", self._json_edit(state.get("disabled_traits", [])))
        self._char_widgets["state.counters"] = self._field(box, "counters", self._json_edit(state.get("counters", {})))
        self._char_widgets["state.last_signals"] = self._field(box, "last_signals", self._json_edit(state.get("last_signals", {})))
        self._char_widgets["state.applied_rules"] = self._field(box, "applied_rules", self._json_edit(state.get("applied_rules", [])))

        sec = self._section("emotion_state.json")
        runtime_l.addWidget(sec)
        box = sec._box
        self._char_widgets["emotion_state.mood"] = self._field(box, "mood", self._line(emotion_state.get("mood", "neutral")))
        self._char_widgets["emotion_state.valence"] = self._field(box, "valence", self._float(emotion_state.get("valence", 0.0)))
        self._char_widgets["emotion_state.arousal"] = self._field(box, "arousal", self._float(emotion_state.get("arousal", 0.0), 0.0, 1.0))
        self._char_widgets["emotion_state.intensity"] = self._field(box, "intensity", self._float(emotion_state.get("intensity", 0.0), 0.0, 1.0))
        self._char_widgets["emotion_state.trigger"] = self._field(box, "trigger", self._line(emotion_state.get("trigger", "")))
        self._char_widgets["emotion_state.cooldown_until_ts"] = self._field(box, "cooldown_until_ts", self._line(emotion_state.get("cooldown_until_ts", "")))
        runtime_l.addStretch()

        # TAB 3 — Persona
        persona = QWidget()
        persona_l = QVBoxLayout(persona)
        tabs.addTab(persona, "Persona")

        sec = self._section("persona_state.json")
        persona_l.addWidget(sec)
        box = sec._box
        self._char_widgets["persona_state.traits"] = self._field(box, "traits", self._json_edit(persona_state.get("traits", {})))
        self._char_widgets["persona_state.mood"] = self._field(box, "mood", self._line(persona_state.get("mood", "thoughtful")))
        self._char_widgets["persona_state.locks"] = self._field(box, "locks", self._json_edit(persona_state.get("locks", {})))
        self._char_widgets["persona_state.relation_state"] = self._field(box, "relation_state", self._json_edit(persona_state.get("relation_state", {})))
        self._char_widgets["persona_state.bans"] = self._field(box, "bans", self._json_edit(persona_state.get("bans", [])))
        self._char_widgets["persona_state.learned"] = self._field(box, "learned", self._json_edit(persona_state.get("learned", {})))
        self._char_widgets["persona_state.stabilizer"] = self._field(box, "stabilizer", self._json_edit(persona_state.get("stabilizer", {})))

        sec = self._section("persona_spec.json")
        persona_l.addWidget(sec)
        box = sec._box
        self._char_widgets["persona_spec.identity"] = self._field(box, "identity", self._json_edit(persona_spec.get("identity", [])))
        self._char_widgets["persona_spec.locks_map"] = self._field(box, "locks_map", self._json_edit(persona_spec.get("locks_map", {})))
        self._char_widgets["persona_spec.bans_template"] = self._field(box, "bans_template", self._line(persona_spec.get("bans_template", "")))
        self._char_widgets["persona_spec.moods"] = self._field(box, "moods", self._json_edit(persona_spec.get("moods", {})))
        self._char_widgets["persona_spec.modes"] = self._field(box, "modes", self._json_edit(persona_spec.get("modes", {})))
        self._char_widgets["persona_spec.trait_order"] = self._field(box, "trait_order", self._json_edit(persona_spec.get("trait_order", [])))
        self._char_widgets["persona_spec.traits_rules"] = self._field(box, "traits_rules", self._json_edit(persona_spec.get("traits_rules", {})))
        persona_l.addStretch()

        # TAB 4 — Identity
        identity = QWidget()
        identity_l = QVBoxLayout(identity)
        tabs.addTab(identity, "Identity")

        sec = self._section("user_addressing.json")
        identity_l.addWidget(sec)
        box = sec._box
        self._char_widgets["user_addressing.canonical_name"] = self._field(box, "canonical_name", self._line(user_addressing.get("canonical_name", "")))
        self._char_widgets["user_addressing.allowed_forms"] = self._field(box, "allowed_forms", self._json_edit(user_addressing.get("allowed_forms", [])))
        self._char_widgets["user_addressing.forbidden_forms"] = self._field(box, "forbidden_forms", self._json_edit(user_addressing.get("forbidden_forms", [])))
        self._char_widgets["user_addressing.allow_diminutives"] = self._field(box, "allow_diminutives", self._check(user_addressing.get("allow_diminutives", False)))
        self._char_widgets["user_addressing.use_name_by_default"] = self._field(box, "use_name_by_default", self._check(user_addressing.get("use_name_by_default", False)))

        sec = self._section("identity_core.json")
        identity_l.addWidget(sec)
        box = sec._box
        self._char_widgets["identity_core.interaction_style"] = self._field(box, "interaction_style", self._json_edit(identity_core.get("interaction_style", {})))
        self._char_widgets["identity_core.boundaries"] = self._field(box, "boundaries", self._json_edit(identity_core.get("boundaries", {})))
        self._char_widgets["identity_core.emotional_handling"] = self._field(box, "emotional_handling", self._json_edit(identity_core.get("emotional_handling", {})))
        self._char_widgets["identity_core.assistant_trait_baseline"] = self._field(box, "assistant_trait_baseline", self._json_edit(identity_core.get("assistant_trait_baseline", {})))
        identity_l.addStretch()

        # TAB 5 — Traits & Rules
        advanced = QWidget()
        advanced_l = QVBoxLayout(advanced)
        tabs.addTab(advanced, "Traits / Rules")

        sec = self._section("traits/builtin.json")
        advanced_l.addWidget(sec)
        self._char_widgets["builtin_traits"] = self._field(sec._box, "builtin_traits", self._json_edit(builtin_traits))

        sec = self._section("traits/learned.json")
        advanced_l.addWidget(sec)
        self._char_widgets["learned_traits"] = self._field(sec._box, "learned_traits", self._json_edit(learned_traits))

        sec = self._section("rules/evolution.json")
        advanced_l.addWidget(sec)
        self._char_widgets["rules"] = self._field(sec._box, "rules", self._json_edit(rules))
        advanced_l.addStretch()

        # Кнопки
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 8, 0, 0)

        if char_id != self._active_id:
            activate_btn = QPushButton("Сделать активным")
            activate_btn.setObjectName("primary_button")
            activate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            activate_btn.clicked.connect(lambda: self._run_action("set_active", id=char_id))
            btn_layout.addWidget(activate_btn)

        save_btn = QPushButton("Сохранить изменения")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._on_save_clicked)
        btn_layout.addWidget(save_btn)

        if char_id not in {"asya", "default"}:
            delete_btn = QPushButton("Удалить")
            delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            delete_btn.setStyleSheet("color: #ef4444;")
            delete_btn.clicked.connect(lambda: self._run_action("delete", id=char_id))
            btn_layout.addWidget(delete_btn)

        self.editor_layout.addLayout(btn_layout)

    def _widget_value(self, widget):
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QPlainTextEdit):
            return self._parse_json_edit(widget, widget.toPlainText())
        return None

    def _set_nested(self, payload: dict, dotted_path: str, value) -> None:
        parts = dotted_path.split(".")
        cur = payload
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    def _on_save_clicked(self):
        updates: dict = {
            "character": {},
            "state": {},
            "persona_state": {},
            "persona_spec": {},
            "emotion_state": {},
            "user_addressing": {},
            "identity_core": {},
        }

        try:
            for path, widget in dict(getattr(self, "_char_widgets", {})).items():
                value = self._widget_value(widget)

                if path in {"builtin_traits", "learned_traits", "rules"}:
                    updates[path] = value
                    continue

                self._set_nested(updates, path, value)

        except Exception as exc:
            MmisMessageBox.critical(self, "Ошибка", f"Некорректный JSON в поле:\n{exc}")
            return

        self._run_action("update", id=self._selected_id, updates=updates)

    def _run_action(self, action, **kwargs):
        self._set_loading(True)
        worker = CharacterActionWorker(self.api, action, **kwargs)

        if action == "update":
            worker.finished.connect(lambda _result: self._on_updated(kwargs.get("id")))
        else:
            worker.finished.connect(self.refresh)

        worker.errored.connect(self._on_error)
        self._workers.append(worker)
        worker.start()

    def _on_updated(self, char_id: str) -> None:
        self._set_loading(False)
        self.refresh()
        self._selected_id = char_id

    def _cleanup_workers(self) -> None:
        for w in list(self._workers):
            if w.isRunning():
                try:
                    w.disconnect()
                    w.terminate()
                    w.wait(500)
                except Exception:
                    pass
            if w in self._workers:
                self._workers.remove(w)

    def deleteLater(self) -> None:
        self._cleanup_workers()
        super().deleteLater()
