# MMis — фикс выбора модели в настройках

## Что исправляем

В `ui/settings_widgets.py` надо заменить текущий селектор модели в настройках на отдельный `ModelNameEditor`.

Проблемы сейчас:

1. **Углы выходят за грани** — popup рисуется как обычный `QFrame` со stylesheet `border-radius`, но у top-level popup Qt иногда оставляет прямоугольный фон/дочерние кнопки визуально залезают в скругления.
2. **Очень медленно открывается** — список моделей запрашивается синхронно при открытии popup, из-за этого UI ждёт API/Ollama.
3. **Нет всех моделей** — список надо брать из `/models`, нормально парсить `models`, кешировать и не заменять его fallback-списком.

---

## 1. Файл

Править:

```text
ui/settings_widgets.py
```

---

## 2. Исправить imports

Сейчас вверху есть примерно так:

```python
from PySide6.QtCore import Signal, Qt, QSize, QRect, QRectF, QTimer, QEvent
```

Замени на:

```python
from PySide6.QtCore import Signal, Qt, QSize, QRect, QRectF, QTimer, QEvent, QThread
```

Сейчас импорт из `ui.chat_shell` такой:

```python
from ui.chat_shell import ToggleSwitch, _ui_font, _to_qcolor, TEXT, MUTED, PlainTextScrollOverlay
```

Замени на:

```python
from ui.chat_shell import (
    ToggleSwitch,
    PaintedButton,
    _ui_font,
    _to_qcolor,
    TEXT,
    MUTED,
    LINE,
    PlainTextScrollOverlay,
)
```

---

## 3. Добавить style для popup моделей

Ниже `_TEXT_EDITOR_POPUP_STYLE` добавь:

```python
_MODEL_SELECTOR_POPUP_STYLE = _TEXT_EDITOR_POPUP_STYLE + """
QFrame#settings_model_selector_popup {
    background: rgba(15, 16, 24, 248);
    border: 1px solid rgba(139, 92, 246, 82);
    border-radius: 12px;
}
QLabel#settings_model_selector_status {
    color: #8f96a3;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 10px;
}
"""
```

---

## 4. Добавить helper для парсинга моделей

Ниже `RestartBadge` или перед `TextEditorPopup` добавь:

```python
def _extract_model_names(raw_models: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for item in list(raw_models or []):
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("model") or item.get("id") or "").strip()
        else:
            name = str(
                getattr(item, "name", "")
                or getattr(item, "model", "")
                or getattr(item, "id", "")
                or ""
            ).strip()

        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)

    return out
```

Это нужно, чтобы не терялись модели, если API/Ollama вернёт не строки, а объекты/словари.

---

## 5. Добавить фоновый worker для загрузки моделей

Ниже helper-а добавь:

```python
class ModelListWorker(QThread):
    loaded = Signal(str, list)
    failed = Signal(str)

    def __init__(self, current: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.current = str(current or "").strip()

    def run(self) -> None:
        try:
            from ui.api_client import ApiClient

            payload = ApiClient().list_models(timeout=2.5)
            runtime = str(payload.get("runtime_model") or self.current or "").strip()
            models = _extract_model_names(payload.get("models"))

            if runtime and runtime not in models:
                models.insert(0, runtime)
            if self.current and self.current not in models:
                models.insert(0, self.current)

            self.loaded.emit(runtime or self.current, models)
        except Exception as exc:
            self.failed.emit(str(exc))
```

Главное: **не вызывать `ApiClient().list_models()` прямо при клике**. Иначе popup будет открываться медленно.

---

## 6. Добавить popup моделей

Ниже `ModelListWorker` добавь:

```python
class ModelSelectorPopup(QFrame):
    modelSelected = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("settings_model_selector_popup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet(_MODEL_SELECTOR_POPUP_STYLE)
        self.setMinimumWidth(240)
        self.setMaximumHeight(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = QLabel("Модели")
        self.title_label.setObjectName("settings_text_editor_title")
        layout.addWidget(self.title_label)

        self.status_label = QLabel("")
        self.status_label.setObjectName("settings_model_selector_status")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        self.list_wrap = QWidget(self)
        self.list_layout = QVBoxLayout(self.list_wrap)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(7)
        layout.addWidget(self.list_wrap)

    def set_status(self, text: str) -> None:
        text = str(text or "").strip()
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))

    def set_models(self, current: str, models: list[str]) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        current = str(current or "").strip()
        models = _extract_model_names(models)
        if current and current not in models:
            models.insert(0, current)
        if not models:
            models = [current or "qwen3:8b"]

        for model in models:
            active = model == current
            btn = PaintedButton(f"{model}{'    ✓' if active else ''}", self)
            btn.set_button_font(_ui_font(pixel_size=12))
            btn.set_button_padding(8, 6, 8, 6)
            btn.set_button_radius(8)
            btn.set_text_alignment(Qt.AlignmentFlag.AlignLeft)
            btn.configure_colors(
                normal_bg="rgba(16,18,22,.28)",
                normal_border=LINE,
                normal_text=MUTED,
                hover_bg="rgba(139,92,246,.10)",
                hover_border="rgba(139,92,246,.18)",
                hover_text=TEXT,
                active_bg="rgba(139,92,246,.18)",
                active_border="rgba(139,92,246,.24)",
                active_text=TEXT,
            )
            btn.set_active(active)
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda _checked=False, name=model: self._select_model(name))
            self.list_layout.addWidget(btn)

        self.adjustSize()

    def _select_model(self, model: str) -> None:
        self.modelSelected.emit(str(model or ""))
        self.close()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        path = QPainterPath()
        # -1 нужен, чтобы маска не вылезала за border на HiDPI/Windows.
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12.0, 12.0)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))
```

Это исправляет углы: popup получает `WA_TranslucentBackground` и реальную rounded-mask, а кнопки не касаются границ из-за `contentsMargins(10, 10, 10, 10)`.

---

## 7. Добавить сам editor модели

Ниже `ModelSelectorPopup` добавь:

```python
class ModelNameEditor(QWidget):
    valueChanged = Signal()

    def __init__(self, value: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self._current = str(value or "").strip()
        self._models_cache: list[str] = [self._current] if self._current else []
        self._popup: ModelSelectorPopup | None = None
        self._worker: ModelListWorker | None = None
        self._loading = False
        self.setProperty("compact_value_control", True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.button = PaintedButton(self._current or "empty", self)
        self.button.set_button_font(_ui_font(pixel_size=11))
        self.button.set_button_padding(8, 0, 8, 0)
        self.button.set_button_radius(8)
        self.button.set_text_alignment(Qt.AlignmentFlag.AlignLeft)
        self.button.configure_colors(
            normal_bg="rgba(255,255,255,0.02)",
            normal_border="rgba(139,92,246,0.34)",
            normal_text=TEXT,
            hover_bg="rgba(139,92,246,0.10)",
            hover_border="rgba(139,92,246,0.42)",
            hover_text=TEXT,
            active_bg="rgba(139,92,246,0.18)",
            active_border="rgba(139,92,246,0.46)",
            active_text=TEXT,
        )
        self.button.clicked.connect(self._request_popup)
        layout.addWidget(self.button)

        # Предзагрузка после построения окна, чтобы первый клик был быстрым.
        QTimer.singleShot(0, self.refresh_models_async)

    def value(self) -> str:
        return self._current

    def set_value(self, value: Any) -> None:
        text = str(value or "").strip()
        if text == self._current:
            return
        self._current = text
        if text and text not in self._models_cache:
            self._models_cache.insert(0, text)
        self._refresh_button()
        if self._popup is not None:
            self._popup.set_models(self._current, self._models_cache)

    def refresh_models_async(self) -> None:
        if self._loading:
            return
        self._loading = True
        if self._popup is not None:
            self._popup.set_status("обновляю список...")

        self._worker = ModelListWorker(self._current, self)
        self._worker.loaded.connect(self._on_models_loaded)
        self._worker.failed.connect(self._on_models_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _request_popup(self) -> None:
        self._ensure_popup()
        self._show_popup()
        # Если в кеше только текущая модель — обновляем в фоне, но popup уже открыт.
        if len(self._models_cache) <= 1:
            self.refresh_models_async()

    def _ensure_popup(self) -> None:
        if self._popup is not None:
            return
        self._popup = ModelSelectorPopup(self.window())
        self._popup.modelSelected.connect(self._select_model)
        self._popup.set_models(self._current, self._models_cache)

    def _show_popup(self) -> None:
        if self._popup is None:
            return

        self._popup.set_models(self._current, self._models_cache)
        self._popup.adjustSize()
        width = max(240, min(420, self._popup.sizeHint().width()))
        height = min(360, max(96, self._popup.sizeHint().height()))
        self._popup.resize(width, height)

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
        self._popup.activateWindow()

    def _select_model(self, model: str) -> None:
        model = str(model or "").strip()
        if not model or model == self._current:
            return
        self._current = model
        if model not in self._models_cache:
            self._models_cache.insert(0, model)
        self._refresh_button()
        self.valueChanged.emit()

    def _on_models_loaded(self, runtime: str, models: list[str]) -> None:
        runtime = str(runtime or "").strip()
        models = _extract_model_names(models)
        if runtime:
            self._current = runtime
        if self._current and self._current not in models:
            models.insert(0, self._current)
        self._models_cache = models
        self._refresh_button()
        if self._popup is not None:
            self._popup.set_status("")
            self._popup.set_models(self._current, self._models_cache)

    def _on_models_failed(self, error_text: str) -> None:
        if self._current and self._current not in self._models_cache:
            self._models_cache.insert(0, self._current)
        if self._popup is not None:
            self._popup.set_status("API недоступно, показан кеш")
            self._popup.set_models(self._current, self._models_cache)

    def _on_worker_finished(self) -> None:
        self._loading = False
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

    def _refresh_button(self) -> None:
        self.button.setText(self._current or "empty")
        self.button.setToolTip(self._current)
        self.updateGeometry()
```

---

## 8. Подключить `ModelNameEditor` в `SettingEditor.value()`

В `SettingEditor.value()` найди блок:

```python
if isinstance(widget, QComboBox):
    return str(widget.currentText())
if isinstance(widget, TextValueEditor):
    return widget.parsed_value()
```

Замени на:

```python
if isinstance(widget, QComboBox):
    return str(widget.currentText())
if isinstance(widget, ModelNameEditor):
    return widget.value()
if isinstance(widget, TextValueEditor):
    return widget.parsed_value()
```

---

## 9. Подключить `ModelNameEditor` в `SettingEditor.set_value()`

В `SettingEditor.set_value()` найди блок:

```python
elif isinstance(widget, QComboBox):
    text = str(value or "")
    idx = widget.findText(text)
    if idx < 0 and text:
        widget.addItem(text)
        idx = widget.findText(text)
    widget.setCurrentIndex(max(0, idx))
elif isinstance(widget, TextValueEditor):
    widget.set_value(value)
```

Замени на:

```python
elif isinstance(widget, QComboBox):
    text = str(value or "")
    idx = widget.findText(text)
    if idx < 0 and text:
        widget.addItem(text)
        idx = widget.findText(text)
    widget.setCurrentIndex(max(0, idx))
elif isinstance(widget, ModelNameEditor):
    widget.set_value(value)
elif isinstance(widget, TextValueEditor):
    widget.set_value(value)
```

---

## 10. Использовать `ModelNameEditor` только для `llm.model_name`

В `SettingEditor._build_widget()` перед `if kind == "bool":` добавь:

```python
if self.spec.path == "llm.model_name":
    widget = ModelNameEditor(value)
    widget.valueChanged.connect(self._emit_changed)
    return widget
```

То есть начало метода должно стать примерно таким:

```python
def _build_widget(self, value: Any) -> QWidget:
    kind = self.spec.kind

    if self.spec.path == "llm.model_name":
        widget = ModelNameEditor(value)
        widget.valueChanged.connect(self._emit_changed)
        return widget

    if kind == "bool":
        ...
```

---

## 11. Важная проверка

После запуска проверь:

1. Popup открывается сразу, даже если Ollama/API думает.
2. Пока модели грузятся, видно `обновляю список...`.
3. Когда API ответит, список обновится без закрытия popup.
4. Углы popup больше не вылезают за границы.
5. В списке должны быть все модели из `/models`, а не только `gemma4:26b` и `qwen3:8b`.

---

## 12. Почему так

Не надо использовать `QComboBox`.

Лучше использовать:

- `PaintedButton` — чтобы кнопка и пункты были как в чате.
- `QFrame` popup со стилем `settings_text_editor_popup` — чтобы цвет был как у эдиторского окна.
- `QThread` worker — чтобы загрузка `/models` не блокировала UI.
- cache `_models_cache` — чтобы popup открывался мгновенно.
- `setMask(QRegion(...))` + `WA_TranslucentBackground` — чтобы скруглённые углы реально обрезались, а не просто выглядели скруглёнными в stylesheet.

