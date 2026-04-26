# MMis — как сделать выбор модели в настройках как в чате

Задача: в настройках заменить текстовый редактор `llm.model_name` на выпадающий выбор моделей. Визуально он должен вести себя как селектор моделей в чате, но выглядеть в стиле эдиторского окна настроек: тёмный popup, фиолетовая рамка, Cascadia Code, компактные кнопки.

Проверено по последнему архиву: `MMis.zip`.

---

## Что использовать

Используй уже готовые части проекта, не делай новый стиль с нуля:

| Что нужно | Что использовать | Почему |
|---|---|---|
| Popup-окно | `QFrame` с флагами `Qt.Popup | Qt.FramelessWindowHint` | ведёт себя как выпадающее окно и закрывается при клике вне него |
| Стиль popup | `_TEXT_EDITOR_POPUP_STYLE` из `ui/settings_widgets.py` | это уже стиль эдиторского окна настроек |
| Кнопки моделей | `PaintedButton` из `ui/chat_shell.py` | такие же рисованные кнопки, как в чате |
| Цвета | `TEXT`, `MUTED`, `LINE` из `ui/chat_shell.py` | чтобы не плодить разные оттенки вручную |
| Шрифт | `_ui_font(pixel_size=12)` | единый UI-шрифт проекта |
| Список моделей | `ApiClient().list_models(timeout=1.2)` | берёт модели из API `/models` |
| Fallback | текущая модель + `qwen3:8b` | чтобы настройки работали даже без API |
| Иконка раскрытия | `ui/assets/settings_chevron_down.svg` | она уже есть в архиве |

Не используй здесь `QComboBox`: он будет выглядеть чужеродно и хуже кастомизируется под текущий UI.

---

## Как оно должно работать

Логика такая:

```text
SettingEditor для llm.model_name
    -> ModelNameEditor
        -> кнопка-превью текущей модели
        -> кнопка-стрелка
        -> ModelPickerPopup
            -> title "Модели"
            -> список PaintedButton
            -> выбранная модель помечается ✓
            -> при выборе обновляется значение настройки
```

То есть это не отдельное окно настроек и не системный dropdown. Это маленький кастомный popup, который живёт рядом со строкой `Основная модель`.

---

# 1. Файл `ui/settings_widgets.py`

## 1.1. Исправить импорт из `ui.chat_shell`

Найди:

```python
from ui.chat_shell import ToggleSwitch, _ui_font, _to_qcolor, TEXT, MUTED, PlainTextScrollOverlay
```

Замени на:

```python
from ui.chat_shell import ToggleSwitch, _ui_font, _to_qcolor, TEXT, MUTED, LINE, PaintedButton, PlainTextScrollOverlay
```

## 1.2. Добавить импорт `ApiClient`

Сразу после импорта из `ui.chat_shell` добавь:

```python
from ui.api_client import ApiClient
```

---

# 2. Файл `ui/settings_widgets.py`

## 2.1. Добавить путь к иконке стрелки

Найди:

```python
_PENCIL_ICON = Path(__file__).resolve().parent / "assets" / "settings_pencil.svg"
```

Сразу после неё добавь:

```python
_CHEVRON_ICON = Path(__file__).resolve().parent / "assets" / "settings_chevron_down.svg"
```

---

# 3. Файл `ui/settings_widgets.py`

## 3.1. Добавить popup выбора модели

Найди конец класса `TextValueEditor`, прямо перед:

```python
class NumericValueEditor(QWidget):
```

Перед ним вставь:

```python
class ModelPickerPopup(QFrame):
    modelSelected = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("settings_text_editor_popup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_TEXT_EDITOR_POPUP_STYLE)
        self.setFixedWidth(240)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(8)

        self.title_label = QLabel("Модели")
        self.title_label.setObjectName("settings_text_editor_title")
        self._layout.addWidget(self.title_label)

        self.list_wrap = QWidget(self)
        self.list_layout = QVBoxLayout(self.list_wrap)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(8)
        self._layout.addWidget(self.list_wrap)

    def load_models(self, current: str) -> None:
        current = str(current or "").strip()
        runtime = ""
        models: list[str] = []

        try:
            payload = ApiClient().list_models(timeout=1.2)
            runtime = str(payload.get("runtime_model") or "").strip()
            available = payload.get("available_models") or payload.get("models") or []
            models = [str(item).strip() for item in available if str(item).strip()]
        except Exception:
            models = []

        merged: list[str] = []
        for name in [current, runtime, *models, "qwen3:8b"]:
            name = str(name or "").strip()
            if name and name not in merged:
                merged.append(name)

        active = current or runtime or (merged[0] if merged else "")
        self._populate(active, merged)

    def _populate(self, active: str, models: list[str]) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not models:
            models = [active or "qwen3:8b"]

        for model in models:
            button = PaintedButton()
            button.set_button_font(_ui_font(pixel_size=12))
            button.set_button_padding(8, 6, 8, 6)
            button.set_button_radius(8)
            button.set_text_alignment(Qt.AlignmentFlag.AlignLeft)
            button.configure_colors(
                normal_bg="rgba(16,18,22,.28)",
                normal_border=LINE,
                normal_text=MUTED,
                hover_bg="rgba(139,92,246,.10)",
                hover_border="rgba(139,92,246,.18)",
                hover_text=TEXT,
                active_bg="rgba(139,92,246,.14)",
                active_border="rgba(139,92,246,.22)",
                active_text=TEXT,
            )
            button.set_active(model == active)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setText(f"{model}{'    ✓' if model == active else ''}")
            button.clicked.connect(lambda _checked=False, name=model: self._select_model(name))
            self.list_layout.addWidget(button)

        self.adjustSize()

    def _select_model(self, model: str) -> None:
        self.modelSelected.emit(str(model or ""))
        self.close()
```

Что тут важно:

- `QFrame` получает `objectName="settings_text_editor_popup"`, поэтому берёт стиль эдиторского окна.
- `PaintedButton` даёт такой же тип кнопок, как в чате.
- `ApiClient().list_models()` берёт список моделей из API.
- `except Exception` нужен специально: настройки не должны ломаться, если API сейчас выключено.

---

# 4. Файл `ui/settings_widgets.py`

## 4.1. Добавить сам редактор модели

Сразу после класса `ModelPickerPopup` вставь:

```python
class ModelNameEditor(QWidget):
    valueChanged = Signal()

    def __init__(self, value: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self._value = str(value or "").strip()
        self._popup: ModelPickerPopup | None = None
        self.setProperty("compact_value_control", True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.preview = QPushButton(self)
        self.preview.setObjectName("settings_value_preview")
        self.preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preview.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.preview.setFixedHeight(24)
        self.preview.clicked.connect(self._request_popup)
        layout.addWidget(self.preview)

        self.arrow_button = QToolButton(self)
        self.arrow_button.setObjectName("settings_value_edit_button")
        self.arrow_button.setIcon(QIcon(str(_CHEVRON_ICON)))
        self.arrow_button.setFixedSize(28, 24)
        self.arrow_button.setIconSize(QSize(13, 13))
        self.arrow_button.setToolTip("Выбрать модель")
        self.arrow_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.arrow_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.arrow_button.clicked.connect(self._request_popup)
        layout.addWidget(self.arrow_button)

        self._refresh_preview()

    def value(self) -> str:
        return self._value

    def set_value(self, value: Any) -> None:
        value = str(value or "").strip()
        if value == self._value:
            return
        self._value = value
        self._refresh_preview()
        self.valueChanged.emit()

    def sizeHint(self) -> QSize:
        text_width = self.preview.fontMetrics().horizontalAdvance(self.preview.text())
        width = min(420, max(120, text_width + 52))
        return QSize(width, 22)

    def minimumSizeHint(self) -> QSize:
        return QSize(110, 22)

    def _ensure_popup(self) -> None:
        if self._popup is not None:
            return
        self._popup = ModelPickerPopup(self.window())
        self._popup.modelSelected.connect(self.set_value)

    def _request_popup(self) -> None:
        self._ensure_popup()
        QTimer.singleShot(0, self._show_popup)

    def _show_popup(self) -> None:
        self._ensure_popup()
        if self._popup is None:
            return

        if self._popup.isVisible():
            self._popup.raise_()
            self._popup.activateWindow()
            return

        self._popup.load_models(self._value)
        self._popup.adjustSize()

        width = max(240, self.width())
        height = self._popup.sizeHint().height()
        self._popup.setFixedWidth(width)
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

    def _refresh_preview(self) -> None:
        text = self._value or "empty"
        self.preview.setText(text)
        self.preview.setToolTip(self._value)
        self.setMaximumWidth(self.sizeHint().width())
        self.updateGeometry()
```

Что тут важно:

- `preview` — это видимое значение модели в строке настроек.
- `arrow_button` — маленькая кнопка раскрытия вместо карандаша.
- `QTimer.singleShot(0, ...)` оставляем, чтобы popup открывался после завершения обработки клика. Это уменьшает шанс багов, как было с editor popup и `restart_required=True`.
- Popup позиционируется под строкой, но если снизу нет места — открывается сверху.

---

# 5. Файл `ui/settings_widgets.py`

## 5.1. Научить `SettingEditor.value()` читать `ModelNameEditor`

В методе `SettingEditor.value()` найди:

```python
        if isinstance(widget, TextValueEditor):
            return widget.parsed_value()
```

Перед ним добавь:

```python
        if isinstance(widget, ModelNameEditor):
            return widget.value()
```

Должно получиться:

```python
        if isinstance(widget, ModelNameEditor):
            return widget.value()
        if isinstance(widget, TextValueEditor):
            return widget.parsed_value()
```

---

# 6. Файл `ui/settings_widgets.py`

## 6.1. Научить `SettingEditor.set_value()` обновлять `ModelNameEditor`

В методе `SettingEditor.set_value()` найди:

```python
        elif isinstance(widget, TextValueEditor):
            widget.set_value(value)
```

Перед ним добавь:

```python
        elif isinstance(widget, ModelNameEditor):
            widget.set_value(value)
```

Должно получиться:

```python
        elif isinstance(widget, ModelNameEditor):
            widget.set_value(value)
        elif isinstance(widget, TextValueEditor):
            widget.set_value(value)
```

---

# 7. Файл `ui/settings_widgets.py`

## 7.1. Подключить новый редактор только для `llm.model_name`

В методе `SettingEditor._build_widget()` найди:

```python
    def _build_widget(self, value: Any) -> QWidget:
        kind = self.spec.kind
        if kind == "bool":
```

Замени на:

```python
    def _build_widget(self, value: Any) -> QWidget:
        if self.spec.path == "llm.model_name":
            widget = ModelNameEditor(value)
            widget.valueChanged.connect(self._emit_changed)
            return widget

        kind = self.spec.kind
        if kind == "bool":
```

Так новый селектор затронет только основную модель, а все остальные строковые настройки останутся со старым эдитором.

---

# 8. Что получится визуально

В строке `Основная модель / llm.model_name` будет:

```text
[qwen3:8b                    ▼]
```

При клике откроется popup:

```text
Модели
[qwen3:8b                 ✓]
[mistral:7b                ]
[qwen coder                ]
```

По стилю:

- фон popup как у эдиторского окна;
- рамка фиолетовая;
- кнопки тёмные и компактные;
- hover фиолетовый;
- активная модель подсвечена;
- галочка стоит у выбранной модели.

---

# 9. Проверка

1. Запусти приложение.
2. Открой настройки.
3. Перейди в группу, где есть `llm.model_name`.
4. Нажми на текущее значение модели или стрелку.
5. Должен открыться popup `Модели`.
6. Выбери другую модель.
7. В правом блоке изменений должна появиться правка `llm.model_name`.
8. Нажми `Save`.
9. Перезапусти приложение и проверь, что выбранная модель сохранилась.

---

# 10. Если модели не подтянулись

Если API выключено или `/models` не отвечает, это нормально. Popup всё равно должен показать:

- текущую модель из настроек;
- fallback `qwen3:8b`.

Если popup вообще не открывается, проверь эти места:

- импортирован ли `ApiClient`;
- импортированы ли `LINE` и `PaintedButton`;
- есть ли `_CHEVRON_ICON`;
- вставлены ли классы `ModelPickerPopup` и `ModelNameEditor` выше `NumericValueEditor`;
- добавлена ли проверка `self.spec.path == "llm.model_name"` в начале `_build_widget()`.
