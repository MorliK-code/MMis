# MMis — исправить углы компактных контролов значений (`ASYA`, `ru_RU`, URL и т.д.)

## Что сейчас происходит

Проблема на скрине уже не в popup-окне редактора.

Сейчас углы вылезают именно у самого компактного контрола значения:

```text
[ ASYA ]
```

Это контрол из `TextValueEditor` в файле:

```text
ui/settings_widgets.py
```

Сейчас он собран из двух отдельных виджетов:

```python
self.preview = QPushButton(...)
self.edit_button = QToolButton(...)
```

А стиль задаётся отдельно:

```css
QPushButton#settings_value_preview {
    border-top-left-radius: 8px;
    border-bottom-left-radius: 8px;
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    border: 1px solid ...
    border-right: 0;
}

QToolButton#settings_value_edit_button {
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    border: 1px solid ...
}
```

На Qt/Windows это часто даёт артефакт: border-radius двух соседних виджетов рисуется отдельно и углы/границы визуально выходят за общий контур.

Правильнее сделать так:

```text
[ общий wrapper с border/radius/background ]
    [preview без border/radius]
    [edit button без внешних углов]
```

То есть рамку и скругление рисует **один общий контейнер**, а не две кнопки отдельно.

---

## 1. Файл `ui/settings_widgets.py`

### Добавить класс `CompactValueFrame`

Добавить класс перед `class TextValueEditor(QWidget):`

```python
class CompactValueFrame(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("settings_compact_value_frame")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        # Страховочная маска: не даём дочерним кнопкам/границам вылезать за радиус.
        path = QPainterPath()
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path.addRoundedRect(rect, 8.0, 8.0)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))
```

`QPainterPath` и `QRegion` у тебя уже импортированы в `ui/settings_widgets.py`.

---

## 2. Файл `ui/settings_widgets.py`

### Переделать `TextValueEditor.__init__()`

Найти внутри `TextValueEditor.__init__()`:

```python
layout = QHBoxLayout(self)
layout.setContentsMargins(0, 0, 0, 0)
layout.setSpacing(0)

self.preview = QPushButton(self)
...
layout.addWidget(self.preview)

self.edit_button = QToolButton(self)
...
layout.addWidget(self.edit_button)
```

Заменить на такую структуру:

```python
root_layout = QHBoxLayout(self)
root_layout.setContentsMargins(0, 0, 0, 0)
root_layout.setSpacing(0)

self.frame = CompactValueFrame(self)
root_layout.addWidget(self.frame)

layout = QHBoxLayout(self.frame)
layout.setContentsMargins(1, 1, 1, 1)
layout.setSpacing(0)

self.preview = QPushButton(self.frame)
self.preview.setObjectName("settings_value_preview")
self.preview.setCursor(Qt.CursorShape.ArrowCursor)
self.preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
self.preview.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
self.preview.setFixedHeight(22)
layout.addWidget(self.preview)

self.edit_button = QToolButton(self.frame)
self.edit_button.setObjectName("settings_value_edit_button")
self.edit_button.setIcon(QIcon(str(_PENCIL_ICON)))
self.edit_button.setFixedSize(28, 22)
self.edit_button.setIconSize(QSize(13, 13))
self.edit_button.setToolTip("Изменить значение")
self.edit_button.setCursor(Qt.CursorShape.PointingHandCursor)
self.edit_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
layout.addWidget(self.edit_button)
```

Остальные connect оставить ниже как были:

```python
self.preview.pressed.connect(self._ensure_popup)
self.edit_button.pressed.connect(self._ensure_popup)
self.preview.clicked.connect(self._request_editor)
self.edit_button.clicked.connect(self._request_editor)
```

---

## 3. Файл `ui/settings_widgets.py`

### Исправить `TextValueEditor.sizeHint()`

Найти:

```python
def sizeHint(self) -> QSize:
    text_width = self.preview.fontMetrics().horizontalAdvance(self.preview.text())
    max_width = 420 if self.as_json or len(self._text) > 34 else 260
    width = min(max_width, max(92, text_width + 48))
    return QSize(width, 22)
```

Заменить на:

```python
def sizeHint(self) -> QSize:
    text_width = self.preview.fontMetrics().horizontalAdvance(self.preview.text())
    max_width = 420 if self.as_json or len(self._text) > 34 else 260
    width = min(max_width, max(92, text_width + 50))
    return QSize(width, 24)
```

---

## 4. Файл `ui/settings_widgets.py`

### Исправить `TextValueEditor.minimumSizeHint()`

Найти:

```python
def minimumSizeHint(self) -> QSize:
    return QSize(86, 22)
```

Заменить на:

```python
def minimumSizeHint(self) -> QSize:
    return QSize(86, 24)
```

---

## 5. Файл `ui/settings_widgets.py`

### Исправить `_refresh_preview()`

Найти:

```python
self.setMaximumWidth(self.sizeHint().width())
```

Оставить, но после него добавить обновление frame:

```python
if hasattr(self, "frame"):
    self.frame.update()
```

Итог:

```python
self.setMaximumWidth(self.sizeHint().width())
if hasattr(self, "frame"):
    self.frame.update()
self.updateGeometry()
```

---

## 6. Файл `ui/settings_styles.py`

### Добавить стиль общего wrapper

Перед блоком:

```css
QPushButton#settings_value_preview {
```

добавить:

```css
QFrame#settings_compact_value_frame {
    min-height: 24px;
    max-height: 24px;
    border-radius: 8px;
    border: 1px solid rgba(139, 92, 246, 46);
    background: rgba(8, 10, 14, 128);
}
```

---

## 7. Файл `ui/settings_styles.py`

### Заменить стиль `settings_value_preview`

Найти текущий блок:

```css
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
```

Заменить на:

```css
QPushButton#settings_value_preview {
    min-height: 22px;
    max-height: 22px;
    border-radius: 0;
    border: 0;
    background: transparent;
    color: #f3f4f6;
    text-align: left;
    padding: 0 7px 1px 7px;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 11px;
}
```

---

## 8. Файл `ui/settings_styles.py`

### Заменить hover preview

Найти:

```css
QPushButton#settings_value_preview:hover {
    background: rgba(8, 10, 14, 108);
    border-color: rgba(255, 255, 255, 22);
    color: #f3f4f6;
}
```

Заменить на:

```css
QPushButton#settings_value_preview:hover {
    background: transparent;
    border: 0;
    color: #f3f4f6;
}
```

---

## 9. Файл `ui/settings_styles.py`

### Заменить стиль edit button

Найти текущий блок:

```css
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
```

Заменить на:

```css
QToolButton#settings_value_edit_button {
    min-width: 28px;
    max-width: 28px;
    min-height: 22px;
    max-height: 22px;
    border-radius: 7px;
    border: 0;
    border-left: 1px solid rgba(139, 92, 246, 38);
    background: rgba(139, 92, 246, 18);
    color: #c4b5fd;
    padding: 0;
    font-size: 12px;
}
```

---

## 10. Файл `ui/settings_styles.py`

### Заменить hover edit button

Найти:

```css
QToolButton#settings_value_edit_button:hover {
    background: rgba(139, 92, 246, 36);
    border-color: rgba(139, 92, 246, 76);
    color: #f3f4f6;
}
```

Заменить на:

```css
QToolButton#settings_value_edit_button:hover {
    background: rgba(139, 92, 246, 34);
    border-left-color: rgba(139, 92, 246, 70);
    color: #f3f4f6;
}
```

---

## 11. Почему это исправит углы

Сейчас углы рисуют два разных виджета:

```text
preview button + edit button
```

И каждый со своими radius/border.

После фикса углы рисует только один виджет:

```text
CompactValueFrame
```

А кнопки внутри становятся прозрачными и не могут вылезти за радиус.

---

## 12. Проверка

Открыть:

```text
Settings -> Main config -> Активный профиль
```

Проверить значение:

```text
ASYA
```

Ожидаемо:

```text
углы не выходят за края
нет фиолетовых хвостиков снизу
контрол выглядит как цельная капсула
карандаш остаётся справа
```

Также проверить:

```text
Название приложения
Локаль приложения
Ollama URL
OpenAI API URL
Web V2 config
Fallback модели
```

---

## Важно

Предыдущая инструкция про `TextEditorPopup` чинит **всплывающее окно редактора**.

А этот фикс чинит **сам компактный контрол значения**.

На твоём последнем скрине проблема именно в компактном контроле, а не в popup.
