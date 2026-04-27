# MMis — углы всё ещё вылезают: правка не применена + финальный фикс через общий wrapper

## Что показала проверка свежего архива

В свежем `MMis.zip` предыдущая правка **не применена**.

В `ui/settings_widgets.py` нет:

```python
class CompactValueFrame
```

В `ui/settings_styles.py` нет:

```css
QFrame#settings_compact_value_frame
```

А `TextValueEditor` и `ModelNameEditor` всё ещё собраны напрямую из двух отдельных виджетов:

```python
layout = QHBoxLayout(self)

self.preview = QPushButton(self)
layout.addWidget(self.preview)

self.edit_button = QToolButton(self)
layout.addWidget(self.edit_button)
```

Из-за этого углы и дальше рисуются двумя соседними контролами отдельно:

```text
[ QPushButton с левыми углами ][ QToolButton с правыми углами ]
```

На Qt/Windows это и даёт фиолетовые хвостики/углы за краями.

---

# Что надо сделать

Нужно исправить **оба** редактора:

```text
TextValueEditor
ModelNameEditor
```

Потому что `ASYA` — это, скорее всего, обычный `TextValueEditor`, а выбор модели — это `ModelNameEditor`. Оба используют один и тот же стиль:

```css
settings_value_preview
settings_value_edit_button
```

---

## 1. Файл `ui/settings_widgets.py`

### Добавить класс `CompactValueFrame`

Добавить перед:

```python
class TextValueEditor(QWidget):
```

вот этот класс:

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

        path = QPainterPath()
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path.addRoundedRect(rect, 8.0, 8.0)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))
```

---

## 2. Файл `ui/settings_widgets.py`

### Переделать `TextValueEditor.__init__()`

Внутри `TextValueEditor.__init__()` найти:

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

Заменить на:

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

Оставить ниже старые connect:

```python
self.preview.pressed.connect(self._ensure_popup)
self.edit_button.pressed.connect(self._ensure_popup)
self.preview.clicked.connect(self._request_editor)
self.edit_button.clicked.connect(self._request_editor)
self._refresh_preview()
```

---

## 3. Файл `ui/settings_widgets.py`

### Переделать `ModelNameEditor.__init__()`

Внутри `ModelNameEditor.__init__()` найти:

```python
layout = QHBoxLayout(self)
layout.setContentsMargins(0, 0, 0, 0)
layout.setSpacing(0)

self.preview = QPushButton(self)
...
layout.addWidget(self.preview)

self.arrow_button = QToolButton(self)
...
layout.addWidget(self.arrow_button)
```

Заменить на:

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
self.preview.setCursor(Qt.CursorShape.PointingHandCursor)
self.preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
self.preview.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
self.preview.setFixedHeight(22)
self.preview.clicked.connect(self._request_popup)
layout.addWidget(self.preview)

self.arrow_button = QToolButton(self.frame)
self.arrow_button.setObjectName("settings_value_edit_button")
self.arrow_button.setIcon(QIcon(str(_CHEVRON_ICON)))
self.arrow_button.setFixedSize(28, 22)
self.arrow_button.setIconSize(QSize(13, 13))
self.arrow_button.setToolTip("Модели")
self.arrow_button.setCursor(Qt.CursorShape.PointingHandCursor)
self.arrow_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
self.arrow_button.clicked.connect(self._request_popup)
layout.addWidget(self.arrow_button)
```

---

## 4. Файл `ui/settings_widgets.py`

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

## 5. Файл `ui/settings_widgets.py`

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

## 6. Файл `ui/settings_widgets.py`

### Исправить `ModelNameEditor.sizeHint()`

Найти:

```python
def sizeHint(self) -> QSize:
    text_width = self.preview.fontMetrics().horizontalAdvance(self.preview.text())
    width = min(420, max(120, text_width + 52))
    return QSize(width, 22)
```

Заменить на:

```python
def sizeHint(self) -> QSize:
    text_width = self.preview.fontMetrics().horizontalAdvance(self.preview.text())
    width = min(420, max(120, text_width + 54))
    return QSize(width, 24)
```

---

## 7. Файл `ui/settings_widgets.py`

### Исправить `ModelNameEditor.minimumSizeHint()`

Найти:

```python
def minimumSizeHint(self) -> QSize:
    return QSize(110, 22)
```

Заменить на:

```python
def minimumSizeHint(self) -> QSize:
    return QSize(110, 24)
```

---

## 8. Файл `ui/settings_widgets.py`

### Обновить `_refresh_preview()` у `TextValueEditor`

В `TextValueEditor._refresh_preview()` найти:

```python
self.setMaximumWidth(self.sizeHint().width())
self.updateGeometry()
```

Заменить на:

```python
self.setMaximumWidth(self.sizeHint().width())
if hasattr(self, "frame"):
    self.frame.update()
self.updateGeometry()
```

---

## 9. Файл `ui/settings_widgets.py`

### Обновить `_refresh_preview()` у `ModelNameEditor`

В `ModelNameEditor._refresh_preview()` найти:

```python
self.setMaximumWidth(self.sizeHint().width())
self.updateGeometry()
```

Заменить на:

```python
self.setMaximumWidth(self.sizeHint().width())
if hasattr(self, "frame"):
    self.frame.update()
self.updateGeometry()
```

---

# Стили

## 10. Файл `ui/settings_styles.py`

### Добавить wrapper-стиль

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

## 11. Файл `ui/settings_styles.py`

### Заменить стиль preview

Найти блок:

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

## 12. Файл `ui/settings_styles.py`

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

## 13. Файл `ui/settings_styles.py`

### Заменить стиль кнопки редактирования

Найти блок:

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

## 14. Файл `ui/settings_styles.py`

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

# Контрольная проверка

После правок выполнить из корня проекта:

```cmd
python -c "from pathlib import Path; w=Path('ui/settings_widgets.py').read_text(encoding='utf-8'); s=Path('ui/settings_styles.py').read_text(encoding='utf-8'); assert 'class CompactValueFrame' in w; assert 'settings_compact_value_frame' in s; assert 'border-top-left-radius: 8px;' not in s[s.find('QPushButton#settings_value_preview'):s.find('QLineEdit#settings_numeric_input')]; print('OK: compact value wrapper applied')"
```

Должно вывести:

```text
OK: compact value wrapper applied
```

Если не вывело — правка не применена.

---

# Проверка в UI

Проверить:

```text
Settings -> Main config -> Активный профиль -> ASYA
Settings -> Main config -> Локаль приложения -> ru_RU
Settings -> LLM / Models -> Основная модель
Settings -> LLM / Models -> Ollama URL
```

Ожидаемо:

```text
углы не выходят за края
нет фиолетовых хвостиков снизу
контрол выглядит как один цельный элемент
карандаш/стрелка остаётся справа
```

---

# Главное

Пока в `settings_styles.py` остаётся:

```css
QPushButton#settings_value_preview {
    border-top-left-radius: 8px;
    ...
}
```

и в `settings_widgets.py` нет:

```python
class CompactValueFrame
```

углы будут вылезать дальше.
