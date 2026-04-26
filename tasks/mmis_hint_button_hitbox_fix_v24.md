# MMis — фикс хитбокса `?` в настройках

## Проблема

На коротких названиях параметров, например:

```text
Debug mode ?
```

подсказка открывается только если навести на самый край иконки `?`.

Причина, скорее всего, в связке:

1. `HintButton` сделан как `QWidget`, но в CSS есть стиль под `QToolButton#hint_button`.
2. `QLabel` с коротким текстом и/или соседний layout может перекрывать часть области `?`.
3. У `HintButton` нет гарантированного верхнего слоя и нет нормального Qt button hitbox.

Нужно сделать `?` настоящей кнопкой `QToolButton`, но рисовать её вручную как раньше.

---

## 1. Файл `ui/settings_window.py`

### В импортах уже должен быть `QToolButton`

Проверить, что есть:

```python
QToolButton,
```

Если нет — добавить в импорт из `PySide6.QtWidgets`.

---

## 2. Файл `ui/settings_window.py`

### Заменить класс `HintButton`

Найти:

```python
class HintButton(QWidget):
```

и заменить весь класс на:

```python
class HintButton(QToolButton):
    activated = Signal(object)
    hovered = Signal(object)
    unhovered = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("hint_button")
        self.setFixedSize(18, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setText("")
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setAutoRaise(True)

        # Важно: QToolButton не должен рисовать свой стандартный фон.
        self.setStyleSheet(
            "QToolButton#hint_button {"
            "border: 0;"
            "background: transparent;"
            "padding: 0px;"
            "margin: 0px;"
            "}"
        )

    def enterEvent(self, event) -> None:
        self.hovered.emit(self)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.unhovered.emit(self)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self)
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        rect = QRectF(0.5, 0.5, 17.0, 17.0)
        painter.setPen(QPen(_to_qcolor("rgba(139,92,246,.40)"), 1))
        painter.setBrush(_to_qcolor("rgba(139,92,246,.16)"))
        painter.drawEllipse(rect)

        painter.setPen(_to_qcolor("#9f8bff"))
        font = _ui_font(pixel_size=10)
        font.setFamily("Segoe UI")
        painter.setFont(font)
        painter.drawText(
            QRectF(0.0, -0.5, 18.0, 18.0),
            int(Qt.AlignmentFlag.AlignCenter),
            "?",
        )
```

---

## 3. Файл `ui/settings_window.py`

### В `_setting_row()` дать `?` верхний слой

Найти:

```python
hint = HintButton()
```

Сразу после этого добавить:

```python
hint.raise_()
```

---

## 4. Файл `ui/settings_window.py`

### Не давать label перехватывать мышь

В `_setting_row()` найти блок создания label:

```python
label = QLabel(_setting_title(spec))
```

После настройки label добавить:

```python
label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
```

Лучшее место — после:

```python
label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
```

Итог:

```python
label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
```

---

## 5. Файл `ui/settings_window.py`

### Убрать слишком широкий фикс label

Сейчас может быть так:

```python
label_width = max(44, label.fontMetrics().horizontalAdvance(label.text()) + 17)
label.setFixedWidth(label_width)
```

Для коротких названий это нормально, но лучше сделать чуть меньше запас:

```python
label_width = max(44, label.fontMetrics().horizontalAdvance(label.text()) + 10)
label.setFixedWidth(label_width)
```

Это уменьшит шанс, что label визуально/геометрически залезает на область `?`.

---

## 6. Файл `ui/settings_window.py`

### Увеличить расстояние между label и `?`

Сейчас может быть:

```python
layout.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignVCenter)
layout.setSpacing(2)
```

Заменить на:

```python
layout.setSpacing(4)
layout.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignVCenter)
```

Важно: `setSpacing(4)` должен быть **до** добавления stretch и лучше не перезаписывать его ниже.

Если ниже есть ещё раз:

```python
layout.setSpacing(2)
```

удалить эту строку.

---

## 7. Файл `ui/settings_styles.py`

### Проверить стиль `hint_button`

Найти:

```css
QToolButton#hint_button {
```

Оставить так:

```css
QToolButton#hint_button {
    min-width: 18px;
    max-width: 18px;
    min-height: 18px;
    max-height: 18px;
    border: 0;
    background: transparent;
    padding: 0;
    margin: 0;
}
```

Если есть стиль:

```css
QWidget#hint_button
```

его можно удалить, потому что теперь `HintButton` — это `QToolButton`.

---

## 8. Проверка

Открыть:

```text
Settings -> Main config -> Приложение -> Debug mode
```

Проверить:

```text
1. Наведение на центр `?` открывает подсказку.
2. Наведение на левую часть `?` открывает подсказку.
3. Наведение на правую часть `?` открывает подсказку.
4. Клик по `?` тоже открывает подсказку.
```

Потом проверить остальные короткие параметры:

```text
Memory
JSON mode
Thinking
Internet
Enabled
Barge-in
Colors
Level
```

---

## Итог

После фикса:

- `?` больше не перекрывается label-ом;
- весь круг `?` становится кликабельным/hoverable;
- короткие названия вроде `Debug mode` больше не ломают наведение;
- стиль и внешний вид иконки остаются прежними.
