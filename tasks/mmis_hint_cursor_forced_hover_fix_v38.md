# MMis — подсказки открываются, но курсор не меняется: финальный UX-фикс hitbox `?`

## Что сейчас в свежем архиве

Проверил свежий `MMis.zip`.

Старый `SettingTitleHint` уже удалён — это хорошо.

Сейчас в `ui/settings_window.py` структура строки уже нормальная:

```python
label = QLabel(_setting_title(spec), row)
hint = HintButton(row)

layout.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignVCenter)
layout.addStretch(1)
```

Также добавлен row-level hitbox:

```python
row.installEventFilter(self)
self._hint_rows[row] = (hint, spec)
```

Поэтому подсказки теперь открываются.

Но осталась UX-проблема: когда подсказка открывается через row-level hitbox, курсор мыши находится **не над самим `HintButton`**, а над расширенной областью вокруг него. Поэтому `QToolButton` не получает настоящий hover, и курсор может оставаться обычной стрелкой.

То есть логика работает, но визуально кажется, что `?` всё ещё не кликабельный.

---

## Цель

Оставить внешний вид как сейчас:

```text
[полное название] [?] [свободное место] [контрол]
```

Но если курсор находится в расширенной зоне вокруг `?`, надо:

```text
1. показывать подсказку
2. менять курсор на PointingHandCursor
3. подсвечивать сам `?` как hover
4. по клику открывать подсказку
5. при уходе — возвращать курсор и скрывать popup
```

---

## 1. Файл `ui/settings_window.py`

### Добавить hover-состояние в `HintButton`

Найти класс:

```python
class HintButton(QToolButton):
```

В `__init__()` после:

```python
self.setAutoRaise(True)
```

добавить:

```python
self._forced_hover = False
```

---

## 2. Файл `ui/settings_window.py`

### Добавить метод `set_forced_hover()` в `HintButton`

Внутрь класса `HintButton`, перед `paintEvent()`, добавить:

```python
def set_forced_hover(self, value: bool) -> None:
    value = bool(value)
    if self._forced_hover == value:
        return
    self._forced_hover = value
    self.setCursor(Qt.CursorShape.PointingHandCursor if value else Qt.CursorShape.ArrowCursor)
    self.update()
```

---

## 3. Файл `ui/settings_window.py`

### В `HintButton.paintEvent()` учитывать forced hover

Найти в `paintEvent()`:

```python
rect = QRectF(2.5, 2.5, 17.0, 17.0)
painter.setPen(QPen(_to_qcolor("rgba(139,92,246,.40)"), 1))
painter.setBrush(_to_qcolor("rgba(139,92,246,.16)"))
painter.drawEllipse(rect)

painter.setPen(_to_qcolor("#9f8bff"))
```

Заменить на:

```python
is_hover = self.underMouse() or bool(getattr(self, "_forced_hover", False))

rect = QRectF(2.5, 2.5, 17.0, 17.0)
border = "rgba(139,92,246,.62)" if is_hover else "rgba(139,92,246,.40)"
bg = "rgba(139,92,246,.24)" if is_hover else "rgba(139,92,246,.16)"
text_color = "#c4b5fd" if is_hover else "#9f8bff"

painter.setPen(QPen(_to_qcolor(border), 1))
painter.setBrush(_to_qcolor(bg))
painter.drawEllipse(rect)

painter.setPen(_to_qcolor(text_color))
```

---

## 4. Файл `ui/settings_window.py`

### Заменить блок row-level hitbox в `eventFilter()`

Найти в `eventFilter()` блок:

```python
if isinstance(watched, QWidget) and watched in self._hint_rows:
    hint, spec = self._hint_rows[watched]

    if event.type() in {
        QEvent.Type.MouseMove,
        QEvent.Type.HoverMove,
        QEvent.Type.Enter,
        QEvent.Type.MouseButtonPress,
    }:
        ...
```

Заменить весь блок `if isinstance(watched, QWidget) and watched in self._hint_rows:` на:

```python
if isinstance(watched, QWidget) and watched in self._hint_rows:
    hint, spec = self._hint_rows[watched]

    if event.type() in {
        QEvent.Type.MouseMove,
        QEvent.Type.HoverMove,
        QEvent.Type.Enter,
        QEvent.Type.MouseButtonPress,
    }:
        try:
            if hasattr(event, "position"):
                pos = event.position().toPoint()
            elif hasattr(event, "pos"):
                pos = event.pos()
            else:
                pos = None

            if pos is not None:
                # Расширенная зона вокруг видимого `?`.
                hint_rect = hint.geometry().adjusted(-6, -4, 6, 4)
                inside = hint_rect.contains(pos)

                if inside:
                    watched.setCursor(Qt.CursorShape.PointingHandCursor)

                    if hasattr(hint, "set_forced_hover"):
                        hint.set_forced_hover(True)

                    self._show_hint_popup(hint, spec)

                    if event.type() == QEvent.Type.MouseButtonPress:
                        event.accept()
                        return True
                else:
                    watched.setCursor(Qt.CursorShape.ArrowCursor)

                    if hasattr(hint, "set_forced_hover"):
                        hint.set_forced_hover(False)

                    self._hide_hint_popup()

        except Exception:
            pass

    elif event.type() in {QEvent.Type.Leave, QEvent.Type.Hide}:
        watched.setCursor(Qt.CursorShape.ArrowCursor)

        if hasattr(hint, "set_forced_hover"):
            hint.set_forced_hover(False)

        self._hide_hint_popup()
```

---

## 5. Файл `ui/settings_window.py`

### В `_setting_row()` убедиться, что row умеет менять курсор

В `_setting_row()` рядом с:

```python
row.setMouseTracking(True)
row.installEventFilter(self)
self._hint_rows[row] = (hint, spec)
```

сделать так:

```python
row.setMouseTracking(True)
row.setCursor(Qt.CursorShape.ArrowCursor)
row.installEventFilter(self)
self._hint_rows[row] = (hint, spec)
```

---

## 6. Файл `ui/settings_window.py`

### В `_hide_hint_popup()` сбрасывать forced hover у всех hint

Найти:

```python
def _hide_hint_popup(self) -> None:
    if self._hint_popup is not None:
        self._hint_popup.hide()
```

Заменить на:

```python
def _hide_hint_popup(self) -> None:
    for hint, _spec in list(getattr(self, "_hint_rows", {}).values()):
        if hasattr(hint, "set_forced_hover"):
            hint.set_forced_hover(False)

    if self._hint_popup is not None:
        self._hint_popup.hide()
```

---

## 7. Почему именно так

Сейчас подсказка открывается не только над реальной кнопкой, а над расширенной областью:

```python
hint.geometry().adjusted(-6, -4, 6, 4)
```

Это правильно.

Но курсор меняет только реальный виджет под мышью.  
Если мышь находится в расширенной зоне row-level hitbox, то под мышью может быть `row`, а не `HintButton`.

Поэтому надо вручную выставить:

```python
watched.setCursor(Qt.CursorShape.PointingHandCursor)
hint.set_forced_hover(True)
```

Тогда визуально всё будет честно:

```text
подсказка открылась → курсор рука → ? подсвечен
```

---

## 8. Проверка

Проверить короткие названия:

```text
Debug mode
API host
API port
Level
Colors
Memory
Enabled
```

Навести:

```text
на центр ?
на край ?
на 2–4px рядом с ?
```

Ожидаемо:

```text
подсказка открывается
курсор становится рукой
? подсвечивается
```

Проверить уход мыши:

```text
курсор возвращается в стрелку
? перестаёт подсвечиваться
popup скрывается
```

Проверить длинные названия:

```text
Название приложения
Локаль приложения
Язык по умолчанию
Активный API endpoint
Локальный API URL
Публичный API URL
```

Ожидаемо:

```text
полное название не режется
? сразу после названия
курсор рука только рядом с ?
```

---

## Итог

Проблема уже не в том, что подсказка не открывается.

Теперь проблема в UX-сигнале:

```text
popup открывается через row-level hitbox,
но курсор не меняется, потому что мышь не над QToolButton
```

Фикс:

```text
в row-level hitbox вручную ставить PointingHandCursor
и включать forced hover у HintButton
```
