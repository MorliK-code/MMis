# MMis — фикс размера `SegmentedSelector` для `Ollama start mode`

## Проблема

У выбора провайдера:

```text
ollama / openai / auto
```

переключатель выглядит нормально, потому что текста достаточно много и текущий `min-width=84` почти не мешает.

А у `Ollama start mode`:

```text
serve / ui
```

текста мало, но `SegmentedSelector` всё равно принудительно держит минимум `84px`, поэтому справа и слева появляется лишнее пустое место.

Причина в `ui/settings_widgets.py`:

```python
self.setMinimumWidth(84)
...
def minimumSizeHint(self) -> QSize:
    return QSize(84, 24)
```

Нужно сделать минимальную ширину динамической, зависящей от текста.

---

## 1. Файл `ui/settings_widgets.py`

### Найти класс

```python
class SegmentedSelector(QWidget):
```

---

## 2. Внутри `SegmentedSelector` заменить константы

### Было

```python
_segment_padding = 20.0
```

### Сделать

```python
_segment_padding = 20.0
_frame_padding = 6.0
_segment_gap = 1.0
_min_segment_width = 24.0
```

---

## 3. В `__init__()` убрать жёсткий minimumWidth

### Найти

```python
self.setMinimumWidth(84)
self.setFixedHeight(24)
```

### Заменить на

```python
self.setMinimumWidth(self.minimumSizeHint().width())
self.setFixedHeight(24)
```

---

## 4. Добавить helper расчёта ширины

Внутри класса `SegmentedSelector`, перед `sizeHint()`, добавить:

```python
def _natural_width(self) -> int:
    if not self._choices:
        return 48

    fm = QFontMetricsF(self._font)
    width = self._frame_padding

    for choice in self._choices:
        segment_width = fm.horizontalAdvance(choice) + self._segment_padding
        width += max(self._min_segment_width, segment_width)

    width += max(0, len(self._choices) - 1) * self._segment_gap
    return int(width + 0.999)
```

---

## 5. Заменить `sizeHint()`

### Было

```python
def sizeHint(self) -> QSize:
    fm = QFontMetricsF(self._font)
    width = 0
    for choice in self._choices:
        width += int(fm.horizontalAdvance(choice) + self._segment_padding + 0.999)
    width += 6 + max(0, len(self._choices) - 1)
    return QSize(max(width, 84), 24)
```

### Должно быть

```python
def sizeHint(self) -> QSize:
    return QSize(self._natural_width(), 24)
```

---

## 6. Заменить `minimumSizeHint()`

### Было

```python
def minimumSizeHint(self) -> QSize:
    return QSize(84, 24)
```

### Должно быть

```python
def minimumSizeHint(self) -> QSize:
    return QSize(self._natural_width(), 24)
```

---

## 7. В `_segment_rects()` заменить расчёт ширины сегментов

### Найти

```python
available = max(1.0, self.width() - 6.0 - max(0, len(self._choices) - 1))
natural = [float(fm.horizontalAdvance(choice) + self._segment_padding) for choice in self._choices]
```

### Заменить на

```python
gap_total = max(0, len(self._choices) - 1) * self._segment_gap
available = max(1.0, self.width() - self._frame_padding - gap_total)
natural = [
    float(max(self._min_segment_width, fm.horizontalAdvance(choice) + self._segment_padding))
    for choice in self._choices
]
```

---

## 8. В `_segment_rects()` заменить добавление gap

### Найти

```python
x += width + 1.0
```

### Заменить на

```python
x += width + self._segment_gap
```

---

## 9. В `RightAlignedSelectorContainer` ничего не растягивать вручную

Оставить как есть:

```python
width = max(self.selector.minimumSizeHint().width(), min(self.selector.sizeHint().width(), self.width()))
self.selector.setGeometry(QRect(max(0, self.width() - width), 0, width, 24))
```

После изменения `minimumSizeHint()` он сам начнёт получать правильную ширину.

---

## Итог

После фикса:

```text
Provider: ollama / openai / auto
```

останется нормальным.

А:

```text
Ollama start mode: serve / ui
```

станет компактным, как выбор провайдера, без лишнего пустого места по бокам.

---

## Проверка

Открыть:

```text
Настройки -> LLM / Models -> Ollama
```

Проверить:

```text
Ollama start mode
```

Ожидаемо:

- переключатель не растянут;
- `serve` и `ui` сидят компактно;
- расстояние до рамок такое же визуально, как у выбора провайдера;
- выбор провайдера не сломался.
