# MMis — уменьшить и стилизовать всплывающие tooltip/описания под стиль приложения

## Что сейчас видно

На скрине вылезает белое стандартное Qt tooltip-окно с большим JSON.  
Это не твой `settings_hint_popup`, а обычный native `QToolTip`, который создаётся через:

```python
setToolTip(...)
```

В свежем архиве такие tooltip есть в `ui/settings_widgets.py`, например:

```python
self.preview.setToolTip(self._text)
self.preview.setToolTip(self._value)
self.edit_button.setToolTip("Edit")
self.arrow_button.setToolTip("Выбрать модель")
```

Главная проблема:

```python
self.preview.setToolTip(self._text)
```

Если значение большое, например JSON, Qt показывает огромную белую подсказку.

Нужно сделать 2 вещи:

```text
1. Стилизовать все QToolTip под MMis.
2. Не отдавать в tooltip полный JSON/длинный текст, а отдавать короткий preview.
```

---

## 1. Файл `ui/settings_styles.py`

### Добавить стиль для tooltip

В начало файла после `_CHEVRON_DOWN = ...` добавить:

```python
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
```

---

## 2. Файл `ui/settings_styles.py`

### Добавить функцию применения tooltip-стиля глобально

Ниже `SETTINGS_TOOLTIP_STYLE` добавить:

```python
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
```

### Зачем

`QToolTip` — это не обычный дочерний widget внутри окна.  
Иногда стиль окна не применяется к tooltip, поэтому надёжнее добавить стиль в `QApplication`.

---

## 3. Файл `ui/settings_window.py`

### Импортировать функцию

Найти:

```python
from ui.settings_styles import SETTINGS_STYLE
```

Заменить на:

```python
from ui.settings_styles import SETTINGS_STYLE, apply_settings_tooltip_style
```

---

## 4. Файл `ui/settings_window.py`

### Применить стиль при создании окна настроек

В `_build_ui()` найти:

```python
self.setStyleSheet(SETTINGS_STYLE)
```

Сразу после добавить:

```python
apply_settings_tooltip_style()
```

Итог:

```python
self.setStyleSheet(SETTINGS_STYLE)
apply_settings_tooltip_style()
```

---

## 5. Файл `ui/settings_widgets.py`

### Добавить helper для коротких tooltip

Найти функцию:

```python
def _preview_text(text: str) -> str:
```

Сразу после неё добавить:

```python
def _tooltip_preview_text(text: str, *, max_chars: int = 420, max_lines: int = 10) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    lines = raw.replace("\r", "\n").splitlines()
    clipped_lines: list[str] = []

    for line in lines[:max_lines]:
        line = line.rstrip()
        if len(line) > 120:
            line = line[:117].rstrip() + "..."
        clipped_lines.append(line)

    result = "\n".join(clipped_lines).strip()

    if len(lines) > max_lines:
        result += "\n..."

    if len(result) > max_chars:
        result = result[:max_chars].rstrip() + "..."

    return result
```

---

## 6. Файл `ui/settings_widgets.py`

### Уменьшить tooltip у `TextValueEditor`

Найти в `TextValueEditor._refresh_preview()`:

```python
self.preview.setToolTip(self._text)
```

Заменить на:

```python
self.preview.setToolTip(_tooltip_preview_text(self._text))
```

### Почему

Теперь при наведении на поле с большим JSON не будет огромного белого окна.

Будет короткий preview в стиле MMis.

---

## 7. Файл `ui/settings_widgets.py`

### Уменьшить tooltip у `ModelNameEditor`

Найти в `ModelNameEditor._refresh_preview()`:

```python
self.preview.setToolTip(self._value)
```

Заменить на:

```python
self.preview.setToolTip(_tooltip_preview_text(self._value, max_chars=120, max_lines=2))
```

---

## 8. Файл `ui/settings_widgets.py`

### Переименовать маленькие tooltip у кнопок

Найти:

```python
self.edit_button.setToolTip("Edit")
```

Заменить на:

```python
self.edit_button.setToolTip("Изменить значение")
```

Найти:

```python
self.arrow_button.setToolTip("Выбрать модель")
```

Можно оставить как есть или сделать чуть короче:

```python
self.arrow_button.setToolTip("Модели")
```

---

## 9. Файл `ui/settings_window.py`

### Сделать окно подсказки `?` чуть меньше

Это уже не белый native tooltip, а твой кастомный `SettingsHintPopup`.

Найти в `SettingsHintPopup.__init__()`:

```python
self.setFixedWidth(314)
```

Заменить на:

```python
self.setFixedWidth(286)
```

Найти:

```python
layout.setContentsMargins(12, 10, 12, 10)
layout.setSpacing(7)
```

Заменить на:

```python
layout.setContentsMargins(10, 8, 10, 8)
layout.setSpacing(5)
```

---

## 10. Файл `ui/settings_window.py`

### Ограничить ширину hint popup

Найти функцию:

```python
def _hint_popup_width(spec: SettingSpec) -> int:
```

В конце сейчас:

```python
return min(360, target)
```

Заменить на:

```python
return min(320, target)
```

И в расчёте заменить большие значения:

```python
target = max(252, title_width + 58, longest_word_width + 90, min(example_width + 58, 360))
if len(description) > 220 or len(example) > 90:
    target = max(target, 334)
elif len(description) > 150 or len(example) > 60:
    target = max(target, 314)
```

на:

```python
target = max(236, title_width + 42, longest_word_width + 62, min(example_width + 42, 300))
if len(description) > 220 or len(example) > 90:
    target = max(target, 300)
elif len(description) > 150 or len(example) > 60:
    target = max(target, 286)
```

---

## 11. Файл `ui/settings_styles.py`

### Сделать `settings_hint_popup` визуально ближе к приложению

Найти:

```css
QFrame#settings_hint_popup {
    background: rgba(15, 16, 24, 248);
    border: 1px solid rgba(139, 92, 246, 82);
    border-radius: 12px;
}
```

Заменить на:

```css
QFrame#settings_hint_popup {
    background: rgba(15, 16, 24, 242);
    border: 1px solid rgba(139, 92, 246, 70);
    border-radius: 10px;
}
```

---

## 12. Файл `ui/settings_styles.py`

### Уменьшить шрифты popup-подсказок

Найти:

```css
QLabel#hint_popup_title {
    color: #c4b5fd;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
    font-weight: 700;
}
```

Заменить на:

```css
QLabel#hint_popup_title {
    color: #c4b5fd;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 10px;
    font-weight: 700;
}
```

Найти:

```css
QLabel#hint_popup_body {
    color: #eef0f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
    line-height: 145%;
}
```

Заменить на:

```css
QLabel#hint_popup_body {
    color: #d8dee9;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 10px;
    line-height: 130%;
}
```

Найти:

```css
QLabel#hint_popup_example {
    color: #f3f4f6;
    background: rgba(139, 92, 246, 34);
    border-top: 1px solid rgba(139, 92, 246, 44);
    border-radius: 8px;
    padding: 7px 9px;
    font-family: Consolas, Cascadia Code, monospace;
    font-size: 10px;
}
```

Заменить на:

```css
QLabel#hint_popup_example {
    color: #f3f4f6;
    background: rgba(139, 92, 246, 24);
    border-top: 1px solid rgba(139, 92, 246, 34);
    border-radius: 7px;
    padding: 5px 7px;
    font-family: Consolas, Cascadia Code, monospace;
    font-size: 9px;
}
```

---

## 13. Проверка

Открыть настройки и навести на поле с большим JSON, например:

```text
Web V2 config
```

Ожидаемо:

```text
больше нет огромного белого окна
tooltip тёмный
tooltip небольшой
длинный JSON обрезан
```

Проверить кнопки:

```text
карандаш редактирования
выбор модели
dropdown модели
```

Ожидаемо:

```text
tooltip тёмный
размер компактный
стиль совпадает с приложением
```

Проверить `?` возле параметров:

```text
popup стал чуть меньше
фон тёмный
граница фиолетовая
шрифт меньше
```

---

## Итог

Главное исправление:

```python
self.preview.setToolTip(self._text)
```

заменить на:

```python
self.preview.setToolTip(_tooltip_preview_text(self._text))
```

И глобально применить:

```css
QToolTip {
    background-color: rgba(15, 16, 24, 248);
    color: #eef0f6;
    border: 1px solid rgba(139, 92, 246, 82);
}
```

После этого стандартные белые tooltip исчезнут.
