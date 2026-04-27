# MMis — кнопки `Сбросить`/`Сохранить` + курсор рукой + фикс съедания `Настройки MMis`

## Что надо исправить

1. У кнопки **Сохранить** сделать 3 состояния цвета:
   - обычное;
   - hover / наведена;
   - pressed / нажата.

2. На кнопки **Сбросить** и **Сохранить** добавить cursor `PointingHandCursor`.

3. Исправить съедание текста `Настройки MMis` в верхней панели.

---

# 1. Файл `ui/settings_window.py`

## 1.1. Исправить верхний заголовок `Настройки MMis`

Найти в `_build_ui()`:

```python
title_col = QVBoxLayout()
title = QLabel("Настройки MMis")
title.setObjectName("settings_title")
subtitle = QLabel("Полупрозрачное окно поверх основного UI - config/settings.py + config.json")
subtitle.setObjectName("settings_subtitle")
title_col.addWidget(title)
title_col.addWidget(subtitle)
top_layout.addLayout(title_col, 1)
```

Заменить на:

```python
title_col = QVBoxLayout()
title_col.setContentsMargins(0, 0, 0, 0)
title_col.setSpacing(1)

title = QLabel("Настройки MMis")
title.setObjectName("settings_title")
title.setMinimumHeight(22)
title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

subtitle = QLabel("Полупрозрачное окно поверх основного UI - config/settings.py + config.json")
subtitle.setObjectName("settings_subtitle")
subtitle.setMinimumHeight(17)
subtitle.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

title_col.addWidget(title)
title_col.addWidget(subtitle)
top_layout.addLayout(title_col, 1)
```

### Зачем

Сейчас `QVBoxLayout` сам ужимает высоту title/subtitle внутри `settings_top` на 58px, и текст `Настройки MMis` визуально режется сверху/снизу.  
Фикс задаёт нормальную минимальную высоту и выравнивание.

---

## 1.2. Добавить курсор рукой на `Сбросить` и `Сохранить`

Найти:

```python
reset_btn = QPushButton("Сбросить")
reset_btn.clicked.connect(self.reload)
self.save_btn = QPushButton("Сохранить")
self.save_btn.setObjectName("primary_button")
self.save_btn.clicked.connect(self._save)
```

Заменить на:

```python
reset_btn = QPushButton("Сбросить")
reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
reset_btn.clicked.connect(self.reload)

self.save_btn = QPushButton("Сохранить")
self.save_btn.setObjectName("primary_button")
self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
self.save_btn.clicked.connect(self._save)
```

---

# 2. Файл `ui/settings_styles.py`

## 2.1. Исправить высоту/line-height заголовка

Найти:

```css
QLabel#settings_title {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 16px;
    font-weight: 700;
}
```

Заменить на:

```css
QLabel#settings_title {
    color: #f3f4f6;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 15px;
    font-weight: 700;
    min-height: 22px;
    padding: 0px;
    margin: 0px;
}
```

### Почему `15px`

На скрине видно, что `16px` для текущей высоты topbar и Cascadia Code чуть цепляет верх/низ.  
`15px + min-height 22px` выглядит почти так же, но не режется.

---

## 2.2. Исправить subtitle

Найти:

```css
QLabel#settings_subtitle, QLabel#settings_muted {
    color: #8f96a3;
    font-family: Cascadia Code, Consolas, monospace;
    font-size: 11px;
}
```

Заменить на:

```css
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
```

### Зачем

Сейчас `settings_subtitle` и `settings_muted` связаны одним стилем.  
Лучше разделить: subtitle в верхней панели должен быть компактнее, а остальные muted-тексты не трогать.

---

# 3. Файл `ui/settings_styles.py`

## 3.1. Сделать обычное состояние кнопки `Сохранить`

Найти текущий блок:

```css
QPushButton#primary_button {
    background: rgba(139, 92, 246, 46);
    border-color: rgba(139, 92, 246, 76);
    color: #ede9fe;
}
```

Заменить на:

```css
QPushButton#primary_button {
    background: rgba(139, 92, 246, 42);
    border: 1px solid rgba(139, 92, 246, 70);
    color: #ede9fe;
}
```

---

## 3.2. Добавить hover состояние

Сразу после блока `QPushButton#primary_button` добавить:

```css
QPushButton#primary_button:hover {
    background: rgba(139, 92, 246, 70);
    border: 1px solid rgba(167, 139, 250, 112);
    color: #ffffff;
}
```

---

## 3.3. Добавить pressed состояние

Сразу после hover добавить:

```css
QPushButton#primary_button:pressed {
    background: rgba(109, 40, 217, 95);
    border: 1px solid rgba(196, 181, 253, 140);
    color: #f5f3ff;
    padding-top: 1px;
    padding-bottom: 0px;
}
```

---

## 3.4. На всякий случай добавить disabled состояние

После pressed добавить:

```css
QPushButton#primary_button:disabled {
    background: rgba(139, 92, 246, 18);
    border: 1px solid rgba(139, 92, 246, 34);
    color: rgba(237, 233, 254, 90);
}
```

---

# 4. Файл `ui/settings_styles.py`

## 4.1. Добавить hover/pressed для обычной кнопки `Сбросить`

У тебя базовый стиль для `QPushButton, QToolButton` уже есть, и общий hover тоже есть:

```css
QPushButton:hover, QToolButton:hover {
    background: rgba(139, 92, 246, 28);
    border-color: rgba(139, 92, 246, 54);
    color: #f3f4f6;
}
```

После него добавить pressed:

```css
QPushButton:pressed, QToolButton:pressed {
    background: rgba(139, 92, 246, 44);
    border-color: rgba(139, 92, 246, 82);
    color: #ffffff;
    padding-top: 1px;
    padding-bottom: 0px;
}
```

### Важно

Этот блок должен стоять **до** `QPushButton#primary_button`, чтобы `primary_button:pressed` мог его переопределить.

Если поставишь после `primary_button:pressed`, он может перебить стиль кнопки `Сохранить`.

---

# 5. Проверка

## Проверить заголовок

Открыть настройки.

Ожидаемо:

```text
Настройки MMis
```

больше не режется сверху/снизу.

Subtitle остаётся под ним:

```text
Полупрозрачное окно поверх основного UI - config/settings.py + config.json
```

---

## Проверить кнопки

Навести на:

```text
Сбросить
Сохранить
```

Ожидаемо:

```text
курсор становится рукой
```

---

## Проверить `Сохранить`

Состояния:

```text
обычная     — мягкий фиолетовый
hover       — ярче
pressed     — темнее/насыщеннее, лёгкое смещение вниз
disabled    — бледная
```

---

# Итог

Главные правки:

```python
reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
```

и:

```css
QPushButton#primary_button
QPushButton#primary_button:hover
QPushButton#primary_button:pressed
```

Плюс для заголовка:

```python
title.setMinimumHeight(22)
subtitle.setMinimumHeight(17)
```
