# MMis — чтобы у всех знаков `?` были нормальные описания

## Что исправляем

В настройках каждый знак `?` берёт текст из `SettingSpec` и/или словаря описаний в `ui/settings_window.py`.

Сейчас часть новых настроек может показывать общий fallback вроде:

```text
Управляет параметром ...
```

Это не подходит. Нужно сделать так, чтобы **каждый `?` имел понятное человеческое описание**, особенно новые настройки Ollama:

```text
ui.ollama.start_mode
ui.ollama.serve_exe
ui.ollama.models_dir
```

---

## 1. Файл `ui/settings_window.py`

### Найти словарь

```python
_TITLE_BY_PATH: dict[str, str] = {
```

В него добавить новые русские названия:

```python
    "ui.ollama.start_mode": "Режим запуска Ollama",
    "ui.ollama.serve_exe": "Путь к Ollama serve",
    "ui.ollama.models_dir": "Папка моделей Ollama",
```

Если там ещё осталось старое поле:

```python
"ui.ollama.server_exe"
```

его лучше убрать или заменить на:

```python
"ui.ollama.serve_exe"
```

---

## 2. Файл `ui/settings_window.py`

### Найти словарь

```python
_DESCRIPTION_BY_PATH: dict[str, str] = {
```

Добавить описания для новых настроек:

```python
    "ui.ollama.start_mode": (
        "Выбирает способ автозапуска Ollama, если её API сейчас недоступен. "
        "serve запускает указанный ollama.exe с аргументом serve. "
        "ui выполняет команду ollama list, чтобы Ollama сама подняла API-контур."
    ),
    "ui.ollama.serve_exe": (
        "Путь к ollama.exe, который будет использоваться в режиме serve. "
        "MMis запустит его как '<ollama.exe> serve'. "
        "Оставь пустым, если ollama доступна из PATH."
    ),
    "ui.ollama.models_dir": (
        "Папка, где Ollama должна искать и хранить модели. "
        "При запуске MMis передаёт этот путь через переменную окружения OLLAMA_MODELS."
    ),
```

Если ещё осталось старое описание:

```python
"ui.ollama.server_exe"
```

заменить ключ на:

```python
"ui.ollama.serve_exe"
```

---

## 3. Файл `ui/settings_schema.py`

### В карточке `Ollama` у новых `SettingSpec` тоже должны быть description

Проверить, чтобы настройки были примерно такими:

```python
SettingSpec(
    "ui.ollama.start_mode",
    "Ollama start mode",
    kind="select",
    options=("serve", "ui"),
    description=(
        "serve запускает Ollama API через '<ollama.exe> serve'. "
        "ui запускает через 'ollama list'."
    ),
    example="serve",
    live=True,
),
SettingSpec(
    "ui.ollama.serve_exe",
    "Ollama serve exe",
    description=(
        "Путь к ollama.exe для режима serve. "
        "Если оставить пустым, будет использоваться команда ollama из PATH."
    ),
    example="C:\\Users\\user\\AppData\\Local\\Programs\\Ollama\\ollama.exe",
    live=True,
),
SettingSpec(
    "ui.ollama.models_dir",
    "Ollama models dir",
    description=(
        "Папка моделей Ollama. Передаётся в окружение как OLLAMA_MODELS."
    ),
    example="D:\\ollama\\models",
    live=True,
),
```

---

## 4. Файл `ui/settings_window.py`

### Сделать проверку покрытия описаний

Добавить функцию рядом с `_hint_description()` или ниже `_spec_for()`:

```python
def _validate_hint_coverage() -> None:
    missing_titles: list[str] = []
    missing_descriptions: list[str] = []

    for category in SETTINGS_CATEGORIES:
        for card in category.cards:
            for spec in card.settings:
                title = str(_TITLE_BY_PATH.get(spec.path) or spec.title or "").strip()
                description = str(_DESCRIPTION_BY_PATH.get(spec.path) or spec.description or "").strip()

                if not title:
                    missing_titles.append(spec.path)

                if not description:
                    missing_descriptions.append(spec.path)

    if missing_titles or missing_descriptions:
        lines: list[str] = []
        if missing_titles:
            lines.append("Нет title для: " + ", ".join(missing_titles))
        if missing_descriptions:
            lines.append("Нет description для: " + ", ".join(missing_descriptions))

        print("[settings hints] " + " | ".join(lines))
```

---

## 5. Файл `ui/settings_window.py`

### Вызвать проверку один раз при создании окна

В `SettingsWindow.__init__()` после:

```python
self._build_ui()
self.reload()
```

добавить:

```python
_validate_hint_coverage()
```

Итог:

```python
self._build_ui()
self.reload()
_validate_hint_coverage()
```

---

## 6. Файл `ui/settings_window.py`

### Усилить fallback, чтобы сразу было видно забытое описание

Найти:

```python
def _hint_description(spec: SettingSpec) -> str:
    text = str(_DESCRIPTION_BY_PATH.get(spec.path) or spec.description or "").strip()
    if not text:
        text = f"Управляет параметром {spec.path}. Значение применяется при сохранении настроек."
    if spec.dangerous:
        text = f"{text}\n\nОпасный параметр: меняй только если понимаешь последствия."
    return text
```

Заменить на:

```python
def _hint_description(spec: SettingSpec) -> str:
    text = str(_DESCRIPTION_BY_PATH.get(spec.path) or spec.description or "").strip()

    if not text:
        text = (
            f"Описание для {spec.path} ещё не задано. "
            "Добавь его в _DESCRIPTION_BY_PATH или в SettingSpec.description."
        )

    if spec.dangerous:
        text = f"{text}\n\nОпасный параметр: меняй только если понимаешь последствия."

    return text
```

### Зачем

Так сразу будет видно, где описание забыли, а не будет казаться, что подсказка нормальная.

---

## 7. Файл `ui/settings_window.py`

### Чтобы `?` показывал описание даже при клике, а не только при hover

Сейчас у тебя уже есть:

```python
hint.hovered.connect(...)
hint.activated.connect(...)
```

Проверить, что это осталось:

```python
hint.hovered.connect(lambda _button, button=hint: self._show_hint_popup(button, self._hint_targets[button]))
hint.activated.connect(lambda _button, button=hint: self._show_hint_popup(button, self._hint_targets[button]))
hint.unhovered.connect(lambda _button: self._hide_hint_popup())
```

Если `activated` нет — добавить. Тогда ЛКМ по `?` тоже будет открывать описание.

---

## 8. Добавить быструю проверку через отдельный скрипт

Создать файл:

```text
tools/check_settings_hints.py
```

С содержимым:

```python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.settings_schema import SETTINGS_CATEGORIES
from ui.settings_window import _DESCRIPTION_BY_PATH, _TITLE_BY_PATH


def main() -> int:
    missing_title: list[str] = []
    missing_description: list[str] = []

    for category in SETTINGS_CATEGORIES:
        for card in category.cards:
            for spec in card.settings:
                title = str(_TITLE_BY_PATH.get(spec.path) or spec.title or "").strip()
                description = str(_DESCRIPTION_BY_PATH.get(spec.path) or spec.description or "").strip()

                if not title:
                    missing_title.append(spec.path)

                if not description:
                    missing_description.append(spec.path)

    if missing_title:
        print("Нет title:")
        for path in missing_title:
            print("  -", path)

    if missing_description:
        print("Нет description:")
        for path in missing_description:
            print("  -", path)

    if missing_title or missing_description:
        return 1

    print("OK: у всех настроек есть title и description для подсказок.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Запуск:

```cmd
python tools\check_settings_hints.py
```

Ожидаемо:

```text
OK: у всех настроек есть title и description для подсказок.
```

---

## 9. Что проверить вручную

Открыть настройки и пройтись по категориям:

```text
Main config
LLM / Models
Memory Core
Web / Internet
Voice
Dialog / UI
Debug / Inspector
Logging
Safety / Tools
```

У каждого `?` должно быть:

```text
1. нормальное название параметра
2. понятное описание
3. пример или список вариантов, если он есть
4. restart required / dangerous / updated бейджи при необходимости
```

Особенно проверить:

```text
Автозапуск Ollama
Ollama start mode
Ollama serve exe
Ollama models dir
Ollama URL
Ollama timeout
Ollama retries
```

---

## Итог

После правки:

- каждый знак `?` показывает нормальное описание;
- новые настройки Ollama больше не показывают пустую/общую подсказку;
- при добавлении новых настроек можно быстро проверить покрытие через `tools/check_settings_hints.py`;
- забытые описания будут сразу видны в консоли и в popup.
