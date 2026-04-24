# План внедрения окна настроек MMis

## 1. Цель
Сделать отдельное окно настроек поверх основного PySide6 UI: слегка прозрачное, в стиле текущего чата, с группами параметров, поиском, подсказками и примерами значений.

## 2. Группы настроек
- Main config: `app`, `startup`, `api`, `paths`, `hardware`.
- LLM / модели: `llm.provider`, `model_name`, `thinking_enabled`, `json_mode_enabled`, `task_models`, `providers.ollama`, `providers.openai`.
- Memory core: `memory`, `memory_core`, retrieval, lifecycle, scoring, context budget, summary.
- Web / Internet: `internet.enabled`, `web_mode`, `web_v2`, search, fetch, trust policy, citations.
- Voice: STT, TTS, voice mode, devices, model names, input/output paths.
- Dialog / UI: greeting policy, console flags, show thinking, runtime toggles.
- Debug / Inspector: memory inspector, raw scores, filtered items, prompt blocks.
- Logging: log level, file, channels, web trace.
- Safety / tools: `safety_mode`, automation/screen flags, prompt filters.

## 3. UX
- Окно открывается по кнопке `⚙` в left rail.
- Фон чата остаётся видимым, затемняется и блюрится.
- Слева список категорий.
- В центре карточки параметров.
- Справа live preview: текущее состояние, JSON diff, предупреждения.
- У каждого параметра есть `?` с описанием и примером.
- Изменённые поля подсвечиваются.
- Опасные параметры вынесены в отдельную красную секцию.

## 4. Техническая реализация
1. Создать `ui/settings_window.py`.
2. Создать модель описания параметров `ui/settings_schema.py`.
3. Читать значения через `get_config_payload(force_reload=True)`.
4. Сохранять изменения через `update_config_values(updates)`.
5. Для вложенных ключей использовать dotted path: `llm.model_name`, `memory.retrieval.top_k`.
6. После сохранения обновлять runtime-состояние, если параметр применяется без перезапуска.
7. Для параметров, требующих перезапуска, показывать бейдж `нужен рестарт`.

## 5. Первые файлы для добавления
- `ui/settings_window.py`
- `ui/settings_schema.py`
- `ui/settings_widgets.py`
- `ui/settings_styles.py`

## 6. Что сделать первым
1. Реализовать окно без сохранения, только чтение `config.json`.
2. Добавить группы и поиск.
3. Добавить tooltips.
4. Добавить сохранение простых значений.
5. Добавить проверку типов.
6. Добавить предупреждения для опасных параметров.
