# STUDIO WORKFLOW (RU)

Этот документ описывает единый модуль `studio` для всех операций с персонажами и modes.

## 1. Команды (публичный контракт)
Поддерживаются только команды:

- `/studio`
- `/studio start [seed]`
- `/studio status`
- `/studio apply`
- `/studio cancel`
- `/apply`
- `/cancel`

Удаленные команды:

- `/specs ...` -> `Команда удалена. �спользуй /studio ...`
- `/studio mode ...` -> `Команда удалена. �спользуй /studio ...`

Важно:

- `/studio apply|cancel` и `/apply|/cancel` обрабатываются локально.
- Эти команды не попадают в основной prompt/chat pipeline.
- После `apply`/`cancel` studio закрывается в этом же ходе (`active=false`).

## 2. Runtime state (single source of truth)
Ключ состояния:

- `studio_generator`

Базовый контракт:

```json
{
  "active": false,
  "phase": "seed_capture",
  "seed_prompt": "",
  "operation_type": "",
  "targets": {
    "character_ids": [],
    "mode_ids": [],
    "scope": "mixed"
  },
  "draft_changes": {
    "characters": {},
    "modes": {},
    "set_active_character": ""
  },
  "confidence_map": {},
  "unresolved_fields": [],
  "step_options": {},
  "last_llm": {},
  "last_result": {}
}
```

## 3. Единый workflow
Р¤Р°Р·С‹:

1. `seed_capture`
2. `clarify`
3. `review`
4. `done`

`operation_type`:

- `create_character`
- `update_character`
- `update_modes`
- `mixed`
- `build_character_pack`

## 4. Intent matrix для character
В `seed_capture` выполняется явный анализ интента по персонажу:

- если в seed явно `create/new/создай/новый` -> `create_character` (без лишнего вопроса create/update);
- если явно `update/change/обнови/измени` -> `update_character` (без лишнего вопроса);
- если запрос смешанный (персонаж + mode) -> `mixed`;
- если интент неоднозначный -> в `clarify` задается вопрос `operation_type`.

## 5. Smart выбор `target_character_id`
Для `target_character_id` используются разные стратегии:

- `create_character`/`build_character_pack`: 3-5 кандидатов `character_id` (LLM + детерминированный fallback), нормализованные в lower `snake_case/kebab`.
- `update_character`/`update_modes`/`mixed`: список существующих `character_id` из `data/specs/characters/*`, отсортированный по релевантности seed.
- если уже есть однозначный target в seed, вопрос может быть пропущен.

Во всех шагах выбора есть `1..N` и `0. Ввести свой вариант`.

Для mode-запросов:

- если в seed есть формат `для <character>` и нет явного `global`, scope автоматически повышается до `character`;
- если в seed одновременно есть `global` и `для <character>`, scope поднимается до `mixed`;
- при `scope=character|mixed` обязательно резолвится `target_character_id`.

## 6. Strict `llm_profile` (dynamic enum)
�сточник значений:

- `config.model_config.PROFILES.keys()`

Правила:

- в шаге `llm_profile` показывается полный актуальный список профилей;
- ручной выбор `0. Ввести свой вариант` для `llm_profile` отключен;
- валидация принимает только значения из текущего списка `PROFILES`;
- safe fallback на `BALANCED` (или первый доступный профиль), если профиль исчез между шагами.

## 7. Формат блока `Что влияет`
Каждый clarify-вопрос содержит единый блок:

- `�зменит: ...`
- `Файлы: ...`
- `Для персонажа: ...`
- `Результат: ...`

Для character-шагов явно указываются целевые файлы:

- `data/specs/characters/<id>/character.json`
- `data/specs/characters/<id>/persona_state.json`
- `data/specs/characters/<id>/persona_spec.json`

Для mode-шагов:

- `data/specs/rules_for_all/taxonomy.json`
- `data/specs/rules_for_all/modes_spec.json`
- плюс `persona_spec.json` персонажа при scope `character|mixed`.

## 8. Apply политика

- Для существующих JSON используется patch-merge, не full rewrite.
- Неизвестные поля сохраняются.
- Удаление (`remove`) выполняется только при явном интенте.
- Существующие `mode` обновляются без дублирования.
- Существующие `character_id` обновляются точечно, без создания дубля.
- Для `create_character`/`update_character`/`mixed` всегда пишутся:
  - specs (`character.json`, `persona_state.json`, `persona_spec.json`, `evolution_spec.json`);
  - `studio_blueprint.json`;
  - artifacts через встроенный pack-механизм (`create_or_update`).
- Для `update_modes` artifacts/blueprint не создаются, если нет character-ветки операции.

## 9. Примеры seed-запросов

- `создай персонажа luna, режим helper, профиль QUALITY`
- `обнови asya: default_mode=engineer, llm_profile=FAST`
- `добавь mode analyst глобально`
- `добавь mode analyst для asya`
- `обнови asya и mode engineer`
- `собери build pack для nova`

## 10. Поведение после `apply/cancel`

- На `apply` при наличии `unresolved_fields` применение блокируется, studio остается активной.
- На успешный `apply` или `cancel` studio закрывается сразу.
- Следующее обычное сообщение пользователя идет напрямую в основной chat route.


