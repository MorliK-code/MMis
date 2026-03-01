# Описание тестов

Ниже кратко описано, что проверяет каждый тест в папке `tests/`.

## tests/test_brain_smoke.py
- `test_brain_handles_chat_and_command`  
  Проверяет smoke-сценарий `Brain`: обработка обычного сообщения, команды (`/nothink`) и следующего сообщения без падений; у ответов есть текст, а роут команды корректный (`command` или `chat`).

## tests/test_cache_usage.py
- `test_disk_ttl_cache_roundtrip`  
  Проверяет, что `DiskTTLCache` сохраняет значение на диск и другой инстанс кеша читает его обратно с теми же данными.
- `test_search_client_uses_disk_cache_between_instances`  
  Проверяет, что `SearchClient` использует дисковый кеш между разными инстансами: второй инстанс получает результат из кеша без сетевого вызова.

## tests/test_character_engine.py
- `test_defaults_created_and_listed`  
  Проверяет, что дефолтный персонаж создается и доступен (`asya`), и что активный персонаж по умолчанию тоже `asya`.
- `test_update_applies_chat_rule_and_builds_prompt`  
  Проверяет, что `update()` применяет rule-изменение трейтов для chat-сценария (например, `playfulness` не уменьшается) и формирует непустой prompt-блок с маркером `[CHAR_BASE]`.
- `test_set_and_remove_trait`  
  Проверяет установку трейта, его появление в списке и корректное удаление.

## tests/test_dialog_policies.py
- `test_is_user_greeting_simple`  
  Проверяет, что простое приветствие распознается как `user_greeting=True`.
- `test_is_user_greeting_exclusion`  
  Проверяет исключение: фразы вида «передай привет» не считаются приветствием пользователя.
- `test_technical_with_greeting_disables_smalltalk`  
  Проверяет, что в техническом сообщении с ошибкой (`Traceback`) smalltalk отключается и приветствие не разрешается.
- `test_long_technical_message`  
  Проверяет детект технического текста и отключение smalltalk на длинном тех-сообщении.
- `test_json_output_cases`  
  Проверяет набор кейсов из JSON-файла в `tests/output/dialog_policies_nametest/cases_20260301.json`: порядок `test_num`, прогон `compute_dialog_flags` и соответствие ожидаемым флагам.
- `test_auto_creates_data_dialog_police_config`  
  Проверяет автосоздание `data/dialog_police.py` при вызове `compute_dialog_flags`.
- `test_local_date_and_region_use_system_timezone`  
  Проверяет, что `local_date` берется из локальной системной таймзоны ПК и что `local_region` не пустой.

## tests/test_event_store.py
- `test_append_get_range_persist`  
  Проверяет `EventStore`: добавление событий, получение по `event_id`, выборки `last/range` и сохранение данных между инстансами (персистентность в файле).

## tests/test_greeting_policy.py
- `test_user_greeting_has_priority_even_in_continuing_smalltalk`  
  Проверяет приоритет пользовательского приветствия: если пользователь здоровается, `allow_greeting=True` даже в `continuing_smalltalk`.
- `test_new_session_allows_greeting_when_not_greeted_today`  
  Проверяет, что в новой сессии при отсутствии приветствия за текущий день автоматическое приветствие разрешается.
- `test_continuing_smalltalk_blocks_auto_greeting`  
  Проверяет, что в состоянии `continuing_smalltalk` авто-приветствие блокируется, даже если сессия формально новая.
- `test_postprocess_strips_leading_greeting_when_disallowed`  
  Проверяет post-filter: если `allow_greeting=False`, стартовое приветствие/вежливая вводная удаляются, оставляя ответ по делу.

## tests/test_llm_payload_logging.py
- `test_ollama_logs_exact_request_payload`  
  Проверяет, что для Ollama в лог (`llm_request_payload`) пишется ровно тот payload, который отправлен в клиент (`chat(**kwargs)`), включая `messages`, их роли/порядок и `system_message`.
- `test_openai_logs_exact_request_payload`  
  Аналогично для OpenAI: лог содержит точный payload `chat.completions.create(...)`, а также корректные `messages`, `message_roles`, `message_order`, `system_message`.

## tests/test_llm_provider_contract.py
- `test_generate_returns_llmresponse`  
  Проверяет контракт провайдера: `generate()` возвращает `LLMResponse` с текстом, валидными usage/timings и корректной структурой данных.

## tests/test_memory_conflicts.py
- `test_birth_year_update_overwrites_current_and_keeps_history`  
  Проверяет разрешение конфликтов в памяти: новое значение факта (`birth_year`) становится текущим, а история сохраняет обе версии.

## tests/test_metadata.py
- `test_extract_balanced_basic_tags`  
  Проверяет, что `MetadataExtractor` в BALANCED-режиме возвращает язык, intent, соответствующий intent-тег и словарь entities.
- `test_extract_fast_mode_is_lightweight`  
  Проверяет, что в FAST-режиме извлечение упрощено (entities пустой), но базовые языковые теги присутствуют.

## tests/test_personality_switching.py
- `test_auto_switch_to_strict_for_coding_intent`  
  Проверяет автопереключение personality на `strict` для coding-intent с достаточной уверенностью.
- `test_cooldown_prevents_switch`  
  Проверяет cooldown: слишком раннее повторное переключение блокируется.
- `test_manual_override_locks_personality`  
  Проверяет ручной override: принудительный выбор personality выполняется и включает lock после переключения.

## tests/test_prompt_budget.py
- `test_prompt_builder_respects_total_budget`  
  Проверяет, что `PromptBuilder` укладывает собранный prompt в общий токен-бюджет (`full_prompt <= total_tokens`) при длинном контексте.

## tests/test_prompt_registry_metadata.py
- `test_frontmatter_is_parsed`  
  Проверяет парсинг frontmatter промпта: `id`, `version`, `tags`, `min_ctx`, а также отсутствие сырого `---` в текстовом теле.

## tests/test_response_hygiene.py
- `test_hygiene_removes_service_lines_and_prefixes`  
  Проверяет очистку служебных строк (`thinking>`, `assistant>`, `user>`, `[model:]`, `[thinking]`) и нормализацию ответа.
- `test_hygiene_dedupes_adjacent_repeats`  
  Проверяет удаление подряд идущих дубликатов одинаковых фраз.
- `test_pipeline_applies_hygiene_in_postprocess`  
  Проверяет, что пайплайн действительно применяет hygiene-постобработку к сырому ответу модели.
- `test_pipeline_replaces_smalltalk_echo`  
  Проверяет замену пустого эхо-smalltalk ответа на более содержательный шаблон.
- `test_pipeline_replaces_generic_short_echo`  
  Проверяет замену слишком короткого/эхо-ответа на нормализованный полезный ответ.

## tests/test_response_pipeline_unwrap.py
- `test_unwraps_safety_output_json`  
  Проверяет распаковку safety-JSON: из `{"output": ...}` извлекается текст для пользователя.
- `test_keeps_json_when_json_mode_enabled`  
  Проверяет, что при `json_mode=True` JSON не распаковывается и возвращается как есть.
- `test_unwraps_json_even_with_thinking_block`  
  Проверяет распаковку JSON, даже если после него идет блок `<think>...</think>`.
- `test_unwraps_in_fast_profile_too`  
  Проверяет, что распаковка работает и в профиле качества FAST.

## tests/test_token_economy.py
- `test_fit_context_blocks_respects_total_minus_reserve`  
  Проверяет `TokenBudgetManager`: итоговые токены не превышают доступный бюджет (`total - reserve`), и обязательный блок пользователя сохраняется.

## tests/test_ui_console_streaming.py
- `test_thinking_first_buffers_answer_until_thinking`  
  Проверяет режим `prefer_thinking_first=True`: ответ буферизуется, сначала выводится thinking, потом assistant-часть.
- `test_no_thinking_preference_streams_answer_immediately`  
  Проверяет режим без приоритета thinking: ответ стримится сразу.
- `test_renderer_extracts_output_from_safety_json_stream`  
  Проверяет инкрементальный парсинг safety-JSON в стриме и извлечение поля `output`.
- `test_sanitize_stream_text_removes_carriage_returns`  
  Проверяет нормализацию `\r` в потоковом тексте.

## tests/test_utils_infra.py
- `test_metrics_and_timer`  
  Проверяет базовые метрики и таймер: счетчики/гейджи/гистограммы корректно обновляются.
- `test_measure_time_backward_compat`  
  Проверяет обратную совместимость `measure_time()` (возвращаемый elapsed > 0).
- `test_validators`  
  Проверяет валидацию tool-call схемы, clamp параметров генерации и безопасное построение пути внутри базовой директории.
- `test_logger_helpers`  
  Проверяет, что вспомогательные функции логгирования конфигурируются и принимают JSON/обычные сообщения без ошибок.
