# MMis — fix: anti-literal memory leakage + prompt-safe memory pipeline

## Что это за фикс

Этот документ описывает **отдельный критический фикс**, который не входит напрямую в шаги 1–7, но по факту нужен **раньше части из них**, потому что сейчас в проекте есть продуктовая ошибка:

- главная LLM может проговаривать внутреннюю память **буквально**;
- в ответ могут проскакивать формулировки уровня:
  - `пользователь чувствует грусть`
  - `пользователь раздражён`
  - `пользователь предпочитает VS Code`
- служебные memory snippets иногда используются не как внутренняя опора для тона/контекста, а как почти готовый текст ответа.

Это не просто косметика. Это ломает:

1. **естественность ответа**;
2. **иллюзию живого общения**;
3. **разделение внутренних и внешних представлений**;
4. **корректную работу памяти как latent-context, а не как шаблонной шпаргалки**.

---

## Главный симптом

Память может доезжать до главной модели в слишком сыром виде.

Пример:

В памяти лежит:

```text
artifact_type = emotional_state
text = "пользователь чувствует грусть"
```

Пользователь пишет:

```text
да не знаю уже
```

Вместо нормального ответа вроде:

```text
Поняла. Тогда давай без лишнего: начнём с самого простого шага...
```

модель может выдать что-то слишком механическое или буквальное, например:

```text
Ты чувствуешь грусть...
```

или даже почти служебную формулировку.

---

# 1. Где именно в архиве проблема проявляется

## 1.1 `core/character_runtime.py`

Проблемный узел: **в prompt-блок уходит почти сырой текст retrieved memories**.

Именно здесь память превращается в строковый блок для main LLM.

### Что это означает

Если в `ctx.retrieved_memories` лежат слишком буквальные записи, то дальше модель видит их уже как обычный текст prompt-а и может:

- повторять их почти дословно;
- путать внутреннее знание и пользовательский ответ;
- воспринимать служебную формулировку как допустимую внешнюю реплику.

### Вывод

`character_runtime.py` нельзя оставлять местом, где память просто форматируется как список строк без дополнительной фильтрации и преобразования.

---

## 1.2 `memory_core/retrieval/context_builder.py`

Проблемный момент: `emotional_state` у тебя сейчас попадает в обычный путь контекста рядом с фактами/предпочтениями.

По смыслу это неверно.

### Почему это ошибка

`emotional_state` — это не обычный recall-факт.

Это чаще всего:
- сигнал тона;
- краткоживущий признак эпизода;
- повод изменить стиль ответа;
- источник response bias;

но **не текст, который надо показывать главной модели как “факт для произнесения”**.

Если такие записи идут в generic facts, модель начинает обращаться с ними как с тем же типом контента, что и:

- имя пользователя,
- рабочий стек,
- проект,
- любимый инструмент.

Это и даёт эффект буквального проговаривания.

---

## 1.3 `memory_core/retrieval/persona_context_builder.py`

Проблемный момент: эмоциональное состояние может тащиться в persona snapshot слишком текстово.

### Почему это опасно

Если в snapshot кладётся не структура, а фраза, например:

```text
"пользователь чувствует грусть"
```

то дальше snapshot начинает работать как user-facing wording.

Хотя persona snapshot должен не рассказывать модели **что произнести**, а подсказывать **как отвечать**.

---

## 1.4 `core/response_pipeline.py`

Проблема: в pipeline уже есть policy-слой, но нет жёсткого правила уровня:

- retrieved memory — это внутренняя опора;
- не цитируй её буквально;
- не пересказывай internal labels;
- эмоции переводить в tone adaptation, а не в прямое проговаривание.

То есть даже если retrieval подаст что-то неидеально, у main prompt сейчас мало встроенной защиты.

---

## 1.5 `memory_core/config.json`

Проблема начинается ещё раньше — на этапе **генерации memory proposals**.

Сейчас memory LLM просят вернуть:

- `text`: краткая формулировка факта (1–2 предложения)

Для `emotional_state` это толкает модель к естественным фразам типа:

- `пользователь чувствует грусть`
- `пользователь разочарован`
- `пользователь устал`

То есть память сама по себе уже генерируется в форме, слишком похожей на готовую человеческую реплику.

Это плохо для внутренней системы.

---

# 2. Корень проблемы

Корневая ошибка не одна, а цепочка:

### Сейчас

```text
event
-> memory LLM proposal
-> artifact.text
-> retrieval
-> retrieved_memories
-> prompt block
-> main LLM answer
```

### И что не так

На каждом этапе используется почти один и тот же `text`.

То есть у проекта пока нет полноценного разделения между:

1. **внутренним каноническим текстом памяти**;
2. **prompt-safe представлением памяти**;
3. **поведенческой интерпретацией памяти**;
4. **разрешённым пользовательским wording**.

А без этого разделения система начинает цитировать саму себя.

---

# 3. Что надо изменить концептуально

Нужно ввести принцип:

## Память — это не готовая реплика

Память должна служить для:
- continuity,
- personalisation,
- retrieval,
- tone control,
- task awareness,

но не быть автоматически готовым текстом ответа.

### Нужно развести 4 уровня

#### 1. Canonical memory text
Внутренняя запись для хранилища.

Пример:

```text
user emotion: sad / low-energy / episode-local
```

#### 2. Prompt-safe memory view
То, что можно показать main LLM.

Пример:

```text
user currently low-bandwidth; prefer gentle concise help
```

#### 3. Behavioral cue
То, как это влияет на ответ.

Пример:

```text
reduce teasing
keep answer compact
lead with one concrete next step
```

#### 4. User-facing wording
То, что уже реально может прозвучать наружу.

Пример:

```text
Поняла. Тогда давай спокойно и по шагам.
```

Именно этого слоя сейчас не хватает.

---

# 4. Что нужно сделать по коду

## 4.1 Ввести `prompt_view` и `exposure_mode` для memory artifacts

### Цель

Чтобы не весь `artifact.text` считался автоматически пригодным для prompt-а.

### Минимальное решение

Добавить в metadata артефакта поля:

```python
metadata = {
    "prompt_view": "user currently low-bandwidth; keep tone gentle",
    "exposure_mode": "latent",
}
```

### Рекомендуемые значения `exposure_mode`

- `latent` — использовать только как скрытую опору, не цитировать
- `prompt_safe` — можно давать в main prompt как внутреннюю инструкцию
- `exact_quote` — допустимо использовать почти буквально, но только для узких случаев

### Где это применять

- `profile_fact`
- `preference`
- `task_state`
- `emotional_state`
- `identity_core`

Но особенно важно это для:
- `emotional_state`
- `identity_core`
- sensitive profile facts

---

## 4.2 Не использовать `artifact.text` как дефолт для вывода в prompt

## Проблема

Если сейчас у тебя логика построена по принципу:

```python
memory_text = row["text"]
```

или близко к этому — это и есть источник утечки.

## Правильный порядок выбора

В prompt должен идти не raw-text, а что-то вроде:

```python
memory_text = (
    row.get("prompt_view")
    or row.get("distilled_text")
    or row.get("summary")
    or ""
)
```

И только в very controlled cases допускать fallback на `text`.

### Пример безопасной логики

```python
exposure = str(row.get("exposure_mode") or "latent").strip().lower()
artifact_type = str(row.get("artifact_type") or "").strip().lower()

if exposure == "latent":
    return None

if artifact_type == "emotional_state":
    return row.get("prompt_view") or row.get("summary") or None

if artifact_type == "identity_core":
    return row.get("prompt_view") or None

return row.get("prompt_view") or row.get("summary") or None
```

То есть **по умолчанию лучше не показывать**, чем показывать сырое.

---

## 4.3 Убрать `emotional_state` из generic facts path

### Почему

Эмоция — это сигнал поведения, а не обычный факт recall-а.

### Что менять

В `memory_core/retrieval/context_builder.py` разделить логику примерно так:

### Было по смыслу

```python
elif artifact.artifact_type in {"fact", "preference", "emotional_state"}:
    facts.append(artifact)
```

### Должно стать

```python
elif artifact.artifact_type in {"fact", "profile_fact", "preference"}:
    facts.append(artifact)
elif artifact.artifact_type == "emotional_state":
    emotional_signals.append(artifact)
```

После этого:
- `facts` идут в recall / memory blocks;
- `emotional_signals` идут в `recent_user_state`, `response_bias`, `tone_hints`.

---

## 4.4 Эмоциональную память хранить структурно, а не разговорной фразой

### Плохой вариант

```python
artifact_type = "emotional_state"
text = "пользователь чувствует грусть"
summary = "грусть пользователя"
```

### Лучше

```python
artifact_type = "emotional_state"
text = "user emotion: sad / low-energy / episode-local"
summary = "sad / low-energy"
metadata = {
    "subject": "user",
    "emotion": "sad",
    "intensity": 0.72,
    "arousal": 0.18,
    "scope": "episode",
    "trigger": "discouragement_or_fatigue",
    "prompt_view": "user currently low-bandwidth; keep tone gentle and concise",
    "exposure_mode": "prompt_safe",
}
```

### Ещё лучше для части случаев

Для короткоживущих сигналов вообще можно делать:

```python
metadata = {
    "emotion": "sad",
    "low_bandwidth": True,
    "frustrated": False,
    "prompt_view": "keep answer short and steady",
    "exposure_mode": "latent",
}
```

И не тащить `text` наружу вообще.

---

## 4.5 Переписать system prompt для memory LLM

Сейчас memory LLM подталкивается к формулировкам на естественном русском языке.

Это удобно человеку, но неудобно системе.

### Что исправить

В `memory_core/config.json` нужно отдельно указать правило:

- для `emotional_state`, `identity_core`, части `task_state` и `preference`
- `text` должен быть **канонической внутренней записью**, а не готовой фразой для общения

### Пример дописываемого правила

```text
Дополнительные правила формулировки:
- Пиши text как внутреннюю каноническую запись памяти, а не как реплику для пользователя.
- Не используй формулировки вида "пользователь чувствует ...", "пользователь любит ...", если это можно выразить в более структурной форме.
- Для emotional_state предпочитай компактную внутреннюю запись, например: "user emotion: sad / low-energy / episode-local".
- Для preference предпочитай канонический формат, например: "prefers examples grounded in own code".
- Для identity_core предпочитай нормализованные формулировки, а не разговорные предложения.
- Если proposal предназначен только для скрытого влияния на стиль ответа, добавляй metadata.prompt_view и metadata.exposure_mode.
```

### Почему это важно

Если memory LLM продолжит генерировать человеческие фразы, downstream-слои всё равно будут постоянно соблазняться использовать их буквально.

---

## 4.6 Ввести отдельный memory distiller перед main prompt

Сейчас у тебя слишком короткая цепочка между retrieval и prompt.

Нужен промежуточный слой:

```text
retrieval result
-> memory distiller
-> prompt-ready memory block
```

## Что делает distiller

Он преобразует memory artifacts в:

- continuity cues
- tone hints
- task anchors
- exact recall facts

### Пример

#### В retrieval пришло

```text
artifact_type = emotional_state
text = user emotion: sad / low-energy / episode-local
```

#### Distiller должен выдать не это, а что-то вроде:

```python
{
  "tone_hints": [
    "User appears low-bandwidth this turn.",
    "Prefer calm direct help.",
  ],
  "exact_recall": [],
}
```

### Где это встраивать

Варианты:

1. отдельный модуль между retrieval и prompt build;
2. либо расширить `context_builder.py` и строить сразу несколько каналов контекста;
3. либо сделать утилиту уровня `memory_prompt_adapter.py`.

### Лучший вариант для твоего проекта

Сейчас безопаснее сделать **небольшой отдельный adapter/distiller**, а не засорять `character_runtime.py` и не перегружать retrieval бизнес-логикой presentation-слоя.

---

## 4.7 Добавить прямые anti-verbatim policy rules в `response_pipeline.py`

Даже если retrieval и distiller станут лучше, нужен **страховочный policy-слой**.

### Что добавить

Пример правил:

```python
_append_policy_rule(
    ctx.policies,
    "Retrieved memory is internal guidance, not user-facing wording. Do not quote recalled memory snippets literally unless the user explicitly asks what you remember.",
)

_append_policy_rule(
    ctx.policies,
    "Convert emotional or profile memory into tone adaptation, continuity, and response strategy — not into direct statements like 'you feel sad' or 'the user is frustrated'.",
)

_append_policy_rule(
    ctx.policies,
    "Never mention internal artifact text, retrieval metadata, memory labels, or system memory summaries in the final reply.",
)
```

### Зачем

Даже если какой-то сырой кусок памяти всё же просочится в prompt, у модели будет явное указание **не повторять его внешне**.

---

## 4.8 Persona snapshot должен получать не raw emotion text, а structured user-state

### Ошибка по смыслу

Если persona layer получает:

```text
"пользователь чувствует грусть"
```

он превращает служебную память в текстовое persona input.

### Должно быть

Persona layer должен получать примерно такое:

```python
recent_user_state = {
    "low_bandwidth": True,
    "frustrated": False,
    "emotion": "sad",
    "intensity": 0.72,
}
```

А уже из этого строить:
- mood adaptation
- warmth adjustments
- teasing reduction
- verbosity reduction

То есть **не текст**, а сигнал.

---

# 5. Какой новый контракт типов нужен

Нужно договориться, что разные типы памяти имеют разные режимы доступа.

## Пример рабочего деления

### exact recall allowed
Можно использовать почти прямо в ответе:
- `profile_fact` — имя, проект, язык, инструменты
- `task_state` — цель, прогресс, блокировки
- `preference` — только если это уместно и не звучит механически

### prompt-safe but not literal
Можно давать в prompt, но нельзя проговаривать буквально:
- `identity_core`
- `emotional_state`
- часть `preference`
- relation continuity markers

### latent only
Используются только для выбора тона/стиля/длины ответа:
- low bandwidth
- frustration markers
- vulnerability markers
- confidence of context continuity

Это нужно явно зафиксировать в коде и в документации, а не оставлять implicit.

---

# 6. Что ещё кроме literal-memory надо доделать по архиву

Ниже — дополнительные вещи, которые я бы внёс в backlog отдельно от шагов 1–7.

## 6.1 Ввести memory presentation layer

Сейчас память у тебя уже достаточно умная как storage, но presentation-слой ещё плоский.

Нужен отдельный слой, отвечающий только за вопрос:

**в каком виде memory можно показывать главной модели**.

Это отдельная ответственность и её не стоит размазывать по:
- retrieval service,
- persona builder,
- character runtime,
- prompt builder одновременно.

---

## 6.2 Развести continuity memory и answer memory

Сейчас один и тот же retrieval-result может использоваться и для continuity, и для ответа.

Но это не одно и то же.

### Нужно различать

- `continuity_only`
- `tone_only`
- `answer_support`
- `exact_recall`
- `task_anchor`

Без этого любой retrieved artifact становится кандидатом на проговаривание.

---

## 6.3 Добавить признаки чувствительности/деликатности памяти

Полезно добавить флаг уровня:

```python
metadata = {
    "sensitivity": "low | medium | high"
}
```

Чтобы:
- высокочувствительные вещи не подавались как обычные facts;
- эмоции и уязвимые состояния не озвучивались без повода;
- личные предпочтения не звучали как досье.

---

## 6.4 Добавить controlled self-recall mode

Иногда пользователь **прямо** спрашивает:
- `что ты про меня помнишь?`
- `что ты вспомнила?`

Вот там можно ослаблять ограничения и использовать более прямой recall.

Но это должен быть **явный отдельный режим**, а не поведение по умолчанию.

---

# 7. Порядок внедрения

## Этап A — быстрый safety fix

Сделать первым, без ожидания больших рефакторов.

### Внести:
1. policy rules против literal memory leakage;
2. убрать `emotional_state` из generic facts;
3. в prompt block использовать `prompt_view/summary`, а не raw `text`;
4. для `exposure_mode = latent` не показывать memory наружу вообще.

### Результат

Уже после этого у тебя резко снизится шанс фраз уровня:
- `пользователь чувствует грусть`
- `пользователь раздражён`

---

## Этап B — нормализация memory proposals

### Внести:
1. обновить `memory_core/config.json`;
2. научить memory LLM писать канонические internal-form записи;
3. начать сохранять `prompt_view` и `exposure_mode`.

### Результат

Даже новые записи памяти будут сразу system-friendly.

---

## Этап C — memory distiller

### Внести:
1. отдельный adapter/distiller;
2. каналы `tone_hints`, `continuity_hints`, `answer_support`, `exact_recall`;
3. persona получает signals, а не raw text.

### Результат

Память начнёт реально работать как когнитивный слой, а не как текстовая вставка.

---

## Этап D — тесты и инспектор

### Внести:
1. тесты на literal leakage;
2. в inspector показывать:
   - raw text,
   - prompt view,
   - exposure mode,
   - distilled output.

### Результат

Можно будет видеть, где именно память превращается в user-facing wording.

---

# 8. Какие тесты обязательно добавить

## 8.1 `test_emotional_memory_not_spoken_literally`

### Дано

В памяти есть:

```python
{
  "artifact_type": "emotional_state",
  "text": "user emotion: sad / low-energy / episode-local",
  "metadata": {
    "prompt_view": "user currently low-bandwidth; keep tone gentle",
    "exposure_mode": "prompt_safe",
  }
}
```

Пользователь пишет:

```text
да не знаю уже
```

### Ожидание

В prompt policy / distilled context есть tone guidance, но нет literal text вида:
- `пользователь чувствует грусть`
- `you feel sad`
- `the user is frustrated`

---

## 8.2 `test_latent_memory_hidden_from_prompt_block`

### Дано

Артефакт:

```python
metadata = {
  "prompt_view": "keep answer shorter",
  "exposure_mode": "latent",
}
```

### Ожидание

Он влияет на response bias, но не появляется в человекочитаемом memory block.

---

## 8.3 `test_preference_memory_not_rendered_as_dossier`

### Дано

Память:

```text
prefers examples grounded in own code
```

### Ожидание

Ответ использует примеры на коде пользователя, но не говорит:

```text
ты предпочитаешь примеры на своём коде
```

если только это не нужно по контексту.

---

## 8.4 `test_self_recall_mode_can_surface_memory`

### Дано

Пользователь спрашивает:

```text
что ты про меня помнишь?
```

### Ожидание

Система может использовать более прямой memory recall, но всё равно:
- не показывает internal labels;
- не выдаёт metadata;
- не цитирует сырой store dump.

---

# 9. Что бы я правил первым в твоём коде

Если делать это быстро и без лишней ломки, я бы шёл так:

## P0

### 1. `core/response_pipeline.py`
Добавить anti-verbatim policy rules.

### 2. `memory_core/retrieval/context_builder.py`
Убрать `emotional_state` из generic facts и перевести в `recent_user_state` / tone hints.

### 3. `core/character_runtime.py`
Перестать выводить raw `text` retrieved memories как дефолтный prompt-block.

## P1

### 4. `memory_core/config.json`
Переписать правила формата proposal для `emotional_state`, `preference`, `identity_core`.

### 5. `persona_context_builder.py`
Передавать в persona-layer signals/state, а не literal phrases.

## P2

### 6. Ввести `memory_prompt_adapter.py` или `memory_distiller.py`
Сделать нормальный presentation-layer для памяти.

---

# 10. Итог

Сейчас главный дополнительный баг в архиве такой:

## У тебя память местами хранится и подаётся слишком “человеческим текстом”

Из-за этого main LLM может:
- проговаривать внутренние записи буквально;
- звучать как диагностический лог;
- ломать ощущение естественной личности.

## Правильная архитектурная цель

Не просто “запретить одну фразу”, а сделать полноценное разделение:

```text
raw memory
!= prompt-safe memory
!= behavioral cue
!= final user wording
```

Пока этого разделения нет, подобные утечки будут возвращаться снова в разных формах.

## Практический вывод

Этот фикс надо ставить **сразу после стабилизации базовых memory шагов**, а часть его — вообще можно внедрить немедленно:

- policy guard;
- hidden/latent exposure mode;
- structured emotional memory;
- removal of emotional_state from generic facts;
- prompt-safe memory view.

Именно это даст самый быстрый эффект против фраз вида:

- `пользователь чувствует грусть`
- `пользователь раздражён`
- `пользователь любит ...`

и вообще против любого буквального проговаривания служебной памяти.
