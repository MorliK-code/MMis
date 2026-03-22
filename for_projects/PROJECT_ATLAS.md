# MMis Atlas: Karta Proekta (chto / zachem / chto mozhno delat)

Etot fail - zhivaya navigatsionnaya karta koda. Ideya prostaya:
- **Chto eto**: rol faila.
- **Zachem eto**: zachem on voobshe nuzhen v arhitekture.
- **Chto mozhno delat**: praktichnye shagi, kotorye mozhno bezopasno vnosit.

---

## 0) Skelet papok

```text
MMis/
  api/ config/ core/ llm/ memory/ metadata/
  modules/ prompt_engine/ prompts/
  ui/ utils/ tests/
  data/ models/
```

---

## 1) Kornevaya paluba (`/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `main.py` | Bootstrap i launcher rezhimov (`api/ui/cli/voice`) | Sobiraet DI-konteiner i zapuskaet prilozhenie | Dobavit novyy rezhim zapuska, prokinut novye zavisimosti |
| `api_main.py` | Uproshchennaya tochka vhoda dlya API | Bystryy start API bez CLI vetok | Derzhat sovmestimost so starimi skriptami |
| `ui_console.py` | Konsolnyy UI-klient k API | Operativnyy chat, komandy i diagnostika | Dobavit komandy (`/character`, `/trait`, `/json`, `/think`) |
| `ui_pyside6.py` | Vhod dlya desktop UI | Zapusk graficheskogo interfeisa | Menyat tolkо startup-obvyazku |
| `requirements.txt` | Zavisimosti Python | Povtoryaemaya ustanovka sredy | Fiksirovat versii, dobavlyat novye pakety |
| `.gitignore` | Pravila ignorirovaniya artefaktov | Chistota repozitoriya | Dobavit runtime puti (`data/cache`, logi) |

---

## 2) API paluba (`api/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `api/app.py` | FastAPI-prilozhenie, endpointy `/chat`, `/chat/stream`, `/models`, `/thinking`, `/json-mode`, `/metadata` | Edinaya HTTP-tochka vhoda dlya UI/console | Dobavit endpointy upravleniya personazhami/traits |
| `api/schemas.py` | Pydantic-modeli request/response | Kontrakt API i validatsiya vhoda | Rasshiryat kontrakty (`character`, `traits`, profile toggles) |
| `api/__init__.py` | Paketnyy modul | Chisty import API paketa | Ostavit minimalnym |

---

## 3) Config paluba (`config/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `config/settings.py` | AppSettings + zagruzka `.env/config.json` | Tsentr vseh feature-flag i putey | Dobavlyat novye flagi (napr. character defaults) |
| `config/model_profiles.py` | Profili generatsii (`FAST/BALANCED/QUALITY/ECONOM`) | Upravlenie temp/top_p/ctx i apparatnymi ogranicheniami | Nastroit profili pod konkretnoe zhelezo |
| `config/paths.py` | Centralizovannye puti (`data`, `models`, `memory`) | Ustranyaet razbros putey po kodu | Dobavit puti dlya novyh subsystem |
| `config/logging_config.py` | Setup loggerov i file handlers | Edinyy format logov | Dobavit JSON-logger i rotaцию |
| `config/__init__.py` | Re-export konfig-obektov | Udobny import config API | Derzhat tonkim |

---

## 4) Core paluba (`core/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `core/brain.py` | Orkestrator obrabotki soobshcheniy (`handle_message`) | Marshrutizatsiya + state + pipeline + persist | Dobavit novye tipy memory_ops/ui_actions |
| `core/response_pipeline.py` | Stadii preprocess/plan/personality/memory/prompt/generate/postprocess/verify | Razdelenie logiki po etapam | Podklyuchat novye stage i profile-vykl/inkl |
| `core/__init__.py` | Lazy-export core klassov | Ustranenie ciklicheskih importov | Derzhat lazy i ne delat eager-importov |

---

## 5) LLM paluba (`llm/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `llm/provider_base.py` | Obshchiy kontrakt providerov (request/response/tool_calls/usage/timing) | Core ne zavisit ot konkretnogo API | Dobavit novye capability pola |
| `llm/ollama_provider.py` | Adapter pod Ollama API | Lokalnaya generatsiya + options + json_mode | Tuning timeout/retry/stream/tool parsing |
| `llm/openai_provider.py` | Adapter pod OpenAI API | Rabota s hosted modelyami po tomu zhe kontraktu | Vklyuchit structured outputs/tool-calling |
| `llm/tokenizer.py` | Otsenka tokenov i truncation strategii | Kontrol budget do vyzova modeli | Dobavit real tokenizer konkretnoi modeli |
| `llm/__init__.py` | Fabrika providera | Vybor backenda (`ollama/openai/auto`) | Rasshirit strategiyu auto-vybora |

---

## 6) Memory paluba (`memory/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `memory/memory_manager.py` | Orkestrator short/long/vector/profile/event | Edinaya tochka ingest/retrieve/write_facts | Dobavit score fusion i rerank |
| `memory/short_memory.py` | Bystroe okno poslednih soobshcheniy | Kontekst dlya prompt tail | Menyat limit/rolling summary pravila |
| `memory/long_memory.py` | Dolgosrochnye text docs/summaries | Hranenie vazhnyh vospominaniy | Dobavit importance decay |
| `memory/vector_store.py` | Vektornyy poisk (abstraction) | Relevatny retrieval po query | Podklyuchit Chroma/FAISS/Qdrant backend |
| `memory/fact_extractor.py` | Izvlechenie strukturirovannyh faktov | Obnovlenie profile-store i pamyati | Rasshirit patterny i quality-mode |
| `memory/profile_store.py` | Versii profilnyh faktov (user/assistant) | Konflikt-rezolyutsiya i istoriya izmeneniy | Dobavit confirm/workflow dlya sensitive factov |
| `memory/event_store.py` | Immutable event-log (`jsonl`) | Debug, audit, vosstanovlenie | Dobavit range/index/filter API |
| `memory/__init__.py` | Paketnyy export | Udobny import storage sloev | Derzhat tonkim |

---

## 7) Metadata paluba (`metadata/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `metadata/metadata_extractor.py` | Orkestrator metadata (lang/intent/emotion/tags/entities/safety) | Signal dlya prompt, memory, style | Dobavit model-assisted extraction rezhim |
| `metadata/language_detector.py` | Language detector + flags (url/code/emoji) | Bazovy kontekst i routing hints | Uluchshit mixed-lang heuristics |
| `metadata/intent_classifier.py` | Klassifikator intenta | Vybor strategii otveta/режima | Dobavit intenty pod novi workflow |
| `metadata/emotion_detector.py` | Detector nastroeniya / tonality | Adaptatsiya stilya i character evolution | Podklyuchit ONNX classifier |
| `metadata/tagger.py` | Finalny tagging-layer | Tehnicheskie i tematicheskie flagi | Rasширять topic/tone tags |
| `metadata/__init__.py` | Lazy-export metadata API | Ustranenie import-cycle | Ne perevodit v eager import |

---

## 8) Character paluba (`modules/character/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `modules/character/storage.py` | Chitaet/pishet `data/characters/*`, sozdaet defaults | Fizicheskoe hranenie personazha, traits, rules, prompts | Dobavit migratsii i backup/restore |
| `modules/character/evaluator.py` | Proverka usloviy rules (`intent/emotion/mode/tags/trait thresholds`) | Reshaet, primenyaetsya li pravilo evolution | Rasshirit DSL usloviy |
| `modules/character/composer.py` | Sobiraet character prompt (base+mood+trait overlays) | Vliyanie personazha na final system block | Dobavit rank overlays po importance |
| `modules/character/engine.py` | Glavny CharacterEngine (update/set/remove/list traits, conflicts, decay, cleanup) | Dinamicheskoe izmenenie haraktera vo vremeni | Dobavit bolshe konflikt-pravil i avto-learn pattern |
| `modules/character/__init__.py` | Export API character subsystem | Tochka vhoda dlya core/prompt_engine | Derzhat yavnoe API sloya |

---

## 9) Moduli-instrumenty (`modules/automation`, `modules/internet`, `modules/screen`, `modules/voice`)

### `modules/automation/`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `browser_controller.py` | Browser actions abstraction | Tool execution bez bizness-logiki | Dobavit safety allowlist i confirmations |
| `os_actions.py` | OS-level actions (files/commands/hotkeys) | Bazovye "ruki" assistenta | Uzhestochit safe-path i command policy |
| `task_executor.py` | Step-by-step task executor | Ispolnenie planov s logami i retry | Dobavit rollback i compensation steps |
| `__init__.py` | Export automation API | Ediny import instrumentov | Derzhat tonkim |

### `modules/internet/`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `search.py` | Structured web search client (SearxNG-first) | Aktualnaya informatsiya i snippety | Derzhat strict endpoint i fallback policy |
| `scraper.py` | Fetch + readable extraction + cleaner integration | Poluchenie chistogo teksta po URL | Kontrolirovat limity i metadata ochistki |
| `content_cleaner.py` | Trafilatura + BS4 boilerplate cleaner | Udalyaet cookie/ads/comments/banner/popups | Rasshiryat noise-patterny pod novye saity |
| `__init__.py` | Export internet API | Udobnaya integratsiya v tools | Derzhat tonkim |

Runbook (Docker web stack):
- `docker compose up --build`
- API: `http://127.0.0.1:8000/health`
- SearxNG: `http://127.0.0.1:8080/search?q=test&format=json`
- Env for API in compose: `MMIS_SEARCH_PROVIDER=searxng`, `MMIS_SEARCH_STRICT_ENDPOINT=true`, `MMIS_SEARCH_API_URL=http://searxng:8080/search?format=json`

### `modules/screen/`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `ocr.py` | OCR extraction + find | Chitaet tekst s ekrana | Dobavit bbox confidence filter |
| `screen_analyzer.py` | Screen context heuristics | UI-context dlya brain/tools | Dobavit app/window classifier |
| `__init__.py` | Export screen API | Integratsiya analyzera | Derzhat tonkim |

### `modules/voice/`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `tts.py` | Text-to-Speech servisy | Ozvuchivanie otvetov | Dobavit cache/audio post-fx |
| `stt.py` | Speech-to-Text servisy | Vvod golosom | Dobavit streaming partials |
| `voice_manager.py` | State machine listening/speaking | Anti-echo, queue, orchestration | Dobavit barge-in rules |
| `__init__.py` | Export voice API | Podklyuchenie voice moduля | Derzhat tonkim |

### `modules/__init__.py`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `modules/__init__.py` | Aggregator modulей | Import odnoi strokoj (`from modules import ...`) | Podderzhivat aktualny export list |

---

## 10) Prompt Engine paluba (`prompt_engine/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `prompt_engine.py` | Sbor final messages iz blokov i budgeta | Tsentr kompozitsii system/user context | Menyat prioritety bucketov |
| `prompt_registry.py` | Registry key -> prompt file | Versioned dostup k promptam | Registrirovat novye prompt keys |
| `prompt_loader.py` | Loader + frontmatter parser + cache/hotreload | Chitaet prompts kak assets | Dobavit checksum + reload watchers |
| `prompt_versioning.py` | Utility dlya versiy promptov | Audit/reproducibility | Dobavit migration map prompt IDs |
| `token_budget_manager.py` | Raschet budgeta i shrink strategii | Ukladyvanie v context window | Menyat weighting i reserve policy |
| `__init__.py` | Lazy-export prompt engine API | Ustranenie import-cycle | Ostavit lazy |

---

## 11) Prompt Assets paluba (`prompts/`)

### `prompts/system/`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `base_system.txt` | Bazovye pravila assistenta | Fundament safety/behavior | Menyat global policy text |


| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `personality_default.txt` | Legacy persona default | Sovmestimost starogo persona sloya | Derzhat krotkim i stabilnym |
| `personality_flirty.txt` | Legacy flirty persona | Stil dlya flirt/relation mode | Podkrutit granitsy tona |
| `personality_strict.txt` | Legacy strict persona | Stil dlya coding/task mode | Utochnit format step-by-step |
| `rules_default.txt` | Obshchie pravila | Edinaya policy-vstavka | Dobavit zaprety i format constraints |
| `style_default.txt` | Legacy style default | Nastroika tonalnosti | Menyat mikrostil otveta |
| `style_flirty.txt` | Legacy style flirty | Variativnost igrivogo tona | Smyagchit ili usilit ton |
| `style_strict.txt` | Legacy style strict | Tehnicheskiy ton | Podnyat formalnost |
| `style_supportive.txt` | Legacy supportive style | Podderzhivaushchiy ton | Kalibrovat empathy intensity |

### `prompts/memory/`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `fact_extraction.txt` | Prompt extraction faktov | Kachestvo struct-facts | Menyat schema i primery |
| `memory_cleanup.txt` | Prompt ochistki memory | Zaschita ot musora | Usilit dedupe pravila |
| `memory_merge.txt` | Prompt merge konfliktov | Konsolidatsiya dublikatov | Menyat prioritety istochnikov |

### `prompts/metadata/`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `emotion_detection.txt` | Prompt dlya emotional labels | Ogranichivaet dopustimye metki | Menyat slovari/emotion set |
| `intent_classification.txt` | Prompt dlya intent labels | Ogranichivaet intent namespace | Dobavit novye intent klasсы |
| `tagging.txt` | Prompt dlya tagger signalov | Upravlyaet final tags | Rasshirit topic/tone tags |

### `prompts/automation/`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `browser_action.txt` | Policy dlya browser tools | Bezopasnost klikov/deystviy | Dobavit allow/deny patterns |
| `os_action.txt` | Policy dlya OS actions | Ogranichenie opasnyh komand | Uzhestochit rule-set |

### `prompts/response/`

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `safety_filter.txt` | Pravila bezopasnogo output | Filtr risk-kontenta | Menyat fallback povедение |
| `formatting.txt` | Pravila formata otveta | Konsistentnost stylya/JSON | Menyat schema format |

---

## 12) UI paluba (`ui/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `ui/app.py` | Osnovnoy desktop UI (chat, stream, thinking, metrics) | Vizualny front-end k API | Dobavit panell char/traits controls |
| `ui/api_client.py` | HTTP/NDJSON client | Kanal svyazi s backend | Dobavit metody `/characters` endpointov |
| `ui/workers.py` | Worker-thread dlya stream requestov | Ne blokirovat UI glavnym potokom | Dobavit cancellation/backpressure |
| `ui/chat_sessions.py` | Session persistence | Istoriya chatov po sessiyam | Dobavit profile-bound sessions |
| `ui/chat_window.py` | Legacy/alt chat window | Sovmestimost vnutri UI sloya | Uprostit ili vypilit posle migratsii |
| `ui/metrics.py` | UI metrika i otobrazhenie | Monitoring generation speed/usage | Dobavit char-specific metrics |
| `ui/voice_adapter.py` | Adapter voice->ui | Integratsiya golosa v UI | Dobavit vad state indicators |
| `ui/constants.py` | Konstanta UI | Edinye defaults | Chistit dedup constants |
| `ui/config.py` | UI-konfig | Lokальные nastroiki okna/tema | Dobavit feature toggles |
| `ui/livecss.py` | Runtime stylesheet reloader | Bystraia iteratsiya dizaina | Dobavit safe fallback style |
| `ui/style.css` | Baza stylizatsii UI | Vneshny vid aplikatsii | Nastroit visual language |
| `ui/assets/.gitkeep` | Placeholder assets dir | Sohranenie papki v git | Kladit shrifty/ikonki |
| `ui/components/.gitkeep` | Placeholder components dir | Mesto pod razdelennye UI widgets | Vynesti krupnye section widgets |
| `ui/styles/style.css` | Dopolnitelny style layer | Razdelenie visual-temy | Variativnye temy/profili |
| `ui/widgets/controls.py` | Reusable controls | Pereispolzuemye UI komponenty | Dobavit toggle dlya character |
| `ui/widgets/filters.py` | Filter widgets | Filtratsiya kontenta/metrik | Dobavit metadata filters |
| `ui/widgets/__init__.py` | Export widgets | Udobny import widgets | Derzhat tonkim |

---

## 13) Utils paluba (`utils/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `utils/logger.py` | Ediny logger helper | Konsistentnye log-formaty | Dobavit structured fields |
| `utils/timers.py` | Timer/context managers | Zamery latency | Vynesti prometheus adapter |
| `utils/metrics.py` | Counters/gauges/hist snapshots | Runtime telemetriya bez tyazhelyh zavisimostey | Dobavit periodic dump |
| `utils/validators.py` | Input/output validators | Safety i schema-checks | Rasshirit validators dlya tools/traits |
| `utils/__init__.py` | Export util API | Udobny import helpers | Derzhat tonkim |

---

## 14) Tests paluba (`tests/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `test_brain_smoke.py` | Smoke test orkestratora | Bystro lovit regress po osnovnomu potoku | Rasshirit scenario routing |
| `test_event_store.py` | Test event store I/O | Nadezhnost zhurnala sobytiy | Dobavit range/filter cases |
| `test_llm_provider_contract.py` | Kontrakt provider_base | Garant obyazatelnogo response format | Pokryt stream/tool edge cases |
| `test_memory_conflicts.py` | Konflikty profile facts | Korrektnaya evolutsiya profile values | Dobavit sensitive confirmation cases |
| `test_metadata.py` | Metadata extraction checks | Stabilnost intent/emotion/lang | Dobavit mixed-lang set |
| `test_personality_switching.py` | Legacy persona switching | Sovmestimost starogo behavior | Postepenno migririvat v character tests |
| `test_prompt_budget.py` | Prompt token budget | Prompt ne vyhodit za limity | Menyat budget weights i proverki |
| `test_prompt_registry_metadata.py` | Prompt frontmatter/registry | Validnost prompt docs | Dobavit version conflict tests |
| `test_response_pipeline_unwrap.py` | Safety JSON unwrap logic | Pravilny output pri json_mode on/off | Dobavit tool-json regression |
| `test_token_economy.py` | Token economy logic | Otrezanie blokov po prioritetam | Dobavit reserve stress cases |
| `test_utils_infra.py` | Utils infra smoke | Bazovaya stabilnost util-sloya | Rasshirit validator coverage |
| `test_character_engine.py` | Character engine add/evolve/remove traits | Garant novoy subsystem logiki | Dobavit konflikty i cleanup matrix |
| `assets/ui.py` | Legacy test asset UI | Ruchnye/visual eksperimenti | Ne ispolzovat kak production code |
| `assets/ui1.py` | Legacy test asset UI var1 | Eksperimentalnye varianty | Derzhat otdelno ot boevogo UI |
| `assets/ui3.py` | Legacy test asset UI var3 | Prototipy states animation | Mozhno vynesti v docs/demo |
| `assets/ui4.py` | Legacy test asset UI var4 | Prototipy interactions | Chistit pri stabilizatsii |

---

## 15) Data paluba (`data/`) — runtime dанные

### `data/characters/`

| File/Dir | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `manifest.json` | Spisok personazhey + aktivnyj | Vybor personazha bez koda | Dobavit novye personazhi |
| `asya/character.json` | Pasport personazha | Statichnye nastroiki haraktera | Menyat default mood/llm_profile |
| `asya/state.json` | Tekushee sostoyanie personazha | Mood + active/disabled traits | Sbrasyvat ili migrirovat state |
| `asya/traits/builtin.json` | Bazovye traits | Startovyy harakter personazha | Kalibrovat nachalnye znacheniya |
| `asya/traits/learned.json` | Dynamically learned traits | Evolutsiya vo vremeni | Chistit perelozhennye/ustarevshie traits |
| `asya/rules/evolution.json` | Pravila evolutsii traits/mood | Avtomatika izmeneniy po kontekstu | Dobavit `when/apply` scenarii |
| `asya/prompts/base.txt` | Bazovyy voice personazha | Osnova style-instruktsiy | Menyat global tone personazha |
| `asya/prompts/moods/*.txt` | Mood-specific overlays | Pereklyuchenie podaci po nastroeniyu | Dobavit novye moods |
| `asya/prompts/overlays/*.txt` | High-intensity overlays | Aktsent pri vysokih trait values | Kalibrovat porogi i tekst |
| `asya/prompts/traits/*.txt` | Trait-specific style snippets | Lokalen vklad trait v rechi | Dobavit trait->prompt map |
| `asya/events.jsonl` | Zhurnal sobytiy personazha | Audit evolutsii i debug | Analizirovat regressii haraktera |

### `data/memory_storage/`

| File/Dir | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `brain_state.json` | Legacy monolit state snapshot | Fallback/sovmestimost state | Ispolzovat dlya backup |
| `brain_state_store/*.json` | Novoe split-state hranilishche | Otdelny fail na kazhduy harakteristiku | Delat point-fix bez perезапisi vsego state |
| `events.jsonl` | Obschiy event log | Audit pipeline/tool/store operatsiy | Retro-analitika i repro bagov |
| `short_memory.json` | Short memory buffer | Prompt tail konteks | Tuning limit summary |
| `long_memory_docs.json` | Long memory docs | Dolgaya pamyat | Ochistka/klasterizatsiya |
| `vector_store.json` | Vektornye zapisi | Retrieval relevatnyh kuskov | Reindex posle migratsii embeddings |
| `user_profile_store.json` | User profile facts + versions | Personalization | Pravka konfliktov factov |
| `assistant_profile_store.json` | Assistant profile facts + versions | Self-profile behavior | Kontrol state drift |
| `metadata/*/metadata_messages.jsonl` | Metadata zhurnaly po modeli | Analitika intent/emotion/tags | Svodki i quality review |
| `metadata/*/metadata_messages.json` | To zhe v JSON masssive | Udobnyy rucnoi razbor | Eksport dlya dashboards |

### `data/logs/` i `data/cache/`

| File/Dir | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `logs/app.log` | Obschiy app log | Operativny debug | Iskat exceptions i latency spikes |
| `logs/llm.log` | LLM-vyzovy | Diagnostika model behavior | Smotret timeout/retry trend |
| `logs/memory.log` | Memory subsystem log | Debug ingest/retrieve | Iskat konflikty fact merge |
| `logs/tools.log` | Tool execution log | Bezopasnost i audit actions | Proveryat opasnye deystviya |
| `logs/ui.log` | UI events log | UI-regression triage | Otladka stream/thinking |
| `cache/` | Runtime cache folder | Uskorenie povtorov | Vvesti TTL/limity size |

---

## 16) Models paluba (`models/`)

| File/Dir | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `models/embeddings/manifest.example.json` | Primer manifest embeddings model | Dokumentiruet format model metadata | Sozdat boevoy manifest dlya real model |
| `models/classifiers/manifest.example.json` | Primer manifest classifier model | Format pod intent/emotion/lang local models | Podklyuchit ONNX classifier bundle |
| `models/quantized/manifest.example.json` | Primer manifest quantized model | Profiling pod CPU/GPU presety | Zavesti per-device builds |
| `models/metadata/config.py` | Lokalnye configi metadata model | Sovmestimost s plugin-like metadata sloem | Ostavit otdelno ot core metadata |
| `models/metadata/extractor.py` | Lokalny extractor (legacy/alt) | Eksperimentalny sloy metadata | Migririvat ili vypilit po strategii |
| `models/metadata/__init__.py` | Paket export metadata model | Udobny import | Derzhat tonkim |

---

## 17) Legacy settings paluba (`settings/`)

| File | Chto eto | Zachem eto | Chto mozhno delat |
|---|---|---|---|
| `settings/config.py` | Legacy config modul | Sovmestimost starogo koda | Planovo svesti v `config/` |
| `settings/model_config.py` | Legacy model config | Sovmestimost pre-refactor | Migririvat v `config/model_profiles.py` |
| `settings/__init__.py` | Legacy package export | Chisty import v perehodny period | Ostavit minimalnym |

---

## 18) Prakticheskie scenarii (chto mozhno delat uzhe seychas)

1. **Dobavit novogo personazha bez Python-koda**:
   - Sozdat papku `data/characters/<id>/` po obrazcu `asya/`.
   - Dopolnit `data/characters/manifest.json`.
   - Perezapusit API/UI i vybrat `/character <id>`.

2. **Nastroit evolyutsiyu haraktera**:
   - Redaktirovat `rules/evolution.json` (`when` + `apply`).
   - Smotret effekt v `events.jsonl` i v metadata logah.

3. **Ruchno upravlyat trait-ami v runtime**:
   - `/trait list`
   - `/trait set sarcasm 0.8`
   - `/trait remove romance`

4. **Tonko kalibrovat styl rechi**:
   - Menyat tekst v `prompts/base.txt`, `prompts/moods/*.txt`, `prompts/traits/*.txt`.

5. **Audit i debug regressiy**:
   - Sravnit `data/memory_storage/metadata/*` + `data/characters/*/events.jsonl` + `data/logs/*.log`.

---

## 19) Note po hygiene

- `data/` i `models/` - runtime/artefakty: ne smeshivat s business-logikoy.
- `core/` dolzhen ostavatsya orchestration-only.
- `modules/*` - realizatsiya "kak delat", ne "chto reshat".
- Dlya novyh subsystem: snachala storage kontrakt -> potom engine -> potom pipeline integration.

---

Eto atlas-versiya `v1`. Posle lyuboy krupnoy restrukturizatsii obnovlyay etot fail v pervuyu ochered.

