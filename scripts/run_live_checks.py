from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import load_config
from core.brain import Brain
from core.character_runtime import CharacterRuntime
from llm import build_provider
from llm.provider_base import LLMProviderBase
from memory.memory_manager import MemoryManager
from memory.memory_models import DocumentIngestRequest, MemoryScope


def _configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _slugify(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9а-яё]+", "-", text, flags=re.I)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "scenario"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    text = str(value or "").strip()
    return [text] if text else []


def _contains_casefold(text: str, needle: str) -> bool:
    return str(needle or "").casefold() in str(text or "").casefold()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig") or "null")


def _parse_scope(value: str) -> MemoryScope:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    for row in MemoryScope:
        if row.value == token:
            return row
    raise ValueError(f"Unsupported memory scope: {value}")


@dataclass(frozen=True)
class LiveCheckDocument:
    source: str
    text: str
    title: str = ""
    scope: str = "project"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, base_dir: Path) -> "LiveCheckDocument":
        row = _as_dict(raw)
        text = str(row.get("text") or "").strip()
        source = str(row.get("source") or row.get("path") or row.get("title") or "document").strip()
        if not text:
            path_text = str(row.get("path") or "").strip()
            if not path_text:
                raise ValueError("Document entry requires either 'text' or 'path'")
            path = Path(path_text)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            text = path.read_text(encoding="utf-8-sig")
            if not source:
                source = str(path)
        return cls(
            source=source or "document",
            text=text,
            title=str(row.get("title") or "").strip(),
            scope=str(row.get("scope") or "project").strip() or "project",
            metadata=_as_dict(row.get("metadata")),
        )


@dataclass(frozen=True)
class LiveCheckTurn:
    text: str
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    expect_contains: list[str] = field(default_factory=list)
    expect_any_contains: list[str] = field(default_factory=list)
    expect_not_contains: list[str] = field(default_factory=list)
    expect_regex: list[str] = field(default_factory=list)
    expect_not_regex: list[str] = field(default_factory=list)
    expect_route: str = ""

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "LiveCheckTurn":
        row = _as_dict(raw)
        text = str(row.get("text") or row.get("user") or "").strip()
        if not text:
            raise ValueError("Turn entry requires 'text'")
        return cls(
            text=text,
            label=str(row.get("label") or row.get("name") or "").strip(),
            meta=_as_dict(row.get("meta")),
            expect_contains=_as_string_list(row.get("expect_contains")),
            expect_any_contains=_as_string_list(row.get("expect_any_contains")),
            expect_not_contains=_as_string_list(row.get("expect_not_contains")),
            expect_regex=_as_string_list(row.get("expect_regex")),
            expect_not_regex=_as_string_list(row.get("expect_not_regex")),
            expect_route=str(row.get("expect_route") or "").strip(),
        )


@dataclass(frozen=True)
class LiveCheckScenario:
    name: str
    description: str = ""
    conversation_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    documents: list[LiveCheckDocument] = field(default_factory=list)
    setup_turns: list[LiveCheckTurn] = field(default_factory=list)
    turns: list[LiveCheckTurn] = field(default_factory=list)
    fresh_brain_per_turn: bool = True
    persist_query_turns: bool = False

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, base_dir: Path) -> "LiveCheckScenario":
        row = _as_dict(raw)
        name = str(row.get("name") or row.get("id") or "").strip()
        if not name:
            raise ValueError("Scenario entry requires 'name'")
        return cls(
            name=name,
            description=str(row.get("description") or "").strip(),
            conversation_id=str(row.get("conversation_id") or "").strip(),
            meta=_as_dict(row.get("meta")),
            documents=[
                LiveCheckDocument.from_raw(item, base_dir=base_dir)
                for item in list(row.get("documents") or [])
            ],
            setup_turns=[
                LiveCheckTurn.from_raw(item)
                for item in list(row.get("setup_turns") or row.get("seed_turns") or [])
            ],
            turns=[LiveCheckTurn.from_raw(item) for item in list(row.get("turns") or [])],
            fresh_brain_per_turn=bool(row.get("fresh_brain_per_turn", row.get("fresh_query_process", True))),
            persist_query_turns=bool(row.get("persist_query_turns", False)),
        )


@dataclass(frozen=True)
class TurnCheckResult:
    ok: bool
    failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TurnRunResult:
    text: str
    route: str
    status: str
    model: str
    thinking: str = ""
    logs: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    memory_ops: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": str(self.text or ""),
            "route": str(self.route or ""),
            "status": str(self.status or ""),
            "model": str(self.model or ""),
            "thinking": str(self.thinking or ""),
            "logs": [str(x) for x in list(self.logs or [])],
            "stats": dict(self.stats or {}),
            "memory_ops": [dict(x) for x in list(self.memory_ops or [])],
            "error": str(self.error or ""),
        }


@dataclass(frozen=True)
class ScenarioRunResult:
    name: str
    conversation_id: str
    work_dir: str
    ok: bool
    turns_run: int
    documents_ingested: int
    failures: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name or ""),
            "conversation_id": str(self.conversation_id or ""),
            "work_dir": str(self.work_dir or ""),
            "ok": bool(self.ok),
            "turns_run": int(self.turns_run),
            "documents_ingested": int(self.documents_ingested),
            "failures": [str(x) for x in list(self.failures or [])],
            "transcript": [dict(x) for x in list(self.transcript or [])],
        }


def _evaluate_expectations(*, turn: LiveCheckTurn, result: TurnRunResult) -> TurnCheckResult:
    failures: list[str] = []
    text = str(result.text or "")

    if str(result.error or "").strip():
        failures.append(f"turn returned error: {result.error}")

    for needle in list(turn.expect_contains or []):
        if not _contains_casefold(text, needle):
            failures.append(f"missing substring: {needle}")

    if turn.expect_any_contains and not any(
        _contains_casefold(text, needle) for needle in list(turn.expect_any_contains or [])
    ):
        failures.append(
            "missing any-of substrings: " + ", ".join(str(x) for x in list(turn.expect_any_contains or []))
        )

    for needle in list(turn.expect_not_contains or []):
        if _contains_casefold(text, needle):
            failures.append(f"forbidden substring present: {needle}")

    for pattern in list(turn.expect_regex or []):
        if not re.search(pattern, text, re.I | re.S):
            failures.append(f"regex did not match: {pattern}")

    for pattern in list(turn.expect_not_regex or []):
        if re.search(pattern, text, re.I | re.S):
            failures.append(f"forbidden regex matched: {pattern}")

    expected_route = str(turn.expect_route or "").strip().lower()
    if expected_route and str(result.route or "").strip().lower() != expected_route:
        failures.append(f"route mismatch: expected {expected_route}, got {result.route}")

    return TurnCheckResult(ok=not failures, failures=failures)


def _scenario_file_scenarios(path: Path) -> list[LiveCheckScenario]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        rows = list(payload.get("scenarios") or [])
    elif isinstance(payload, list):
        rows = list(payload)
    else:
        raise ValueError(f"Unsupported scenario file payload in {path}")
    return [LiveCheckScenario.from_raw(item, base_dir=path.parent) for item in rows]


def _builtin_scenarios() -> dict[str, list[LiveCheckScenario]]:
    def scenario(
        *,
        name: str,
        description: str,
        conversation_id: str,
        setup_turns: list[LiveCheckTurn] | None = None,
        turns: list[LiveCheckTurn] | None = None,
        documents: list[LiveCheckDocument] | None = None,
    ) -> LiveCheckScenario:
        return LiveCheckScenario(
            name=name,
            description=description,
            conversation_id=conversation_id,
            setup_turns=list(setup_turns or []),
            turns=list(turns or []),
            documents=list(documents or []),
            fresh_brain_per_turn=True,
            persist_query_turns=False,
        )

    facts = LiveCheckScenario(
        name="facts_exact_recall",
        description="Exact self recall for name, age, GPU, RAM, Python and OS.",
        conversation_id="live-facts",
        setup_turns=[
            LiveCheckTurn(text="Меня зовут Паша."),
            LiveCheckTurn(text="Мне уже 22 года."),
            LiveCheckTurn(text="У меня RTX 3050 Ti, 32 ГБ ОЗУ, Python 3.11 и Windows 11."),
        ],
        turns=[
            LiveCheckTurn(
                text="Как меня зовут?",
                expect_any_contains=["Паша"],
                expect_not_contains=["не помню", "не вижу в памяти", "как посмотреть"],
            ),
            LiveCheckTurn(
                text="Сколько мне лет?",
                expect_any_contains=["22"],
                expect_not_contains=["не помню", "не вижу в памяти"],
            ),
            LiveCheckTurn(
                text="Какая у меня видеокарта?",
                expect_any_contains=["RTX 3050 Ti", "3050 Ti", "RTX 3050", "3050"],
                expect_not_contains=["не помню", "не вижу в памяти", "wmic", "как посмотреть", "проверь сам"],
            ),
            LiveCheckTurn(
                text="Сколько у меня ОЗУ?",
                expect_any_contains=["32"],
                expect_not_contains=["Паша", "не помню", "winver", "python --version", "как посмотреть"],
            ),
            LiveCheckTurn(
                text="Какой у меня Python?",
                expect_any_contains=["3.11"],
                expect_not_contains=["не помню", "python --version", "как посмотреть"],
            ),
            LiveCheckTurn(
                text="Какая у меня ОС?",
                expect_any_contains=["Windows 11", "Windows"],
                expect_not_contains=["не помню", "winver", "как посмотреть"],
            ),
        ],
        fresh_brain_per_turn=True,
        persist_query_turns=False,
    )
    claim_uses = LiveCheckScenario(
        name="claims_uses_recall",
        description="Claim recall for tools the user uses.",
        conversation_id="live-claims-uses",
        setup_turns=[LiveCheckTurn(text="Я использую VS Code как основной редактор.")],
        turns=[
            LiveCheckTurn(
                text="Какой редактор я использую?",
                expect_any_contains=["VS Code", "vscode"],
                expect_not_contains=["не помню", "проверь сам"],
            )
        ],
        fresh_brain_per_turn=True,
        persist_query_turns=False,
    )
    claim_owns = LiveCheckScenario(
        name="claims_owns_recall",
        description="Claim recall for things the user owns.",
        conversation_id="live-claims-owns",
        setup_turns=[LiveCheckTurn(text="У меня холодильник Samsung.")],
        turns=[
            LiveCheckTurn(
                text="Какой у меня холодильник?",
                expect_any_contains=["Samsung"],
                expect_not_contains=["не помню", "проверь сам"],
            )
        ],
        fresh_brain_per_turn=True,
        persist_query_turns=False,
    )
    dialog = LiveCheckScenario(
        name="dialog_contextual_recall",
        description="Contextual recall for what was discussed and decided.",
        conversation_id="live-dialog",
        setup_turns=[
            LiveCheckTurn(text="Давай не хранить мысли ассистента в памяти."),
            LiveCheckTurn(text="Сначала доделываем память, потом веб."),
        ],
        turns=[
            LiveCheckTurn(
                text="Что мы решили по памяти?",
                expect_any_contains=[
                    "не хранить мысли ассистента",
                    "не сохранять мысли ассистента",
                    "мысли ассистента не хранить",
                ],
            ),
            LiveCheckTurn(
                text="Какой у нас план дальше?",
                expect_contains=["пам"],
                expect_any_contains=["веб", "web"],
            ),
        ],
        fresh_brain_per_turn=True,
        persist_query_turns=False,
    )
    documents = LiveCheckScenario(
        name="document_code_recall",
        description="Document/query-driven recall for code evidence.",
        conversation_id="live-docs",
        documents=[
            LiveCheckDocument(
                source="sample_ollama_client.py",
                title="sample_ollama_client.py",
                text=(
                    "import ollama\n\n"
                    "def call_ollama(prompt: str) -> str:\n"
                    "    response = ollama.chat(model='qwen3:8b', messages=[{'role': 'user', 'content': prompt}])\n"
                    "    return response['message']['content']\n"
                ),
                scope="project",
                metadata={"path": "sample_ollama_client.py"},
            )
        ],
        turns=[
            LiveCheckTurn(
                text="Где в коде вызывается ollama?",
                expect_any_contains=["ollama.chat", "call_ollama", "ollama"],
                expect_not_contains=["не вижу в коде", "не наш", "не могу проверить"],
            )
        ],
        fresh_brain_per_turn=True,
        persist_query_turns=False,
    )
    facts_name_exact = scenario(
        name="facts_name_exact",
        description="A1: exact self recall for name.",
        conversation_id="live-facts-name",
        setup_turns=[LiveCheckTurn(text="меня Паша зовут")],
        turns=[
            LiveCheckTurn(
                text="как меня зовут?",
                expect_any_contains=["Паша"],
                expect_not_contains=["вроде", "если не ошибаюсь", "не помню", "не вижу в памяти"],
            )
        ],
    )
    facts_age_exact = scenario(
        name="facts_age_exact",
        description="A2: exact self recall for age.",
        conversation_id="live-facts-age",
        setup_turns=[LiveCheckTurn(text="мне 21 год")],
        turns=[
            LiveCheckTurn(
                text="сколько мне лет?",
                expect_any_contains=["21"],
                expect_not_contains=["вроде", "если не ошибаюсь", "не помню", "не вижу в памяти"],
            )
        ],
    )
    facts_gpu_exact = scenario(
        name="facts_gpu_exact",
        description="A3: exact self recall for GPU.",
        conversation_id="live-facts-gpu",
        setup_turns=[LiveCheckTurn(text="у меня rtx 3050 ti")],
        turns=[
            LiveCheckTurn(
                text="какая у меня видеокарта?",
                expect_any_contains=["RTX 3050 Ti", "3050 Ti", "RTX 3050"],
                expect_not_contains=["не помню", "не вижу в памяти", "посмотри сам", "проверь сам", "wmic", "как посмотреть"],
            )
        ],
    )
    facts_hardware_colloquial = scenario(
        name="facts_hardware_colloquial",
        description="A4: colloquial hardware extraction and recall.",
        conversation_id="live-facts-hardware-colloquial",
        setup_turns=[LiveCheckTurn(text="у меня 3050ti с 4gb памяти и 32 гб озу")],
        turns=[
            LiveCheckTurn(
                text="напомни что у меня по железу",
                expect_any_contains=["3050 Ti", "RTX 3050 Ti", "RTX 3050"],
                expect_contains=["4", "32"],
                expect_not_contains=["не помню", "не вижу в памяти", "посмотри сам", "проверь сам"],
                expect_regex=[r"(?i)(rtx\s*3050|3050\s*ti)"],
            )
        ],
    )
    facts_python_exact = scenario(
        name="facts_python_exact",
        description="A5: exact self recall for Python version.",
        conversation_id="live-facts-python",
        setup_turns=[LiveCheckTurn(text="сейчас на python 3.11 сижу")],
        turns=[
            LiveCheckTurn(
                text="какой у меня питон?",
                expect_any_contains=["3.11", "Python 3.11"],
                expect_not_contains=["не помню", "не вижу в памяти", "python --version", "как посмотреть"],
            )
        ],
    )
    facts_os_exact = scenario(
        name="facts_os_exact",
        description="A6: exact self recall for OS.",
        conversation_id="live-facts-os",
        setup_turns=[LiveCheckTurn(text="щас на windows 11")],
        turns=[
            LiveCheckTurn(
                text="какая у меня система?",
                expect_any_contains=["Windows 11", "Windows"],
                expect_not_contains=["не помню", "не вижу в памяти", "winver", "как посмотреть"],
            )
        ],
    )
    facts_os_update_current_only = scenario(
        name="facts_os_update_current_only",
        description="A7: updated OS should supersede old one.",
        conversation_id="live-facts-os-update",
        setup_turns=[
            LiveCheckTurn(text="я на windows 11"),
            LiveCheckTurn(text="теперь уже linux поставил"),
        ],
        turns=[
            LiveCheckTurn(
                text="какая у меня сейчас ос?",
                expect_any_contains=["Linux", "linux", "Линукс"],
                expect_not_contains=["Windows 11", "Windows"],
            )
        ],
    )
    facts_python_update_current_only = scenario(
        name="facts_python_update_current_only",
        description="A8: updated Python should supersede old one.",
        conversation_id="live-facts-python-update",
        setup_turns=[
            LiveCheckTurn(text="был python 3.10"),
            LiveCheckTurn(text="теперь уже 3.12 поставил"),
        ],
        turns=[
            LiveCheckTurn(
                text="какой у меня сейчас python?",
                expect_any_contains=["3.12", "Python 3.12"],
                expect_not_contains=["3.10", "python --version", "не помню"],
            )
        ],
    )
    claims_like_rose_eyes = scenario(
        name="claims_like_rose_eyes",
        description="B1: open claim recall for a specific liked thing.",
        conversation_id="live-claims-rose-eyes",
        setup_turns=[LiveCheckTurn(text="я люблю смотреть в глаза Розе")],
        turns=[
            LiveCheckTurn(
                text="что я люблю в Розе?",
                expect_any_contains=["глаз", "в глаза", "глаза Розы", "смотреть в глаза"],
                expect_not_contains=["не помню"],
            )
        ],
    )
    claims_dislike_noisy_places = scenario(
        name="claims_dislike_noisy_places",
        description="B2: open claim recall for dislikes.",
        conversation_id="live-claims-dislike-noise",
        setup_turns=[LiveCheckTurn(text="я не люблю шумные места")],
        turns=[
            LiveCheckTurn(
                text="что я не люблю?",
                expect_any_contains=["шумные места", "шумные", "шум"],
                expect_not_contains=["не помню"],
            )
        ],
    )
    claims_owns_fridge_brand = scenario(
        name="claims_owns_fridge_brand",
        description="B3: claim recall for appliance brand.",
        conversation_id="live-claims-fridge",
        setup_turns=[LiveCheckTurn(text="у меня холодильник Samsung")],
        turns=[
            LiveCheckTurn(
                text="какой у меня холодильник?",
                expect_any_contains=["Samsung"],
                expect_not_contains=["не помню"],
            )
        ],
    )
    claims_preference_lotus_smell = scenario(
        name="claims_preference_lotus_smell",
        description="B4: preference recall should prefer the specific phrase.",
        conversation_id="live-claims-lotus",
        setup_turns=[LiveCheckTurn(text="мне нравится запах цветка лотоса")],
        turns=[
            LiveCheckTurn(
                text="что мне нравится?",
                expect_any_contains=["лотоса", "лотос", "запах цветка лотоса", "запах лотоса"],
                expect_not_contains=["не помню"],
            )
        ],
    )
    claims_garbage_not_promoted = scenario(
        name="claims_garbage_not_promoted",
        description="B5: garbage phrase should not become a useful claim.",
        conversation_id="live-claims-garbage",
        setup_turns=[LiveCheckTurn(text="ну это как бы такое")],
        turns=[
            LiveCheckTurn(
                text="что я люблю?",
                expect_not_contains=["ну это", "как бы", "такое"],
            )
        ],
    )
    dialog_reason_no_assistant_thoughts = scenario(
        name="dialog_reason_no_assistant_thoughts",
        description="C1: dialog recall should explain why assistant thoughts were rejected.",
        conversation_id="live-dialog-no-thoughts",
        setup_turns=[
            LiveCheckTurn(text="не думаю что надо хранить мысли ассистента"),
            LiveCheckTurn(text="да, они шумят retrieval"),
            LiveCheckTurn(text="тогда лучше хранить только решения и факты"),
            LiveCheckTurn(text="ок, так и делаем"),
        ],
        turns=[
            LiveCheckTurn(
                text="почему мы решили не хранить мысли ассистента?",
                expect_any_contains=["шум", "retrieval", "шумят", "засор", "мешают"],
                expect_contains=["мысли ассистента"],
                expect_not_contains=["не помню"],
            )
        ],
    )
    dialog_memory_design_summary = scenario(
        name="dialog_memory_design_summary",
        description="C2: contextual dialog summary for memory discussion.",
        conversation_id="live-dialog-memory-summary",
        setup_turns=[
            LiveCheckTurn(text="давай разделим память на facts и claims"),
            LiveCheckTurn(text="да"),
            LiveCheckTurn(text="и ещё отдельно эпизоды диалога"),
            LiveCheckTurn(text="согласна"),
        ],
        turns=[
            LiveCheckTurn(
                text="что мы обсуждали по памяти?",
                expect_contains=["памят"],
                expect_any_contains=["facts", "claims", "факты", "claims"],
                expect_regex=[r"(?i)(эпиз|диалог|dialog)"],
                expect_not_contains=["не помню"],
            )
        ],
    )
    dialog_plan_memory_before_web = scenario(
        name="dialog_plan_memory_before_web",
        description="C3: contextual dialog recall for agreed plan.",
        conversation_id="live-dialog-plan",
        setup_turns=[
            LiveCheckTurn(text="сначала доделываем память, потом веб"),
            LiveCheckTurn(text="ок"),
            LiveCheckTurn(text="по вебу потом отдельно вернёмся"),
            LiveCheckTurn(text="да"),
        ],
        turns=[
            LiveCheckTurn(
                text="какой у нас план?",
                expect_contains=["пам"],
                expect_any_contains=["веб", "web"],
                expect_not_contains=["не помню"],
            )
        ],
    )
    documents_character_death = scenario(
        name="documents_character_death",
        description="D1: answer should come from the loaded story text.",
        conversation_id="live-doc-character-death",
        documents=[
            LiveCheckDocument(
                source="story_excerpt.txt",
                title="story_excerpt.txt",
                text=(
                    "Глава 3.\n"
                    "Персонаж Арман умер, когда утонул в реке во время шторма.\n"
                    "После этого деревня долго вспоминала тот шторм.\n"
                ),
                scope="project",
                metadata={"path": "story_excerpt.txt"},
            )
        ],
        turns=[
            LiveCheckTurn(
                text="как умер персонаж?",
                expect_any_contains=["утонул", "в реке", "во время шторма"],
                expect_not_contains=["не знаю", "не вижу в тексте"],
            )
        ],
    )
    documents_code_ollama_call = scenario(
        name="documents_code_ollama_call",
        description="D2: answer should come from document/code retrieval.",
        conversation_id="live-doc-code-ollama",
        documents=[
            LiveCheckDocument(
                source="sample_ollama_client.py",
                title="sample_ollama_client.py",
                text=(
                    "import ollama\n\n"
                    "def call_ollama(prompt: str) -> str:\n"
                    "    response = ollama.chat(model='qwen3:8b', messages=[{'role': 'user', 'content': prompt}])\n"
                    "    return response['message']['content']\n"
                ),
                scope="project",
                metadata={"path": "sample_ollama_client.py"},
            )
        ],
        turns=[
            LiveCheckTurn(
                text="где у меня вызывается ollama?",
                expect_any_contains=["ollama.chat", "call_ollama", "sample_ollama_client.py"],
                expect_not_contains=["не вижу в коде", "не нашёл", "не могу проверить"],
            )
        ],
    )
    documents_do_not_override_self_facts = scenario(
        name="documents_do_not_override_self_facts",
        description="D3: document mention must not override user self fact.",
        conversation_id="live-doc-vs-self-fact",
        documents=[
            LiveCheckDocument(
                source="runtime_notes.txt",
                title="runtime_notes.txt",
                text=(
                    "Dev notes.\n"
                    "The sample service in this repository still runs on Python 3.9.\n"
                    "Migration to Python 3.11 is planned later.\n"
                ),
                scope="project",
                metadata={"path": "runtime_notes.txt"},
            )
        ],
        setup_turns=[LiveCheckTurn(text="я на python 3.11")],
        turns=[
            LiveCheckTurn(
                text="какой у меня python?",
                expect_any_contains=["3.11", "Python 3.11"],
                expect_not_contains=["3.9", "python --version", "не помню"],
            )
        ],
    )
    noise_assistant_echo_does_not_dominate_gpu_recall = scenario(
        name="noise_assistant_echo_does_not_dominate_gpu_recall",
        description="E1: assistant echo reply must not become the main memory source.",
        conversation_id="live-noise-assistant-echo",
        setup_turns=[LiveCheckTurn(text="у меня rtx 3050 ti")],
        turns=[
            LiveCheckTurn(
                text="какая у меня видеокарта?",
                expect_any_contains=["RTX 3050 Ti", "3050 Ti", "RTX 3050"],
                expect_not_contains=["не помню", "посмотри сам", "проверь сам"],
            )
        ],
    )
    noise_old_assistant_miss_does_not_override_python_fact = scenario(
        name="noise_old_assistant_miss_does_not_override_python_fact",
        description="E2: old assistant miss/help reply must not beat later exact fact.",
        conversation_id="live-noise-assistant-miss",
        setup_turns=[
            LiveCheckTurn(text="какой у меня python?"),
            LiveCheckTurn(text="у меня python 3.11"),
        ],
        turns=[
            LiveCheckTurn(
                text="какой у меня python?",
                expect_any_contains=["3.11", "Python 3.11"],
                expect_not_contains=["не помню", "python --version", "посмотри", "проверь"],
            )
        ],
    )
    noise_false_number_does_not_become_gpu = scenario(
        name="noise_false_number_does_not_become_gpu",
        description="E3: unrelated number must not become RAM/GPU fact.",
        conversation_id="live-noise-false-number",
        setup_turns=[LiveCheckTurn(text="у меня 3050 сообщений в логе")],
        turns=[
            LiveCheckTurn(
                text="какая у меня видеокарта?",
                expect_any_contains=["не вижу", "не помню", "нет точного", "не знаю", "не нахожу"],
                expect_not_regex=[r"(?i)\brtx\s*3050\b", r"(?i)\b3050\s*ti\b", r"(?i)\b3050\b"],
            )
        ],
    )
    return {
        "facts": [
            facts,
            facts_name_exact,
            facts_age_exact,
            facts_gpu_exact,
            facts_hardware_colloquial,
            facts_python_exact,
            facts_os_exact,
            facts_os_update_current_only,
            facts_python_update_current_only,
        ],
        "claims": [
            claim_uses,
            claim_owns,
            claims_like_rose_eyes,
            claims_dislike_noisy_places,
            claims_owns_fridge_brand,
            claims_preference_lotus_smell,
            claims_garbage_not_promoted,
        ],
        "dialog": [
            dialog,
            dialog_reason_no_assistant_thoughts,
            dialog_memory_design_summary,
            dialog_plan_memory_before_web,
        ],
        "documents": [
            documents,
            documents_character_death,
            documents_code_ollama_call,
            documents_do_not_override_self_facts,
        ],
        "noise": [
            noise_assistant_echo_does_not_dominate_gpu_recall,
            noise_old_assistant_miss_does_not_override_python_fact,
            noise_false_number_does_not_become_gpu,
        ],
        "all": [
            facts,
            facts_name_exact,
            facts_age_exact,
            facts_gpu_exact,
            facts_hardware_colloquial,
            facts_python_exact,
            facts_os_exact,
            facts_os_update_current_only,
            facts_python_update_current_only,
            claim_uses,
            claim_owns,
            claims_like_rose_eyes,
            claims_dislike_noisy_places,
            claims_owns_fridge_brand,
            claims_preference_lotus_smell,
            claims_garbage_not_promoted,
            dialog,
            dialog_reason_no_assistant_thoughts,
            dialog_memory_design_summary,
            dialog_plan_memory_before_web,
            documents,
            documents_character_death,
            documents_code_ollama_call,
            documents_do_not_override_self_facts,
            noise_assistant_echo_does_not_dominate_gpu_recall,
            noise_old_assistant_miss_does_not_override_python_fact,
            noise_false_number_does_not_become_gpu,
        ],
    }


def _build_ad_hoc_scenario(args: argparse.Namespace) -> LiveCheckScenario | None:
    turns = [str(x or "").strip() for x in list(args.turn or []) if str(x or "").strip()]
    if not turns:
        return None
    documents: list[LiveCheckDocument] = []
    for raw_path in list(args.document_file or []):
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        documents.append(
            LiveCheckDocument(
                source=str(path),
                title=path.name,
                text=path.read_text(encoding="utf-8-sig"),
                scope="project",
                metadata={"path": str(path)},
            )
        )
    query_turns = [LiveCheckTurn(text=text) for text in turns]
    if query_turns:
        last = query_turns[-1]
        query_turns[-1] = LiveCheckTurn(
            text=last.text,
            label=last.label,
            meta=dict(last.meta or {}),
            expect_contains=_as_string_list(args.expect_contains),
            expect_any_contains=_as_string_list(args.expect_any_contains),
            expect_not_contains=_as_string_list(args.expect_not_contains),
            expect_regex=_as_string_list(args.expect_regex),
            expect_not_regex=_as_string_list(args.expect_not_regex),
            expect_route=str(args.expect_route or "").strip(),
        )
    return LiveCheckScenario(
        name="ad_hoc",
        description="Ad-hoc live project check from CLI turns.",
        conversation_id=str(args.conversation_id or "live-ad-hoc"),
        documents=documents,
        turns=query_turns,
        fresh_brain_per_turn=False,
        persist_query_turns=False,
    )


def _provider_name_for_arg(value: str) -> str | None:
    token = str(value or "").strip().lower()
    if token in {"", "config", "default", "auto"}:
        return None
    return token


def _build_provider_instance(*, provider_name: str | None, model: str) -> LLMProviderBase:
    default_model = str(model or "").strip() or None
    return build_provider(provider_name, default_model=default_model)


def _cleanup_temp_dir(temp_dir: tempfile.TemporaryDirectory[str] | None) -> None:
    if temp_dir is None:
        return
    try:
        temp_dir.cleanup()
    except Exception as exc:
        print(f"Warning: could not fully remove temp artifacts: {exc}", file=sys.stderr)


def _scenario_meta(
    *,
    scenario: LiveCheckScenario,
    conversation_id: str,
    turn_meta: dict[str, Any],
    store_turn: bool,
) -> dict[str, Any]:
    return {
        **dict(scenario.meta or {}),
        **dict(turn_meta or {}),
        "source": "live_check",
        "conversation_id": conversation_id,
        "store_turn": bool(store_turn),
    }


def _build_brain(
    *,
    provider: LLMProviderBase,
    memory_root: Path,
    state_root: Path,
    conversation_id: str,
) -> Brain:
    state_root.mkdir(parents=True, exist_ok=True)
    memory_root.mkdir(parents=True, exist_ok=True)
    state_manager = CharacterRuntime(
        state_path=state_root / "brain_state.json",
        state_store_dir=state_root / "brain_state_store",
    )
    if conversation_id:
        try:
            state_manager.new_conversation(conversation_id=conversation_id)
        except Exception:
            pass
    memory_manager = MemoryManager(root_dir=memory_root)
    return Brain(provider=provider, state_manager=state_manager, memory_manager=memory_manager)


def _close_brain(brain: Brain | None) -> None:
    if brain is None:
        return
    memory_manager = getattr(brain, "memory_manager", None)
    close = getattr(memory_manager, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _ingest_documents(*, brain: Brain, scenario: LiveCheckScenario, conversation_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list(scenario.documents or []):
        request = DocumentIngestRequest(
            text=str(row.text or ""),
            source=str(row.source or ""),
            namespace=conversation_id,
            scope=_parse_scope(str(row.scope or "project")),
            title=str(row.title or ""),
            metadata={
                **dict(row.metadata or {}),
                "conversation_id": conversation_id,
                "source": "live_check_document",
            },
        )
        result = brain.memory_manager.ingest_document(request)
        out.append(
            {
                "source": str(row.source or ""),
                "title": str(row.title or ""),
                "document_id": str(result.document.id or ""),
                "chunks": len(list(result.chunks or [])),
            }
        )
    return out


def _run_turn(*, brain: Brain, text: str, meta: dict[str, Any]) -> TurnRunResult:
    result = brain.handle_message(text, meta=meta)
    return TurnRunResult(
        text=str(result.text or ""),
        route=str(result.route or ""),
        status=str(result.status or ""),
        model=str(result.stats.get("served_model") or meta.get("model") or ""),
        thinking=str(result.thinking or ""),
        logs=[str(x) for x in list(result.logs or [])],
        stats=dict(result.stats or {}),
        memory_ops=[dict(x) for x in list(result.memory_ops or [])],
        error=str(result.error or ""),
    )


def _run_scenario(
    *,
    scenario: LiveCheckScenario,
    provider: LLMProviderBase,
    base_work_dir: Path,
    show_thinking: bool,
    show_logs: bool,
    json_mode: bool,
    stop_on_fail: bool,
) -> ScenarioRunResult:
    scenario_slug = _slugify(scenario.name)
    scenario_dir = (base_work_dir / scenario_slug).resolve()
    shared_memory_root = (scenario_dir / "memory_root").resolve()
    setup_state_root = (scenario_dir / "setup_state").resolve()
    conversation_id = str(scenario.conversation_id or scenario_slug).strip().lower() or scenario_slug
    failures: list[str] = []
    transcript: list[dict[str, Any]] = []
    documents_ingested = 0
    turns_run = 0

    setup_brain: Brain | None = None
    try:
        setup_brain = _build_brain(
            provider=provider,
            memory_root=shared_memory_root,
            state_root=setup_state_root,
            conversation_id=conversation_id,
        )
        ingested = _ingest_documents(brain=setup_brain, scenario=scenario, conversation_id=conversation_id)
        documents_ingested = len(ingested)
        for item in ingested:
            transcript.append({"kind": "document", **dict(item)})

        for index, turn in enumerate(list(scenario.setup_turns or []), start=1):
            meta = _scenario_meta(
                scenario=scenario,
                conversation_id=conversation_id,
                turn_meta=turn.meta,
                store_turn=True,
            )
            result = _run_turn(brain=setup_brain, text=turn.text, meta=meta)
            turns_run += 1
            transcript.append(
                {
                    "kind": "setup_turn",
                    "index": index,
                    "label": str(turn.label or ""),
                    "user": str(turn.text or ""),
                    "meta": dict(meta or {}),
                    "result": result.to_dict(),
                }
            )
            if not json_mode:
                print(f"[setup {index}] you> {turn.text}")
                print(f"assistant> {result.text}")
                print(f"[route: {result.route or '-'} | model: {result.model or '-'}]")
                if show_thinking and result.thinking:
                    print(f"[thinking]\n{result.thinking}")
                if show_logs and result.logs:
                    print("[logs]")
                    for row in list(result.logs or []):
                        print(row)
            if result.error:
                failures.append(f"setup turn {index} failed: {result.error}")
                if stop_on_fail:
                    return ScenarioRunResult(
                        name=scenario.name,
                        conversation_id=conversation_id,
                        work_dir=str(scenario_dir),
                        ok=False,
                        turns_run=turns_run,
                        documents_ingested=documents_ingested,
                        failures=failures,
                        transcript=transcript,
                    )

        shared_query_brain: Brain | None = None
        if not scenario.fresh_brain_per_turn:
            shared_query_brain = _build_brain(
                provider=provider,
                memory_root=shared_memory_root,
                state_root=(scenario_dir / "query_state_shared").resolve(),
                conversation_id=conversation_id,
            )

        try:
            for index, turn in enumerate(list(scenario.turns or []), start=1):
                active_brain = shared_query_brain
                temp_brain: Brain | None = None
                if active_brain is None:
                    temp_brain = _build_brain(
                        provider=provider,
                        memory_root=shared_memory_root,
                        state_root=(scenario_dir / f"query_state_{index:02d}").resolve(),
                        conversation_id=conversation_id,
                    )
                    active_brain = temp_brain
                meta = _scenario_meta(
                    scenario=scenario,
                    conversation_id=conversation_id,
                    turn_meta=turn.meta,
                    store_turn=bool(scenario.persist_query_turns),
                )
                result = _run_turn(brain=active_brain, text=turn.text, meta=meta)
                turns_run += 1
                check = _evaluate_expectations(turn=turn, result=result)
                transcript.append(
                    {
                        "kind": "query_turn",
                        "index": index,
                        "label": str(turn.label or ""),
                        "user": str(turn.text or ""),
                        "meta": dict(meta or {}),
                        "result": result.to_dict(),
                        "check": {"ok": bool(check.ok), "failures": list(check.failures or [])},
                    }
                )
                if not json_mode:
                    print(f"[check {index}] you> {turn.text}")
                    print(f"assistant> {result.text}")
                    print(f"[route: {result.route or '-'} | model: {result.model or '-'} | ok: {str(check.ok).lower()}]")
                    if show_thinking and result.thinking:
                        print(f"[thinking]\n{result.thinking}")
                    if show_logs and result.logs:
                        print("[logs]")
                        for row in list(result.logs or []):
                            print(row)
                    if check.failures:
                        print("[expectation_failures]")
                        for row in list(check.failures or []):
                            print(f"- {row}")
                if not check.ok:
                    failures.extend([f"turn {index}: {row}" for row in list(check.failures or [])])
                    if stop_on_fail:
                        _close_brain(temp_brain)
                        return ScenarioRunResult(
                            name=scenario.name,
                            conversation_id=conversation_id,
                            work_dir=str(scenario_dir),
                            ok=False,
                            turns_run=turns_run,
                            documents_ingested=documents_ingested,
                            failures=failures,
                            transcript=transcript,
                        )
                _close_brain(temp_brain)
        finally:
            _close_brain(shared_query_brain)
    finally:
        _close_brain(setup_brain)

    return ScenarioRunResult(
        name=scenario.name,
        conversation_id=conversation_id,
        work_dir=str(scenario_dir),
        ok=not failures,
        turns_run=turns_run,
        documents_ingested=documents_ingested,
        failures=failures,
        transcript=transcript,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live end-to-end MMis checks through the real Brain/LLM path.",
    )
    parser.add_argument("--suite", choices=sorted(_builtin_scenarios()), default="facts")
    parser.add_argument("--scenario-file", action="append", default=[], help="JSON file with scenarios.")
    parser.add_argument("--list", action="store_true", help="List built-in suites/scenarios and exit.")
    parser.add_argument("--provider", default="config", help="Provider override: ollama, openai, or config.")
    parser.add_argument("--model", default="", help="Model override for the chosen provider.")
    parser.add_argument("--memory-dir", default="", help="Base directory for live-check artifacts.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep temporary artifacts directory.")
    parser.add_argument("--show-thinking", action="store_true", help="Print returned thinking text.")
    parser.add_argument("--show-logs", action="store_true", help="Print BrainResult logs.")
    parser.add_argument("--json", action="store_true", help="Print final run result as JSON.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failed live check.")
    parser.add_argument("--skip-healthcheck", action="store_true", help="Skip provider healthcheck.")
    parser.add_argument("--conversation-id", default="", help="Conversation id for ad-hoc --turn mode.")
    parser.add_argument("--turn", action="append", default=[], help="Ad-hoc user turn. Pass multiple times.")
    parser.add_argument("--document-file", action="append", default=[], help="Document file for ad-hoc mode.")
    parser.add_argument("--expect-contains", action="append", default=[], help="Required substring on final ad-hoc turn.")
    parser.add_argument("--expect-any-contains", action="append", default=[], help="Any-of substrings on final ad-hoc turn.")
    parser.add_argument("--expect-not-contains", action="append", default=[], help="Forbidden substring on final ad-hoc turn.")
    parser.add_argument("--expect-regex", action="append", default=[], help="Required regex on final ad-hoc turn.")
    parser.add_argument("--expect-not-regex", action="append", default=[], help="Forbidden regex on final ad-hoc turn.")
    parser.add_argument("--expect-route", default="", help="Expected route for the final ad-hoc turn.")
    return parser.parse_args()


def _print_list() -> None:
    suites = _builtin_scenarios()
    print("Built-in live suites:")
    for suite_name, rows in suites.items():
        print(f"- {suite_name}")
        for row in list(rows or []):
            print(f"  - {row.name}")


def _collect_scenarios(args: argparse.Namespace) -> list[LiveCheckScenario]:
    scenarios: list[LiveCheckScenario] = []
    suites = _builtin_scenarios()
    if not list(args.turn or []):
        scenarios.extend(list(suites.get(str(args.suite or "facts"), [])))
    for raw_path in list(args.scenario_file or []):
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        scenarios.extend(_scenario_file_scenarios(path))
    ad_hoc = _build_ad_hoc_scenario(args)
    if ad_hoc is not None:
        scenarios.append(ad_hoc)
    return scenarios


def main() -> int:
    _configure_stdout()
    args = _parse_args()
    if args.list:
        _print_list()
        return 0

    scenarios = _collect_scenarios(args)
    if not scenarios:
        print("No live scenarios selected.", file=sys.stderr)
        return 2

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if str(args.memory_dir or "").strip():
        base_work_dir = Path(str(args.memory_dir)).expanduser().resolve()
        base_work_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="mmis-live-check-")
        base_work_dir = Path(temp_dir.name).resolve()

    provider_name = _provider_name_for_arg(str(args.provider or "config"))
    provider = _build_provider_instance(provider_name=provider_name, model=str(args.model or ""))
    provider_health: dict[str, Any] = {}
    if not bool(args.skip_healthcheck):
        try:
            health = provider.healthcheck()
            provider_health = {
                "ok": bool(health.ok),
                "provider": str(health.provider or ""),
                "detail": str(health.detail or ""),
                "model": str(health.model or ""),
            }
            if not health.ok:
                print(f"Provider healthcheck failed: {health.provider} | {health.detail}", file=sys.stderr)
                _cleanup_temp_dir(temp_dir)
                return 2
        except Exception as exc:
            print(f"Provider healthcheck failed: {exc}", file=sys.stderr)
            _cleanup_temp_dir(temp_dir)
            return 2

    cfg = load_config()
    resolved_provider = str(provider_name or getattr(cfg, "llm_default_provider", "") or "").strip() or "config"
    resolved_model = str(args.model or provider_health.get("model") or getattr(cfg, "model_name", "") or "").strip()

    if not args.json:
        print(f"Live checks: {len(scenarios)} scenario(s)")
        print(f"Provider: {resolved_provider}")
        print(f"Model: {resolved_model or '-'}")
        print(f"Artifacts: {base_work_dir}")
        print("")

    failed = False
    scenario_results: list[ScenarioRunResult] = []
    try:
        for index, scenario in enumerate(list(scenarios or []), start=1):
            if not args.json:
                print("=" * 100)
                print(f"Scenario {index}/{len(scenarios)}: {scenario.name}")
                if scenario.description:
                    print(scenario.description)
                print(f"Conversation: {scenario.conversation_id or _slugify(scenario.name)}")
                print(f"Work dir: {(base_work_dir / _slugify(scenario.name)).resolve()}")
                print("-" * 100)
            result = _run_scenario(
                scenario=scenario,
                provider=provider,
                base_work_dir=base_work_dir,
                show_thinking=bool(args.show_thinking),
                show_logs=bool(args.show_logs),
                json_mode=bool(args.json),
                stop_on_fail=bool(args.stop_on_fail),
            )
            scenario_results.append(result)
            failed = failed or (not result.ok)
            if not args.json:
                print("-" * 100)
                print(f"Scenario result: {'PASS' if result.ok else 'FAIL'} | turns={result.turns_run} | documents={result.documents_ingested}")
                for row in list(result.failures or []):
                    print(f"- {row}")
                print("")
                if failed and args.stop_on_fail:
                    break
    finally:
        if temp_dir is not None and not args.keep_artifacts:
            _cleanup_temp_dir(temp_dir)

    summary = {
        "ok": not failed,
        "provider": resolved_provider,
        "model": resolved_model,
        "provider_health": dict(provider_health or {}),
        "artifacts_dir": str(base_work_dir),
        "scenarios": [row.to_dict() for row in list(scenario_results or [])],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("=" * 100)
        print(f"Final result: {'PASS' if summary['ok'] else 'FAIL'} | scenarios={len(scenario_results)}")
        print(f"Artifacts dir: {base_work_dir}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
