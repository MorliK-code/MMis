from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_live_checks import (
    LiveCheckTurn,
    TurnRunResult,
    _build_ad_hoc_scenario,
    _builtin_scenarios,
    _evaluate_expectations,
    _scenario_file_scenarios,
)


def test_evaluate_expectations_passes_on_matching_response() -> None:
    turn = LiveCheckTurn(
        text="Как меня зовут?",
        expect_contains=["Паша"],
        expect_not_contains=["не помню"],
        expect_regex=[r"Паш"],
        expect_route="chat",
    )
    result = TurnRunResult(
        text="Тебя зовут Паша.",
        route="chat",
        status="ok",
        model="test-model",
    )

    check = _evaluate_expectations(turn=turn, result=result)

    assert check.ok is True
    assert check.failures == []


def test_evaluate_expectations_reports_multiple_failures() -> None:
    turn = LiveCheckTurn(
        text="Какая у меня видеокарта?",
        expect_contains=["RTX"],
        expect_any_contains=["3050 Ti", "4060"],
        expect_not_contains=["не помню"],
        expect_route="chat",
    )
    result = TurnRunResult(
        text="Я не помню точную модель.",
        route="command",
        status="ok",
        model="test-model",
    )

    check = _evaluate_expectations(turn=turn, result=result)

    assert check.ok is False
    assert any("missing substring: RTX" == row for row in check.failures)
    assert any("missing any-of substrings: 3050 Ti, 4060" == row for row in check.failures)
    assert any("forbidden substring present: не помню" == row for row in check.failures)
    assert any("route mismatch: expected chat, got command" == row for row in check.failures)


def test_build_ad_hoc_scenario_attaches_expectations_to_last_turn(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.py"
    doc_path.write_text("print('hello')\n", encoding="utf-8")
    args = SimpleNamespace(
        turn=["Меня зовут Паша.", "Как меня зовут?"],
        conversation_id="live-ad-hoc",
        document_file=[str(doc_path)],
        expect_contains=["Паша"],
        expect_any_contains=[],
        expect_not_contains=["не помню"],
        expect_regex=[],
        expect_not_regex=[],
        expect_route="chat",
    )

    scenario = _build_ad_hoc_scenario(args)

    assert scenario is not None
    assert len(scenario.documents) == 1
    assert len(scenario.turns) == 2
    assert scenario.turns[0].expect_contains == []
    assert scenario.turns[1].expect_contains == ["Паша"]
    assert scenario.turns[1].expect_not_contains == ["не помню"]
    assert scenario.turns[1].expect_route == "chat"


def test_builtin_suites_are_present() -> None:
    suites = _builtin_scenarios()

    assert set(suites) == {"facts", "claims", "dialog", "documents", "noise", "all"}
    assert any(row.name == "facts_exact_recall" for row in suites["facts"])
    assert any(row.name == "facts_name_exact" for row in suites["facts"])
    assert any(row.name == "facts_os_update_current_only" for row in suites["facts"])
    assert any(row.name == "claims_uses_recall" for row in suites["claims"])
    assert any(row.name == "claims_like_rose_eyes" for row in suites["claims"])
    assert any(row.name == "claims_garbage_not_promoted" for row in suites["claims"])
    assert any(row.name == "dialog_contextual_recall" for row in suites["dialog"])
    assert any(row.name == "dialog_reason_no_assistant_thoughts" for row in suites["dialog"])
    assert any(row.name == "document_code_recall" for row in suites["documents"])
    assert any(row.name == "documents_do_not_override_self_facts" for row in suites["documents"])
    assert any(row.name == "noise_old_assistant_miss_does_not_override_python_fact" for row in suites["noise"])


def test_scenario_file_loader_supports_list_payload(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            [
                {
                    "name": "sample",
                    "conversation_id": "conv-1",
                    "turns": [{"text": "Как меня зовут?", "expect_contains": ["Паша"]}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = _scenario_file_scenarios(scenario_path)

    assert len(rows) == 1
    assert rows[0].name == "sample"
    assert rows[0].conversation_id == "conv-1"
    assert rows[0].turns[0].expect_contains == ["Паша"]
