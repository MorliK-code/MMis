from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.widgets import MemoryInspectorPanel


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _find_top_level(panel_tree, label: str):
    for index in range(panel_tree.topLevelItemCount()):
        item = panel_tree.topLevelItem(index)
        if item is not None and item.text(0) == label:
            return item
    return None


def test_memory_inspector_panel_builds_tree_and_raw_json_views() -> None:
    _app()
    panel = MemoryInspectorPanel()
    panel.set_snapshot(
        {
            "request_id": "req-7",
            "user_text": "какой у меня python?",
            "memory_retrieval": {
                "query": "какой у меня python?",
                "selected_facts": [
                    {
                        "type": "fact",
                        "predicate": "environment_runtime_python",
                        "value": "3.11",
                        "score": 0.92,
                        "reason": "exact_self_fact",
                        "source": "profile/global",
                    }
                ],
                "selected_episodes": [
                    {
                        "episode_id": "ep_00124",
                        "topic": "memory_governor",
                        "score": 0.81,
                        "summary_short": "Обсуждали конфликтную логику фактов",
                        "open_questions": ["как группировать soft_singleton?"],
                    }
                ],
                "filtered_out": [{"reason": "assistant_reply_not_allowed"}],
                "confidence": {"top_selected_score": 0.92},
            },
            "memory_governor": {
                "decisions": [
                    {
                        "group": "environment.python",
                        "action": "supersede_old",
                        "winner": "Python 3.11",
                        "loser": "Python 3.10",
                        "reason": "singleton_latest_wins",
                    }
                ]
            },
            "identity_core": {
                "snapshot": {
                    "character_id": "asya",
                    "addressing": {"canonical_name": "Паша"},
                    "assistant_trait_baseline": {"warmth_baseline": 0.62},
                    "debug": {
                        "sources": {
                            "addressing": {"canonical_name": "identity_core"},
                            "assistant_trait_baseline": {"warmth_baseline": "identity_core"},
                        }
                    },
                },
                "protected_keys": ["addressing.canonical_name"],
                "pending_overrides": [
                    {
                        "key": "addressing.canonical_name",
                        "value": "Павел",
                        "reason": "override_requires_confirmation",
                    }
                ],
                "trait_baselines": {"warmth_baseline": 0.62},
                "active_identity_keys": {
                    "addressing": ["canonical_name"],
                    "assistant_trait_baseline": ["warmth_baseline"],
                },
            },
            "persona_snapshot": {"mood": "neutral"},
            "active_task": {"topic": "memory design", "status": "active"},
            "prompt_pack": {"memory_block": "[SELF_FACTS]\n- Python: 3.11", "token_estimate": 123},
        }
    )

    assert panel.tabs.count() == 6
    assert "request_id: req-7" in panel.meta_label.text()

    retrieval_tree = panel.retrieval_page.tree
    assert retrieval_tree.topLevelItemCount() >= 3

    selected_facts = _find_top_level(retrieval_tree, "Selected Facts")
    assert selected_facts is not None
    assert selected_facts.childCount() == 1
    assert "environment_runtime_python" in selected_facts.child(0).text(0)
    assert "score=0.92" == selected_facts.child(0).text(1)

    retrieval_tree.setCurrentItem(selected_facts.child(0))
    assert "predicate: environment_runtime_python" in panel.retrieval_page.details_view.toPlainText()
    assert "\"exact_self_fact\"" in panel.retrieval_page.raw_view.toPlainText()

    governor_tree = panel.governor_page.tree
    decisions = _find_top_level(governor_tree, "Decisions")
    assert decisions is not None
    assert decisions.childCount() == 1
    assert "environment.python" in decisions.child(0).text(0)

    identity_tree = panel.identity_core_page.tree
    protected_keys = _find_top_level(identity_tree, "Protected Keys")
    assert protected_keys is not None
    assert protected_keys.childCount() == 1
    pending = _find_top_level(identity_tree, "Pending Overrides")
    assert pending is not None
    assert pending.childCount() == 1
    identity_tree.setCurrentItem(pending.child(0))
    assert "override_requires_confirmation" in panel.identity_core_page.details_view.toPlainText()
    assert "\"addressing.canonical_name\"" in panel.identity_core_page.raw_view.toPlainText()


def test_memory_inspector_panel_respects_display_modes() -> None:
    _app()
    panel = MemoryInspectorPanel()
    panel.configure_display(
        enabled=True,
        show_raw_scores=False,
        show_filtered_items=False,
        show_prompt_blocks=False,
    )
    panel.set_snapshot(
        {
            "request_id": "req-8",
            "user_text": "debug modes",
            "memory_retrieval": {
                "selected_facts": [
                    {
                        "predicate": "environment_runtime_python",
                        "value": "3.11",
                        "score": 0.92,
                    }
                ],
                "filtered_out": [{"reason": "assistant_reply_not_allowed"}],
                "confidence": {"top_selected_score": 0.92},
            },
            "prompt_pack": {
                "token_estimate": 111,
                "memory_block": "[SELF_FACTS]",
                "persona_block": "persona",
                "active_task_block": "task",
            },
        }
    )

    selected_facts = _find_top_level(panel.retrieval_page.tree, "Selected Facts")
    assert selected_facts is not None
    assert selected_facts.childCount() == 1
    assert selected_facts.child(0).text(1) != "score=0.92"
    assert _find_top_level(panel.retrieval_page.tree, "Confidence") is None
    assert _find_top_level(panel.retrieval_page.tree, "Filtered Out") is None
    assert _find_top_level(panel.prompt_page.tree, "Memory Block") is None
