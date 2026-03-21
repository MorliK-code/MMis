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
    panel.configure_display(
        enabled=True,
        show_raw_scores=True,
        show_filtered_items=True,
        show_prompt_blocks=True,
    )
    panel.set_snapshot(
        {
            "request_id": "req-7",
            "user_text": "РєР°РєРѕР№ Сѓ РјРµРЅСЏ python?",
            "memory_retrieval": {
                "query": "РєР°РєРѕР№ Сѓ РјРµРЅСЏ python?",
                "recall_mode": "exact_fact_recall",
                "request_plan": {
                    "mode": "profile",
                    "topic_hints": ["python", "runtime"],
                },
                "fanout_queries": [
                    {"label": "raw_user_query", "query_text": "какой у меня python?"},
                    {"label": "topic_focused", "query_text": "python runtime"},
                ],
                "selected_facts": [
                    {
                        "type": "fact",
                        "predicate": "environment_runtime_python",
                        "value": "3.11",
                        "score": 0.92,
                        "reason": "exact_self_fact",
                        "source": "profile/global",
                        "why_selected": ["exact_match_boost", "semantic_similarity"],
                        "score_breakdown": {"exact_match_boost": 1.0, "semantic_similarity": 0.89},
                    }
                ],
                "selected_episodes": [
                    {
                        "episode_id": "ep_00124",
                        "topic": "memory_governor",
                        "score": 0.81,
                        "summary_short": "РћР±СЃСѓР¶РґР°Р»Рё РєРѕРЅС„Р»РёРєС‚РЅСѓСЋ Р»РѕРіРёРєСѓ С„Р°РєС‚РѕРІ",
                        "open_questions": ["РєР°Рє РіСЂСѓРїРїРёСЂРѕРІР°С‚СЊ soft_singleton?"],
                    }
                ],
                "filtered_out": [{"reason": "assistant_reply_not_allowed"}],
                "score_breakdowns": [
                    {
                        "record_id": "fact:python",
                        "final_score": 0.92,
                        "semantic_similarity": 0.89,
                        "exact_match_boost": 1.0,
                    }
                ],
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
                    "addressing": {"canonical_name": "РџР°С€Р°"},
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
                        "value": "РџР°РІРµР»",
                        "reason": "override_requires_confirmation",
                    }
                ],
                "trait_baselines": {"warmth_baseline": 0.62},
                "active_identity_keys": {
                    "addressing": ["canonical_name"],
                    "assistant_trait_baseline": ["warmth_baseline"],
                },
            },
            "persona_snapshot": {
                "mood": "neutral",
                "relation_continuity": {"is_followup": True},
                "recent_user_state": {"low_bandwidth": False},
            },
            "active_task": {"topic": "memory design", "status": "active", "source": "continuation"},
            "prompt_pack": {
                "memory_block": "[SELF_FACTS]\n- Python: 3.11",
                "token_estimate": 123,
                "included_memory_blocks": ["SELF_FACTS", "ACTIVE_TASK"],
            },
            "tool_loop": {
                "agent_loop": True,
                "agent_tool_calls": 1,
                "agent_passes": 2,
                "executed_tools": [
                    {"iteration": 1, "tool": "memory_retrieve", "call_id": "call_memory_1", "ok": True}
                ],
                "memory_reasoning_snapshot": {
                    "profile_facts": {"prefers_short_answers": True},
                    "active_task": {"status": "active"},
                },
            },
            "final_answer_meta": {"route": "chat", "memory_reasoning_used": True},
        }
    )

    assert panel.tabs.count() == 7
    assert "request_id: req-7" in panel.meta_label.text()

    retrieval_tree = panel.retrieval_page.tree
    assert retrieval_tree.topLevelItemCount() >= 5

    query_plan = _find_top_level(retrieval_tree, "Query Plan")
    assert query_plan is not None
    retrieval_tree.setCurrentItem(query_plan)
    assert "recall_mode: exact_fact_recall" in panel.retrieval_page.details_view.toPlainText()

    fanout = _find_top_level(retrieval_tree, "Fan-out Queries")
    assert fanout is not None
    assert fanout.childCount() == 2

    selected_facts = _find_top_level(retrieval_tree, "Selected Facts")
    assert selected_facts is not None
    assert selected_facts.childCount() == 1
    assert "environment_runtime_python" in selected_facts.child(0).text(0)
    assert "score=0.92" == selected_facts.child(0).text(1)

    retrieval_tree.setCurrentItem(selected_facts.child(0))
    assert "predicate: environment_runtime_python" in panel.retrieval_page.details_view.toPlainText()
    assert "\"exact_self_fact\"" in panel.retrieval_page.raw_view.toPlainText()

    rerank = _find_top_level(retrieval_tree, "Rerank")
    assert rerank is not None
    assert rerank.childCount() == 1

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

    active_task_tree = panel.active_task_page.tree
    task = _find_top_level(active_task_tree, "Task")
    assert task is not None
    active_task_tree.setCurrentItem(task)
    assert "source: continuation" in panel.active_task_page.details_view.toPlainText()

    loop_tree = panel.loop_page.tree
    overview = _find_top_level(loop_tree, "Overview")
    assert overview is not None
    loop_tree.setCurrentItem(overview)
    assert "agent_tool_calls: 1" in panel.loop_page.details_view.toPlainText()
    memory_reasoning = _find_top_level(loop_tree, "Memory Reasoning")
    assert memory_reasoning is not None


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
    hit_summary = _find_top_level(panel.retrieval_page.tree, "Hit Summary")
    assert hit_summary is not None
    assert hit_summary.text(1) == "dict[1]"
    assert _find_top_level(panel.retrieval_page.tree, "Filtered Out") is None
    assert _find_top_level(panel.retrieval_page.tree, "Rerank") is None
    assert _find_top_level(panel.prompt_page.tree, "Memory Block") is None
