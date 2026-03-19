from __future__ import annotations

from modules.character.persona_snapshot_builder import PersonaSnapshotBuilder


def test_persona_snapshot_builder_prefers_flat_profile_hints_and_active_facts() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "preferred_editor": "vscode",
            "preferred_language": "python",
            "assistant_warmth": 0.7,
            "active_facts": {
                "user.identity_name": {
                    "predicate": "identity_name",
                    "value": "Паша",
                },
                "user.environment_runtime_python": {
                    "predicate": "environment_runtime_python",
                    "value": "python 3.11",
                },
                "user.project_name#1": {
                    "predicate": "project_name",
                    "value": "MMis",
                },
                "user.project_name#2": {
                    "predicate": "project_name",
                    "value": "AnotherProject",
                },
            },
        },
        memory_context={},
        state={"active_mode": "chatting"},
        meta={},
    )

    assert snapshot.character_id == "asya"
    assert snapshot.user_profile_hints["user_name"] == "Паша"
    assert snapshot.user_profile_hints["preferred_editor"] == "vscode"
    assert snapshot.user_profile_hints["preferred_language"] == "python"
    assert snapshot.user_profile_hints["python"] == "python 3.11"
    assert snapshot.user_profile_hints["project_name"] == ["MMis", "AnotherProject"]
    assert snapshot.user_addressing["canonical_name"] == "Паша"
    assert snapshot.stable_traits["warmth"] == 0.7


def test_persona_snapshot_builder_does_not_use_raw_memory_selection_as_truth() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot=None,
        memory_context={
            "selected": [
                {"text": "old noisy memory"},
                {"text": "assistant said something irrelevant"},
            ],
            "retrieved_memories": [
                {"text": "raw snippet should not become profile hint"},
            ],
        },
        state={},
        meta={},
    )

    assert snapshot.user_profile_hints == {}
    assert snapshot.debug["memory_blocks"] == []


def test_persona_snapshot_builder_splits_persistent_and_turn_local_signals() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "preferred_editor": "vscode",
            "relation_trust": 0.91,
            "assistant_directness": 0.88,
            "name_allowed_forms": ["Паша", "Паша"],
            "use_name_by_default": True,
        },
        memory_context={"blocks": {"self_facts": "...", "relevant_claims": "..."}} ,
        state={
            "active_mode": "engineer",
            "character_trait_defaults": {
                "warmth": 0.68,
                "verbosity": 0.55,
                "teasing": 0.46,
            },
            "user_addressing": {"canonical_name": "Паша", "allow_diminutives": True},
        },
        meta={
            "emotion": "sad",
            "is_technical": True,
            "metadata_tags": ["needs_short_answer"],
            "active_mode": "debugger",
        },
    )

    assert snapshot.user_profile_hints["preferred_editor"] == "vscode"
    assert snapshot.relation_state["trust"] == 0.91
    assert snapshot.user_addressing["canonical_name"] == "Паша"
    assert snapshot.user_addressing["allow_diminutives"] is True
    assert snapshot.user_addressing["use_name_by_default"] is True
    assert snapshot.user_addressing["allowed_forms"] == ["Паша", "Паша"]
    assert round(float(snapshot.stable_traits["warmth"]), 4) == 0.78
    assert round(float(snapshot.stable_traits["directness"]), 4) == 1.0
    assert snapshot.response_bias["technical_mode"] == 1.0
    assert snapshot.response_bias["needs_short_answer"] == 1.0
    assert snapshot.response_bias["frustration_softening"] == 1.0
    assert snapshot.active_mode == "debugger"
    assert snapshot.mood == "sad"
    assert snapshot.debug["profile_keys"]
    assert snapshot.debug["memory_blocks"] == ["relevant_claims", "self_facts"]
    sources = dict(snapshot.debug.get("sources") or {})
    assert dict(sources.get("baseline_traits") or {}).get("warmth") == 0.68
    assert dict(sources.get("dynamic_trait_modifiers") or {}).get("warmth_delta") == 0.1
    assert dict(sources.get("dynamic_trait_modifiers") or {}).get("directness_delta") == 0.12
    assert "warmth" in list(sources.get("stable_traits_from_character_defaults") or [])
    assert list(sources.get("stable_traits_ignored_runtime") or []) == []


def test_persona_snapshot_builder_reports_sources_in_debug() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "identity_name": "Паша",
            "assistant_warmth": 0.8,
            "relation_trust": 0.9,
        },
        memory_context={"blocks": {"self_facts": "- identity_name: Паша"}},
        state={
            "mood": "neutral",
            "active_mode": "chatting",
            "character_trait_defaults": {"empathy": 0.77},
            "user_addressing": {"allowed_forms": ["Паша"]},
        },
        meta={
            "emotion": "frustrated",
            "is_technical": True,
            "metadata_tags": ["needs_short_answer"],
        },
    )

    sources = dict(snapshot.debug.get("sources") or {})
    assert sources.get("mood_source") == "meta.emotion"
    assert sources.get("active_mode_source") == "state.active_mode"
    assert "identity_name" in list(sources.get("persistent_profile_keys") or [])
    assert "relation_trust" in list(sources.get("relation_fields_from_persistent") or [])
    assert "empathy" in list(sources.get("stable_traits_from_character_defaults") or [])
    assert "meta.is_technical" in list(sources.get("response_bias_sources") or [])
    assert dict(sources.get("addressing_source") or {}).get("allowed_forms") == "runtime_override"
    assert dict(sources.get("trait_overrides") or {}).get("empathy", {}).get("source") == "character_specs_default"


def test_persona_snapshot_builder_applies_trait_priority_order() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "assistant_warmth": 0.52,
            "assistant_directness": 0.44,
            "assistant_sarcasm": 0.34,
        },
        memory_context={},
        state={
            "character_trait_defaults": {
                "warmth": 0.31,
                "directness": 0.36,
                "sarcasm": 0.28,
            },
            "identity_core": {
                "assistant_trait_baseline": {
                    "warmth_baseline": 0.66,
                    "directness_baseline": 0.82,
                    "sarcasm_ceiling": 0.12,
                }
            },
        },
        meta={
            "emotion": "frustrated",
            "is_technical": True,
        },
    )

    assert round(float(snapshot.stable_traits["warmth"]), 4) == 0.76
    assert round(float(snapshot.stable_traits["directness"]), 4) == 0.94
    assert round(float(snapshot.stable_traits["sarcasm"]), 4) == 0.0
    sources = dict(snapshot.debug.get("sources") or {})
    assert list(sources.get("trait_layer_priority") or []) == [
        "character_specs_defaults",
        "active_profile_snapshot",
        "identity_core",
        "turn_local_modifiers",
    ]
    assert dict(sources.get("trait_overrides") or {}).get("warmth", {}).get("source") == "identity_core_baseline"
    assert dict(sources.get("trait_overrides") or {}).get("directness", {}).get("source") == "identity_core_baseline"


def test_persona_snapshot_builder_uses_identity_core_above_active_profile_for_addressing() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "identity_name": "Паша",
            "name_allowed_forms": ["Паша"],
        },
        identity_core_snapshot={
            "addressing": {
                "canonical_name": "Паш",
                "allowed_forms": ["Паш"],
                "forbidden_forms": ["Пашка"],
                "use_name_by_default": False,
            }
        },
        memory_context={},
        state={},
        meta={},
    )

    assert snapshot.user_addressing["canonical_name"] == "Паш"
    assert snapshot.user_addressing["allowed_forms"] == ["Паш"]
    assert snapshot.user_profile_hints["user_name"] == "Паш"
    sources = dict(snapshot.debug.get("sources") or {})
    assert "canonical_name" in list(sources.get("user_addressing_from_identity_core") or [])
    assert dict(sources.get("addressing_source") or {}).get("canonical_name") == "identity_core"


def test_persona_snapshot_builder_uses_identity_core_interaction_style_for_persistent_persona_bias() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "assistant_directness": 0.35,
            "assistant_sarcasm": 0.05,
            "relation_technical_collaboration": 0.40,
        },
        identity_core_snapshot={
            "interaction_style": {
                "prefers_directness": 0.82,
                "prefers_short_answers": 0.55,
                "allows_light_teasing": True,
                "technical_collaboration_style": "high",
            }
        },
        memory_context={},
        state={},
        meta={},
    )

    assert snapshot.stable_traits["directness"] == 0.82
    assert float(snapshot.stable_traits["verbosity"]) < 0.6
    assert float(snapshot.stable_traits["teasing"]) >= 0.38
    assert snapshot.relation_state["technical_collaboration"] == 0.88
    assert snapshot.user_profile_hints["technical_collaboration_style"] == "high"
    sources = dict(snapshot.debug.get("sources") or {})
    assert "directness" in list(sources.get("stable_traits_from_identity_core") or [])
    assert "technical_collaboration" in list(sources.get("relation_fields_from_identity_core") or [])
    assert dict(sources.get("interaction_style_source") or {}).get("prefers_short_answers") == "identity_core"


def test_persona_snapshot_builder_uses_identity_core_trait_baseline_before_turn_local_modifiers() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "assistant_warmth": 0.30,
            "assistant_directness": 0.25,
            "assistant_sarcasm": 0.42,
            "assistant_empathy": 0.40,
            "assistant_professionalism": 0.44,
        },
        identity_core_snapshot={
            "assistant_trait_baseline": {
                "warmth_baseline": 0.68,
                "directness_baseline": 0.82,
                "empathy_floor": 0.72,
                "professionalism_floor": 0.80,
                "sarcasm_ceiling": 0.18,
            }
        },
        memory_context={},
        state={},
        meta={
            "emotion": "frustrated",
            "is_technical": True,
        },
    )

    assert round(float(snapshot.stable_traits["warmth"]), 4) == 0.78
    assert round(float(snapshot.stable_traits["directness"]), 4) == 0.94
    assert round(float(snapshot.stable_traits["empathy"]), 4) == 0.8
    assert round(float(snapshot.stable_traits["professionalism"]), 4) == 0.86
    assert round(float(snapshot.stable_traits["sarcasm"]), 4) == 0.0
    sources = dict(snapshot.debug.get("sources") or {})
    assert dict(sources.get("assistant_trait_baseline_source") or {}).get("warmth_baseline") == "identity_core"
    assert dict(sources.get("baseline_traits") or {}).get("warmth") == 0.68
    assert dict(sources.get("dynamic_trait_modifiers") or {}).get("warmth_delta") == 0.1


def test_persona_snapshot_builder_keeps_identity_core_boundaries() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={},
        identity_core_snapshot={
            "boundaries": {
                "avoid_overloaded_intros": True,
                "avoid_baby_talk": True,
                "do_not_invent_user_facts": True,
            }
        },
        memory_context={},
        state={},
        meta={},
    )

    assert snapshot.boundaries["avoid_baby_talk"] is True
    assert snapshot.boundaries["do_not_invent_user_facts"] is True
    sources = dict(snapshot.debug.get("sources") or {})
    assert "avoid_baby_talk" in list(sources.get("boundary_fields_from_identity_core") or [])
    assert dict(sources.get("boundary_source") or {}).get("avoid_baby_talk") == "identity_core"


def test_persona_snapshot_builder_keeps_identity_core_emotional_handling() -> None:
    builder = PersonaSnapshotBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={},
        identity_core_snapshot={
            "emotional_handling": {
                "deescalate_on_irritation": True,
                "treat_short_replies_as_low_bandwidth": True,
                "warmth_upshift_on_user_distress": 0.24,
                "playfulness_downshift_on_user_distress": 0.33,
            }
        },
        memory_context={},
        state={},
        meta={"emotion": "frustrated"},
    )

    assert snapshot.emotional_handling["deescalate_on_irritation"] is True
    assert snapshot.emotional_handling["treat_short_replies_as_low_bandwidth"] is True
    assert snapshot.response_bias["frustration_softening"] == 1.0
    assert snapshot.response_bias["warmth_upshift"] == 0.24
    assert snapshot.response_bias["playfulness_downshift"] == 0.33
    sources = dict(snapshot.debug.get("sources") or {})
    assert "deescalate_on_irritation" in list(sources.get("emotional_handling_fields_from_identity_core") or [])
    assert dict(sources.get("emotional_handling_source") or {}).get("deescalate_on_irritation") == "identity_core"
