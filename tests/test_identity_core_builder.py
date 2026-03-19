from __future__ import annotations

import tempfile
from pathlib import Path

from modules.character.identity_core import IdentityCoreBuilder
from modules.character.storage import CharacterStorage


def test_identity_core_builder_prefers_stored_addressing_over_active_profile() -> None:
    builder = IdentityCoreBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "identity_name": "Паша",
            "name_allowed_forms": ["Паша"],
            "use_name_by_default": True,
        },
        stored_identity_core={
            "addressing": {
                "canonical_name": "Паш",
                "allowed_forms": ["Паш"],
                "forbidden_forms": ["Пашка"],
                "use_name_by_default": False,
                "allow_diminutives": False,
                "updated_at": "2026-03-19T00:00:00",
            }
        },
    )

    row = snapshot.to_dict()
    assert row["addressing"]["canonical_name"] == "Паш"
    assert row["addressing"]["allowed_forms"] == ["Паш"]
    assert row["addressing"]["forbidden_forms"] == ["Пашка"]
    assert row["addressing"]["use_name_by_default"] is False
    assert dict(dict(row["debug"].get("sources") or {}).get("addressing") or {}).get("canonical_name") == "identity_core"


def test_identity_core_builder_keeps_interaction_style_and_profile_fallbacks() -> None:
    builder = IdentityCoreBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "assistant_directness": 0.41,
            "assistant_sarcasm": 0.30,
            "relation_technical_collaboration": 0.81,
        },
        stored_identity_core={
            "interaction_style": {
                "prefers_directness": 0.82,
                "prefers_short_answers": 0.55,
                "allows_light_teasing": True,
                "technical_collaboration_style": "high",
            }
        },
    )

    row = snapshot.to_dict()
    style = dict(row.get("interaction_style") or {})
    assert style.get("prefers_directness") == 0.82
    assert style.get("prefers_short_answers") == 0.55
    assert style.get("allows_light_teasing") is True
    assert style.get("technical_collaboration_style") == "high"
    assert dict(dict(row.get("debug") or {}).get("sources") or {}).get("interaction_style", {}).get("prefers_directness") == "identity_core"


def test_identity_core_builder_uses_memory_identity_core_snapshot_above_runtime_storage() -> None:
    builder = IdentityCoreBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={},
        stored_identity_core={
            "addressing": {
                "canonical_name": "Павел",
                "allowed_forms": ["Павел"],
            },
            "boundaries": {
                "avoid_baby_talk": False,
            },
        },
        memory_identity_core_snapshot={
            "addressing": {
                "canonical_name": "Паша",
                "allowed_forms": ["Паша", "Паш"],
            },
            "boundaries": {
                "avoid_baby_tone": True,
                "avoid_inventing_user_facts": True,
            },
            "emotional_rules": {
                "frustration_softening": 0.24,
                "reduce_teasing_when_user_irritated": 0.33,
            },
            "assistant_trait_baseline": {
                "warmth_baseline": 0.61,
            },
        },
    )

    row = snapshot.to_dict()
    assert dict(row.get("addressing") or {}).get("canonical_name") == "Паша"
    assert dict(row.get("boundaries") or {}).get("avoid_baby_talk") is True
    assert dict(row.get("boundaries") or {}).get("do_not_invent_user_facts") is True
    assert dict(row.get("emotional_handling") or {}).get("deescalate_on_irritation") is True
    assert dict(row.get("emotional_handling") or {}).get("warmth_upshift_on_user_distress") == 0.24
    assert dict(row.get("emotional_handling") or {}).get("playfulness_downshift_on_user_distress") == 0.33
    assert dict(row.get("assistant_trait_baseline") or {}).get("warmth_baseline") == 0.61
    assert "memory_identity_core" in list(dict(dict(row.get("debug") or {}).get("input_layers") or {}).get("priority") or [])


def test_character_storage_syncs_user_addressing_into_identity_core() -> None:
    root = Path(tempfile.mkdtemp(prefix="mmis-identity-core-"))
    storage = CharacterStorage(root=root)

    storage.save_user_addressing(
        "asya",
        {
            "canonical_name": "Паша",
            "allowed_forms": ["Паша", "Паш"],
            "forbidden_forms": ["Пашка"],
            "use_name_by_default": False,
            "allow_diminutives": False,
            "updated_at": "2026-03-19T12:00:00",
        },
    )

    identity_core = dict(storage.load_identity_core("asya") or {})
    addressing = dict(identity_core.get("addressing") or {})
    assert addressing.get("canonical_name") == "Паша"
    assert addressing.get("allowed_forms") == ["Паша", "Паш"]
    assert addressing.get("forbidden_forms") == ["Пашка"]
    assert identity_core.get("updated_at") == "2026-03-19T12:00:00"


def test_character_storage_persists_identity_core_interaction_style() -> None:
    root = Path(tempfile.mkdtemp(prefix="mmis-identity-core-style-"))
    storage = CharacterStorage(root=root)

    storage.save_identity_core(
        "asya",
        {
            "interaction_style": {
                "prefers_directness": 0.82,
                "prefers_short_answers": 0.55,
                "allows_light_teasing": True,
                "technical_collaboration_style": "high",
            }
        },
    )

    identity_core = dict(storage.load_identity_core("asya") or {})
    style = dict(identity_core.get("interaction_style") or {})
    assert style == {
        "prefers_directness": 0.82,
        "prefers_short_answers": 0.55,
        "allows_light_teasing": True,
        "technical_collaboration_style": "high",
    }


def test_identity_core_builder_keeps_boundaries() -> None:
    builder = IdentityCoreBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={},
        stored_identity_core={
            "boundaries": {
                "avoid_overloaded_intros": True,
                "avoid_baby_talk": True,
                "avoid_overformal_tone": True,
                "do_not_invent_user_facts": True,
            }
        },
    )

    row = snapshot.to_dict()
    boundaries = dict(row.get("boundaries") or {})
    assert boundaries.get("avoid_baby_talk") is True
    assert boundaries.get("do_not_invent_user_facts") is True
    assert (
        dict(dict(row.get("debug") or {}).get("sources") or {})
        .get("boundaries", {})
        .get("avoid_baby_talk")
        == "identity_core"
    )


def test_character_storage_persists_identity_core_boundaries() -> None:
    root = Path(tempfile.mkdtemp(prefix="mmis-identity-core-boundaries-"))
    storage = CharacterStorage(root=root)

    storage.save_identity_core(
        "asya",
        {
            "boundaries": {
                "avoid_overloaded_intros": True,
                "avoid_baby_talk": True,
                "avoid_overformal_tone": True,
                "do_not_invent_user_facts": True,
            }
        },
    )

    identity_core = dict(storage.load_identity_core("asya") or {})
    boundaries = dict(identity_core.get("boundaries") or {})
    assert boundaries == {
        "avoid_overloaded_intros": True,
        "avoid_baby_talk": True,
        "avoid_overformal_tone": True,
        "do_not_invent_user_facts": True,
    }


def test_identity_core_builder_keeps_emotional_handling() -> None:
    builder = IdentityCoreBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={},
        stored_identity_core={
            "emotional_handling": {
                "deescalate_on_irritation": True,
                "treat_short_replies_as_low_bandwidth": True,
                "warmth_upshift_on_user_distress": 0.24,
                "playfulness_downshift_on_user_distress": 0.33,
            }
        },
    )

    row = snapshot.to_dict()
    emotional_handling = dict(row.get("emotional_handling") or {})
    assert emotional_handling == {
        "deescalate_on_irritation": True,
        "treat_short_replies_as_low_bandwidth": True,
        "warmth_upshift_on_user_distress": 0.24,
        "playfulness_downshift_on_user_distress": 0.33,
    }
    assert (
        dict(dict(row.get("debug") or {}).get("sources") or {})
        .get("emotional_handling", {})
        .get("deescalate_on_irritation")
        == "identity_core"
    )


def test_character_storage_persists_identity_core_emotional_handling() -> None:
    root = Path(tempfile.mkdtemp(prefix="mmis-identity-core-emotion-"))
    storage = CharacterStorage(root=root)

    storage.save_identity_core(
        "asya",
        {
            "emotional_handling": {
                "deescalate_on_irritation": True,
                "treat_short_replies_as_low_bandwidth": True,
                "warmth_upshift_on_user_distress": 0.24,
                "playfulness_downshift_on_user_distress": 0.33,
            }
        },
    )

    identity_core = dict(storage.load_identity_core("asya") or {})
    emotional_handling = dict(identity_core.get("emotional_handling") or {})
    assert emotional_handling == {
        "deescalate_on_irritation": True,
        "treat_short_replies_as_low_bandwidth": True,
        "warmth_upshift_on_user_distress": 0.24,
        "playfulness_downshift_on_user_distress": 0.33,
    }


def test_identity_core_builder_keeps_assistant_trait_baseline() -> None:
    builder = IdentityCoreBuilder()

    snapshot = builder.build(
        character_id="asya",
        active_profile_snapshot={
            "assistant_warmth": 0.52,
            "assistant_directness": 0.48,
        },
        stored_identity_core={
            "assistant_trait_baseline": {
                "warmth_baseline": 0.68,
                "directness_baseline": 0.82,
                "empathy_floor": 0.72,
                "professionalism_floor": 0.80,
                "sarcasm_ceiling": 0.18,
            }
        },
    )

    row = snapshot.to_dict()
    baseline = dict(row.get("assistant_trait_baseline") or {})
    assert baseline == {
        "warmth_baseline": 0.68,
        "directness_baseline": 0.82,
        "empathy_floor": 0.72,
        "professionalism_floor": 0.8,
        "sarcasm_ceiling": 0.18,
    }
    assert (
        dict(dict(row.get("debug") or {}).get("sources") or {})
        .get("assistant_trait_baseline", {})
        .get("warmth_baseline")
        == "identity_core"
    )


def test_character_storage_persists_identity_core_assistant_trait_baseline() -> None:
    root = Path(tempfile.mkdtemp(prefix="mmis-identity-core-traits-"))
    storage = CharacterStorage(root=root)

    storage.save_identity_core(
        "asya",
        {
            "assistant_trait_baseline": {
                "warmth_baseline": 0.68,
                "directness_baseline": 0.82,
                "empathy_floor": 0.72,
                "professionalism_floor": 0.80,
                "sarcasm_ceiling": 0.18,
            }
        },
    )

    identity_core = dict(storage.load_identity_core("asya") or {})
    baseline = dict(identity_core.get("assistant_trait_baseline") or {})
    assert baseline == {
        "warmth_baseline": 0.68,
        "directness_baseline": 0.82,
        "empathy_floor": 0.72,
        "professionalism_floor": 0.8,
        "sarcasm_ceiling": 0.18,
    }


def test_character_storage_strips_non_identity_core_fields() -> None:
    root = Path(tempfile.mkdtemp(prefix="mmis-identity-core-strip-"))
    storage = CharacterStorage(root=root)

    storage.save_identity_core(
        "asya",
        {
            "addressing": {
                "canonical_name": "Паша",
            },
            "interaction_style": {
                "prefers_directness": 0.82,
            },
            "boundaries": {
                "avoid_baby_talk": True,
            },
            "emotional_handling": {
                "deescalate_on_irritation": True,
            },
            "project_name": "MMis",
            "environment_runtime_python": "3.11",
            "environment_ram_gb": 32,
            "environment_gpu_model": "RTX 3050 Ti",
            "mood": "frustrated",
            "active_mode": "pirate",
            "temporary_style": {"roleplay": True},
        },
    )

    identity_core = dict(storage.load_identity_core("asya") or {})
    assert set(identity_core.keys()) == {
        "schema_version",
        "addressing",
        "interaction_style",
        "boundaries",
        "emotional_handling",
        "assistant_trait_baseline",
        "updated_at",
    }
    assert "project_name" not in identity_core
    assert "environment_runtime_python" not in identity_core
    assert "environment_ram_gb" not in identity_core
    assert "environment_gpu_model" not in identity_core
    assert "mood" not in identity_core
    assert "active_mode" not in identity_core
    assert "temporary_style" not in identity_core
