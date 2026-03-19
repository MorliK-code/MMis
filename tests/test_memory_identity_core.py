from __future__ import annotations

from memory.identity_core import (
    IDENTITY_CORE_KEYS,
    IdentityCoreCandidate,
    IdentityCoreManager,
    IdentityCoreRecord,
)


def test_identity_core_record_to_dict_roundtrip_shape() -> None:
    row = IdentityCoreRecord(
        key="addressing.canonical_name",
        value="Pasha",
        confidence=0.95,
        source="user_confirmed",
        requires_confirmation_to_override=True,
        updated_at=123.0,
    )

    assert row.to_dict() == {
        "key": "addressing.canonical_name",
        "value": "Pasha",
        "confidence": 0.95,
        "source": "user_confirmed",
        "requires_confirmation_to_override": True,
        "updated_at": 123.0,
    }


def test_identity_core_manager_builds_snapshot_from_grouped_rows() -> None:
    manager = IdentityCoreManager()

    snapshot = manager.build_snapshot(
        rows=[
            {
                "key": "addressing.canonical_name",
                "value": "Pasha",
                "confidence": 1.0,
                "updated_at": 10.0,
            },
            {
                "key": "interaction_style.prefers_directness",
                "value": 0.82,
                "confidence": 0.9,
                "updated_at": 9.0,
            },
            {
                "key": "boundaries.avoid_baby_talk",
                "value": True,
                "confidence": 0.95,
                "updated_at": 8.0,
            },
            {
                "key": "emotional_handling.deescalate_on_irritation",
                "value": True,
                "confidence": 0.97,
                "updated_at": 11.0,
            },
            {
                "key": "assistant_trait_baseline.warmth",
                "value": 0.66,
                "confidence": 0.88,
                "updated_at": 12.0,
            },
        ]
    )

    assert snapshot.addressing["canonical_name"] == "Pasha"
    assert snapshot.interaction_style["prefers_directness"] == 0.82
    assert snapshot.boundaries["avoid_baby_tone"] is True
    assert snapshot.emotional_rules["frustration_softening"] == 1.0
    assert snapshot.assistant_trait_baseline["warmth_baseline"] == 0.66


def test_identity_core_manager_prefers_higher_confidence_then_newer_value() -> None:
    manager = IdentityCoreManager()

    snapshot = manager.build_snapshot(
        rows=[
            {
                "key": "addressing.canonical_name",
                "value": "Pasha",
                "confidence": 0.60,
                "updated_at": 100.0,
            },
            {
                "key": "addressing.canonical_name",
                "value": "Pash",
                "confidence": 0.95,
                "updated_at": 10.0,
            },
        ]
    )

    assert snapshot.addressing["canonical_name"] == "Pash"


def test_identity_core_manager_blocks_override_without_confirmation() -> None:
    manager = IdentityCoreManager()

    assert manager.should_allow_override(
        key="addressing.canonical_name",
        old_value="Pasha",
        new_value="Pashka",
        explicit_confirmation=False,
    ) is False
    assert manager.should_allow_override(
        key="addressing.canonical_name",
        old_value="Pasha",
        new_value="Pashka",
        explicit_confirmation=True,
    ) is True
    assert manager.should_allow_override(
        key="boundaries.avoid_baby_talk",
        old_value=True,
        new_value=True,
        explicit_confirmation=False,
    ) is True


def test_identity_core_manager_blocks_protected_assistant_trait_override_without_confirmation() -> None:
    manager = IdentityCoreManager()

    assert manager.should_allow_override(
        key="assistant_traits.warmth_baseline",
        old_value=0.58,
        new_value=0.72,
        explicit_confirmation=False,
    ) is False


def test_identity_core_manager_allows_non_protected_override_without_confirmation() -> None:
    manager = IdentityCoreManager()

    assert manager.should_allow_override(
        key="interaction.prefers_examples_on_user_code",
        old_value=True,
        new_value=False,
        explicit_confirmation=False,
    ) is True


def test_identity_core_keys_whitelist_contains_canonical_keys() -> None:
    assert "addressing.canonical_name" in IDENTITY_CORE_KEYS
    assert "interaction.prefers_examples_on_user_code" in IDENTITY_CORE_KEYS
    assert "boundaries.avoid_inventing_user_facts" in IDENTITY_CORE_KEYS
    assert "emotional.frustration_softening" in IDENTITY_CORE_KEYS
    assert "assistant_traits.sarcasm_ceiling" in IDENTITY_CORE_KEYS


def test_identity_core_manager_ignores_non_whitelisted_keys() -> None:
    manager = IdentityCoreManager()

    snapshot = manager.build_snapshot(
        rows=[
            {"key": "addressing.canonical_name", "value": "Pasha", "confidence": 1.0},
            {"key": "project_name", "value": "MMis", "confidence": 1.0},
            {"key": "environment_runtime_python", "value": "3.11", "confidence": 1.0},
            {"key": "boundaries.avoid_overformal_tone", "value": True, "confidence": 1.0},
            {"key": "today_mode", "value": "pirate", "confidence": 1.0},
        ]
    )

    row = snapshot.to_dict()
    assert row["addressing"]["canonical_name"] == "Pasha"
    assert row["interaction_style"] == {}
    assert row["boundaries"] == {}
    assert row["emotional_rules"] == {}
    assert row["assistant_trait_baseline"] == {}


def test_identity_core_policy_allows_canonical_name_from_strong_self_identification() -> None:
    manager = IdentityCoreManager()
    candidate = IdentityCoreCandidate(
        record=IdentityCoreRecord(
            key="addressing.canonical_name",
            value="Pasha",
            confidence=0.91,
            source="fact:self_identification",
        ),
        signals={"stable_self_identification": True},
    )

    decision = manager.decide_write(
        candidate=candidate,
        existing_record=None,
        source_role="user",
    )

    assert decision.allow_write is True
    assert decision.reason == "canonical_name_strong_self_identification"


def test_identity_core_policy_blocks_canonical_name_override_without_confirmation() -> None:
    manager = IdentityCoreManager()
    candidate = IdentityCoreCandidate(
        record=IdentityCoreRecord(
            key="addressing.canonical_name",
            value="Pashka",
            confidence=0.95,
            source="fact:self_identification",
        ),
        signals={"stable_self_identification": True},
    )

    decision = manager.decide_write(
        candidate=candidate,
        existing_record=IdentityCoreRecord(
            key="addressing.canonical_name",
            value="Pasha",
            confidence=0.93,
            source="user_confirmed",
        ),
        source_role="user",
    )

    assert decision.allow_write is False
    assert decision.reason == "override_requires_confirmation"


def test_identity_core_policy_allows_forbidden_form_from_direct_prohibition() -> None:
    manager = IdentityCoreManager()
    candidate = IdentityCoreCandidate(
        record=IdentityCoreRecord(
            key="addressing.forbidden_forms",
            value=["Pashka"],
            confidence=0.97,
            source="user_direct_prohibition",
        ),
        signals={"direct_user_prohibition": True},
    )

    decision = manager.decide_write(
        candidate=candidate,
        existing_record=None,
        source_role="user",
    )

    assert decision.allow_write is True
    assert decision.reason == "direct_user_prohibition"


def test_identity_core_policy_blocks_assistant_traits_without_explicit_confirmation() -> None:
    manager = IdentityCoreManager()
    candidate = IdentityCoreCandidate(
        record=IdentityCoreRecord(
            key="assistant_traits.warmth_baseline",
            value=0.68,
            confidence=1.0,
            source="user_preference",
        ),
    )

    decision = manager.decide_write(
        candidate=candidate,
        existing_record=None,
        source_role="user",
    )

    assert decision.allow_write is False
    assert decision.reason == "assistant_traits_require_system_or_confirmation"


def test_identity_core_manager_extracts_boundary_and_interaction_candidates_from_message() -> None:
    manager = IdentityCoreManager()

    rows = manager.extract_candidates_from_message(
        text="Не называй меня Пашка и лучше показывай примеры на моем коде.",
        metadata={},
        source_role="user",
        now_ts=10.0,
    )

    payload = [item.to_dict() for item in rows]
    assert any(dict(item.get("record") or {}).get("key") == "addressing.forbidden_forms" for item in payload)
    assert any(dict(item.get("record") or {}).get("key") == "interaction.prefers_examples_on_user_code" for item in payload)
    forbidden = next(
        (dict(item.get("record") or {}) for item in payload if dict(item.get("record") or {}).get("key") == "addressing.forbidden_forms"),
        {},
    )
    assert "Пашка" in list(forbidden.get("value") or [])


def test_identity_core_manager_extracts_canonical_name_candidate_from_fact() -> None:
    manager = IdentityCoreManager()

    class _Fact:
        predicate = "identity_name"
        value = "Pasha"
        confidence = 0.90
        updated_at = 12.0
        metadata = {}

    rows = manager.extract_candidates_from_fact(
        fact=_Fact(),
        source_role="user",
        now_ts=12.0,
    )

    assert len(rows) == 1
    assert rows[0].record.key == "addressing.canonical_name"
    assert rows[0].record.value == "Pasha"
