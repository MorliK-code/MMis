from __future__ import annotations

import tempfile
from pathlib import Path

from modules.character.composer import CharacterComposer
from modules.character.storage import CharacterStorage


def _storage() -> CharacterStorage:
    root = Path(tempfile.mkdtemp(prefix="mmis-character-composer-"))
    return CharacterStorage(root=root)


def test_compose_uses_persona_snapshot_user_addressing_and_mood_without_persisting() -> None:
    storage = _storage()
    composer = CharacterComposer()
    before_persona = dict(storage.load_persona_state("asya") or {})
    before_addressing = dict(storage.load_user_addressing("asya") or {})

    result = composer.compose(
        storage=storage,
        character_id="asya",
        character={"default_mood": "thoughtful"},
        state={},
        traits={},
        persona_snapshot={
            "mood": "frustrated",
            "user_addressing": {
                "canonical_name": "Паша",
                "allowed_forms": ["Паша"],
                "use_name_by_default": True,
            },
            "relation_state": {
                "familiarity": 0.91,
                "trust": 0.88,
            },
        },
    )

    assert result.mood == "frustrated"
    assert "canonical_name: Паша" in result.prompt
    assert dict(storage.load_persona_state("asya") or {}) == before_persona
    assert dict(storage.load_user_addressing("asya") or {}) == before_addressing


def test_compose_softly_overlays_snapshot_stable_traits() -> None:
    storage = _storage()
    composer = CharacterComposer()
    persona_state = dict(storage.load_persona_state("asya") or {})
    persona_state["traits"] = {"warmth": 0.20}
    storage.save_persona_state("asya", persona_state)

    result = composer.compose(
        storage=storage,
        character_id="asya",
        character={"default_mood": "thoughtful"},
        state={},
        traits={},
        persona_snapshot={
            "stable_traits": {
                "warmth": 0.80,
            },
        },
    )

    warmth = float(result.effective_traits.get("warmth") or 0.0)
    assert 0.55 <= warmth <= 0.61


def test_compose_applies_response_bias_only_to_current_compose() -> None:
    storage = _storage()
    composer = CharacterComposer()
    before_persona = dict(storage.load_persona_state("asya") or {})

    result = composer.compose(
        storage=storage,
        character_id="asya",
        character={"default_mood": "thoughtful"},
        state={},
        traits={},
        persona_snapshot={
            "response_bias": {
                "technical_mode": 1.0,
                "needs_short_answer": 1.0,
                "frustration_softening": 1.0,
            },
        },
    )

    assert float(result.style_coefficients.get("strictness") or 0.0) > 0.5
    assert float(result.style_coefficients.get("verbosity") or 1.0) < 0.5
    assert float(result.style_coefficients.get("warmth") or 0.0) > 0.5
    assert "response_bias" not in dict(storage.load_persona_state("asya") or {})
    assert dict(storage.load_persona_state("asya") or {}) == before_persona


def test_compose_applies_snapshot_boundaries_into_prompt_without_persisting() -> None:
    storage = _storage()
    composer = CharacterComposer()
    before_persona = dict(storage.load_persona_state("asya") or {})

    result = composer.compose(
        storage=storage,
        character_id="asya",
        character={"default_mood": "thoughtful"},
        state={},
        traits={},
        persona_snapshot={
            "boundaries": {
                "avoid_overloaded_intros": True,
                "avoid_baby_talk": True,
                "avoid_overformal_tone": True,
                "do_not_invent_user_facts": True,
            }
        },
    )

    assert "[PERSONA_BOUNDARIES]" in result.prompt
    assert "Do not invent facts about the user" in result.prompt
    assert "Do not use baby-talk" in result.prompt
    assert dict(storage.load_persona_state("asya") or {}) == before_persona


def test_compose_applies_snapshot_emotional_handling_into_prompt_and_style_bias() -> None:
    storage = _storage()
    composer = CharacterComposer()
    before_persona = dict(storage.load_persona_state("asya") or {})

    result = composer.compose(
        storage=storage,
        character_id="asya",
        character={"default_mood": "thoughtful"},
        state={},
        traits={},
        persona_snapshot={
            "response_bias": {
                "warmth_upshift": 0.24,
                "playfulness_downshift": 0.33,
            },
            "emotional_handling": {
                "deescalate_on_irritation": True,
                "treat_short_replies_as_low_bandwidth": True,
                "warmth_upshift_on_user_distress": 0.24,
                "playfulness_downshift_on_user_distress": 0.33,
            },
        },
    )

    assert "[PERSONA_EMOTIONAL_HANDLING]" in result.prompt
    assert "de-escalate first and then move to solving" in result.prompt
    assert "Treat short user replies as low bandwidth" in result.prompt
    assert float(result.style_coefficients.get("warmth") or 0.0) > 0.5
    assert float(result.style_coefficients.get("sarcasm") or 1.0) < 0.5
    assert dict(storage.load_persona_state("asya") or {}) == before_persona


def test_compose_applies_identity_core_trait_baseline_and_addressing_priority() -> None:
    storage = _storage()
    composer = CharacterComposer()
    before_addressing = dict(storage.load_user_addressing("asya") or {})

    persona_state = dict(storage.load_persona_state("asya") or {})
    persona_state["traits"] = {
        "warmth": 0.20,
        "sarcasm": 0.74,
        "empathy": 0.30,
        "professionalism": 0.40,
    }
    storage.save_persona_state("asya", persona_state)
    before_persona = dict(storage.load_persona_state("asya") or {})

    result = composer.compose(
        storage=storage,
        character_id="asya",
        character={"default_mood": "thoughtful"},
        state={},
        traits={},
        identity_core_snapshot={
            "addressing": {
                "canonical_name": "Паш",
                "allowed_forms": ["Паш"],
            },
            "assistant_trait_baseline": {
                "warmth_baseline": 0.66,
                "empathy_floor": 0.72,
                "professionalism_floor": 0.80,
                "sarcasm_ceiling": 0.18,
            },
        },
    )

    assert "canonical_name: Паш" in result.prompt
    assert float(result.effective_traits.get("warmth") or 0.0) == 0.66
    assert float(result.effective_traits.get("empathy") or 0.0) == 0.72
    assert float(result.effective_traits.get("sarcasm") or 1.0) == 0.18
    assert dict(storage.load_persona_state("asya") or {}) == before_persona
    assert dict(storage.load_user_addressing("asya") or {}) == before_addressing
