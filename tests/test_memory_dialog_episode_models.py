from __future__ import annotations

from memory.dialog_episode_models import DialogEpisode


def test_dialog_episode_roundtrip() -> None:
    episode = DialogEpisode(
        id="episode:memory-1",
        topic="memory architecture",
        turn_ids=["turn-1", "turn-2"],
        summary_short="Discussed staged memory architecture.",
        summary_reasoning="Separated facts, claims, episodes, and docs into different layers.",
        decisions=["stabilize memory base before new subsystems"],
        open_questions=["how to synthesize dialog episodes from turns"],
        participants=["user", "assistant"],
        salience=0.84,
        topic_keys=["memory", "architecture"],
        entity_keys=["memory_manager", "retrieval"],
        created_at=100.0,
        updated_at=120.0,
    )

    restored = DialogEpisode.from_dict(episode.to_dict())

    assert restored == episode
