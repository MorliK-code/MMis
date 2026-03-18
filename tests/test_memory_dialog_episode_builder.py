from __future__ import annotations

from memory.dialog_episode_builder import DialogEpisodeBoundary, DialogEpisodeBuilder, DialogTurn


def test_dialog_episode_builder_detects_topic_shift_boundary() -> None:
    builder = DialogEpisodeBuilder(min_turns_for_episode=2)
    turns = [
        DialogTurn(turn_id="turn-1", role="user", text="Давай добьем память.", topic="memory", ts=100.0),
        DialogTurn(turn_id="turn-2", role="assistant", text="Ок, сначала retrieval.", topic="memory", ts=110.0),
        DialogTurn(turn_id="turn-3", role="user", text="Теперь глянем веб.", topic="web", ts=120.0),
    ]

    boundary = builder.evaluate_boundary(turns)

    assert boundary == DialogEpisodeBoundary(should_build=True, reason="topic_shift", split_index=2)


def test_dialog_episode_builder_detects_explicit_completion_boundary() -> None:
    builder = DialogEpisodeBuilder(min_turns_for_episode=2)
    turns = [
        DialogTurn(turn_id="turn-1", role="user", text="Давай соберем эпизоды.", topic="memory", ts=100.0),
        DialogTurn(turn_id="turn-2", role="assistant", text="Договорились, идем дальше.", topic="memory", ts=120.0),
    ]

    boundary = builder.evaluate_boundary(turns)

    assert boundary == DialogEpisodeBoundary(should_build=True, reason="explicit_completion", split_index=2)


def test_dialog_episode_builder_detects_time_gap_boundary() -> None:
    builder = DialogEpisodeBuilder(min_turns_for_episode=2, time_gap_sec=300.0)
    turns = [
        DialogTurn(turn_id="turn-1", role="user", text="Поговорим про память.", topic="memory", ts=100.0),
        DialogTurn(turn_id="turn-2", role="assistant", text="Да, начнем с фактов.", topic="memory", ts=180.0),
        DialogTurn(turn_id="turn-3", role="user", text="Спустя время вернулся.", topic="memory", ts=800.0),
    ]

    boundary = builder.evaluate_boundary(turns)

    assert boundary == DialogEpisodeBoundary(should_build=True, reason="time_gap", split_index=2)


def test_dialog_episode_builder_detects_window_limit_boundary() -> None:
    builder = DialogEpisodeBuilder(min_turns_for_episode=2, max_window_turns=3, time_gap_sec=10_000.0)
    turns = [
        DialogTurn(turn_id="turn-1", role="user", text="Шаг один.", topic="memory", ts=100.0),
        DialogTurn(turn_id="turn-2", role="assistant", text="Шаг два.", topic="memory", ts=110.0),
        DialogTurn(turn_id="turn-3", role="user", text="Шаг три.", topic="memory", ts=120.0),
    ]

    boundary = builder.evaluate_boundary(turns)

    assert boundary == DialogEpisodeBoundary(should_build=True, reason="window_limit", split_index=3)


def test_dialog_episode_builder_builds_dialog_episode_payload() -> None:
    builder = DialogEpisodeBuilder(min_turns_for_episode=2)
    turns = [
        DialogTurn(
            turn_id="turn-1",
            role="user",
            text="Давай добьем память, у меня RTX 3050 Ti.",
            topic="memory",
            ts=100.0,
            tags=["topic_memory"],
            metadata={"memory_views": {"entity_keys": ["rtx_3050_ti", "gpu"]}},
        ),
        DialogTurn(
            turn_id="turn-2",
            role="assistant",
            text="Договорились, следующим шагом делаем dialog episode builder.",
            topic="memory",
            ts=110.0,
            tags=["topic_memory"],
        ),
        DialogTurn(
            turn_id="turn-3",
            role="user",
            text="Что осталось открыть по эпизодам?",
            topic="memory",
            ts=120.0,
            tags=["topic_memory"],
        ),
    ]

    episode = builder.build_episode(turns, episode_id="episode:test", now_ts=500.0)

    assert episode is not None
    assert episode.id == "episode:test"
    assert episode.topic == "memory"
    assert episode.turn_ids == ["turn-1", "turn-2", "turn-3"]
    assert episode.participants == ["user", "assistant"]
    assert any("Договорились" in row for row in episode.decisions)
    assert any("Что осталось" in row for row in episode.open_questions)
    assert "rtx_3050_ti" in episode.entity_keys
    assert "memory" in episode.topic_keys
    assert episode.summary_short
    assert episode.summary_reasoning
    assert episode.salience > 0.5
    assert episode.created_at == 500.0
    assert episode.updated_at == 500.0


def test_dialog_episode_builder_build_if_needed_uses_boundary_slice() -> None:
    builder = DialogEpisodeBuilder(min_turns_for_episode=2)
    turns = [
        DialogTurn(turn_id="turn-1", role="user", text="Сначала память.", topic="memory", ts=100.0),
        DialogTurn(turn_id="turn-2", role="assistant", text="Да, добиваем facts.", topic="memory", ts=110.0),
        DialogTurn(turn_id="turn-3", role="user", text="Теперь веб.", topic="web", ts=120.0),
    ]

    boundary, episode = builder.build_if_needed(turns, episode_id="episode:sliced", now_ts=300.0)

    assert boundary == DialogEpisodeBoundary(should_build=True, reason="topic_shift", split_index=2)
    assert episode is not None
    assert episode.id == "episode:sliced"
    assert episode.turn_ids == ["turn-1", "turn-2"]
    assert episode.topic == "memory"
