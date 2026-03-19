from __future__ import annotations

import time

from memory.dialog_episode_builder import DialogEpisodeBuilder, DialogTurn
from memory.memory_models import MemoryLevel, MemoryType
from test_memory_e2e_scenarios import (
    _build_context_and_prompt,
    _ingest_message,
    _manager,
    _record,
    _system_prompt,
)


def _store_dialog_episode(
    *,
    namespace: str,
    episode_id: str,
    turns_source: list[tuple[str, str, str]],
):
    manager = _manager()
    built_turns: list[DialogTurn] = []
    for idx, (role, text, topic) in enumerate(turns_source):
        _ingest_message(manager, namespace=namespace, role=role, text=text)
        row = next(
            record
            for record in reversed(list(manager._store.iter_records(namespace=namespace)))
            if record.memory_type == MemoryType.MESSAGE and str(record.text or "") == text
        )
        built_turns.append(
            DialogTurn(
                turn_id=str(row.id),
                role=role,
                text=text,
                ts=float(row.created_at or time.time()) + idx,
                topic=topic,
                metadata=dict(row.metadata or {}),
            )
        )

    builder = DialogEpisodeBuilder()
    episode = builder.build_episode(built_turns, episode_id=episode_id)
    assert episode is not None
    manager._store.upsert(
        _record(
            str(episode.id),
            str(episode.summary_short),
            namespace=namespace,
            memory_type=MemoryType.EPISODE,
            level=MemoryLevel.L2_EPISODIC,
            metadata={
                "dialog_episode": episode.to_dict(),
                "topic": episode.topic,
                "summary_short": episode.summary_short,
                "summary_reasoning": episode.summary_reasoning,
                "turn_ids": list(episode.turn_ids or []),
                "decisions": list(episode.decisions or []),
                "open_questions": list(episode.open_questions or []),
                "participants": list(episode.participants or []),
                "topic_keys": list(episode.topic_keys or []),
                "entity_keys": list(episode.entity_keys or []),
                "salience": float(episode.salience or 0.0),
            },
        )
    )
    return manager, episode


def _prompt_memory_text(ctx) -> str:  # noqa: ANN001
    system_prompt = _system_prompt(ctx)
    prompt_memory = str(dict(ctx.state.get("prompt_memory_block") or {}).get("text") or "")
    return (system_prompt + "\n" + prompt_memory).strip()


def test_golden_dialog_why_reasoning_explains_assistant_thoughts_decision() -> None:
    manager, episode = _store_dialog_episode(
        namespace="golden-dialog-why",
        episode_id="episode:assistant-thoughts",
        turns_source=[
            ("user", "We should not store assistant thoughts in long-term memory.", "memory"),
            ("assistant", "Right, assistant thoughts would pollute retrieval and create noisy self-loops.", "memory"),
            ("user", "Let's keep them only in debug logs.", "memory"),
            ("assistant", "We decided to keep assistant thoughts in debug only.", "memory"),
        ],
    )

    result, ctx = _build_context_and_prompt(
        manager,
        namespace="golden-dialog-why",
        user_message="Why did we decide not to store assistant thoughts?",
        top_k=4,
    )

    assert episode.summary_reasoning
    assert "pollute retrieval" in str(episode.summary_reasoning).lower()
    assert episode.decisions
    assert any(
        "debug" in str(item).lower() and ("only" in str(item).lower() or "log" in str(item).lower())
        for item in list(episode.decisions or [])
    )
    assert result.recall_mode == "contextual_recall"
    block = str(result.blocks.get("recalled_dialog") or "")
    assert "Reasoning:" in block
    assert "pollute retrieval" in block.lower()
    assert "Decisions:" in block
    assert "debug" in block.lower()
    assert "[RECALLED_DIALOG]" in _prompt_memory_text(ctx)


def test_golden_dialog_memory_discussion_has_meaningful_summary_and_topic() -> None:
    manager, episode = _store_dialog_episode(
        namespace="golden-dialog-memory",
        episode_id="episode:memory-overview",
        turns_source=[
            ("user", "Привет", "привет"),
            ("assistant", "Ок", "ок"),
            ("user", "Давай разделим память на facts, claims и dialog episodes.", "memory"),
            ("assistant", "И document memory оставим отдельным контуром.", "memory"),
        ],
    )

    result, ctx = _build_context_and_prompt(
        manager,
        namespace="golden-dialog-memory",
        user_message="What did we discuss about memory?",
        top_k=4,
    )

    assert str(episode.topic) == "memory"
    assert "привет" not in str(episode.summary_short).lower()
    assert not str(episode.summary_short).lower().startswith("ok")
    assert "facts, claims" in str(episode.summary_short).lower()
    block = str(result.blocks.get("recalled_dialog") or "")
    assert "Summary:" in block
    assert "facts, claims" in block.lower()
    assert "Topic: memory" in block
    assert "[EPISODIC_MEMORIES]" not in _prompt_memory_text(ctx)


def test_golden_dialog_plan_recall_surfaces_memory_then_web_decision() -> None:
    manager, episode = _store_dialog_episode(
        namespace="golden-dialog-plan",
        episode_id="episode:memory-plan",
        turns_source=[
            ("user", "First we finish memory, then we return to web.", "memory planning"),
            ("assistant", "Agreed, memory first and web second.", "memory planning"),
            ("user", "По вебу потом вернемся отдельно.", "memory planning"),
            ("assistant", "We decided: first memory, then web.", "memory planning"),
        ],
    )

    result, ctx = _build_context_and_prompt(
        manager,
        namespace="golden-dialog-plan",
        user_message="What is our plan now?",
        top_k=4,
    )

    assert result.recall_mode == "contextual_recall"
    block = str(result.blocks.get("recalled_dialog") or "")
    assert "Decisions:" in block
    assert "first memory" in block.lower()
    assert "then web" in block.lower()
    assert any("memory" in str(item).lower() and "web" in str(item).lower() for item in list(episode.decisions or []))
    assert "[RECALLED_DIALOG]" in _prompt_memory_text(ctx)


def test_golden_dialog_new_episode_does_not_merge_with_previous_greeting() -> None:
    manager, episode = _store_dialog_episode(
        namespace="golden-dialog-web",
        episode_id="episode:web-followup",
        turns_source=[
            ("user", "Привет", "привет"),
            ("assistant", "Поняла", "поняла"),
            ("user", "Теперь по вебу: document evidence надо держать отдельно от chat memory.", "web"),
            ("assistant", "Да, web evidence should stay isolated from ordinary dialog memory.", "web"),
        ],
    )

    result, ctx = _build_context_and_prompt(
        manager,
        namespace="golden-dialog-web",
        user_message="What did we discuss about web?",
        top_k=4,
    )

    assert str(episode.topic) == "web"
    assert "привет" not in str(episode.summary_short).lower()
    assert "поняла" not in str(episode.summary_short).lower()
    assert not str(episode.summary_short).lower().startswith("привет")
    block = str(result.blocks.get("recalled_dialog") or "")
    assert "document evidence" in block.lower()
    assert "ordinary dialog memory" in block.lower()
    assert "[EPISODIC_MEMORIES]" not in _prompt_memory_text(ctx)


def test_golden_dialog_why_query_prefers_episode_hit_over_raw_messages() -> None:
    manager, episode = _store_dialog_episode(
        namespace="golden-dialog-ranking",
        episode_id="episode:python-policy",
        turns_source=[
            ("user", "Why are we using singleton latest wins for python facts?", "memory"),
            ("assistant", "Because environment.python should keep one active canonical value.", "memory"),
            ("user", "So old python versions should become superseded?", "memory"),
            ("assistant", "Yes, we decided singleton latest wins for python facts.", "memory"),
        ],
    )
    _ingest_message(manager, namespace="golden-dialog-ranking", role="user", text="why python latest wins?")
    _ingest_message(manager, namespace="golden-dialog-ranking", role="assistant", text="python latest wins because it keeps one active fact")

    result, ctx = _build_context_and_prompt(
        manager,
        namespace="golden-dialog-ranking",
        user_message="Why did we choose singleton latest wins for python?",
        top_k=4,
    )

    assert result.recall_mode == "contextual_recall"
    assert result.dialog_episode_hits
    top_hit = dict(result.dialog_episode_hits[0] or {})
    assert str(top_hit.get("record_id") or "") == str(episode.id)
    assert "singleton latest wins" in str(dict(top_hit.get("episode") or {}).get("summary_reasoning") or "").lower()
    block = str(result.blocks.get("recalled_dialog") or "")
    assert "singleton latest wins" in block.lower()
    prompt_text = _prompt_memory_text(ctx)
    assert "[RECALLED_DIALOG]" in prompt_text
    assert "[EPISODIC_MEMORIES]" not in prompt_text
