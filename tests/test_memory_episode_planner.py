from __future__ import annotations

from memory.episode_planner import EpisodePlanner


def test_episode_planner_continues_short_follow_up_for_active_task() -> None:
    planner = EpisodePlanner()

    assert planner.should_continue_active_task(
        user_text="continue",
        active_task={
            "task_id": "task:episode:memory-plan",
            "topic": "memory plan",
            "status": "active",
            "summary_short": "Finish memory before web.",
        },
        meta={},
    ) is True


def test_episode_planner_does_not_continue_on_switch_topic_marker() -> None:
    planner = EpisodePlanner()

    assert planner.should_continue_active_task(
        user_text="by the way, another topic",
        active_task={
            "task_id": "task:episode:memory-plan",
            "topic": "memory plan",
            "status": "active",
        },
        meta={},
    ) is False


def test_episode_planner_uses_followup_like_meta_for_continuation() -> None:
    planner = EpisodePlanner()

    assert planner.should_continue_active_task(
        user_text="yeah",
        active_task={
            "task_id": "task:episode:memory-plan",
            "topic": "memory plan",
            "status": "active",
        },
        meta={"followup_like": True},
    ) is True


def test_episode_planner_builds_active_task_from_structured_episode_hit() -> None:
    planner = EpisodePlanner()

    task = planner.resolve_active_task(
        user_text="return to that memory plan",
        memory_context={
            "dialog_episode_hits": [
                {
                    "record_id": "episode:memory-plan",
                    "score": 0.78,
                    "summary_short": "Discussed staged memory architecture.",
                    "summary_reasoning": "Finish memory before web and keep retrieval clean.",
                    "decisions": ["finish memory before web"],
                    "episode": {
                        "id": "episode:memory-plan",
                        "topic": "memory design",
                        "summary_short": "Discussed staged memory architecture.",
                        "summary_reasoning": "Finish memory before web and keep retrieval clean.",
                        "decisions": ["finish memory before web"],
                        "open_questions": ["how to build fusion retrieval"],
                        "updated_at": 88.0,
                    },
                }
            ],
            "blocks": {
                "recalled_dialog": (
                    "Topic: memory design\n"
                    "Summary: Discussed staged memory architecture.\n"
                    "Decisions:\n"
                    "- finish memory before web\n"
                    "Open questions:\n"
                    "- how to build fusion retrieval"
                ),
            },
            "selected": [
                {
                    "id": "episode:memory-plan",
                    "memory_type": "episode",
                    "score": 0.70,
                    "metadata": {
                        "dialog_episode": {
                            "id": "episode:memory-plan",
                            "topic": "memory design",
                            "summary_short": "Discussed staged memory architecture.",
                            "decisions": ["finish memory before web"],
                            "open_questions": ["how to build fusion retrieval"],
                        }
                    },
                }
            ],
        },
        state={},
        meta={"now_ts": 100.0},
    )

    assert task is not None
    assert task.task_id == "task:episode:memory-plan"
    assert task.topic == "memory design"
    assert task.status == "waiting_user"
    assert task.current_goal == "Finish memory before web and keep retrieval clean."
    assert task.next_steps == ["finish memory before web"]
    assert task.decisions == ["finish memory before web"]
    assert task.open_questions == ["how to build fusion retrieval"]
    assert task.updated_at == 100.0


def test_episode_planner_does_not_create_active_task_from_recalled_dialog_without_episode_hit() -> None:
    planner = EpisodePlanner()

    task = planner.resolve_active_task(
        user_text="return to that memory plan",
        memory_context={
            "blocks": {
                "recalled_dialog": (
                    "Topic: memory design\n"
                    "Summary: Discussed staged memory architecture.\n"
                    "Decisions:\n"
                    "- finish memory before web\n"
                    "Open questions:\n"
                    "- how to build fusion retrieval"
                ),
            },
            "selected": [
                {
                    "id": "message:memory-plan",
                    "memory_type": "message",
                    "score": 0.92,
                    "text": "finish memory before web",
                    "metadata": {},
                }
            ],
        },
        state={},
        meta={"now_ts": 100.0},
    )

    assert task is None


def test_episode_planner_builds_fallback_active_task_from_runtime_hints() -> None:
    planner = EpisodePlanner()

    task = planner.resolve_active_task(
        user_text="return to the plan",
        memory_context={
            "open_questions": ["how to build fusion retrieval"],
            "current_decisions": ["finish memory before web"],
        },
        state={},
        meta={"now_ts": 100.0},
    )

    assert task is not None
    assert task.task_id.startswith("task:runtime:")
    assert task.status == "waiting_user"
    assert task.current_goal == "finish memory before web"
    assert task.decisions == ["finish memory before web"]
    assert task.open_questions == ["how to build fusion retrieval"]
    assert task.updated_at == 100.0


def test_episode_planner_does_not_activate_closed_episode_from_decisions_only() -> None:
    planner = EpisodePlanner()

    task = planner.resolve_active_task(
        user_text="return to that memory plan",
        memory_context={
            "dialog_episode_hits": [
                {
                    "record_id": "episode:memory-plan",
                    "score": 0.79,
                    "summary_short": "Finished the memory plan.",
                    "summary_reasoning": "",
                    "decisions": ["finish memory before web"],
                    "episode": {
                        "id": "episode:memory-plan",
                        "topic": "memory design",
                        "summary_short": "Finished the memory plan.",
                        "status": "done",
                        "decisions": ["finish memory before web"],
                        "open_questions": [],
                    },
                }
            ],
        },
        state={},
        meta={"now_ts": 100.0},
    )

    assert task is None


def test_episode_planner_maybe_close_marks_done() -> None:
    planner = EpisodePlanner()

    closed = planner.maybe_close_active_task(
        user_text="done, we are finished",
        active_task={
            "task_id": "task:episode:memory-plan",
            "topic": "memory design",
            "status": "active",
        },
    )

    assert closed is not None
    assert closed["status"] == "done"


def test_episode_planner_maybe_close_marks_done_for_resolved() -> None:
    planner = EpisodePlanner()

    closed = planner.maybe_close_active_task(
        user_text="resolved, we are done here",
        active_task={
            "task_id": "task:episode:memory-plan",
            "topic": "memory design",
            "status": "active",
        },
    )

    assert closed is not None
    assert closed["status"] == "done"


def test_episode_planner_maybe_close_marks_blocked() -> None:
    planner = EpisodePlanner()

    closed = planner.maybe_close_active_task(
        user_text="we are blocked and stuck",
        active_task={
            "task_id": "task:episode:memory-plan",
            "topic": "memory design",
            "status": "active",
        },
    )

    assert closed is not None
    assert closed["status"] == "blocked"


def test_episode_planner_closes_existing_task_when_matching_episode_is_done() -> None:
    planner = EpisodePlanner()

    task = planner.resolve_active_task(
        user_text="continue",
        memory_context={
            "dialog_episode_hits": [
                {
                    "record_id": "episode:memory-plan",
                    "score": 0.83,
                    "episode": {
                        "id": "episode:memory-plan",
                        "topic": "memory design",
                        "summary_short": "Finished the memory plan.",
                        "status": "done",
                        "decisions": ["finish memory before web"],
                        "open_questions": [],
                        "updated_at": 91.0,
                    },
                }
            ],
        },
        state={
            "active_task": {
                "task_id": "task:episode:memory-plan",
                "topic": "memory design",
                "status": "active",
                "summary_short": "Discussed staged memory architecture.",
                "current_goal": "finish memory before web",
                "source_episode_id": "episode:memory-plan",
            }
        },
        meta={"now_ts": 100.0},
    )

    assert task is not None
    assert task.status == "done"
    assert task.updated_at == 100.0


def test_episode_planner_restores_previous_task_from_task_history() -> None:
    planner = EpisodePlanner()

    task = planner.resolve_active_task(
        user_text="return to previous task",
        memory_context={
            "task_continuity": {
                "active_task": {
                    "task_id": "task:current-web",
                    "topic": "web isolation",
                    "status": "active",
                    "current_goal": "Finish web isolation cleanup.",
                },
                "task_history": [
                    {
                        "task_id": "task:memory-loop",
                        "topic": "memory loop",
                        "status": "blocked",
                        "current_goal": "Finish Memory -> Persona -> Prompt wiring.",
                        "next_steps": ["stabilize persona snapshot"],
                        "open_questions": ["how to update task continuity after replies"],
                        "decisions": ["use episode planner as continuity source"],
                    }
                ],
            }
        },
        state={},
        meta={"now_ts": 120.0},
    )

    assert task is not None
    assert task.task_id == "task:memory-loop"
    assert task.status == "blocked"
    assert task.current_goal == "Finish Memory -> Persona -> Prompt wiring."
    assert task.planner_source == "task_history"
    assert task.planner_reason == "return_to_previous_task"


def test_episode_planner_updates_active_task_from_assistant_reply() -> None:
    planner = EpisodePlanner()

    task = planner.update_active_task_after_assistant_reply(
        assistant_text=(
            "Let's continue the memory loop.\n"
            "1. Stabilize episode continuity writes.\n"
            "2. Sync task continuity into memory manager."
        ),
        active_task={
            "task_id": "task:memory-loop",
            "topic": "memory loop",
            "status": "waiting_user",
            "current_goal": "Need a continuity plan.",
            "open_questions": ["how do we persist task state?"],
            "decisions": ["keep episode planner central"],
        },
        meta={"now_ts": 130.0},
    )

    assert task is not None
    assert task.task_id == "task:memory-loop"
    assert task.status == "active"
    assert task.next_steps == [
        "Stabilize episode continuity writes",
        "Sync task continuity into memory manager",
    ]
    assert "keep episode planner central" in list(task.decisions or [])
    assert task.open_questions == []
    assert task.planner_source == "assistant_reply"
    assert task.planner_reason == "assistant_progress_update"
