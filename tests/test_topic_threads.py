from __future__ import annotations

from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.sqlite_db import Database
from memory_core.topic.topic_router import TopicRouter
from memory_core.topic.topic_store import TopicStore


def _build_topic_router(tmp_path):
    db = Database(str(tmp_path / "memory.db"))
    artifact_store = ArtifactStore(db)
    store = TopicStore(artifact_store)
    router = TopicRouter(store)
    return db, store, router


def test_topic_router_builds_short_human_title(tmp_path) -> None:
    _db, _store, router = _build_topic_router(tmp_path)

    decision = router.route_turn(
        text="Давай разберем как построить дом из дерева, желе и клея",
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        current_state={},
        meta={},
    )

    assert decision.title in {"постройка дома", "дом дерева желе клея"}
    assert decision.title.lower() not in {"чат", "tasks", "general"}


def test_topic_router_allows_independent_question_despite_recent_thread(tmp_path) -> None:
    _db, _store, router = _build_topic_router(tmp_path)

    first = router.route_turn(
        text="Давай разберем как построить дом из дерева, желе и клея",
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        current_state={},
        meta={},
    )
    second = router.route_turn(
        text="почему ollama list пустой?",
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        current_state={"topic_thread_id": first.thread_id},
        meta={},
    )

    assert second.thread_id != first.thread_id
    assert second.is_new_thread is True
    assert second.reason == "new_topic_low_match"


def test_topic_router_keeps_side_followup_inside_current_focus(tmp_path) -> None:
    _db, _store, router = _build_topic_router(tmp_path)

    first = router.route_turn(
        text="Давай разберем как построить дом из дерева, желе и клея",
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        current_state={},
        meta={},
    )
    second = router.route_turn(
        text="а как сделать фундамент?",
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        current_state={"topic_thread_id": first.thread_id},
        meta={},
    )

    assert second.thread_id == first.thread_id
    assert second.reason in {"short_followup", "continue_current_topic", "prefer_current_topic", "semantic_match"}
