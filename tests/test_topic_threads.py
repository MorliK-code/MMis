from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.memory_inspector_router as inspector_router_module
from api.memory_inspector_router import create_memory_inspector_router
from core.response_pipeline import MemoryRetrieveStage, MemoryWriteStage, PipelineContext, ResponsePipeline, TopicRoutingStage, _agent_loop_tools
from llm.provider_base import ToolCall
from memory_core.config_manager import RetrievalConfig
from memory_core.governor.governor import Governor
from memory_core.inspect.inspector_service import MemoryInspectorService
from memory_core.planner.episode_planner import EpisodePlanner
from memory_core.processors.memory_llm_processor import ArtifactProposal, MemoryLLMResult
from memory_core.retrieval.retrieval_service import RetrievalService
from memory_core.schemas import MemoryArtifact, MemoryEnvelope, MemoryQuery
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.event_store import EventStore
from memory_core.storage.job_queue_store import JobQueueStore
from memory_core.storage.sqlite_db import Database
from memory_core.topic import TopicRouteDecision, TopicRouter, TopicStore, TopicSummaryBuilder, TopicThread, TopicToolService
from memory_core.worker.background_worker import BackgroundWorker
from prompt_engine.prompt_engine import PromptEngine


def _build_stores(tmp_path):
    db = Database(str(tmp_path / "topic_threads.db"))
    return db, ArtifactStore(db), EventStore(db)


def test_topic_router_reuses_existing_thread_for_similar_followup(tmp_path) -> None:
    _, artifact_store, _ = _build_stores(tmp_path)
    store = TopicStore(artifact_store)
    router = TopicRouter(store)

    first = router.route_turn(
        text="Давай разберем память и continuity в одном чате",
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        current_state={},
        meta={},
    )
    second = router.route_turn(
        text="А как лучше переделать continuity?",
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        current_state={"topic_thread_id": first.thread_id},
        meta={},
    )

    assert first.is_new_thread is True
    assert second.is_new_thread is False
    assert second.thread_id == first.thread_id
    assert second.reason in {"semantic_match", "prefer_current_topic", "short_followup"}


def test_episode_planner_scopes_active_episode_by_topic_thread_id(tmp_path) -> None:
    _, artifact_store, event_store = _build_stores(tmp_path)
    planner = EpisodePlanner(artifact_store=artifact_store, event_store=event_store)

    first = planner.get_or_create_episode(
        session_id="chat-1",
        workspace_id="global",
        topic_thread_id="thr-memory",
    )
    second = planner.get_or_create_episode(
        session_id="chat-1",
        workspace_id="global",
        topic_thread_id="thr-ui",
    )
    first_again = planner.get_or_create_episode(
        session_id="chat-1",
        workspace_id="global",
        topic_thread_id="thr-memory",
    )

    assert first.episode_id != second.episode_id
    assert first_again.episode_id == first.episode_id
    assert first.metadata["topic_thread_id"] == "thr-memory"
    assert second.metadata["topic_thread_id"] == "thr-ui"


def test_retrieval_service_prioritizes_same_topic_continuity(tmp_path) -> None:
    db, artifact_store, _ = _build_stores(tmp_path)
    service = RetrievalService(
        db=db,
        vector_index=None,
        retrieval_config=RetrievalConfig(
            always_load=[],
            sometimes_load=[],
            never_load_if_decay="immediate",
            never_load_if_status="superseded",
            never_load_if_confidence_below=0.0,
        ),
    )

    artifact_store.create(
        MemoryArtifact(
            artifact_id="episode-other",
            artifact_type="episode_event",
            source_event_id="event-1",
            text="Старая UI тема",
            summary="UI",
            metadata={"session_id": "chat-1", "topic_thread_id": "thr-ui", "confidence": 0.9},
            workspace_id="global",
            updated_at=200.0,
            created_at=200.0,
        )
    )
    artifact_store.create(
        MemoryArtifact(
            artifact_id="episode-memory",
            artifact_type="episode_event",
            source_event_id="event-2",
            text="Текущая тема про память",
            summary="Memory",
            metadata={"session_id": "chat-1", "topic_thread_id": "thr-memory", "confidence": 0.9},
            workspace_id="global",
            updated_at=100.0,
            created_at=100.0,
        )
    )

    continuity = service._collect_continuity_artifacts(
        MemoryQuery(
            text="память",
            workspace_id="global",
            session_id="chat-1",
            topic_thread_id="thr-memory",
            top_k=8,
        )
    )

    assert [artifact.artifact_id for artifact in continuity[:2]] == ["episode-memory", "episode-other"]


def test_topic_routing_stage_updates_state_and_meta() -> None:
    class _Router:
        def route_turn(self, **kwargs):
            return TopicRouteDecision(
                thread_id="thr-memory",
                topic_key="memory",
                title="Память / continuity",
                reason="semantic_match",
                score=0.88,
                related_thread_ids=["thr-ui"],
            )

    ctx = PipelineContext(
        route="chat",
        user_msg="давай еще про память",
        state={"conversation_id": "chat-1", "context_tags": {}},
        meta={"conversation_id": "chat-1"},
        retrieved_memories=[],
        traits={},
        policies={},
        profile="BALANCED",
        clean_user_msg="давай еще про память",
    )

    updated = TopicRoutingStage(topic_router=_Router()).run(ctx)

    assert updated.state["topic_thread_id"] == "thr-memory"
    assert updated.state["active_topic_thread_id"] == "thr-memory"
    assert updated.meta["topic_thread_id"] == "thr-memory"
    assert updated.meta["topic_key"] == "memory"
    assert updated.tags["topic"] == "memory"
    assert updated.state["context_tags"]["topic"] == "memory"
    assert updated.state["topic_stack"][0]["thread_id"] == "thr-memory"


def test_memory_write_stage_attaches_topic_metadata_to_turn_ops() -> None:
    ctx = PipelineContext(
        route="chat",
        user_msg="hello",
        state={
            "conversation_id": "chat-1",
            "topic_thread_id": "thr-memory",
            "topic_key": "memory",
            "topic_thread_title": "Память / continuity",
        },
        meta={"conversation_id": "chat-1"},
        retrieved_memories=[],
        traits={},
        policies={},
        profile="BALANCED",
        clean_user_msg="hello",
        text="hi back",
        tags={"topic": "memory"},
    )

    result = MemoryWriteStage(memory_manager=object()).run(ctx)
    user_op = next(item for item in result.memory_ops if item["op"] == "turn_user")
    assistant_op = next(item for item in result.memory_ops if item["op"] == "turn_assistant")

    assert user_op["tags"]["topic_thread_id"] == "thr-memory"
    assert assistant_op["tags"]["topic_key"] == "memory"
    assert assistant_op["tags"]["topic_title"] == "Память / continuity"


def test_response_pipeline_profiles_include_topic_routing() -> None:
    pipeline = ResponsePipeline(provider=object())

    assert "topic_routing" in pipeline._resolve_stage_names("FAST", {}, {})
    assert "topic_routing" in pipeline._resolve_stage_names("BALANCED", {}, {})
    assert "topic_routing" in pipeline._resolve_stage_names("QUALITY", {}, {})


def test_inspector_service_lists_topics_and_topic_details(tmp_path) -> None:
    _, artifact_store, _ = _build_stores(tmp_path)
    topic_store = TopicStore(artifact_store)
    thread = topic_store.create_thread(
        TopicThread.create(
            visible_chat_id="chat-1",
            workspace_id="global",
            session_id="chat-1",
            topic_key="memory",
            title="Память / continuity",
            tags=["memory", "architecture"],
        )
    )
    artifact_store.create(
        MemoryArtifact(
            artifact_id="task-1",
            artifact_type="task_state",
            source_event_id="event-1",
            text="Переделать continuity retrieval",
            summary="continuity task",
            metadata={
                "topic_thread_id": thread.thread_id,
                "task_status": "open",
                "open_questions": ["Как хранить topic_thread_id в episode?"],
                "decisions": ["Сначала делаем topic summaries без отдельной topic table"],
            },
            workspace_id="global",
        )
    )
    artifact_store.create(
        MemoryArtifact(
            artifact_id="episode-1",
            artifact_type="episode_event",
            source_event_id="event-2",
            text="Эпизод про память",
            summary="memory episode",
            metadata={"topic_thread_id": thread.thread_id},
            workspace_id="global",
        )
    )

    memory_core = SimpleNamespace(service=SimpleNamespace(artifact_store=artifact_store))
    service = MemoryInspectorService(memory_core)

    topics = service.list_topics(visible_chat_id="chat-1", limit=20)
    details = service.get_topic_details(thread.thread_id)

    assert len(topics) == 1
    assert topics[0]["thread_id"] == thread.thread_id
    assert details is not None
    assert details["episode_count"] == 1
    assert details["open_questions"] == ["Как хранить topic_thread_id в episode?"]
    assert details["current_decisions"] == ["Сначала делаем topic summaries без отдельной topic table"]
    assert details["artifact_count"] == 2


def test_memory_inspector_router_exposes_topic_endpoints() -> None:
    original = inspector_router_module.MemoryInspectorService

    class _FakeInspectorService:
        def __init__(self, memory_core):
            self.memory_core = memory_core

        def list_topics(self, **kwargs):
            return [{"thread_id": "thr-memory", "title": "Память"}]

        def get_topic_details(self, thread_id: str):
            return {"thread_id": thread_id, "title": "Память", "artifact_count": 3}

        def get_overview(self):
            return {}

        def list_artifacts(self, **kwargs):
            return []

        def list_jobs(self, **kwargs):
            return []

        def get_pipeline_trace(self, **kwargs):
            return []

        def get_checks(self):
            return []

        def get_raw_event(self, event_id):
            return None

        def get_artifact(self, artifact_id):
            return None

        def get_event_trace(self, event_id):
            return None

        def retry_job(self, job_id):
            return {"ok": True}

        def delete_job(self, job_id):
            return {"ok": True}

    inspector_router_module.MemoryInspectorService = _FakeInspectorService
    try:
        app = FastAPI()
        app.include_router(create_memory_inspector_router(memory_core=object()))
        client = TestClient(app)

        topics_response = client.get("/api/memory-core/topics")
        detail_response = client.get("/api/memory-core/topics/thr-memory")

        assert topics_response.status_code == 200
        assert topics_response.json()[0]["thread_id"] == "thr-memory"
        assert detail_response.status_code == 200
        assert detail_response.json()["artifact_count"] == 3
    finally:
        inspector_router_module.MemoryInspectorService = original


def test_topic_tool_service_search_read_and_related(tmp_path) -> None:
    _, artifact_store, _ = _build_stores(tmp_path)
    store = TopicStore(artifact_store)
    memory_thread = store.create_thread(
        TopicThread.create(
            visible_chat_id="chat-1",
            workspace_id="global",
            session_id="chat-1",
            topic_key="memory_architecture",
            title="Memory / continuity / hidden topics",
            tags=["memory", "continuity", "architecture"],
            related_thread_ids=["thr-ui"],
        )
    )
    ui_thread = TopicThread.create(
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        topic_key="browser_ui",
        title="Browser UI for MMis",
        tags=["ui", "browser", "memory"],
        metadata={"created_by": "test"},
    )
    ui_thread.thread_id = "thr-ui"
    store.create_thread(ui_thread)

    artifact_store.create(
        MemoryArtifact(
            artifact_id="task-memory",
            artifact_type="task_state",
            source_event_id="event-1",
            text="Переделать continuity retrieval",
            summary="Continuity retrieval task",
            metadata={
                "topic_thread_id": memory_thread.thread_id,
                "task_status": "open",
                "open_questions": ["Как читать hidden topics через tools?"],
                "decisions": ["Сначала открываем тему через topic_read, потом углубляем retrieval"],
            },
            workspace_id="global",
        )
    )
    artifact_store.create(
        MemoryArtifact(
            artifact_id="episode-memory",
            artifact_type="episode_event",
            source_event_id="event-2",
            text="Эпизод про hidden topic threads",
            summary="Topic thread episode",
            metadata={"topic_thread_id": memory_thread.thread_id},
            workspace_id="global",
        )
    )

    service = TopicToolService(store)
    search = service.search_topics(
        query="continuity hidden topics",
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        limit=5,
    )
    read = service.read_topic(memory_thread.thread_id, limit=5, workspace_id="global")
    related = service.related_topics(
        memory_thread.thread_id,
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        limit=5,
    )

    assert search["topics"][0]["thread_id"] == memory_thread.thread_id
    assert read is not None
    assert read["open_questions"] == ["Как читать hidden topics через tools?"]
    assert read["current_decisions"] == ["Сначала открываем тему через topic_read, потом углубляем retrieval"]
    assert read["recent_episodes"][0]["artifact_id"] == "episode-memory"
    assert related["topics"][0]["thread_id"] == "thr-ui"


def test_topic_summary_builder_persists_summary_and_links(tmp_path) -> None:
    _, artifact_store, _ = _build_stores(tmp_path)
    store = TopicStore(artifact_store)
    memory_thread = store.create_thread(
        TopicThread.create(
            visible_chat_id="chat-1",
            workspace_id="global",
            session_id="chat-1",
            topic_key="memory_architecture",
            title="Memory / continuity / hidden topics",
            tags=["memory", "continuity"],
        )
    )
    ui_thread = TopicThread.create(
        visible_chat_id="chat-1",
        workspace_id="global",
        session_id="chat-1",
        topic_key="browser_ui",
        title="Browser UI",
        tags=["ui", "browser", "memory"],
    )
    ui_thread.thread_id = "thr-ui"
    store.create_thread(ui_thread)

    governor = Governor(artifact_store)
    envelope = MemoryEnvelope(
        event_id="event-topic",
        text="Нужно собрать summaries и связи тем.",
        workspace_id="global",
        session_id="chat-1",
        metadata={
            "visible_chat_id": "chat-1",
            "topic_thread_id": memory_thread.thread_id,
            "topic_key": "memory_architecture",
            "topic_title": "Memory / continuity / hidden topics",
            "related_topic_thread_ids": ["thr-ui"],
        },
    )
    proposal = ArtifactProposal(
        artifact_type="task_state",
        text="Переделать retrieval внутри hidden topic threads",
        summary="Topic retrieval task",
        confidence=0.92,
        scope="task",
        decay="slow",
        retrieve_when=["memory", "continuity"],
        action="create",
        metadata={
            "task_status": "open",
            "open_questions": ["Как держать topic summaries стабильными?"],
            "decisions": ["Сначала делаем summaries без отдельной topic table"],
        },
    )
    decision = governor._create_artifact(proposal, envelope)
    builder = TopicSummaryBuilder(store)
    snapshot = builder.rebuild_thread(memory_thread.thread_id, workspace_id="global")

    assert decision.artifact is not None
    assert snapshot is not None
    artifact = artifact_store.get_by_id(decision.artifact.artifact_id)
    updated_thread = store.get_thread(memory_thread.thread_id)
    assert artifact is not None
    assert updated_thread is not None
    assert artifact.metadata["topic_thread_id"] == memory_thread.thread_id
    assert artifact.metadata["related_topic_thread_ids"] == ["thr-ui"]
    assert snapshot.open_questions == ["Как держать topic summaries стабильными?"]
    assert snapshot.current_decisions == ["Сначала делаем summaries без отдельной topic table"]
    assert "Focus:" in updated_thread.summary
    assert "Open questions:" in updated_thread.summary
    assert updated_thread.related_thread_ids[0] == "thr-ui"

    task_links = store.list_links(src_artifact_id=artifact.artifact_id, link_type="task_in_topic")
    question_links = store.list_links(src_artifact_id=artifact.artifact_id, link_type="question_in_topic")
    decision_links = store.list_links(src_artifact_id=artifact.artifact_id, link_type="decision_in_topic")
    related_links = store.list_links(src_artifact_id=memory_thread.thread_id, link_type="related_topic")
    assert task_links[0]["dst_artifact_id"] == memory_thread.thread_id
    assert question_links[0]["metadata"]["items"] == ["Как держать topic summaries стабильными?"]
    assert decision_links[0]["metadata"]["items"] == ["Сначала делаем summaries без отдельной topic table"]
    assert related_links[0]["dst_artifact_id"] == "thr-ui"


def test_background_worker_refreshes_topic_summary_after_job(tmp_path) -> None:
    db, artifact_store, event_store = _build_stores(tmp_path)
    job_queue = JobQueueStore(db)
    store = TopicStore(artifact_store)
    thread = store.create_thread(
        TopicThread.create(
            visible_chat_id="chat-1",
            workspace_id="global",
            session_id="chat-1",
            topic_key="memory_architecture",
            title="Memory / continuity / hidden topics",
            tags=["memory"],
        )
    )

    envelope = MemoryEnvelope(
        event_id="event-worker-topic",
        source_kind="assistant",
        payload_type="message",
        text="Сначала делаем stable topic summaries, потом topic graph.",
        workspace_id="global",
        session_id="chat-1",
        metadata={
            "visible_chat_id": "chat-1",
            "topic_thread_id": thread.thread_id,
            "topic_key": "memory_architecture",
            "topic_title": "Memory / continuity / hidden topics",
        },
    )
    event_store.append(envelope)
    job_queue.enqueue(
        event_id=envelope.event_id,
        job_type=job_queue.TYPE_MEMORY_LLM_PROCESS,
        payload={},
        priority=5,
    )

    def _memory_llm(_envelope):
        return MemoryLLMResult(
            event_id=_envelope.event_id,
            importance=0.9,
            should_process=True,
            proposals=[
                ArtifactProposal(
                    artifact_type="task_state",
                    text="Собрать stable topic summaries",
                    summary="Topic summary task",
                    confidence=0.9,
                    scope="task",
                    decay="slow",
                    retrieve_when=["memory", "topics"],
                    action="create",
                    metadata={
                        "task_status": "open",
                        "open_questions": ["Как ограничить шум между related topics?"],
                        "decisions": ["Summary builder запускаем после governor"],
                    },
                )
            ],
        )

    worker = BackgroundWorker(
        job_queue=job_queue,
        event_store=event_store,
        memory_llm_processor=_memory_llm,
        governor=Governor(artifact_store).decide,
        artifact_store=artifact_store,
    )

    handled = worker.process_one_job()
    updated_thread = store.get_thread(thread.thread_id)
    task_links = store.list_links(link_type="task_in_topic", dst_artifact_id=thread.thread_id)

    assert handled is True
    assert updated_thread is not None
    assert "Summary builder запускаем после governor" in list(updated_thread.metadata.get("current_decisions") or [])
    assert "Как ограничить шум между related topics?" in list(updated_thread.metadata.get("open_questions") or [])
    assert task_links


def test_memory_retrieve_stage_injects_topic_blocks_into_prompt_context(tmp_path) -> None:
    _, artifact_store, _ = _build_stores(tmp_path)
    store = TopicStore(artifact_store)
    thread = store.create_thread(
        TopicThread.create(
            visible_chat_id="chat-1",
            workspace_id="global",
            session_id="chat-1",
            topic_key="memory_architecture",
            title="Memory / continuity / hidden topics",
            tags=["memory", "continuity"],
            related_thread_ids=["thr-ui"],
        )
    )
    store.touch_thread(
        thread.thread_id,
        summary="Focus: stable topic summaries. Decisions: summary builder runs after governor. Open questions: how to keep related topics quiet?",
        metadata={
            "open_questions": ["How to keep related topics quiet?"],
            "current_decisions": ["Summary builder runs after governor"],
        },
    )

    class _FakeMemoryCore:
        def query(self, **kwargs):
            return {
                "context_blocks": [],
                "hits": [],
                "citations": [],
                "blocks": {},
                "selected": [],
                "dropped": [],
                "open_questions": [],
                "current_decisions": [],
            }

    ctx = PipelineContext(
        route="chat",
        user_msg="напомни что у нас по hidden topics",
        state={"conversation_id": "chat-1", "topic_thread_id": thread.thread_id},
        meta={"conversation_id": "chat-1", "topic_store": store},
        retrieved_memories=[],
        traits={},
        policies={},
        profile="BALANCED",
        clean_user_msg="напомни что у нас по hidden topics",
    )

    ctx = MemoryRetrieveStage(memory_manager=_FakeMemoryCore()).run(ctx)
    block = PromptEngine._build_memory_retrieval_block(
        blocks={},
        memory_blocks=dict(ctx.memory_context.get("blocks") or {}),
        state_map={},
    )

    assert "Summary builder runs after governor" in list(ctx.memory_context.get("current_decisions") or [])
    assert "How to keep related topics quiet?" in list(ctx.memory_context.get("open_questions") or [])
    assert "[CURRENT_TOPIC]" in block
    assert "[TOPIC_OPEN_QUESTIONS]" in block
    assert "[TOPIC_DECISIONS]" in block


def test_agent_loop_exposes_and_executes_topic_tools(tmp_path) -> None:
    _, artifact_store, _ = _build_stores(tmp_path)
    store = TopicStore(artifact_store)
    thread = store.create_thread(
        TopicThread.create(
            visible_chat_id="chat-1",
            workspace_id="global",
            session_id="chat-1",
            topic_key="memory_architecture",
            title="Memory / continuity / hidden topics",
            tags=["memory", "continuity"],
        )
    )
    artifact_store.create(
        MemoryArtifact(
            artifact_id="task-memory",
            artifact_type="task_state",
            source_event_id="event-1",
            text="Переделать continuity retrieval",
            summary="Continuity retrieval task",
            metadata={
                "topic_thread_id": thread.thread_id,
                "task_status": "open",
                "open_questions": ["Как хранить related topics?"],
                "decisions": ["Связи тем держим в artifact_links"],
            },
            workspace_id="global",
        )
    )

    memory_core = SimpleNamespace(service=SimpleNamespace(artifact_store=artifact_store))
    pipeline = ResponsePipeline(provider=object(), memory_manager=memory_core)
    ctx = PipelineContext(
        route="chat",
        user_msg="Открой тему про memory continuity",
        state={
            "conversation_id": "chat-1",
            "topic_thread_id": thread.thread_id,
        },
        meta={
            "conversation_id": "chat-1",
            "memory_core": memory_core,
            "memory_manager": memory_core,
        },
        retrieved_memories=[],
        traits={},
        policies={},
        profile="AUTONOMOUS",
        clean_user_msg="Открой тему про memory continuity",
    )

    tool_names = {tool.name for tool in _agent_loop_tools(ctx)}
    assert {"topic_read", "topic_search", "topic_related"} <= tool_names

    generate_stage = pipeline._stages["generate"]
    search_payload, search_error = generate_stage._execute_agent_tool_call(
        ctx,
        ToolCall(id="topic-search-1", name="topic_search", arguments={"query": "memory continuity", "limit": 3}),
    )
    read_payload, read_error = generate_stage._execute_agent_tool_call(
        ctx,
        ToolCall(id="topic-read-1", name="topic_read", arguments={"thread_id": thread.thread_id, "limit": 4}),
    )
    related_payload, related_error = generate_stage._execute_agent_tool_call(
        ctx,
        ToolCall(id="topic-related-1", name="topic_related", arguments={"thread_id": thread.thread_id, "limit": 4}),
    )

    search = json.loads(search_payload)
    read = json.loads(read_payload)
    related = json.loads(related_payload)

    assert search_error is False
    assert read_error is False
    assert related_error is False
    assert search["result"]["topics"][0]["thread_id"] == thread.thread_id
    assert read["result"]["thread"]["thread_id"] == thread.thread_id
    assert read["result"]["open_questions"] == ["Как хранить related topics?"]
    assert read["result"]["current_decisions"] == ["Связи тем держим в artifact_links"]
    assert related["result"]["thread_id"] == thread.thread_id
