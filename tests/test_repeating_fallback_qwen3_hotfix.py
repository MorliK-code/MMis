from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from core.brain import Brain
from core.response_pipeline import PipelineResult
from memory_core.identity.identity_core import IdentityCore
from memory_core.retrieval.persona_context_builder import PersonaContextBuilder
from memory_core.schemas import MemoryArtifact, MemoryEnvelope


class _FakePipeline:
    def run(self, **kwargs):
        return PipelineResult(text="Нормальный ответ", logs=["pipeline_ok"])


class _FakeStateManager:
    def __init__(self) -> None:
        self._state = {
            "conversation_id": "conv-hotfix",
            "active_character_id": "asya",
            "quality_profile": "BALANCED",
            "context_tags": {},
        }

    def snapshot(self):
        raw = dict(self._state)
        return SimpleNamespace(
            raw=raw,
            mode="chat",
            conversation_id=str(raw.get("conversation_id") or "conv-hotfix"),
            turn_id=1,
            quality_profile=str(raw.get("quality_profile") or "BALANCED"),
            active_character_id=str(raw.get("active_character_id") or "asya"),
            active_goal="",
            dialog_summary="",
            active_tasks=[],
            history=[],
            context_tags=dict(raw.get("context_tags") or {}),
            active_mode="chatting",
            mode_lock=False,
            mode_until=0.0,
            last_signals=[],
            last_actions=[],
            web_mode="off",
            thinking_enabled=False,
            output_format={},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def update_on_user_message(self, *_args, **_kwargs) -> None:
        return None

    def update_on_event(self, *_args, **_kwargs) -> None:
        return None

    def update_on_assistant_message(self, *_args, **_kwargs) -> None:
        return None

    def patch(self, values: dict) -> None:
        self._state.update(dict(values or {}))

    def get(self, key: str, default=None):
        return self._state.get(key, default)

    def get_meta(self, _character_id: str):
        return SimpleNamespace(llm_profile="")

    def set_mode(self, *_args, **_kwargs) -> None:
        return None

    def set_mode_lock(self, *_args, **_kwargs) -> None:
        return None

    def set_output_format(self, *_args, **_kwargs) -> None:
        return None

    def set_active_personality(self, *_args, **_kwargs) -> None:
        return None

    def set_active_character(self, *_args, **_kwargs) -> None:
        return None

    def set_last_tool_result(self, *_args, **_kwargs) -> None:
        return None

    def set_dialog_summary(self, text: str) -> None:
        self._state["dialog_summary"] = str(text or "")


class _ModernArtifactStore:
    def __init__(self, artifacts: list[MemoryArtifact] | None = None) -> None:
        self.rows = list(artifacts or [])
        self.created_ids: list[str] = []
        self.updated_ids: list[str] = []

    def list_artifacts(
        self,
        artifact_type: str | None = None,
        workspace_id: str | None = None,
        status: str | None = None,
        namespace: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryArtifact]:
        rows = list(self.rows)
        if artifact_type is not None:
            rows = [row for row in rows if row.artifact_type == artifact_type]
        if workspace_id is not None:
            rows = [row for row in rows if row.workspace_id == workspace_id]
        if status is not None:
            rows = [row for row in rows if row.status == status]
        if namespace is not None:
            rows = [row for row in rows if row.namespace == namespace]
        return rows[offset : offset + limit]

    def create(self, artifact: MemoryArtifact) -> None:
        self.rows.append(artifact)
        self.created_ids.append(str(artifact.artifact_id))

    def update(self, artifact: MemoryArtifact) -> None:
        self.updated_ids.append(str(artifact.artifact_id))


class _ModernEventStore:
    def __init__(self, events: list[MemoryEnvelope] | None = None) -> None:
        self.rows = list(events or [])

    def list_events(
        self,
        workspace_id: str | None = None,
        session_id: str | None = None,
        source_kind: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEnvelope]:
        rows = list(self.rows)
        if workspace_id is not None:
            rows = [row for row in rows if row.workspace_id == workspace_id]
        if session_id is not None:
            rows = [row for row in rows if row.session_id == session_id]
        if source_kind is not None:
            rows = [row for row in rows if row.source_kind == source_kind]
        return rows[offset : offset + limit]


def test_brain_preserves_generated_answer_when_persist_turns_fails() -> None:
    brain = Brain(
        provider=object(),
        state_manager=_FakeStateManager(),
        memory_core=SimpleNamespace(),
        response_pipeline=_FakePipeline(),
    )

    with patch.object(brain, "_persist_turns", side_effect=AttributeError("broken debug snapshot")):
        result = brain.handle_message(
            "привет",
            meta={"conversation_id": "conv-hotfix", "track_state": False, "store_turn": False},
        )

    assert result.text == "Нормальный ответ"
    assert result.status == "ok"
    assert any("persist_warning=AttributeError:broken debug snapshot" == row for row in list(result.logs or []))


def test_identity_core_uses_modern_artifact_store_api_for_create_and_update() -> None:
    store = _ModernArtifactStore()
    core = IdentityCore(store)

    created = core.set_user_name("Паша", workspace_id="asya")
    updated = core.set_user_name("Миша", workspace_id="asya")
    profile = core.get_profile("asya")

    assert created.artifact_id == updated.artifact_id
    assert len(store.created_ids) == 1
    assert len(store.updated_ids) == 1
    assert profile.user_name == "миша"


def test_persona_context_builder_uses_modern_store_api() -> None:
    artifact_store = _ModernArtifactStore(
        [
            MemoryArtifact(
                artifact_type="profile_fact",
                source_event_id="evt-profile",
                text="Пишет на Python",
                summary="python",
                workspace_id="asya",
                status="active",
            ),
            MemoryArtifact(
                artifact_type="preference",
                source_event_id="evt-pref",
                text="Любит короткие ответы",
                summary="short answers",
                workspace_id="asya",
                status="active",
            ),
            MemoryArtifact(
                artifact_type="task_state",
                source_event_id="evt-task",
                text="Чинит runtime баги",
                summary="task",
                workspace_id="asya",
                status="active",
            ),
            MemoryArtifact(
                artifact_type="episode_event",
                source_event_id="evt-episode",
                text="Диагностирует repeating fallback",
                summary="episode",
                workspace_id="asya",
                status="active",
                metadata={"session_id": "sess-1"},
            ),
            MemoryArtifact(
                artifact_type="emotional_state",
                source_event_id="evt-emotion",
                text="Немного раздражен",
                summary="emotion",
                workspace_id="asya",
                status="active",
                metadata={"session_id": "sess-1", "decay": "fast"},
            ),
        ]
    )
    event_store = _ModernEventStore(
        [
            MemoryEnvelope(
                event_id="evt-1",
                source_kind="user",
                payload_type="message",
                text="привет",
                workspace_id="asya",
                session_id="sess-1",
            )
        ]
    )

    snapshot = PersonaContextBuilder(artifact_store=artifact_store, event_store=event_store).build(
        workspace_id="asya",
        session_id="sess-1",
    )

    assert snapshot.user_profile_facts == ["Пишет на Python"]
    assert snapshot.user_preferences == ["Любит короткие ответы"]
    assert snapshot.current_task == "Чинит runtime баги"
    assert snapshot.current_episode == "Диагностирует repeating fallback"
    assert snapshot.emotional_state == "Немного раздражен"
    assert snapshot.recent_events == ["[user] привет"]
