"""
Тесты на исправление Governor update/merge/supersede.

См. tasks/fix_governor_step1.md
"""

import pytest
from memory_core.governor.governor import Governor
from memory_core.processors.memory_llm_processor import ArtifactProposal
from memory_core.schemas import MemoryArtifact, MemoryEnvelope
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.storage.sqlite_db import Database


def test_governor_update_updates_existing_artifact(tmp_path):
    """
    Тест: Governor._update_artifact() обновляет существующий артефакт,
    а не создаёт дубль.
    """
    db = Database(str(tmp_path / "test.db"))
    store = ArtifactStore(db)
    governor = Governor(store)

    envelope = MemoryEnvelope(text="Я работаю в VS Code", workspace_id="global", namespace="default")

    existing = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id=envelope.event_id,
        text="Пользователь работает в VS Code",
        summary="VS Code",
        metadata={"confidence": 0.9},
        namespace=envelope.namespace,
        workspace_id=envelope.workspace_id,
    )
    store.create(existing)

    proposal = ArtifactProposal(
        artifact_type="profile_fact",
        text="Пользователь работает в VS Code и Python",
        summary="VS Code + Python",
        confidence=0.7,
        scope="profile",
        decay="none",
        retrieve_when=["coding"],
        action="update",
    )

    decision = governor._update_artifact(proposal, existing, envelope)

    assert decision.action == "update"

    row = store.get_by_id(existing.artifact_id)
    assert row is not None
    assert row.text == "Пользователь работает в VS Code и Python"
    assert row.summary == "VS Code + Python"
    assert row.metadata["last_governor_action"] == "update"
    assert row.metadata["last_source_event_id"] == envelope.event_id

    # Проверяем, что только одна запись
    artifacts = store.list_artifacts(limit=10)
    assert len(artifacts) == 1


def test_governor_merge_updates_existing_artifact_in_place(tmp_path):
    """
    Тест: Governor._merge_artifacts() обновляет существующий артефакт,
    а не создаёт дубль.
    """
    db = Database(str(tmp_path / "test.db"))
    store = ArtifactStore(db)
    governor = Governor(store)

    envelope = MemoryEnvelope(text="Я пишу на Python", workspace_id="global", namespace="default")

    existing = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id=envelope.event_id,
        text="Пользователь пишет код",
        summary="Кодинг",
        metadata={"confidence": 0.6},
        namespace=envelope.namespace,
        workspace_id=envelope.workspace_id,
    )
    store.create(existing)

    proposal = ArtifactProposal(
        artifact_type="profile_fact",
        text="Пользователь пишет код на Python",
        summary="Python coding",
        confidence=0.65,
        scope="profile",
        decay="none",
        retrieve_when=["coding", "python"],
        action="merge",
    )

    decision = governor._merge_artifacts(proposal, existing, envelope)

    assert decision.action == "merge"

    row = store.get_by_id(existing.artifact_id)
    assert row is not None
    assert "Python" in row.text
    assert row.metadata["last_governor_action"] == "merge"
    assert row.metadata["merge_count"] >= 1
    assert row.metadata["last_source_event_id"] == envelope.event_id

    # Проверяем, что только одна запись
    artifacts = store.list_artifacts(limit=10)
    assert len(artifacts) == 1


def test_governor_supersede_marks_old_and_creates_new(tmp_path):
    """
    Тест: Governor._supersede_artifacts() помечает старые как superseded
    и создаёт новый артефакт.
    """
    db = Database(str(tmp_path / "test.db"))
    store = ArtifactStore(db)
    governor = Governor(store)

    envelope = MemoryEnvelope(text="Теперь пользователь работает в PyCharm", workspace_id="global", namespace="default")

    old_1 = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id="old-event-1",
        text="Пользователь работает в VS Code",
        summary="VS Code",
        metadata={"confidence": 0.5},
        namespace=envelope.namespace,
        workspace_id=envelope.workspace_id,
    )
    old_2 = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id="old-event-2",
        text="Пользователь в основном использует VS Code",
        summary="VS Code main",
        metadata={"confidence": 0.55},
        namespace=envelope.namespace,
        workspace_id=envelope.workspace_id,
    )
    store.create(old_1)
    store.create(old_2)

    proposal = ArtifactProposal(
        artifact_type="profile_fact",
        text="Пользователь теперь работает в PyCharm",
        summary="PyCharm",
        confidence=0.9,
        scope="profile",
        decay="none",
        retrieve_when=["coding", "ide"],
        action="supersede",
    )

    decision = governor._supersede_artifacts(proposal, [old_1, old_2], envelope)

    assert decision.action == "supersede"
    assert len(decision.superseded_artifact_ids) == 2

    updated_old_1 = store.get_by_id(old_1.artifact_id)
    updated_old_2 = store.get_by_id(old_2.artifact_id)
    new_artifact = store.get_by_id(decision.artifact.artifact_id)

    # Проверяем старые артефакты
    assert updated_old_1.status == "superseded"
    assert updated_old_2.status == "superseded"
    assert updated_old_1.metadata["superseded_by"] == new_artifact.artifact_id
    assert updated_old_2.metadata["superseded_by"] == new_artifact.artifact_id
    assert updated_old_1.metadata["last_governor_action"] == "supersede_old"
    assert updated_old_2.metadata["last_governor_action"] == "supersede_old"

    # Проверяем новый артефакт
    assert new_artifact is not None
    assert new_artifact.status == "active"
    assert new_artifact.metadata["supersedes"] == [old_1.artifact_id, old_2.artifact_id]
    assert new_artifact.metadata["last_governor_action"] == "supersede_new"

    # Проверяем, что всего 3 записи (2 старых + 1 новый)
    artifacts = store.list_artifacts(limit=10)
    assert len(artifacts) == 3


def test_governor_update_does_not_raise_unique_constraint(tmp_path):
    """
    Тест: Governor._update_artifact() не вызывает UNIQUE constraint failed.
    """
    db = Database(str(tmp_path / "test.db"))
    store = ArtifactStore(db)
    governor = Governor(store)

    envelope = MemoryEnvelope(text="Тест", workspace_id="global", namespace="default")

    # Создаём артефакт
    existing = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id=envelope.event_id,
        text="Тестовый текст",
        summary="Тест",
        metadata={"confidence": 0.5},
        namespace=envelope.namespace,
        workspace_id=envelope.workspace_id,
    )
    store.create(existing)

    proposal = ArtifactProposal(
        artifact_type="profile_fact",
        text="Обновлённый текст",
        summary="Обновление",
        confidence=0.6,
        scope="profile",
        decay="none",
        retrieve_when=["test"],
        action="update",
    )

    # Не должно падать с UNIQUE constraint failed
    decision = governor._update_artifact(proposal, existing, envelope)
    assert decision.action == "update"


def test_governor_merge_does_not_raise_unique_constraint(tmp_path):
    """
    Тест: Governor._merge_artifacts() не вызывает UNIQUE constraint failed.
    """
    db = Database(str(tmp_path / "test.db"))
    store = ArtifactStore(db)
    governor = Governor(store)

    envelope = MemoryEnvelope(text="Тест", workspace_id="global", namespace="default")

    # Создаём артефакт
    existing = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id=envelope.event_id,
        text="Тестовый текст",
        summary="Тест",
        metadata={"confidence": 0.5},
        namespace=envelope.namespace,
        workspace_id=envelope.workspace_id,
    )
    store.create(existing)

    proposal = ArtifactProposal(
        artifact_type="profile_fact",
        text="Дополнение к тексту",
        summary="Мердж",
        confidence=0.55,
        scope="profile",
        decay="none",
        retrieve_when=["test"],
        action="merge",
    )

    # Не должно падать с UNIQUE constraint failed
    decision = governor._merge_artifacts(proposal, existing, envelope)
    assert decision.action == "merge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
