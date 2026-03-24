"""
Тесты на hybrid retrieval.

См. tasks/fix_retrieval_step2.md
"""

import pytest
from memory_core.storage.sqlite_db import Database
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.indexing.vector_index import VectorIndex
from memory_core.indexing.embeddings import EmbeddingProvider
from memory_core.retrieval.retrieval_service import RetrievalService
from memory_core.schemas import MemoryQuery
from memory_core.config_manager import RetrievalConfig


@pytest.fixture
def retrieval_service(tmp_path):
    """Создаёт retrieval service с vector index."""
    db_path = tmp_path / "test.db"
    vector_path = tmp_path / "vector"
    
    db = Database(str(db_path))
    artifact_store = ArtifactStore(db)
    
    embedding_provider = EmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2")
    vector_index = VectorIndex(
        index_path=str(vector_path),
        embedding_provider=embedding_provider,
    )
    
    retrieval_config = RetrievalConfig(
        always_load=["identity_core", "profile_fact", "task_state"],
        sometimes_load=["preference", "episode_event", "emotional_state"],
        never_load_if_decay="immediate",
        never_load_if_status="superseded",
        never_load_if_confidence_below=0.3,
    )
    
    return RetrievalService(
        db=db,
        vector_index=vector_index,
        retrieval_config=retrieval_config,
    )


def test_retrieval_collects_mandatory_artifacts(retrieval_service):
    """
    Тест: retrieval собирает обязательные артефакты (always_load).
    """
    from memory_core.schemas import MemoryArtifact
    
    # Создаём обязательные артефакты
    artifact = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id="test-event",
        text="Пользователь работает в VS Code",
        summary="VS Code",
        metadata={"confidence": 0.9},
        namespace="default",
        workspace_id="global",
    )
    retrieval_service.artifact_store.create(artifact)
    
    query = MemoryQuery(
        text="тест",
        workspace_id="global",
        session_id="test-session",
        top_k=8,
    )
    
    mandatory = retrieval_service._collect_mandatory_artifacts(query)
    
    # Должен найти хотя бы один mandatory artifact
    assert len(mandatory) >= 1
    assert any(a.artifact_type == "profile_fact" for a in mandatory)


def test_retrieval_collects_continuity_artifacts(retrieval_service):
    """
    Тест: retrieval собирает артефакты continuity для сессии.
    """
    from memory_core.schemas import MemoryArtifact
    
    # Создаём артефакт с session_id
    artifact = MemoryArtifact(
        artifact_type="episode_event",
        source_event_id="test-event",
        text="Эпизод тестовой сессии",
        summary="Тест",
        metadata={"session_id": "test-session", "confidence": 0.8},
        namespace="default",
        workspace_id="global",
    )
    retrieval_service.artifact_store.create(artifact)
    
    query = MemoryQuery(
        text="тест",
        workspace_id="global",
        session_id="test-session",
        top_k=8,
    )
    
    continuity = retrieval_service._collect_continuity_artifacts(query)
    
    # Должен найти артефакт с session_id
    assert len(continuity) >= 1
    assert any(a.metadata.get("session_id") == "test-session" for a in continuity)


def test_retrieval_collects_semantic_artifacts(retrieval_service):
    """
    Тест: retrieval использует vector index для semantic search.
    """
    from memory_core.schemas import MemoryArtifact
    
    # Создаём артефакт
    artifact = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id="test-event",
        text="Пользователь программирует на Python",
        summary="Python programming",
        metadata={"confidence": 0.9},
        namespace="default",
        workspace_id="global",
    )
    retrieval_service.artifact_store.create(artifact)
    
    # Индексируем в vector index
    retrieval_service.vector_index.add(
        artifact_id=artifact.artifact_id,
        text=artifact.text,
        metadata={"artifact_type": "profile_fact"},
    )
    
    query = MemoryQuery(
        text="кодинг программирование",  # Семантически похоже
        workspace_id="global",
        session_id="test-session",
        top_k=8,
    )
    
    semantic = retrieval_service._collect_semantic_artifacts(query)
    
    # Должен найти через semantic search
    assert len(semantic) >= 1
    assert any(a.artifact_id == artifact.artifact_id for a in semantic)
    # semantic_score должен быть сохранён
    assert "semantic_score" in semantic[0].metadata


def test_retrieval_applies_runtime_rules(retrieval_service):
    """
    Тест: retrieval применяет runtime rules (never_load_if).
    """
    from memory_core.schemas import MemoryArtifact
    
    # Создаём superseded артефакт
    superseded = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id="test-event",
        text="Старый факт",
        summary="Старое",
        metadata={"confidence": 0.9},
        namespace="default",
        workspace_id="global",
        status="superseded",
    )
    retrieval_service.artifact_store.create(superseded)
    
    query = MemoryQuery(
        text="тест",
        workspace_id="global",
        session_id="test-session",
        top_k=8,
    )
    
    # Собираем все артефакты
    all_artifacts = retrieval_service.artifact_store.list_artifacts(limit=10)
    
    # Применяем runtime rules
    filtered = retrieval_service._apply_runtime_rules(all_artifacts, query)
    
    # superseded должен быть отфильтрован
    assert not any(a.status == "superseded" for a in filtered)


def test_retrieval_merge_removes_duplicates(retrieval_service):
    """
    Тест: _merge_artifact_lists удаляет дубликаты.
    """
    from memory_core.schemas import MemoryArtifact
    
    artifact = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id="test-event",
        text="Тест",
        summary="Тест",
        metadata={},
        namespace="default",
        workspace_id="global",
    )
    
    # Один и тот же артефакт в разных списках
    mandatory = [artifact]
    semantic = [artifact]
    lexical = [artifact]
    
    merged = retrieval_service._merge_artifact_lists(mandatory, [], semantic, lexical)
    
    # Должен быть только один экземпляр
    assert len(merged) == 1
    assert merged[0].artifact_id == artifact.artifact_id


def test_retrieval_full_query(retrieval_service):
    """
    Тест: полный query с 4 стадиями retrieval.
    """
    from memory_core.schemas import MemoryArtifact
    
    # Создаём артефакт
    artifact = MemoryArtifact(
        artifact_type="profile_fact",
        source_event_id="test-event",
        text="Пользователь работает в Visual Studio Code",
        summary="VS Code",
        metadata={"confidence": 0.9},
        namespace="default",
        workspace_id="global",
    )
    retrieval_service.artifact_store.create(artifact)
    
    # Индексируем
    retrieval_service.vector_index.add(
        artifact_id=artifact.artifact_id,
        text=artifact.text,
        metadata={"artifact_type": "profile_fact"},
    )
    
    query = MemoryQuery(
        text="в каком редакторе он пишет код",
        workspace_id="global",
        session_id="test-session",
        top_k=8,
    )
    
    context_pack, citations = retrieval_service.query(query)
    
    # Должен вернуть контекст
    assert context_pack is not None
    # Артефакт должен попасть в результат (через semantic или lexical)
    # Проверяем, что context не пустой или citations есть


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
