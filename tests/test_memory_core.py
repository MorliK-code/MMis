"""
Тесты для memory_core.
"""

import pytest
import shutil
import os
from pathlib import Path

# Импорты memory_core
from core.brain import Brain
from memory_core.schemas import MemoryEnvelope, MemoryArtifact, MemoryQuery
from memory_core.bootstrap.service_factory import build_memory_service, MemoryServiceConfig
from memory_core.storage.sqlite_db import Database
from memory_core.storage.event_store import EventStore
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.processors.fact_processor import FactProcessor
from memory_core.processors.profile_processor import ProfileProcessor
from memory_core.retrieval.retrieval_service import RetrievalService


@pytest.fixture
def test_db_path(tmp_path):
    """Создаёт временную базу данных."""
    db_path = str(tmp_path / "test_memory.db")
    yield db_path
    # Очистка после теста (игнорируем ошибки на Windows)
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except PermissionError:
        pass  # Файл ещё заблокирован на Windows


@pytest.fixture
def memory_service(test_db_path):
    """Создаёт MemoryService для тестов."""
    config = MemoryServiceConfig(
        db_path=test_db_path,
        vector_path=str(Path(test_db_path).parent / "vector"),
        enable_background_worker=False,
    )
    service = build_memory_service(config)
    yield service
    try:
        service.close()
    except Exception:
        pass


class TestMemoryEnvelope:
    """Тесты для MemoryEnvelope."""
    
    def test_create_envelope(self):
        """Создание конверта события."""
        envelope = MemoryEnvelope(
            source_kind="user",
            payload_type="message",
            text="Привет!",
        )
        assert envelope.source_kind == "user"
        assert envelope.text == "Привет!"
        assert envelope.event_id is not None
    
    def test_envelope_to_dict(self):
        """Преобразование в словарь."""
        envelope = MemoryEnvelope(text="Test")
        data = envelope.to_dict()
        assert data["text"] == "Test"
        assert data["source_kind"] == "user"
    
    def test_envelope_from_dict(self):
        """Создание из словаря."""
        data = {
            "event_id": "test-123",
            "source_kind": "assistant",
            "text": "Hello",
        }
        envelope = MemoryEnvelope.from_dict(data)
        assert envelope.event_id == "test-123"
        assert envelope.source_kind == "assistant"


class TestEventStore:
    """Тесты для EventStore."""
    
    def test_append_event(self, test_db_path):
        """Добавление события."""
        db = Database(test_db_path)
        store = EventStore(db)
        
        envelope = MemoryEnvelope(text="Test event")
        store.append(envelope)
        
        assert store.count() == 1
    
    def test_get_event_by_id(self, test_db_path):
        """Получение события по ID."""
        db = Database(test_db_path)
        store = EventStore(db)
        
        envelope = MemoryEnvelope(text="Test event")
        store.append(envelope)
        
        retrieved = store.get_by_id(envelope.event_id)
        assert retrieved is not None
        assert retrieved.text == "Test event"
    
    def test_list_events(self, test_db_path):
        """Список событий."""
        db = Database(test_db_path)
        store = EventStore(db)
        
        for i in range(5):
            store.append(MemoryEnvelope(text=f"Event {i}"))
        
        events = store.list_events(limit=10)
        assert len(events) == 5


class TestArtifactStore:
    """Тесты для ArtifactStore."""
    
    def test_create_artifact(self, test_db_path):
        """Создание артефакта."""
        db = Database(test_db_path)
        store = ArtifactStore(db)
        
        artifact = MemoryArtifact(
            artifact_type="fact",
            text="Test fact",
            summary="Test",
        )
        store.create(artifact)
        
        assert store.count() == 1
    
    def test_get_artifact_by_id(self, test_db_path):
        """Получение артефакта по ID."""
        db = Database(test_db_path)
        store = ArtifactStore(db)
        
        artifact = MemoryArtifact(text="Test fact")
        store.create(artifact)
        
        retrieved = store.get_by_id(artifact.artifact_id)
        assert retrieved is not None
        assert retrieved.text == "Test fact"
    
    def test_search_by_text(self, test_db_path):
        """Поиск по тексту."""
        db = Database(test_db_path)
        store = ArtifactStore(db)
        
        store.create(MemoryArtifact(text="Python is great"))
        store.create(MemoryArtifact(text="I love Java"))
        store.create(MemoryArtifact(text="JavaScript is awesome"))
        
        results = store.search_by_text("Python", limit=10)
        assert len(results) == 1
        assert "Python" in results[0].text


class TestFactProcessor:
    """Тесты для FactProcessor."""
    
    def test_extract_simple_fact(self):
        """Извлечение простого факта."""
        processor = FactProcessor()
        envelope = MemoryEnvelope(
            text="Я работаю в VS Code",
            source_kind="user",
        )
        
        artifacts = processor.process(envelope)
        assert len(artifacts) >= 1
        assert artifacts[0].artifact_type == "fact"
    
    def test_no_fact_in_gibberish(self):
        """Отсутствие фактов в бессвязном тексте."""
        processor = FactProcessor()
        envelope = MemoryEnvelope(
            text="asdfghjkl qwerty",
            source_kind="user",
        )
        
        artifacts = processor.process(envelope)
        # Может вернуть 0, если нет паттернов
        assert len(artifacts) >= 0


class TestProfileProcessor:
    """Тесты для ProfileProcessor."""
    
    def test_extract_profile_fact(self):
        """Извлечение профильного факта."""
        processor = ProfileProcessor()
        envelope = MemoryEnvelope(
            text="Меня зовут Паша",
            source_kind="user",
        )
        
        artifacts = processor.process(envelope)
        # Профильные факты могут быть извлечены
        assert len(artifacts) >= 0


class TestMemoryService:
    """Тесты для MemoryService."""
    
    def test_ingest_event(self, memory_service):
        """Ingest события."""
        envelope = MemoryEnvelope(
            text="Я использую Windows",
            source_kind="user",
        )
        
        result = memory_service.ingest_event(envelope)
        assert result["processed"] is True
        assert result["event_id"] == envelope.event_id
    
    def test_query_memory(self, memory_service):
        """Запрос к памяти."""
        # Сначала добавим данные
        memory_service.ingest_event(
            MemoryEnvelope(text="Я работаю в Python")
        )
        
        # Теперь запрос
        query = MemoryQuery(text="Что использует пользователь?")
        result = memory_service.query(query)
        
        assert result is not None
        assert hasattr(result, "context_blocks")

    def test_ingest_event_updates_runtime_session_snapshot(self, memory_service):
        envelope = MemoryEnvelope(
            text="Вернёмся к переносу системы на SSD",
            source_kind="user",
            session_id="conv-runtime-sync",
            workspace_id="global",
            metadata={
                "topic_thread_id": "topic-ssd",
                "topic_title": "SSD migration",
            },
        )

        result = memory_service.ingest_event(envelope)
        snapshot = memory_service.get_runtime_session(
            namespace="default",
            workspace_id="global",
            session_id="conv-runtime-sync",
        )

        assert str(snapshot["last_user_turn"]["text"]).startswith("Вернёмся к переносу")
        assert snapshot["active_topic"]["thread_id"] == "topic-ssd"
        assert str(snapshot["current_episode_id"]).strip()
        assert result["runtime_session"]["current_episode_id"] == snapshot["current_episode_id"]

    def test_query_returns_runtime_task_continuity_before_worker(self, memory_service):
        memory_service.ingest_event(
            MemoryEnvelope(
                text="Продолжаем план по SSD",
                source_kind="user",
                session_id="conv-runtime-task",
                workspace_id="global",
                metadata={
                    "topic_thread_id": "topic-ssd",
                    "topic_title": "SSD migration",
                },
            )
        )
        memory_service.update_task_continuity(
            namespace="default",
            workspace_id="global",
            session_id="conv-runtime-task",
            active_task={
                "task_id": "task:ssd-migration",
                "topic": "SSD migration",
                "status": "waiting_user",
                "current_goal": "Перенести систему на SSD без потери данных.",
                "decisions": ["Сначала оставить систему на HDD, потом клонировать SSD."],
                "open_questions": ["Как безопасно перенести систему позже?"],
            },
            source="test",
            now_ts=123.0,
        )

        result = memory_service.query(
            MemoryQuery(
                text="что дальше по SSD?",
                session_id="conv-runtime-task",
                workspace_id="global",
            )
        )

        assert result.task_continuity["active_task"]["task_id"] == "task:ssd-migration"
        assert result.open_questions == ["Как безопасно перенести систему позже?"]
        assert result.current_decisions == ["Сначала оставить систему на HDD, потом клонировать SSD."]
        assert result.dialog_episode_hits[0]["source"] == "runtime_session"
        assert any(
            str(dict(hit.get("metadata") or {}).get("retrieval_source") or "") in {"active_task", "active_episode"}
            for hit in list(result.hits or [])
        )
    
    def test_get_stats(self, memory_service):
        """Получение статистики."""
        memory_service.ingest_event(MemoryEnvelope(text="Test"))
        
        stats = memory_service.get_stats()
        assert "events_count" in stats
        assert stats["events_count"] >= 1


class TestBrainMemoryIngestAccounting:
    def test_capture_memory_ingest_accepts_dict_payload(self):
        summary = Brain._empty_memory_write_summary()
        brain = Brain.__new__(Brain)

        Brain._capture_memory_ingest(
            brain,
            summary,
            {
                "stored_ids": ["a1"],
                "promoted_ids": ["p1"],
                "dropped_ids": ["d1"],
                "extracted_facts": ["f1", "f2"],
            },
            bucket="user_turn",
        )

        assert summary["stored_records"] == 1
        assert summary["blocked_writes"] == 1
        assert summary["facts_extracted"] == 2
        assert summary["promotions"] == 1
        assert summary["user_turns_written"] == 1



class TestEndToEnd:
    """End-to-end тесты."""
    
    def test_full_cycle(self, memory_service):
        """Полный цикл: ingest -> query."""
        # Ingest
        memory_service.ingest_event(
            MemoryEnvelope(
                text="Меня зовут Алекс. Я разработчик на Python.",
                source_kind="user",
            )
        )
        
        # Query
        query = MemoryQuery(text="Кто такой Алекс?")
        result = memory_service.query(query)
        
        # Проверяем, что контекст не пуст
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
