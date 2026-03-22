"""
Memory Trace Builder - построение трассировки.
"""

from typing import Any
from memory_core.storage.sqlite_db import Database
from memory_core.storage.event_store import EventStore
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.schemas import MemoryTrace


class MemoryTraceBuilder:
    """
    Построитель трассировки памяти.
    
    Строит полную трассировку:
    event -> artifacts -> indexed -> retrieved
    """
    
    def __init__(self, db: Database):
        """
        Инициализирует построитель трассировки.
        
        Args:
            db: Экземпляр Database.
        """
        self.db = db
        self.event_store = EventStore(db)
        self.artifact_store = ArtifactStore(db)
    
    def build_trace(self, event_id: str) -> MemoryTrace:
        """
        Строит полную трассировку события.
        
        Args:
            event_id: ID события.
            
        Returns:
            MemoryTrace.
        """
        # Получаем событие
        event = self.event_store.get_by_id(event_id)
        event_dict = event.to_dict() if event else None
        
        if not event:
            return MemoryTrace(
                event=None,
                artifacts=[],
                indexed=False,
                retrieved=False,
            )
        
        # Получаем артефакты
        artifacts = self.artifact_store.get_by_source_event(event_id)
        artifact_dicts = [a.to_dict() for a in artifacts]
        
        # Проверяем индексацию
        indexed = len(artifacts) > 0
        
        # Для retrieved нужно проверять отдельно
        # (был ли этот event использован в retrieval)
        retrieved = self._check_if_retrieved(event_id)
        
        return MemoryTrace(
            event=event_dict,
            artifacts=artifact_dicts,
            indexed=indexed,
            retrieved=retrieved,
        )
    
    def _check_if_retrieved(self, event_id: str) -> bool:
        """
        Проверяет, было ли событие использовано в retrieval.
        
        Args:
            event_id: ID события.
            
        Returns:
            True, если событие было использовано.
        """
        # Пока простая проверка: если есть артефакты, считаем что использовано
        # В будущем можно вести журнал retrieval запросов
        artifacts = self.artifact_store.get_by_source_event(event_id)
        return len(artifacts) > 0
    
    def build_full_trace(
        self,
        event_id: str,
        include_links: bool = True,
    ) -> dict[str, Any]:
        """
        Строит полную трассировку с деталями.
        
        Args:
            event_id: ID события.
            include_links: Включать ли связи.
            
        Returns:
            Словарь с полной трассировкой.
        """
        trace = self.build_trace(event_id)
        
        result = {
            "event": trace.event,
            "artifacts": trace.artifacts,
            "indexed": trace.indexed,
            "retrieved": trace.retrieved,
        }
        
        if include_links:
            # Добавляем связи между артефактами
            result["links"] = self._get_artifact_links(event_id)
        
        return result
    
    def _get_artifact_links(self, event_id: str) -> list[dict[str, Any]]:
        """
        Получает связи между артефактами события.
        
        Args:
            event_id: ID события.
            
        Returns:
            Список связей.
        """
        # Получаем артефакты
        artifacts = self.artifact_store.get_by_source_event(event_id)
        artifact_ids = [a.artifact_id for a in artifacts]
        
        # Пока возвращаем пустой список
        # В будущем можно загружать из artifact_links таблицы
        return []
