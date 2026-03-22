"""
Event Store - хранилище сырых событий (канонический источник истины).
"""

import json
import sqlite3
from typing import Any
from memory_core.storage.sqlite_db import Database
from memory_core.schemas import MemoryEnvelope
from memory_core.errors import EventStoreError


class EventStore:
    """
    Хранилище сырых событий.
    
    Raw events - это канонический источник истины.
    Все события записываются и никогда не удаляются (только архивируются).
    """
    
    def __init__(self, db: Database):
        """
        Инициализирует Event Store.
        
        Args:
            db: Экземпляр Database.
        """
        self.db = db
    
    def append(self, envelope: MemoryEnvelope) -> None:
        """
        Добавляет событие в хранилище.
        
        Args:
            envelope: Конверт события.
        """
        sql = """
            INSERT INTO events (
                event_id, source_kind, payload_type, text, 
                metadata_json, namespace, workspace_id, session_id, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            envelope.event_id,
            envelope.source_kind,
            envelope.payload_type,
            envelope.text,
            json.dumps(envelope.metadata),
            envelope.namespace,
            envelope.workspace_id,
            envelope.session_id,
            envelope.ts,
        )
        try:
            self.db.execute(sql, params)
        except Exception as e:
            raise EventStoreError(f"Failed to append event: {e}")
    
    def get_by_id(self, event_id: str) -> MemoryEnvelope | None:
        """
        Получает событие по ID.
        
        Args:
            event_id: ID события.
            
        Returns:
            MemoryEnvelope или None, если не найдено.
        """
        sql = "SELECT * FROM events WHERE event_id = ?"
        row = self.db.fetchone(sql, (event_id,))
        
        if row is None:
            return None
        
        return self._row_to_envelope(row)
    
    def list_events(
        self,
        workspace_id: str | None = None,
        session_id: str | None = None,
        source_kind: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryEnvelope]:
        """
        Получает список событий с фильтрацией.
        
        Args:
            workspace_id: Фильтр по workspace.
            session_id: Фильтр по session.
            source_kind: Фильтр по типу источника.
            limit: Максимальное количество результатов.
            offset: Смещение.
            
        Returns:
            Список MemoryEnvelope.
        """
        conditions = []
        params: list[Any] = []
        
        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if source_kind:
            conditions.append("source_kind = ?")
            params.append(source_kind)
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        sql = f"""
            SELECT * FROM events 
            {where_clause}
            ORDER BY ts DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        rows = self.db.fetchall(sql, tuple(params))
        return [self._row_to_envelope(row) for row in rows]
    
    def list_by_session(self, session_id: str, limit: int = 100) -> list[MemoryEnvelope]:
        """
        Получает события по session_id.
        
        Args:
            session_id: ID сессии.
            limit: Максимальное количество результатов.
            
        Returns:
            Список MemoryEnvelope.
        """
        return self.list_events(session_id=session_id, limit=limit)
    
    def list_by_workspace(self, workspace_id: str, limit: int = 100) -> list[MemoryEnvelope]:
        """
        Получает события по workspace_id.
        
        Args:
            workspace_id: ID workspace.
            limit: Максимальное количество результатов.
            
        Returns:
            Список MemoryEnvelope.
        """
        return self.list_events(workspace_id=workspace_id, limit=limit)
    
    def count(self, workspace_id: str | None = None) -> int:
        """
        Подсчитывает количество событий.
        
        Args:
            workspace_id: Фильтр по workspace.
            
        Returns:
            Количество событий.
        """
        if workspace_id:
            sql = "SELECT COUNT(*) FROM events WHERE workspace_id = ?"
            row = self.db.fetchone(sql, (workspace_id,))
        else:
            sql = "SELECT COUNT(*) FROM events"
            row = self.db.fetchone(sql)
        
        return row[0] if row else 0
    
    def _row_to_envelope(self, row: sqlite3.Row) -> MemoryEnvelope:
        """Преобразует строку БД в MemoryEnvelope."""
        return MemoryEnvelope(
            event_id=row["event_id"],
            source_kind=row["source_kind"],
            payload_type=row["payload_type"],
            text=row["text"],
            metadata=json.loads(row["metadata_json"]),
            namespace=row["namespace"],
            workspace_id=row["workspace_id"],
            session_id=row["session_id"],
            ts=row["ts"],
        )
