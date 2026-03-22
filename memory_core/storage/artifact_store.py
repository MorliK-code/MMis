"""
Artifact Store - хранилище нормализованных артефактов.
"""

import json
import sqlite3
import time
from typing import Any
from memory_core.storage.sqlite_db import Database
from memory_core.schemas import MemoryArtifact
from memory_core.errors import ArtifactStoreError


class ArtifactStore:
    """
    Хранилище артефактов памяти.
    
    Артефакты создаются процессорами из сырых событий.
    """
    
    def __init__(self, db: Database):
        """
        Инициализирует Artifact Store.
        
        Args:
            db: Экземпляр Database.
        """
        self.db = db
    
    def create(self, artifact: MemoryArtifact) -> None:
        """
        Создаёт артефакт.
        
        Args:
            artifact: Артефакт для создания.
        """
        sql = """
            INSERT INTO artifacts (
                artifact_id, artifact_type, source_event_id, text, summary,
                metadata_json, namespace, workspace_id, status, 
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            artifact.artifact_id,
            artifact.artifact_type,
            artifact.source_event_id,
            artifact.text,
            artifact.summary,
            json.dumps(artifact.metadata),
            artifact.namespace,
            artifact.workspace_id,
            artifact.status,
            artifact.created_at,
            artifact.updated_at,
        )
        try:
            self.db.execute(sql, params)
        except Exception as e:
            raise ArtifactStoreError(f"Failed to create artifact: {e}")
    
    def create_many(self, artifacts: list[MemoryArtifact]) -> None:
        """
        Создаёт несколько артефактов за раз.
        
        Args:
            artifacts: Список артефактов.
        """
        sql = """
            INSERT INTO artifacts (
                artifact_id, artifact_type, source_event_id, text, summary,
                metadata_json, namespace, workspace_id, status, 
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params_list = [
            (
                a.artifact_id,
                a.artifact_type,
                a.source_event_id,
                a.text,
                a.summary,
                json.dumps(a.metadata),
                a.namespace,
                a.workspace_id,
                a.status,
                a.created_at,
                a.updated_at,
            )
            for a in artifacts
        ]
        try:
            self.db.executemany(sql, params_list)
        except Exception as e:
            raise ArtifactStoreError(f"Failed to create artifacts: {e}")
    
    def get_by_id(self, artifact_id: str) -> MemoryArtifact | None:
        """
        Получает артефакт по ID.
        
        Args:
            artifact_id: ID артефакта.
            
        Returns:
            MemoryArtifact или None, если не найдено.
        """
        sql = "SELECT * FROM artifacts WHERE artifact_id = ?"
        row = self.db.fetchone(sql, (artifact_id,))
        
        if row is None:
            return None
        
        return self._row_to_artifact(row)
    
    def get_by_source_event(self, source_event_id: str) -> list[MemoryArtifact]:
        """
        Получает артефакты по ID исходного события.
        
        Args:
            source_event_id: ID исходного события.
            
        Returns:
            Список MemoryArtifact.
        """
        sql = "SELECT * FROM artifacts WHERE source_event_id = ? ORDER BY created_at ASC"
        rows = self.db.fetchall(sql, (source_event_id,))
        return [self._row_to_artifact(row) for row in rows]
    
    def list_artifacts(
        self,
        artifact_type: str | None = None,
        workspace_id: str | None = None,
        status: str | None = None,
        namespace: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryArtifact]:
        """
        Получает список артефактов с фильтрацией.
        
        Args:
            artifact_type: Фильтр по типу артефакта.
            workspace_id: Фильтр по workspace.
            status: Фильтр по статусу.
            namespace: Фильтр по namespace.
            limit: Максимальное количество результатов.
            offset: Смещение.
            
        Returns:
            Список MemoryArtifact.
        """
        conditions = []
        params: list[Any] = []
        
        if artifact_type:
            conditions.append("artifact_type = ?")
            params.append(artifact_type)
        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        sql = f"""
            SELECT * FROM artifacts 
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        rows = self.db.fetchall(sql, tuple(params))
        return [self._row_to_artifact(row) for row in rows]
    
    def update_status(self, artifact_id: str, status: str) -> None:
        """
        Обновляет статус артефакта.
        
        Args:
            artifact_id: ID артефакта.
            status: Новый статус.
        """
        sql = """
            UPDATE artifacts 
            SET status = ?, updated_at = ?
            WHERE artifact_id = ?
        """
        params = (status, time.time(), artifact_id)
        try:
            self.db.execute(sql, params)
        except Exception as e:
            raise ArtifactStoreError(f"Failed to update artifact status: {e}")
    
    def update(self, artifact: MemoryArtifact) -> None:
        """
        Обновляет артефакт.
        
        Args:
            artifact: Артефакт с обновлёнными данными.
        """
        sql = """
            UPDATE artifacts 
            SET text = ?, summary = ?, metadata_json = ?, 
                status = ?, updated_at = ?
            WHERE artifact_id = ?
        """
        params = (
            artifact.text,
            artifact.summary,
            json.dumps(artifact.metadata),
            artifact.status,
            time.time(),
            artifact.artifact_id,
        )
        try:
            self.db.execute(sql, params)
        except Exception as e:
            raise ArtifactStoreError(f"Failed to update artifact: {e}")
    
    def search_by_text(
        self,
        query: str,
        workspace_id: str | None = None,
        artifact_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryArtifact]:
        """
        Полнотекстовый поиск по артефактам.
        
        Args:
            query: Поисковый запрос.
            workspace_id: Фильтр по workspace.
            artifact_types: Фильтр по типам артефактов.
            limit: Максимальное количество результатов.
            
        Returns:
            Список MemoryArtifact.
        """
        conditions = []
        params: list[Any] = []
        
        # Поиск по тексту (простой LIKE поиск)
        conditions.append("(text LIKE ? OR summary LIKE ?)")
        like_query = f"%{query}%"
        params.extend([like_query, like_query])
        
        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        
        if artifact_types:
            placeholders = ", ".join("?" for _ in artifact_types)
            conditions.append(f"artifact_type IN ({placeholders})")
            params.extend(artifact_types)
        
        # Только активные артефакты
        conditions.append("status = ?")
        params.append("active")
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        sql = f"""
            SELECT * FROM artifacts 
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        
        rows = self.db.fetchall(sql, tuple(params))
        return [self._row_to_artifact(row) for row in rows]
    
    def count(self, workspace_id: str | None = None) -> int:
        """
        Подсчитывает количество артефактов.
        
        Args:
            workspace_id: Фильтр по workspace.
            
        Returns:
            Количество артефактов.
        """
        if workspace_id:
            sql = "SELECT COUNT(*) FROM artifacts WHERE workspace_id = ?"
            row = self.db.fetchone(sql, (workspace_id,))
        else:
            sql = "SELECT COUNT(*) FROM artifacts"
            row = self.db.fetchone(sql)
        
        return row[0] if row else 0
    
    def _row_to_artifact(self, row: sqlite3.Row) -> MemoryArtifact:
        """Преобразует строку БД в MemoryArtifact."""
        return MemoryArtifact(
            artifact_id=row["artifact_id"],
            artifact_type=row["artifact_type"],
            source_event_id=row["source_event_id"],
            text=row["text"],
            summary=row["summary"],
            metadata=json.loads(row["metadata_json"]),
            namespace=row["namespace"],
            workspace_id=row["workspace_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
