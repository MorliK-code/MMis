"""
Workspace Store - хранилище workspace и документов.
"""

import json
import time
from typing import Any
from memory_core.storage.sqlite_db import Database
from memory_core.errors import WorkspaceStoreError


class WorkspaceStore:
    """
    Хранилище workspace и источников документов.
    
    Workspace - это скоуп для группировки памяти по проектам/задачам.
    """
    
    def __init__(self, db: Database):
        """
        Инициализирует Workspace Store.
        
        Args:
            db: Экземпляр Database.
        """
        self.db = db
    
    def create_workspace(
        self,
        workspace_id: str,
        title: str,
        goal: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Создаёт workspace.
        
        Args:
            workspace_id: ID workspace.
            title: Заголовок.
            goal: Цель workspace.
            metadata: Дополнительные метаданные.
        """
        sql = """
            INSERT INTO workspaces (
                workspace_id, title, goal, metadata_json, 
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        now = time.time()
        params = (
            workspace_id,
            title,
            goal,
            json.dumps(metadata or {}),
            now,
            now,
        )
        try:
            self.db.execute(sql, params)
        except Exception as e:
            raise WorkspaceStoreError(f"Failed to create workspace: {e}")
    
    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        """
        Получает workspace по ID.
        
        Args:
            workspace_id: ID workspace.
            
        Returns:
            Словарь с данными workspace или None.
        """
        sql = "SELECT * FROM workspaces WHERE workspace_id = ?"
        row = self.db.fetchone(sql, (workspace_id,))
        
        if row is None:
            return None
        
        return {
            "workspace_id": row["workspace_id"],
            "title": row["title"],
            "goal": row["goal"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    
    def list_workspaces(self) -> list[dict[str, Any]]:
        """
        Получает список всех workspace.
        
        Returns:
            Список словарей с данными workspace.
        """
        sql = "SELECT * FROM workspaces ORDER BY updated_at DESC"
        rows = self.db.fetchall(sql)
        
        return [
            {
                "workspace_id": row["workspace_id"],
                "title": row["title"],
                "goal": row["goal"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
    
    def update_workspace(
        self,
        workspace_id: str,
        title: str | None = None,
        goal: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Обновляет workspace.
        
        Args:
            workspace_id: ID workspace.
            title: Новый заголовок.
            goal: Новая цель.
            metadata: Новые метаданные.
        """
        updates = []
        params: list[Any] = []
        
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if goal is not None:
            updates.append("goal = ?")
            params.append(goal)
        if metadata is not None:
            updates.append("metadata_json = ?")
            params.append(json.dumps(metadata))
        
        if not updates:
            return
        
        updates.append("updated_at = ?")
        params.append(time.time())
        params.append(workspace_id)
        
        sql = f"""
            UPDATE workspaces 
            SET {", ".join(updates)}
            WHERE workspace_id = ?
        """
        try:
            self.db.execute(sql, params)
        except Exception as e:
            raise WorkspaceStoreError(f"Failed to update workspace: {e}")
    
    def add_source(
        self,
        source_id: str,
        workspace_id: str,
        source_kind: str,
        title: str,
        content_ref: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Добавляет источник в workspace.
        
        Args:
            source_id: ID источника.
            workspace_id: ID workspace.
            source_kind: Тип источника (file | url | text | code).
            title: Заголовок.
            content_ref: Ссылка на контент (путь к файлу, URL, или ID).
            metadata: Дополнительные метаданные.
        """
        sql = """
            INSERT INTO workspace_sources (
                source_id, workspace_id, source_kind, title, 
                content_ref, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            source_id,
            workspace_id,
            source_kind,
            title,
            content_ref,
            json.dumps(metadata or {}),
            time.time(),
        )
        try:
            self.db.execute(sql, params)
        except Exception as e:
            raise WorkspaceStoreError(f"Failed to add source: {e}")
    
    def get_source(self, source_id: str) -> dict[str, Any] | None:
        """
        Получает источник по ID.
        
        Args:
            source_id: ID источника.
            
        Returns:
            Словарь с данными источника или None.
        """
        sql = "SELECT * FROM workspace_sources WHERE source_id = ?"
        row = self.db.fetchone(sql, (source_id,))
        
        if row is None:
            return None
        
        return {
            "source_id": row["source_id"],
            "workspace_id": row["workspace_id"],
            "source_kind": row["source_kind"],
            "title": row["title"],
            "content_ref": row["content_ref"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }
    
    def list_sources(
        self,
        workspace_id: str | None = None,
        source_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Получает список источников.
        
        Args:
            workspace_id: Фильтр по workspace.
            source_kind: Фильтр по типу источника.
            
        Returns:
            Список словарей с данными источников.
        """
        conditions = []
        params: list[Any] = []
        
        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        if source_kind:
            conditions.append("source_kind = ?")
            params.append(source_kind)
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        sql = f"""
            SELECT * FROM workspace_sources 
            {where_clause}
            ORDER BY created_at DESC
        """
        rows = self.db.fetchall(sql, tuple(params))
        
        return [
            {
                "source_id": row["source_id"],
                "workspace_id": row["workspace_id"],
                "source_kind": row["source_kind"],
                "title": row["title"],
                "content_ref": row["content_ref"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    
    def delete_source(self, source_id: str) -> None:
        """
        Удаляет источник.
        
        Args:
            source_id: ID источника.
        """
        sql = "DELETE FROM workspace_sources WHERE source_id = ?"
        try:
            self.db.execute(sql, (source_id,))
        except Exception as e:
            raise WorkspaceStoreError(f"Failed to delete source: {e}")
    
    def workspace_exists(self, workspace_id: str) -> bool:
        """
        Проверяет существование workspace.
        
        Args:
            workspace_id: ID workspace.
            
        Returns:
            True, если workspace существует.
        """
        sql = "SELECT 1 FROM workspaces WHERE workspace_id = ? LIMIT 1"
        row = self.db.fetchone(sql, (workspace_id,))
        return row is not None
