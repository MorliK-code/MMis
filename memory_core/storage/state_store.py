"""
State Store - хранилище краткоживущего состояния.
"""

import json
import time
from typing import Any
from memory_core.storage.sqlite_db import Database
from memory_core.errors import StateStoreError


class StateStore:
    """
    Хранилище краткоживущего состояния.
    
    Здесь хранится только текущее, быстро меняющееся состояние:
    - активная задача
    - текущая тема
    - незакрытая ветка
    - текущий workspace
    - short-lived plan
    
    НЕ для долгосрочной памяти!
    """
    
    def __init__(self, db: Database):
        """
        Инициализирует State Store.
        
        Args:
            db: Экземпляр Database.
        """
        self.db = db
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Устанавливает значение состояния.
        
        Args:
            key: Ключ состояния.
            value: Значение (любой JSON-сериализуемый объект).
        """
        sql = """
            INSERT OR REPLACE INTO runtime_state (
                state_key, state_value_json, updated_at
            ) VALUES (?, ?, ?)
        """
        params = (key, json.dumps(value), time.time())
        try:
            self.db.execute(sql, params)
        except Exception as e:
            raise StateStoreError(f"Failed to set state: {e}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Получает значение состояния.
        
        Args:
            key: Ключ состояния.
            default: Значение по умолчанию.
            
        Returns:
            Значение состояния или default.
        """
        sql = "SELECT state_value_json FROM runtime_state WHERE state_key = ?"
        row = self.db.fetchone(sql, (key,))
        
        if row is None:
            return default
        
        return json.loads(row["state_value_json"])
    
    def delete_state(self, key: str) -> None:
        """
        Удаляет состояние.
        
        Args:
            key: Ключ состояния.
        """
        sql = "DELETE FROM runtime_state WHERE state_key = ?"
        try:
            self.db.execute(sql, (key,))
        except Exception as e:
            raise StateStoreError(f"Failed to delete state: {e}")
    
    def list_keys(self, prefix: str | None = None) -> list[str]:
        """
        Получает список ключей состояния.
        
        Args:
            prefix: Опциональный префикс для фильтрации.
            
        Returns:
            Список ключей.
        """
        if prefix:
            sql = "SELECT state_key FROM runtime_state WHERE state_key LIKE ? ORDER BY state_key"
            rows = self.db.fetchall(sql, (f"{prefix}%",))
        else:
            sql = "SELECT state_key FROM runtime_state ORDER BY state_key"
            rows = self.db.fetchall(sql)
        
        return [row["state_key"] for row in rows]
    
    def clear_all(self) -> None:
        """Очищает все состояния."""
        sql = "DELETE FROM runtime_state"
        try:
            self.db.execute(sql)
        except Exception as e:
            raise StateStoreError(f"Failed to clear all states: {e}")
    
    # === Специализированные методы для часто используемых состояний ===
    
    def set_current_workspace(self, workspace_id: str) -> None:
        """Устанавливает текущий workspace."""
        self.set_state("current_workspace", workspace_id)
    
    def get_current_workspace(self, default: str = "global") -> str:
        """Получает текущий workspace."""
        return self.get_state("current_workspace", default)
    
    def set_active_task(self, task: dict[str, Any]) -> None:
        """Устанавливает активную задачу."""
        self.set_state("active_task", task)
    
    def get_active_task(self, default: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Получает активную задачу."""
        return self.get_state("active_task", default)
    
    def clear_active_task(self) -> None:
        """Очищает активную задачу."""
        self.delete_state("active_task")
    
    def set_current_topic(self, topic: str) -> None:
        """Устанавливает текущую тему."""
        self.set_state("current_topic", topic)
    
    def get_current_topic(self, default: str = "") -> str:
        """Получает текущую тему."""
        return self.get_state("current_topic", default)
    
    def set_session_data(self, session_id: str, data: dict[str, Any]) -> None:
        """Устанавливает данные сессии."""
        self.set_state(f"session:{session_id}", data)
    
    def get_session_data(self, session_id: str, default: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Получает данные сессии."""
        return self.get_state(f"session:{session_id}", default)
    
    def delete_session_data(self, session_id: str) -> None:
        """Удаляет данные сессии."""
        self.delete_state(f"session:{session_id}")
