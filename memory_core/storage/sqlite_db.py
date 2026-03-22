"""
SQLite база данных для memory_core.
"""

import sqlite3
import os
from pathlib import Path
from typing import Any


class Database:
    """
    Управление SQLite соединением для memory_core.
    
    - Открывает базу
    - Включает WAL режим
    - Создаёт таблицы
    - Предоставляет connection/cursor
    """
    
    def __init__(self, db_path: str):
        """
        Инициализирует базу данных.
        
        Args:
            db_path: Путь к файлу базы данных.
        """
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        
        # Убеждаемся, что директория существует
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self) -> None:
        """Инициализирует базу данных: соединение, WAL, таблицы."""
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None  # Autocommit
        )
        self._conn.row_factory = sqlite3.Row
        
        # Включаем WAL режим для лучшей производительности
        self._execute("PRAGMA journal_mode=WAL")
        
        # Включаем foreign keys
        self._execute("PRAGMA foreign_keys=ON")
        
        # Создаём таблицы
        self._create_tables()
    
    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Выполняет SQL запрос."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        return cursor
    
    def _executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Выполняет SQL запрос с несколькими наборами параметров."""
        assert self._conn is not None
        cursor = self._conn.cursor()
        cursor.executemany(sql, params_list)
        return cursor
    
    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Выполняет SELECT и возвращает все строки."""
        cursor = self._execute(sql, params)
        return cursor.fetchall()
    
    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Выполняет SELECT и возвращает одну строку."""
        cursor = self._execute(sql, params)
        return cursor.fetchone()
    
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Выполняет SQL запрос (INSERT, UPDATE, DELETE)."""
        return self._execute(sql, params)
    
    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Выполняет SQL запрос с несколькими наборами параметров."""
        return self._executemany(sql, params_list)
    
    def _create_tables(self) -> None:
        """Создаёт все таблицы."""
        self._create_events_table()
        self._create_artifacts_table()
        self._create_artifact_links_table()
        self._create_workspaces_table()
        self._create_workspace_sources_table()
        self._create_runtime_state_table()
    
    def _create_events_table(self) -> None:
        """Создаёт таблицу events."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                source_kind TEXT NOT NULL,
                payload_type TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                namespace TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        
        # Индексы для ускорения поиска
        self._execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_kind)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_events_workspace ON events(workspace_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
    
    def _create_artifacts_table(self) -> None:
        """Создаёт таблицу artifacts."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                text TEXT NOT NULL,
                summary TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                namespace TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        
        # Индексы
        self._execute("CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_artifacts_workspace ON artifacts(workspace_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_artifacts_source_event ON artifacts(source_event_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(status)")
    
    def _create_artifact_links_table(self) -> None:
        """Создаёт таблицу связей между артефактами."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS artifact_links (
                link_id TEXT PRIMARY KEY,
                src_artifact_id TEXT NOT NULL,
                dst_artifact_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        
        self._execute("CREATE INDEX IF NOT EXISTS idx_links_src ON artifact_links(src_artifact_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_links_dst ON artifact_links(dst_artifact_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_links_type ON artifact_links(link_type)")
    
    def _create_workspaces_table(self) -> None:
        """Создаёт таблицу workspaces."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                goal TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
    
    def _create_workspace_sources_table(self) -> None:
        """Создаёт таблицу workspace_sources."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS workspace_sources (
                source_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                title TEXT NOT NULL,
                content_ref TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        
        self._execute("CREATE INDEX IF NOT EXISTS idx_sources_workspace ON workspace_sources(workspace_id)")
    
    def _create_runtime_state_table(self) -> None:
        """Создаёт таблицу runtime_state."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS runtime_state (
                state_key TEXT PRIMARY KEY,
                state_value_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
    
    def close(self) -> None:
        """Закрывает соединение с базой данных."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    @property
    def connection(self) -> sqlite3.Connection:
        """Возвращает SQLite соединение."""
        assert self._conn is not None
        return self._conn
