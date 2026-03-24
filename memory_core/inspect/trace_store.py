"""
Memory Trace Store — хранилище для persistent trace.

Сохраняет turn-level trace в JSONL формат для последующей инспекции.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.logger import get_logger


LOGGER = get_logger(__name__)


class MemoryTraceStore:
    """
    Хранилище trace для Memory Inspector.

    Пишет в JSONL формат:
    - data/logs/memory_trace/trace.jsonl
    
    Каждый row содержит:
    - trace_id
    - request_id
    - conversation_id
    - turn_id
    - pipeline trace (retrieval, governor, persona, etc.)
    - memory events/artifacts/jobs
    - links to web_trace
    """

    def __init__(self, trace_dir: str | None = None):
        """
        Инициализирует trace store.

        Args:
            trace_dir: Директория для trace файлов.
        """
        if trace_dir is None:
            trace_dir = "data/logs/memory_trace"
        
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        # Основной JSONL файл
        self.trace_file = self.trace_dir / "trace.jsonl"
        
        # Кэш для быстрого поиска по trace_id
        self._trace_index: dict[str, int] = {}  # trace_id -> line_number
        self._load_index()

    def _load_index(self) -> None:
        """Загружает индекс trace_id -> line_number."""
        if not self.trace_file.exists():
            return
        
        try:
            with self.trace_file.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        trace_id = row.get("trace_id", "")
                        if trace_id:
                            self._trace_index[trace_id] = line_num
                    except Exception:
                        continue
        except Exception as e:
            LOGGER.error(f"Failed to load trace index: {e}")

    def append_turn_trace(self, row: dict[str, Any]) -> None:
        """
        Добавляет turn trace.

        Args:
            row: Trace row с pipeline/memory данными.
        """
        # Гарантируем наличие обязательных полей
        if not row.get("trace_id"):
            row["trace_id"] = f"trace_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        if not row.get("created_at"):
            row["created_at"] = datetime.now().isoformat()
        
        # Пишем в JSONL
        try:
            with self.trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            
            # Обновляем индекс
            trace_id = row["trace_id"]
            with self.trace_file.open("r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            self._trace_index[trace_id] = line_count - 1
            
            LOGGER.debug(f"Appended turn trace: {trace_id}")
        except Exception as e:
            LOGGER.error(f"Failed to append turn trace: {e}")

    def append_worker_trace(self, trace_id: str, row: dict[str, Any]) -> None:
        """
        Добавляет worker trace к существующему turn trace.

        Args:
            trace_id: ID turn trace.
            row: Worker trace row.
        """
        # Для worker trace используем отдельный файл
        worker_trace_file = self.trace_dir / f"{trace_id}_worker.jsonl"
        
        try:
            if not row.get("created_at"):
                row["created_at"] = datetime.now().isoformat()
            
            with worker_trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            
            LOGGER.debug(f"Appended worker trace: {trace_id}")
        except Exception as e:
            LOGGER.error(f"Failed to append worker trace: {e}")

    def list_traces(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Получает последние trace.

        Args:
            limit: Максимальное количество.

        Returns:
            Список trace rows.
        """
        if not self.trace_file.exists():
            return []
        
        result = []
        try:
            with self.trace_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Читаем с конца (последние сначала)
            for line in reversed(lines[-limit:]):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    result.append(row)
                except Exception:
                    continue
        except Exception as e:
            LOGGER.error(f"Failed to list traces: {e}")
        
        return result

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """
        Получает trace по ID.

        Args:
            trace_id: ID trace.

        Returns:
            Trace row или None.
        """
        if trace_id not in self._trace_index:
            return None
        
        line_num = self._trace_index[trace_id]
        
        try:
            with self.trace_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == line_num:
                        line = line.strip()
                        if not line:
                            return None
                        return json.loads(line)
        except Exception as e:
            LOGGER.error(f"Failed to get trace: {e}")
        
        return None

    def get_worker_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """
        Получает worker trace для turn.

        Args:
            trace_id: ID turn trace.

        Returns:
            Список worker trace rows.
        """
        worker_trace_file = self.trace_dir / f"{trace_id}_worker.jsonl"
        
        if not worker_trace_file.exists():
            return []
        
        result = []
        try:
            with worker_trace_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as e:
            LOGGER.error(f"Failed to get worker trace: {e}")
        
        return result

    def find_by_event_id(self, event_id: str) -> dict[str, Any] | None:
        """
        Ищет trace по event_id.

        Args:
            event_id: ID события.

        Returns:
            Trace row или None.
        """
        try:
            with self.trace_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        memory = row.get("memory", {})
                        event_ids = memory.get("event_ids", [])
                        if event_id in event_ids:
                            return row
                    except Exception:
                        continue
        except Exception as e:
            LOGGER.error(f"Failed to find by event_id: {e}")
        
        return None

    def find_by_request_id(self, request_id: str) -> dict[str, Any] | None:
        """
        Ищет trace по request_id.

        Args:
            request_id: ID запроса.

        Returns:
            Trace row или None.
        """
        try:
            with self.trace_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if row.get("request_id") == request_id:
                            return row
                    except Exception:
                        continue
        except Exception as e:
            LOGGER.error(f"Failed to find by request_id: {e}")
        
        return None

    def clear(self) -> None:
        """Очищает все trace."""
        if self.trace_file.exists():
            self.trace_file.unlink()
        self._trace_index.clear()
        LOGGER.info("Cleared trace store")
