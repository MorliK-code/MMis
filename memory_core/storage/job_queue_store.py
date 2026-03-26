"""
JobQueueStore - хранилище очереди задач для асинхронной обработки памяти.

Управляет таблицей ingest_jobs:
- queued: стоит в очереди
- processing: обрабатывается (с lease/lock)
- retry_wait: ошибка, ждёт повтор
- done: успешно
- dead: окончательно сломалось
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from memory_core.storage.sqlite_db import Database


@dataclass(slots=True)
class IngestJob:
    """
    Задача обработки памяти.
    """
    job_id: str
    event_id: str
    job_type: str  # 'memory_llm_process', 'reindex', 'backfill'
    status: str  # queued | processing | retry_wait | done | dead
    priority: int = 5  # 1 = highest, 9 = lowest
    attempts: int = 0
    max_attempts: int = 3
    locked_by: str | None = None
    locked_at: float | None = None
    available_at: float = 0.0
    error_text: str | None = None
    payload_json: str = "{}"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "job_id": self.job_id,
            "event_id": self.event_id,
            "job_type": self.job_type,
            "status": self.status,
            "priority": self.priority,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "locked_by": self.locked_by,
            "locked_at": self.locked_at,
            "available_at": self.available_at,
            "error_text": self.error_text,
            "payload_json": self.payload_json,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "IngestJob":
        """Создаёт из SQLite строки."""
        return cls(
            job_id=row["job_id"],
            event_id=row["event_id"],
            job_type=row["job_type"],
            status=row["status"],
            priority=row["priority"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            locked_by=row["locked_by"],
            locked_at=row["locked_at"],
            available_at=row["available_at"],
            error_text=row["error_text"],
            payload_json=row["payload_json"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def payload(self) -> dict[str, Any]:
        """Возвращает распарсенный payload."""
        return json.loads(self.payload_json) if self.payload_json else {}


class JobQueueStore:
    """
    Хранилище очереди задач.

    Поддерживает:
    - Добавление задач в очередь
    - Получение следующей доступной задачи (с lock)
    - Обновление статуса
    - Retry с backoff
    - Очистку зависших задач (lease timeout)
    """

    # Статусы задач
    STATUS_QUEUED = "queued"
    STATUS_PROCESSING = "processing"
    STATUS_RETRY_WAIT = "retry_wait"
    STATUS_DONE = "done"
    STATUS_DEAD = "dead"

    # Типы задач
    TYPE_MEMORY_LLM_PROCESS = "memory_llm_process"
    TYPE_REINDEX = "reindex"
    TYPE_BACKFILL = "backfill"

    # Lease timeout в секундах (2 минуты)
    LEASE_TIMEOUT_SEC = 120.0

    # Backoff между попытками в секундах
    RETRY_BACKOFF_SEC = 30.0

    def __init__(self, db: Database):
        """
        Инициализирует хранилище.

        Args:
            db: Экземпляр Database.
        """
        self.db = db

    def enqueue(
        self,
        event_id: str,
        job_type: str,
        payload: dict[str, Any],
        priority: int = 5,
        max_attempts: int = 3,
    ) -> str:
        """
        Добавляет задачу в очередь.

        Args:
            event_id: ID события.
            job_type: Тип задачи.
            payload: Данные задачи.
            priority: Приоритет (1=highest, 9=lowest).
            max_attempts: Максимальное количество попыток.

        Returns:
            job_id созданной задачи.
        """
        job_id = str(uuid.uuid4())
        now = time.time()
        payload_json = json.dumps(payload)

        self.db.execute(
            """
            INSERT INTO ingest_jobs (
                job_id, event_id, job_type, status, priority,
                attempts, max_attempts, available_at, payload_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                event_id,
                job_type,
                self.STATUS_QUEUED,
                priority,
                0,
                max_attempts,
                now,  # available сразу
                payload_json,
                now,
                now,
            ),
        )

        return job_id

    def dequeue(
        self,
        worker_id: str,
        job_type: str | None = None,
        max_priority: int = 9,
    ) -> IngestJob | None:
        """
        Получает следующую доступную задачу и блокирует её.

        Args:
            worker_id: ID воркера (для lock).
            job_type: Фильтр по типу задачи (None = любой).
            max_priority: Максимальный приоритет (по умолчанию 9).

        Returns:
            IngestJob или None, если нет доступных задач.
        """
        now = time.time()

        # Сначала освобождаем зависшие задачи (lease timeout)
        self._release_expired_leases(now)

        # Формируем запрос
        base_query = """
            SELECT * FROM ingest_jobs
            WHERE status = ? AND available_at <= ? AND priority <= ?
        """
        params: list[Any] = [self.STATUS_QUEUED, now, max_priority]

        if job_type:
            base_query += " AND job_type = ?"
            params.append(job_type)

        base_query += " ORDER BY priority ASC, available_at ASC LIMIT 1"

        row = self.db.fetchone(base_query, tuple(params))
        if not row:
            return None

        # Блокируем задачу (acquire lock)
        job = IngestJob.from_row(row)
        self.db.execute(
            """
            UPDATE ingest_jobs
            SET status = ?, locked_by = ?, locked_at = ?, updated_at = ?, error_text = NULL
            WHERE job_id = ? AND (locked_by IS NULL OR locked_at IS NULL OR locked_at <= ?)
            """,
            (
                self.STATUS_PROCESSING,
                worker_id,
                now,
                now,
                job.job_id,
                now - self.LEASE_TIMEOUT_SEC,
            ),
        )

        # Проверяем, удалось ли заблокировать
        updated_row = self.db.fetchone(
            "SELECT * FROM ingest_jobs WHERE job_id = ?",
            (job.job_id,),
        )
        if not updated_row or updated_row["status"] != self.STATUS_PROCESSING:
            # Не удалось заблокировать, другой воркер опередил
            return None

        return IngestJob.from_row(updated_row)

    def complete(self, job_id: str) -> None:
        """
        Помечает задачу как выполненную.

        Args:
            job_id: ID задачи.
        """
        now = time.time()
        self.db.execute(
            """
            UPDATE ingest_jobs
            SET status = ?, updated_at = ?, error_text = NULL,
                locked_by = NULL, locked_at = NULL
            WHERE job_id = ?
            """,
            (self.STATUS_DONE, now, job_id),
        )

    def fail(
        self,
        job_id: str,
        error_text: str,
        retry: bool = True,
    ) -> bool:
        """
        Помечает задачу как неудачную.

        Args:
            job_id: ID задачи.
            error_text: Текст ошибки.
            retry: Попытаться ли повторить.

        Returns:
            True если задача будет повторена, False если dead.
        """
        now = time.time()

        row = self.db.fetchone(
            "SELECT attempts, max_attempts FROM ingest_jobs WHERE job_id = ?",
            (job_id,),
        )
        if not row:
            return False

        attempts = row["attempts"] + 1
        max_attempts = row["max_attempts"]

        if retry and attempts < max_attempts:
            # Планируем retry с backoff
            available_at = now + (self.RETRY_BACKOFF_SEC * attempts)  # экспоненциальный backoff
            self.db.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, attempts = ?, error_text = ?,
                    available_at = ?, updated_at = ?,
                    locked_by = NULL, locked_at = NULL
                WHERE job_id = ?
                """,
                (
                    self.STATUS_RETRY_WAIT,
                    attempts,
                    error_text,
                    available_at,
                    now,
                    job_id,
                ),
            )
            # Сразу возвращаем в queued
            self.db.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, available_at = ?
                WHERE job_id = ?
                """,
                (self.STATUS_QUEUED, available_at, job_id),
            )
            return True
        else:
            # Мёртвая задача
            self.db.execute(
                """
                UPDATE ingest_jobs
                SET status = ?, attempts = ?, error_text = ?,
                    updated_at = ?, locked_by = NULL, locked_at = NULL
                WHERE job_id = ?
                """,
                (self.STATUS_DEAD, attempts, error_text, now, job_id),
            )
            return False

    def requeue_immediately(self, job_id: str, error_text: str = "") -> bool:
        """
        Немедленно возвращает задачу в queued без retry backoff.
        """
        now = time.time()
        cursor = self.db.execute(
            """
            UPDATE ingest_jobs
            SET status = ?, error_text = ?, available_at = ?, updated_at = ?,
                locked_by = NULL, locked_at = NULL
            WHERE job_id = ?
            """,
            (
                self.STATUS_QUEUED,
                error_text,
                now,
                now,
                job_id,
            ),
        )
        return bool(cursor.rowcount)

    def _release_expired_leases(self, now: float) -> int:
        """
        Освобождает задачи с истёкшим lease.

        Args:
            now: Текущее время.

        Returns:
            Количество освобождённых задач.
        """
        cursor = self.db.execute(
            """
            UPDATE ingest_jobs
            SET status = ?, locked_by = NULL, locked_at = NULL, updated_at = ?
            WHERE status = ? AND locked_at <= ?
            """,
            (
                self.STATUS_QUEUED,
                now,
                self.STATUS_PROCESSING,
                now - self.LEASE_TIMEOUT_SEC,
            ),
        )
        return cursor.rowcount

    def get_stats(self) -> dict[str, Any]:
        """
        Получает статистику очереди.

        Returns:
            Статистика по задачам.
        """
        stats = {}

        # Количество по статусам
        row = self.db.fetchone(
            """
            SELECT
                SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) as queued,
                SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                SUM(CASE WHEN status = 'retry_wait' THEN 1 ELSE 0 END) as retry_wait,
                SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as done,
                SUM(CASE WHEN status = 'dead' THEN 1 ELSE 0 END) as dead
            FROM ingest_jobs
            """
        )
        if row:
            stats["queued"] = row["queued"] or 0
            stats["processing"] = row["processing"] or 0
            stats["retry_wait"] = row["retry_wait"] or 0
            stats["done"] = row["done"] or 0
            stats["dead"] = row["dead"] or 0

        # Количество по типам
        rows = self.db.fetchall(
            """
            SELECT job_type, COUNT(*) as count
            FROM ingest_jobs
            WHERE status IN ('queued', 'processing', 'retry_wait')
            GROUP BY job_type
            """
        )
        stats["by_type"] = {row["job_type"]: row["count"] for row in rows}

        return stats

    def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
    ) -> list[IngestJob]:
        """
        Получает список задач.

        Args:
            status: Фильтр по статусу.
            job_type: Фильтр по типу.
            limit: Максимальное количество.

        Returns:
            Список IngestJob.
        """
        query = "SELECT * FROM ingest_jobs WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if job_type:
            query += " AND job_type = ?"
            params.append(job_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.db.fetchall(query, tuple(params))
        return [IngestJob.from_row(row) for row in rows]

    def cleanup_done(self, older_than_sec: float = 3600.0) -> int:
        """
        Очищает выполненные задачи.

        Args:
            older_than_sec: Удалять задачи старше этого времени (в секундах).

        Returns:
            Количество удалённых задач.
        """
        now = time.time()
        cutoff = now - older_than_sec

        cursor = self.db.execute(
            "DELETE FROM ingest_jobs WHERE status = ? AND updated_at <= ?",
            (self.STATUS_DONE, cutoff),
        )
        return cursor.rowcount
