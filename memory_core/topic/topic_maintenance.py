from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from memory_core.topic.topic_models import TopicThread
from memory_core.topic.topic_store import TopicStore
from memory_core.topic.topic_summary import TopicSummaryBuilder


@dataclass(slots=True)
class TopicMaintenanceResult:
    thread_id: str
    old_status: str
    new_status: str
    summary_rebuilt: bool
    summary_stale: bool
    artifact_count: int
    reason: str


class TopicMaintenanceService:
    SLEEP_AFTER_SEC = 7 * 24 * 60 * 60
    ARCHIVE_AFTER_SEC = 30 * 24 * 60 * 60
    TRIVIAL_DELETE_AFTER_SEC = 6 * 60 * 60

    def __init__(self, topic_store: TopicStore):
        self.topic_store = topic_store

    def maintain_thread(
        self,
        thread_id: str,
        *,
        workspace_id: str = "",
        allow_summary_rebuild: bool = True,
        trigger: str = "maintenance",
    ) -> TopicMaintenanceResult | None:
        thread = self.topic_store.get_thread(thread_id)
        if thread is None:
            return None

        summary_rebuilt = False
        if allow_summary_rebuild and TopicSummaryBuilder.needs_refresh(thread):
            source = "lazy_rebuild" if trigger.startswith("read") else "background_rebuild"
            snapshot = TopicSummaryBuilder(self.topic_store).rebuild_thread(
                thread.thread_id,
                workspace_id=str(workspace_id or thread.workspace_id or "").strip(),
                source=source,
                refresh_reason=trigger,
            )
            if snapshot is not None:
                summary_rebuilt = True
                thread = self.topic_store.get_thread(thread.thread_id) or thread

        artifacts = self.topic_store.list_thread_artifacts(
            thread.thread_id,
            workspace_id=str(workspace_id or thread.workspace_id or "").strip(),
            limit=200,
        )
        next_status, reason = self._resolve_status(thread, artifacts)
        metadata = self._build_metadata(thread, artifacts, next_status, reason)
        old_status = str(thread.status or "active").strip() or "active"

        if next_status != old_status or self._metadata_needs_update(thread, metadata):
            self.topic_store.touch_thread(
                thread.thread_id,
                status=next_status,
                metadata=metadata,
            )
            thread = self.topic_store.get_thread(thread.thread_id) or thread

        return TopicMaintenanceResult(
            thread_id=thread.thread_id,
            old_status=old_status,
            new_status=str(thread.status or next_status).strip() or next_status,
            summary_rebuilt=summary_rebuilt,
            summary_stale=bool(dict(thread.metadata or {}).get("summary_stale")),
            artifact_count=len(artifacts),
            reason=reason,
        )

    def maintain_scope(
        self,
        *,
        visible_chat_id: str = "",
        workspace_id: str = "",
        session_id: str = "",
        active_thread_id: str = "",
        limit: int = 32,
    ) -> list[TopicMaintenanceResult]:
        threads = self.topic_store.list_threads(
            visible_chat_id=str(visible_chat_id or "").strip(),
            workspace_id=str(workspace_id or "").strip(),
            session_id=str(session_id or "").strip(),
            include_hidden=True,
            limit=max(1, int(limit or 32)),
        )
        results: list[TopicMaintenanceResult] = []
        active_id = str(active_thread_id or "").strip()
        for thread in threads:
            if str(thread.thread_id or "").strip() == active_id:
                continue
            result = self.maintain_thread(
                thread.thread_id,
                workspace_id=str(workspace_id or thread.workspace_id or "").strip(),
                allow_summary_rebuild=False,
                trigger="scope_maintenance",
            )
            if result is not None:
                results.append(result)
        return results

    def _resolve_status(
        self,
        thread: TopicThread,
        artifacts: list[dict[str, Any]],
    ) -> tuple[str, str]:
        current_status = str(thread.status or "active").strip().lower() or "active"
        if current_status in {"merged", "deleted"}:
            return current_status, "preserve_terminal_status"

        metadata = dict(thread.metadata or {})
        now = time.time()
        meaningful_at = float(metadata.get("last_meaningful_activity_at") or thread.created_at or now)
        idle_for_sec = max(0.0, now - meaningful_at)
        has_active_task = self._has_active_tasks(artifacts)
        has_open_questions = bool(_clean_strings(metadata.get("open_questions")))
        has_decisions = bool(_clean_strings(metadata.get("current_decisions")))

        if self._is_trivial_smalltalk_topic(thread, artifacts) and idle_for_sec >= float(self.TRIVIAL_DELETE_AFTER_SEC):
            return "deleted", "trivial_smalltalk_cleanup"
        if idle_for_sec >= float(self.ARCHIVE_AFTER_SEC) and not has_active_task and not has_open_questions:
            return "archived", "inactive_archive"
        if idle_for_sec >= float(self.SLEEP_AFTER_SEC) and not has_active_task:
            return "sleeping", "inactive_sleep"
        if current_status in {"sleeping", "archived"} and (
            idle_for_sec < float(self.SLEEP_AFTER_SEC) or has_active_task or has_open_questions or has_decisions
        ):
            return "active", "reactivated"
        return "active", "keep_active"

    def _build_metadata(
        self,
        thread: TopicThread,
        artifacts: list[dict[str, Any]],
        next_status: str,
        reason: str,
    ) -> dict[str, Any]:
        metadata = dict(thread.metadata or {})
        last_artifact_activity_at = max((float(row.get("updated_at") or 0.0) for row in artifacts), default=0.0)
        return {
            "topic_status": next_status,
            "topic_status_reason": reason,
            "last_lifecycle_at": time.time(),
            "artifact_count": len(artifacts),
            "last_artifact_activity_at": max(
                float(metadata.get("last_artifact_activity_at") or 0.0),
                float(last_artifact_activity_at or 0.0),
            ),
            "summary_stale": bool(metadata.get("summary_stale")),
        }

    @staticmethod
    def _metadata_needs_update(thread: TopicThread, metadata: dict[str, Any]) -> bool:
        existing = dict(thread.metadata or {})
        for key, value in metadata.items():
            if key == "last_lifecycle_at":
                continue
            if existing.get(key) != value:
                return True
        return False

    @staticmethod
    def _has_active_tasks(artifacts: list[dict[str, Any]]) -> bool:
        inactive_statuses = {"done", "completed", "closed", "archived", "cancelled", "canceled"}
        for row in list(artifacts or []):
            if str(row.get("artifact_type") or "").strip().lower() not in {"task", "task_state"}:
                continue
            metadata = dict(row.get("metadata") or {})
            task_status = str(metadata.get("task_status") or row.get("status") or "active").strip().lower()
            if task_status not in inactive_statuses:
                return True
        return False

    @staticmethod
    def _is_trivial_smalltalk_topic(thread: TopicThread, artifacts: list[dict[str, Any]]) -> bool:
        if artifacts:
            return False
        metadata = dict(thread.metadata or {})
        if _clean_strings(metadata.get("open_questions")) or _clean_strings(metadata.get("current_decisions")):
            return False
        turns = _clean_strings(metadata.get("recent_turn_texts"))
        if not turns:
            last_user_text = " ".join(str(metadata.get("last_user_text") or "").strip().split())
            turns = [last_user_text] if last_user_text else []
        if not turns or len(turns) > 2:
            return False
        return all(_looks_like_smalltalk(turn) for turn in turns)


def _clean_strings(items: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        clean = " ".join(str(item or "").strip().split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _looks_like_smalltalk(text: str) -> bool:
    clean = " ".join(str(text or "").strip().lower().split())
    if not clean:
        return True
    if len(clean.split()) > 4:
        return False
    markers = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "cool",
        "yes",
        "no",
        "yo",
        "привет",
        "здарова",
        "спасибо",
        "ок",
        "ага",
        "понял",
        "ясно",
        "да",
        "нет",
    }
    return clean in markers
