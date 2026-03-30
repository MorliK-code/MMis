from __future__ import annotations

import json
import time
import uuid
from typing import Any

from memory_core.storage.artifact_store import ArtifactStore
from memory_core.topic.topic_models import TopicThread


class TopicStore:
    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def create_thread(self, thread: TopicThread) -> TopicThread:
        self.artifact_store.create(thread.to_artifact())
        return thread

    def upsert_thread(self, thread: TopicThread) -> TopicThread:
        artifact = thread.to_artifact()
        existing = self.artifact_store.get_by_id(thread.thread_id)
        if existing is None:
            self.artifact_store.create(artifact)
        else:
            self.artifact_store.update(artifact)
        return thread

    def get_thread(self, thread_id: str) -> TopicThread | None:
        artifact = self.artifact_store.get_by_id(str(thread_id or "").strip())
        if artifact is None or artifact.artifact_type != "topic_thread":
            return None
        return TopicThread.from_artifact(artifact)

    def list_threads(
        self,
        *,
        visible_chat_id: str = "",
        workspace_id: str = "",
        session_id: str = "",
        status: str | None = None,
        limit: int = 100,
    ) -> list[TopicThread]:
        artifacts = self.artifact_store.list_artifacts(
            artifact_type="topic_thread",
            workspace_id=(str(workspace_id or "").strip() or None),
            status=("active" if status == "active" else None) if status else None,
            limit=max(limit * 3, limit),
        )
        result: list[TopicThread] = []
        want_visible_chat = str(visible_chat_id or "").strip()
        want_session = str(session_id or "").strip()
        want_status = str(status or "").strip().lower()
        for artifact in artifacts:
            thread = TopicThread.from_artifact(artifact)
            if want_visible_chat and thread.visible_chat_id != want_visible_chat:
                continue
            if want_session and thread.session_id != want_session:
                continue
            if want_status and str(thread.status or "").strip().lower() != want_status:
                continue
            result.append(thread)
        result.sort(key=lambda item: float(item.updated_at or 0.0), reverse=True)
        return result[:limit]

    def touch_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        tags: list[str] | None = None,
        related_thread_ids: list[str] | None = None,
        replace_related_thread_ids: bool = False,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
    ) -> TopicThread | None:
        thread = self.get_thread(thread_id)
        if thread is None:
            return None
        if title is not None and str(title or "").strip():
            thread.title = str(title).strip()
        if summary is not None:
            thread.summary = str(summary or "").strip()
        if tags is not None:
            thread.tags = _merge_unique(thread.tags, tags)
        if related_thread_ids is not None:
            if replace_related_thread_ids:
                thread.related_thread_ids = _merge_unique([], related_thread_ids)
            else:
                thread.related_thread_ids = _merge_unique(thread.related_thread_ids, related_thread_ids)
        if metadata:
            merged = dict(thread.metadata or {})
            merged.update(dict(metadata))
            thread.metadata = merged
        if status:
            thread.status = str(status).strip() or thread.status
        thread.updated_at = time.time()
        self.upsert_thread(thread)
        return thread

    def list_thread_artifacts(
        self,
        thread_id: str,
        *,
        workspace_id: str = "",
        limit: int = 200,
        include_topic_thread_artifacts: bool = False,
    ) -> list[dict[str, Any]]:
        target = str(thread_id or "").strip()
        if not target:
            return []
        artifacts = self.artifact_store.list_artifacts(
            workspace_id=(str(workspace_id or "").strip() or None),
            limit=max(limit * 3, limit),
        )
        result: list[dict[str, Any]] = []
        for artifact in artifacts:
            if not include_topic_thread_artifacts and artifact.artifact_type == "topic_thread":
                continue
            meta = dict(artifact.metadata or {})
            if str(meta.get("topic_thread_id") or "").strip() != target:
                continue
            result.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "artifact_type": artifact.artifact_type,
                    "text": artifact.text,
                    "summary": artifact.summary,
                    "status": artifact.status,
                    "created_at": artifact.created_at,
                    "updated_at": artifact.updated_at,
                    "metadata": meta,
                }
            )
        result.sort(key=lambda item: float(item.get("updated_at") or 0.0), reverse=True)
        return result[:limit]

    def upsert_link(
        self,
        src_artifact_id: str,
        dst_artifact_id: str,
        link_type: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        src_id = str(src_artifact_id or "").strip()
        dst_id = str(dst_artifact_id or "").strip()
        clean_type = str(link_type or "").strip()
        if not src_id or not dst_id or not clean_type:
            return None
        payload = dict(metadata or {})
        now = time.time()
        link_id = _make_link_id(src_id, dst_id, clean_type)
        existing = self.artifact_store.db.fetchone(
            "SELECT link_id FROM artifact_links WHERE link_id = ?",
            (link_id,),
        )
        params = (json.dumps(payload), now, link_id)
        if existing is None:
            self.artifact_store.db.execute(
                """
                INSERT INTO artifact_links (
                    link_id,
                    src_artifact_id,
                    dst_artifact_id,
                    link_type,
                    metadata_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (link_id, src_id, dst_id, clean_type, json.dumps(payload), now),
            )
        else:
            self.artifact_store.db.execute(
                """
                UPDATE artifact_links
                SET metadata_json = ?, created_at = ?
                WHERE link_id = ?
                """,
                params,
            )
        return {
            "link_id": link_id,
            "src_artifact_id": src_id,
            "dst_artifact_id": dst_id,
            "link_type": clean_type,
            "metadata": payload,
            "created_at": now,
        }

    def replace_links(
        self,
        src_artifact_id: str,
        link_type: str,
        targets: list[tuple[str, dict[str, Any] | None]] | None,
    ) -> list[dict[str, Any]]:
        src_id = str(src_artifact_id or "").strip()
        clean_type = str(link_type or "").strip()
        if not src_id or not clean_type:
            return []
        desired = [
            (str(dst_id or "").strip(), dict(meta or {}))
            for dst_id, meta in list(targets or [])
            if str(dst_id or "").strip()
        ]
        desired_ids = [dst_id for dst_id, _ in desired]
        if desired_ids:
            placeholders = ", ".join("?" for _ in desired_ids)
            self.artifact_store.db.execute(
                f"""
                DELETE FROM artifact_links
                WHERE src_artifact_id = ? AND link_type = ?
                  AND dst_artifact_id NOT IN ({placeholders})
                """,
                tuple([src_id, clean_type, *desired_ids]),
            )
        else:
            self.artifact_store.db.execute(
                "DELETE FROM artifact_links WHERE src_artifact_id = ? AND link_type = ?",
                (src_id, clean_type),
            )
        result: list[dict[str, Any]] = []
        for dst_id, meta in desired:
            row = self.upsert_link(src_id, dst_id, clean_type, metadata=meta)
            if row is not None:
                result.append(row)
        return result

    def list_links(
        self,
        *,
        src_artifact_id: str = "",
        dst_artifact_id: str = "",
        link_type: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        clean_src = str(src_artifact_id or "").strip()
        clean_dst = str(dst_artifact_id or "").strip()
        clean_type = str(link_type or "").strip()
        if clean_src:
            conditions.append("src_artifact_id = ?")
            params.append(clean_src)
        if clean_dst:
            conditions.append("dst_artifact_id = ?")
            params.append(clean_dst)
        if clean_type:
            conditions.append("link_type = ?")
            params.append(clean_type)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.artifact_store.db.fetchall(
            f"""
            SELECT *
            FROM artifact_links
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple([*params, max(1, int(limit or 200))]),
        )
        return [_row_to_link(row) for row in rows]


def _merge_unique(existing: list[str] | None, extra: list[str] | None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in list(existing or []) + list(extra or []):
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _make_link_id(src_artifact_id: str, dst_artifact_id: str, link_type: str) -> str:
    seed = f"{src_artifact_id}|{dst_artifact_id}|{link_type}"
    return f"link_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:24]}"


def _row_to_link(row: Any) -> dict[str, Any]:
    return {
        "link_id": row["link_id"],
        "src_artifact_id": row["src_artifact_id"],
        "dst_artifact_id": row["dst_artifact_id"],
        "link_type": row["link_type"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }
