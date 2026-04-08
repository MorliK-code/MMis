from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from memory_core.topic.topic_linker import RelatedTopicLink, TopicRelationshipLinker
from memory_core.topic.topic_models import TopicThread
from memory_core.topic.topic_store import TopicStore


@dataclass(slots=True)
class TopicSummarySnapshot:
    thread_id: str
    summary: str
    open_questions: list[str]
    current_decisions: list[str]
    related_thread_ids: list[str]
    linked_tasks: list[dict[str, Any]]
    recent_episodes: list[dict[str, Any]]
    artifact_count: int
    episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "summary": self.summary,
            "open_questions": list(self.open_questions or []),
            "current_decisions": list(self.current_decisions or []),
            "related_thread_ids": list(self.related_thread_ids or []),
            "linked_tasks": [dict(row) for row in list(self.linked_tasks or [])],
            "recent_episodes": [dict(row) for row in list(self.recent_episodes or [])],
            "artifact_count": int(self.artifact_count or 0),
            "episode_count": int(self.episode_count or 0),
        }


class TopicSummaryBuilder:
    def __init__(
        self,
        topic_store: TopicStore,
        *,
        relationship_linker: TopicRelationshipLinker | None = None,
    ):
        self.topic_store = topic_store
        self.relationship_linker = relationship_linker or TopicRelationshipLinker(topic_store)

    def rebuild_thread(
        self,
        thread_id: str,
        *,
        workspace_id: str = "",
    ) -> TopicSummarySnapshot | None:
        thread = self.topic_store.get_thread(thread_id)
        if thread is None:
            return None

        artifacts = self.topic_store.list_thread_artifacts(
            thread.thread_id,
            workspace_id=str(workspace_id or thread.workspace_id or "").strip(),
            limit=200,
        )
        thread_meta = dict(thread.metadata or {})
        open_questions = self._collect_open_questions(artifacts)
        current_decisions = self._collect_decisions(artifacts)
        if not open_questions:
            open_questions = _clean_strings(thread_meta.get("open_questions"))[:10]
        if not current_decisions:
            current_decisions = _clean_strings(thread_meta.get("current_decisions"))[:10]
        linked_tasks = self._collect_linked_tasks(artifacts)
        recent_episodes = self._collect_recent_episodes(artifacts)
        episode_count = sum(
            1 for row in artifacts if str(row.get("artifact_type") or "").strip().lower() == "episode_event"
        )
        related_topics = self.relationship_linker.resolve_related_topics(
            thread,
            artifacts=artifacts,
            limit=6,
        )
        related_thread_ids = [row.thread_id for row in related_topics]
        summary = self._build_summary(
            thread=thread,
            linked_tasks=linked_tasks,
            recent_episodes=recent_episodes,
            open_questions=open_questions,
            current_decisions=current_decisions,
        )

        self._sync_artifact_links(
            thread=thread,
            artifacts=artifacts,
            open_questions=open_questions,
            current_decisions=current_decisions,
        )
        self.relationship_linker.sync_related_topic_links(thread.thread_id, related_topics)

        now = time.time()
        metadata = {
            "artifact_count": len(artifacts),
            "episode_count": int(episode_count),
            "open_questions": list(open_questions[:10]),
            "current_decisions": list(current_decisions[:10]),
            "linked_tasks": [dict(row) for row in linked_tasks[:10]],
            "recent_episodes": [dict(row) for row in recent_episodes[:8]],
            "summary_updated_at": now,
            "summary_build_version": 1,
        }
        self.topic_store.touch_thread(
            thread.thread_id,
            summary=summary,
            related_thread_ids=related_thread_ids,
            replace_related_thread_ids=True,
            metadata=metadata,
        )
        return TopicSummarySnapshot(
            thread_id=thread.thread_id,
            summary=summary,
            open_questions=open_questions[:10],
            current_decisions=current_decisions[:10],
            related_thread_ids=related_thread_ids,
            linked_tasks=linked_tasks[:10],
            recent_episodes=recent_episodes[:8],
            artifact_count=len(artifacts),
            episode_count=episode_count,
        )

    @staticmethod
    def needs_refresh(thread: TopicThread | None) -> bool:
        if thread is None:
            return False
        summary = str(thread.summary or "").strip()
        metadata = dict(thread.metadata or {})
        summary_updated_at = float(metadata.get("summary_updated_at") or 0.0)
        summary_build_version = int(metadata.get("summary_build_version") or 0)
        updated_at = float(thread.updated_at or 0.0)
        if not summary:
            return True
        if summary_build_version < 1:
            return True
        if summary_updated_at <= 0:
            return True
        return updated_at > (summary_updated_at + 1e-6)

    def _sync_artifact_links(
        self,
        *,
        thread: TopicThread,
        artifacts: list[dict[str, Any]],
        open_questions: list[str],
        current_decisions: list[str],
    ) -> None:
        for row in list(artifacts or []):
            artifact_id = str(row.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            metadata = dict(row.get("metadata") or {})
            base_meta = {
                "topic_thread_id": thread.thread_id,
                "topic_key": str(thread.topic_key or "").strip(),
            }
            self.topic_store.replace_links(
                artifact_id,
                "belongs_to_topic",
                [(thread.thread_id, dict(base_meta))],
            )
            if str(row.get("artifact_type") or "").strip().lower() == "episode_event":
                self.topic_store.replace_links(
                    artifact_id,
                    "episode_in_topic",
                    [(thread.thread_id, dict(base_meta))],
                )
            else:
                self.topic_store.replace_links(artifact_id, "episode_in_topic", [])
            if str(row.get("artifact_type") or "").strip().lower() in {"task", "task_state"}:
                task_meta = {
                    **base_meta,
                    "task_status": str(metadata.get("task_status") or row.get("status") or "").strip() or "active",
                    "open_questions": list(_clean_strings(metadata.get("open_questions"))[:5]),
                    "decisions": list(_clean_strings(metadata.get("decisions") or metadata.get("current_decisions"))[:5]),
                }
                self.topic_store.replace_links(
                    artifact_id,
                    "task_in_topic",
                    [(thread.thread_id, task_meta)],
                )
            else:
                self.topic_store.replace_links(artifact_id, "task_in_topic", [])
            artifact_questions = self._extract_questions_from_row(row)
            if artifact_questions:
                self.topic_store.replace_links(
                    artifact_id,
                    "question_in_topic",
                    [(thread.thread_id, {**base_meta, "items": artifact_questions[:5]})],
                )
            else:
                self.topic_store.replace_links(artifact_id, "question_in_topic", [])
            artifact_decisions = self._extract_decisions_from_row(row)
            if artifact_decisions:
                self.topic_store.replace_links(
                    artifact_id,
                    "decision_in_topic",
                    [(thread.thread_id, {**base_meta, "items": artifact_decisions[:5]})],
                )
            else:
                self.topic_store.replace_links(artifact_id, "decision_in_topic", [])

        self.topic_store.replace_links(
            thread.thread_id,
            "question_in_topic",
            [(thread.thread_id, {"items": open_questions[:10]})] if open_questions else [],
        )
        self.topic_store.replace_links(
            thread.thread_id,
            "decision_in_topic",
            [(thread.thread_id, {"items": current_decisions[:10]})] if current_decisions else [],
        )

    def _collect_open_questions(self, artifacts: list[dict[str, Any]]) -> list[str]:
        items: list[str] = []
        for row in list(artifacts or []):
            for item in self._extract_questions_from_row(row):
                if item not in items:
                    items.append(item)
        return items[:10]

    def _collect_decisions(self, artifacts: list[dict[str, Any]]) -> list[str]:
        items: list[str] = []
        for row in list(artifacts or []):
            for item in self._extract_decisions_from_row(row):
                if item not in items:
                    items.append(item)
        return items[:10]

    def _collect_linked_tasks(self, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in list(artifacts or []):
            artifact_type = str(row.get("artifact_type") or "").strip().lower()
            if artifact_type not in {"task", "task_state"}:
                continue
            metadata = dict(row.get("metadata") or {})
            result.append(
                {
                    "artifact_id": str(row.get("artifact_id") or "").strip(),
                    "text": str(row.get("text") or "").strip(),
                    "summary": str(row.get("summary") or row.get("text") or "").strip(),
                    "status": str(metadata.get("task_status") or row.get("status") or "").strip() or "active",
                    "open_questions": self._extract_questions_from_row(row)[:4],
                    "decisions": self._extract_decisions_from_row(row)[:4],
                    "updated_at": float(row.get("updated_at") or 0.0),
                }
            )
        result.sort(key=lambda item: float(item.get("updated_at") or 0.0), reverse=True)
        return result

    def _collect_recent_episodes(self, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in list(artifacts or []):
            if str(row.get("artifact_type") or "").strip().lower() != "episode_event":
                continue
            result.append(
                {
                    "artifact_id": str(row.get("artifact_id") or "").strip(),
                    "summary": str(row.get("summary") or row.get("text") or "").strip(),
                    "updated_at": float(row.get("updated_at") or 0.0),
                }
            )
        result.sort(key=lambda item: float(item.get("updated_at") or 0.0), reverse=True)
        return result

    def _build_summary(
        self,
        *,
        thread: TopicThread,
        linked_tasks: list[dict[str, Any]],
        recent_episodes: list[dict[str, Any]],
        open_questions: list[str],
        current_decisions: list[str],
    ) -> str:
        parts: list[str] = []
        focus = [
            str(item.get("summary") or item.get("text") or "").strip()
            for item in list(linked_tasks or [])[:2]
            if str(item.get("summary") or item.get("text") or "").strip()
        ]
        if not focus:
            focus = [
                str(item.get("summary") or "").strip()
                for item in list(recent_episodes or [])[:2]
                if str(item.get("summary") or "").strip()
            ]
        if not focus:
            metadata = dict(thread.metadata or {})
            focus = _clean_strings(metadata.get("recent_turn_texts"))[:2]
        if not focus:
            last_user_text = " ".join(str(dict(thread.metadata or {}).get("last_user_text") or "").strip().split())
            if last_user_text:
                focus = [last_user_text[:180]]
        if focus:
            parts.append("Focus: " + "; ".join(focus[:2]))
        if current_decisions:
            parts.append("Decisions: " + "; ".join(list(current_decisions or [])[:2]))
        if open_questions:
            parts.append("Open questions: " + "; ".join(list(open_questions or [])[:2]))
        summary = " ".join(part.strip() for part in parts if str(part).strip()).strip()
        if not summary:
            summary = str(thread.summary or thread.title or thread.topic_key or thread.thread_id).strip()
        return summary[:480]

    def _extract_questions_from_row(self, row: dict[str, Any]) -> list[str]:
        metadata = dict(row.get("metadata") or {})
        items: list[str] = list(_clean_strings(metadata.get("open_questions")))
        if not items:
            items.extend(_extract_questions_from_text(row.get("text")))
        if not items:
            items.extend(_extract_questions_from_text(row.get("summary")))
        return items[:5]

    def _extract_decisions_from_row(self, row: dict[str, Any]) -> list[str]:
        metadata = dict(row.get("metadata") or {})
        items = list(_clean_strings(metadata.get("decisions") or metadata.get("current_decisions")))
        if not items:
            items.extend(_extract_decisions_from_text(row.get("text")))
        if not items:
            items.extend(_extract_decisions_from_text(row.get("summary")))
        return items[:5]


def _clean_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in list(values or []):
        clean = " ".join(str(value or "").strip().split())
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _extract_questions_from_text(text: Any) -> list[str]:
    raw = str(text or "").strip()
    if "?" not in raw:
        return []
    result: list[str] = []
    for chunk in raw.replace("?", "?\n").splitlines():
        clean = " ".join(chunk.strip().split())
        if "?" not in clean or len(clean) < 6:
            continue
        if clean not in result:
            result.append(clean)
    return result[:3]


def _extract_decisions_from_text(text: Any) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    markers = [
        "решили",
        "договорились",
        "пришли к выводу",
        "итог:",
        "вывод:",
        "решение:",
    ]
    result: list[str] = []
    lower = raw.lower()
    for marker in markers:
        idx = lower.find(marker)
        if idx < 0:
            continue
        start = raw.rfind(". ", 0, idx)
        start = 0 if start < 0 else start + 2
        end = raw.find(". ", idx)
        end = len(raw) if end < 0 else end
        clean = " ".join(raw[start:end].strip().split())
        if clean and clean not in result:
            result.append(clean[:200])
    return result[:3]
