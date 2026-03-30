from __future__ import annotations

import re
from typing import Any

from llm.provider_base import ToolSpec
from memory_core.topic.topic_models import TopicThread
from memory_core.topic.topic_store import TopicStore


def topic_read_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="topic_read",
        description="Read summary, recent episodes, open questions, linked tasks, and recent artifacts of a topic thread.",
        input_schema={
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "Topic thread id to inspect.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Maximum number of recent artifacts to include.",
                },
            },
            "required": ["thread_id"],
            "additionalProperties": False,
        },
    )


def topic_search_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="topic_search",
        description="Search hidden topic threads inside the current visible chat by subject, keyword, title, summary, or tags.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic search query.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 12,
                    "description": "Maximum number of topics to return.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "sleeping", "archived", "all"],
                    "description": "Optional status filter.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def topic_related_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="topic_related",
        description="Return topics related to a given topic thread using explicit links and lightweight similarity.",
        input_schema={
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "Topic thread id to expand.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 12,
                    "description": "Maximum number of related topics to return.",
                },
            },
            "required": ["thread_id"],
            "additionalProperties": False,
        },
    )


def topic_tools_list() -> list[ToolSpec]:
    return [
        topic_read_tool_spec(),
        topic_search_tool_spec(),
        topic_related_tool_spec(),
    ]


class TopicToolService:
    def __init__(self, topic_store: TopicStore):
        self.topic_store = topic_store

    def search_topics(
        self,
        *,
        query: str,
        visible_chat_id: str = "",
        workspace_id: str = "",
        session_id: str = "",
        status: str | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        clean_query = str(query or "").strip()
        threads = self.topic_store.list_threads(
            visible_chat_id=str(visible_chat_id or "").strip(),
            workspace_id=str(workspace_id or "").strip(),
            session_id=str(session_id or "").strip(),
            status=None if str(status or "").strip().lower() == "all" else status,
            limit=max(int(limit or 8) * 4, 24),
        )
        if not clean_query:
            ranked = [(thread, 0.0) for thread in threads]
        else:
            ranked = [
                (thread, self._search_score(clean_query, thread))
                for thread in threads
            ]
            ranked = [item for item in ranked if item[1] > 0.0]
        ranked.sort(key=lambda item: (float(item[1]), float(item[0].updated_at or 0.0)), reverse=True)
        topics = [
            self._compact_thread(thread, artifact_count=self._artifact_count(thread), score=score)
            for thread, score in ranked[: max(1, int(limit or 8))]
        ]
        return {
            "query": clean_query,
            "count": len(topics),
            "topics": topics,
        }

    def read_topic(
        self,
        thread_id: str,
        *,
        limit: int = 20,
        workspace_id: str = "",
    ) -> dict[str, Any] | None:
        thread = self.topic_store.get_thread(thread_id)
        if thread is None:
            return None

        artifacts = self.topic_store.list_thread_artifacts(
            thread.thread_id,
            workspace_id=str(workspace_id or thread.workspace_id or "").strip(),
            limit=max(int(limit or 20) * 3, 24),
        )

        thread_meta = dict(thread.metadata or {})
        open_questions: list[str] = [
            str(item).strip()
            for item in list(thread_meta.get("open_questions") or [])
            if str(item).strip()
        ]
        current_decisions: list[str] = [
            str(item).strip()
            for item in list(thread_meta.get("current_decisions") or [])
            if str(item).strip()
        ]
        linked_tasks: list[dict[str, Any]] = []
        recent_episodes: list[dict[str, Any]] = []
        recent_artifacts: list[dict[str, Any]] = []

        for row in artifacts:
            metadata = dict(row.get("metadata") or {})
            for item in list(metadata.get("open_questions") or []):
                value = str(item or "").strip()
                if value and value not in open_questions:
                    open_questions.append(value)
            for item in list(metadata.get("decisions") or metadata.get("current_decisions") or []):
                value = str(item or "").strip()
                if value and value not in current_decisions:
                    current_decisions.append(value)
            if str(row.get("artifact_type") or "") == "episode_event":
                recent_episodes.append(
                    {
                        "artifact_id": row.get("artifact_id"),
                        "summary": str(row.get("summary") or row.get("text") or "").strip(),
                        "updated_at": row.get("updated_at"),
                    }
                )
            if str(row.get("artifact_type") or "") in {"task", "task_state"}:
                linked_tasks.append(
                    {
                        "artifact_id": row.get("artifact_id"),
                        "text": str(row.get("text") or "").strip(),
                        "summary": str(row.get("summary") or "").strip(),
                        "status": str(metadata.get("task_status") or row.get("status") or "").strip(),
                    }
                )
            recent_artifacts.append(
                {
                    "artifact_id": row.get("artifact_id"),
                    "artifact_type": row.get("artifact_type"),
                    "summary": str(row.get("summary") or row.get("text") or "").strip(),
                    "updated_at": row.get("updated_at"),
                    "status": row.get("status"),
                }
            )

        related_topics = self.related_topics(
            thread.thread_id,
            visible_chat_id=thread.visible_chat_id,
            workspace_id=thread.workspace_id,
            session_id=thread.session_id,
            limit=min(6, max(1, int(limit or 20))),
        ).get("topics", [])

        return {
            "thread": self._compact_thread(thread, artifact_count=len(artifacts)),
            "summary": str(thread.summary or "").strip(),
            "open_questions": open_questions[:10],
            "current_decisions": current_decisions[:10],
            "linked_tasks": linked_tasks[:10],
            "recent_episodes": recent_episodes[: min(8, int(limit or 20))],
            "recent_artifacts": recent_artifacts[: max(1, int(limit or 20))],
            "related_topics": related_topics,
        }

    def related_topics(
        self,
        thread_id: str,
        *,
        visible_chat_id: str = "",
        workspace_id: str = "",
        session_id: str = "",
        limit: int = 8,
    ) -> dict[str, Any]:
        thread = self.topic_store.get_thread(thread_id)
        if thread is None:
            return {"thread_id": str(thread_id or "").strip(), "count": 0, "topics": []}

        threads = self.topic_store.list_threads(
            visible_chat_id=str(visible_chat_id or thread.visible_chat_id or "").strip(),
            workspace_id=str(workspace_id or thread.workspace_id or "").strip(),
            session_id=str(session_id or thread.session_id or "").strip(),
            limit=max(int(limit or 8) * 4, 24),
        )

        explicit: list[TopicThread] = []
        seen: set[str] = {thread.thread_id}
        for related_id in list(thread.related_thread_ids or []):
            related = self.topic_store.get_thread(related_id)
            if related is None or related.thread_id in seen:
                continue
            seen.add(related.thread_id)
            explicit.append(related)

        target_tokens = self._thread_tokens(thread)
        target_tags = {str(item or "").strip().lower() for item in list(thread.tags or []) if str(item or "").strip()}
        scored: list[tuple[TopicThread, float]] = []
        for candidate in threads:
            if candidate.thread_id in seen:
                continue
            score = 0.0
            if thread.thread_id in set(candidate.related_thread_ids or []):
                score += 0.45
            score += self._overlap_ratio(target_tags, {str(item or "").strip().lower() for item in list(candidate.tags or []) if str(item or "").strip()}) * 0.30
            score += self._overlap_ratio(target_tokens, self._thread_tokens(candidate)) * 0.25
            if score <= 0.0:
                continue
            scored.append((candidate, min(1.0, score)))
        scored.sort(key=lambda item: (float(item[1]), float(item[0].updated_at or 0.0)), reverse=True)

        ordered: list[dict[str, Any]] = [
            self._compact_thread(item, artifact_count=self._artifact_count(item), relation="explicit")
            for item in explicit
        ]
        for candidate, score in scored:
            if len(ordered) >= max(1, int(limit or 8)):
                break
            ordered.append(
                self._compact_thread(
                    candidate,
                    artifact_count=self._artifact_count(candidate),
                    relation="similar",
                    score=score,
                )
            )
        return {
            "thread_id": thread.thread_id,
            "count": len(ordered),
            "topics": ordered[: max(1, int(limit or 8))],
        }

    def _artifact_count(self, thread: TopicThread) -> int:
        return len(
            self.topic_store.list_thread_artifacts(
                thread.thread_id,
                workspace_id=thread.workspace_id,
                limit=200,
            )
        )

    @staticmethod
    def _compact_thread(
        thread: TopicThread,
        *,
        artifact_count: int = 0,
        relation: str = "",
        score: float | None = None,
    ) -> dict[str, Any]:
        payload = {
            "thread_id": thread.thread_id,
            "topic_key": str(thread.topic_key or "").strip(),
            "title": str(thread.title or thread.topic_key or thread.thread_id).strip(),
            "status": str(thread.status or "").strip() or "active",
            "summary": str(thread.summary or "").strip(),
            "tags": list(thread.tags or []),
            "updated_at": float(thread.updated_at or 0.0),
            "artifact_count": int(artifact_count),
        }
        if relation:
            payload["relation"] = relation
        if score is not None:
            payload["score"] = round(float(score), 4)
        return payload

    @staticmethod
    def _search_score(query: str, thread: TopicThread) -> float:
        query_tokens = _tokenize(query)
        hay_text = " ".join(
            part
            for part in (
                thread.topic_key,
                thread.title,
                thread.summary,
                " ".join(thread.tags),
            )
            if str(part or "").strip()
        )
        hay_text_lower = hay_text.lower()
        hay_tokens = _tokenize(hay_text)
        if not query_tokens and not str(query or "").strip():
            return 0.0
        overlap = TopicToolService._overlap_ratio(query_tokens, hay_tokens)
        exact = 0.0
        clean_query = str(query or "").strip().lower()
        if clean_query and clean_query in hay_text_lower:
            exact += 0.55
            if clean_query == str(thread.topic_key or "").strip().lower():
                exact += 0.15
        return min(1.0, overlap * 0.45 + exact)

    @staticmethod
    def _thread_tokens(thread: TopicThread) -> set[str]:
        return _tokenize(
            " ".join(
                part
                for part in (
                    thread.topic_key,
                    thread.title,
                    thread.summary,
                    " ".join(thread.tags),
                )
                if str(part or "").strip()
            )
        )

    @staticmethod
    def _overlap_ratio(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        shared = left & right
        return len(shared) / max(len(left), len(right), 1)


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_]{2,}", str(text or "").lower())
        if len(token) >= 2
    }
