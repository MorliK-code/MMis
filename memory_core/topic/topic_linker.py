from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory_core.topic.topic_models import TopicThread
from memory_core.topic.topic_store import TopicStore


@dataclass(slots=True)
class RelatedTopicLink:
    thread_id: str
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "score": round(float(self.score or 0.0), 4),
            "source": str(self.source or "").strip() or "similarity",
        }


class TopicRelationshipLinker:
    def __init__(self, topic_store: TopicStore):
        self.topic_store = topic_store

    def resolve_related_topics(
        self,
        thread: TopicThread,
        *,
        artifacts: list[dict[str, Any]] | None = None,
        limit: int = 6,
    ) -> list[RelatedTopicLink]:
        threads = self.topic_store.list_threads(
            visible_chat_id=str(thread.visible_chat_id or "").strip(),
            workspace_id=str(thread.workspace_id or "").strip(),
            session_id=str(thread.session_id or "").strip(),
            limit=max(int(limit or 6) * 5, 30),
        )
        artifacts = list(artifacts or [])
        explicit_ids = [
            str(item).strip()
            for item in list(thread.related_thread_ids or [])
            if str(item).strip()
        ]
        artifact_ids = _collect_artifact_related_topic_ids(artifacts)
        current_tags = _collect_tags(thread, artifacts)
        current_tokens = _collect_tokens(thread, artifacts)
        ranked: dict[str, RelatedTopicLink] = {}

        def remember(candidate_id: str, score: float, source: str) -> None:
            clean_id = str(candidate_id or "").strip()
            if not clean_id or clean_id == thread.thread_id:
                return
            if self.topic_store.get_thread(clean_id) is None:
                return
            existing = ranked.get(clean_id)
            if existing is not None and float(existing.score or 0.0) >= float(score or 0.0):
                return
            ranked[clean_id] = RelatedTopicLink(
                thread_id=clean_id,
                score=min(1.0, max(0.0, float(score or 0.0))),
                source=str(source or "").strip() or "similarity",
            )

        for candidate_id in explicit_ids:
            remember(candidate_id, 0.95, "explicit")
        for candidate_id in artifact_ids:
            remember(candidate_id, 0.85, "artifact_reference")

        for candidate in threads:
            if candidate.thread_id == thread.thread_id:
                continue
            score = 0.0
            candidate_tags = _collect_tags(candidate, [])
            candidate_tokens = _collect_tokens(candidate, [])
            score += _overlap_ratio(current_tags, candidate_tags) * 0.45
            score += _overlap_ratio(current_tokens, candidate_tokens) * 0.55
            if thread.thread_id in set(candidate.related_thread_ids or []):
                score += 0.1
            if score < 0.35:
                continue
            remember(candidate.thread_id, score, "semantic_overlap")

        ordered = sorted(
            ranked.values(),
            key=lambda item: (
                float(item.score or 0.0),
                float((self.topic_store.get_thread(item.thread_id) or thread).updated_at or 0.0),
            ),
            reverse=True,
        )
        return ordered[: max(1, int(limit or 6))]

    def sync_related_topic_links(
        self,
        thread_id: str,
        related_topics: list[RelatedTopicLink] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        targets: list[tuple[str, dict[str, Any]]] = []
        for item in list(related_topics or []):
            row = item.to_dict() if hasattr(item, "to_dict") else dict(item or {})
            target_id = str(row.get("thread_id") or "").strip()
            if not target_id:
                continue
            targets.append(
                (
                    target_id,
                    {
                        "score": round(float(row.get("score") or 0.0), 4),
                        "source": str(row.get("source") or "").strip() or "similarity",
                    },
                )
            )
        return self.topic_store.replace_links(str(thread_id or "").strip(), "related_topic", targets)


def _collect_artifact_related_topic_ids(artifacts: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for row in list(artifacts or []):
        metadata = dict(row.get("metadata") or {})
        for value in list(metadata.get("related_topic_thread_ids") or metadata.get("related_topic_ids") or []):
            clean = str(value or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            result.append(clean)
    return result


def _collect_tags(thread: TopicThread, artifacts: list[dict[str, Any]]) -> set[str]:
    values: set[str] = {
        str(item or "").strip().lower()
        for item in list(thread.tags or [])
        if str(item or "").strip()
    }
    for row in list(artifacts or []):
        metadata = dict(row.get("metadata") or {})
        for value in list(metadata.get("retrieve_when") or metadata.get("tags") or []):
            clean = str(value or "").strip().lower()
            if clean:
                values.add(clean)
    return values


def _collect_tokens(thread: TopicThread, artifacts: list[dict[str, Any]]) -> set[str]:
    parts = [
        str(thread.topic_key or "").strip(),
        str(thread.title or "").strip(),
        str(thread.summary or "").strip(),
    ]
    for row in list(artifacts or [])[:12]:
        parts.append(str(row.get("summary") or "").strip())
        parts.append(str(row.get("text") or "").strip())
    tokens: set[str] = set()
    for raw in parts:
        for token in _tokenize(raw):
            tokens.add(token)
    return tokens


def _tokenize(text: str) -> list[str]:
    clean = str(text or "").lower()
    buff: list[str] = []
    token = []
    for char in clean:
        if char.isalnum():
            token.append(char)
            continue
        if len(token) >= 3:
            buff.append("".join(token))
        token = []
    if len(token) >= 3:
        buff.append("".join(token))
    return buff


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap <= 0:
        return 0.0
    return float(overlap) / float(max(len(left), len(right), 1))
