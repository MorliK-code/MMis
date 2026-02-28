from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from memory.event_store import EventStore
from memory.fact_extractor import Fact, FactExtractor, MODE_BALANCED
from memory.long_memory import LongMemory
from memory.profile_store import AssistantProfileStore, UserProfileStore
from memory.short_memory import ShortMemory
from memory.vector_store import VectorStore
from prompt_engine.prompt_registry import PromptRegistry
from utils.logger import get_logger, log_json


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class MemoryItem:
    id: str
    text: str
    score: float
    source: str
    tags: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": float(self.score),
            "source": self.source,
            "tags": list(self.tags or []),
            "metadata": dict(self.metadata or {}),
        }


class MemoryManager:
    """Memory orchestrator: ingestion, retrieval, facts, profiles, context pack."""

    def __init__(
        self,
        *,
        short_memory: ShortMemory | None = None,
        long_memory: LongMemory | None = None,
        vector_store: VectorStore | None = None,
        fact_extractor: FactExtractor | None = None,
        user_profile_store: UserProfileStore | None = None,
        assistant_profile_store: AssistantProfileStore | None = None,
        event_store: EventStore | None = None,
        retrieve_score_threshold: float = 0.28,
        prompt_registry: PromptRegistry | None = None,
    ):
        self.short_memory = short_memory or ShortMemory(limit=80, summary_trigger=60)
        self.long_memory = long_memory or LongMemory()
        self.vector_store = vector_store or VectorStore(dim=128)
        self.fact_extractor = fact_extractor or FactExtractor()
        self.user_profile_store = user_profile_store or UserProfileStore()
        self.assistant_profile_store = assistant_profile_store or AssistantProfileStore()
        self.event_store = event_store or EventStore()
        self.retrieve_score_threshold = max(0.0, min(1.0, float(retrieve_score_threshold)))
        self._prompt_registry = prompt_registry or PromptRegistry()

    def ingest_message(
        self,
        role: str,
        text: str,
        metadata: dict | None = None,
        ts: float | None = None,
        **ids,
    ) -> None:
        content = str(text or "").strip()
        if not content:
            return
        role_norm = _normalize_role(role)
        meta = dict(metadata or {})
        now = float(ts or time.time())
        tags = [str(x) for x in list(meta.get("tags") or []) if str(x).strip()]
        lang = str(meta.get("lang") or "")
        intent = str(meta.get("intent") or "")
        emotion = str(meta.get("mood") or meta.get("emotion") or "")
        has_code = bool(meta.get("has_code") or ("has_code" in {t.lower() for t in tags}))
        trace_id = str(ids.get("trace_id") or meta.get("trace_id") or "")
        model = str(ids.get("model") or meta.get("model") or "")
        source = str(ids.get("source") or meta.get("source") or "text")
        profile = str(ids.get("quality_profile") or meta.get("quality_profile") or MODE_BALANCED).upper()
        profile_id = str(ids.get("profile_id") or meta.get("user_id") or "default")

        event = self.event_store.append(
            {
                "ts": now,
                "type": f"{role_norm}_message",
                "payload": {
                    "role": role_norm,
                    "text": content,
                    "metadata": meta,
                    "ids": {k: v for k, v in ids.items()},
                },
                "tags": tags,
                "trace_id": trace_id,
                "model": model,
                "latency_ms": float(meta.get("latency_ms") or 0.0),
            }
        )
        event_id = str(event.get("event_id") or "")
        log_json(
            LOGGER,
            "memory_ingest_message",
            event_id=event_id,
            role=role_norm,
            text_chars=len(content),
            tags=len(tags),
            lang=lang,
            intent=intent,
            emotion=emotion,
            model=model,
            source=source,
        )

        self.short_memory.append(
            {
                "id": event_id,
                "role": role_norm,
                "type": "message",
                "text": content,
                "ts": now,
                "lang": lang,
                "intent": intent,
                "emotion": emotion,
                "tags": tags,
                "has_code": has_code,
                "meta": {
                    "trace_id": trace_id,
                    "source": source,
                    "model": model,
                    "conversation_id": ids.get("conversation_id"),
                    "turn_id": ids.get("turn_id"),
                },
            }
        )

        # Message-level index into vector store.
        self.vector_store.upsert(
            id=f"msg:{event_id}",
            text=content,
            embedding=None,
            metadata={
                "type": "message",
                "role": role_norm,
                "lang": lang,
                "topic": str(meta.get("topic") or ""),
                "user_id": profile_id,
                "event_id": event_id,
                "source": source,
                "ts": now,
                "tags": tags,
            },
        )

        if role_norm == "assistant":
            importance = _clamp01(float(meta.get("importance") or 0.42))
            confidence = _clamp01(float(meta.get("confidence") or 0.62))
            doc = self.long_memory.add_doc(
                text=content,
                meta={"event_id": event_id, "role": role_norm, "source": source, "trace_id": trace_id},
                source="chat",
                tags=tags,
                importance=importance,
                confidence=confidence,
            )
            self.vector_store.upsert(
                id=f"doc:{doc.id}",
                text=doc.text,
                metadata={
                    "type": "summary",
                    "role": role_norm,
                    "lang": lang,
                    "topic": str(meta.get("topic") or ""),
                    "user_id": profile_id,
                    "doc_id": doc.id,
                    "event_id": event_id,
                    "source": "chat",
                    "ts": now,
                    "tags": tags,
                    "importance": doc.importance,
                    "confidence": doc.confidence,
                },
            )

        facts = self.fact_extractor.extract(
            text=content,
            metadata={"event_id": event_id, **meta},
            speaker=role_norm,
            mode=profile,
        )
        if facts:
            self.write_facts(facts)

    def retrieve(self, query: str, k: int = 8, filters: dict | None = None) -> list[MemoryItem]:
        text = str(query or "").strip()
        if not text:
            return []
        limit = max(1, int(k))
        filt = dict(filters or {})
        threshold = float(filt.get("score_threshold", self.retrieve_score_threshold))

        short_tail_n = int(filt.get("short_n", max(20, limit * 3)))
        short_items = self.short_memory.tail(short_tail_n)
        short_ranked: list[MemoryItem] = []
        for idx, row in enumerate(reversed(short_items)):
            src_text = str(row.get("text") or "")
            if not src_text:
                continue
            recency_score = 1.0 - (idx / max(1, len(short_items)))
            similarity = _text_similarity(text, src_text)
            score = _clamp01(0.55 * recency_score + 0.45 * similarity)
            if score < threshold:
                continue
            short_ranked.append(
                MemoryItem(
                    id=str(row.get("id") or f"short:{idx}"),
                    text=src_text,
                    score=score,
                    source="short",
                    tags=[str(x) for x in list(row.get("tags") or []) if str(x).strip()],
                    metadata=dict(row.get("meta") or {}),
                )
            )

        vector_hits = self.vector_store.query_text(
            query_text=text,
            k=max(limit * 4, 12),
            filters=filt.get("vector_filters") if isinstance(filt.get("vector_filters"), dict) else None,
        )
        long_ranked: list[MemoryItem] = []
        for _id, score, meta in vector_hits:
            if float(score) < threshold:
                continue
            src_text = str(meta.get("text") or "")
            if not src_text:
                continue
            long_ranked.append(
                MemoryItem(
                    id=str(_id),
                    text=src_text,
                    score=_clamp01(float(score)),
                    source=str(meta.get("type") or "long"),
                    tags=[str(x) for x in list(meta.get("tags") or []) if str(x).strip()],
                    metadata=dict(meta or {}),
                )
            )

        combined = self._dedupe_items(short_ranked + long_ranked)
        combined = self._apply_retrieval_prompt(combined, threshold=threshold)
        combined.sort(key=lambda x: float(x.score), reverse=True)
        out = combined[:limit]
        log_json(
            LOGGER,
            "memory_retrieve",
            query_chars=len(text),
            requested_k=limit,
            threshold=round(float(threshold), 3),
            short_hits=len(short_ranked),
            vector_hits=len(long_ranked),
            returned=len(out),
        )
        return out

    def write_facts(self, facts: list[Fact]) -> None:
        wrote = 0
        for fact in list(facts or []):
            if not isinstance(fact, Fact):
                continue
            target = self.assistant_profile_store if fact.subject == "assistant" else self.user_profile_store
            profile_result = target.update_fact(fact, profile_id="default")
            status = str(profile_result.get("status") or "")
            needs_confirmation = bool(profile_result.get("needs_confirmation", False))

            # Persist fact as doc + vector (versioned, not silent overwrite).
            fact_text = f"{fact.subject}.{fact.key}={fact.value}"
            tags = ["fact", f"fact_{fact.op}", f"subject_{fact.subject}", f"key_{fact.key}"]
            if needs_confirmation:
                tags.append("needs_confirmation")

            doc = self.long_memory.add_doc(
                text=fact_text,
                meta={
                    "fact": fact.to_dict(),
                    "status": status,
                    "needs_confirmation": needs_confirmation,
                },
                source="fact",
                tags=tags,
                importance=0.86 if fact.op in {"update", "remove"} else 0.72,
                confidence=_clamp01(float(fact.confidence)),
            )
            self.vector_store.upsert(
                id=f"fact:{doc.id}",
                text=fact_text,
                embedding=None,
                metadata={
                    "type": "fact",
                    "topic": "profile",
                    "lang": "",
                    "user_id": "default",
                    "doc_id": doc.id,
                    "source": "fact",
                    "ts": time.time(),
                    "tags": tags,
                    "status": status,
                    "needs_confirmation": needs_confirmation,
                },
            )
            self.event_store.append(
                {
                    "type": "system",
                    "payload": {
                        "kind": "fact_write",
                        "fact": fact.to_dict(),
                        "status": status,
                        "needs_confirmation": needs_confirmation,
                        "doc_id": doc.id,
                    },
                    "tags": tags,
                }
            )
            wrote += 1
        if wrote:
            log_json(LOGGER, "memory_write_facts", facts=wrote)

    def build_context_pack(
        self,
        query: str,
        *,
        k: int = 8,
        filters: dict | None = None,
        tail_n: int = 10,
    ) -> dict[str, Any]:
        tail = self.short_memory.tail(max(1, int(tail_n)))
        retrieved = self.retrieve(query=query, k=k, filters=filters)
        user_summary = self._profile_summary(self.user_profile_store, "default")
        assistant_summary = self._profile_summary(self.assistant_profile_store, "default")
        return {
            "tail": tail,
            "retrieved": [x.to_dict() for x in retrieved],
            "short_summary": self.short_memory.rolling_summary(),
            "profile_summary": {
                "user": user_summary,
                "assistant": assistant_summary,
            },
        }

    @staticmethod
    def _dedupe_items(items: list[MemoryItem]) -> list[MemoryItem]:
        out: list[MemoryItem] = []
        for item in items:
            duplicate = False
            for kept in out:
                if _text_similarity(item.text, kept.text) >= 0.92:
                    duplicate = True
                    if item.score > kept.score:
                        out.remove(kept)
                        out.append(item)
                    break
            if not duplicate:
                out.append(item)
        return out

    @staticmethod
    def _profile_summary(store, profile_id: str) -> str:
        # store.history/get is key-based; collect current profile map directly.
        try:
            rows = store._data.get("profiles", {}).get(profile_id, {})  # noqa: SLF001
        except Exception:
            rows = {}
        parts = []
        for key, value in dict(rows or {}).items():
            if not isinstance(value, dict):
                continue
            if value.get("value") is None:
                continue
            short = str(value.get("value"))
            parts.append(f"{key}={short}")
            if len(parts) >= 12:
                break
        return "; ".join(parts)

    def _apply_retrieval_prompt(self, items: list[MemoryItem], *, threshold: float) -> list[MemoryItem]:
        prompt_text = ""
        try:
            prompt_text = str(self._prompt_registry.get_text("memory.cleanup") or "").lower()
        except Exception:
            prompt_text = ""
        if not prompt_text:
            return items

        out = list(items)
        if "low-confidence" in prompt_text or "low confidence" in prompt_text:
            out = [x for x in out if float(x.score) >= float(threshold)]
        if "remove duplicates" in prompt_text:
            out = self._dedupe_items(out)
        return out


def _normalize_role(value: str) -> str:
    role = str(value or "user").strip().lower()
    if role in {"assistant", "ai", "bot"}:
        return "assistant"
    if role in {"user", "human"}:
        return "user"
    return role or "user"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _text_similarity(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    if union <= 0:
        return 0.0
    jaccard = inter / union
    len_ratio = min(len(a), len(b)) / max(1, max(len(a), len(b)))
    return _clamp01(0.75 * jaccard + 0.25 * len_ratio)


def _tokens(value: str) -> set[str]:
    import re

    return {x.lower() for x in re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", str(value or "")) if x}
