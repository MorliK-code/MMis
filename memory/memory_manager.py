from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from config.settings import load_config
from metadata.entity_extractor import extract_entities, flatten_entity_tags, infer_topics_from_entities
from memory.event_store import EventStore
from memory.fact_extractor import Fact, FactExtractor, MODE_BALANCED
from memory.long_memory import LongMemory
from memory.profile_store import AssistantProfileStore, UserProfileStore
from memory.short_memory import ShortMemory
from memory.vector_store import VectorStore
from prompt_engine.prompt_registry import PromptRegistry
from utils.datetime_local import now_local_ts, parse_time_to_epoch, to_local_iso
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
    confidence: float = 0.0
    priority: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        confidence = self.confidence if float(self.confidence or 0.0) > 0.0 else float(self.score)
        priority = self.priority if float(self.priority or 0.0) > 0.0 else float(self.score)
        return {
            "id": self.id,
            "text": self.text,
            "score": float(self.score),
            "source": self.source,
            "tags": list(self.tags or []),
            "confidence": _clamp01(confidence),
            "priority": _clamp01(priority),
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
        facts_scope: str | None = None,
        include_pending_facts_in_retrieval: bool | None = None,
        confirmation_ttl_sec: int | None = None,
        confirmation_max_turn_distance: int = 3,
        prompt_registry: PromptRegistry | None = None,
    ):
        cfg = load_config()
        self.short_memory = short_memory if short_memory is not None else ShortMemory(limit=80, summary_trigger=60)
        self.long_memory = long_memory if long_memory is not None else LongMemory()
        self.vector_store = vector_store if vector_store is not None else VectorStore(dim=128)
        self.fact_extractor = fact_extractor if fact_extractor is not None else FactExtractor()
        self.user_profile_store = user_profile_store if user_profile_store is not None else UserProfileStore()
        self.assistant_profile_store = (
            assistant_profile_store if assistant_profile_store is not None else AssistantProfileStore()
        )
        self.event_store = event_store if event_store is not None else EventStore()
        self.retrieve_score_threshold = max(0.0, min(1.0, float(retrieve_score_threshold)))
        self._prompt_registry = prompt_registry or PromptRegistry()
        self.facts_scope = _normalize_facts_scope(
            facts_scope if facts_scope is not None else getattr(cfg, "memory_facts_scope", "user_only")
        )
        include_pending_default = bool(getattr(cfg, "memory_include_pending_facts_in_retrieval", False))
        self.include_pending_facts_in_retrieval = (
            bool(include_pending_facts_in_retrieval)
            if include_pending_facts_in_retrieval is not None
            else include_pending_default
        )
        self.confirmation_ttl_sec = max(
            1,
            int(
                confirmation_ttl_sec
                if confirmation_ttl_sec is not None
                else int(getattr(cfg, "memory_confirmation_ttl_sec", 300))
            ),
        )
        self.confirmation_max_turn_distance = max(1, int(confirmation_max_turn_distance))

    def ingest_message(
        self,
        role: str,
        text: str,
        thinking: str = "",
        metadata: dict | None = None,
        ts: float | str | None = None,
        **ids,
    ) -> None:
        content = str(text or "").strip()
        if not content:
            return
        role_norm = _normalize_role(role)
        meta = dict(metadata or {})
        now_iso = to_local_iso(ts, default=now_local_ts())
        tags = [str(x).strip().lower() for x in list(meta.get("tags") or []) if str(x).strip()]
        entities = _normalize_entities(meta.get("entities"))
        inferred_entities = extract_entities(content)
        entities = _merge_entity_maps(entities, inferred_entities)
        if entities:
            meta["entities"] = dict(entities)
            tags.extend(flatten_entity_tags(entities))
            tags.extend(infer_topics_from_entities(entities))
        tags = _dedupe_tags(tags)
        topics = _topic_list_from_tags(tags=tags, primary=str(meta.get("topic") or ""))
        lang = str(meta.get("lang") or "")
        intent = str(meta.get("intent") or "")
        emotion = str(meta.get("mood") or meta.get("emotion") or "")
        has_code = bool(meta.get("has_code") or ("has_code" in set(tags)))
        trace_id = str(ids.get("trace_id") or meta.get("trace_id") or "")
        model = str(ids.get("model") or meta.get("model") or "")
        source = str(ids.get("source") or meta.get("source") or "text")
        profile = str(ids.get("quality_profile") or meta.get("quality_profile") or MODE_BALANCED).upper()
        profile_id = str(ids.get("profile_id") or meta.get("user_id") or "default")
        topic = _primary_topic(meta=meta, tags=tags)

        event = self.event_store.append(
            {
                "ts": now_iso,
                "type": f"{role_norm}_message",
                "payload": {
                    "role": role_norm,
                    "text": content,
                    "thinking": thinking,
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
                "thinking": thinking,
                "ts": now_iso,
                "lang": lang,
                "intent": intent,
                "emotion": emotion,
                "tags": tags,
                "topic": topic,
                "topics": topics,
                "active_mode": str(meta.get("active_mode") or "").strip().lower(),
                "has_code": has_code,
                "persona_snapshot": _normalize_persona_snapshot(meta.get("persona_snapshot")),
                "meta": {
                    "trace_id": trace_id,
                    "source": source,
                    "model": model,
                    "conversation_id": ids.get("conversation_id"),
                    "turn_id": ids.get("turn_id"),
                    "entities": dict(entities),
                    "topic": topic,
                    "topics": topics,
                    "active_mode": str(meta.get("active_mode") or "").strip().lower(),
                    "persona_snapshot": _normalize_persona_snapshot(meta.get("persona_snapshot")),
                    "personality_id": str(meta.get("personality_id") or "").strip().lower(),
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
                "topic": topic,
                "user_id": profile_id,
                "event_id": event_id,
                "source": source,
                "ts": now_iso,
                "tags": tags,
                "entities": dict(entities),
            },
        )

        if role_norm == "assistant":
            importance = _clamp01(float(meta.get("importance") or 0.42))
            confidence = _clamp01(float(meta.get("confidence") or 0.62))
            doc = self.long_memory.add_doc(
                text=content,
                thinking=thinking,
                meta={
                    "event_id": event_id,
                    "role": role_norm,
                    "source": source,
                    "trace_id": trace_id,
                    "topic": topic,
                    "topics": topics,
                    "entities": dict(entities),
                    "profile_id": profile_id,
                    "active_mode": str(meta.get("active_mode") or "").strip().lower(),
                    "persona_snapshot": _normalize_persona_snapshot(meta.get("persona_snapshot")),
                },
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
                    "topic": topic,
                    "user_id": profile_id,
                    "doc_id": doc.id,
                    "event_id": event_id,
                    "source": "chat",
                    "ts": now_iso,
                    "tags": tags,
                    "entities": dict(entities),
                    "topics": topics,
                    "importance": doc.importance,
                    "confidence": doc.confidence,
                },
            )

        allow_fact_extract = (role_norm == "user") or (self.facts_scope != "user_only")
        facts: list[Fact] = []
        if allow_fact_extract:
            facts = self.fact_extractor.extract(
                text=content,
                metadata={"event_id": event_id, **meta},
                speaker=role_norm,
                mode=profile,
            )
        if facts:
            self.write_facts(facts, profile_id=profile_id)
        elif role_norm == "user" and _looks_like_confirmation_message(content):
            self._confirm_pending_with_context(
                profile_id=profile_id,
                event_id=event_id,
                confirmation_text=content,
            )

        if role_norm == "assistant":
            self._refresh_confirmation_context_from_assistant(
                profile_id=profile_id,
                assistant_text=content,
                assistant_event_id=event_id,
                turn_id=ids.get("turn_id"),
            )

    def retrieve(self, query: str, k: int = 8, filters: dict | None = None) -> list[MemoryItem]:
        text = str(query or "").strip()
        if not text:
            return []
        limit = max(1, int(k))
        filt = dict(filters or {})
        threshold = float(filt.get("score_threshold", self.retrieve_score_threshold))
        include_pending = bool(
            filt.get(
                "include_pending_facts",
                filt.get("include_pending_facts_in_retrieval", self.include_pending_facts_in_retrieval),
            )
        )

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
                    metadata={
                        **dict(row.get("meta") or {}),
                        "type": str(row.get("type") or "message"),
                        "role": str(row.get("role") or ""),
                        "ts": str(row.get("ts") or ""),
                    },
                    confidence=score,
                    priority=score,
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
            if not include_pending and _is_unconfirmed_fact_metadata(meta):
                continue
            src_text = str(meta.get("text") or "")
            if not src_text:
                continue
            confidence = _clamp01(float(meta.get("confidence") or score))
            priority = _clamp01(float(meta.get("priority") or score))
            long_ranked.append(
                MemoryItem(
                    id=str(_id),
                    text=src_text,
                    score=_clamp01(float(score)),
                    source=str(meta.get("type") or "long"),
                    tags=[str(x) for x in list(meta.get("tags") or []) if str(x).strip()],
                    metadata=dict(meta or {}),
                    confidence=confidence,
                    priority=priority,
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
            include_pending=bool(include_pending),
            short_hits=len(short_ranked),
            vector_hits=len(long_ranked),
            returned=len(out),
        )
        return out

    def write_facts(self, facts: list[Fact], *, profile_id: str = "default") -> None:
        wrote = 0
        indexed = 0
        for fact in list(facts or []):
            if not isinstance(fact, Fact):
                continue
            if self.facts_scope == "user_only" and str(fact.subject or "").strip().lower() != "user":
                continue
            target = self.assistant_profile_store if fact.subject == "assistant" else self.user_profile_store
            target_profile_id = profile_id if fact.subject == "user" else "default"
            profile_result = target.update_fact(fact, profile_id=target_profile_id)
            status = str(profile_result.get("status") or "")
            needs_confirmation = bool(profile_result.get("needs_confirmation", False))
            tags = ["fact", f"fact_{fact.op}", f"subject_{fact.subject}", f"key_{fact.key}"]
            if status:
                tags.append(f"status_{status}")
            if status in {"pending", "conflict_pending"}:
                tags.append("pending_fact")
            if status in {"confirmed", "confirmed_by_user"}:
                tags.append("confirmed_fact")
            if needs_confirmation:
                tags.append("needs_confirmation")
            tags = _dedupe_tags(tags)
            should_index_retrieval = status in {"confirmed", "confirmed_by_user"}
            if should_index_retrieval:
                fact_text = f"{fact.subject}.{fact.key}={fact.value}"
                doc = self.long_memory.add_doc(
                    text=fact_text,
                    meta={
                        "fact": fact.to_dict(),
                        "status": status,
                        "needs_confirmation": needs_confirmation,
                        "profile_id": target_profile_id,
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
                        "user_id": target_profile_id,
                        "doc_id": doc.id,
                        "source": "fact",
                        "ts": now_local_ts(),
                        "tags": tags,
                        "status": status,
                        "needs_confirmation": needs_confirmation,
                        "confidence": _clamp01(float(fact.confidence)),
                        "priority": _clamp01(float(fact.confidence)),
                    },
                )
                indexed += 1
            self.event_store.append(
                {
                    "type": "system",
                    "payload": {
                        "kind": "fact_write",
                        "fact": fact.to_dict(),
                        "status": status,
                        "needs_confirmation": needs_confirmation,
                        "indexed": bool(should_index_retrieval),
                        "profile_id": target_profile_id,
                    },
                    "tags": tags,
                }
            )
            wrote += 1
        if wrote:
            log_json(LOGGER, "memory_write_facts", facts=wrote, indexed=indexed)

    def _persist_confirmed_pending(
        self,
        *,
        rows: list[dict[str, Any]],
        subject: str,
        profile_id: str,
        event_id: str,
    ) -> None:
        for row in list(rows or []):
            key = str(row.get("key") or "").strip()
            if not key:
                continue
            value = row.get("value")
            text = f"{subject}.{key}={value}"
            tags = _dedupe_tags(
                [
                    "fact",
                    "fact_confirm",
                    "confirmed_fact",
                    "status_confirmed_by_user",
                    f"subject_{subject}",
                    f"key_{key}",
                ]
            )
            confidence = _clamp01(float(row.get("confidence") or 0.8))
            doc = self.long_memory.add_doc(
                text=text,
                source="fact",
                tags=tags,
                importance=0.8,
                confidence=confidence,
                meta={
                    "fact": {
                        "subject": subject,
                        "key": key,
                        "value": value,
                        "op": str(row.get("op") or "add"),
                    },
                    "status": "confirmed_by_user",
                    "profile_id": profile_id,
                    "pending_confirmed_from_event": event_id,
                },
            )
            self.vector_store.upsert(
                id=f"fact:{doc.id}",
                text=text,
                embedding=None,
                metadata={
                    "type": "fact",
                    "topic": "profile",
                    "lang": "",
                    "user_id": profile_id,
                    "doc_id": doc.id,
                    "source": "fact",
                    "ts": now_local_ts(),
                    "tags": tags,
                    "status": "confirmed_by_user",
                    "needs_confirmation": False,
                    "confidence": confidence,
                    "priority": confidence,
                },
            )
            self.event_store.append(
                {
                    "type": "system",
                    "payload": {
                        "kind": "fact_confirm",
                        "subject": subject,
                        "profile_id": profile_id,
                        "event_id": event_id,
                        "fact": dict(row),
                        "doc_id": doc.id,
                    },
                    "tags": tags,
                }
            )

    def _confirm_pending_with_context(self, *, profile_id: str, event_id: str, confirmation_text: str) -> None:
        context = dict(self.user_profile_store.get_confirmation_context(profile_id=profile_id) or {})
        if not context:
            log_json(
                LOGGER,
                "confirm_skipped_no_context",
                profile_id=profile_id,
                event_id=event_id,
                reason="missing_context",
            )
            return

        context_age_sec = max(0.0, time.time() - float(context.get("created_at") or 0.0))
        ttl_sec = max(1, int(context.get("ttl_sec") or self.confirmation_ttl_sec))
        assistant_event_id = str(context.get("assistant_event_id") or "").strip()
        turn_distance = self._distance_from_event(assistant_event_id)
        max_turn_distance = max(1, int(context.get("max_turn_distance") or self.confirmation_max_turn_distance))
        key_list = [str(x).strip().lower() for x in list(context.get("keys") or []) if str(x).strip()]
        expected_values = {
            str(k).strip().lower(): v
            for k, v in dict(context.get("expected_values") or {}).items()
            if str(k).strip()
        }

        reason = ""
        if context_age_sec > float(ttl_sec):
            reason = "expired"
        elif not assistant_event_id:
            reason = "missing_assistant_event"
        elif turn_distance is None or int(turn_distance) > int(max_turn_distance):
            reason = "turn_distance"
        elif not key_list:
            reason = "missing_keys"

        if reason:
            self.user_profile_store.clear_confirmation_context(profile_id=profile_id)
            log_json(
                LOGGER,
                "confirm_skipped_no_context",
                profile_id=profile_id,
                event_id=event_id,
                reason=reason,
                age_sec=round(context_age_sec, 3),
                ttl_sec=ttl_sec,
                turn_distance=turn_distance,
            )
            return

        confirmed_rows = self.user_profile_store.confirm_pending(
            profile_id=profile_id,
            limit=max(1, len(key_list)),
            keys=key_list,
            expected_values=expected_values,
        )
        self.user_profile_store.clear_confirmation_context(profile_id=profile_id)
        if not confirmed_rows:
            log_json(
                LOGGER,
                "confirm_skipped_no_context",
                profile_id=profile_id,
                event_id=event_id,
                reason="candidate_mismatch",
                keys=key_list,
            )
            return
        self._persist_confirmed_pending(
            rows=confirmed_rows,
            subject="user",
            profile_id=profile_id,
            event_id=event_id,
        )
        log_json(
            LOGGER,
            "confirm_pending_applied",
            profile_id=profile_id,
            event_id=event_id,
            confirmed=len(confirmed_rows),
            text_chars=len(str(confirmation_text or "")),
        )

    def _refresh_confirmation_context_from_assistant(
        self,
        *,
        profile_id: str,
        assistant_text: str,
        assistant_event_id: str,
        turn_id: Any,
    ) -> None:
        pending = dict(self.user_profile_store.get_pending_facts(profile_id=profile_id) or {})
        if not pending:
            self.user_profile_store.clear_confirmation_context(profile_id=profile_id)
            return
        if not _looks_like_confirmation_prompt(assistant_text):
            return

        candidates = _pending_candidates_from_map(pending)
        if not candidates:
            return
        key_hints = _extract_fact_key_hints(assistant_text)
        if key_hints:
            keys = [key for key in candidates.keys() if key in key_hints]
        elif len(candidates) == 1:
            keys = list(candidates.keys())
        else:
            # If key is ambiguous, skip setting context to avoid accidental confirms.
            return
        if not keys:
            return

        expected_values = {key: candidates.get(key) for key in keys}
        explicit_values = _extract_expected_values_from_prompt(assistant_text, keys=set(keys))
        for key, value in explicit_values.items():
            expected_values[key] = value
        context = {
            "assistant_event_id": assistant_event_id,
            "created_at": float(time.time()),
            "ttl_sec": int(self.confirmation_ttl_sec),
            "max_turn_distance": int(self.confirmation_max_turn_distance),
            "keys": list(keys),
            "expected_values": dict(expected_values),
            "turn_id": _safe_int(turn_id, 0),
        }
        self.user_profile_store.set_confirmation_context(profile_id=profile_id, context=context)
        log_json(
            LOGGER,
            "confirm_context_set",
            profile_id=profile_id,
            assistant_event_id=assistant_event_id,
            keys=keys,
        )

    def _distance_from_event(self, event_id: str) -> int | None:
        key = str(event_id or "").strip()
        if not key:
            return None
        scan = max(20, int(getattr(self.short_memory, "limit", 80)))
        rows = self.short_memory.tail(scan)
        if not rows:
            return None
        idx = -1
        for i in range(len(rows) - 1, -1, -1):
            if str(rows[i].get("id") or "").strip() == key:
                idx = i
                break
        if idx < 0:
            return None
        return max(0, len(rows) - 1 - idx)

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
        assistant_summary = ""
        if self.facts_scope != "user_only":
            assistant_summary = self._profile_summary(self.assistant_profile_store, "default")
        return {
            "tail": tail,
            "retrieved": [x.to_dict() for x in retrieved],
            "short_summary": self.short_memory.rolling_summary(),
            "short_summary_meta": self.short_memory.rolling_summary_meta(),
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
        confirmed_rows: dict[str, Any] = {}
        try:
            confirmed_rows = dict(store.get_confirmed_facts(profile_id=profile_id))
        except Exception:
            confirmed_rows = {}
        parts = []
        for key, value in dict(confirmed_rows or {}).items():
            if not isinstance(value, dict):
                continue
            status = str(value.get("status") or "confirmed").strip().lower()
            if status not in {"confirmed", "confirmed_by_user"}:
                continue
            short = str(value.get("value"))
            parts.append(f"{key}={short}")
            if len(parts) >= 12:
                break
        if parts:
            return "; ".join(parts)

        # Legacy fallback if confirmed bucket is not populated yet.
        try:
            rows = store._data.get("profiles", {}).get(profile_id, {})  # noqa: SLF001
        except Exception:
            rows = {}
        for key, value in dict(rows or {}).items():
            if not isinstance(value, dict):
                continue
            if value.get("value") is None:
                continue
            if bool(value.get("needs_confirmation", False)):
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


def _dedupe_tags(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        item = str(value or "").strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_entities(payload: Any) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, list[str]] = {}
    for raw_key, raw_values in payload.items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        values: list[str] = []
        seen: set[str] = set()
        for raw_value in list(raw_values or []):
            item = str(raw_value or "").strip()
            if not item:
                continue
            low = item.lower()
            if low in seen:
                continue
            seen.add(low)
            values.append(item)
        if values:
            out[key] = values
    return out


def _merge_entity_maps(base: dict[str, list[str]], extra: dict[str, list[str]]) -> dict[str, list[str]]:
    out = {str(k): list(v) for k, v in dict(base or {}).items()}
    for raw_key, raw_values in dict(extra or {}).items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        current = [str(x).strip() for x in list(out.get(key) or []) if str(x).strip()]
        seen = {x.lower() for x in current}
        for raw_value in list(raw_values or []):
            item = str(raw_value or "").strip()
            if not item:
                continue
            low = item.lower()
            if low in seen:
                continue
            seen.add(low)
            current.append(item)
        if current:
            out[key] = current
    return out


def _primary_topic(*, meta: dict[str, Any], tags: list[str]) -> str:
    raw = str(meta.get("topic") or "").strip().lower()
    if raw:
        return raw.replace("topic_", "", 1)
    for tag in list(tags or []):
        item = str(tag or "").strip().lower()
        if item.startswith("topic_"):
            return item.replace("topic_", "", 1)
    return ""


def _topic_list_from_tags(*, tags: list[str], primary: str = "") -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    raw_primary = str(primary or "").strip().lower()
    if raw_primary:
        raw_primary = raw_primary.replace("topic_", "", 1)
        seen.add(raw_primary)
        out.append(raw_primary)
    for tag in list(tags or []):
        item = str(tag or "").strip().lower()
        if not item.startswith("topic_"):
            continue
        token = item.replace("topic_", "", 1)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _normalize_persona_snapshot(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    mood = str(payload.get("mood") or "").strip().lower()
    if mood:
        out["mood"] = mood
    traits_raw = dict(payload.get("traits") or {})
    traits: dict[str, float] = {}
    for key in ("warmth", "sarcasm", "verbosity", "strictness", "teasing", "empathy"):
        if key not in traits_raw:
            continue
        try:
            traits[key] = _clamp01(float(traits_raw.get(key)))
        except Exception:
            continue
    if traits:
        out["traits"] = traits
    character_id = str(payload.get("character_id") or "").strip().lower()
    if character_id:
        out["character_id"] = character_id
    active_mode = str(payload.get("active_mode") or "").strip().lower()
    if active_mode:
        out["active_mode"] = active_mode
    return out


def _looks_like_confirmation_message(text: str) -> bool:
    src = str(text or "").strip().lower()
    if not src or len(src) > 64 or "?" in src:
        return False
    compact = re.sub(r"[^a-z0-9\u0400-\u04ff' ]+", " ", src)
    compact = re.sub(r"\s+", " ", compact).strip()
    if not compact:
        return False
    markers = {
        "yes",
        "yep",
        "yup",
        "correct",
        "exactly",
        "right",
        "true",
        "affirmative",
        "da",
        "verno",
        "tochno",
        "\u0434\u0430",
        "\u0432\u0435\u0440\u043d\u043e",
        "\u0442\u043e\u0447\u043d\u043e",
    }
    if compact in markers:
        return True
    phrases = {
        "that's right",
        "you are right",
        "yes correct",
        "yes exactly",
        "\u0434\u0430 \u0432\u0435\u0440\u043d\u043e",
    }
    if compact in phrases:
        return True
    return False


def _looks_like_confirmation_prompt(text: str) -> bool:
    src = str(text or "").strip().lower()
    if not src or len(src) > 400:
        return False
    if "?" not in src:
        return False
    markers = (
        "correct",
        "right",
        "is that",
        "am i right",
        "confirm",
        "yes or no",
        "\u0432\u0435\u0440\u043d\u043e",
        "\u043f\u0440\u0430\u0432\u0438\u043b\u044c\u043d\u043e",
        "\u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0434\u0438",
        "\u044d\u0442\u043e \u0442\u0430\u043a",
        "\u0434\u0430 \u0438\u043b\u0438 \u043d\u0435\u0442",
    )
    return any(marker in src for marker in markers)


def _extract_fact_key_hints(text: str) -> set[str]:
    src = str(text or "").lower()
    hints: set[str] = set()
    patterns = {
        "name": (r"\bname\b", r"\bcalled\b", r"\b\u0438\u043c\u044f\b", r"\b\u0437\u043e\u0432\u0443\u0442\b"),
        "age": (r"\bage\b", r"\byears old\b", r"\b\u0432\u043e\u0437\u0440\u0430\u0441\u0442\b", r"\b\u043b\u0435\u0442\b"),
        "birth_year": (
            r"\bbirth year\b",
            r"\bborn in\b",
            r"\bborn\b",
            r"\b\u0433\u043e\u0434 \u0440\u043e\u0436\u0434\u0435\u043d\u0438\u044f\b",
            r"\b\u0440\u043e\u0434\u0438\u043b",
        ),
        "location": (r"\blocation\b", r"\bfrom\b", r"\blive\b", r"\b\u0433\u043e\u0440\u043e\u0434\b", r"\b\u0436\u0438\u0432\u0443\b"),
        "likes": (r"\blike\b", r"\blikes\b", r"\b\u043b\u044e\u0431\u043b", r"\b\u043d\u0440\u0430\u0432\u0438\u0442\u0441\u044f\b"),
        "dislikes": (r"\bdislike\b", r"\bhate\b", r"\b\u043d\u0435 \u043b\u044e\u0431\u043b", r"\b\u043d\u0435\u043d\u0430\u0432\u0438\u0436\u0443\b"),
        "device": (r"\bdevice\b", r"\biphone\b", r"\bandroid\b", r"\bwindows\b", r"\blinux\b", r"\bmac\b"),
    }
    for key, rows in patterns.items():
        for pattern in rows:
            if re.search(pattern, src):
                hints.add(key)
                break
    return hints


def _extract_expected_values_from_prompt(text: str, *, keys: set[str]) -> dict[str, Any]:
    src = str(text or "").strip()
    out: dict[str, Any] = {}
    if not src:
        return out
    if "birth_year" in keys:
        m = re.search(r"\b(19\d{2}|20\d{2})\b", src)
        if m:
            out["birth_year"] = str(m.group(1))
    if "age" in keys:
        m = re.search(r"\b(\d{1,3})\s*(?:years?\s*old|\u043b\u0435\u0442)?\b", src, re.I)
        if m:
            out["age"] = str(m.group(1))
    if "name" in keys:
        m = re.search(r"\b(?:name is|called|zovut|\u0437\u043e\u0432\u0443\u0442)\s+([A-Za-z\u0400-\u04ff' -]{2,32})", src, re.I)
        if m:
            out["name"] = str(m.group(1)).strip()
    if "device" in keys:
        m = re.search(r"\b(iphone|android|windows|linux|mac(?:book|os)?)\b", src, re.I)
        if m:
            out["device"] = str(m.group(1)).strip().lower()
    return out


def _pending_candidates_from_map(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_key, raw_items in dict(payload or {}).items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        items = [dict(x) for x in list(raw_items or []) if isinstance(x, dict)]
        if not items:
            continue
        items.sort(
            key=lambda x: (
                int(x.get("count") or 0),
                float(x.get("confidence") or 0.0),
                parse_time_to_epoch(x.get("updated_at"), 0.0),
            ),
            reverse=True,
        )
        candidate = dict(items[0] or {})
        if "value" not in candidate:
            continue
        out[key] = candidate.get("value")
    return out


def _is_unconfirmed_fact_metadata(metadata: dict[str, Any]) -> bool:
    meta = dict(metadata or {})
    if str(meta.get("type") or "").strip().lower() != "fact":
        return False
    status = str(meta.get("status") or "").strip().lower()
    if not status:
        return True
    return status not in {"confirmed", "confirmed_by_user"}


def _normalize_facts_scope(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw == "user_only":
        return "user_only"
    return "all"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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

    return {x.lower() for x in re.findall(r"[A-Za-z\u0400-\u04ff0-9_]+", str(value or "")) if x}

