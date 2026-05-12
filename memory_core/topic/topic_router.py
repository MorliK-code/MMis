from __future__ import annotations

import re
import time
from typing import Any, Callable

from memory_core.processors.episode_processor import EpisodeProcessor
from memory_core.topic.topic_models import TopicRouteDecision, TopicThread
from memory_core.topic.topic_store import TopicStore


class TopicRouter:
    _SOFT_SWITCH_MARKERS = (
        "\u0442\u0435\u043f\u0435\u0440\u044c \u043f\u0440\u043e",
        "\u0430 \u0442\u0435\u043f\u0435\u0440\u044c \u043f\u0440\u043e",
        "\u0434\u0430\u0432\u0430\u0439 \u0442\u0435\u043f\u0435\u0440\u044c \u043f\u0440\u043e",
        "\u043f\u0435\u0440\u0435\u0439\u0434\u0435\u043c \u043a",
        "\u043f\u0435\u0440\u0435\u0439\u0434\u0451\u043c \u043a",
        "\u0441\u043c\u0435\u043d\u0438\u043c \u0442\u0435\u043c\u0443",
        "\u0434\u0440\u0443\u0433\u043e\u0439 \u0432\u043e\u043f\u0440\u043e\u0441",
        "\u043e\u0442\u0434\u0435\u043b\u044c\u043d\u044b\u0439 \u0432\u043e\u043f\u0440\u043e\u0441",
        "another topic",
        "new topic",
        "separate question",
        "switch topic",
    )
    _CONTINUATION_PREFIXES = (
        "\u0430 ",
        "\u0438 ",
        "\u043d\u043e ",
        "\u043d\u0443 ",
        "\u0434\u0430\u0432\u0430\u0439 ",
        "\u0434\u0430\u0432\u0430\u0439 \u0435\u0449\u0435",
        "\u043f\u043e\u0434\u0440\u043e\u0431\u043d\u0435\u0435",
        "\u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0430\u0439",
        "\u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0438\u043c",
        "\u0430 \u0435\u0441\u043b\u0438",
        "\u0430 \u043a\u0430\u043a",
        "\u0430 \u0447\u0442\u043e",
        "\u0447\u0442\u043e \u0435\u0441\u043b\u0438",
        "and ",
        "also ",
        "then ",
        "continue",
        "go on",
        "what about",
    )
    _RECENT_THREAD_TTL_SEC = 30 * 60
    _NEW_TOPIC_MARKERS = (
        "\u0434\u0440\u0443\u0433\u0430\u044f \u0442\u0435\u043c\u0430",
        "\u043d\u043e\u0432\u044b\u0439 \u0432\u043e\u043f\u0440\u043e\u0441",
        "\u0435\u0449\u0435 \u0432\u043e\u043f\u0440\u043e\u0441",
        "\u0435\u0449\u0451 \u0432\u043e\u043f\u0440\u043e\u0441",
        "\u043e\u0442\u0434\u0435\u043b\u044c\u043d\u043e",
        "\u043a\u0441\u0442\u0430\u0442\u0438",
    )
    _RETURN_MARKERS = (
        "\u0432\u0435\u0440\u043d\u0435\u043c\u0441\u044f \u043a",
        "\u0432\u0435\u0440\u043d\u0451\u043c\u0441\u044f \u043a",
        "\u043f\u043e \u043f\u043e\u0432\u043e\u0434\u0443",
        "\u043d\u0430\u0441\u0447\u0435\u0442",
        "\u043d\u0430\u0441\u0447\u0451\u0442",
        "\u0447\u0442\u043e \u0442\u0430\u043c \u0441",
        "\u043e\u0431\u0440\u0430\u0442\u043d\u043e \u043a",
    )

    def __init__(
        self,
        topic_store: TopicStore,
        *,
        hint_extractor: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.topic_store = topic_store
        self._hint_extractor = hint_extractor or EpisodeProcessor().detect_topic_hints

    def route_turn(
        self,
        *,
        text: str,
        visible_chat_id: str,
        workspace_id: str,
        session_id: str,
        current_state: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> TopicRouteDecision:
        clean_text = str(text or "").strip()
        state = dict(current_state or {})
        meta_map = dict(meta or {})
        current_thread_id = str(
            state.get("active_topic_thread_id")
            or state.get("topic_thread_id")
            or meta_map.get("topic_thread_id")
            or ""
        ).strip()
        threads = self.topic_store.list_threads(
            visible_chat_id=visible_chat_id,
            workspace_id=workspace_id,
            session_id=session_id,
            limit=32,
        )
        hints = self._safe_hints(clean_text)
        hinted_key = self._derive_topic_key(clean_text, hints)
        explicit_return = self._extract_explicit_target(clean_text, self._RETURN_MARKERS)
        explicit_new = self._contains_marker(clean_text, self._NEW_TOPIC_MARKERS + self._SOFT_SWITCH_MARKERS)
        current_thread = next((item for item in threads if item.thread_id == current_thread_id), None)

        if not threads:
            return self._create_thread_decision(
                visible_chat_id=visible_chat_id,
                workspace_id=workspace_id,
                session_id=session_id,
                topic_key=hinted_key,
                title=self._derive_title(clean_text, hints),
                seed_text=clean_text,
                tags=hints,
                reason="bootstrap",
                score=1.0,
            )

        ranked = self._score_threads(
            clean_text,
            hints=hints,
            hinted_key=hinted_key,
            threads=threads,
            current_thread_id=current_thread_id,
            active_task=dict(state.get("active_task") or {}),
        )
        best_thread = ranked[0][0] if ranked else None
        best_score = ranked[0][1] if ranked else 0.0
        current_score = next(
            (score for thread, score in ranked if current_thread is not None and thread.thread_id == current_thread.thread_id),
            0.0,
        )
        related_ids = [thread.thread_id for thread, score in ranked[1:3] if score >= 0.45]
        looks_like_continuation = self._looks_like_continuation(clean_text)
        recent_current = self._is_recent_thread(current_thread)

        if explicit_return:
            matched = self._match_explicit_target(explicit_return, threads)
            if matched is not None:
                self.topic_store.touch_thread(
                    matched.thread_id,
                    summary=self._route_summary_for_existing_thread(matched, clean_text),
                    tags=hints,
                    related_thread_ids=related_ids,
                    metadata=self._route_metadata_for_existing_thread(matched, clean_text, "explicit_return", max(best_score, 0.86)),
                    status="active",
                )
                return TopicRouteDecision(
                    thread_id=matched.thread_id,
                    topic_key=matched.topic_key,
                    title=matched.title,
                    reason="explicit_return",
                    score=max(best_score, 0.86),
                    related_thread_ids=related_ids,
                    metadata={"explicit_target": explicit_return},
                )

        if explicit_new:
            return self._create_thread_decision(
                visible_chat_id=visible_chat_id,
                workspace_id=workspace_id,
                session_id=session_id,
                topic_key=hinted_key,
                title=self._derive_title(clean_text, hints),
                seed_text=clean_text,
                tags=hints,
                related_thread_ids=related_ids,
                reason="explicit_new_topic",
                score=0.0,
            )

        if self._is_short_followup(clean_text) and current_thread is not None:
            self.topic_store.touch_thread(
                current_thread.thread_id,
                summary=self._route_summary_for_existing_thread(current_thread, clean_text),
                tags=hints,
                metadata=self._route_metadata_for_existing_thread(current_thread, clean_text, "short_followup", max(best_score, 0.74)),
                status="active",
            )
            return TopicRouteDecision(
                thread_id=current_thread.thread_id,
                topic_key=current_thread.topic_key,
                title=current_thread.title,
                reason="short_followup",
                score=max(best_score, 0.74),
                related_thread_ids=related_ids,
            )

        if (
            current_thread is not None
            and best_thread is not None
            and best_thread.thread_id != current_thread.thread_id
            and best_score >= 0.88
            and current_score <= 0.18
            and not looks_like_continuation
            and not recent_current
            and not dict(state.get("active_task") or {})
        ):
            self.topic_store.touch_thread(
                best_thread.thread_id,
                summary=self._route_summary_for_existing_thread(best_thread, clean_text),
                tags=hints,
                related_thread_ids=related_ids,
                metadata=self._route_metadata_for_existing_thread(best_thread, clean_text, "semantic_switch_existing", best_score),
                status="active",
            )
            return TopicRouteDecision(
                thread_id=best_thread.thread_id,
                topic_key=best_thread.topic_key,
                title=best_thread.title,
                reason="semantic_switch_existing",
                score=best_score,
                related_thread_ids=related_ids,
            )

        if best_thread is not None and best_score >= 0.72 and (
            current_thread is None or best_thread.thread_id == current_thread.thread_id
        ):
            self.topic_store.touch_thread(
                best_thread.thread_id,
                summary=self._route_summary_for_existing_thread(best_thread, clean_text),
                tags=hints,
                related_thread_ids=related_ids,
                metadata=self._route_metadata_for_existing_thread(best_thread, clean_text, "semantic_match", best_score),
                status="active",
            )
            return TopicRouteDecision(
                thread_id=best_thread.thread_id,
                topic_key=best_thread.topic_key,
                title=best_thread.title,
                reason="semantic_match",
                score=best_score,
                related_thread_ids=related_ids,
            )

        if current_thread is not None:
            if current_score >= 0.5:
                self.topic_store.touch_thread(
                    current_thread.thread_id,
                    summary=self._route_summary_for_existing_thread(current_thread, clean_text),
                    tags=hints,
                    related_thread_ids=related_ids,
                    metadata=self._route_metadata_for_existing_thread(current_thread, clean_text, "prefer_current_topic", current_score),
                    status="active",
                )
                return TopicRouteDecision(
                    thread_id=current_thread.thread_id,
                    topic_key=current_thread.topic_key,
                    title=current_thread.title,
                    reason="prefer_current_topic",
                    score=current_score,
                    related_thread_ids=related_ids,
                )
            if current_score >= 0.40 or looks_like_continuation or recent_current:
                reason = "continue_current_topic"
                score = current_score
                if current_score >= 0.40 and not looks_like_continuation and not recent_current:
                    reason = "prefer_current_topic"
                elif looks_like_continuation:
                    score = max(score, 0.68)
                elif recent_current:
                    score = max(score, 0.62)
                self.topic_store.touch_thread(
                    current_thread.thread_id,
                    summary=self._route_summary_for_existing_thread(current_thread, clean_text),
                    tags=hints,
                    related_thread_ids=related_ids,
                    metadata=self._route_metadata_for_existing_thread(current_thread, clean_text, reason, score),
                    status="active",
                )
                return TopicRouteDecision(
                    thread_id=current_thread.thread_id,
                    topic_key=current_thread.topic_key,
                    title=current_thread.title,
                    reason=reason,
                    score=score,
                    related_thread_ids=related_ids,
                )

        return self._create_thread_decision(
            visible_chat_id=visible_chat_id,
            workspace_id=workspace_id,
            session_id=session_id,
            topic_key=hinted_key,
            title=self._derive_title(clean_text, hints),
            seed_text=clean_text,
            tags=hints,
            related_thread_ids=related_ids,
            reason="new_topic_low_match",
            score=best_score,
        )

    def _create_thread_decision(
        self,
        *,
        visible_chat_id: str,
        workspace_id: str,
        session_id: str,
        topic_key: str,
        title: str,
        seed_text: str,
        tags: list[str],
        related_thread_ids: list[str] | None = None,
        reason: str,
        score: float,
    ) -> TopicRouteDecision:
        now = time.time()
        fast_summary = self._build_fast_summary(seed_text, fallback_title=title or topic_key)
        thread = TopicThread.create(
            visible_chat_id=visible_chat_id,
            workspace_id=workspace_id,
            session_id=session_id,
            topic_key=topic_key,
            title=title,
            tags=tags,
            related_thread_ids=related_thread_ids,
            metadata={
                "created_by": "topic_router",
                "last_route_reason": reason,
                "last_route_score": float(score),
                "last_user_text": str(seed_text or "").strip(),
                "recent_turn_texts": self._merge_recent_turn_texts([], seed_text),
                "summary_updated_at": now,
                "summary_build_version": 0,
                "summary_source": "topic_router_fastpath",
                "summary_stale": True,
                "summary_refresh_reason": reason,
                "summary_input_updated_at": now,
                "topic_status": "active",
                "last_meaningful_activity_at": now,
            },
        )
        thread.summary = fast_summary
        self.topic_store.create_thread(thread)
        return TopicRouteDecision(
            thread_id=thread.thread_id,
            topic_key=thread.topic_key,
            title=thread.title,
            reason=reason,
            score=float(score),
            is_new_thread=True,
            related_thread_ids=list(thread.related_thread_ids or []),
        )

    def _score_threads(
        self,
        text: str,
        *,
        hints: list[str],
        hinted_key: str,
        threads: list[TopicThread],
        current_thread_id: str,
        active_task: dict[str, Any],
    ) -> list[tuple[TopicThread, float]]:
        active_task_text = " ".join(
            str(active_task.get(key) or "")
            for key in ("topic", "summary_short", "current_goal")
        ).strip()
        text_tokens = self._tokenize(text)
        hint_tokens = set(str(item).strip().lower() for item in list(hints or []) if str(item).strip())
        ranked: list[tuple[TopicThread, float]] = []
        now = time.time()
        for thread in threads:
            base = " ".join(
                part
                for part in (
                    thread.topic_key,
                    thread.title,
                    thread.summary,
                    " ".join(thread.tags),
                )
                if str(part or "").strip()
            )
            thread_tokens = self._tokenize(base)
            semantic_similarity = self._overlap_ratio(text_tokens, thread_tokens)
            tag_overlap = self._overlap_ratio(hint_tokens, set(str(item).lower() for item in list(thread.tags or [])))
            keyword_overlap = 1.0 if hinted_key and hinted_key == str(thread.topic_key or "").strip().lower() else 0.0
            recency_bonus = max(0.0, min(1.0, 1.0 - ((now - float(thread.updated_at or now)) / 86400.0)))
            active_task_bonus = 0.0
            if active_task_text and self._overlap_ratio(self._tokenize(active_task_text), thread_tokens) > 0.0:
                active_task_bonus = 1.0
            score = (
                semantic_similarity * 0.55
                + tag_overlap * 0.15
                + keyword_overlap * 0.10
                + recency_bonus * 0.10
                + active_task_bonus * 0.10
            )
            if thread.thread_id == current_thread_id:
                score += 0.08
            ranked.append((thread, min(1.0, max(0.0, score))))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _match_explicit_target(self, needle: str, threads: list[TopicThread]) -> TopicThread | None:
        target_tokens = self._tokenize(needle)
        if not target_tokens:
            return None
        matches: list[tuple[TopicThread, float]] = []
        for thread in threads:
            haystack = self._tokenize(" ".join([thread.topic_key, thread.title, thread.summary, " ".join(thread.tags)]))
            score = self._overlap_ratio(target_tokens, haystack)
            if score > 0.0:
                matches.append((thread, score))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[0][0] if matches else None

    def _safe_hints(self, text: str) -> list[str]:
        try:
            hints = list(self._hint_extractor(text) or [])
        except Exception:
            hints = []
        seen: set[str] = set()
        result: list[str] = []
        for item in hints:
            value = str(item or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result[:8]

    @staticmethod
    def _derive_title(text: str, hints: list[str]) -> str:
        if hints:
            return " / ".join(item.replace("_", " ") for item in hints[:3])
        words = [word for word in str(text or "").strip().split() if word]
        title = " ".join(words[:6]).strip()
        return title if len(title) <= 80 else title[:77].rstrip() + "..."

    @staticmethod
    def _derive_topic_key(text: str, hints: list[str]) -> str:
        if hints:
            return str(hints[0]).strip().lower()
        tokens = [token for token in TopicRouter._tokenize(text) if len(token) >= 4]
        if not tokens:
            return "general"
        return "_".join(tokens[:3])[:48].strip("_") or "general"

    @staticmethod
    def _extract_explicit_target(text: str, markers: tuple[str, ...]) -> str:
        lowered = str(text or "").strip().lower()
        for marker in markers:
            idx = lowered.find(marker)
            if idx < 0:
                continue
            tail = lowered[idx + len(marker):].strip(" .,:;!?-")
            if tail:
                return tail
        return ""

    @staticmethod
    def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
        lowered = str(text or "").strip().lower()
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _is_short_followup(text: str) -> bool:
        words = [word for word in str(text or "").strip().split() if word]
        if len(words) > 4:
            return False
        lowered = str(text or "").strip().lower()
        if any(marker in lowered for marker in TopicRouter._NEW_TOPIC_MARKERS):
            return False
        return bool(words)

    @staticmethod
    def _looks_like_continuation(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        if any(lowered.startswith(marker) for marker in TopicRouter._CONTINUATION_PREFIXES):
            return True
        words = [word for word in lowered.split() if word]
        if len(words) <= 6 and lowered.endswith("?"):
            return True
        if len(words) <= 3:
            return True
        return False

    @staticmethod
    def _is_recent_thread(thread: TopicThread | None) -> bool:
        if thread is None:
            return False
        updated_at = float(thread.updated_at or 0.0)
        if updated_at <= 0:
            return False
        return (time.time() - updated_at) <= float(TopicRouter._RECENT_THREAD_TTL_SEC)

    @staticmethod
    def _build_fast_summary(text: str, *, fallback_title: str = "") -> str:
        clean = " ".join(str(text or "").strip().split())
        if clean:
            return ("Focus: " + clean[:220]).strip()[:480]
        return str(fallback_title or "").strip()[:480]

    @staticmethod
    def _merge_recent_turn_texts(existing: list[str] | None, text: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in list(existing or []) + [text]:
            clean = " ".join(str(item or "").strip().split())
            if not clean or clean in seen:
                continue
            seen.add(clean)
            result.append(clean[:180])
        return result[-4:]

    def _route_summary_for_existing_thread(self, thread: TopicThread, text: str) -> str | None:
        metadata = dict(thread.metadata or {})
        build_version = int(metadata.get("summary_build_version") or 0)
        if build_version >= 1 and str(thread.summary or "").strip():
            return None
        return self._build_fast_summary(text, fallback_title=thread.title or thread.topic_key)

    def _route_metadata_for_existing_thread(
        self,
        thread: TopicThread,
        text: str,
        reason: str,
        score: float,
    ) -> dict[str, Any]:
        metadata = dict(thread.metadata or {})
        now = time.time()
        payload = {
            "last_route_reason": reason,
            "last_route_score": float(score),
            "last_user_text": str(text or "").strip(),
            "recent_turn_texts": self._merge_recent_turn_texts(metadata.get("recent_turn_texts"), text),
            "summary_stale": True,
            "summary_refresh_reason": reason,
            "summary_input_updated_at": now,
            "topic_status": "active",
            "last_meaningful_activity_at": now,
        }
        build_version = int(metadata.get("summary_build_version") or 0)
        if build_version < 1 or not str(thread.summary or "").strip():
            payload["summary_updated_at"] = now
            payload["summary_build_version"] = 0
            payload["summary_source"] = "topic_router_fastpath"
        return payload

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z\u0400-\u04FF0-9_]{3,}", str(text or "").lower())
            if len(token) >= 3
        }

    @staticmethod
    def _overlap_ratio(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        inter = left & right
        return len(inter) / max(len(left), len(right), 1)
