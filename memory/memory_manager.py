import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from memory.fact_extractor import extract_facts
from memory.event_extractor import extract_events_llm
from memory.assistant_fact_extractor import extract_assistant_self
from memory.user_memory_selector import extract_user_memory_note
from memory.logging_utils import ErrorMetrics, get_memory_logger


class MemoryManager:
    def __init__(
        self,
        short_memory,
        long_memory,
        user_profile,
        assistant_profile,
        event_store,
        chat_log,
        distance_threshold: float = 0.65,
    ):
        self.short = short_memory
        self.long = long_memory
        self.user_profile = user_profile
        self.assistant_profile = assistant_profile
        self.events = event_store
        self.log = chat_log
        self.distance_threshold = distance_threshold
        self.logger = get_memory_logger()
        self.error_metrics = ErrorMetrics()
        self._postprocess_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mm-postprocess")
        self._shutting_down = False
        self.dislike_penalty_default = 0.20
        self.like_bonus_default = 0.03

    @staticmethod
    def _input_size(*values: Any) -> int:
        return sum(len(str(v or "")) for v in values)

    @staticmethod
    def _extract_user_do_not_say(user_text: str) -> List[str]:
        text = str(user_text or "").strip()
        if not text:
            return []

        lowered = text.lower()
        triggers = (
            "РЅРµ РіРѕРІРѕСЂРё",
            "РЅРµР»СЊР·СЏ РіРѕРІРѕСЂРёС‚СЊ",
            "РЅРµ РїСЂРѕРёР·РЅРѕСЃРё",
            "РЅРµ СѓРїРѕС‚СЂРµР±Р»СЏР№",
            "РЅРµ РёСЃРїРѕР»СЊР·СѓР№",
            "Р·Р°РїСЂРµС‰",
        )
        if not any(t in lowered for t in triggers):
            return []

        out: List[str] = []
        seen: set[str] = set()

        def _push(raw: str) -> None:
            value = re.sub(r"\s+", " ", str(raw or "")).strip(" \t\r\n\"'`В«В»вЂњвЂќ.,;:!?")
            if not value or len(value) > 80:
                return
            key = value.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(value)

        for m in re.finditer(r"[\"'`В«вЂњвЂќ]([^\"'`В»вЂњвЂќ]{1,80})[\"'`В»вЂњвЂќ]", text):
            _push(m.group(1))

        for m in re.finditer(
            r"(?:РЅРµ\s+РіРѕРІРѕСЂРё|РЅРµР»СЊР·СЏ\s+РіРѕРІРѕСЂРёС‚СЊ|РЅРµ\s+РїСЂРѕРёР·РЅРѕСЃРё|РЅРµ\s+СѓРїРѕС‚СЂРµР±Р»СЏР№|РЅРµ\s+РёСЃРїРѕР»СЊР·СѓР№)\s+"
            r"(?:СЃР»РѕРІРѕ|СЃР»РѕРІР°|С„СЂР°Р·Сѓ|С„СЂР°Р·Р°|С„СЂР°Р·С‹)\s+([A-Za-zРђ-РЇР°-СЏРЃС‘0-9_-]{1,40})",
            text,
            flags=re.IGNORECASE,
        ):
            _push(m.group(1))

        return out

    def _log_error(
        self,
        operation: str,
        error: Exception,
        input_size: int,
        category: Optional[str] = None,
    ) -> None:
        self.logger.warning(
            "operation=%s error=%s input_size=%s",
            operation,
            error,
            input_size,
        )
        if category:
            self.error_metrics.increment(category)

    def store_turn(self, user_text: str, assistant_text: str) -> None:
        if self._shutting_down:
            return
        # РќРµР±Р»РѕРєРёСЂСѓСЋС‰РёР№ РїСѓС‚СЊ: С‚РѕР»СЊРєРѕ Р±С‹СЃС‚СЂР°СЏ Р·Р°РїРёСЃСЊ РІ short memory + РїРѕСЃС‚Р°РЅРѕРІРєР° Р·Р°РґР°С‡Рё РІ РѕС‡РµСЂРµРґСЊ.
        self.short.add("user", user_text)
        self.short.add("assistant", assistant_text)

        try:
            self.log.append("user", user_text)
            self.log.append("assistant", assistant_text)
        except Exception as exc:
            self._log_error(
                operation="chat_log_append",
                error=exc,
                input_size=self._input_size(user_text, assistant_text),
            )

        self._postprocess_executor.submit(self._safe_process_turn, user_text, assistant_text)

    def _safe_process_turn(self, user_text: str, assistant_text: str) -> None:
        try:
            self._process_turn(user_text, assistant_text)
        except Exception as exc:
            self._log_error(
                operation="process_turn_background",
                error=exc,
                input_size=self._input_size(user_text, assistant_text),
            )

    def _process_turn(self, user_text: str, assistant_text: str) -> None:
        t0 = time.perf_counter()

        # 1) РџРѕР»РЅС‹Р№ Р»РѕРі
        try:
            evs = extract_events_llm(user_text)
            if isinstance(evs, list):
                for ev in evs:
                    if isinstance(ev, dict):
                        self.events.add(ev)
        except Exception as exc:
            self._log_error(
                operation="extract_events",
                error=exc,
                input_size=self._input_size(user_text),
                category="extractor_error",
            )

        # 4) Р’РµРєС‚РѕСЂРЅР°СЏ РїР°РјСЏС‚СЊ (РґР»СЏ recall РїРѕ СЃРјС‹СЃР»Сѓ)
        try:
            combined = f"User: {user_text}\nAssistant: {assistant_text}"
            self.long.add(combined, meta={"type": "dialog_turn"})
        except Exception as exc:
            self._log_error(
                operation="long_memory_add",
                error=exc,
                input_size=self._input_size(user_text, assistant_text),
                category="embedding_error",
            )

        # 5) Р¤Р°РєС‚С‹ РёР· СЃРѕРѕР±С‰РµРЅРёСЏ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ
        try:
            result = extract_facts(user_text)
            self._apply_fact_result(user_text, result)
        except Exception as exc:
            self._log_error(
                operation="extract_facts",
                error=exc,
                input_size=self._input_size(user_text),
                category="extractor_error",
            )

        try:
            forbidden = self._extract_user_do_not_say(user_text)
            if forbidden:
                self.assistant_profile.apply_fact_patch("assertion", {"do_not_say": forbidden})
        except Exception as exc:
            self._log_error(
                operation="assistant_do_not_say_from_user",
                error=exc,
                input_size=self._input_size(user_text),
                category="profile_write_error",
            )

        # 5b) LLM decides if this turn should become a compact long-term user note.
        try:
            note_res = extract_user_memory_note(user_text, assistant_text)
            note = str(note_res.get("note", "") or "").strip()
            remember = bool(note_res.get("remember", False))
            confidence = float(note_res.get("confidence", 0.0) or 0.0)
            if remember and note and confidence >= 0.55:
                self.user_profile.add_memory_note(note)
        except Exception as exc:
            self._log_error(
                operation="extract_user_memory_note",
                error=exc,
                input_size=self._input_size(user_text, assistant_text),
                category="extractor_error",
            )

        # 6) Р¤Р°РєС‚С‹ Рѕ СЃР°РјРѕР№ Р°СЃСЃРёСЃС‚РµРЅС‚РєРµ (РёР· РµС‘ РѕС‚РІРµС‚Р°)
        try:
            self_res = extract_assistant_self(assistant_text)
            pol = self_res.get("polarity", "none")
            facts = self_res.get("facts", {}) or {}
            conf = float(self_res.get("confidence", 0.0))
            if conf >= 0.7 and facts:
                self.assistant_profile.apply_fact_patch(pol, facts)
        except Exception as exc:
            self._log_error(
                operation="extract_assistant_self",
                error=exc,
                input_size=self._input_size(assistant_text),
                category="extractor_error",
            )

    def recall(self, query: str, n_results: int = 5) -> List[str]:
        try:
            t0 = time.perf_counter()
            found = self.long.search(query, n_results=n_results)
        except Exception as exc:
            self._log_error(
                operation="recall",
                error=exc,
                input_size=self._input_size(query),
                category="embedding_error",
            )
            return []

        disliked = self.recent_disliked_assistant_texts(limit=12)
        recalled: List[str] = []
        for item in found:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            doc = item[0]
            dist = item[1]
            meta = item[2] if len(item) >= 3 and isinstance(item[2], dict) else {}
            if not doc or dist is None:
                continue

            # Hard filter: skip chunks that match explicitly disliked assistant outputs.
            doc_l = str(doc).lower()
            if any((snippet and snippet in doc_l) for snippet in disliked):
                continue

            effective_dist = float(dist)
            score = float(meta.get("feedback_score", 0.0) or 0.0)
            if score < 0:
                effective_dist += float(meta.get("feedback_penalty", self.dislike_penalty_default))
            elif score > 0:
                effective_dist -= float(meta.get("feedback_bonus", self.like_bonus_default))

            if effective_dist <= self.distance_threshold:
                recalled.append(doc)
        return recalled

    def register_assistant_feedback(
        self,
        user_text: str,
        assistant_text: str,
        feedback: int,
        penalty: float | None = None,
    ) -> None:
        if self._shutting_down:
            return
        self._postprocess_executor.submit(
            self._register_assistant_feedback_sync,
            user_text,
            assistant_text,
            feedback,
            penalty,
        )

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        self._shutting_down = True
        try:
            self._postprocess_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError:
            self._postprocess_executor.shutdown(wait=wait)
        except Exception:
            pass

    def _register_assistant_feedback_sync(
        self,
        user_text: str,
        assistant_text: str,
        feedback: int,
        penalty: float | None = None,
    ) -> None:
        rating = 1 if int(feedback) > 0 else -1
        use_penalty = float(penalty) if penalty is not None else self.dislike_penalty_default
        use_penalty = max(0.0, min(use_penalty, 1.0))

        try:
            self.events.add(
                {
                    "type": "assistant_feedback",
                    "who": "user",
                    "what": "assistant_reply_feedback",
                    "rating": rating,
                    "penalty": use_penalty if rating < 0 else 0.0,
                    "assistant_text": assistant_text,
                    "user_text": user_text,
                    "importance": "high",
                    "tags": ["feedback", "like" if rating > 0 else "dislike"],
                }
            )
        except Exception as exc:
            self._log_error(
                operation="events_add_feedback",
                error=exc,
                input_size=self._input_size(user_text, assistant_text, rating),
            )

        try:
            self.log.append("feedback", f"rating={rating}; penalty={use_penalty:.2f}; assistant={assistant_text}")
        except Exception as exc:
            self._log_error(
                operation="chat_log_feedback_append",
                error=exc,
                input_size=self._input_size(user_text, assistant_text, rating),
            )

        try:
            combined = f"User: {user_text}\nAssistant: {assistant_text}"
            self.long.add(
                combined,
                meta={
                    "type": "dialog_feedback",
                    "feedback_score": float(rating),
                    "feedback_penalty": use_penalty if rating < 0 else 0.0,
                    "feedback_bonus": self.like_bonus_default if rating > 0 else 0.0,
                },
            )
        except Exception as exc:
            self._log_error(
                operation="long_memory_add_feedback",
                error=exc,
                input_size=self._input_size(user_text, assistant_text, rating),
                category="embedding_error",
            )

    def recent_disliked_assistant_texts(self, limit: int = 20) -> List[str]:
        try:
            events = self.events.last(300)
        except Exception as exc:
            self._log_error(
                operation="events_read_feedback",
                error=exc,
                input_size=self._input_size(limit),
            )
            return []

        out: List[str] = []
        seen: set[str] = set()
        for ev in reversed(events):
            if not isinstance(ev, dict):
                continue
            if ev.get("type") != "assistant_feedback":
                continue
            try:
                rating = int(ev.get("rating", 0))
            except Exception:
                rating = 0
            if rating >= 0:
                continue
            text = str(ev.get("assistant_text") or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(key[:160])
            if len(out) >= limit:
                break
        return out

    def last_events(self, n: int = 20) -> List[Dict[str, Any]]:
        try:
            return self.events.last(n)
        except Exception as exc:
            self._log_error(
                operation="last_events",
                error=exc,
                input_size=self._input_size(n),
            )
            return []

    def last_event_of_type(self, event_type: str) -> Optional[Dict[str, Any]]:
        try:
            return self.events.last_of_type(event_type)
        except Exception as exc:
            self._log_error(
                operation="last_event_of_type",
                error=exc,
                input_size=self._input_size(event_type),
            )
            return None

    def recent_chat_messages(self, n: int = 20) -> List[Dict[str, Any]]:
        try:
            rows = self.log.last_messages(n=n, roles=("user", "assistant"))
            return rows if isinstance(rows, list) else []
        except Exception as exc:
            self._log_error(
                operation="recent_chat_messages",
                error=exc,
                input_size=self._input_size(n),
            )
            return []

    def relevant_chat_messages(self, query: str, limit: int = 6, scan_last: int = 400) -> List[Dict[str, Any]]:
        try:
            rows = self.log.relevant_messages(
                query=query,
                limit=limit,
                scan_last=scan_last,
                roles=("user", "assistant"),
            )
            return rows if isinstance(rows, list) else []
        except Exception as exc:
            self._log_error(
                operation="relevant_chat_messages",
                error=exc,
                input_size=self._input_size(query, limit, scan_last),
            )
            return []

    def _apply_fact_result(self, user_text: str, result: Any) -> None:
        if not isinstance(result, dict):
            return

        about = (result.get("about") or "none").strip().lower()
        polarity = (result.get("polarity") or "none").strip().lower()
        try:
            confidence = float(result.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        facts = result.get("facts") or {}
        if not isinstance(facts, dict) or not facts:
            return

        can_write_profile = confidence >= 0.55 and polarity in ("assertion", "correction")

        if about == "user" and can_write_profile:
            try:
                self.user_profile.merge(facts)
            except Exception as exc:
                self._log_error(
                    operation="user_profile_merge",
                    error=exc,
                    input_size=self._input_size(user_text, facts),
                    category="profile_write_error",
                )
            return

        # Р•СЃР»Рё РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РіРѕРІРѕСЂРёС‚ С„Р°РєС‚С‹ РїСЂРѕ Р°СЃСЃРёСЃС‚РµРЅС‚РєСѓ вЂ” РґРѕР±Р°РІР»СЏРµРј РєР°Рє Р·Р°РјРµС‚РєСѓ/СЃРѕР±С‹С‚РёРµ (РЅРµ РјРµРЅСЏРµРј РїСЂРѕС„РёР»СЊ)
        # Explicit wording constraints from the user can update assistant profile.
        if about == "assistant" and confidence >= 0.55 and polarity in ("assertion", "correction"):
            assistant_patch: Dict[str, Any] = {}
            if isinstance(facts.get("do_not_say"), list) and facts.get("do_not_say"):
                assistant_patch["do_not_say"] = facts.get("do_not_say")
            if isinstance(facts.get("signature_phrases"), list) and facts.get("signature_phrases"):
                assistant_patch["signature_phrases"] = facts.get("signature_phrases")

            if assistant_patch:
                try:
                    self.assistant_profile.apply_fact_patch("assertion", assistant_patch)
                except Exception as exc:
                    self._log_error(
                        operation="assistant_profile_merge_from_user_claim",
                        error=exc,
                        input_size=self._input_size(user_text, assistant_patch),
                        category="profile_write_error",
                    )
                return
            try:
                self.events.add(
                    {
                        "type": "note",
                        "who": "user",
                        "what": f"user_claim_about_assistant: {facts}",
                        "when": None,
                        "where": None,
                        "importance": "low",
                        "tags": ["about_assistant"],
                        "source_text": user_text,
                    }
                )
            except Exception as exc:
                self._log_error(
                    operation="events_add_about_assistant",
                    error=exc,
                    input_size=self._input_size(user_text, facts),
                )
            return

        should_write_events_only = (
            about in ("user", "assistant", "other")
            and (confidence < 0.70 or polarity in ("rumor", "uncertain"))
        )
        if should_write_events_only:
            tag = "rumor" if polarity == "rumor" else "uncertain" if polarity == "uncertain" else "low_confidence"
            try:
                self.events.add(
                    {
                        "type": "note",
                        "who": None,
                        "what": f"{tag}_about_{about}: {facts}",
                        "when": None,
                        "where": None,
                        "importance": "low",
                        "tags": [tag],
                        "source_text": user_text,
                    }
                )
            except Exception as exc:
                self._log_error(
                    operation="events_add_uncertain_or_rumor",
                    error=exc,
                    input_size=self._input_size(user_text, facts),
                )
            return


