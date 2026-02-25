import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from memory.fact_extractor import extract_facts
from memory.event_extractor import extract_events_llm
from memory.assistant_fact_extractor import extract_assistant_self
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

    @staticmethod
    def _input_size(*values: Any) -> int:
        return sum(len(str(v or "")) for v in values)

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

    def error_metrics_snapshot(self) -> Dict[str, int]:
        return self.error_metrics.snapshot()

        self._store_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._store_worker = threading.Thread(target=self._store_worker_loop, daemon=True)
        self._store_worker.start()

    def store_turn(self, user_text: str, assistant_text: str) -> None:
        # Неблокирующий путь: только быстрая запись в short memory + постановка задачи в очередь.
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

    def _process_turn(self, user_text: str, assistant_text: str) -> None:
        t0 = time.perf_counter()

        # 1) Полный лог
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

        # 3) Оперативная память (критичная операция — fail-safe не меняем)
        self.short.add("user", user_text)
        self.short.add("assistant", assistant_text)

        # 4) Векторная память (для recall по смыслу)
        try:
            embed_t0 = time.perf_counter()
            combined = f"User: {user_text}\nAssistant: {assistant_text}"
            self.long.add(combined, meta={"type": "dialog_turn"})
        except Exception as exc:
            self._log_error(
                operation="long_memory_add",
                error=exc,
                input_size=self._input_size(user_text, assistant_text),
                category="embedding_error",
            )

        # 5) Факты из сообщения пользователя
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

        # 6) Факты о самой ассистентке (из её ответа)
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

        recalled: List[str] = []
        for item in found:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            doc = item[0]
            dist = item[1]
            if doc and dist is not None and dist <= self.distance_threshold:
                recalled.append(doc)
        return recalled

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

        can_write_profile = confidence >= 0.70 and polarity in ("assertion", "correction")

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

        # Если пользователь говорит факты про ассистентку — добавляем как заметку/событие (не меняем профиль)
        if about == "assistant" and confidence >= 0.70 and polarity in ("assertion", "correction"):
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
        pol = self_res.get("polarity", "none")
        facts = self_res.get("facts", {}) or {}
        conf = float(self_res.get("confidence", 0.0))
        if conf >= 0.7 and facts:
            self.assistant_profile.apply_fact_patch(pol, facts)
