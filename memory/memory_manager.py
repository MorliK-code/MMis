import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from memory.turn_metadata_extractor import extract_turn_metadata

logger = logging.getLogger(__name__)


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

        self._store_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._store_worker = threading.Thread(target=self._store_worker_loop, daemon=True)
        self._store_worker.start()

    def store_turn(self, user_text: str, assistant_text: str) -> None:
        # Неблокирующий путь: только быстрая запись в short memory + постановка задачи в очередь.
        self.short.add("user", user_text)
        self.short.add("assistant", assistant_text)

        try:
            self._store_queue.put_nowait((user_text, assistant_text))
        except Exception:
            logger.exception("failed to enqueue store_turn; fallback to sync processing")
            self._process_turn(user_text, assistant_text)

    def _store_worker_loop(self) -> None:
        while True:
            user_text, assistant_text = self._store_queue.get()
            try:
                self._process_turn(user_text, assistant_text)
            except Exception:
                logger.exception("background store_turn processing failed")
            finally:
                self._store_queue.task_done()

    def _process_turn(self, user_text: str, assistant_text: str) -> None:
        t0 = time.perf_counter()

        # 1) Полный лог
        try:
            self.log.append("user", user_text)
            self.log.append("assistant", assistant_text)
        except Exception:
            logger.exception("chat log append failed")

        # 2) Векторная память (для recall по смыслу)
        try:
            embed_t0 = time.perf_counter()
            combined = f"User: {user_text}\nAssistant: {assistant_text}"
            self.long.add(combined, meta={"type": "dialog_turn"})
            logger.info("latency.embeddings_add_ms=%.2f", (time.perf_counter() - embed_t0) * 1000)
        except Exception:
            logger.exception("long memory add failed")

        # 3) Единый extractor (facts + events + assistant_self)
        metadata = extract_turn_metadata(user_text, assistant_text)

        # 3.1) События
        events = metadata.get("events")
        if isinstance(events, list):
            for ev in events:
                if isinstance(ev, dict):
                    ev.setdefault("source_text", user_text)
                    try:
                        self.events.add(ev)
                    except Exception:
                        logger.exception("event add failed")

        # 3.2) Факты про пользователя/assistant claims
        self._apply_fact_result(user_text, metadata.get("facts"))

        # 3.3) Факты о самой ассистентке
        self._apply_assistant_self(metadata.get("assistant_self"))

        logger.info("latency.store_turn_total_ms=%.2f", (time.perf_counter() - t0) * 1000)

    def recall(self, query: str, n_results: int = 5) -> List[str]:
        try:
            t0 = time.perf_counter()
            found = self.long.search(query, n_results=n_results)
            logger.info("latency.embeddings_search_ms=%.2f", (time.perf_counter() - t0) * 1000)
        except Exception:
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
        except Exception:
            return []

    def last_event_of_type(self, event_type: str) -> Optional[Dict[str, Any]]:
        try:
            return self.events.last_of_type(event_type)
        except Exception:
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
            self.user_profile.merge(facts)
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
            except Exception:
                logger.exception("failed to store user claim about assistant")
            return

        # Неуверенные/слухи — тоже в события
        if polarity in ("rumor", "uncertain") and about in ("user", "assistant", "other"):
            try:
                self.events.add(
                    {
                        "type": "note",
                        "who": None,
                        "what": f"{polarity}_about_{about}: {facts}",
                        "when": None,
                        "where": None,
                        "importance": "low",
                        "tags": ["rumor"] if polarity == "rumor" else ["uncertain"],
                        "source_text": user_text,
                    }
                )
            except Exception:
                logger.exception("failed to store uncertain/rumor fact")
            return

    def _apply_assistant_self(self, self_res: Any) -> None:
        if not isinstance(self_res, dict):
            return
        pol = self_res.get("polarity", "none")
        facts = self_res.get("facts", {}) or {}
        conf = float(self_res.get("confidence", 0.0))
        if conf >= 0.7 and facts:
            self.assistant_profile.apply_fact_patch(pol, facts)
