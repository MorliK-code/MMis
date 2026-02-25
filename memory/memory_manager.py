from typing import Any, Dict, List, Optional

from memory.fact_extractor import extract_facts
from memory.event_extractor import extract_events_llm
from memory.assistant_fact_extractor import extract_assistant_self


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

    def store_turn(self, user_text: str, assistant_text: str) -> None:
        # 1) Полный лог
        try:
            self.log.append("user", user_text)
            self.log.append("assistant", assistant_text)
        except Exception:
            pass

        # 2) События (из реплики пользователя)
        try:
            evs = extract_events_llm(user_text)
            if isinstance(evs, list):
                for ev in evs:
                    if isinstance(ev, dict):
                        self.events.add(ev)
        except Exception:
            pass

        # 3) Оперативная память
        self.short.add("user", user_text)
        self.short.add("assistant", assistant_text)

        # 4) Векторная память (для recall по смыслу)
        try:
            combined = f"User: {user_text}\nAssistant: {assistant_text}"
            self.long.add(combined, meta={"type": "dialog_turn"})
        except Exception:
            pass

        # 5) Факты из сообщения пользователя
        try:
            result = extract_facts(user_text)
            self._apply_fact_result(user_text, result)
        except Exception:
            pass

        # 6) Факты о самой ассистентке (из её ответа)
        try:
            self_res = extract_assistant_self(assistant_text)
            pol = self_res.get("polarity", "none")
            facts = self_res.get("facts", {}) or {}
            conf = float(self_res.get("confidence", 0.0))
            if conf >= 0.7 and facts:
                self.assistant_profile.apply_fact_patch(pol, facts)
        except Exception:
            pass

    def recall(self, query: str, n_results: int = 5) -> List[str]:
        try:
            found = self.long.search(query, n_results=n_results)
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

        if about == "assistant" and can_write_profile:
            self.assistant_profile.merge(facts)
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
            except Exception:
                pass
            return
