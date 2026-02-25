import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional

import ollama

from config import EXTRACTOR_TIMEOUT_SEC, FAST_MODE, MODEL_NAME

logger = logging.getLogger(__name__)

TURN_METADATA_SYSTEM = (
    "Ты извлекаешь метаданные хода диалога для памяти.\\n"
    "Верни ТОЛЬКО JSON-объект строго такого формата:\\n"
    "{"
    "\\\"facts\\\": {"
    "\\\"speaker\\\":\\\"user\\\","
    "\\\"about\\\":\\\"user|assistant|other|none\\\","
    "\\\"polarity\\\":\\\"assertion|correction|rumor|uncertain|none\\\","
    "\\\"confidence\\\": number,"
    "\\\"facts\\\": { ... }"
    "},"
    "\\\"events\\\": ["
    "{"
    "\\\"type\\\":\\\"call|message|meeting|reminder|task|promise|plan|purchase|idea|preference|fact|location|health|mood|relationship|deadline|other\\\","
    "\\\"who\\\":string|null,"
    "\\\"what\\\":string|null,"
    "\\\"when\\\":string|null,"
    "\\\"where\\\":string|null,"
    "\\\"importance\\\":\\\"low|normal|high\\\","
    "\\\"tags\\\":[string],"
    "\\\"source_text\\\":string"
    "}"
    "],"
    "\\\"assistant_self\\\": {"
    "\\\"about\\\":\\\"assistant\\\","
    "\\\"polarity\\\":\\\"assertion|correction|retraction|none\\\","
    "\\\"confidence\\\": number,"
    "\\\"facts\\\": {"
    "\\\"likes\\\": [string],"
    "\\\"dislikes\\\": [string],"
    "\\\"interests\\\": [string],"
    "\\\"do_not_say\\\": [string],"
    "\\\"signature_phrases\\\": [string]"
    "}"
    "}"
    "}\\n\\n"
    "Правила:\\n"
    "- Извлекай только явно сказанное, ничего не выдумывай.\\n"
    "- facts: НЕ извлекай факты из вопросов пользователя.\\n"
    "- events: если событий нет, верни пустой массив.\\n"
    "- assistant_self: анализируй только assistant_text (если он передан). Если фактов нет, верни polarity='none', confidence=0, facts={}.\\n"
)

FACT_ALLOWED_KEYS = {
    "name", "birth_year", "birth_month", "birth_day", "profession", "projects", "likes", "habits"
}
SELF_ALLOWED_KEYS = {"likes", "dislikes", "interests", "do_not_say", "signature_phrases"}


def _default_metadata() -> Dict[str, Any]:
    return {
        "facts": {"speaker": "user", "about": "none", "polarity": "none", "confidence": 0.0, "facts": {}},
        "events": [],
        "assistant_self": {"about": "assistant", "polarity": "none", "confidence": 0.0, "facts": {}},
    }


def _extract_json_obj(text: str) -> Dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return _default_metadata()

    try:
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError
    except Exception:
        return _default_metadata()

    out = _default_metadata()

    facts = obj.get("facts")
    if isinstance(facts, dict):
        out["facts"]["speaker"] = facts.get("speaker", "user")
        out["facts"]["about"] = facts.get("about", "none")
        out["facts"]["polarity"] = facts.get("polarity", "none")
        try:
            out["facts"]["confidence"] = float(facts.get("confidence", 0.0))
        except Exception:
            out["facts"]["confidence"] = 0.0

        raw_facts = facts.get("facts")
        if isinstance(raw_facts, dict):
            out["facts"]["facts"] = {k: v for k, v in raw_facts.items() if k in FACT_ALLOWED_KEYS}

    events = obj.get("events")
    if isinstance(events, list):
        cleaned_events: List[Dict[str, Any]] = []
        for e in events:
            if not isinstance(e, dict):
                continue
            cleaned_events.append(
                {
                    "type": e.get("type", "other"),
                    "who": e.get("who"),
                    "what": e.get("what"),
                    "when": e.get("when"),
                    "where": e.get("where"),
                    "importance": e.get("importance", "normal"),
                    "tags": e.get("tags") if isinstance(e.get("tags"), list) else [],
                    "source_text": e.get("source_text"),
                }
            )
        out["events"] = cleaned_events

    assistant_self = obj.get("assistant_self")
    if isinstance(assistant_self, dict):
        out["assistant_self"]["about"] = assistant_self.get("about", "assistant")
        out["assistant_self"]["polarity"] = assistant_self.get("polarity", "none")
        try:
            out["assistant_self"]["confidence"] = float(assistant_self.get("confidence", 0.0))
        except Exception:
            out["assistant_self"]["confidence"] = 0.0

        self_facts = assistant_self.get("facts")
        if isinstance(self_facts, dict):
            cleaned_self_facts: Dict[str, List[str]] = {}
            for k, v in self_facts.items():
                if k not in SELF_ALLOWED_KEYS or not isinstance(v, list):
                    continue
                vv = [str(x).strip() for x in v if str(x).strip()]
                if vv:
                    cleaned_self_facts[k] = vv
            out["assistant_self"]["facts"] = cleaned_self_facts

    return out


def _build_user_prompt(user_text: str, assistant_text: Optional[str]) -> str:
    assistant_block = assistant_text if assistant_text is not None else ""
    return (
        f"user_text:\n{user_text}\n\n"
        f"assistant_text:\n{assistant_block}\n\n"
        "Верни JSON по формату выше."
    )


def _extract_turn_metadata_internal(user_text: str, assistant_text: Optional[str]) -> Dict[str, Any]:
    if "?" in user_text and not assistant_text:
        return _default_metadata()

    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": TURN_METADATA_SYSTEM},
            {"role": "user", "content": _build_user_prompt(user_text, assistant_text)},
        ],
        options={"temperature": 0},
    )
    return _extract_json_obj((resp.get("message", {}) or {}).get("content", "").strip())


def extract_turn_metadata(user_text: str, assistant_text: Optional[str] = None) -> Dict[str, Any]:
    if FAST_MODE:
        return _default_metadata()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_extract_turn_metadata_internal, user_text, assistant_text)
        try:
            result = future.result(timeout=EXTRACTOR_TIMEOUT_SEC)
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info("latency.extractors_ms=%.2f", latency_ms)
            return result
        except TimeoutError:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning("extract_turn_metadata timeout after %.2f ms", latency_ms)
            return _default_metadata()
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.exception("extract_turn_metadata failed after %.2f ms", latency_ms)
            return _default_metadata()
