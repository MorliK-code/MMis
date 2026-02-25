from prompts.extractors import FACT_SYSTEM
from prompts.handlers import run_chat_prompt
from prompts.json_extract import extract_json_object

ALLOWED_FACT_KEYS = {
    "user": {
        "name",
        "birth_year",
        "birth_month",
        "birth_day",
        "profession",
        "projects",
        "likes",
        "dislikes",
        "interests",
        "habits",
        "bio",
        "communication_style",
        "boundaries",
        "values",
        "goals",
        "persona_notes",
    },
    "assistant": {
        "name",
        "profession",
        "projects",
        "likes",
        "dislikes",
        "interests",
        "bio",
        "communication_style",
        "boundaries",
        "values",
        "goals",
        "persona_notes",
    },
}


def _default_result() -> dict:
    return {"speaker": "user", "about": "none", "polarity": "none", "confidence": 0.0, "facts": {}}


def _extract_json(text: str) -> dict:
    obj = extract_json_object(text, _default_result())

    obj.setdefault("speaker", "user")
    obj.setdefault("about", "none")
    obj.setdefault("polarity", "none")
    obj.setdefault("confidence", 0.0)
    obj.setdefault("facts", {})

    about = str(obj.get("about", "none")).strip().lower()
    facts = obj.get("facts", {})
    if not isinstance(facts, dict):
        facts = {}
    allowed_keys = ALLOWED_FACT_KEYS.get(about, set())
    obj["facts"] = {k: v for k, v in facts.items() if k in allowed_keys} if allowed_keys else {}
    return obj


def _is_likely_question_only(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return True

    if "?" not in normalized:
        return False

    lowered = normalized.lower()
    has_statement_markers = any(
        marker in lowered for marker in (" я ", " мне ", " у меня ", " мой ", " моя ", " мы ", " я,", "я ")
    )
    return lowered.endswith("?") and not has_statement_markers


def extract_facts(user_text: str) -> dict:
    if _is_likely_question_only(user_text):
        return _default_result()

    resp = run_chat_prompt(
        task_type="fact_extraction",
        messages=[
            {"role": "system", "content": FACT_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        default_content="{}",
    )
    return _extract_json(resp.content)
