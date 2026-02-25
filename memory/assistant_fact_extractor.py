from prompts.extractors import SELF_SYSTEM
from prompts.handlers import run_chat_prompt
from prompts.json_extract import extract_json_object


def _extract_json_obj(text: str) -> dict:
    obj = extract_json_object(text, {"about": "assistant", "polarity": "none", "confidence": 0.0, "facts": {}})

    obj.setdefault("about", "assistant")
    obj.setdefault("polarity", "none")
    obj.setdefault("confidence", 0.0)
    obj.setdefault("facts", {})

    facts = obj.get("facts", {})
    if not isinstance(facts, dict):
        facts = {}

    cleaned = {}
    for k in ("likes", "dislikes", "interests", "do_not_say", "signature_phrases"):
        v = facts.get(k)
        if isinstance(v, list) and v:
            vv = [str(x).strip() for x in v if str(x).strip()]
            if vv:
                cleaned[k] = vv

    obj["facts"] = cleaned
    return obj


def extract_assistant_self(assistant_text: str) -> dict:
    resp = run_chat_prompt(
        task_type="assistant_fact_extraction",
        messages=[
            {"role": "system", "content": SELF_SYSTEM},
            {"role": "user", "content": assistant_text},
        ],
        default_content="{}",
    )
    return _extract_json_obj(resp.content)
