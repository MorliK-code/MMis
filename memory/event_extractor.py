from prompts.extractors import EVENT_SYSTEM
from prompts.handlers import run_chat_prompt
from prompts.json_extract import extract_json_array


def extract_events_llm(user_text: str) -> list[dict]:
    resp = run_chat_prompt(
        task_type="event_extraction",
        messages=[
            {"role": "system", "content": EVENT_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        default_content="[]",
    )
    events = extract_json_array(resp.content)

    cleaned = []
    for e in events:
        if not isinstance(e, dict):
            continue
        e.setdefault("type", "other")
        e.setdefault("who", None)
        e.setdefault("what", None)
        e.setdefault("when", None)
        e.setdefault("where", None)
        e.setdefault("importance", "normal")
        e.setdefault("tags", [])
        e["source_text"] = user_text
        cleaned.append(e)

    return cleaned
