import json
import re
import ollama
from config import MODEL_NAME, build_ollama_options

EVENT_SYSTEM = (
    "Ты извлекаешь события из реплики пользователя для памяти.\n"
    "Верни ТОЛЬКО JSON-массив. Если событий нет — верни []\n"
    "Каждый элемент массива — объект:\n"
    "{"
    "\"type\":\"call|message|meeting|reminder|task|promise|plan|purchase|idea|preference|fact|location|health|mood|relationship|deadline|other\","
    "\"who\":string|null,"
    "\"what\":string|null,"
    "\"when\":string|null,"
    "\"where\":string|null,"
    "\"importance\":\"low|normal|high\","
    "\"tags\":[string],"
    "\"source_text\":string"
    "}\n"
    "Правила:\n"
    "- who: имя/кто связан, если есть (Гарри, мама, босс)\n"
    "- when: если есть 'через минуту/завтра/в пятницу' — запиши строкой как сказано\n"
    "- what: кратко что произошло/что нужно сделать\n"
    "- source_text: оригинальная реплика пользователя\n"
    "- Не выдумывай факты. Только то, что явно сказано.\n"
)

def _extract_json_array(text: str):
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def extract_events_llm(user_text: str) -> list[dict]:
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": EVENT_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        options=build_ollama_options("event_extraction")
    )
    raw = resp["message"]["content"].strip()
    events = _extract_json_array(raw)

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
