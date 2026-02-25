import json
import re
import ollama
from config import MODEL_NAME

SELF_SYSTEM = (
    "Ты анализируешь ТОЛЬКО ответ ассистентки и извлекаешь факты о НЕЙ.\n"
    "Верни ТОЛЬКО JSON объекта формата:\n"
    "{"
    "\"about\":\"assistant\","
    "\"polarity\":\"assertion|correction|retraction|none\","
    "\"confidence\": number,"
    "\"facts\": {"
    "  \"likes\": [string],"
    "  \"dislikes\": [string],"
    "  \"interests\": [string]"
    "  \"do_not_say\": [string],"
    "  \"signature_phrases\": [string]"
    "}"
    "}\n\n"
    "Правила:\n"
    "- Извлекай только то, что явно сказано в тексте.\n"
    "- Извлекай так-же, явно указанов в тексте что нельзя говорить.\n"
    "- Извлекай явно указанные фразы, которые ассистентка может говорить в тексте.\n"
    "- Если ассистентка сказала 'я не люблю X' -> dislikes += X.\n"
    "- Если сказала 'я люблю X' / 'мне нравится X' -> likes += X.\n"
    "- Если сказала 'раньше любила, но теперь нет' -> polarity='retraction' и факт должен убирать из likes и/или добавлять в dislikes.\n"
    "- Если фактов нет -> {\"about\":\"assistant\",\"polarity\":\"none\",\"confidence\":0,\"facts\":{}}"
)

def _extract_json_obj(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"about":"assistant","polarity":"none","confidence":0.0,"facts":{}}
    try:
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError
    except Exception:
        return {"about":"assistant","polarity":"none","confidence":0.0,"facts":{}}

    obj.setdefault("about", "assistant")
    obj.setdefault("polarity", "none")
    obj.setdefault("confidence", 0.0)
    obj.setdefault("facts", {})

    facts = obj.get("facts", {})
    if not isinstance(facts, dict):
        facts = {}
    cleaned = {}
    for k in ("likes", "dislikes", "interests", ):
        v = facts.get(k)
        if isinstance(v, list) and v:
            cleaned[k] = [str(x).strip() for x in v if str(x).strip()]
    obj["facts"] = cleaned
    return obj

def extract_assistant_self(assistant_text: str) -> dict:
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SELF_SYSTEM},
            {"role": "user", "content": assistant_text},
        ],
        options={"temperature": 0}
    )
    return _extract_json_obj(resp["message"]["content"].strip())