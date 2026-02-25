import json
import re
import ollama
from config import MODEL_NAME

SELF_SYSTEM = (
    "Ты анализируешь ТОЛЬКО ответ ассистентки и извлекаешь факты о НЕЙ.\n"
    "Верни ТОЛЬКО JSON объекта формата:\n"
    "{\n"
    "  \"about\": \"assistant\",\n"
    "  \"polarity\": \"assertion|correction|retraction|none\",\n"
    "  \"confidence\": number,\n"
    "  \"facts\": {\n"
    "    \"likes\": [string],\n"
    "    \"dislikes\": [string],\n"
    "    \"interests\": [string],\n"
    "    \"do_not_say\": [string],\n"
    "    \"signature_phrases\": [string]\n"
    "  }\n"
    "}\n\n"
    "Правила:\n"
    "- Извлекай только то, что явно сказано в тексте.\n"
    "- Если ассистентка просто поддерживает пользователя (\"у нас много общего\") без явного 'я люблю' — НЕ считать это фактом.\n"
    "- Если ассистентка сказала 'я не люблю X' -> dislikes += X.\n"
    "- Если сказала 'я люблю X' / 'мне нравится X' -> likes += X.\n"
    "- Если сказала 'раньше любила, но теперь нет' -> polarity='retraction' (убрать из likes/интерсов).\n"
    "- Если фактов нет -> {\"about\":\"assistant\",\"polarity\":\"none\",\"confidence\":0,\"facts\":{}}\n"
)


def _extract_json_obj(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"about": "assistant", "polarity": "none", "confidence": 0.0, "facts": {}}

    try:
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError
    except Exception:
        return {"about": "assistant", "polarity": "none", "confidence": 0.0, "facts": {}}

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
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SELF_SYSTEM},
            {"role": "user", "content": assistant_text},
        ],
        options={"temperature": 0},
    )
    return _extract_json_obj(resp["message"]["content"].strip())
