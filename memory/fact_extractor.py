import json
import re
import ollama
from config import MODEL_NAME, build_ollama_options

FACT_SYSTEM = (
    "Ты извлекаешь структурированные факты из сообщения пользователя.\n"
    "Верни ТОЛЬКО JSON объект строго такого вида:\n"
    "{"
    "\"speaker\":\"user\","
    "\"about\":\"user|assistant|other|none\","
    "\"polarity\":\"assertion|correction|rumor|uncertain|none\","
    "\"confidence\": number,"
    "\"facts\": { ... }"
    "}\n\n"
    "Правила:\n"
    "- speaker всегда 'user' (это сообщение пользователя).\n"
    "- about='user' если факты про пользователя (я/мне/у меня/мой).\n"
    "- about='assistant' если факты про ассистентку (ты/тебе/у тебя/твой).\n"
    "- about='other' если факт про третье лицо (Гарри, мама, друг).\n"
    "- about='none' если фактов нет или это вопрос.\n"
    "- polarity:\n"
    "  * correction если пользователь исправляет прошлое (\"нет\", \"не\", \"а неееет\")\n"
    "  * rumor если есть маркеры слуха (\"я слышал\", \"говорят\")\n"
    "  * uncertain если (\"кажется\", \"вроде\")\n"
    "  * assertion если уверенное утверждение\n"
    "- НЕ извлекай факты из вопросов.\n"
    "- НЕ выдумывай факты.\n"
)

ALLOWED_KEYS = {
    "name", "birth_year", "birth_month", "birth_day",
    "profession", "projects", "likes", "habits"
}

def _extract_json(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"speaker":"user","about":"none","polarity":"none","confidence":0.0,"facts":{}}
    try:
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            raise ValueError
    except Exception:
        return {"speaker":"user","about":"none","polarity":"none","confidence":0.0,"facts":{}}

    obj.setdefault("speaker", "user")
    obj.setdefault("about", "none")
    obj.setdefault("polarity", "none")
    obj.setdefault("confidence", 0.0)
    obj.setdefault("facts", {})

    facts = obj.get("facts", {})
    if not isinstance(facts, dict):
        facts = {}
    facts = {k: v for k, v in facts.items() if k in ALLOWED_KEYS}
    obj["facts"] = facts
    return obj

def extract_facts(user_text: str) -> dict:
    if "?" in user_text:
        return {"speaker":"user","about":"none","polarity":"none","confidence":0.0,"facts":{}}

    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": FACT_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        options=build_ollama_options("fact_extraction")
    )
    return _extract_json(resp["message"]["content"].strip())
