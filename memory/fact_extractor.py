import json
import ollama
from config import MODEL_NAME

FACT_SYSTEM = (
    "Ты извлекаешь устойчивые факты о пользователе для профиля. "
    "Возвращай ТОЛЬКО JSON-объект. "
    "Если новых фактов нет — верни пустой объект {}. "
    "Не включай случайные детали. "
    "Примеры фактов: имя, город (если явно сказал), предпочтения, долгие проекты, "
    "технический стек, язык общения."
)

def extract_facts(user_text: str, assistant_text: str) -> dict:
    promt = (
        f"User said: {user_text}\n"
        f"Assistant replied: {assistant_text}\n"
        "Extract new stable facts about the user (if any)."
    )

    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": FACT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0}
    )

    raw = resp["message"]["content"].strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}