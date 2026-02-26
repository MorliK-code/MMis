import json
import re

import ollama

from config import MODEL_NAME, build_ollama_options

MEMORY_NOTE_SYSTEM = (
    "Ты извлекаешь ОДНУ краткую заметку о пользователе для долгой памяти ассистента. "
    "Если в сообщении нет устойчивой полезной информации о предпочтениях, планах, контексте жизни, "
    "верни remember=false. Верни только JSON: "
    "{\"remember\": boolean, \"note\": string, \"confidence\": number}."
)


def _default() -> dict:
    return {"remember": False, "note": "", "confidence": 0.0}


def _parse_json(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return _default()
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return _default()

    remember = bool(obj.get("remember", False))
    note = str(obj.get("note", "") or "").strip()
    try:
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    if len(note) > 220:
        note = note[:220].rstrip() + "..."

    return {"remember": remember, "note": note, "confidence": confidence}


def extract_user_memory_note(user_text: str, assistant_text: str = "") -> dict:
    if not str(user_text or "").strip():
        return _default()

    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": MEMORY_NOTE_SYSTEM},
            {
                "role": "user",
                "content": f"user_text:\n{user_text}\n\nassistant_text:\n{assistant_text}\n\nВерни JSON.",
            },
        ],
        options=build_ollama_options("fact_extraction"),
    )
    content = (resp.get("message", {}) or {}).get("content", "")
    return _parse_json(content)
