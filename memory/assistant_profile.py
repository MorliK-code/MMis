import json
import os
from typing import Dict, Any, List

DEFAULT_ASSISTANT_PROFILE: Dict[str, Any] = {
    "style": "лёгкая ирония",
    "humor_level": 2,
    "tone": "дружелюбная",
    "address": "ты",
    "gender": "женский",
    "talkativeness": 1,
    "emoji_level": 1,
    "do_not_say": [
        "я персональный ассистент",
        "я не имею памяти",
        "я ассистент",
        "я не могу запоминать информацию",
        "я не могу помнить",
        "я ИИ",
        "регистрация",
        "платформа",
        "я здесь, чтобы помочь"
    ],
    "signature_phrases": [], 
}

class AssistantProfile:
    def __init__(self, path: str = "assistant_profile.json"):
        self.path = path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = DEFAULT_ASSISTANT_PROFILE
            self.save()
        
    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def merge(self, patch: Dict[str, Any]):
        if not patch:
            return
        for k, v in patch.items():
            if v is None:
                continue
        if isinstance(v, list):
            cur = self.data.get(k, [])
            if not isinstance(cur, list):
                cur = []
            for item in v:
                if item not in cur:
                    cur.append(item)
            self.data[k] = cur
        else:
            self.data[k] = v
        self.save()

    def apply_fact_patch(self, polarity: str, facts: dict):
        if not facts:
            return

        for k in ("likes", "dislikes", "interests", "do_not_say", "signature_phrases"):
            if k not in self.data or not isinstance(self.data.get(k), list):
                self.data[k] = []

        def add_unique(key, items):
            for it in items:
                if it not in self.data[key]:
                    self.data[key].append(it)

        def remove_items(key, items):
            self.data[key] = [x for x in self.data[key] if x not in items]

        polarity = (polarity or "none").lower()

        if polarity in ("assertion", "correction"):
            if "likes" in facts:
                add_unique("likes", facts["likes"])
                remove_items("dislikes", facts["likes"])
            if "dislikes" in facts:
                add_unique("dislikes", facts["dislikes"])
                remove_items("likes", facts["dislikes"])
            if "interests" in facts:
                add_unique("interests", facts["interests"])
            if "do_not_say" in facts:
                add_unique("do_not_say", facts["do_not_say"])
            if "signature_phrases" in facts:
                add_unique("signature_phrases", facts["signature_phrases"])

            for k, items in facts.items():
                if isinstance(items, list):
                    remove_items(k, items)

        self.save()

    def summary(self) -> str:
        d = self.data
        return (
            "Профиль ассистентки:\n"
            f"Стиль: {d.get('style', '')}\n"
            f"Юмор: {d.get('humor_level', 0)}\n"
            f"Тон: {d.get('tone', '')}\n"
            f"Обращение: {d.get('address', '')}\n"
            f"Эмодзи: {d.get('emoji_level', 0)}\n"
            f"Разговорчивость: {d.get('talkativeness', 0)}\n"
            f"Пол: {d.get('gender', '')}\n"
        )
