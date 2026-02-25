import json
from pathlib import Path
from typing import Any, Dict, List, Union

DEFAULT_ASSISTANT_PROFILE: Dict[str, Any] = {
    "name": "Вероника",
    "short_name": "Ника",
    "style": "лёгкая ирония",
    "humor_level": 2,
    "tone": "дружелюбная",
    "address": "ты",
    "gender": "женский",
    "talkativeness": 1,
    "emoji_level": 1,
    "likes": [],
    "dislikes": [],
    "interests": [],
    "do_not_say": [
        "я персональный ассистент",
        "я не имею памяти",
        "я ассистент",
        "я не могу запоминать информацию",
        "я не могу помнить",
        "я ИИ",
        "регистрация",
        "платформа",
        "я здесь, чтобы помочь",
    ],
    "signature_phrases": [],
}

IMMUTABLE_BASE_KEYS = (
    "name",
    "short_name",
    "style",
    "humor_level",
    "tone",
    "address",
    "gender",
    "talkativeness",
    "emoji_level",
)

LIST_PROFILE_KEYS = ("likes", "dislikes", "interests", "do_not_say", "signature_phrases")


class AssistantProfile:
    def __init__(self, path: Union[str, Path] = "assistant_profile.json"):
        self.path = Path(path)
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self):
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = DEFAULT_ASSISTANT_PROFILE.copy()

        # Базовые параметры ассистентки должны подтягиваться из DEFAULT всегда,
        # чтобы правки в коде применялись без ручного удаления профиля.
        for key in IMMUTABLE_BASE_KEYS:
            self.data[key] = DEFAULT_ASSISTANT_PROFILE.get(key)

        # Для списков сохраняем обученные значения и добавляем дефолтные.
        for key in LIST_PROFILE_KEYS:
            existing = self._normalize_to_list(self.data.get(key))
            defaults = self._normalize_to_list(DEFAULT_ASSISTANT_PROFILE.get(key))
            merged = defaults.copy()
            for item in existing:
                if item not in merged:
                    merged.append(item)
            self.data[key] = merged

        # На случай будущих новых ключей.
        for key, value in DEFAULT_ASSISTANT_PROFILE.items():
            if key not in self.data:
                self.data[key] = value

        self.save()

    def save(self):
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, list):
            normalized: List[str] = []
            for item in value:
                item_str = str(item).strip()
                if item_str and item_str not in normalized:
                    normalized.append(item_str)
            return normalized

        if isinstance(value, str):
            return value.strip()

        return value

    @classmethod
    def _normalize_to_list(cls, items: Any) -> List[str]:
        if items is None:
            return []
        if not isinstance(items, list):
            items = [items]
        normalized = cls._normalize_value(items)
        return normalized if isinstance(normalized, list) else []

    def merge(self, patch: Dict[str, Any]):
        """Смешивает поля профиля. Списки — дополняет уникальными элементами."""
        if not patch:
            return

        for k, v in patch.items():
            if v is None:
                continue

            normalized = self._normalize_value(v)
            if isinstance(normalized, list):
                cur = self.data.get(k, [])
                if not isinstance(cur, list):
                    cur = []
                for item in normalized:
                    if item not in cur:
                        cur.append(item)
                self.data[k] = cur
            else:
                self.data[k] = normalized

        self.save()

    def apply_fact_patch(self, polarity: str, facts: dict):
        """Обновляет likes/dislikes/interests и т.п. на основе извлечённых фактов."""
        if not facts:
            return

        polarity = (polarity or "none").lower().strip()

        # гарантируем списки
        for k in ("likes", "dislikes", "interests", "do_not_say", "signature_phrases"):
            if k not in self.data or not isinstance(self.data.get(k), list):
                self.data[k] = []

        def add_unique(key: str, items):
            for it in self._normalize_to_list(items):
                if it not in self.data[key]:
                    self.data[key].append(it)

        def remove_items(key: str, items):
            remove_set = set(self._normalize_to_list(items))
            if not remove_set:
                return
            self.data[key] = [x for x in self.data[key] if x not in remove_set]

        if polarity in ("assertion", "correction"):
            if "likes" in facts:
                add_unique("likes", facts.get("likes"))
                remove_items("dislikes", facts.get("likes"))
            if "dislikes" in facts:
                add_unique("dislikes", facts.get("dislikes"))
                remove_items("likes", facts.get("dislikes"))
            if "interests" in facts:
                add_unique("interests", facts.get("interests"))
            if "do_not_say" in facts:
                add_unique("do_not_say", facts.get("do_not_say"))
            if "signature_phrases" in facts:
                add_unique("signature_phrases", facts.get("signature_phrases"))

        elif polarity == "retraction":
            # "я раньше любила X, но теперь нет" — убрать X из likes/interests и/или dislikes
            if "likes" in facts:
                remove_items("likes", facts.get("likes"))
            if "dislikes" in facts:
                remove_items("dislikes", facts.get("dislikes"))
            if "interests" in facts:
                remove_items("interests", facts.get("interests"))

        self.save()

    def summary(self) -> str:
        d = self.data
        # Важно: это для LLM — не для вывода пользователю.
        likes = ", ".join(d.get("likes", [])[:8])
        dislikes = ", ".join(d.get("dislikes", [])[:8])
        interests = ", ".join(d.get("interests", [])[:8])
        do_not_say = ", ".join(d.get("do_not_say", [])[:12])

        return (
            "Профиль ассистентки (используй для согласованности, но НЕ перечисляй пользователю без прямого вопроса):\n"
            f"Стиль: {d.get('style', '')}. Тон: {d.get('tone', '')}. Юмор: {d.get('humor_level', 0)}.\n"
            f"Разговорчивость: {d.get('talkativeness', 0)}. Эмодзи: {d.get('emoji_level', 0)}.\n"
            + (f"Likes: {likes}.\n" if likes else "")
            + (f"Dislikes: {dislikes}.\n" if dislikes else "")
            + (f"Interests: {interests}.\n" if interests else "")
            + (f"Запрещённые фразы: {do_not_say}.\n" if do_not_say else "")
        )
