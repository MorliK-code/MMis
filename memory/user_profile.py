import json
import os
from typing import Any, Dict, List

class UserProfile:
    def __init__(self, path: str = "user_profile.json"):
        self.path = path
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
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

    def merge(self, facts: dict):
        if not facts:
            return

        for k, v in facts.items():
            if v is None:
                continue

            normalized = self._normalize_value(v)
            if isinstance(normalized, list):
                current = self.data.get(k, [])
                if not isinstance(current, list):
                    current = []
                for item in normalized:
                    if item not in current:
                        current.append(item)
                self.data[k] = current
            else:
                self.data[k] = normalized

        self.save()

    def get_summary(self) -> str:
        if not self.data:
            return "Профиль пользователя пока пуст."
        lines = [f"- {k}: {v}" for k, v in self.data.items()]
        return "Профиль пользователя:\n" + "\n".join(lines)