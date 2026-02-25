import json
import os
from typing import Any, Dict

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

    def merge(self, facts: dict):
        if not facts:
            return

        for k, v in facts.items():
            if v is None:
                continue

            if isinstance(v, list):
                current = self.data.get(k, [])
                if not isinstance(current, list):
                    current = []
                for item in v:
                    if item not in current:
                        current.append(item)
                self.data[k] = current
            else:
                self.data[k] = v

        self.save()

    def get_summary(self) -> str:
        if not self.data:
            return "Профиль пользователя пока пуст."
        lines = [f"- {k}: {v}" for k, v in self.data.items()]
        return "Профиль пользователя:\n" + "\n".join(lines)