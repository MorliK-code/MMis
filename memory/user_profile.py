import json
from pathlib import Path
from typing import Any, Dict, List, Union


class UserProfile:
    def __init__(self, path: Union[str, Path] = "user_profile.json"):
        self.path = Path(path)
        self.data: Dict[str, Any] = {}
        self.load()

    def load(self):
        if self.path.exists():
            with self.path.open("r", encoding="utf-8-sig") as f:
                self.data = json.load(f)
        else:
            self.data = {}

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

    def add_memory_note(self, note: str):
        text = str(note or "").strip()
        if not text:
            return
        notes = self.data.get("memory_notes", [])
        if not isinstance(notes, list):
            notes = []
        if text not in notes:
            notes.append(text)
            self.data["memory_notes"] = notes[-120:]
            self.save()

    def get_summary(self) -> str:
        if not self.data:
            return "Профиль пользователя пока пуст."

        lines: List[str] = []
        for k, v in self.data.items():
            if k == "memory_notes" and isinstance(v, list):
                if v:
                    lines.append("- memory_notes:")
                    lines.extend(f"  - {x}" for x in v[-10:])
                continue
            lines.append(f"- {k}: {v}")

        return "Профиль пользователя:\n" + "\n".join(lines)
