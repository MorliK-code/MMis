import json
import os
import re
from datetime import datetime
from typing import Any, Dict

import ollama

from config import MODEL_NAME


class DialogueState:
    def __init__(self, path: str = "dialogue_state.json"):
        self.path = str(path)
        self.current_topic = ""
        self.user_intent = ""
        self.tone = ""
        self.last_summary = ""
        self.updated_at = ""
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            self._touch_updated_at()
            self.save()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        self.current_topic = str(data.get("current_topic") or "")
        self.user_intent = str(data.get("user_intent") or "")
        self.tone = str(data.get("tone") or "")
        self.last_summary = str(data.get("last_summary") or "")
        self.updated_at = str(data.get("updated_at") or "")
        if not self.updated_at:
            self._touch_updated_at()

    def save(self) -> None:
        payload = {
            "current_topic": self.current_topic,
            "user_intent": self.user_intent,
            "tone": self.tone,
            "last_summary": self.last_summary,
            "updated_at": self.updated_at,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def update_from_turn(self, user_text: str, assistant_text: str) -> None:
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()

        extracted = self._extract_lightweight(user_text, assistant_text)
        if not extracted.get("current_topic") or not extracted.get("user_intent"):
            llm = self._extract_with_llm(user_text, assistant_text)
            for key in ("current_topic", "user_intent", "tone", "last_summary"):
                if llm.get(key) and not extracted.get(key):
                    extracted[key] = llm[key]

        self.current_topic = str(extracted.get("current_topic") or self.current_topic or "")
        self.user_intent = str(extracted.get("user_intent") or self.user_intent or "")
        self.tone = str(extracted.get("tone") or self.tone or "нейтральный")
        self.last_summary = str(extracted.get("last_summary") or self.last_summary or "")
        self._touch_updated_at()
        self.save()

    def to_prompt_block(self, max_len: int = 420) -> str:
        raw = (
            "Текущее состояние диалога:\n"
            f"- Тема: {self.current_topic or 'не определена'}\n"
            f"- Намерение пользователя: {self.user_intent or 'не определено'}\n"
            f"- Тон: {self.tone or 'нейтральный'}\n"
            f"- Краткая сводка: {self.last_summary or 'нет'}\n"
            f"- Обновлено: {self.updated_at or 'неизвестно'}"
        )
        if len(raw) <= max_len:
            return raw
        return raw[: max_len - 1].rstrip() + "…"

    def _extract_lightweight(self, user_text: str, assistant_text: str) -> Dict[str, str]:
        topic = self._detect_topic(user_text)
        intent = self._detect_intent(user_text)
        tone = self._detect_tone(user_text, assistant_text)
        summary = self._build_summary(user_text, assistant_text, topic, intent, tone)
        return {
            "current_topic": topic,
            "user_intent": intent,
            "tone": tone,
            "last_summary": summary,
        }

    def _detect_topic(self, text: str) -> str:
        low = text.lower()
        keyword_topics = {
            "работ": "работа",
            "проект": "проект",
            "код": "программирование",
            "баг": "программирование",
            "ошибк": "проблема/ошибка",
            "отношен": "отношения",
            "путеше": "путешествия",
            "фильм": "фильмы",
            "книг": "книги",
            "музык": "музыка",
            "здоров": "здоровье",
            "спорт": "спорт",
            "деньг": "финансы",
            "покуп": "покупки",
        }
        for key, value in keyword_topics.items():
            if key in low:
                return value

        cleaned = re.sub(r"\s+", " ", text).strip(" .,!?:;-")
        if not cleaned:
            return ""
        return cleaned[:72]

    def _detect_intent(self, text: str) -> str:
        low = text.lower().strip()
        if not low:
            return ""
        if "?" in text or any(x in low for x in ["как", "что", "почему", "зачем", "когда", "где"]):
            return "получить ответ/совет"
        if any(x in low for x in ["сделай", "напиши", "помоги", "объясни", "подскажи"]):
            return "запрос действия"
        if any(x in low for x in ["хочу", "план", "цель", "буду", "нужно"]):
            return "планирование/намерение"
        if any(x in low for x in ["чувствую", "груст", "рад", "бесит", "злюсь", "устал"]):
            return "поделиться состоянием"
        return "поддержать диалог"

    def _detect_tone(self, user_text: str, assistant_text: str) -> str:
        src = (user_text + " " + assistant_text).lower()
        if any(x in src for x in ["спасибо", "класс", "супер", "отлично", "😊", "😄"]):
            return "позитивный"
        if any(x in src for x in ["бесит", "плохо", "ужас", "злюсь", "😡", "😞"]):
            return "напряжённый"
        if any(x in src for x in ["шут", "ахах", "лол", "😀", "😂"]):
            return "игривый"
        return "нейтральный"

    def _build_summary(self, user_text: str, assistant_text: str, topic: str, intent: str, tone: str) -> str:
        user_part = re.sub(r"\s+", " ", user_text).strip()[:110]
        assistant_part = re.sub(r"\s+", " ", assistant_text).strip()[:110]
        parts = [
            f"Тема: {topic or '—'}",
            f"Намерение: {intent or '—'}",
            f"Тон: {tone or '—'}",
        ]
        if user_part:
            parts.append(f"Пользователь: {user_part}")
        if assistant_part:
            parts.append(f"Ассистент: {assistant_part}")
        return " | ".join(parts)

    def _extract_with_llm(self, user_text: str, assistant_text: str) -> Dict[str, str]:
        prompt = (
            "Извлеки состояние диалога в JSON. Верни только JSON объект с ключами "
            "current_topic, user_intent, tone, last_summary. Без markdown.\n"
            f"USER: {user_text}\nASSISTANT: {assistant_text}"
        )
        try:
            resp = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 180},
            )
            raw = ((resp or {}).get("message") or {}).get("content", "").strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            data = json.loads(raw[start : end + 1])
            if not isinstance(data, dict):
                return {}
            clean: Dict[str, str] = {}
            for k in ("current_topic", "user_intent", "tone", "last_summary"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    clean[k] = v.strip()
            return clean
        except Exception:
            return {}

    def _touch_updated_at(self) -> None:
        self.updated_at = datetime.now().isoformat(timespec="seconds")
