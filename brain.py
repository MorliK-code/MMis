import ollama
import re

from config import MODEL_NAME, RESPONSE_NUM_PREDICT

SYSTEM_PROMPT = (
    "Ты — живая девушка-собеседница в чате (не формально).\n"
    "Стиль: лёгкая ирония, дружелюбно, можно слегка подшучивать.\n"
    "Приоритет: коротко и по делу, без заходов издалека.\n"
    "Жёсткие правила ответа:\n"
    "- 1–2 коротких предложения.\n"
    "- Запрещены многосоставные и перегруженные конструкции с несколькими смысловыми блоками в одном предложении.\n"
    "- Без списков, без лекций, без лишних уточнений.\n"
    "- Не начинай с 'Здравствуйте', 'Приветствую', 'Добрый день', 'Я очень рада…'.\n"
    "- Не повторяй факты о пользователе и о себе, если об этом не спрашивали.\n"
    "- Не присваивай себе вкусы пользователя (не говори 'я тоже люблю', если это не твой закреплённый факт).\n"
    "- Максимум 1 вопрос в конце, и только если он реально нужен.\n"
)


class Brain:
    def __init__(self, memory_manager):
        self.mm = memory_manager

    def _truncate(self, s: str, limit: int = 220) -> str:
        s = (s or "").strip()
        if len(s) <= limit:
            return s
        return s[:limit].rstrip() + "…"

    def _segment_sentences(self, text: str) -> list[str]:
        chunks = re.findall(r"[^.!?…]+(?:[.!?…]+(?=\s|$)|$)", text or "")
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def _trim_by_words(self, text: str, limit: int = 150) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if len(text) <= limit:
            return text

        candidate = text[:limit]
        cut_at = candidate.rfind(" ")
        if cut_at > 0:
            candidate = candidate[:cut_at]
        return candidate.rstrip(" ,;:-") + "…"

    def _postprocess_reply(self, reply: str) -> str:
        reply = re.sub(r"\s+", " ", (reply or "")).strip()
        if not reply:
            return ""

        parts = self._segment_sentences(reply)
        if parts:
            reply = " ".join(parts[:2])
        return self._trim_by_words(reply, limit=150)

    def think(self, user_input: str) -> str:
        recalled = self.mm.recall(user_input, n_results=5)
        events = self.mm.last_events(20)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Профили (для согласованности)
        try:
            messages.append({"role": "system", "content": self.mm.assistant_profile.summary()})
        except Exception:
            pass
        try:
            messages.append({"role": "system", "content": self.mm.user_profile.get_summary()})
        except Exception:
            pass

        # Воспоминания — только самые релевантные, и в усечённом виде, чтобы не тащить болтовню
        if recalled:
            short_recalled = [self._truncate(x, 240) for x in recalled[:3]]
            memories_block = "\n\n".join(short_recalled)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Релевантные воспоминания (используй только если реально помогают ответить кратко; "
                        "НЕ цитируй и НЕ пересказывай их пользователю):\n" + memories_block
                    ),
                }
            )

        # События — тоже кратко
        if events:
            lines = []
            for e in events[-6:]:
                t = e.get("type", "other")
                what = e.get("what")
                ts = e.get("ts")
                if what:
                    lines.append(f"- {ts}: {t}: {self._truncate(str(what), 140)}")
            if lines:
                messages.append({"role": "system", "content": "Последние заметки/события:\n" + "\n".join(lines)})

        # Контекст последних реплик
        messages.extend(self.mm.short.get())

        messages.append({"role": "user", "content": user_input})

        resp = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.6,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.2,
                "mirostat": 0,
                "num_predict": min(RESPONSE_NUM_PREDICT, 48),
                "stop": ["\n\n", "\n-", "Пользователь:", "User:"],
            },
        )
        reply = self._postprocess_reply((resp.get("message", {}) or {}).get("content", ""))

        self.mm.store_turn(user_input, reply)
        return reply
