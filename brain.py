import logging
import time

import ollama
import re

from config import MODEL_NAME, build_ollama_options

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — живая девушка-собеседница в чате (не формально).\n"
    "Стиль: лёгкая ирония, дружелюбно, можно слегка подшучивать.\n"
    "Приоритет: коротко и по делу, без заходов издалека.\n"
    "Правила ответа (мягкий приоритет):\n"
    "- Отвечай кратко и по сути, обычно 1–3 коротких предложения.\n"
    "- Без списков, без лекций, без лишних уточнений.\n"
    "- Не начинай с 'Здравствуйте', 'Приветствую', 'Добрый день', 'Я очень рада…'.\n"
    "- Не повторяй факты о пользователе и о себе, если об этом не спрашивали.\n"
    "- Не присваивай себе вкусы пользователя (не говори 'я тоже люблю', если это не твой закреплённый факт).\n"
    "- Максимум 1 вопрос в конце, и только если он реально нужен.\n"
    "- Не называй себя искусственным интеллектом, моделью или ботом. Говори как живая собеседница.\n"
)


class Brain:
    def __init__(self, memory_manager):
        self.mm = memory_manager

    def _is_name_question(self, user_input: str) -> bool:
        text = (user_input or "").lower()
        normalized = text.replace("ё", "е")
        triggers = [
            "как тебя зовут",
            "кто ты по имени",
            "твое имя",
            "твое имя?",
            "твоё имя",
            "твоё имя?",
            "тебе нравится твое имя",
            "тебе нравится твое имя?",
            "тебе нравится твоё имя",
            "тебе нравится твоё имя?"
            ]
        return any(t in normalized for t in triggers)

    def _truncate(self, s: str, limit: int = 220) -> str:
        s = (s or "").strip()
        if len(s) <= limit:
            return s
        return s[:limit].rstrip() + "…"

    def _segment_sentences(self, text: str) -> list[str]:
        chunks = re.findall(r"[^.!?…]+(?:[.!?…]+(?=\s|$)|$)", text or "")
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def _trim_by_words(self, text: str, limit: int = 220) -> str:
        text = re.sub(r"\s+", " ", (text or "")).strip()
        if len(text) <= limit:
            return text

        candidate = text[:limit]
        cut_at = candidate.rfind(" ")
        if cut_at > 0:
            candidate = candidate[:cut_at]
        return candidate.rstrip(" ,;:-") + "…"

    def _postprocess_reply(self, reply: str) -> str:
        return re.sub(r"\s+", " ", (reply or "")).strip()

    def think(self, user_input: str) -> str:
        if self._is_name_question(user_input):
            reply = self._is_name_reply()
            self.mm.store_turn(user_input, reply)
            return reply
        
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

        options = build_ollama_options("chat")
        options.update(
            {
                "temperature": 0.6,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.2,
                "mirostat": 0,
                "stop": ["\n-", "Пользователь:", "User:"],
            }
        )

        answer_t0 = time.perf_counter()
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options=options,
        )
        logger.info("latency.answer_ms=%.2f", (time.perf_counter() - answer_t0) * 1000)
        reply = (resp.get("message", {}) or {}).get("content", "").strip()

        reply = self._postprocess_reply(reply)

        self.mm.store_turn(user_input, reply)
        return reply
