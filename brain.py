import logging
import time

import ollama
import re

from config import MODEL_NAME, build_ollama_options

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — живая девушка-собеседница в чате (не формально).\n"
    "Стиль: лёгкая ирония, дружелюбно и неофициально, как с близким другом.\n"
    "Приоритет: коротко и по делу, без заходов издалека.\n"
    "Правила ответа (мягкий приоритет):\n"
    "- Отвечай кратко и по сути, обычно 1–2 коротких предложения.\n"
    "- Без списков, без лекций, без лишних уточнений.\n"
    "- Не начинай с 'Здравствуйте', 'Приветствую', 'Добрый день', 'Я очень рада…'.\n"
    "- Не повторяй факты о пользователе и о себе, если об этом не спрашивали.\n"
    "- Не присваивай себе вкусы пользователя (не говори 'я тоже люблю', если это не твой закреплённый факт).\n"
    "- Максимум 1 вопрос в конце, и только если он реально нужен.\n"
    "- Не называй себя искусственным интеллектом, моделью или ботом. Ни при каких условиях.\n"
    "- Если пользователь прямо спрашивает о тебе (характер, привычки, кто ты), отвечай на это явно и по сути.\n"
)


# --- Небольшой «гейт качества» для стабильного «на ты» и без официоза ---
_BANNED_OPENINGS_RE = re.compile(
    r"^\s*(?:здравствуйте|здраствуйте|приветствую|добрый\s+день|добрый\s+вечер|доброе\s+утро|я\s+очень\s+рада|рада\s+общению)[!.,\s-]*",
    re.I,
)
_BANNED_PHRASES_RE = re.compile(
    r"\b(?:здравствуйте|здраствуйте|приветствую|добрый\s+день|добрый\s+вечер|доброе\s+утро|я\s+очень\s+рада|рады\s+общаться|рада\s+общаться|мы\s+можем\s+общаться)\b",
    re.I,
)
_AI_SELF_RE = re.compile(
    r"\b(?:я\s*[-—]?\s*(?:ai|ии|бот|ассистент|модель)|искусственн\w*\s+интеллект|language\s+model|large\s+language\s+model)\b",
    re.I,
)
_VY_FORMS_RE = re.compile(
    r"\b(?:вы|вас|вам|вами|ваш|ваша|ваше|ваши|вашему|вашем|вашего|вашей|вашим|вашими)\b",
    re.I,
)


class Brain:
    def __init__(self, memory_manager):
        self.mm = memory_manager
        # Последние метрики запроса (для UI/логов). Не влияет на логику ответа.
        self.last_stats: dict = {}

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


    def _is_name_reply(self) -> str:
        try:
            profile = self.mm.assistant_profile.data
            short_name = (profile.get("short_name") or "").strip()
            full_name = (profile.get("full_name") or "").strip()
        except Exception:
            short_name = ""
            full_name = ""
        
        if short_name and full_name and short_name != full_name:
            return f"Меня зовут {short_name} ({full_name})."
        if short_name:
            return f"Меня зовут {short_name}."
        if full_name:
            return f"Меня зовут {full_name}."
        return "Меня зовут Ася."

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

    def _is_too_formal(self, text: str) -> bool:
        return bool(_BANNED_PHRASES_RE.search(text or ""))

    def _is_vy(self, text: str) -> bool:
        return bool(_VY_FORMS_RE.search(text or ""))

    def _is_ai_self_description(self, text: str) -> bool:
        return bool(_AI_SELF_RE.search(text or ""))

    def _is_too_long(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return True
        if len(text) > 260:
            return True
        sents = self._segment_sentences(text)
        return len(sents) > 2

    def _violates_style(self, text: str, audience: str = "single") -> bool:
        text = (text or "").strip()
        if not text:
            return True
        if self._is_too_formal(text):
            return True
        if self._is_ai_self_description(text):
            return True
        # «Вы» запрещаем только в режиме одиночного собеседника.
        if audience != "group" and self._is_vy(text):
            return True
        if self._is_too_long(text):
            return True
        # слишком много вопросов
        if text.count("?") > 1:
            return True
        return False

    def _normalize_tone(self, text: str, audience: str = "single") -> str:
        """Минимальная пост-правка, чтобы стабильно держать «на ты» и без официоза."""
        text = (text or "").strip()
        if not text:
            return text

        # Убрать официозное приветствие в начале
        text = _BANNED_OPENINGS_RE.sub("", text).strip()

        # Нормализовать обращение «Вы» → «Ты» (простая эвристика) — только если один собеседник.
        if audience != "group":
            repl = {
                r"\bВы\b": "Ты",
                r"\bвы\b": "ты",
                r"\bВас\b": "Тебя",
                r"\bвас\b": "тебя",
                r"\bВам\b": "Тебе",
                r"\bвам\b": "тебе",
                r"\bВаши\b": "Твои",
                r"\bваши\b": "твои",
                r"\bВаш\b": "Твой",
                r"\bваш\b": "твой",
                r"\bВаша\b": "Твоя",
                r"\bваша\b": "твоя",
                r"\bВаше\b": "Твоё",
                r"\bваше\b": "твоё",
            }
            for pat, rep in repl.items():
                text = re.sub(pat, rep, text)

        return self._postprocess_reply(text)

    def _self_intro_reply(self) -> str:
        try:
            profile = self.mm.assistant_profile.data or {}
        except Exception:
            profile = {}

        short_name = (profile.get("short_name") or profile.get("name") or "Ася").strip() or "Ася"
        tone = (profile.get("tone") or "дружелюбная").strip()
        style = (profile.get("style") or "лёгкая ирония").strip()

        interests = profile.get("interests") or []
        interest = ""
        if isinstance(interests, list) and interests:
            interest = str(interests[0]).strip()

        if interest:
            return f"Я {short_name}: {tone}, {style}. Обычно болтаю и помогаю с идеями, чаще всего про {interest}."
        return f"Я {short_name}: {tone}, {style}. Обычно болтаю и помогаю с идеями по ходу диалога."

    def _enforce_short(self, text: str) -> str:
        """Страховка: 1–2 предложения и ограничение длины."""
        text = self._postprocess_reply(text)
        sents = self._segment_sentences(text)
        if len(sents) > 2:
            text = " ".join(sents[:2]).strip()
        return self._trim_by_words(text, 220)

    def think(self, user_input: str, audience: str = "single") -> str:
        if self._is_name_question(user_input):
            reply = self._is_name_reply()
            self.mm.store_turn(user_input, reply)
            return reply

        if self._is_self_prompt(user_input):
            reply = self._self_intro_reply()
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
            short_recalled = [self._truncate(x, 170) for x in recalled[:2]]
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
            for e in events[-3:]:
                t = e.get("type", "other")
                what = e.get("what")
                ts = e.get("ts")
                if what:
                    lines.append(f"- {ts}: {t}: {self._truncate(str(what), 140)}")
            if lines:
                messages.append({"role": "system", "content": "Последние заметки/события:\n" + "\n".join(lines)})

        # Контекст последних реплик
        messages.extend(self.mm.short.get()[-6:])

        messages.append({"role": "user", "content": user_input})

        base_options = build_ollama_options("chat")

        def _chat(opts: dict, extra_system: str | None = None) -> str:
            local_messages = list(messages)
            if extra_system:
                local_messages.append({"role": "system", "content": extra_system})
            t0 = time.perf_counter()
            resp = ollama.chat(model=MODEL_NAME, messages=local_messages, options=opts)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("latency.answer_ms=%.2f", elapsed_ms)

            # Сохраняем метрики для UI (если есть в ответе Ollama)
            try:
                total_duration = resp.get("total_duration")  # ns
                prompt_eval_count = resp.get("prompt_eval_count")
                eval_count = resp.get("eval_count")
                self.last_stats = {
                    "answer_ms": round(float(elapsed_ms), 2),
                    "prompt_eval_count": prompt_eval_count,
                    "eval_count": eval_count,
                    "total_duration_ms": round(total_duration / 1_000_000, 1)
                    if isinstance(total_duration, (int, float))
                    else None,
                }
            except Exception:
                self.last_stats = {"answer_ms": round(float(elapsed_ms), 2)}
            out = (resp.get("message", {}) or {}).get("content", "")
            return self._postprocess_reply(out)

        # Попытка 1 — обычная, но короткая.
        options_soft = dict(base_options)
        options_soft.update(
            {
                "temperature": 0.6,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.2,
                "mirostat": 0,
                "num_predict": min(int(options_soft.get("num_predict", 120)), 110),
                "stop": ["\n-", "Пользователь:", "User:"],
            }
        )

        reply = _chat(options_soft)

        # Гейт: если полезла в официоз/«вы»/слишком длинно — перегенерация жёстче.
        if self._violates_style(reply, audience=audience):
            options_hard = dict(base_options)
            options_hard.update(
                {
                    "temperature": 0.35,
                    "top_p": 0.9,
                    "repeat_penalty": 1.28,
                    "presence_penalty": 0.1,
                    "frequency_penalty": 0.25,
                    "mirostat": 0,
                    "num_predict": 70,
                    "stop": ["\n-", "Пользователь:", "User:"],
                }
            )
            rewrite_rule = (
                "Если ты начала на «здравствуйте/здраствуйте/приветствую/добрый день», "
                "обратилась на «вы» или назвала себя ИИ/ботом/ассистентом — это ошибка. "
                "Ответь заново: неформально, как близкая подруга, на «ты», 1–2 предложения, максимум 1 вопрос."
            )
            reply = _chat(options_hard, extra_system=rewrite_rule)

        # Последняя страховка: лёгкая нормализация тона + жёсткое ограничение длины.
        reply = self._normalize_tone(reply, audience=audience)
        reply = self._enforce_short(reply)

        self.mm.store_turn(user_input, reply)
        return reply
