import logging
import re

from prompts.chat import SYSTEM_CHAT_PROMPT
from prompts.handlers import PromptResponse, run_chat_prompt

logger = logging.getLogger(__name__)

_BANNED_OPENINGS_RE = re.compile(
    r"^\s*(?:здравствуйте|здраствуйте|приветствую|добрый\s+день|добрый\s+вечер|доброе\s+утро|я\s+очень\s+рада|рада\s+общению)[!.,\s-]*",
    re.I,
)


class Brain:
    def __init__(self, memory_manager):
        self.mm = memory_manager
        self.last_stats: PromptResponse | None = None

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

    def _is_too_long(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return True
        if len(text) > 320:
            return True
        return len(self._segment_sentences(text)) > 3

    def _normalize_tone(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text
        text = _BANNED_OPENINGS_RE.sub("", text).strip()
        return self._postprocess_reply(text)

    def _enforce_short(self, text: str) -> str:
        text = self._postprocess_reply(text)
        sents = self._segment_sentences(text)
        if len(sents) > 3:
            text = " ".join(sents[:3]).strip()
        return self._trim_by_words(text, 260)

    def _run_chat(self, messages: list[dict[str, str]], options_override: dict | None = None) -> str:
        response = run_chat_prompt(
            task_type="chat",
            messages=messages,
            options_override=options_override,
        )
        self.last_stats = response
        return self._postprocess_reply(response.content)

    def think(self, user_input: str, audience: str = "single") -> str:
        recalled = self.mm.recall(user_input, n_results=5)
        events = self.mm.last_events(20)

        messages = [{"role": "system", "content": SYSTEM_CHAT_PROMPT}]

        try:
            messages.append({"role": "system", "content": self.mm.assistant_profile.summary()})
        except Exception:
            pass
        try:
            messages.append({"role": "system", "content": self.mm.user_profile.get_summary()})
        except Exception:
            pass

        if recalled:
            short_recalled = [self._truncate(x, 170) for x in recalled[:2]]
            memories_block = "\n\n".join(short_recalled)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Релевантная память (используй только если помогает ответить точнее; "
                        "не цитируй и не пересказывай её пользователю):\n" + memories_block
                    ),
                }
            )

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

        messages.extend(self.mm.short.get()[-6:])
        messages.append({"role": "user", "content": user_input})

        options_soft = {
            "temperature": 0.55,
            "top_p": 0.9,
            "repeat_penalty": 1.12,
            "presence_penalty": 0.15,
            "frequency_penalty": 0.15,
            "mirostat": 0,
            "num_predict": 140,
            "stop": ["\n-", "Пользователь:", "User:"],
        }

        reply = self._run_chat(messages, options_override=options_soft)

        if self._is_too_long(reply):
            options_hard = {
                "temperature": 0.35,
                "top_p": 0.85,
                "repeat_penalty": 1.2,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                "mirostat": 0,
                "num_predict": 90,
                "stop": ["\n-", "Пользователь:", "User:"],
            }
            rewrite_rule = "Переформулируй кратко и по делу: максимум 3 коротких предложения, без списков и официоза."
            reply = self._run_chat(messages + [{"role": "system", "content": rewrite_rule}], options_override=options_hard)

        reply = self._normalize_tone(reply)
        reply = self._enforce_short(reply)

        self.mm.store_turn(user_input, reply)
        return reply
