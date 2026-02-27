import logging
import json
import re
import time
from urllib import request as urllib_request

import ollama

from config import (
    MODEL_FALLBACKS,
    MODEL_NAME,
    MMIS_CHAT_EVENTS_LIMIT,
    MMIS_CHAT_PROOFREAD,
    MMIS_CHAT_PROOFREAD_STRICT,
    MMIS_CHAT_RECALL_RESULTS,
    build_ollama_options,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Личный чат. Отвечай на русском языке естественно и свободно.\n"
    "Если пользователь просит длинный ответ, давай длинный ответ."
)

_BANNED_OPENINGS_RE = re.compile(
    r"^\s*(?:здравствуйте|здраствуйте|приветствую|добрый\s+день|добрый\s+вечер|доброе\s+утро|я\s+очень\s+рада|рада\s+общению)[!.,\s-]*",
    re.I,
)
_CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_THINK_TAG_RE = re.compile(r"<\s*/?\s*think\b[^>]*>", re.I)


class Brain:
    def __init__(self, memory_manager):
        self.mm = memory_manager
        self.last_stats: dict = {}
        self.last_thinking: str = ""
        self._runtime_model = MODEL_NAME
        self._thinking_enabled = True
        self._stream_think_mode = False
        self._stream_tag_pending = ""
        self._models_cache: list[str] = []
        self._models_cache_ts: float = 0.0
        self._models_cache_ttl_sec: float = 2.0

    def is_thinking_enabled(self) -> bool:
        return bool(self._thinking_enabled)

    def set_thinking_enabled(self, enabled: bool) -> None:
        self._thinking_enabled = bool(enabled)

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

    @staticmethod
    def _compact_block(text: str, max_chars: int = 700, max_lines: int = 14) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        s = "\n".join(lines)
        if len(s) > max_chars:
            s = s[:max_chars].rstrip() + "…"
        return s

    def _postprocess_reply(self, reply: str) -> str:
        s = str(reply or "")
        s = self._strip_think_tags(s)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        return _THINK_TAG_RE.sub("", str(text or ""))

    def _assistant_do_not_say(self, limit: int | None = None) -> list[str]:
        try:
            d = getattr(self.mm.assistant_profile, "data", {}) or {}
        except Exception:
            d = {}

        raw = d.get("do_not_say", [])
        if not isinstance(raw, list):
            raw = [raw]

        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if not text or self._looks_broken_text(text):
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)

        if isinstance(limit, int) and limit > 0:
            return out[:limit]
        return out

    def _assistant_guardrails_hint(self) -> str:
        banned = self._assistant_do_not_say(limit=24)
        if not banned:
            return ""
        lines = "\n".join(f"- {self._truncate(x, 80)}" for x in banned)
        return (
            "Строгий запрет: никогда не используй в ответе следующие слова/фразы:\n"
            f"{lines}\n"
            "Если мысль требует этих слов, переформулируй без них."
        )

    def _enforce_do_not_say(self, text: str) -> str:
        out = str(text or "")
        banned = self._assistant_do_not_say()
        if not out or not banned:
            return self._postprocess_reply(out)

        for token in banned:
            if re.search(r"\s", token):
                pattern = re.compile(re.escape(token), flags=re.IGNORECASE)
            else:
                pattern = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)", flags=re.IGNORECASE)
            out = pattern.sub("", out)

        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"([,.;:!?]){2,}", r"\1", out)
        out = re.sub(r"\s{2,}", " ", out).strip(" \t\r\n,;:-")

        if not out:
            out = "Поняла."
        return self._postprocess_reply(out)

    @staticmethod
    def _strip_cjk(text: str) -> str:
        return _CJK_RE.sub("", text or "")

    @staticmethod
    def _looks_broken_text(text: str) -> bool:
        s = str(text or "")
        if not s:
            return False
        # Typical mojibake fragments from broken Cyrillic transcoding.
        bad = len(re.findall(r"(?:Р.|С.|Ѓ.|Ђ.|Ќ.|љ.|ў.)", s))
        return bad >= 6

    def _normalize_tone(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text
        text = _BANNED_OPENINGS_RE.sub("", text).strip()
        text = re.sub(
            r"^\s*привет[!.,\s-]*я\s+(?:ai|ии|нейросеть|ассистент)[^.!?]*[.!?]?\s*",
            "",
            text,
            flags=re.I,
        ).strip()
        return self._postprocess_reply(text)

    def _assistant_identity_hint(self) -> str:
        # Короткая identity-подсказка для fast-mode без тяжёлого profile summary.
        try:
            d = getattr(self.mm.assistant_profile, "data", {}) or {}
        except Exception:
            d = {}
        name = str(d.get("short_name") or d.get("name") or "Она").strip() or "Она"
        tone = str(d.get("tone") or "дружелюбная").strip() or "дружелюбная"
        address = str(d.get("address") or "ты").strip() or "ты"
        return (
            f"Персона: тебя зовут {name}. "
            f"Стиль: {tone}, личный чат. "
            f"Обращайся к пользователю на '{address}'. "
            "Не используй формулировки про AI-ассистента."
        )

    @staticmethod
    def _extract_thinking_and_answer(raw_text: str, explicit_thinking: str = "") -> tuple[str, str]:
        text = str(raw_text or "")
        explicit = str(explicit_thinking or "").strip()
        think_parts: list[str] = []
        if explicit:
            think_parts.append(explicit)

        # Qwen-like thinking tags.
        tags = re.findall(r"<\s*think\b[^>]*>(.*?)<\s*/\s*think\s*>", text, flags=re.S | re.I)
        for t in tags:
            t = str(t or "").strip()
            if t:
                think_parts.append(t)

        answer = re.sub(r"<\s*think\b[^>]*>.*?<\s*/\s*think\s*>", "", text, flags=re.S | re.I).strip()
        answer = Brain._strip_think_tags(answer).strip()
        thinking = "\n\n".join(x for x in think_parts if x)
        return thinking, answer

    def _consume_visible_stream_text(self, piece: str) -> str:
        data = (self._stream_tag_pending or "") + str(piece or "")
        if not data:
            return ""
        out: list[str] = []
        i = 0
        open_tag = "<think>"
        close_tag = "</think>"
        low = data.lower()

        while i < len(data):
            if self._stream_think_mode:
                idx = low.find(close_tag, i)
                if idx < 0:
                    rem = data[i:]
                    self._stream_tag_pending = rem[-8:] if len(rem) > 8 else rem
                    return "".join(out)
                i = idx + len(close_tag)
                self._stream_think_mode = False
                continue

            idx = low.find(open_tag, i)
            if idx < 0:
                rem = data[i:]
                keep = 0
                for k in range(1, len(open_tag)):
                    if rem.lower().endswith(open_tag[:k]):
                        keep = k
                if keep:
                    out.append(rem[:-keep])
                    self._stream_tag_pending = rem[-keep:]
                else:
                    out.append(rem)
                    self._stream_tag_pending = ""
                visible = "".join(out)
                return self._strip_think_tags(visible)

            out.append(data[i:idx])
            i = idx + len(open_tag)
            self._stream_think_mode = True

        self._stream_tag_pending = ""
        visible = "".join(out)
        return self._strip_think_tags(visible)

    @staticmethod
    def _looks_like_prompt_leak(text: str) -> bool:
        s = str(text or "").lower()
        if not s:
            return False
        markers = (
            "system prompt",
            "внутренние инструкции",
            "служебные данные",
            "role:",
            "assistant:",
            "user:",
            "```",
        )
        return any(m in s for m in markers)

    def _proofread_reply(self, reply: str, messages: list[dict], keep_alive, base_options: dict) -> str:
        source = self._postprocess_reply(reply)
        if not source:
            return source

        local_messages = list(messages)
        strict_tail = (
            " Проверяй максимально строго: орфография, пунктуация, согласование времен и падежей."
            if MMIS_CHAT_PROOFREAD_STRICT
            else ""
        )
        local_messages.append(
            {
                "role": "system",
                "content": (
                    "Сделай только вычитку последнего ответа на русском: исправь орфографию, пунктуацию и грамматику, "
                    "не меняй смысл, стиль, тон и объем. Верни только финальный исправленный текст без комментариев."
                    + strict_tail
                ),
            }
        )
        local_messages.append({"role": "assistant", "content": source})

        options = dict(base_options)
        options.update(
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "mirostat": 0,
                "num_predict": min(int(options.get("num_predict", 160)), 170),
                "stop": ["\n-"],
            }
        )

        try:
            resp = ollama.chat(
                model=self._runtime_model,
                messages=local_messages,
                think=False,
                options=options,
                keep_alive=keep_alive,
            )
            corrected = (resp.get("message", {}) or {}).get("content", "")
        except Exception:
            return source

        corrected = self._postprocess_reply(self._strip_cjk(corrected))
        if not corrected:
            return source
        if self._looks_like_prompt_leak(corrected):
            return source
        # Avoid destructive rewrites: correction should stay close to source text.
        if len(corrected) > max(len(source) * 1.6, len(source) + 80):
            return source
        return corrected

    def _recover_reply_from_thinking(
        self,
        user_input: str,
        thinking: str,
        keep_alive,
        base_options: dict,
    ) -> str:
        t = self._postprocess_reply(self._strip_cjk(thinking))
        if not t:
            return ""
        options = dict(base_options)
        options.update(
            {
                "temperature": 0.25,
                "top_p": 0.9,
                "repeat_penalty": 1.05,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "mirostat": 0,
            }
        )
        msgs = [
            {
                "role": "system",
                "content": (
                    "Сформируй финальный ответ пользователю на русском по внутренним заметкам. "
                    "Никаких тегов <think>, никаких служебных пояснений. Только готовый ответ."
                ),
            },
            {"role": "user", "content": f"Запрос пользователя: {user_input}\n\nВнутренние заметки:\n{t}"},
        ]
        try:
            resp = ollama.chat(
                model=self._runtime_model,
                messages=msgs,
                think=False,
                options=options,
                keep_alive=keep_alive,
            )
            out = (resp.get("message", {}) or {}).get("content", "")
            _, answer = self._extract_thinking_and_answer(out, "")
            return self._postprocess_reply(self._strip_cjk(answer))
        except Exception:
            return ""

    @staticmethod
    def _needs_memory_lookup(user_input: str) -> bool:
        text = (user_input or "").strip().lower()
        if not text:
            return False
        if len(text) >= 60:
            return True
        triggers = (
            "помнишь",
            "напомни",
            "мы обсуждали",
            "в прошлый раз",
            "раньше",
            "мой",
            "моя",
            "мои",
            "мне нравится",
            "как меня",
            "что я",
        )
        return any(t in text for t in triggers)

    @staticmethod
    def _needs_chat_history_lookup(user_input: str) -> bool:
        text = (user_input or "").strip().lower()
        if not text:
            return False
        triggers = (
            "помнишь",
            "напомни",
            "что я писал",
            "что ты отвечала",
            "мы обсуждали",
            "в прошлый раз",
            "раньше",
            "вчера",
            "до этого",
            "история",
            "диалог",
        )
        if any(t in text for t in triggers):
            return True
        # Длинные уточняющие вопросы тоже могут требовать истории.
        return len(text) >= 140

    @staticmethod
    def _is_light_prompt_query(user_input: str) -> bool:
        text = (user_input or "").strip().lower()
        if not text:
            return True
        # Для большинства коротких/средних запросов нужен быстрый старт без heavy-context.
        if Brain._needs_memory_lookup(text):
            return False
        return len(text) <= 120

    def _build_messages(self, user_input: str) -> list[dict]:
        light_mode = self._is_light_prompt_query(user_input)
        use_recall = (
            MMIS_CHAT_RECALL_RESULTS > 0 and self._needs_memory_lookup(user_input)
        )
        use_chat_history = self._needs_chat_history_lookup(user_input)
        recalled = self.mm.recall(user_input, n_results=MMIS_CHAT_RECALL_RESULTS) if use_recall else []
        include_heavy_context = use_recall and (not light_mode)
        events_limit = MMIS_CHAT_EVENTS_LIMIT if include_heavy_context else 0
        events = self.mm.last_events(events_limit) if events_limit > 0 else []

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "system", "content": self._assistant_identity_hint()})
        if self._thinking_enabled:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Для сложных вопросов можешь использовать внутренний reasoning/thinking. "
                        "В финальном ответе пользователю показывай только итоговый ответ."
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Не используй reasoning/thinking и не выводи теги <think>. "
                        "Отвечай сразу итоговым ответом."
                    ),
                }
            )
        if include_heavy_context:
            try:
                assistant_summary = self.mm.assistant_profile.summary()
                if assistant_summary and not self._looks_broken_text(assistant_summary):
                    compact = self._compact_block(assistant_summary, max_chars=650, max_lines=12)
                    if compact:
                        messages.append({"role": "system", "content": compact})
            except Exception:
                pass
            try:
                user_summary = self.mm.user_profile.get_summary()
                if user_summary and not self._looks_broken_text(user_summary):
                    compact = self._compact_block(user_summary, max_chars=650, max_lines=12)
                    if compact:
                        messages.append({"role": "system", "content": compact})
            except Exception:
                pass

        if recalled:
            short_recalled = [self._truncate(x, 170) for x in recalled[:2] if not self._looks_broken_text(x)]
            memories_block = "\n\n".join(short_recalled)
            if memories_block:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Релевантная память (используй только если помогает ответить точнее; "
                            "не цитируй и не пересказывай ее пользователю):\n" + memories_block
                        ),
                    }
                )

        if events:
            lines = []
            for e in events[-3:]:
                t = e.get("type", "other")
                what = e.get("what")
                ts = e.get("ts")
                if what and not self._looks_broken_text(str(what)):
                    lines.append(f"- {ts}: {t}: {self._truncate(str(what), 140)}")
            if lines:
                messages.append({"role": "system", "content": "Последние заметки/события:\n" + "\n".join(lines)})

        if use_chat_history:
            rows = self.mm.relevant_chat_messages(user_input, limit=6, scan_last=500)
            history_lines = []
            for r in rows:
                role = str(r.get("role", "")).lower()
                content = str(r.get("content", "") or "")
                if not content:
                    continue
                if self._looks_broken_text(content):
                    continue
                if role not in {"user", "assistant"}:
                    continue
                speaker = "Ты" if role == "user" else "Она"
                clean = self._strip_cjk(self._truncate(content, 150))
                if clean:
                    history_lines.append(f"- {speaker}: {clean}")
            if history_lines:
                block = self._compact_block("\n".join(history_lines), max_chars=760, max_lines=8)
                if block:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Релевантные фрагменты истории чата (используй для точности, "
                                "не цитируй дословно и не показывай как служебный блок):\n" + block
                            ),
                        }
                    )

        short_ctx = self.mm.short.get()[-4:] if light_mode else self.mm.short.get()[-6:]
        messages.extend(short_ctx)
        messages.append({"role": "user", "content": user_input})
        return messages

    def think(self, user_input: str, audience: str = "single", store_turn: bool = True) -> str:
        prep_t0 = time.perf_counter()
        messages = self._build_messages(user_input)
        prep_ms = (time.perf_counter() - prep_t0) * 1000
        self._stream_think_mode = False
        self._stream_tag_pending = ""

        base_options = build_ollama_options("chat")
        keep_alive = base_options.pop("keep_alive", None)

        def _chat(opts: dict, extra_system: str | None = None) -> tuple[str, str]:
            local_messages = list(messages)
            if extra_system:
                local_messages.append({"role": "system", "content": extra_system})

            t0 = time.perf_counter()
            try:
                resp = ollama.chat(
                    model=self._runtime_model,
                    messages=local_messages,
                    think=(True if self._thinking_enabled else False),
                    options=opts,
                    keep_alive=keep_alive,
                )
            except Exception as exc:
                if "not found" not in str(exc).lower():
                    friendly = self._friendly_ollama_error(exc)
                    if friendly:
                        raise RuntimeError(friendly) from exc
                    raise
                resolved = self._resolve_existing_model()
                if not resolved:
                    raise
                self._runtime_model = resolved
                resp = ollama.chat(
                    model=self._runtime_model,
                    messages=local_messages,
                    think=(True if self._thinking_enabled else False),
                    options=opts,
                    keep_alive=keep_alive,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("latency.answer_ms=%.2f", elapsed_ms)

            try:
                total_duration = resp.get("total_duration")
                prompt_eval_count = resp.get("prompt_eval_count")
                eval_count = resp.get("eval_count")
                eval_duration = resp.get("eval_duration")
                prompt_eval_duration = resp.get("prompt_eval_duration")
                self.last_stats = {
                    "answer_ms": round(float(elapsed_ms), 2),
                    "prep_ms": round(float(prep_ms), 2),
                    "prompt_eval_count": prompt_eval_count,
                    "eval_count": eval_count,
                    "total_duration_ms": round(total_duration / 1_000_000, 1)
                    if isinstance(total_duration, (int, float))
                    else None,
                    "eval_duration_ms": round(eval_duration / 1_000_000, 1)
                    if isinstance(eval_duration, (int, float))
                    else None,
                    "prompt_eval_duration_ms": round(prompt_eval_duration / 1_000_000, 1)
                    if isinstance(prompt_eval_duration, (int, float))
                    else None,
                }
            except Exception:
                self.last_stats = {"answer_ms": round(float(elapsed_ms), 2), "prep_ms": round(float(prep_ms), 2)}

            msg = (resp.get("message", {}) or {})
            out = msg.get("content", "")
            thinking_raw = msg.get("thinking", "")
            thinking, answer = self._extract_thinking_and_answer(out, thinking_raw)
            return (
                self._postprocess_reply(self._strip_cjk(answer)),
                self._postprocess_reply(self._strip_cjk(thinking)),
            )

        options_soft = dict(base_options)
        options_soft.update(
            {
                "temperature": 0.78,
                "top_p": 0.95,
                "repeat_penalty": 1.03,
                "presence_penalty": 0.02,
                "frequency_penalty": 0.02,
                "mirostat": 0,
            }
        )

        reply, thinking = _chat(options_soft)

        reply = self._strip_cjk(reply)
        self.last_thinking = self._postprocess_reply(self._strip_cjk(thinking)) if self._thinking_enabled else ""
        if not reply.strip() and self._thinking_enabled:
            recovered = self._recover_reply_from_thinking(user_input, self.last_thinking, keep_alive, base_options)
            if recovered:
                reply = recovered
        if not reply.strip():
            reply = "Ответ не сгенерировался. Напиши запрос еще раз."

        if store_turn:
            self.mm.store_turn(user_input, reply)
        return reply

    def _resolve_existing_model(self) -> str | None:
        preferred = []
        for x in [self._runtime_model, MODEL_NAME, *MODEL_FALLBACKS]:
            y = str(x or "").strip()
            if y and y not in preferred:
                preferred.append(y)
        installed = self.list_local_models()
        if not installed:
            return None
        for cand in preferred:
            if cand in installed:
                return cand
        return installed[0] if installed else None

    @staticmethod
    def _friendly_ollama_error(exc: Exception) -> str | None:
        s = str(exc or "")
        low = s.lower()
        if ("winerror 10061" in low) or ("connection refused" in low) or ("all connection attempts failed" in low):
            return (
                "Нет подключения к Ollama. Запусти Ollama и повтори.\n"
                "Команда: ollama serve"
            )
        if "timed out" in low:
            return "Ollama не ответил вовремя. Попробуй еще раз."
        return None

    def list_local_models(self, force_refresh: bool = False) -> list[str]:
        now = time.time()
        if (not force_refresh) and self._models_cache and (now - self._models_cache_ts) < self._models_cache_ttl_sec:
            return list(self._models_cache)

        try:
            resp = ollama.list()
        except Exception:
            resp = None

        rows = []
        if isinstance(resp, dict):
            rows = resp.get("models", []) or []
        elif resp is not None:
            rows = getattr(resp, "models", None) or []

        installed: list[str] = []
        seen: set[str] = set()
        for row in rows or []:
            name = ""
            if isinstance(row, dict):
                name = str(row.get("name") or row.get("model") or row.get("id") or "").strip()
            else:
                name = str(
                    getattr(row, "name", "")
                    or getattr(row, "model", "")
                    or getattr(row, "id", "")
                    or ""
                ).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            installed.append(name)

        if not installed:
            # Window-safe HTTP fallback to local Ollama API (no subprocess/console popups).
            try:
                req = urllib_request.Request("http://127.0.0.1:11434/api/tags", method="GET")
                with urllib_request.urlopen(req, timeout=2.5) as resp_raw:
                    payload = json.loads(resp_raw.read().decode("utf-8", errors="ignore") or "{}")
                rows2 = payload.get("models", []) if isinstance(payload, dict) else []
                for row in rows2 or []:
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("name") or row.get("model") or row.get("id") or "").strip()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    installed.append(name)
            except Exception:
                pass

        self._models_cache = list(installed)
        self._models_cache_ts = now
        return installed

    def get_runtime_model(self) -> str:
        return str(self._runtime_model or MODEL_NAME)

    def set_runtime_model(self, model_name: str) -> bool:
        target = str(model_name or "").strip()
        if not target:
            return False
        installed = self.list_local_models(force_refresh=True)
        if target not in installed:
            return False
        self._runtime_model = target
        return True

    def think_stream(self, user_input: str, on_chunk=None, on_thinking_chunk=None, store_turn: bool = True) -> str:
        prep_t0 = time.perf_counter()
        messages = self._build_messages(user_input)
        prep_ms = (time.perf_counter() - prep_t0) * 1000

        opts = dict(build_ollama_options("chat"))
        keep_alive = opts.pop("keep_alive", None)
        opts.update(
            {
                "temperature": 0.78,
                "top_p": 0.95,
                "repeat_penalty": 1.03,
                "presence_penalty": 0.02,
                "frequency_penalty": 0.02,
                "mirostat": 0,
            }
        )

        t0 = time.perf_counter()
        raw_parts: list[str] = []
        parts: list[str] = []
        last_chunk = {}
        try:
            stream = ollama.chat(
                model=self._runtime_model,
                messages=messages,
                think=(True if self._thinking_enabled else False),
                options=opts,
                keep_alive=keep_alive,
                stream=True,
            )
        except Exception as exc:
            if "not found" not in str(exc).lower():
                friendly = self._friendly_ollama_error(exc)
                if friendly:
                    raise RuntimeError(friendly) from exc
                raise
            resolved = self._resolve_existing_model()
            if not resolved:
                raise
            self._runtime_model = resolved
            stream = ollama.chat(
                model=self._runtime_model,
                messages=messages,
                think=(True if self._thinking_enabled else False),
                options=opts,
                keep_alive=keep_alive,
                stream=True,
            )

        for chunk in stream:
            last_chunk = chunk or {}
            msg = ((chunk or {}).get("message", {}) or {})
            piece = msg.get("content", "")
            if piece:
                raw_piece = self._strip_cjk(piece)
                raw_parts.append(raw_piece)
                visible_piece = self._consume_visible_stream_text(raw_piece)
                if visible_piece:
                    parts.append(visible_piece)
                    if callable(on_chunk):
                        try:
                            on_chunk(visible_piece)
                        except Exception:
                            pass
            tpiece = msg.get("thinking", "")
            if tpiece and self._thinking_enabled:
                # Thinking is accumulated separately and never shown in streaming bubble.
                try:
                    if not hasattr(self, "_thinking_parts"):
                        self._thinking_parts = []
                    clean_think = self._strip_cjk(str(tpiece))
                    self._thinking_parts.append(clean_think)
                    if callable(on_thinking_chunk):
                        try:
                            on_thinking_chunk(clean_think)
                        except Exception:
                            pass
                except Exception:
                    pass

        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            self.last_stats = {
                "answer_ms": round(float(elapsed_ms), 2),
                "prep_ms": round(float(prep_ms), 2),
                "prompt_eval_count": last_chunk.get("prompt_eval_count"),
                "eval_count": last_chunk.get("eval_count"),
                "total_duration_ms": round(last_chunk.get("total_duration", 0) / 1_000_000, 1)
                if isinstance(last_chunk.get("total_duration"), (int, float))
                else None,
                "eval_duration_ms": round(last_chunk.get("eval_duration", 0) / 1_000_000, 1)
                if isinstance(last_chunk.get("eval_duration"), (int, float))
                else None,
                "prompt_eval_duration_ms": round(last_chunk.get("prompt_eval_duration", 0) / 1_000_000, 1)
                if isinstance(last_chunk.get("prompt_eval_duration"), (int, float))
                else None,
            }
        except Exception:
            self.last_stats = {"answer_ms": round(float(elapsed_ms), 2), "prep_ms": round(float(prep_ms), 2)}

        thinking_raw = ""
        try:
            thinking_raw = "".join(getattr(self, "_thinking_parts", []) or [])
        except Exception:
            thinking_raw = ""
        self._thinking_parts = []

        raw_text = "".join(raw_parts)
        thinking, answer = self._extract_thinking_and_answer(raw_text, thinking_raw)
        if not answer.strip():
            answer = "".join(parts)
        reply = self._postprocess_reply(self._strip_cjk(answer))
        self.last_thinking = self._postprocess_reply(self._strip_cjk(thinking)) if self._thinking_enabled else ""
        if not reply.strip() and self._thinking_enabled:
            recovered = self._recover_reply_from_thinking(user_input, self.last_thinking, keep_alive, opts)
            if recovered:
                reply = recovered
        if not reply.strip():
            reply = "Ответ не сгенерировался. Напиши запрос еще раз."
        if store_turn:
            self.mm.store_turn(user_input, reply)
        return reply


