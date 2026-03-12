from __future__ import annotations

import re
from typing import Any

from modules.character.dialog_policies import sanitize_user_addressing_text, trim_leading_greeting


class RuleEvaluator:
    def matches(self, when: dict[str, Any] | None, *, ctx: dict[str, Any], traits: dict[str, Any]) -> bool:
        cond = dict(when or {})
        if not cond:
            return True

        if "intent" in cond and not _match_multi(str(ctx.get("intent") or ""), cond.get("intent")):
            return False
        if "emotion" in cond and not _match_multi(str(ctx.get("emotion") or ""), cond.get("emotion")):
            return False
        if "mode" in cond and not _match_multi(str(ctx.get("mode") or ""), cond.get("mode")):
            return False

        tags = {str(x).strip().lower() for x in list(ctx.get("tags") or []) if str(x).strip()}
        any_tags = {str(x).strip().lower() for x in list(cond.get("tags_any") or []) if str(x).strip()}
        all_tags = {str(x).strip().lower() for x in list(cond.get("tags_all") or []) if str(x).strip()}
        if any_tags and tags.isdisjoint(any_tags):
            return False
        if all_tags and not all_tags.issubset(tags):
            return False

        min_trait = dict(cond.get("min_trait") or {})
        for key, value in min_trait.items():
            if _trait_value(traits, str(key)) < _to_float(value, 0.0):
                return False

        max_trait = dict(cond.get("max_trait") or {})
        for key, value in max_trait.items():
            if _trait_value(traits, str(key)) > _to_float(value, 1.0):
                return False

        return True


def _match_multi(actual: str, expected) -> bool:
    src = str(actual or "").strip().lower()
    if isinstance(expected, (list, tuple, set)):
        values = [str(x).strip().lower() for x in list(expected or []) if str(x).strip()]
        return src in values
    return src == str(expected or "").strip().lower()


def _trait_value(traits: dict[str, Any], key: str) -> float:
    row = dict(traits.get(str(key).strip().lower()) or {})
    value = row.get("value")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return _to_float(value, 0.0)


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class ResponseConstraintEvaluator:
    """Deterministic post-filter for hard response constraints."""

    _APOLOGY_RE = re.compile(
        r"\b(sorry|apolog(?:y|ize|ise)|извини|извините|прошу\s+прощения|прости|простите)\b",
        flags=re.IGNORECASE,
    )
    _BANNED_PREFIX_RE = re.compile(
        r"^\s*(?:hello|hi|hey|yo|привет|здравствуй(?:те)?|добр(?:ый|ое)\s+(?:день|вечер|утро))[\s,!.?:;-]*",
        flags=re.IGNORECASE,
    )
    _ASK_BACK_RE = re.compile(
        r"(\b(а\s+ты|как\s+ты|как\s+дела|что\s+у\s+тебя)\b.*?\??$)|(\b(and\s+you|how\s+about\s+you)\b.*?\??$)",
        flags=re.IGNORECASE,
    )
    _SERVICE_MARKER_RE = re.compile(r"^(thinking|assistant|user|system)\s*>\s*", flags=re.IGNORECASE)
    _SPACE_RE = re.compile(r"\s+")
    _APOLOGY_FALLBACK = "Поняла. Перейду сразу к сути."

    def __init__(self, forbidden_phrases: list[str] | None = None):
        self.forbidden_phrases = [str(x or "").strip() for x in list(forbidden_phrases or []) if str(x or "").strip()]

    def enforce(
        self,
        text: str,
        *,
        user_text: str = "",
        dialog_mode: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        address_terms_policy: dict[str, Any] | None = None,
        user_addressing: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        mode = dict(dialog_mode or {})
        meta = dict(metadata or {})
        terms_policy = dict(address_terms_policy or {})
        out = str(text or "").strip()
        applied: list[str] = []
        if not out:
            return out, applied

        out, did = self._remove_forbidden_prefixes(out, greeting_allowed=bool(mode.get("greeting_allowed", True)))
        if did:
            applied.append("forbidden_start")

        out, did = self._remove_forbidden_phrases(
            out,
            smalltalk_allowed=bool(mode.get("smalltalk_allowed", True)),
            greeting_allowed=bool(mode.get("greeting_allowed", True)),
        )
        if did:
            applied.append("forbidden_phrases")

        out, did = self._remove_ask_back(out, smalltalk_allowed=bool(mode.get("smalltalk_allowed", True)))
        if did:
            applied.append("ask_back_removed")

        out, term_applied = self._apply_address_terms_policy(out, terms_policy)
        if term_applied:
            applied.extend(term_applied)

        out, name_applied = sanitize_user_addressing_text(out, user_addressing)
        if name_applied:
            applied.extend(name_applied)

        out, did = self._remove_unneeded_apology(out, user_text=user_text, metadata=meta)
        if did:
            applied.append("apology_removed")

        out, did = self._remove_repetitions(out)
        if did:
            applied.append("repetition_removed")

        out = self._cleanup(out)
        return out, applied

    def _remove_forbidden_prefixes(self, text: str, *, greeting_allowed: bool) -> tuple[str, bool]:
        src = str(text or "").strip()
        if not src:
            return "", False
        if greeting_allowed:
            return src, False
        trimmed = trim_leading_greeting(src)
        trimmed = self._BANNED_PREFIX_RE.sub("", trimmed, count=1).strip()
        return (trimmed or src), bool(trimmed and trimmed != src)

    def _remove_forbidden_phrases(
        self,
        text: str,
        *,
        smalltalk_allowed: bool,
        greeting_allowed: bool,
    ) -> tuple[str, bool]:
        src = str(text or "")
        if not src:
            return "", False
        phrases = list(self.forbidden_phrases or [])
        if not greeting_allowed:
            phrases.extend(["чем помочь", "how can i help"])
        if not smalltalk_allowed:
            phrases.extend(["как дела", "how are you", "what's up"])
        out = src
        changed = False
        for phrase in phrases:
            item = str(phrase or "").strip()
            if not item:
                continue
            pattern = re.compile(re.escape(item), flags=re.IGNORECASE)
            new_out = pattern.sub("", out)
            if new_out != out:
                changed = True
                out = new_out
        return out, changed

    def _remove_ask_back(self, text: str, *, smalltalk_allowed: bool) -> tuple[str, bool]:
        src = str(text or "").strip()
        if not src or smalltalk_allowed:
            return src, False
        lines = [x.strip() for x in src.split("\n") if x.strip()]
        kept: list[str] = []
        changed = False
        for line in lines:
            if self._ASK_BACK_RE.search(line):
                changed = True
                continue
            kept.append(line)
        if not kept:
            return src, False
        return "\n".join(kept).strip(), changed

    def _apply_address_terms_policy(self, text: str, policy: dict[str, Any]) -> tuple[str, list[str]]:
        src = str(text or "").strip()
        if not src:
            return "", []

        terms = [str(x or "").strip().lower() for x in list(policy.get("terms_list") or []) if str(x or "").strip()]
        allowed_term = str(policy.get("allowed_term") or "").strip().lower()
        if allowed_term and allowed_term not in terms:
            terms.append(allowed_term)
        banned_terms = [str(x or "").strip().lower() for x in list(policy.get("banned_terms_effective") or []) if str(x or "").strip()]
        use_term_now = bool(policy.get("use_term_now", False))
        banned_active = bool(policy.get("banned_terms_active", False)) or bool(banned_terms)
        applied: list[str] = []

        if not terms and not banned_terms:
            return src, applied

        out = src
        if banned_active:
            remove_terms = sorted(set(banned_terms or terms))
            out, changed = self._remove_terms(out, remove_terms, remove_all=True)
            if changed:
                applied.append("terms_removed_banned")
            return out, applied

        if not use_term_now:
            remove_terms = sorted(set(terms))
            out, changed = self._remove_terms(out, remove_terms, remove_all=True)
            if changed:
                applied.append("terms_removed_disallowed")
            return out, applied

        keep_term = allowed_term or (terms[0] if terms else "")
        out, limited = self._limit_terms_to_one(out, keep_term=keep_term, terms=terms)
        if limited:
            applied.append("terms_limited_to_one")
        if self._count_terms(out, keep_term) == 1:
            applied.append("terms_kept_once")
        return out, applied

    def _remove_terms(self, text: str, terms: list[str], *, remove_all: bool) -> tuple[str, bool]:
        out = str(text or "")
        changed = False
        for term in list(terms or []):
            item = str(term or "").strip().lower()
            if not item:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(item)}(?!\w)", flags=re.IGNORECASE | re.UNICODE)
            if remove_all:
                new_out = pattern.sub("", out)
            else:
                new_out = pattern.sub("", out, count=1)
            if new_out != out:
                changed = True
                out = new_out
        out = re.sub(r"\s{2,}", " ", out).strip(" ,.!?:;\n\t")
        return out, changed

    def _limit_terms_to_one(self, text: str, *, keep_term: str, terms: list[str]) -> tuple[str, bool]:
        src = str(text or "")
        if not src or not keep_term:
            return src, False
        pattern = re.compile(rf"(?<!\w){re.escape(keep_term)}(?!\w)", flags=re.IGNORECASE | re.UNICODE)
        matches = list(pattern.finditer(src))
        if len(matches) <= 1:
            others = [x for x in terms if x and x != keep_term]
            out, changed = self._remove_terms(src, others, remove_all=True)
            return out, changed

        pieces: list[str] = []
        last = 0
        kept = 0
        changed = False
        for match in matches:
            pieces.append(src[last : match.start()])
            if kept == 0:
                pieces.append(src[match.start() : match.end()])
                kept = 1
            else:
                changed = True
            last = match.end()
        pieces.append(src[last:])
        out = "".join(pieces)
        out, removed_other = self._remove_terms(out, [x for x in terms if x and x != keep_term], remove_all=True)
        changed = changed or removed_other
        out = re.sub(r"\s{2,}", " ", out).strip(" ,.!?:;\n\t")
        return out, changed

    def _count_terms(self, text: str, term: str) -> int:
        src = str(text or "")
        item = str(term or "").strip()
        if not src or not item:
            return 0
        pattern = re.compile(rf"(?<!\w){re.escape(item)}(?!\w)", flags=re.IGNORECASE | re.UNICODE)
        return len(list(pattern.finditer(src)))

    def _remove_unneeded_apology(self, text: str, *, user_text: str, metadata: dict[str, Any]) -> tuple[str, bool]:
        src = str(text or "").strip()
        if not src:
            return "", False
        if self._has_error_context(user_text=user_text, metadata=metadata):
            return src, False

        lines = [x.strip() for x in src.split("\n") if x.strip()]
        if not lines:
            return src, False
        changed = False
        kept: list[str] = []
        for line in lines:
            if self._APOLOGY_RE.search(line):
                changed = True
                continue
            kept.append(line)
        if not kept and changed:
            return self._APOLOGY_FALLBACK, True
        if not kept:
            return src, False
        return "\n".join(kept).strip(), changed

    def _remove_repetitions(self, text: str) -> tuple[str, bool]:
        src = str(text or "").strip()
        if not src:
            return "", False
        lines = [x.strip() for x in src.split("\n") if x.strip()]
        kept: list[str] = []
        prev = ""
        changed = False
        for line in lines:
            key = self._dedupe_key(line)
            if key and key == prev:
                changed = True
                continue
            kept.append(line)
            prev = key
        out = "\n".join(kept).strip()
        return out, changed

    def _cleanup(self, text: str) -> str:
        lines = []
        for raw in str(text or "").split("\n"):
            row = str(raw or "").strip()
            if not row:
                continue
            row = self._SERVICE_MARKER_RE.sub("", row).strip()
            if row:
                lines.append(row)
        out = "\n".join(lines).strip()
        out = re.sub(r"\n{3,}", "\n\n", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
        return out.strip()

    def _has_error_context(self, *, user_text: str, metadata: dict[str, Any]) -> bool:
        if bool(metadata.get("had_error")):
            return True
        if bool(metadata.get("tool_error")):
            return True
        if bool(metadata.get("provider_error")):
            return True
        low = str(user_text or "").lower()
        return any(x in low for x in ("traceback", "exception", "ошибка", "error", "failed", "не работает"))

    def _dedupe_key(self, text: str) -> str:
        src = str(text or "").strip().lower()
        src = re.sub(r"[\"'`]", "", src)
        src = re.sub(r"[^\w\s]", " ", src, flags=re.UNICODE)
        src = self._SPACE_RE.sub(" ", src)
        return src.strip()

