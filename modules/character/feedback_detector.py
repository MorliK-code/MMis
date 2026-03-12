from __future__ import annotations

import re
from typing import Iterable

_SPACES_RE = re.compile(r"\s+", flags=re.UNICODE)
_CYRILLIC_RE = re.compile(r"[а-яё]", flags=re.IGNORECASE)
_MOJIBAKE_RE = re.compile(r"[РСЃ]", flags=re.UNICODE)

_STOP_TOKENS = {
    "так",
    "это",
    "эту",
    "этот",
    "эти",
    "слово",
    "фразу",
    "фраза",
    "word",
    "phrase",
    "it",
    "that",
}

_BAN_MARKER_RE = re.compile(
    r"(?:не\s+говори|не\s+используй|не\s+пиши|don't\s+(?:say|use|write)|do\s+not\s+(?:say|use|write)|avoid\s+(?:saying|using))",
    flags=re.IGNORECASE,
)
_UNBAN_MARKER_RE = re.compile(
    r"(?:можно\s+говорить|можешь\s+(?:говорить|использовать)|you\s+can\s+(?:say|use)|you\s+may\s+(?:say|use))",
    flags=re.IGNORECASE,
)

_QUOTED_PART_RE = re.compile(
    r"\"([^\"]{2,180})\"|'([^']{2,180})'|«([^»]{2,180})»|“([^”]{2,180})”",
    flags=re.IGNORECASE,
)
_TOKEN_AFTER_MARKER_RE = re.compile(
    r"(?:не\s+говори|не\s+используй|не\s+пиши|don't\s+(?:say|use|write)|do\s+not\s+(?:say|use|write)|avoid\s+(?:saying|using))"
    r"(?:\s+(?:слово|фразу|phrase|word))?\s+(?:[:\-]\s*)?([a-zа-яё0-9_\-]{2,80})",
    flags=re.IGNORECASE,
)
_UNBAN_TOKEN_AFTER_MARKER_RE = re.compile(
    r"(?:можно\s+говорить|можешь\s+(?:говорить|использовать)|you\s+can\s+(?:say|use)|you\s+may\s+(?:say|use))"
    r"(?:\s+(?:слово|фразу|phrase|word))?\s+(?:[:\-]\s*)?([a-zа-яё0-9_\-]{2,80})",
    flags=re.IGNORECASE,
)
_COLON_PHRASE_RE = re.compile(
    r"(?:не\s+говори|don't\s+say|do\s+not\s+say|avoid\s+saying)\s*[:\-]\s*([^.!?\n]{2,180})",
    flags=re.IGNORECASE,
)
_UNBAN_COLON_PHRASE_RE = re.compile(
    r"(?:можно\s+говорить|you\s+can\s+say)\s*[:\-]\s*([^.!?\n]{2,180})",
    flags=re.IGNORECASE,
)

_NAME_FORM_BODY = r"[A-Za-z\u0400-\u04FF][A-Za-z\u0400-\u04FF\-]{1,31}"
_NAME_CORRECTION_RE = re.compile(
    rf"(?:^|[\s,;:])(?:\u043d\u0435|not)\s+({_NAME_FORM_BODY})\s*,?\s*(?:\u0430|but)\s+({_NAME_FORM_BODY})(?:$|[\s,.!?;:])",
    flags=re.IGNORECASE,
)
_MY_NAME_IS_RE = re.compile(
    rf"(?:\u043c\u0435\u043d\u044f\s+\u0437\u043e\u0432\u0443\u0442|\u043c\u043e(?:\u0435|\u0451)\s+\u0438\u043c\u044f|my\s+name\s+is)\s+({_NAME_FORM_BODY})(?:$|[\s,.!?;:])",
    flags=re.IGNORECASE,
)
_ADDRESS_ME_AS_RE = re.compile(
    rf"(?:\u043e\u0431\u0440\u0430\u0449\u0430\u0439\u0441\u044f\s+\u043a\u043e\s+\u043c\u043d\u0435|\u0437\u043e\u0432\u0438\s+\u043c\u0435\u043d\u044f|\u043d\u0430\u0437\u044b\u0432\u0430\u0439\s+\u043c\u0435\u043d\u044f|call\s+me)\s+({_NAME_FORM_BODY})(?:$|[\s,.!?;:])",
    flags=re.IGNORECASE,
)
_SHORT_ALLOWED_NAME_RE = re.compile(
    rf"^\s*(?:\u043c\u043e\u0436\u043d\u043e|\u043b\u0443\u0447\u0448\u0435|\u043f\u0440\u043e\u0441\u0442\u043e)\s+({_NAME_FORM_BODY})(?:$|[\s,.!?;:])",
    flags=re.IGNORECASE,
)
_FORBID_NAME_FORM_RE = re.compile(
    rf"(?:\u043d\u0435\s+\u043d\u0430\u0437\u044b\u0432\u0430\u0439\s+\u043c\u0435\u043d\u044f|\u043d\u0435\s+\u0437\u043e\u0432\u0438\s+\u043c\u0435\u043d\u044f|don't\s+call\s+me)\s+({_NAME_FORM_BODY})(?:$|[\s,.!?;:])",
    flags=re.IGNORECASE,
)
_DISABLE_DIMINUTIVES_RE = re.compile(
    r"(?:\u043d\u0435\s+\u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439\s+(?:\u043b\u0430\u0441\u043a\u0430\u0442\u0435\u043b\u044c\w+|\u0443\u043c\u0435\u043d\u044c\u0448\u0438\u0442\u0435\u043b\u044c\w+)\s+\u0444\u043e\u0440\u043c\w*|"
    r"\u0431\u0435\u0437\s+(?:\u043b\u0430\u0441\u043a\u0430\u0442\u0435\u043b\u044c\w+|\u0443\u043c\u0435\u043d\u044c\u0448\u0438\u0442\u0435\u043b\u044c\w+)\s+\u0444\u043e\u0440\u043c\w*|"
    r"\u043d\u0435\s+\u043a\u043e\u0432\u0435\u0440\u043a\u0430\u0439\s+\u0438\u043c\u044f|"
    r"\u043d\u0435\s+\u0438\u0441\u043a\u0430\u0436\u0430\u0439\s+\u0438\u043c\u044f|"
    r"don't\s+use\s+diminutives?)",
    flags=re.IGNORECASE,
)
_ENABLE_DIMINUTIVES_RE = re.compile(
    r"(?:\u043c\u043e\u0436\u043d\u043e\s+(?:\u043b\u0430\u0441\u043a\u0430\u0442\u0435\u043b\u044c\w+|\u0443\u043c\u0435\u043d\u044c\u0448\u0438\u0442\u0435\u043b\u044c\w+)\s+\u0444\u043e\u0440\u043c\w*|"
    r"\u043c\u043e\u0436\u0435\u0448\u044c\s+\u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u044c\s+(?:\u043b\u0430\u0441\u043a\u0430\u0442\u0435\u043b\u044c\w+|\u0443\u043c\u0435\u043d\u044c\u0448\u0438\u0442\u0435\u043b\u044c\w+)\s+\u0444\u043e\u0440\u043c\w*|"
    r"you\s+can\s+use\s+diminutives?)",
    flags=re.IGNORECASE,
)

_FEEDBACK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "less_compliments",
        (
            "\\b(?:\\u043c\\u0435\\u043d\\u044c\\u0448\\u0435|\\u043f\\u043e\\u043c\\u0435\\u043d\\u044c\\u0448\\u0435)\\s+\\u043a\\u043e\\u043c\\u043f\\u043b\\u0438\\u043c\\u0435\\u043d\\u0442\\u043e\\u0432\\b",
            r"\bfewer\s+compliments\b",
            r"\bless\s+compliments\b",
        ),
    ),
    (
        "more_compliments",
        (
            "\\b(?:\\u0431\\u043e\\u043b\\u044c\\u0448\\u0435|\\u043f\\u043e\\u0431\\u043e\\u043b\\u044c\\u0448\\u0435)\\s+\\u043a\\u043e\\u043c\\u043f\\u043b\\u0438\\u043c\\u0435\\u043d\\u0442\\u043e\\u0432\\b",
            r"\bmore\s+compliments\b",
            r"\bcompliment\s+me\s+more\b",
        ),
    ),
    (
        "no_compliments",
        (
            "\\b\\u0431\\u0435\\u0437\\s+\\u043a\\u043e\\u043c\\u043f\\u043b\\u0438\\u043c\\u0435\\u043d\\u0442\\u043e\\u0432\\b",
            "\\b\\u043d\\u0435\\s+\\u043d\\u0430\\u0434\\u043e\\s+\\u043a\\u043e\\u043c\\u043f\\u043b\\u0438\\u043c\\u0435\\u043d\\u0442\\u043e\\u0432\\b",
            r"\bno\s+compliments\b",
            r"\bdon't\s+compliment\s+me\b",
        ),
    ),
    (
        "less_warmth",
        (
            r"\b(меньше|менее)\s+дружелюб",
            r"\bless\s+warm",
            r"\bless\s+friendly",
        ),
    ),
    (
        "more_warmth",
        (
            r"\bбольше\s+дружелюб",
            r"\bтеплее\b",
            r"\bmore\s+warm",
            r"\bbe\s+warmer",
        ),
    ),
    (
        "no_teasing",
        (
            r"\bне\s+подкалывай\b",
            r"\bбез\s+подкол",
            r"\bdon't\s+tease\b",
            r"\bno\s+teasing\b",
        ),
    ),
    (
        "more_teasing",
        (
            r"\bможно\s+подкалывать\b",
            r"\bбольше\s+подкол",
            r"\bmore\s+teasing\b",
            r"\btease\s+more\b",
        ),
    ),
    (
        "less_sarcasm",
        (
            r"\bменьше\s+сарказм",
            r"\bбез\s+сарказм",
            r"\bless\s+sarcasm\b",
            r"\bavoid\s+sarcasm\b",
        ),
    ),
    (
        "more_sarcasm",
        (
            r"\bбольше\s+сарказм",
            r"\bmore\s+sarcasm\b",
        ),
    ),
    (
        "shorter_answers",
        (
            r"\bкороче\b",
            r"\bкратко\b",
            r"\bпокороче\b",
            r"\bshorter\s+answers\b",
            r"\bbe\s+brief\b",
        ),
    ),
    (
        "longer_answers",
        (
            r"\bподробнее\b",
            r"\bдетальнее\b",
            r"\bразвернуто\b",
            "\\b(?:\\u0431\\u043e\\u043b\\u044c\\u0448\\u0435|\\u0434\\u043e\\u0431\\u0430\\u0432\\u044c)\\s+\\u0444\\u0438\\u0434\\u0431\\u0435\\u043a\\s+\\u043f\\u0430\\u0442\\u0442?\\u0435\\u0440\\u043d\\u043e\\u0432?\\b",
            "\\bmore\\s+feedback\\s+patterns?\\b",
            "\\badd\\s+more\\s+feedback\\s+patterns?\\b",
            r"\blonger\s+answers\b",
            r"\bmore\s+details\b",
        ),
    ),
    (
        "be_strict",
        (
            r"\bпостроже\b",
            r"\bстроже\b",
            r"\bбудь\s+строже\b",
            r"\bbe\s+strict\b",
        ),
    ),
    (
        "be_softer",
        (
            r"\bпомягче\b",
            r"\bмягче\b",
            r"\bне\s+будь\s+так(ой|ой)\s+строг",
            r"\bbe\s+softer\b",
        ),
    ),
    (
        "lock_feminine",
        (
            r"\bпиши\s+в\s+женск(?:ом|ом)\s+роде\b",
            r"\bвсегда\s+женск(?:ий|ом)\s+род\b",
            r"\buse\s+feminine\b",
            r"\bfemale\s+gender\b",
        ),
    ),
    (
        "avoid_wording",
        (
            r"\bтак\s+не\s+говори\b",
            r"\bне\s+говори\s+так\b",
            r"\bне\s+формулируй\s+так\b",
            r"\bdon't\s+say\s+it\s+like\s+that\b",
            r"\bdon't\s+phrase\s+it\s+like\s+that\b",
        ),
    ),
)


def detect_feedback(text: str) -> list[str]:
    src_raw = _normalize_text_preserve_case(text)
    src = src_raw.lower()
    if not src_raw:
        return []

    out: list[str] = []

    for token, patterns in _FEEDBACK_PATTERNS:
        if _has_any(src, patterns):
            out.append(token)

    for term in _extract_ban_terms(src):
        out.append(f"ban_word:{term}")

    for term in _extract_unban_terms(src):
        out.append(f"unban_word:{term}")

    out.extend(_extract_user_addressing_feedback(src_raw, src))

    uniq: list[str] = []
    seen: set[str] = set()
    for item in out:
        value = str(item or "").strip()
        key = _feedback_dedupe_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(value)
    return uniq


def _extract_ban_terms(src: str) -> list[str]:
    out: list[str] = []
    out.extend(_extract_quoted_after_marker(src, marker_re=_BAN_MARKER_RE))
    out.extend(_extract_phrases_by_regex(src, _COLON_PHRASE_RE))
    out.extend(_extract_tokens_by_regex(src, _TOKEN_AFTER_MARKER_RE))
    return _uniq_terms(out)


def _extract_unban_terms(src: str) -> list[str]:
    out: list[str] = []
    out.extend(_extract_quoted_after_marker(src, marker_re=_UNBAN_MARKER_RE))
    out.extend(_extract_phrases_by_regex(src, _UNBAN_COLON_PHRASE_RE))
    out.extend(_extract_tokens_by_regex(src, _UNBAN_TOKEN_AFTER_MARKER_RE))
    return _uniq_terms(out)


def _extract_quoted_after_marker(src: str, *, marker_re: re.Pattern[str]) -> list[str]:
    out: list[str] = []
    for marker in marker_re.finditer(src):
        tail = src[marker.end() : marker.end() + 220]
        for quoted in _QUOTED_PART_RE.finditer(tail):
            token = next((x for x in quoted.groups() if x), "")
            norm = _normalize_term(token)
            if norm:
                out.append(norm)
    return out


def _extract_tokens_by_regex(src: str, pattern: re.Pattern[str]) -> list[str]:
    out: list[str] = []
    for match in pattern.finditer(src):
        norm = _normalize_term(match.group(1))
        if norm:
            out.append(norm)
    return out


def _extract_phrases_by_regex(src: str, pattern: re.Pattern[str]) -> list[str]:
    out: list[str] = []
    for match in pattern.finditer(src):
        norm = _normalize_term(match.group(1))
        if norm:
            out.append(norm)
    return out


def _normalize_term(value: str) -> str:
    token = _normalize_text(value)
    token = token.strip(".,!?;:()[]{}\"'`“”«»")
    token = _SPACES_RE.sub(" ", token).strip()
    if not token:
        return ""
    if token in _STOP_TOKENS:
        return ""
    if len(token) < 2:
        return ""
    return token


def _uniq_terms(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in values:
        token = str(row or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _has_any(src: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if re.search(pat, src, flags=re.IGNORECASE):
            return True
    return False


def _normalize_text(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    raw = _repair_mojibake(raw)
    raw = raw.lower()
    return _SPACES_RE.sub(" ", raw)


def _normalize_text_preserve_case(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return ""
    raw = _repair_mojibake(raw)
    return _SPACES_RE.sub(" ", raw).strip()


def _repair_mojibake(text: str) -> str:
    src = str(text or "")
    if not src:
        return ""
    if _CYRILLIC_RE.search(src):
        return src
    if not _MOJIBAKE_RE.search(src):
        return src
    for src_enc, dst_enc in (("latin1", "utf-8"), ("cp1251", "utf-8")):
        try:
            fixed = src.encode(src_enc, errors="ignore").decode(dst_enc, errors="ignore")
        except Exception:
            continue
        if _CYRILLIC_RE.search(fixed):
            return fixed
    return src


def _extract_user_addressing_feedback(src_raw: str, src_lower: str) -> list[str]:
    out: list[str] = []

    for match in _NAME_CORRECTION_RE.finditer(src_raw):
        wrong = _normalize_name_form(match.group(1))
        right = _normalize_name_form(match.group(2))
        if not wrong or not right or wrong.casefold() == right.casefold():
            continue
        out.append(f"set_canonical_name:{right}")
        out.append(f"allow_name_form:{right}")
        out.append(f"forbid_name_form:{wrong}")

    for pattern in (_MY_NAME_IS_RE, _ADDRESS_ME_AS_RE, _SHORT_ALLOWED_NAME_RE):
        for match in pattern.finditer(src_raw):
            if pattern is _ADDRESS_ME_AS_RE:
                prefix = src_raw[max(0, match.start() - 4) : match.start()].lower()
                if prefix.endswith("не ") or prefix.endswith("not "):
                    continue
            name = _normalize_name_form(match.group(1))
            if not name:
                continue
            out.append(f"set_canonical_name:{name}")
            out.append(f"allow_name_form:{name}")

    for match in _FORBID_NAME_FORM_RE.finditer(src_raw):
        name = _normalize_name_form(match.group(1))
        if name:
            out.append(f"forbid_name_form:{name}")

    if _DISABLE_DIMINUTIVES_RE.search(src_lower):
        out.append("disable_diminutives")
    if _ENABLE_DIMINUTIVES_RE.search(src_lower):
        out.append("enable_diminutives")
    return out


def _normalize_name_form(value: str) -> str:
    token = _normalize_text_preserve_case(value)
    token = token.strip(".,!?;:()[]{}\"'`вЂњвЂќВ«В»")
    token = _SPACES_RE.sub(" ", token).strip()
    if not token or " " in token:
        return ""
    if len(token) < 2 or len(token) > 32:
        return ""
    if any(ch.isdigit() for ch in token):
        return ""
    if token.casefold() in _STOP_TOKENS:
        return ""
    return token


def _feedback_dedupe_key(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if ":" not in text:
        return text.lower()
    kind, payload = text.split(":", 1)
    return f"{kind.strip().lower()}:{payload.strip().casefold()}"
