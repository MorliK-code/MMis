from __future__ import annotations

import re


_COMMAND_PREFIXES = ("/web", "/no-web")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїҐґ0-9][A-Za-zА-Яа-яЁёІіЇїҐґ0-9._\-]*")
_LETTER_RE = r"A-Za-zА-Яа-яЁёІіЇїҐґ"
_LEADING_FILLER_RE = re.compile(
    r"^(?:\s*(?:а|ну|и|но|слушай|слушай-ка|кстати|короче|вообще|ладно|вот|теперь)\s+)+",
    flags=re.I,
)
_LEADING_SEARCH_RE = re.compile(
    r"^(?:(?:можешь(?:\s+ли)?|можно|сможешь|не\s+мог\s+бы(?:\s+ты)?|could\s+you|can\s+you|please)\s+)?"
    r"(?:(?:мне|нам)\s+)?"
    r"(?:(?:найди(?:-ка)?|найти|поищи(?:-ка)?|поискать|подскажи(?:-ка)?|скажи(?:-ка)?|"
    r"глянь(?:-ка)?|посмотри(?:-ка)?|проверь(?:-ка)?|check|find|search|lookup|look\s+up)\s+)+"
    r"(?:(?:мне|нам)\s+)?",
    flags=re.I,
)
_TRAILING_GENERIC_RE = re.compile(
    r"(?:\s+(?:в|по)\s+(?:интернете|интернету|сети|инете|web|вебе|веб))+$|"
    r"(?:\s+(?:online|онлайн))+$",
    flags=re.I,
)
_TRAILING_POLITE_RE = re.compile(r"(?:\s+(?:please|pls|пожалуйста))+$", flags=re.I)
_SEGMENT_SPLIT_RE = re.compile(
    rf"[?!;\n]+|\.{{2,}}|…+|(?<=[{_LETTER_RE}0-9])\.(?=\s+[{_LETTER_RE}])|(?:\s+[—-]\s+)"
)
_TAIL_SPLIT_RE = re.compile(r"\s*,\s+")
_SEARCH_CORE_ANCHOR_RE = re.compile(
    r"\b(?:что|где|когда|какой|какая|какие|сколько|latest|current|today|price|pricing|cost|"
    r"weather|forecast|temperature|rate|exchange|version|release|news|курс|цена|стоимость|"
    r"погода|прогноз|температура|версия|релиз|новости|найди|поищи|посмотри|проверь|search|find|check)\b",
    flags=re.I,
)
_QUESTIONISH_RE = re.compile(
    r"\b(?:что|где|когда|какой|какая|какие|сколько|как|почему|latest|current|today|price|"
    r"weather|forecast|rate|version|news|курс|цена|погода|прогноз|версия|новости)\b",
    flags=re.I,
)
_QUESTION_START_RE = re.compile(
    r"\b(?:что|где|когда|какой|какая|какие|сколько|как|почему|what|where|when|which|how|why)\b",
    flags=re.I,
)
_CHATTER_TOKENS = {
    "а",
    "ну",
    "и",
    "но",
    "слушай",
    "кстати",
    "короче",
    "вообще",
    "ладно",
    "вот",
    "теперь",
    "просто",
    "надеюсь",
    "думаю",
    "хочу",
    "хотел",
    "хотела",
    "нужно",
    "надо",
    "мне",
    "нам",
    "я",
    "мы",
    "ты",
    "же",
    "можешь",
    "можно",
    "сможешь",
    "значит",
    "уже",
    "наконец",
    "пожалуйста",
    "please",
    "pls",
    "just",
    "maybe",
    "can",
    "could",
    "you",
}
_CANDIDATE_KIND_BONUS = {
    "full": 0.0,
    "segment": 0.6,
    "tail": 1.4,
    "search_tail": 3.8,
    "question_clause": 5.2,
    "segment_question_clause": 5.6,
}


def strip_web_command_prefix(text: str) -> str:
    src = str(text or "").strip()
    low = src.lower()
    for token in _COMMAND_PREFIXES:
        if low == token:
            return ""
        if low.startswith(token + " "):
            return src[len(token) :].strip()
    return src


def normalize_search_text(text: str) -> str:
    debug = analyze_search_text(text)
    return str(debug.get("extracted_search_core") or debug.get("normalized_query") or "").strip()


def extract_search_core(text: str) -> str:
    src = strip_web_command_prefix(text)
    if not src:
        return ""
    cleaned_full = _cleanup_search_text(src)
    return _extract_search_core_from_cleaned(src=src, cleaned_full=cleaned_full)


def analyze_search_text(text: str) -> dict[str, str]:
    original = str(text or "").strip()
    src = strip_web_command_prefix(text)
    if not src:
        return {
            "original_query": original,
            "normalized_query": "",
            "search_core": "",
            "extracted_search_core": "",
        }

    cleaned_full = _cleanup_search_text(src)
    search_core = _extract_search_core_from_cleaned(src=src, cleaned_full=cleaned_full)
    return {
        "original_query": original,
        "normalized_query": cleaned_full,
        "search_core": search_core,
        "extracted_search_core": search_core,
    }


def _extract_search_core_from_cleaned(*, src: str, cleaned_full: str) -> str:
    full_tokens = _tokenize(cleaned_full)
    if not full_tokens:
        return ""
    if len(full_tokens) <= 6:
        return cleaned_full

    candidates: list[tuple[str, str, int, int, str]] = []

    def _push(candidate: str, raw: str, segment_index: int, tail_index: int, kind: str) -> None:
        value = _cleanup_search_text(candidate)
        if not value:
            return
        row = (value, raw, int(segment_index), int(tail_index), str(kind or "full"))
        if row not in candidates:
            candidates.append(row)

    _push(cleaned_full, src, 0, 0, "full")

    full_question_clause = _extract_last_question_clause(cleaned_full)
    if full_question_clause:
        _push(full_question_clause, cleaned_full, 0, 1, "question_clause")

    segments = _split_segments(src)
    for seg_index, raw_segment in enumerate(segments):
        _push(raw_segment, raw_segment, seg_index, 0, "segment")

        question_clause = _extract_last_question_clause(raw_segment)
        if question_clause:
            _push(question_clause, raw_segment, seg_index, 1, "segment_question_clause")

        tails = _split_tails(raw_segment)
        for tail_index, raw_tail in enumerate(tails, start=1):
            _push(raw_tail, raw_segment, seg_index, tail_index, "tail")

        tail = _extract_last_search_tail(raw_segment)
        if tail:
            _push(tail, raw_segment, seg_index, len(tails) + 1, "search_tail")

    best = max(
        candidates,
        key=lambda item: _candidate_score(
            candidate=item[0],
            raw=item[1],
            segment_index=item[2],
            tail_index=item[3],
            kind=item[4],
            total_segments=max(1, len(segments)),
        ),
    )
    best_text = str(best[0] or "").strip()
    if len(_tokenize(best_text)) <= 2 and len(full_tokens) <= 10:
        return cleaned_full
    return best_text


def _cleanup_search_text(text: str) -> str:
    src = str(text or "").strip()
    prev = None
    while src and src != prev:
        prev = src
        src = _LEADING_FILLER_RE.sub("", src, count=1).strip()
        src = _LEADING_SEARCH_RE.sub("", src, count=1).strip()
        src = src.rstrip(" ?!.,")
        src = _TRAILING_GENERIC_RE.sub("", src).strip()
        src = _TRAILING_POLITE_RE.sub("", src).strip()
    src = re.sub(r"\s+", " ", src).strip(" \t\r\n,;:-")
    return src.rstrip(" ?!.,")


def _split_segments(text: str) -> list[str]:
    src = str(text or "").strip()
    if not src:
        return []
    out: list[str] = []
    for segment in _SEGMENT_SPLIT_RE.split(src):
        item = str(segment or "").strip(" \t\r\n,;:-")
        if item:
            out.append(item)
    return out


def _split_tails(text: str) -> list[str]:
    src = str(text or "").strip()
    if not src:
        return []
    out: list[str] = []
    for tail in _TAIL_SPLIT_RE.split(src):
        item = str(tail or "").strip(" \t\r\n,;:-")
        if item:
            out.append(item)
    return out


def _extract_last_search_tail(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    matches = list(_SEARCH_CORE_ANCHOR_RE.finditer(src))
    if not matches:
        return ""
    anchor = matches[-1]
    return str(src[anchor.start() :] or "").strip(" \t\r\n,;:-")


def _extract_last_question_clause(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    matches = list(_QUESTION_START_RE.finditer(src))
    if not matches:
        return ""
    anchor = matches[-1]
    return str(src[anchor.start() :] or "").strip(" \t\r\n,;:-")


def _candidate_score(
    *,
    candidate: str,
    raw: str,
    segment_index: int,
    tail_index: int,
    kind: str,
    total_segments: int,
) -> tuple[float, ...]:
    tokens = _tokenize(candidate)
    if not tokens:
        return (-999.0,)

    content = [token for token in tokens if token not in _CHATTER_TOKENS]
    chatter_count = len(tokens) - len(content)
    has_anchor = bool(_SEARCH_CORE_ANCHOR_RE.search(raw))
    is_questionish = bool("?" in str(raw or "")) or bool(_QUESTIONISH_RE.search(candidate))
    starts_question = bool(_QUESTION_START_RE.match(candidate))
    starts_anchor = bool(_SEARCH_CORE_ANCHOR_RE.match(candidate))
    numeric_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    content_density = float(len(content)) / float(max(1, len(tokens)))
    raw_tokens = _tokenize(raw)
    compression = max(0, len(raw_tokens) - len(tokens))
    kind_bonus = float(_CANDIDATE_KIND_BONUS.get(str(kind or "").strip().lower(), 0.0))

    score = float(len(content)) * 2.0
    score += kind_bonus
    score += 4.0 if has_anchor else 0.0
    score += 3.0 if starts_question else 0.0
    score += 2.0 if starts_anchor else 0.0
    score += 2.2 if is_questionish else 0.0
    score += 1.2 * float(numeric_tokens)
    score += 2.5 * content_density
    score += 1.4 * float(min(compression, 8)) * float(starts_question)
    score += 1.0 * float(min(compression, 8)) * float(starts_anchor)
    score += 1.4 * float(segment_index >= max(0, total_segments - 1))
    score += 1.4 * float(tail_index > 0)
    score -= max(0, len(tokens) - 10) * 1.8
    score -= 1.4 * float(chatter_count)
    score -= 2.0 * float(chatter_count >= 3)
    score -= 2.5 * float(len(content) <= 1)

    return (
        score,
        float(starts_question),
        float(starts_anchor),
        float(has_anchor),
        float(is_questionish),
        float(kind_bonus),
        float(compression),
        float(len(content)),
        -float(len(tokens)),
        float(segment_index),
        float(tail_index),
    )


def _tokenize(text: str) -> list[str]:
    return [str(token).strip().lower() for token in _WORD_RE.findall(str(text or "")) if str(token).strip()]
