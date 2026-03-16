from __future__ import annotations

import re


_QUERY_COMMANDS = {"/web", "/no-web"}
_COMMAND_ARG_TOKENS = {"on", "off", "auto", "true", "false", "yes", "no", "1", "0", "lock", "unlock"}
_SINGLE_ARG_COMMANDS = {"/mode", "/model", "/character", "/persona", "/modes"}
_LEADING_SLASH_COMMAND_RE = re.compile(r"^/(?P<name>[A-Za-z][A-Za-z0-9._-]*)")
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ][0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ._-]*")
_LETTER_RE = r"0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ"

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
_TRAILING_PREFERENCE_RE = re.compile(
    r"(?:\s*[,.-]?\s*(?:желательно|по возможности|если можно|if possible|preferably)\b[^?!]*)$",
    flags=re.I,
)
_SEGMENT_SPLIT_RE = re.compile(
    rf"[?!;\n]+|\.{{2,}}|…+|(?<=[{_LETTER_RE}])\.(?=\s+[{_LETTER_RE}])|(?:\s+[—-]\s+)"
)
_TAIL_SPLIT_RE = re.compile(r"\s*,\s+")
_QUESTION_START_RE = re.compile(
    r"\b(?:что|где|когда|какой|какая|какие|сколько|как|почему|what|where|when|which|how|why)\b",
    flags=re.I,
)
_SEARCH_CORE_ANCHOR_RE = re.compile(
    r"\b(?:что|где|когда|какой|какая|какие|сколько|latest|current|today|price|pricing|cost|"
    r"weather|forecast|temperature|rate|exchange|version|release|news|дата|date|"
    r"курс|цена|стоимость|погода|прогноз|температура|версия|релиз|новости|"
    r"найди|поищи|посмотри|проверь|search|find|check)\b",
    flags=re.I,
)
_QUESTIONISH_RE = re.compile(
    r"\b(?:что|где|когда|какой|какая|какие|сколько|как|почему|latest|current|today|price|"
    r"weather|forecast|rate|version|news|дата|date|курс|цена|погода|прогноз|версия|новости)\b",
    flags=re.I,
)
_LEADING_MODEL_QUERY_WRAPPER_RE = re.compile(
    r"^(?:(?:а|ну|и|но|ладно|слушай|слушай-ка|короче|вот|теперь|так\s+вот)\s+)*"
    r"(?:(?:ты|вы)\s+)?"
    r"(?:(?:же|точно|вообще|там|снова|опять|реально)\s+)*"
    r"(?:(?:вр[её]шь|ошиб(?:лась|аешься)|неправильно|неверно|смотришь|видишь|"
    r"можешь\s+увидеть|проверяешь|получаешь|не\s+путаешь)\s+)+",
    flags=re.I,
)
_LEADING_TOOL_STATUS_WRAPPER_RE = re.compile(
    r"^(?:что\s+)?(?:интернет|web|веб|поиск|search|доступ)\s+"
    r"(?:(?:есть|работает|будет|появился|включен|включён|доступен)\s+)*",
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
    "wrapper_tail": 4.8,
    "search_tail": 3.8,
    "question_clause": 5.2,
    "segment_question_clause": 5.6,
}


def strip_web_command_prefix(text: str) -> str:
    stripped, _removed = _strip_service_command_prefixes(text)
    return stripped


def strip_service_command_prefix(text: str) -> str:
    stripped, _removed = _strip_service_command_prefixes(text)
    return stripped


def normalize_search_text(text: str) -> str:
    debug = analyze_search_text(text)
    return str(debug.get("extracted_search_core") or debug.get("normalized_query") or "").strip()


def extract_search_core(text: str) -> str:
    src, _removed = _strip_service_command_prefixes(text)
    if not src:
        return ""
    cleaned_full = _cleanup_search_text(src)
    return _extract_search_core_from_cleaned(src=src, cleaned_full=cleaned_full)


def analyze_search_text(text: str) -> dict[str, object]:
    original = str(text or "").strip()
    src, removed_commands = _strip_service_command_prefixes(text)
    if not src:
        removed_text = str(removed_commands or "").strip()
        return {
            "original_query": original,
            "normalized_query": "",
            "search_core": "",
            "extracted_search_core": "",
            "removed_wrapper_text": removed_text,
            "removed_wrapper_fragments": _split_removed_wrapper_fragments(removed_text),
        }

    cleaned_full = _cleanup_search_text(src)
    search_core = _extract_search_core_from_cleaned(src=src, cleaned_full=cleaned_full)
    removed_wrapper_text = _derive_removed_wrapper_text(
        removed_commands=removed_commands,
        normalized_query=cleaned_full,
        search_core=search_core,
    )
    removed_wrapper_fragments = _split_removed_wrapper_fragments(removed_wrapper_text)
    return {
        "original_query": original,
        "normalized_query": cleaned_full,
        "search_core": search_core,
        "extracted_search_core": search_core,
        "removed_wrapper_text": removed_wrapper_text,
        "removed_wrapper_fragments": removed_wrapper_fragments,
    }


def _extract_search_core_from_cleaned(*, src: str, cleaned_full: str) -> str:
    full_tokens = _tokenize(cleaned_full)
    if not full_tokens:
        return ""
    if len(full_tokens) <= 6:
        wrapper_tail = _strip_non_search_wrapper(cleaned_full)
        if wrapper_tail and wrapper_tail != cleaned_full:
            return wrapper_tail
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

    wrapper_tail = _strip_non_search_wrapper(cleaned_full)
    if wrapper_tail:
        _push(wrapper_tail, cleaned_full, 0, 1, "wrapper_tail")

    full_question_clause = _extract_last_question_clause(cleaned_full)
    if full_question_clause:
        _push(full_question_clause, cleaned_full, 0, 1, "question_clause")

    segments = _split_segments(src)
    for seg_index, raw_segment in enumerate(segments):
        _push(raw_segment, raw_segment, seg_index, 0, "segment")

        wrapped_segment = _strip_non_search_wrapper(raw_segment)
        if wrapped_segment:
            _push(wrapped_segment, raw_segment, seg_index, 1, "wrapper_tail")

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
        src = _TRAILING_PREFERENCE_RE.sub("", src).strip()
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


def _strip_non_search_wrapper(text: str) -> str:
    src = _cleanup_search_text(text)
    if not src:
        return ""
    prev = None
    while src and src != prev:
        prev = src
        src = _strip_prefix_once(src, _LEADING_MODEL_QUERY_WRAPPER_RE)
        src = _strip_prefix_once(src, _LEADING_TOOL_STATUS_WRAPPER_RE)
        if src:
            clause = _extract_last_question_clause(src)
            if clause and clause != src and len(_tokenize(clause)) >= 2:
                src = clause
        src = _cleanup_search_text(src)
    return src


def _strip_prefix_once(text: str, pattern: re.Pattern[str]) -> str:
    src = str(text or "").strip()
    match = pattern.match(src)
    if not match:
        return src
    return str(src[match.end() :] or "").strip(" \t\r\n,;:-")


def _strip_service_command_prefixes(text: str) -> tuple[str, str]:
    src = str(text or "").strip()
    removed: list[str] = []
    while src.startswith("/"):
        match = _LEADING_SLASH_COMMAND_RE.match(src)
        if not match:
            break
        command = "/" + str(match.group("name") or "")
        end = int(match.end())
        rest = str(src[end:] or "")
        rest_lstrip = rest.lstrip()
        consumed = str(src[:end] or "").strip()
        token_count = _leading_command_arg_count(command=command, rest=rest_lstrip)
        if token_count > 0:
            end = end + _consume_rest_tokens(rest=rest, token_count=token_count)
            consumed = str(src[:end] or "").strip()
        removed.append(consumed)
        src = str(src[end:] or "").strip()
    return src, " ".join(part for part in removed if part).strip()


def _leading_command_arg_count(*, command: str, rest: str) -> int:
    normalized = str(command or "").strip().lower()
    if not normalized or not str(rest or "").strip():
        return 0
    tokens = [str(x or "").strip() for x in str(rest or "").split() if str(x or "").strip()]
    if not tokens:
        return 0
    head = str(tokens[0] or "").strip().lower()
    if normalized in _QUERY_COMMANDS:
        return 0
    if normalized == "/mode_lock":
        return 1 if head in _COMMAND_ARG_TOKENS else 0
    if normalized == "/output":
        if head == "status":
            return 1
        if head in {"parameters", "summary"}:
            if len(tokens) >= 2 and str(tokens[1] or "").strip().lower() in _COMMAND_ARG_TOKENS:
                return 2
            return 1
        return 0
    if normalized in _SINGLE_ARG_COMMANDS:
        return 1
    return 1 if head in _COMMAND_ARG_TOKENS else 0


def _consume_rest_tokens(*, rest: str, token_count: int) -> int:
    if int(token_count) <= 0:
        return 0
    src = str(rest or "")
    offset = 0
    remaining = src
    left = int(token_count)
    while left > 0 and remaining:
        stripped = remaining.lstrip()
        offset += len(remaining) - len(stripped)
        remaining = stripped
        if not remaining:
            break
        match = re.match(r"\S+", remaining)
        if not match:
            break
        offset += int(match.end())
        remaining = remaining[int(match.end()) :]
        left -= 1
    return offset


def _derive_removed_wrapper_text(*, removed_commands: str, normalized_query: str, search_core: str) -> str:
    parts: list[str] = []
    commands = str(removed_commands or "").strip()
    if commands:
        parts.append(commands)
    normalized = str(normalized_query or "").strip()
    extracted = str(search_core or "").strip()
    if normalized and extracted and normalized != extracted:
        low = normalized.casefold()
        needle = extracted.casefold()
        index = low.find(needle)
        if index >= 0:
            prefix = normalized[:index].strip(" \t\r\n,;:-")
            suffix = normalized[index + len(extracted) :].strip(" \t\r\n,;:-")
            if prefix:
                parts.append(prefix)
            if suffix:
                parts.append(suffix)
        else:
            parts.append(normalized)
    seen: set[str] = set()
    out: list[str] = []
    for item in parts:
        token = str(item or "").strip()
        if not token:
            continue
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return " | ".join(out)


def _split_removed_wrapper_fragments(value: str) -> list[str]:
    out: list[str] = []
    for part in str(value or "").split("|"):
        item = str(part or "").strip()
        if item and item not in out:
            out.append(item)
    return out


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
