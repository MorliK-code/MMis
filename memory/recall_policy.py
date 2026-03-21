from __future__ import annotations

import re
from dataclasses import dataclass

from memory.memory_models import MemoryLevel, MemoryRecord, MemoryType

RECALL_MODE_EXACT = "exact"
RECALL_MODE_CONTEXTUAL = "contextual"
RECALL_MODE_AMBIENT = "ambient"
RECALL_MODE_DOCUMENT = "document"

_VALID_CLAIM_RECALL_MODES = {
    RECALL_MODE_EXACT,
    RECALL_MODE_CONTEXTUAL,
    RECALL_MODE_AMBIENT,
    RECALL_MODE_DOCUMENT,
}

_SELF_QUERY_RE = re.compile(
    r"(?:какая|какой|какое|напомни|подскажи|скажи|what(?:'s| is)|remind me)[^?.!\n]{0,56}(?:у меня|мой|мою|моя|моё|my)\b",
    re.I,
)
_EXACT_FACT_DOMAIN_RE = re.compile(
    r"(?:"
    r"видюх|видеокарт|gpu|graphics card|video card|карточк|карта|"
    r"python|питон|пайтон|"
    r"os|windows|linux|ubuntu|macos|операционк|ос|винда|винды|система|"
    r"озу|ram|оператив|"
    r"имя|name|зовут|"
    r"возраст|age|лет|год"
    r")",
    re.I,
)
_EXACT_FACT_SELF_HINT_RE = re.compile(
    r"(?:у\s+меня|\bмой\b|\bмоя\b|\bмою\b|\bмоё\b|\bmy\b|\bme\b|\bmine\b|\bменя\b|\bмне\b)",
    re.I,
)
_EXACT_FACT_RECALL_HINT_RE = re.compile(
    r"(?:"
    r"какая|какой|какое|сколько|напомни|подскажи|скажи|помнишь|помниш|помни|"
    r"what(?:'s| is)|which|remember|remind|"
    r"что\s+там\s+у\s+меня\s+по|"
    r"что\s+у\s+меня\s+за|"
    r"как\s+меня\s+зовут|"
    r"сколько\s+мне\s+лет"
    r")",
    re.I,
)
_CONTEXTUAL_RECALL_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:о чем|о ч[её]м|what did we|what were we|what did we discuss)", re.I),
    re.compile(
        r"(?:что мы обсуждали|что обсуждали|что решили|к чему пришли|what we decided|what did we decide|what conclusion did we reach)",
        re.I,
    ),
    re.compile(
        r"(?:почему (?:так сделали|(?:мы\s+)?(?:решили|выбрали))|why did we (?:do that|decide|choose)|why we (?:did that|decided|chose))",
        re.I,
    ),
    re.compile(
        r"(?:какой у нас был план|какой (?:был|у нас) план|что по плану|what was our plan|what was the plan|what is our plan|next steps?)",
        re.I,
    ),
    re.compile(r"(?:что ты советовала|what did you suggest|what did you advise)", re.I),
)
_DOCUMENT_RECALL_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:что было в главе|what was in chapter|chapter\s+\d+|глава\s+\d+)", re.I),
    re.compile(r"(?:где .*вызывается|где .*объявляется|where .*called|where .*declared)", re.I),
    re.compile(r"(?:где в коде|where in (?:the )?code|класс|class|function|method|код|файл|документ|chapter|глава)", re.I),
)
_CLAIM_QUERY_RE = re.compile(
    r"(?:нравит|люблю|обожаю|like|love|favorite|любим|не люблю|ненавиж|hate|dislike|использ|пользуюсь|use|using|предпочит|prefer|what do i use|what do i own|what do i like|what do i hate|что\s+я\s+использую|что\s+у\s+меня\s+есть)\b",
    re.I,
)
_SPONTANEOUS_CLAIM_QUERY_RE = re.compile(
    r"(?:что мне нравится|что я люблю|что я не люблю|what do i like|what do i hate)",
    re.I,
)
_CONTEXTUAL_FACT_PREDICATES = {"decision", "task", "task_goal", "agreed_plan"}
_CLAIM_SELF_HINTS: tuple[str, ...] = (
    "\u0443 \u043c\u0435\u043d\u044f",
    "\u043c\u043d\u0435",
    "\u043c\u0435\u043d\u044f",
    "\u043c\u043e\u0439",
    "\u043c\u043e\u044f",
    "\u043c\u043e\u044e",
    "\u043c\u043e\u0451",
    "my",
    "me",
    "i ",
)
_CLAIM_QUERY_FALLBACK_HINTS: tuple[str, ...] = (
    "\u043d\u0440\u0430\u0432\u0438\u0442",
    "\u043b\u044e\u0431\u043b\u044e",
    "\u043e\u0431\u043e\u0436\u0430\u044e",
    "\u043b\u044e\u0431\u0438\u043c",
    "\u043d\u0435 \u043b\u044e\u0431\u043b\u044e",
    "\u043d\u0435\u043d\u0430\u0432\u0438\u0436",
    "\u0442\u0435\u0440\u043f\u0435\u0442\u044c \u043d\u0435 \u043c\u043e\u0433\u0443",
    "\u0438\u0441\u043f\u043e\u043b\u044c\u0437",
    "\u043f\u043e\u043b\u044c\u0437\u0443\u044e\u0441\u044c",
    "\u044e\u0437\u0430\u044e",
    "\u043f\u0440\u0435\u0434\u043f\u043e\u0447\u0438\u0442",
)
_SPONTANEOUS_CLAIM_FALLBACK_PHRASES: tuple[str, ...] = (
    "\u0447\u0442\u043e \u043c\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0442",
    "\u0447\u0442\u043e \u044f \u043b\u044e\u0431\u043b\u044e",
    "\u0447\u0442\u043e \u044f \u043d\u0435 \u043b\u044e\u0431\u043b\u044e",
    "what do i like",
    "what do i hate",
)


def _contains_any_fragment(text: str, fragments: tuple[str, ...]) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    return any(str(fragment).strip().lower() in low for fragment in fragments if str(fragment).strip())


@dataclass(frozen=True)
class QueryRecallProfile:
    mode: str = ""
    self_like: bool = False
    claim_like: bool = False
    contextual_dialog: bool = False
    document_query: bool = False
    spontaneous_claim: bool = False


def normalize_claim_recall_mode(mode: str, *, fallback: str = RECALL_MODE_CONTEXTUAL) -> str:
    value = str(mode or "").strip().lower()
    if value in _VALID_CLAIM_RECALL_MODES:
        return value
    return str(fallback or RECALL_MODE_CONTEXTUAL).strip().lower() or RECALL_MODE_CONTEXTUAL


def default_claim_recall_mode(*, predicate: str, promotion_level: str = "") -> str:
    predicate_norm = str(predicate or "").strip().lower()
    promotion_norm = str(promotion_level or "").strip().lower()
    if predicate_norm in {"owns", "uses"}:
        return RECALL_MODE_EXACT
    if predicate_norm in {"likes", "dislikes"}:
        if promotion_norm == "strong_claim":
            return RECALL_MODE_AMBIENT
        return RECALL_MODE_CONTEXTUAL
    return RECALL_MODE_CONTEXTUAL


def default_claim_spontaneous_recall(*, predicate: str, promotion_level: str = "") -> bool:
    predicate_norm = str(predicate or "").strip().lower()
    promotion_norm = str(promotion_level or "").strip().lower()
    return predicate_norm in {"likes", "dislikes"} and promotion_norm == "strong_claim"


def classify_query_recall_profile(query_text: str) -> QueryRecallProfile:
    text = str(query_text or "").strip()
    low = text.lower()
    self_like = bool(_SELF_QUERY_RE.search(text))
    if not self_like and _EXACT_FACT_DOMAIN_RE.search(low):
        has_self_hint = bool(_EXACT_FACT_SELF_HINT_RE.search(low))
        has_recall_hint = bool(_EXACT_FACT_RECALL_HINT_RE.search(low)) or "?" in text
        self_like = has_self_hint and has_recall_hint
    contextual_dialog = any(pattern.search(low) for pattern in _CONTEXTUAL_RECALL_RULES)
    document_query = any(pattern.search(low) for pattern in _DOCUMENT_RECALL_RULES)
    claim_like = bool(_CLAIM_QUERY_RE.search(low)) or (
        _contains_any_fragment(low, _CLAIM_QUERY_FALLBACK_HINTS)
        and _contains_any_fragment(low, _CLAIM_SELF_HINTS)
    )
    spontaneous_claim = bool(_SPONTANEOUS_CLAIM_QUERY_RE.search(low)) or _contains_any_fragment(
        low,
        _SPONTANEOUS_CLAIM_FALLBACK_PHRASES,
    )

    mode = ""
    if self_like and not claim_like:
        mode = "exact_fact_recall"
    elif contextual_dialog:
        mode = "contextual_recall"
    elif document_query:
        mode = "document_recall"

    return QueryRecallProfile(
        mode=mode,
        self_like=self_like,
        claim_like=claim_like,
        contextual_dialog=contextual_dialog,
        document_query=document_query,
        spontaneous_claim=spontaneous_claim,
    )


def record_recall_mode(record: MemoryRecord) -> str:
    if record.level == MemoryLevel.L4_DOCUMENT or record.memory_type in {
        MemoryType.DOCUMENT,
        MemoryType.DOCUMENT_CHUNK,
    }:
        return RECALL_MODE_DOCUMENT

    meta = dict(record.metadata or {})
    if record.memory_type == MemoryType.CLAIM:
        claim = dict(meta.get("claim") or {})
        return normalize_claim_recall_mode(claim.get("recall_mode"), fallback=RECALL_MODE_CONTEXTUAL)

    if record.memory_type == MemoryType.EPISODE:
        return RECALL_MODE_CONTEXTUAL

    if record.memory_type == MemoryType.FACT:
        fact = dict(meta.get("fact") or {})
        predicate = str(fact.get("predicate") or "").strip().lower()
        if predicate in _CONTEXTUAL_FACT_PREDICATES:
            return RECALL_MODE_CONTEXTUAL
        return RECALL_MODE_EXACT

    if record.memory_type in {MemoryType.MESSAGE, MemoryType.SUMMARY, MemoryType.SEMANTIC}:
        return RECALL_MODE_CONTEXTUAL

    return RECALL_MODE_CONTEXTUAL


def recall_policy_adjustment(*, profile: QueryRecallProfile, record: MemoryRecord) -> float:
    mode = record_recall_mode(record)
    is_claim = record.memory_type == MemoryType.CLAIM
    is_episode = record.memory_type == MemoryType.EPISODE
    is_fact = record.memory_type == MemoryType.FACT

    if mode == RECALL_MODE_DOCUMENT:
        return 0.22 if profile.document_query else -0.36

    if mode == RECALL_MODE_EXACT:
        if is_fact and profile.mode == "exact_fact_recall":
            return 0.10
        if is_claim and profile.claim_like:
            return 0.10
        if profile.document_query:
            return -0.24
        if profile.contextual_dialog:
            return -0.20
        if profile.self_like:
            return -0.18 if is_claim else -0.08
        if profile.claim_like:
            return -0.16 if is_fact else 0.0
        return -0.10 if is_claim else -0.06

    if mode == RECALL_MODE_CONTEXTUAL:
        if is_episode:
            if profile.contextual_dialog:
                return 0.18
            if profile.document_query or profile.self_like:
                return -0.26
            if profile.claim_like:
                return -0.10
            return -0.06
        if is_claim:
            if profile.claim_like:
                return 0.10
            if profile.contextual_dialog or profile.document_query or profile.self_like:
                return -0.18
            return -0.08
        if profile.contextual_dialog:
            return 0.08
        if profile.document_query or profile.self_like:
            return -0.12
        return 0.0

    if mode == RECALL_MODE_AMBIENT:
        if profile.spontaneous_claim or profile.claim_like:
            return 0.04
        if profile.document_query or profile.contextual_dialog or profile.self_like:
            return -0.24
        return -0.12

    return 0.0
