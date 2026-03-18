from __future__ import annotations

import re
from dataclasses import dataclass

from memory.memory_models import MemoryLevel, MemoryRecord, MemoryType


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
    re.compile(r"(?:что мы обсуждали|что обсуждали|что решили|what we decided|what did we decide)", re.I),
    re.compile(r"(?:почему так сделали|why did we do that|why we did that)", re.I),
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


@dataclass(frozen=True)
class QueryRecallProfile:
    mode: str = ""
    self_like: bool = False
    claim_like: bool = False
    contextual_dialog: bool = False
    document_query: bool = False
    spontaneous_claim: bool = False


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
    claim_like = bool(_CLAIM_QUERY_RE.search(low))
    spontaneous_claim = bool(_SPONTANEOUS_CLAIM_QUERY_RE.search(low))

    mode = ""
    if self_like:
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
        return "document"

    meta = dict(record.metadata or {})
    if record.memory_type == MemoryType.CLAIM:
        claim = dict(meta.get("claim") or {})
        return str(claim.get("recall_mode") or "contextual").strip().lower() or "contextual"

    if record.memory_type == MemoryType.EPISODE:
        return "contextual"

    if record.memory_type == MemoryType.FACT:
        fact = dict(meta.get("fact") or {})
        predicate = str(fact.get("predicate") or "").strip().lower()
        if predicate in _CONTEXTUAL_FACT_PREDICATES:
            return "contextual"
        return "exact"

    if record.memory_type in {MemoryType.MESSAGE, MemoryType.SUMMARY, MemoryType.SEMANTIC}:
        return "contextual"

    return "contextual"


def recall_policy_adjustment(*, profile: QueryRecallProfile, record: MemoryRecord) -> float:
    mode = record_recall_mode(record)

    if mode == "document":
        return 0.20 if profile.document_query else -0.30

    if mode == "exact":
        if profile.mode == "exact_fact_recall":
            return 0.08
        if profile.document_query:
            return -0.18
        if profile.contextual_dialog:
            return -0.14
        return -0.04

    if mode == "contextual":
        if record.memory_type == MemoryType.EPISODE:
            if profile.contextual_dialog:
                return 0.16
            if profile.document_query or profile.self_like:
                return -0.22
            return -0.04
        if record.memory_type == MemoryType.CLAIM:
            if profile.claim_like:
                return 0.08
            if profile.contextual_dialog or profile.document_query or profile.self_like:
                return -0.12
            return -0.04
        if profile.contextual_dialog:
            return 0.08
        if profile.document_query or profile.self_like:
            return -0.10
        return 0.0

    if mode == "ambient":
        if profile.spontaneous_claim or profile.claim_like:
            return 0.04
        if profile.document_query or profile.contextual_dialog or profile.self_like:
            return -0.20
        return -0.08

    return 0.0
