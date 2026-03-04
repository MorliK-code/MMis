from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from metadata.taxonomy import normalize_intent


@dataclass(frozen=True)
class IntentResult:
    label: str
    conf: float
    alt_labels: list[str] = field(default_factory=list)


_PATTERNS = {
    "bug_report": [
        r"\btraceback\b",
        r"\bexception\b",
        r"\berror\b",
        r"\bdebug\b",
        r"\bstack trace\b",
        r"\binternal server error\b",
        r"\blogs?\b",
        r"```",
        r"[A-Za-z]:\\",
        r"\bbug\b",
    ],
    "code_review": [
        r"\bcode review\b",
        r"\breview this code\b",
        r"\blook at this diff\b",
        r"\bревью\b",
    ],
    "planning": [
        r"\bplan\b",
        r"\broadmap\b",
        r"\bphases?\b",
        r"\bmilestones?\b",
        r"\bэтап",
        r"\bплан\b",
    ],
    "clarification": [
        r"\bclarify\b",
        r"\bwhat do you mean\b",
        r"\bcan you explain\b",
        r"\bуточни\b",
        r"\bне понял\b",
    ],
    "task": [
        r"\bсделай\b|\bвыполни\b|\bнастрой\b|\bсоздай\b",
        r"\bdo\b|\bmake\b|\bbuild\b|\bconfigure\b|\bplease\b",
    ],
    "question": [
        r"\?$",
        r"\bкак\b|\bчто\b|\bпочему\b|\bзачем\b|\bгде\b",
        r"\bhow\b|\bwhat\b|\bwhy\b|\bwhere\b|\bwhen\b",
    ],
}


def classify(text: str, lang: str, context_tags: dict[str, Any] | None = None) -> IntentResult:
    src = str(text or "").strip()
    lower = src.lower()
    tags = dict(context_tags or {})

    scores = {
        "chat": 0.22,
        "question": 0.16,
        "task": 0.16,
        "bug_report": 0.12,
        "code_review": 0.08,
        "planning": 0.07,
        "clarification": 0.07,
    }

    if not src:
        return IntentResult(label="chat", conf=0.35, alt_labels=["question"])

    for label, patterns in _PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, src, re.I | re.S):
                scores[label] += 0.28

    if "?" in src:
        scores["question"] += 0.15
    if any(token in lower for token in ("help", "помоги")):
        scores["task"] += 0.14
    if any(token in lower for token in ("review", "ревью", "pull request", "pr ")):
        scores["code_review"] += 0.28
    if any(token in lower for token in ("git", "branch", "commit", "merge", "pytest")):
        scores["task"] += 0.16
    if "```" in src:
        scores["task"] += 0.20
    if any(token in lower for token in ("traceback", "exception", "internal server error", "doesn't work", "не работает")):
        scores["bug_report"] += 0.32
    if any(token in lower for token in ("plan", "roadmap", "этап", "план")):
        scores["planning"] += 0.25
    if any(token in lower for token in ("уточни", "clarify", "что имеешь в виду")):
        scores["clarification"] += 0.24
    if lang in {"mixed", "unknown"} and len(src) > 40:
        scores["question"] += 0.04

    mode = str(tags.get("mode") or "").lower()
    if mode == "coding":
        scores["task"] += 0.14
    if mode == "task":
        scores["task"] += 0.12
    if str(tags.get("last_intent") or "") == "bug_report":
        scores["bug_report"] += 0.08

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    conf = best_score / max(0.001, best_score + second_score)
    conf = max(0.0, min(1.0, conf))

    alt = [normalize_intent(name) for name, score in ordered[1:4] if score >= 0.18]
    return IntentResult(label=normalize_intent(best_label), conf=conf, alt_labels=alt)


def classify_intent(text: str) -> str:
    result = classify(text=text, lang="unknown", context_tags=None)
    return result.label
