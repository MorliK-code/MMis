from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.internet.web.web_models import QueryClassification, WebEvidencePack, WebPolicyDecision


@dataclass(frozen=True)
class WebMemoryBridgeResult:
    prompt_items: list[dict[str, Any]]
    memory_candidates: list[dict[str, Any]]


class WebMemoryBridge:
    def __init__(self, *, ttl_days: dict[str, int] | None = None):
        self._ttl_days = _ttl_defaults(ttl_days)

    def build(
        self,
        *,
        evidence_pack: WebEvidencePack,
        decision: WebPolicyDecision,
        classification: QueryClassification,
    ) -> WebMemoryBridgeResult:
        prompt_items: list[dict[str, Any]] = []
        memory_candidates: list[dict[str, Any]] = []

        stability = _stability_category(classification.primary_category)
        ttl_days = int(self._ttl_days.get(stability, self._ttl_days["default"]))
        ttl_sec = max(3600, ttl_days * 86400)

        for item in list(evidence_pack.items or []):
            snippet = str(item.snippet or "").strip()
            body = str(item.text or "").strip()
            marker = "[WEB]" if body else "[WEB_SEARCH]"
            confidence = 0.82 if body else 0.68
            priority = 0.86 if body else 0.72
            prompt_row = {
                "text": (
                    f"{marker} {item.title}\n"
                    f"source_url: {item.url}\n"
                    f"source_domain: {item.domain}\n"
                    f"source_published: {item.published_date or '-'}\n"
                    f"fetched_at: {item.fetched_at or '-'}\n"
                    f"snippet: {snippet}\n"
                    f"text: {body if body else '(page scrape blocked, snippet-only fallback)'}"
                ).strip(),
                "confidence": float(confidence),
                "priority": float(priority),
                "source": str(item.domain or "web"),
                "topic": "web:evidence",
                "relevant": True,
                "source_url": str(item.url),
                "source_domain": str(item.domain),
                "published_date": str(item.published_date or ""),
                "fetched_at": str(item.fetched_at or ""),
                "clean_method": str(item.clean_method or ("search_snippet_fallback" if not body else "scrape")),
                "web_v2_origin": True,
                "web_v2_stability": stability,
                "web_v2_ttl_sec": int(ttl_sec),
            }
            prompt_items.append(prompt_row)

            if _is_stable_candidate(item, stability=stability):
                memory_candidates.append(
                    {
                        "source_url": str(item.url),
                        "source_domain": str(item.domain),
                        "title": str(item.title),
                        "stability": stability,
                        "ttl_sec": int(ttl_sec),
                        "trust_tier": str(item.trust_tier),
                        "fetched_at": str(item.fetched_at or ""),
                    }
                )

        return WebMemoryBridgeResult(prompt_items=prompt_items, memory_candidates=memory_candidates)


def _is_stable_candidate(item, *, stability: str) -> bool:
    tier = str(getattr(item, "trust_tier", "") or "").strip().lower()
    if stability in {"prices", "news"}:
        return False
    return tier in {"official", "institutional", "reputable_tech"}


def _stability_category(primary_category: str) -> str:
    value = str(primary_category or "").strip().lower()
    if value in {"price"}:
        return "prices"
    if value in {"news"}:
        return "news"
    if value in {"version"}:
        return "versions"
    if value in {"mixed"}:
        return "market_compare"
    return "docs_summary"


def _ttl_defaults(payload: dict[str, int] | None) -> dict[str, int]:
    out = {
        "default": 7,
        "prices": 1,
        "versions": 14,
        "news": 2,
        "docs_summary": 30,
        "market_compare": 7,
    }
    for key, value in dict(payload or {}).items():
        name = str(key or "").strip()
        if not name:
            continue
        try:
            out[name] = max(1, int(value))
        except Exception:
            continue
    return out
