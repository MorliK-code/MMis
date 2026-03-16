from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.internet.web.web_models import QueryClassification, WebEvidencePack, WebPolicyDecision


@dataclass(frozen=True)
class WebMemoryBridgeResult:
    prompt_items: list[dict[str, Any]]
    memory_candidates: list[dict[str, Any]]
    memory_writes: list[dict[str, Any]]


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
        memory_writes: list[dict[str, Any]] = []

        stability = _stability_category(classification.primary_category)
        ttl_days = int(self._ttl_days.get(stability, self._ttl_days["default"]))
        ttl_sec = max(3600, ttl_days * 86400)
        stable_limit = 3
        temporary_limit = 2
        stable_count = 0
        temporary_count = 0

        for item in list(evidence_pack.items or []):
            snippet = str(item.snippet or "").strip()
            body = str(item.text or "").strip()
            marker = "[WEB]" if body else "[WEB_SEARCH]"
            confidence = 0.82 if body else 0.68
            priority = 0.86 if body else 0.72
            prompt_row = {
                "title": str(item.title or "").strip(),
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
                "trust_tier": str(item.trust_tier or ""),
                "web_v2_origin": True,
                "web_v2_stability": stability,
                "web_v2_ttl_sec": int(ttl_sec),
                "web_v2_conflict_flags": list(item.conflict_flags or []),
                "web_v2_source_audit": dict(getattr(item, "audit", {}) or {}),
            }
            prompt_items.append(prompt_row)

            if _is_stable_candidate(item, stability=stability):
                if stable_count < stable_limit:
                    write_row = _build_stable_write(
                        item=item,
                        classification=classification,
                        decision=decision,
                        stability=stability,
                    )
                    if write_row:
                        memory_writes.append(write_row)
                        stable_count += 1
                        memory_candidates.append(
                            {
                                "source_url": str(item.url),
                                "source_domain": str(item.domain),
                                "title": str(item.title),
                                "stability": stability,
                                "ttl_sec": int(ttl_sec),
                                "trust_tier": str(item.trust_tier),
                                "fetched_at": str(item.fetched_at or ""),
                                "source_audit": dict(getattr(item, "audit", {}) or {}),
                                "write_scope": "project",
                                "write_type": "stable",
                            }
                        )
                continue

            if _is_temporary_candidate(stability=stability):
                if temporary_count >= temporary_limit:
                    continue
                temp_row = _build_temporary_write(
                    item=item,
                    classification=classification,
                    decision=decision,
                    stability=stability,
                    ttl_sec=ttl_sec,
                )
                if temp_row:
                    memory_writes.append(temp_row)
                    temporary_count += 1
                    memory_candidates.append(
                        {
                            "source_url": str(item.url),
                            "source_domain": str(item.domain),
                            "title": str(item.title),
                            "stability": stability,
                            "ttl_sec": int(ttl_sec),
                            "trust_tier": str(item.trust_tier),
                            "fetched_at": str(item.fetched_at or ""),
                            "source_audit": dict(getattr(item, "audit", {}) or {}),
                            "write_scope": "temporary",
                            "write_type": "temporary",
                        }
                    )

        return WebMemoryBridgeResult(
            prompt_items=prompt_items,
            memory_candidates=memory_candidates,
            memory_writes=memory_writes,
        )


def _is_stable_candidate(item, *, stability: str) -> bool:
    tier = str(getattr(item, "trust_tier", "") or "").strip().lower()
    if stability in {"prices", "news", "market_compare"}:
        return False
    if not _is_memory_safe_candidate(item):
        return False
    return tier in {
        "official_docs",
        "official_repo",
        "official_repo_release",
        "vendor_docs",
        "institutional",
        "reputable_tech",
    }


def _is_temporary_candidate(*, stability: str) -> bool:
    return stability in {"prices", "news", "market_compare"}


def _is_memory_safe_candidate(item) -> bool:
    audit = dict(getattr(item, "audit", {}) or {})
    if list(getattr(item, "conflict_flags", []) or []):
        return False
    if float(audit.get("numeric_conflict_severity") or 0.0) >= 0.55:
        return False
    if float(getattr(item, "quality_score", 0.0) or 0.0) < 0.42:
        return False
    return True


def _build_stable_write(
    *,
    item,
    classification: QueryClassification,
    decision: WebPolicyDecision,
    stability: str,
) -> dict[str, Any] | None:
    source_url = str(getattr(item, "url", "") or "").strip()
    source_domain = str(getattr(item, "domain", "") or "").strip().lower()
    if not source_url or not source_domain:
        return None
    facts = [str(x or "").strip() for x in list(getattr(item, "key_facts", []) or []) if str(x or "").strip()]
    canonical_fact = facts[0] if facts else str(getattr(item, "snippet", "") or "").strip()
    if not canonical_fact:
        return None
    text = (
        f"[WEB_STABLE] {str(getattr(item, 'title', '') or '').strip()}\n"
        f"fact: {canonical_fact}\n"
        f"source_url: {source_url}\n"
        f"source_domain: {source_domain}\n"
        f"published_at: {str(getattr(item, 'published_date', '') or '').strip() or '-'}\n"
        f"fetched_at: {str(getattr(item, 'fetched_at', '') or '').strip() or '-'}"
    ).strip()
    return {
        "write_type": "stable",
        "scope": "project",
        "memory_type": "semantic",
        "text": text,
        "confidence": _stable_confidence(item),
        "importance": 0.72,
        "metadata": {
            "source": "web_v2",
            "web_v2_origin": True,
            "web_v2_write_type": "stable",
            "web_v2_stability": str(stability),
            "web_v2_mode": str(getattr(decision, "mode", "") or ""),
            "web_v2_query_type": str(getattr(classification, "query_type", "") or ""),
            "web_v2_primary_category": str(getattr(classification, "primary_category", "") or ""),
            "source_url": source_url,
            "source_domain": source_domain,
            "source_published": str(getattr(item, "published_date", "") or ""),
            "fetched_at": str(getattr(item, "fetched_at", "") or ""),
            "trust_tier": str(getattr(item, "trust_tier", "") or ""),
            "conflict_flags": list(getattr(item, "conflict_flags", []) or []),
            "web_v2_source_audit": dict(getattr(item, "audit", {}) or {}),
        },
    }


def _build_temporary_write(
    *,
    item,
    classification: QueryClassification,
    decision: WebPolicyDecision,
    stability: str,
    ttl_sec: int,
) -> dict[str, Any] | None:
    source_url = str(getattr(item, "url", "") or "").strip()
    source_domain = str(getattr(item, "domain", "") or "").strip().lower()
    if not source_url or not source_domain:
        return None
    facts = [str(x or "").strip() for x in list(getattr(item, "key_facts", []) or []) if str(x or "").strip()]
    fact = facts[0] if facts else str(getattr(item, "snippet", "") or "").strip()
    if not fact:
        return None
    text = (
        f"[WEB_TEMP] {str(getattr(item, 'title', '') or '').strip()}\n"
        f"fact: {fact}\n"
        f"source_url: {source_url}\n"
        f"source_domain: {source_domain}\n"
        f"fetched_at: {str(getattr(item, 'fetched_at', '') or '').strip() or '-'}"
    ).strip()
    return {
        "write_type": "temporary",
        "scope": "temporary",
        "memory_type": "semantic",
        "ttl_sec": max(3600, int(ttl_sec)),
        "text": text,
        "confidence": 0.62,
        "importance": 0.46,
        "metadata": {
            "source": "web_v2",
            "web_v2_origin": True,
            "web_v2_write_type": "temporary",
            "web_v2_stability": str(stability),
            "web_v2_mode": str(getattr(decision, "mode", "") or ""),
            "web_v2_query_type": str(getattr(classification, "query_type", "") or ""),
            "web_v2_primary_category": str(getattr(classification, "primary_category", "") or ""),
            "source_url": source_url,
            "source_domain": source_domain,
            "source_published": str(getattr(item, "published_date", "") or ""),
            "fetched_at": str(getattr(item, "fetched_at", "") or ""),
            "trust_tier": str(getattr(item, "trust_tier", "") or ""),
            "conflict_flags": list(getattr(item, "conflict_flags", []) or []),
            "web_v2_source_audit": dict(getattr(item, "audit", {}) or {}),
        },
    }


def _stable_confidence(item) -> float:
    tier = str(getattr(item, "trust_tier", "") or "").strip().lower()
    if tier in {"official_docs", "official_repo_release", "vendor_docs"}:
        return 0.86
    if tier in {"official_repo", "institutional"}:
        return 0.80
    return 0.72


def _stability_category(primary_category: str) -> str:
    value = str(primary_category or "").strip().lower()
    if value in {"price", "finance", "weather"}:
        return "prices"
    if value in {"news", "external"}:
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
