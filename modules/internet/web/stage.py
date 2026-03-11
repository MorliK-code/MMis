from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from typing import Any

from config.settings import load_config
from modules.internet.search import SearchClient, SearchResult
from modules.internet.scraper import WebScraper
from modules.internet.web.confidence import assess_confidence
from modules.internet.web.freshness import assess_freshness
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.query_planner import build_query_plan
from modules.internet.web.result_processor import build_evidence_pack
from modules.internet.web.search_manager import SearchExecutionResult, SearchManager
from modules.internet.web.source_ranker import rank_sources
from modules.internet.web.web_memory_bridge import WebMemoryBridge
from modules.internet.web.web_models import WebSearchMode
from modules.internet.web.web_policy import WebPolicyEngine, config_from_dict, default_policy_config
from utils.logger import get_logger, log_json


LOGGER = get_logger(__name__)
_SMALLTALK_RE = re.compile(r"\b(hello|hi|how are you|привет|как дела|что нового)\b", flags=re.I)


@dataclass
class WebStageConfig:
    k_search: int = 5
    k_fetch: int = 2
    max_text_chars: int = 900


class WebStageV2:
    name = "web_retrieve"

    def __init__(
        self,
        *,
        search_client: SearchClient | None = None,
        scraper: WebScraper | None = None,
        cfg: WebStageConfig | None = None,
    ):
        app = load_config()
        self._app = app
        self._cfg = cfg or WebStageConfig()
        self._search = search_client or SearchClient(
            endpoint=app.search_api_url,
            provider=app.search_provider,
            strict_endpoint=bool(app.search_strict_endpoint),
            timeout_s=float(app.search_timeout_sec),
            cache_dir=app.cache_dir,
            cache_ttl_s=900,
        )
        self._scraper = scraper or WebScraper(
            cache_dir=app.cache_dir,
            cache_ttl_s=1800,
            timeout_s=int(app.web_fetch_timeout_sec),
            retries=int(app.web_fetch_retries),
            clean_max_chars=int(app.web_clean_max_chars),
            clean_min_chars=int(app.web_clean_min_chars),
            clean_language_hint=str(app.web_clean_language_hint or ""),
        )

        web_v2_cfg = _as_dict(getattr(app, "web_v2", {}))
        policy_payload = _as_dict(web_v2_cfg)
        self._policy = WebPolicyEngine(config_from_dict(policy_payload) if policy_payload else default_policy_config())

        ttl_days = _as_dict(web_v2_cfg.get("ttl_days") or web_v2_cfg.get("ttl"))
        self._memory_bridge = WebMemoryBridge(ttl_days={k: _to_int(v, 0) for k, v in ttl_days.items()})
        self._search_manager = SearchManager(
            search_client=self._search,
            scraper=self._scraper,
            cooldown_seconds=_to_int(web_v2_cfg.get("cooldown_seconds"), 45),
            retry_policy=_as_dict(web_v2_cfg.get("retry_policy")),
        )

        self._preferred_domains = [
            str(x or "").strip().lower()
            for x in list(web_v2_cfg.get("preferred_domains") or [])
            if str(x or "").strip()
        ]
        self._blocked_domains = [
            str(x or "").strip().lower()
            for x in list(web_v2_cfg.get("blocked_domains") or [])
            if str(x or "").strip()
        ]
        citations_cfg = _as_dict(web_v2_cfg.get("citations"))
        self._citation_max_fact = max(1, _to_int(citations_cfg.get("max_items_fact"), 1))
        self._citation_max_compare = max(1, _to_int(citations_cfg.get("max_items_compare"), 3))

    def run(self, ctx):
        trace = str(_as_dict(getattr(ctx, "meta", {})).get("web_trace_id") or "").strip() or "-"
        mode = _resolve_web_mode(_as_dict(ctx.meta), _as_dict(ctx.state))
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_mode"] = mode

        text = str(getattr(ctx, "clean_user_msg", "") or getattr(ctx, "user_msg", "") or "").strip()
        preview = _clip(text.replace("\n", " "), 120)
        ctx.logs.append(f"stage=web_retrieve start trace={trace} mode={mode} text_len={len(text)} text_preview={preview}")
        log_json(LOGGER, "web_retrieve_start", trace=trace, mode=mode, text_len=len(text), text_preview=preview)
        _emit_trace_event(ctx, "web_retrieve_start", {"mode": mode, "text_len": len(text), "text_preview": preview})

        if not text:
            _set_web_flags(
                ctx,
                web_query_intent="generic",
                web_search_mode=WebSearchMode.NO_SEARCH.value,
                fresh_required=False,
                fresh_missing=False,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve skipped(empty_text) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="empty_text")
            _emit_trace_event(ctx, "web_retrieve_skip", {"reason": "empty_text"})
            return ctx

        user_override = _resolve_user_override(
            text=text,
            meta=_as_dict(getattr(ctx, "meta", {})),
            state=_as_dict(getattr(ctx, "state", {})),
        )
        query = _strip_web_prefix(text)
        if not query:
            _set_web_flags(
                ctx,
                web_query_intent="generic",
                web_search_mode=WebSearchMode.NO_SEARCH.value,
                fresh_required=False,
                fresh_missing=False,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve skipped(empty_query) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="empty_query")
            _emit_trace_event(ctx, "web_retrieve_skip", {"reason": "empty_query"})
            return ctx

        classification = classify_query(query, metadata_tags=_as_list(_as_dict(getattr(ctx, "tags", {})).get("metadata_tags")))
        if _is_smalltalk_query(query, ctx_tags=_as_dict(getattr(ctx, "tags", {}))):
            _set_web_flags(
                ctx,
                web_query_intent="generic",
                web_search_mode=WebSearchMode.NO_SEARCH.value,
                fresh_required=False,
                fresh_missing=False,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve skipped(smalltalk_hard_skip) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", trace=trace, reason="smalltalk_hard_skip")
            _emit_trace_event(ctx, "web_retrieve_skip", {"reason": "smalltalk_hard_skip"})
            return ctx

        confidence = assess_confidence(
            query=query,
            classification=classification,
            retrieved_memories=_as_list(getattr(ctx, "retrieved_memories", [])),
            memory_context=_as_dict(getattr(ctx, "memory_context", {})),
        )
        web_items = [x for x in _as_list(getattr(ctx, "retrieved_memories", [])) if _is_web_item(_as_dict(x))]
        freshness = assess_freshness(
            query=query,
            classification=classification,
            web_items=web_items,
            ttl_days=_as_dict(_as_dict(getattr(self._app, "web_v2", {})).get("ttl_days") or _as_dict(getattr(self._app, "web_v2", {})).get("ttl")),
        )
        compat_intent = _compat_query_intent(query)

        _emit_trace_event(ctx, "web_classification", classification.to_dict())
        _emit_trace_event(ctx, "web_confidence_assessment", confidence.to_dict())
        _emit_trace_event(ctx, "web_freshness_assessment", freshness.to_dict())

        decision = self._policy.decide(
            query=query,
            classification=classification,
            confidence=confidence,
            freshness=freshness,
            web_mode=mode,
            internet_enabled=bool(self._app.internet_enabled),
            user_override=user_override,
            policy_context=_as_dict(getattr(ctx, "meta", {})),
        )
        decision_payload = decision.to_dict()
        decision_payload["classification"] = classification.to_dict()
        decision_payload["confidence"] = confidence.to_dict()
        decision_payload["freshness"] = freshness.to_dict()
        decision_payload["user_override"] = str(user_override or "")
        _emit_trace_event(ctx, "web_policy_decision", decision_payload)

        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_query"] = str(query)
            ctx.meta["web_user_override"] = str(user_override or "")
            ctx.meta["web_decision_breakdown"] = dict(decision.decision_breakdown or {})
            ctx.meta["web_query_classification"] = classification.to_dict()
            ctx.meta["web_confidence_assessment"] = confidence.to_dict()
            ctx.meta["web_freshness_assessment"] = freshness.to_dict()
            ctx.meta["web_need_score"] = float(decision.web_need_score)

        _set_web_flags(
            ctx,
            web_query_intent=compat_intent,
            web_search_mode=decision.mode.value,
            fresh_required=bool(freshness.needs_refresh or classification.requires_freshness),
            fresh_missing=False,
            web_used=False,
        )

        ctx.logs.append(
            "stage=web_retrieve decision "
            f"trace={trace} mode={decision.mode.value} should_search={int(bool(decision.should_search))} "
            f"score={decision.web_need_score:.3f} reason={decision.reason} intent={compat_intent} "
            f"local_scope_cap={int(bool(decision.local_scope_cap_applied))} category_penalty={decision.category_penalty_applied:.3f}"
        )
        log_json(
            LOGGER,
            "web_retrieve_decision",
            trace=trace,
            mode=str(decision.mode.value),
            should_search=bool(decision.should_search),
            decision_score=float(decision.web_need_score),
            decision_reason=str(decision.reason),
            category_penalty=float(decision.category_penalty_applied),
            local_scope_cap=bool(decision.local_scope_cap_applied),
            decision_breakdown=dict(decision.decision_breakdown or {}),
        )
        _emit_trace_event(
            ctx,
            "web_retrieve_decision",
            {
                "mode": str(decision.mode.value),
                "should_search": bool(decision.should_search),
                "decision_score": float(decision.web_need_score),
                "decision_reason": str(decision.reason),
                "decision_breakdown": dict(decision.decision_breakdown or {}),
            },
        )

        if not bool(decision.should_search):
            fresh_required = bool(freshness.needs_refresh or classification.requires_freshness)
            fresh_missing = bool(fresh_required)
            _set_web_flags(
                ctx,
                web_query_intent=compat_intent,
                web_search_mode=decision.mode.value,
                fresh_required=fresh_required,
                fresh_missing=fresh_missing,
                web_used=False,
            )
            if isinstance(getattr(ctx, "meta", None), dict):
                ctx.meta["web_guardrail_local_reply"] = bool(
                    fresh_missing and compat_intent in {"fx_rate", "weather"}
                )
            ctx.logs.append(f"stage=web_retrieve skipped(policy_no_search) trace={trace}")
            log_json(
                LOGGER,
                "web_retrieve_skip",
                trace=trace,
                reason="policy_no_search",
                mode=str(decision.mode.value),
                score=float(decision.web_need_score),
            )
            _emit_trace_event(
                ctx,
                "web_retrieve_skip",
                {"reason": "policy_no_search", "mode": str(decision.mode.value), "score": float(decision.web_need_score)},
            )
            return ctx

        plan = build_query_plan(
            query=query,
            classification=classification,
            decision=decision,
            preferred_domains=self._preferred_domains,
        )
        _emit_trace_event(ctx, "web_query_plan", plan.to_dict())

        ctx.logs.append(
            "stage=web_retrieve search_start "
            f"trace={trace} mode={decision.mode.value} max_queries={decision.budget.max_queries} "
            f"max_sources={decision.budget.max_sources} max_fetches={_budget_fetches(decision)}"
        )
        _emit_trace_event(
            ctx,
            "web_search_request",
            {
                "query": str(query),
                "mode": str(decision.mode.value),
                "queries": plan.all_queries(),
                "query_roles": dict(plan.query_roles or {}),
                "max_queries": int(decision.budget.max_queries),
                "max_sources": int(decision.budget.max_sources),
                "max_fetches": int(_budget_fetches(decision)),
            },
        )

        try:
            executed = self._search_manager.execute(
                plan=plan,
                decision=decision,
                classification=classification,
                freshness=freshness,
            )
        except Exception as exc:
            fresh_required = bool(freshness.needs_refresh or classification.requires_freshness)
            _set_web_flags(
                ctx,
                web_query_intent=compat_intent,
                web_search_mode=decision.mode.value,
                fresh_required=fresh_required,
                fresh_missing=fresh_required,
                web_used=False,
            )
            if isinstance(getattr(ctx, "meta", None), dict):
                ctx.meta["web_guardrail_local_reply"] = bool(
                    fresh_required and compat_intent in {"fx_rate", "weather"}
                )
            ctx.logs.append(f"stage=web_retrieve search_fail trace={trace} err={type(exc).__name__}")
            log_json(LOGGER, "web_retrieve_search_fail", trace=trace, error=type(exc).__name__)
            _emit_trace_event(ctx, "web_search_fail", {"error": type(exc).__name__, "mode": str(decision.mode.value)})
            return ctx

        ranked = rank_sources(
            items=list(executed.results),
            preferred_domains=self._preferred_domains,
            blocked_domains=self._blocked_domains,
        )
        domains = _top_domains([x.item for x in ranked], limit=5)
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_result_count"] = int(len(ranked))
            ctx.meta["web_result_domains"] = list(domains)

        ctx.logs.append(
            "stage=web_retrieve search_done "
            f"trace={trace} results={len(ranked)} domains={','.join(domains) if domains else '-'}"
        )
        _emit_trace_event(
            ctx,
            "web_search_results",
            {
                "query": str(query),
                "mode": str(decision.mode.value),
                "queries_used": list(executed.queries_used),
                "scout_queries_used": list(getattr(executed, "scout_queries_used", []) or []),
                "focused_queries_used": list(getattr(executed, "focused_queries_used", []) or []),
                "retries_used": int(getattr(executed, "retries_used", 0) or 0),
                "cooldown_applied": bool(getattr(executed, "cooldown_applied", False)),
                "result_count": len(ranked),
                "domains": list(domains),
                "top_results": [
                    {
                        "rank": idx + 1,
                        "title": str(row.item.title or ""),
                        "url": str(row.item.url or ""),
                        "domain": str(row.item.source or ""),
                        "published_date": str(row.item.published_date or ""),
                        "score_total": float(row.item.score or 0.0),
                        "score_breakdown": dict(row.item.score_breakdown or {}),
                        "snippet_500": _snippet500(str(row.item.snippet or "")),
                        "snippet_sha1": _sha1_text(str(row.item.snippet or "")),
                        "snippet_len": len(str(row.item.snippet or "")),
                    }
                    for idx, row in enumerate(list(ranked)[:5])
                ],
            },
        )
        _emit_trace_event(
            ctx,
            "web_budget_update",
            {
                "mode": str(decision.mode.value),
                "budget": (
                    decision.budget.to_dict()
                    if hasattr(decision.budget, "to_dict")
                    else {
                        "max_queries": int(getattr(decision.budget, "max_queries", 0) or 0),
                        "max_sources": int(getattr(decision.budget, "max_sources", 0) or 0),
                        "max_pages": int(getattr(decision.budget, "max_pages", 0) or 0),
                        "max_fetches": int(_budget_fetches(decision)),
                    }
                ),
                "used": {
                    "queries": int(len(list(executed.queries_used or []))),
                    "sources": int(len(ranked)),
                    "fetches": int(len(list(executed.fetched_pages or {}))),
                    "scout_queries": int(len(list(getattr(executed, "scout_queries_used", []) or []))),
                    "focused_queries": int(len(list(getattr(executed, "focused_queries_used", []) or []))),
                },
                "cooldown_applied": bool(getattr(executed, "cooldown_applied", False)),
                "retries_used": int(getattr(executed, "retries_used", 0) or 0),
            },
        )

        evidence = build_evidence_pack(
            ranked_results=ranked,
            fetched_pages=executed.fetched_pages,
            decision=decision,
            classification=classification,
        )
        bridge = self._memory_bridge.build(
            evidence_pack=evidence,
            decision=decision,
            classification=classification,
        )
        prompt_items = list(bridge.prompt_items or [])
        web_evidence_context = _build_web_evidence_context(
            evidence=evidence,
            prompt_items=prompt_items,
            mode=str(decision.mode.value),
            query=str(query),
        )
        _emit_trace_event(
            ctx,
            "web_evidence_pack",
            {
                "items": int(len(list(evidence.items or []))),
                "citations": int(len(list(evidence.compact_citations or []))),
                "key_facts": int(len(list(evidence.key_facts or []))),
                "conflicting_sources": bool(evidence.conflicting_sources),
                "conflict_notes": list(evidence.conflict_notes or []),
            },
        )

        fetched = 0
        for item in list(prompt_items):
            row = dict(item or {})
            text_payload = str(row.get("text") or "")
            if not text_payload:
                continue
            if "search_snippet_fallback" in str(row.get("clean_method") or ""):
                ctx.logs.append(
                    "stage=web_retrieve fetch_snippet_fallback "
                    f"trace={trace} domain={str(row.get('source_domain') or '-')} snippet_len={len(str(row.get('text') or ''))}"
                )
            else:
                ctx.logs.append(
                    "stage=web_retrieve fetch_ok "
                    f"trace={trace} domain={str(row.get('source_domain') or '-')} text_len={len(str(row.get('text') or ''))}"
                )
            _emit_trace_event(
                ctx,
                "web_fetch_item",
                {
                    "url": str(row.get("source_url") or ""),
                    "domain": str(row.get("source_domain") or ""),
                    "status": 0 if row.get("clean_method") == "search_snippet_fallback" else 200,
                    "clean_method": str(row.get("clean_method") or ""),
                    "text_500": _snippet500(str(row.get("text") or "")),
                    "text_sha1": _sha1_text(str(row.get("text") or "")),
                    "text_len": len(str(row.get("text") or "")),
                },
            )
            fetched += 1

        web_used = bool(fetched > 0)
        fresh_required = bool(freshness.needs_refresh or classification.requires_freshness)
        fresh_missing = bool(fresh_required and not web_used)
        _set_web_flags(
            ctx,
            web_query_intent=compat_intent,
            web_search_mode=decision.mode.value,
            fresh_required=fresh_required,
            fresh_missing=fresh_missing,
            web_used=web_used,
        )

        planned_memory_writes = list(bridge.memory_writes or [])
        stored_memory_writes = False
        if bool(_as_dict(getattr(ctx, "meta", {})).get("store_turn", True)):
            if isinstance(getattr(ctx, "memory_ops", None), list) and planned_memory_writes:
                namespace = str(
                    _pick(
                        _as_dict(getattr(ctx, "meta", {})).get("conversation_id"),
                        _as_dict(getattr(ctx, "state", {})).get("conversation_id"),
                        "default",
                    )
                ).strip() or "default"
                ctx.memory_ops.append(
                    {
                        "op": "web_memory_write",
                        "namespace": namespace,
                        "items": planned_memory_writes,
                    }
                )
                stored_memory_writes = True
        _emit_trace_event(
            ctx,
            "web_memory_write",
            {
                "planned": int(len(planned_memory_writes)),
                "stable": int(sum(1 for x in planned_memory_writes if str(_as_dict(x).get("write_type") or "") == "stable")),
                "temporary": int(sum(1 for x in planned_memory_writes if str(_as_dict(x).get("write_type") or "") == "temporary")),
                "queued": bool(stored_memory_writes),
            },
        )
        ctx.logs.append(
            "stage=web_retrieve memory_write "
            f"planned={len(planned_memory_writes)} queued={int(bool(stored_memory_writes))}"
        )

        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_fetched"] = int(fetched)
            ctx.meta["web_used"] = bool(web_used)
            ctx.meta["web_citations"] = list(_adaptive_citations(evidence.compact_citations, classification, max_fact=self._citation_max_fact, max_compare=self._citation_max_compare))
            ctx.meta["web_evidence_pack"] = evidence.to_dict()
            ctx.meta["web_evidence_context"] = dict(web_evidence_context)
            ctx.meta["web_prompt_items"] = list(prompt_items)
            ctx.meta["web_key_facts"] = list(evidence.key_facts or [])
            ctx.meta["web_trust_hints"] = list(evidence.trust_hints or [])
            ctx.meta["web_freshness_summary"] = str(evidence.freshness_summary or "")
            ctx.meta["web_memory_candidates"] = list(bridge.memory_candidates)
            ctx.meta["web_memory_write_plan"] = list(planned_memory_writes)
            ctx.meta["web_search_execution"] = {
                "queries_used": list(executed.queries_used),
                "scout_queries_used": list(getattr(executed, "scout_queries_used", []) or []),
                "focused_queries_used": list(getattr(executed, "focused_queries_used", []) or []),
                "retries_used": int(getattr(executed, "retries_used", 0) or 0),
                "cooldown_applied": bool(getattr(executed, "cooldown_applied", False)),
                "max_queries": int(decision.budget.max_queries),
                "max_sources": int(decision.budget.max_sources),
                "max_fetches": int(_budget_fetches(decision)),
            }
            ctx.meta["web_guardrail_local_reply"] = bool(
                fresh_missing and compat_intent in {"fx_rate", "weather"}
            )
        if isinstance(getattr(ctx, "state", None), dict):
            ctx.state["web_last_mode"] = str(decision.mode.value)
            ctx.state["web_last_query"] = str(query)
            ctx.state["web_last_used"] = bool(web_used)
            ctx.state["web_evidence_context"] = dict(web_evidence_context)

        ctx.logs.append(
            "stage=web_retrieve done "
            f"trace={trace} mode={decision.mode.value} fetched={fetched} web_used={str(web_used).lower()} "
            f"fresh_missing={str(fresh_missing).lower()}"
        )
        _emit_trace_event(
            ctx,
            "web_retrieve_done",
            {
                "mode": str(decision.mode.value),
                "fetched": int(fetched),
                "web_used": bool(web_used),
                "fresh_missing": bool(fresh_missing),
                "citations": list(_as_dict(getattr(ctx, "meta", {})).get("web_citations") or []),
            },
        )
        return ctx


def _adaptive_citations(
    citations: list[str],
    classification,
    *,
    max_fact: int,
    max_compare: int,
) -> list[str]:
    rows = [str(x or "").strip() for x in list(citations or []) if str(x or "").strip()]
    if not rows:
        return []
    if str(getattr(classification, "query_type", "") or "") == "external_factual":
        return rows[: max(1, int(max_fact))]
    return rows[: max(1, int(max_compare))]


def _resolve_web_mode(meta: dict[str, Any], state: dict[str, Any]) -> str:
    raw = _pick(meta.get("web_mode"), state.get("web_mode"), "auto").lower()
    if raw in {"on", "off", "auto"}:
        return raw
    return "auto"


def _strip_web_prefix(text: str) -> str:
    src = str(text or "").strip()
    low = src.lower()
    for token in ("/web", "/no-web"):
        if low == token:
            return ""
        if low.startswith(token + " "):
            return src[len(token) :].strip()
    return src


def _resolve_user_override(*, text: str, meta: dict[str, Any], state: dict[str, Any]) -> str:
    from_context = _normalize_override_token(_pick(meta.get("web_override"), state.get("web_override"), ""))
    if from_context:
        return from_context
    low = str(text or "").strip().lower()
    if low == "/web" or low.startswith("/web "):
        return "web"
    if low == "/no-web" or low.startswith("/no-web "):
        return "no-web"
    return ""


def _normalize_override_token(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw in {"web", "/web", "on", "force_web", "force-web"}:
        return "web"
    if raw in {"no-web", "/no-web", "off", "no_web", "disable_web", "disable-web"}:
        return "no-web"
    return ""


def _compat_query_intent(query: str) -> str:
    low = str(query or "").strip().lower()
    if any(token in low for token in ("usd", "uah", "eur", "forex", "курс", "exchange rate")):
        return "fx_rate"
    if any(token in low for token in ("weather", "forecast", "погод", "температур")):
        return "weather"
    if any(token in low for token in ("latest", "release", "version", "news", "релиз", "версия", "новост")):
        return "news_release"
    return "generic"


def _budget_fetches(decision) -> int:
    budget = getattr(decision, "budget", None)
    if budget is None:
        return 0
    if hasattr(budget, "effective_max_fetches"):
        try:
            return max(0, int(budget.effective_max_fetches()))
        except Exception:
            pass
    try:
        return max(0, int(getattr(budget, "max_fetches")))
    except Exception:
        pass
    try:
        return max(0, int(getattr(budget, "max_pages")))
    except Exception:
        return 0


def _build_web_evidence_context(
    *,
    evidence,
    prompt_items: list[dict[str, Any]],
    mode: str,
    query: str,
) -> dict[str, Any]:
    compact_citations = [str(x or "").strip() for x in list(getattr(evidence, "compact_citations", []) or []) if str(x or "").strip()]
    key_facts = [str(x or "").strip() for x in list(getattr(evidence, "key_facts", []) or []) if str(x or "").strip()]
    trust_hints = [str(x or "").strip() for x in list(getattr(evidence, "trust_hints", []) or []) if str(x or "").strip()]
    conflict_notes = [str(x or "").strip() for x in list(getattr(evidence, "conflict_notes", []) or []) if str(x or "").strip()]
    summary = str(getattr(evidence, "summary", "") or "").strip()
    freshness_summary = str(getattr(evidence, "freshness_summary", "") or "").strip()
    conflicting_sources = bool(getattr(evidence, "conflicting_sources", False))

    source_rows: list[dict[str, Any]] = []
    for row in list(prompt_items or []):
        item = _as_dict(row)
        source_rows.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("source_url") or "").strip(),
                "domain": str(item.get("source_domain") or "").strip().lower(),
                "published_at": str(item.get("published_date") or "").strip(),
                "fetched_at": str(item.get("fetched_at") or "").strip(),
                "trust_tier": str(item.get("trust_tier") or "").strip(),
                "clean_method": str(item.get("clean_method") or "").strip(),
            }
        )

    return {
        "mode": str(mode or "").strip(),
        "query": str(query or "").strip(),
        "summary": summary,
        "freshness_summary": freshness_summary,
        "key_facts": key_facts,
        "compact_citations": compact_citations,
        "trust_hints": trust_hints,
        "conflicting_sources": conflicting_sources,
        "conflict_notes": conflict_notes,
        "sources": source_rows,
        "prompt_block": _render_web_evidence_block(
            mode=str(mode or "").strip(),
            summary=summary,
            freshness_summary=freshness_summary,
            key_facts=key_facts,
            compact_citations=compact_citations,
            trust_hints=trust_hints,
            conflicting_sources=conflicting_sources,
            conflict_notes=conflict_notes,
            sources=source_rows,
        ),
    }


def _render_web_evidence_block(
    *,
    mode: str,
    summary: str,
    freshness_summary: str,
    key_facts: list[str],
    compact_citations: list[str],
    trust_hints: list[str],
    conflicting_sources: bool,
    conflict_notes: list[str],
    sources: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    if mode:
        lines.append(f"- mode: {mode}")
    if summary:
        lines.append(f"- summary: {summary}")
    if freshness_summary:
        lines.append(f"- freshness: {freshness_summary}")

    if key_facts:
        lines.append("- key_facts:")
        for fact in list(key_facts)[:8]:
            lines.append(f"  - {fact}")

    if compact_citations:
        lines.append("- citations:")
        for citation in list(compact_citations)[:4]:
            lines.append(f"  - {citation}")

    if trust_hints:
        lines.append("- trust_hints:")
        for hint in list(trust_hints)[:4]:
            lines.append(f"  - {hint}")

    lines.append(f"- source_conflicts: {'yes' if conflicting_sources else 'no'}")
    if conflicting_sources and conflict_notes:
        lines.append("- conflict_notes:")
        for note in list(conflict_notes)[:4]:
            lines.append(f"  - {note}")

    if sources:
        lines.append("- sources:")
        for row in list(sources)[:5]:
            domain = str(_as_dict(row).get("domain") or "").strip() or "unknown"
            url = str(_as_dict(row).get("url") or "").strip()
            published = str(_as_dict(row).get("published_at") or "").strip() or "-"
            fetched = str(_as_dict(row).get("fetched_at") or "").strip() or "-"
            lines.append(f"  - {domain} | published={published} | fetched={fetched} | {url}")

    if not lines:
        return ""
    return "[WEB_EVIDENCE]\n" + "\n".join(lines).strip()


def _set_web_flags(
    ctx,
    *,
    web_query_intent: str,
    web_search_mode: str,
    fresh_required: bool,
    fresh_missing: bool,
    web_used: bool,
) -> None:
    tags = _as_dict(getattr(ctx, "tags", {}))
    tags["web_query_intent"] = str(web_query_intent or "generic")
    tags["web_search_mode"] = str(web_search_mode or WebSearchMode.NO_SEARCH.value)
    tags["web_fresh_required"] = "true" if bool(fresh_required) else "false"
    tags["web_fresh_missing"] = "true" if bool(fresh_missing) else "false"
    tags["web_used"] = "true" if bool(web_used) else "false"
    tags["web_response_style"] = "factual_direct" if web_query_intent in {"fx_rate", "weather", "news_release"} else "default"
    setattr(ctx, "tags", tags)

    if isinstance(getattr(ctx, "meta", None), dict):
        ctx.meta["web_query_intent"] = str(web_query_intent or "generic")
        ctx.meta["web_search_mode"] = str(web_search_mode or WebSearchMode.NO_SEARCH.value)
        ctx.meta["web_fresh_required"] = bool(fresh_required)
        ctx.meta["web_fresh_missing"] = bool(fresh_missing)
        ctx.meta["web_used"] = bool(web_used)
        ctx.meta["web_response_style"] = str(tags.get("web_response_style") or "default")


def _emit_trace_event(ctx, event: str, payload: dict[str, Any]) -> None:
    emitter = _as_dict(getattr(ctx, "meta", {})).get("emit_web_trace_event")
    if callable(emitter):
        try:
            emitter(str(event or "").strip(), dict(payload or {}))
            return
        except Exception:
            pass
    trace = str(_as_dict(getattr(ctx, "meta", {})).get("web_trace_id") or "").strip() or "-"
    log_json(LOGGER, str(event or "").strip(), trace=trace, **dict(payload or {}))


def _pick(*values) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _snippet500(value: str) -> str:
    return str(value or "")[:500]


def _sha1_text(value: str) -> str:
    src = str(value or "")
    if not src:
        return ""
    return hashlib.sha1(src.encode("utf-8", errors="ignore")).hexdigest()


def _clip(text: str, max_chars: int) -> str:
    src = str(text or "").strip()
    n = max(16, int(max_chars or 120))
    if len(src) <= n:
        return src
    return src[: n - 3].rstrip() + "..."


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    return [value]


def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _is_web_item(item: dict[str, Any]) -> bool:
    source = str(item.get("source") or "").strip().lower()
    topic = str(item.get("topic") or "").strip().lower()
    text = str(item.get("text") or "").strip().lower()
    return bool(
        source in {"web", "search", "internet"}
        or topic.startswith("web:")
        or text.startswith("[web]")
        or text.startswith("[web_search]")
        or "source_url:" in text
    )


def _top_domains(results: list[SearchResult], *, limit: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in list(results or []):
        domain = str(getattr(row, "source", "") or "").strip().lower()
        if not domain or domain in seen:
            continue
        seen.add(domain)
        out.append(domain)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _is_smalltalk_query(query: str, *, ctx_tags: dict[str, Any]) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    intent = str(ctx_tags.get("intent") or "").strip().lower()
    if intent in {"chat", "smalltalk", "chatting"} and _SMALLTALK_RE.search(text):
        return True
    return False
