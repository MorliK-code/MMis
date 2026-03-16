from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import load_config
from modules.internet.search import SearchClient, SearchResult
from modules.internet.scraper import WebScraper
from modules.internet.web.confidence import assess_confidence
from modules.internet.web.continuity import ContinuityConfig, build_continuity_patch, resolve_continuation
from modules.internet.web.domain_reputation import DomainReputationStore, DomainTrustPolicy
from modules.internet.web.freshness import assess_freshness
from modules.internet.web.numeric_facts import infer_numeric_profile, requires_strict_numeric_evidence
from modules.internet.web.query_classifier import classify_query
from modules.internet.web.query_planner import build_query_plan
from modules.internet.web.query_text import normalize_search_text, strip_service_command_prefix
from modules.internet.web.result_processor import build_evidence_pack
from modules.internet.web.search_manager import SearchExecutionResult, SearchManager
from modules.internet.web.source_ranker import build_source_audit_entry, rank_sources
from modules.internet.web.web_memory_bridge import WebMemoryBridge
from modules.internet.web.web_models import WebPolicyDecision, WebSearchMode
from modules.internet.web.web_policy import WebPolicyEngine, config_from_dict, default_policy_config
from modules.nlu.intent_resolver import IntentResolver
from modules.nlu.segment_classifier import SegmentClassifier
from modules.nlu.segmenter import SemanticSegmenter
from utils.logger import append_human_log, get_logger, log_json


LOGGER = get_logger(__name__)
_GEO_PATTERN = re.compile(
    r"(?:\b(?:in|at|for)\s+|(?:\bв\b|\bво\b|\bпо\b|\bдля\b)\s+)([A-Za-zА-Яа-яЁёІіЇїЄєҐґ\-]{2,48})",
    flags=re.I,
)
_LOCATION_FACT_RE = re.compile(r"location_place\s*[:=]\s*([A-Za-zА-Яа-яЁёІіЇїЄєҐґ\- ]{2,64})", flags=re.I)
_UA_GEO_TOKENS = (
    "ukraine",
    "украин",
    "україн",
    "kyiv",
    "kiev",
    "киев",
    "київ",
    "lviv",
    "львов",
    "львів",
    "odessa",
    "odesa",
    "одесс",
    "одес",
    "kharkiv",
    "харьков",
    "харків",
    "dnipro",
    "днепр",
    "дніпро",
    "ua",
)
_UA_PREFERRED_DOMAINS = {
    "finance": [
        "bank.gov.ua",
        "minfin.com.ua",
        "finance.ua",
    ],
    "weather": [
        "sinoptik.ua",
        "meteo.ua",
    ],
    "news": [
        "ukrinform.ua",
    ],
    "generic": [],
}
_NON_GEO_HINTS = {
    "internet",
    "интернет",
    "интернете",
    "сеть",
    "сети",
    "web",
    "веб",
    "site",
    "сайт",
    "source",
    "источник",
    "documentation",
    "документация",
}
_SMALLTALK_RE = re.compile(r"\b(hello|hi|how are you|привет|как дела|что нового)\b", flags=re.I)

_INTENT_ALIGNMENT_TARGET_CATEGORIES = {
    "fx_rate": {"finance", "price"},
    "weather": {"weather"},
    "news_release": {"news", "version", "docs"},
}
_GENERIC_TOP_LEVEL_INTENTS = {
    "",
    "chat",
    "question",
    "clarification",
    "smalltalk",
    "general",
    "conversation",
    "search_query",
    "finance_query",
    "weather_query",
    "news_query",
}
_PROTECTED_TOP_LEVEL_INTENTS = {
    "task",
    "bug_report",
    "planning",
    "code_review",
    "coding_question",
    "command",
    "system_event",
}


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
            search_policy=_as_dict(web_v2_cfg.get("search_policy")),
        )

        self._trust_policy = DomainTrustPolicy.from_config(web_v2_cfg)
        self._preferred_domains = list(self._trust_policy.discovery_domains(query_category="generic"))
        self._blocked_domains = list(self._trust_policy.blocked_domains)
        citations_cfg = _as_dict(web_v2_cfg.get("citations"))
        self._citation_max_fact = max(1, _to_int(citations_cfg.get("max_items_fact"), 1))
        self._citation_max_compare = max(1, _to_int(citations_cfg.get("max_items_compare"), 3))
        reputation_cfg = _as_dict(web_v2_cfg.get("domain_reputation") or web_v2_cfg.get("reputation"))
        reputation_path = _pick(
            reputation_cfg.get("storage_path"),
            web_v2_cfg.get("domain_reputation_file"),
            str(Path(app.memory_dir).expanduser().resolve() / "web_domain_reputation.json"),
        )
        self._domain_reputation = DomainReputationStore(
            storage_path=str(reputation_path),
            config=reputation_cfg,
        )
        self._nlu_segmenter = SemanticSegmenter()
        self._nlu_segment_classifier = SegmentClassifier()
        self._nlu_intent_resolver = IntentResolver()
        continuation_cfg = _as_dict(web_v2_cfg.get("continuation"))
        self._continuity_cfg = ContinuityConfig(
            ttl_minutes=max(1, _to_int(continuation_cfg.get("ttl_minutes"), 20)),
            max_user_turns=max(2, _to_int(continuation_cfg.get("max_user_turns"), 6)),
            short_followup_max_tokens=max(3, _to_int(continuation_cfg.get("short_followup_max_tokens"), 9)),
        )
        quality_cfg = _as_dict(web_v2_cfg.get("evidence_quality"))
        self._quality_min_score = max(0.0, min(1.0, _to_float(quality_cfg.get("min_score"), 0.46)))
        self._quality_min_usable = max(1, _to_int(quality_cfg.get("min_usable_results"), 2))
        self._quality_min_domains = max(1, _to_int(quality_cfg.get("min_unique_domains"), 2))
        self._quality_min_trusted = max(0, _to_int(quality_cfg.get("min_trusted_count"), 1))

    def run(self, ctx):
        trace = _trace_token(ctx)
        mode = _resolve_web_mode(_as_dict(ctx.meta), _as_dict(ctx.state))
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_mode"] = mode

        text = str(getattr(ctx, "clean_user_msg", "") or getattr(ctx, "user_msg", "") or "").strip()
        preview = _clip(text.replace("\n", " "), 120)
        ctx.logs.append(f"stage=web_retrieve start trace={trace} mode={mode} text_len={len(text)} text_preview={preview}")
        log_json(LOGGER, "web_retrieve_start", context=_stage_log_context(ctx), trace_id=trace, mode=mode, text_len=len(text), text_preview=preview)
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
            log_json(LOGGER, "web_retrieve_skip", context=_stage_log_context(ctx), trace_id=trace, reason="empty_text")
            _emit_web_summary(
                ctx,
                summary="used=false reason=empty_text",
                payload=_build_skip_web_trace_summary(query="", mode=str(mode or ""), reason="empty_text"),
            )
            _emit_trace_event(ctx, "web_retrieve_skip", {"reason": "empty_text"})
            return ctx

        if str(getattr(ctx, "route", "") or "").strip().lower() == "command":
            _set_web_flags(
                ctx,
                web_query_intent="generic",
                web_search_mode=WebSearchMode.NO_SEARCH.value,
                fresh_required=False,
                fresh_missing=False,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve skipped(command_route) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", context=_stage_log_context(ctx), trace_id=trace, reason="command_route")
            _emit_web_summary(
                ctx,
                summary="used=false reason=command_route",
                payload=_build_skip_web_trace_summary(query="", mode=str(mode or ""), reason="command_route"),
            )
            _emit_trace_event(ctx, "web_retrieve_skip", {"reason": "command_route"})
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
            log_json(LOGGER, "web_retrieve_skip", context=_stage_log_context(ctx), trace_id=trace, reason="empty_query")
            _emit_web_summary(
                ctx,
                summary="used=false reason=empty_query",
                payload=_build_skip_web_trace_summary(query="", mode=str(mode or ""), reason="empty_query"),
            )
            _emit_trace_event(ctx, "web_retrieve_skip", {"reason": "empty_query"})
            return ctx

        continuation = resolve_continuation(
            query=query,
            state=_as_dict(getattr(ctx, "state", {})),
            config=self._continuity_cfg,
        )
        effective_query = str(continuation.resolved_query or query).strip() or query
        continuity_base_query = str(continuation.base_query or query).strip() or query
        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["continuation_ref"] = str(continuation.continuation_ref or "")
            ctx.meta["context_confidence"] = float(continuation.context_confidence or 0.0)
            ctx.meta["query_effective"] = str(effective_query)
            ctx.meta["query_base"] = str(continuity_base_query)
        _emit_trace_event(
            ctx,
            "web_context_link",
            {
                "used": bool(continuation.used),
                "continuation_ref": str(continuation.continuation_ref or ""),
                "context_confidence": float(continuation.context_confidence or 0.0),
                "reason": str(continuation.reason or ""),
                "base_query": str(continuity_base_query),
                "original_query": str(query),
                "effective_query": str(effective_query),
            },
        )

        ctx_tags = _as_dict(getattr(ctx, "tags", {}))
        classification = classify_query(
            effective_query,
            metadata_tags=_as_list(ctx_tags.get("metadata_tags")),
            original_text=query,
        )
        nlu_intents = _extract_nlu_intents_from_query(
            query=effective_query,
            segmenter=self._nlu_segmenter,
            segment_classifier=self._nlu_segment_classifier,
            intent_resolver=self._nlu_intent_resolver,
        )
        if _is_smalltalk_query(effective_query, ctx_tags=ctx_tags):
            nlu_intents = _merge_lower_tokens(nlu_intents, ["smalltalk"])
        _emit_trace_event(ctx, "web_nlu_signals", {"nlu_intents": list(nlu_intents)})
        if (
            user_override != "web"
            and _is_smalltalk_query(effective_query, ctx_tags=ctx_tags)
            and not classification.requires_freshness
            and not classification.is_external_fact_question
            and not _has_external_nlu_intent(nlu_intents)
        ):
            _set_web_flags(
                ctx,
                web_query_intent="generic",
                web_search_mode=WebSearchMode.NO_SEARCH.value,
                fresh_required=False,
                fresh_missing=False,
                web_used=False,
            )
            ctx.logs.append(f"stage=web_retrieve skipped(smalltalk_hard_skip) trace={trace}")
            log_json(LOGGER, "web_retrieve_skip", context=_stage_log_context(ctx), trace_id=trace, reason="smalltalk_hard_skip")
            _emit_web_summary(
                ctx,
                summary="used=false reason=smalltalk_hard_skip",
                payload=_build_skip_web_trace_summary(query=str(effective_query), mode=str(mode or ""), reason="smalltalk_hard_skip"),
            )
            _emit_trace_event(ctx, "web_retrieve_skip", {"reason": "smalltalk_hard_skip"})
            return ctx

        confidence = assess_confidence(
            query=effective_query,
            classification=classification,
            retrieved_memories=_as_list(getattr(ctx, "retrieved_memories", [])),
            memory_context=_as_dict(getattr(ctx, "memory_context", {})),
        )
        web_items = [x for x in _as_list(getattr(ctx, "retrieved_memories", [])) if _is_web_item(_as_dict(x))]
        freshness = assess_freshness(
            query=effective_query,
            classification=classification,
            web_items=web_items,
            ttl_days=_as_dict(_as_dict(getattr(self._app, "web_v2", {})).get("ttl_days") or _as_dict(getattr(self._app, "web_v2", {})).get("ttl")),
        )
        compat_intent = _compat_query_intent(effective_query, classification=classification)

        _emit_trace_event(ctx, "web_classification", classification.to_dict())
        _emit_trace_event(ctx, "web_confidence_assessment", confidence.to_dict())
        _emit_trace_event(ctx, "web_freshness_assessment", freshness.to_dict())
        policy_context = _as_dict(getattr(ctx, "meta", {}))
        if nlu_intents:
            policy_context["nlu_intents"] = _merge_lower_tokens(policy_context.get("nlu_intents"), nlu_intents)

        decision = self._policy.decide(
            query=effective_query,
            classification=classification,
            confidence=confidence,
            freshness=freshness,
            web_mode=mode,
            internet_enabled=bool(self._app.internet_enabled),
            user_override=user_override,
            policy_context=policy_context,
        )
        decision_payload = decision.to_dict()
        decision_payload["classification"] = classification.to_dict()
        decision_payload["confidence"] = confidence.to_dict()
        decision_payload["freshness"] = freshness.to_dict()
        decision_payload["user_override"] = str(user_override or "")
        _emit_trace_event(ctx, "web_policy_decision", decision_payload)
        intent_alignment = _resolve_intent_alignment(
            original_intent=str(_as_dict(getattr(ctx, "tags", {})).get("intent") or ""),
            web_intent=compat_intent,
            classification=classification,
            decision=decision,
        )
        _apply_intent_alignment(ctx, intent_alignment)
        if str(intent_alignment.get("web_intent") or "").strip():
            _emit_trace_event(ctx, "web_intent_alignment", dict(intent_alignment))

        geo_decision = _resolve_geo_hint(
            original_query=query,
            effective_query=effective_query,
            state=_as_dict(getattr(ctx, "state", {})),
            memory_context=_as_dict(getattr(ctx, "memory_context", {})),
            retrieved_memories=_as_list(getattr(ctx, "retrieved_memories", [])),
            classification=classification,
        )
        geo_hint = str(geo_decision.get("value") or "").strip()
        policy_category = str(getattr(classification, "primary_category", "") or "").strip().lower()
        category_domains = self._trust_policy.discovery_domains(query_category=policy_category)
        geo_domains = _geo_preferred_domains(geo_hint, query_category=policy_category)
        preferred_domains = _merge_domains(category_domains, geo_domains)
        effective_trust_policy = self._trust_policy.with_runtime(
            preferred_domains_by_category={policy_category or "generic": geo_domains}
        )
        _emit_trace_event(ctx, "web_geo_hint", dict(geo_decision))

        if isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta["web_query"] = str(query)
            ctx.meta["web_query_effective"] = str(effective_query)
            ctx.meta["web_user_override"] = str(user_override or "")
            ctx.meta["web_decision_breakdown"] = dict(decision.decision_breakdown or {})
            ctx.meta["web_query_classification"] = classification.to_dict()
            ctx.meta["web_confidence_assessment"] = confidence.to_dict()
            ctx.meta["web_freshness_assessment"] = freshness.to_dict()
            ctx.meta["web_need_score"] = float(decision.web_need_score)
            ctx.meta["web_nlu_intents"] = list(nlu_intents)
            ctx.meta["web_trust_policy"] = effective_trust_policy.to_dict()
            ctx.meta["web_geo_hint_debug"] = dict(geo_decision)
            ctx.meta["web_intent_alignment"] = dict(intent_alignment)
            ctx.meta["web_correction_challenge"] = bool(getattr(classification, "is_correction_challenge", False))
            if geo_hint:
                ctx.meta["geo_hint"] = str(geo_hint)

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
            context=_stage_log_context(ctx),
            trace_id=trace,
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
                context=_stage_log_context(ctx),
                trace_id=trace,
                reason="policy_no_search",
                mode=str(decision.mode.value),
                score=float(decision.web_need_score),
            )
            _emit_web_summary(
                ctx,
                summary=(
                    f"used=false reason=policy_no_search mode={str(decision.mode.value)} "
                    f"score={float(decision.web_need_score):.3f}"
                ),
                payload=_build_skip_web_trace_summary(
                    query=str(effective_query),
                    mode=str(decision.mode.value),
                    reason="policy_no_search",
                    classification=classification,
                    decision_score=float(decision.web_need_score),
                    compat_intent=compat_intent,
                    intent_alignment=intent_alignment,
                ),
            )
            _emit_trace_event(
                ctx,
                "web_retrieve_skip",
                {"reason": "policy_no_search", "mode": str(decision.mode.value), "score": float(decision.web_need_score)},
            )
            continuation_patch = build_continuity_patch(
                state=_as_dict(getattr(ctx, "state", {})),
                query=effective_query,
                resolved_intent=compat_intent,
                resolved_concepts=list(nlu_intents),
                resolved_location=geo_hint,
                active_task={
                    "task_id": _task_id_for_query(continuity_base_query),
                    "query": str(continuity_base_query),
                    "base_query": str(continuity_base_query),
                    "latest_query": str(effective_query),
                    "resolved_intent": str(compat_intent),
                    "resolved_concepts": list(nlu_intents),
                    "resolved_location": str(geo_hint or ""),
                    "turn_id": int(_to_int(_as_dict(getattr(ctx, "state", {})).get("turn_id"), 0)),
                },
                continuation_ref=str(continuation.continuation_ref or ""),
                context_confidence=float(continuation.context_confidence or 0.0),
            )
            _apply_continuity_patch(ctx, continuation_patch)
            return ctx

        ctx.logs.append(
            "stage=web_retrieve search_start "
            f"trace={trace} mode={decision.mode.value} max_queries={decision.budget.max_queries} "
            f"max_sources={decision.budget.max_sources} max_fetches={_budget_fetches(decision)}"
        )

        try:
            selected = _execute_search_cycle(
                query=effective_query,
                classification=classification,
                decision=decision,
                freshness=freshness,
                preferred_domains=preferred_domains,
                trust_policy=effective_trust_policy,
                geo_hint=geo_hint,
                search_manager=self._search_manager,
                domain_reputation=self._domain_reputation,
                memory_bridge=self._memory_bridge,
                compat_intent=compat_intent,
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
            log_json(LOGGER, "web_retrieve_search_fail", context=_stage_log_context(ctx), trace_id=trace, error=type(exc).__name__)
            _emit_web_summary(
                ctx,
                summary=f"used=false reason=search_fail error={type(exc).__name__}",
                payload=_build_skip_web_trace_summary(
                    query=str(effective_query),
                    mode=str(decision.mode.value),
                    reason="search_fail",
                    classification=classification,
                    decision_score=float(decision.web_need_score),
                    error=str(type(exc).__name__),
                    compat_intent=compat_intent,
                    intent_alignment=intent_alignment,
                ),
            )
            _emit_trace_event(ctx, "web_search_fail", {"error": type(exc).__name__, "mode": str(decision.mode.value)})
            continuation_patch = build_continuity_patch(
                state=_as_dict(getattr(ctx, "state", {})),
                query=effective_query,
                resolved_intent=compat_intent,
                resolved_concepts=list(nlu_intents),
                resolved_location=geo_hint,
                active_task={
                    "task_id": _task_id_for_query(continuity_base_query),
                    "query": str(continuity_base_query),
                    "base_query": str(continuity_base_query),
                    "latest_query": str(effective_query),
                    "resolved_intent": str(compat_intent),
                    "resolved_concepts": list(nlu_intents),
                    "resolved_location": str(geo_hint or ""),
                    "turn_id": int(_to_int(_as_dict(getattr(ctx, "state", {})).get("turn_id"), 0)),
                },
                continuation_ref=str(continuation.continuation_ref or ""),
                context_confidence=float(continuation.context_confidence or 0.0),
            )
            _apply_continuity_patch(ctx, continuation_patch)
            return ctx

        quality = _as_dict(selected.get("quality"))
        quality_score = float(quality.get("score") or 0.0)
        upgraded_decision = self._policy.upgrade_mode_once(decision)
        escalated = False
        if (
            upgraded_decision.mode != decision.mode
            and quality_score < self._quality_min_score
            and bool(decision.should_search)
        ):
            base_mode = str(decision.mode.value)
            base_score = float(quality_score)
            try:
                escalated_cycle = _execute_search_cycle(
                    query=effective_query,
                    classification=classification,
                    decision=upgraded_decision,
                    freshness=freshness,
                    preferred_domains=preferred_domains,
                    trust_policy=effective_trust_policy,
                    geo_hint=geo_hint,
                    search_manager=self._search_manager,
                    domain_reputation=self._domain_reputation,
                    memory_bridge=self._memory_bridge,
                    compat_intent=compat_intent,
                )
                escalated_quality = _as_dict(escalated_cycle.get("quality"))
                escalated_score = float(escalated_quality.get("score") or 0.0)
                escalated = True
                if escalated_score >= quality_score:
                    selected = escalated_cycle
                    decision = upgraded_decision
                    quality = escalated_quality
                    quality_score = escalated_score
                _emit_trace_event(
                    ctx,
                    "web_search_escalation",
                    {
                        "attempted": True,
                        "from_mode": base_mode,
                        "to_mode": str(upgraded_decision.mode.value),
                        "quality_before": round(float(base_score), 4),
                        "quality_after": round(float(escalated_score), 4),
                    },
                )
            except Exception as exc:
                _emit_trace_event(
                    ctx,
                    "web_search_escalation",
                    {
                        "attempted": True,
                        "from_mode": base_mode,
                        "to_mode": str(upgraded_decision.mode.value),
                        "error": type(exc).__name__,
                    },
                )

        plan = selected["plan"]
        executed = selected["executed"]
        ranked = selected["ranked"]
        domains = selected["domains"]
        domain_reputation = selected["domain_reputation"]
        evidence = selected["evidence"]
        bridge = selected["bridge"]
        prompt_items = selected["prompt_items"]
        web_evidence_context = selected["web_evidence_context"]
        source_audit = list(selected.get("source_audit") or [])
        trace_query = _trace_query_text(query=effective_query, plan=plan)

        if isinstance(getattr(ctx, "meta", None), dict):
            plan_debug = _as_dict(getattr(plan, "debug", {}) or {})
            query_debug = _as_dict(plan_debug.get("query_text"))
            search_debug = _as_dict(getattr(executed, "search_debug", {}) or {})
            search_context = _as_dict(getattr(executed, "search_context", {}) or {})
            ctx.meta["web_result_count"] = int(len(ranked))
            ctx.meta["web_result_domains"] = list(domains)
            ctx.meta["web_domain_reputation"] = dict(domain_reputation)
            ctx.meta["web_evidence_quality"] = dict(quality)
            ctx.meta["web_source_audit"] = list(source_audit)
            ctx.meta["web_trace_query"] = str(trace_query)
            ctx.meta["web_search_core"] = str(query_debug.get("extracted_search_core") or query_debug.get("search_core") or "")
            ctx.meta["web_removed_wrapper_text"] = str(query_debug.get("removed_wrapper_text") or "")
            ctx.meta["web_planner_category"] = str(plan_debug.get("strategy") or "")
            ctx.meta["web_search_debug"] = dict(search_debug)
            ctx.meta["web_search_context"] = dict(search_context)

        plan_payload = plan.to_dict()
        if geo_hint:
            plan_payload["geo_hint"] = str(geo_hint)
        _emit_trace_event(ctx, "web_query_plan", plan_payload)
        query_debug = _as_dict(_as_dict(getattr(plan, "debug", {}) or {}).get("query_text"))
        _emit_trace_event(
            ctx,
            "web_search_request",
            {
                "query": str(trace_query),
                "effective_query": str(effective_query),
                "search_core": str(query_debug.get("extracted_search_core") or query_debug.get("search_core") or ""),
                "removed_wrapper_text": str(query_debug.get("removed_wrapper_text") or ""),
                "mode": str(decision.mode.value),
                "queries": plan.all_queries(),
                "query_roles": dict(plan.query_roles or {}),
                "max_queries": int(decision.budget.max_queries),
                "max_sources": int(decision.budget.max_sources),
                "max_fetches": int(_budget_fetches(decision)),
            },
        )
        ctx.logs.append(
            "stage=web_retrieve search_done "
            f"trace={trace} results={len(ranked)} domains={','.join(domains) if domains else '-'} "
            f"quality={quality_score:.3f} escalated={int(bool(escalated))} "
            f"search_success={int(bool(_as_dict(getattr(executed, 'search_debug', {}) or {}).get('effective_success', bool(getattr(executed, 'results', [])))))} "
            f"reason={str(_as_dict(getattr(executed, 'search_debug', {}) or {}).get('effective_success_reason') or '-')} "
            f"confidence={float(_to_float(_as_dict(getattr(executed, 'search_debug', {}) or {}).get('effective_search_confidence'), 0.0) or 0.0):.3f} "
            f"region_bias={str(_as_dict(getattr(executed, 'search_debug', {}) or {}).get('region_bias') or '-')}"
        )
        _emit_trace_event(
            ctx,
            "web_search_results",
            {
                "query": str(trace_query),
                "effective_query": str(effective_query),
                "mode": str(decision.mode.value),
                "queries_used": list(executed.queries_used),
                "scout_queries_used": list(getattr(executed, "scout_queries_used", []) or []),
                "focused_queries_used": list(getattr(executed, "focused_queries_used", []) or []),
                "query_runs": list(getattr(executed, "query_runs", []) or [])[:8],
                "retries_used": int(getattr(executed, "retries_used", 0) or 0),
                "cooldown_applied": bool(getattr(executed, "cooldown_applied", False)),
                "result_count": len(ranked),
                "raw_result_count": int(_to_int(_as_dict(getattr(executed, "search_debug", {}) or {}).get("raw_result_count"), len(list(getattr(executed, "results", []) or []))) or 0),
                "raw_results_count": int(_to_int(_as_dict(getattr(executed, "search_debug", {}) or {}).get("raw_results_count"), len(list(getattr(executed, "results", []) or []))) or len(list(getattr(executed, "results", []) or []))),
                "reported_result_count": _to_int(_as_dict(getattr(executed, "search_debug", {}) or {}).get("reported_result_count"), None),
                "reported_number_of_results": _to_int(_as_dict(getattr(executed, "search_debug", {}) or {}).get("reported_number_of_results"), None),
                "usable_results_count": int(_to_int(_as_dict(getattr(executed, "search_debug", {}) or {}).get("usable_results_count"), len(list(getattr(executed, "results", []) or []))) or len(list(getattr(executed, "results", []) or []))),
                "effective_search_success": bool(_as_dict(getattr(executed, "search_debug", {}) or {}).get("effective_success", bool(getattr(executed, "results", [])))),
                "effective_search_success_reason": str(_as_dict(getattr(executed, "search_debug", {}) or {}).get("effective_success_reason") or ("results_present" if getattr(executed, "results", []) else "empty_results")),
                "effective_search_confidence": float(_to_float(_as_dict(getattr(executed, "search_debug", {}) or {}).get("effective_search_confidence"), 0.0) or 0.0),
                "engine_success_count": int(_to_int(_as_dict(getattr(executed, "search_debug", {}) or {}).get("engine_success_count"), 0) or 0),
                "engine_failure_count": int(_to_int(_as_dict(getattr(executed, "search_debug", {}) or {}).get("engine_failure_count"), 0) or 0),
                "engine_health": dict(_as_dict(getattr(executed, "search_debug", {}) or {}).get("engine_health") or {}),
                "query_locale": str(_as_dict(getattr(executed, "search_debug", {}) or {}).get("query_locale") or ""),
                "region_bias": str(_as_dict(getattr(executed, "search_debug", {}) or {}).get("region_bias") or ""),
                "region_bias_reasons": [str(x or "").strip() for x in list(_as_dict(getattr(executed, "search_debug", {}) or {}).get("region_bias_reasons") or []) if str(x or "").strip()][:4],
                "search_debug": dict(getattr(executed, "search_debug", {}) or {}),
                "domains": list(domains),
                "geo_hint": str(geo_hint or ""),
                "domain_reputation": dict(domain_reputation),
                "trust_policy": effective_trust_policy.to_dict(),
                "fetch_summary": dict(getattr(executed, "fetch_summary", {}) or {}),
                "warnings": list(getattr(executed, "warnings", []) or []),
                "source_audit_summary": {
                    "total": int(len(source_audit)),
                    "selected": int(sum(1 for row in source_audit if bool(_as_dict(row).get("selected_for_evidence")))),
                    "filtered": int(sum(1 for row in source_audit if str(_as_dict(row).get("filtered_out_reason") or "").strip())),
                    "blocked": int(sum(1 for row in source_audit if bool(_as_dict(row).get("blocked_hit")))),
                },
                "top_results": [
                    {
                        "rank": idx + 1,
                        "title": str(row.item.title or ""),
                        "url": str(row.item.url or ""),
                        "domain": str(row.item.source or ""),
                        "trust_tier": str(row.trust_tier or ""),
                        "policy_state": str(_as_dict(row.item.raw).get("v2_policy_state") or ""),
                        "manual_override": str(_as_dict(row.item.raw).get("v2_manual_override") or ""),
                        "published_date": str(row.item.published_date or ""),
                        "score_total": float(row.item.score or 0.0),
                        "score_breakdown": dict(row.item.score_breakdown or {}),
                        "reputation_score": float(_as_dict(row.item.raw).get("v2_reputation_score", 0.0) or 0.0),
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
            "web_search_summary",
            {
                "query": str(trace_query),
                "category": str(getattr(classification, "primary_category", "") or ""),
                "planner_category": str(_as_dict(getattr(plan, "debug", {}) or {}).get("strategy") or ""),
                "search_core": str(query_debug.get("extracted_search_core") or query_debug.get("search_core") or ""),
                "region_bias": str(_as_dict(getattr(executed, "search_debug", {}) or {}).get("region_bias") or ""),
                "query_locale": str(_as_dict(getattr(executed, "search_debug", {}) or {}).get("query_locale") or ""),
                "engine_health": dict(_as_dict(getattr(executed, "search_debug", {}) or {}).get("engine_health") or {}),
                "engines_failed": list(_as_dict(getattr(executed, "search_debug", {}) or {}).get("engines_failed") or [])[:6],
                "raw_results_count": int(_to_int(_as_dict(getattr(executed, "search_debug", {}) or {}).get("raw_result_count"), 0) or 0),
                "usable_results_count": int(_to_int(_as_dict(getattr(executed, "search_debug", {}) or {}).get("usable_results_count"), len(list(getattr(executed, "results", []) or []))) or len(list(getattr(executed, "results", []) or []))),
                "effective_search_success": bool(_as_dict(getattr(executed, "search_debug", {}) or {}).get("effective_success", bool(getattr(executed, "results", [])))),
                "effective_search_confidence": float(_to_float(_as_dict(getattr(executed, "search_debug", {}) or {}).get("effective_search_confidence"), 0.0) or 0.0),
                "top_domains": [str(x or "").strip().lower() for x in list(_as_dict(getattr(executed, "search_debug", {}) or {}).get("top_domains") or domains) if str(x or "").strip()][:6],
                "warnings": list(getattr(executed, "warnings", []) or [])[:8],
            },
        )
        _emit_trace_event(
            ctx,
            "web_source_audit",
            {
                "query": str(trace_query),
                "effective_query": str(effective_query),
                "mode": str(decision.mode.value),
                "sources": list(source_audit)[:12],
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
        _emit_trace_event(
            ctx,
            "web_evidence_quality",
            dict(quality),
        )
        _emit_trace_event(
            ctx,
            "web_evidence_pack",
            {
                "items": int(len(list(evidence.items or []))),
                "source_audit": int(len(list(source_audit or []))),
                "citations": int(len(list(evidence.compact_citations or []))),
                "key_facts": int(len(list(evidence.key_facts or []))),
                "conflicting_sources": bool(evidence.conflicting_sources),
                "conflict_notes": list(evidence.conflict_notes or []),
                "numeric_candidates_selected": int(quality.get("numeric_candidates_selected") or 0),
                "numeric_candidates_rejected": int(quality.get("numeric_candidates_rejected") or 0),
                "conflict_severity": float(_to_float(quality.get("conflict_severity"), 0.0) or 0.0),
                "evidence_strength": float(_to_float(quality.get("evidence_strength"), 0.0) or 0.0),
                "final_factual_confidence": float(_to_float(quality.get("final_factual_confidence"), 0.0) or 0.0),
                "factual_basis": dict(quality.get("factual_basis") or {}),
                "selection_summary": dict(getattr(evidence, "selection_summary", {}) or {}),
            },
        )
        reputation_update = self._domain_reputation.learn_from_evidence(list(evidence.items or []))
        _emit_trace_event(
            ctx,
            "web_domain_reputation_update",
            dict(reputation_update or {}),
        )

        clarify_needed = _needs_clarifying_question(
            quality=quality,
            classification=classification,
            min_score=self._quality_min_score,
            min_usable=self._quality_min_usable,
            min_domains=self._quality_min_domains,
            min_trusted=self._quality_min_trusted,
        )
        if clarify_needed and isinstance(getattr(ctx, "meta", None), dict):
            question = _build_clarifying_question(
                original_query=query,
                effective_query=effective_query,
                compat_intent=compat_intent,
                geo_hint=geo_hint,
            )
            ctx.meta["web_clarifying_question"] = question
            ctx.meta["web_clarify_reason"] = "low_evidence_quality"
            ctx.meta["web_skip_citations"] = True
            _emit_trace_event(
                ctx,
                "web_clarify_required",
                {
                    "reason": "low_evidence_quality",
                    "question": str(question),
                    "quality": dict(quality),
                },
            )
        elif isinstance(getattr(ctx, "meta", None), dict):
            ctx.meta.pop("web_clarifying_question", None)
            ctx.meta.pop("web_clarify_reason", None)
            ctx.meta.pop("web_skip_citations", None)

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
                        "trace_id": _stage_log_context(ctx).get("trace_id"),
                        "request_id": _stage_log_context(ctx).get("request_id"),
                        "turn_id": _stage_log_context(ctx).get("turn_id"),
                        "conversation_id": _stage_log_context(ctx).get("conversation_id"),
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
            ctx.meta["web_synthesis_caution"] = {
                "cautious": bool(quality.get("cautious_synthesis")),
                "selected_avg_quality": float(_to_float(quality.get("selected_avg_quality"), 0.0) or 0.0),
                "topical_filtered_sources": int(quality.get("topical_filtered_sources") or 0),
                "has_conflict": bool(quality.get("has_conflict")),
                "conflict_severity": float(_to_float(quality.get("conflict_severity"), 0.0) or 0.0),
                "evidence_strength": float(_to_float(quality.get("evidence_strength"), 0.0) or 0.0),
                "final_factual_confidence": float(_to_float(quality.get("final_factual_confidence"), 0.0) or 0.0),
            }
            ctx.meta["web_prompt_items"] = list(prompt_items)
            ctx.meta["web_key_facts"] = list(evidence.key_facts or [])
            ctx.meta["web_trust_hints"] = list(evidence.trust_hints or [])
            ctx.meta["web_freshness_summary"] = str(evidence.freshness_summary or "")
            ctx.meta["web_memory_candidates"] = list(bridge.memory_candidates)
            ctx.meta["web_memory_write_plan"] = list(planned_memory_writes)
            ctx.meta["web_domain_reputation_update"] = dict(reputation_update or {})
            ctx.meta["web_search_execution"] = {
                "queries_used": list(executed.queries_used),
                "scout_queries_used": list(getattr(executed, "scout_queries_used", []) or []),
                "focused_queries_used": list(getattr(executed, "focused_queries_used", []) or []),
                "query_runs": list(getattr(executed, "query_runs", []) or []),
                "retries_used": int(getattr(executed, "retries_used", 0) or 0),
                "cooldown_applied": bool(getattr(executed, "cooldown_applied", False)),
                "max_queries": int(decision.budget.max_queries),
                "max_sources": int(decision.budget.max_sources),
                "max_fetches": int(_budget_fetches(decision)),
                "fetch_summary": dict(getattr(executed, "fetch_summary", {}) or {}),
                "warnings": list(getattr(executed, "warnings", []) or []),
            }
            ctx.meta["web_guardrail_local_reply"] = bool(
                fresh_missing and compat_intent in {"fx_rate", "weather"}
            )
        if isinstance(getattr(ctx, "state", None), dict):
            ctx.state["web_last_mode"] = str(decision.mode.value)
            ctx.state["web_last_query"] = str(effective_query)
            ctx.state["web_last_query_raw"] = str(query)
            ctx.state["web_last_used"] = bool(web_used)
            ctx.state["web_evidence_context"] = dict(web_evidence_context)

        continuation_patch = build_continuity_patch(
            state=_as_dict(getattr(ctx, "state", {})),
            query=effective_query,
            resolved_intent=compat_intent,
            resolved_concepts=list(nlu_intents),
            resolved_location=geo_hint,
            active_task={
                "task_id": _task_id_for_query(continuity_base_query),
                "query": str(continuity_base_query),
                "base_query": str(continuity_base_query),
                "latest_query": str(effective_query),
                "resolved_intent": str(compat_intent),
                "resolved_concepts": list(nlu_intents),
                "resolved_location": str(geo_hint or ""),
                "turn_id": int(_to_int(_as_dict(getattr(ctx, "state", {})).get("turn_id"), 0)),
                "mode": str(decision.mode.value),
                "web_used": bool(web_used),
                "quality_score": float(quality_score),
            },
            continuation_ref=str(continuation.continuation_ref or ""),
            context_confidence=float(continuation.context_confidence or 0.0),
        )
        _apply_continuity_patch(ctx, continuation_patch)

        compact_summary = _build_compact_web_trace_summary(
            query=str(trace_query),
            effective_query=str(effective_query),
            plan=plan,
            decision=decision,
            classification=classification,
            freshness=freshness,
            executed=executed,
            evidence=evidence,
            quality=quality,
            source_audit=source_audit,
            geo_hint=str(geo_hint or ""),
            clarify_needed=bool(clarify_needed),
            fresh_missing=bool(fresh_missing),
            web_used=bool(web_used),
            compat_intent=compat_intent,
            intent_alignment=intent_alignment,
        )
        _emit_web_summary(
            ctx,
            summary=(
                f"used={str(web_used).lower()} mode={str(decision.mode.value)} "
                f"results={len(ranked)} fetched={fetched} quality={quality_score:.3f}"
            ),
            payload=compact_summary,
        )
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


def _execute_search_cycle(
    *,
    query: str,
    classification,
    decision: WebPolicyDecision,
    freshness,
    preferred_domains: list[str],
    trust_policy: DomainTrustPolicy,
    geo_hint: str,
    search_manager: SearchManager,
    domain_reputation: DomainReputationStore,
    memory_bridge: WebMemoryBridge,
    compat_intent: str,
) -> dict[str, Any]:
    plan = build_query_plan(
        query=query,
        classification=classification,
        decision=decision,
        preferred_domains=preferred_domains,
        geo_hint=geo_hint,
    )
    executed = search_manager.execute(
        plan=plan,
        decision=decision,
        classification=classification,
        freshness=freshness,
        preferred_domains=preferred_domains,
        geo_hint=geo_hint,
    )
    reputation_scores = domain_reputation.snapshot_scores()
    reputation_stats = domain_reputation.snapshot_stats()
    pre_rank_audit = [
        build_source_audit_entry(
            item=row,
            preferred_domains=preferred_domains,
            trust_policy=trust_policy,
            reputation_scores=reputation_scores,
            reputation_stats=reputation_stats,
            query_category=str(getattr(classification, "primary_category", "") or ""),
            query_text=str(query),
            geo_hint=geo_hint,
        )
        for row in list(executed.results or [])
    ]
    ranked = rank_sources(
        items=list(executed.results),
        preferred_domains=preferred_domains,
        trust_policy=trust_policy,
        reputation_scores=reputation_scores,
        reputation_stats=reputation_stats,
        query_intent=compat_intent,
        query_category=str(getattr(classification, "primary_category", "") or ""),
        query_text=str(query),
        geo_hint=geo_hint,
        limit_hint=int(getattr(decision.budget, "max_sources", 0) or 0),
    )
    domains = _top_domains([x.item for x in ranked], limit=5)
    domain_reputation_map = {
        str(domain): float(reputation_scores.get(str(domain).strip().lower(), 0.0))
        for domain in list(domains or [])
    }
    evidence = build_evidence_pack(
        ranked_results=ranked,
        fetched_pages=executed.fetched_pages,
        decision=decision,
        classification=classification,
        query_text=str(query),
    )
    source_audit = _merge_source_audit(pre_rank_audit=pre_rank_audit, pack_audit=list(getattr(evidence, "source_audit", []) or []))
    bridge = memory_bridge.build(
        evidence_pack=evidence,
        decision=decision,
        classification=classification,
    )
    prompt_items = list(bridge.prompt_items or [])
    quality = _assess_evidence_quality(
        ranked=ranked,
        evidence=evidence,
        freshness=freshness,
        classification=classification,
        source_audit=source_audit,
        query_text=str(query),
    )
    web_evidence_context = _build_web_evidence_context(
        evidence=evidence,
        prompt_items=prompt_items,
        mode=str(decision.mode.value),
        query=str(query),
        source_audit=source_audit,
        quality=quality,
    )
    return {
        "plan": plan,
        "executed": executed,
        "ranked": ranked,
        "domains": domains,
        "domain_reputation": domain_reputation_map,
        "evidence": evidence,
        "bridge": bridge,
        "prompt_items": prompt_items,
        "web_evidence_context": web_evidence_context,
        "source_audit": source_audit,
        "quality": quality,
    }


def _assess_evidence_quality(
    *,
    ranked: list[Any],
    evidence,
    freshness,
    classification,
    source_audit: list[dict[str, Any]] | None = None,
    query_text: str = "",
) -> dict[str, Any]:
    items = list(getattr(evidence, "items", []) or [])
    audits = [dict(x or {}) for x in list(source_audit or []) if isinstance(x, dict)]
    selected_audits = [audit for audit in audits if bool(audit.get("selected_for_evidence"))]
    selection_summary = dict(getattr(evidence, "selection_summary", {}) or {})
    query_category = str(getattr(classification, "primary_category", "") or "")
    numeric_profile = infer_numeric_profile(
        query_category=query_category,
        query_text=query_text,
    )
    strict_numeric = requires_strict_numeric_evidence(
        query_category=query_category,
        query_text=query_text,
    )
    numeric_selected = int(
        sum(len(list(audit.get("numeric_candidates_selected") or [])) for audit in selected_audits)
    )
    numeric_rejected = int(
        sum(len(list(audit.get("numeric_candidates_rejected") or [])) for audit in selected_audits)
    )
    usable_results = int(len(items))
    unique_domains = int(len({str(getattr(item, "domain", "") or "").strip().lower() for item in items if str(getattr(item, "domain", "") or "").strip()}))
    trusted_count = int(
        sum(
            1
            for item in items
            if str(getattr(item, "trust_tier", "") or "").strip().lower()
            not in {"general_web", "forum_discussion", "blog_random", "unknown", "degraded_source", "policy_risky", "policy_degraded"}
        )
    )
    selected_avg_quality = float(
        sum(float(audit.get("quality_score") or 0.0) for audit in selected_audits) / max(1, len(selected_audits))
        if selected_audits
        else 0.0
    )
    weak_selected = int(sum(1 for audit in selected_audits if float(audit.get("quality_score") or 0.0) < 0.28))
    topical_filtered = int(
        sum(
            1
            for audit in audits
            if str(audit.get("filtered_out_reason") or "").strip().lower() in {"topical_mismatch", "unsupported_source_type", "missing_relevant_numeric"}
        )
    )
    conflict_severity = _conflict_severity(evidence=evidence, source_audit=audits)
    conflict_notes = [str(x or "").strip() for x in list(getattr(evidence, "conflict_notes", []) or []) if str(x or "").strip()]
    rejected_type_mismatch = bool(
        any(
            str(audit.get("filtered_out_reason") or "").strip().lower() in {"rate_type_mismatch", "historical_rate_mismatch"}
            for audit in selected_audits + audits
        )
    )
    type_mismatch_only = bool(
        any(note.startswith("rate_type_mismatch:") for note in conflict_notes)
        and not bool(getattr(evidence, "conflicting_sources", False))
    )
    if rejected_type_mismatch and not bool(getattr(evidence, "conflicting_sources", False)):
        type_mismatch_only = True
    consensus_score = 0.0
    if usable_results > 0:
        consensus_score += 0.25
    if unique_domains >= 2:
        consensus_score += 0.20
    if trusted_count >= 1:
        consensus_score += 0.20
    if selected_avg_quality >= 0.48:
        consensus_score += 0.20
    elif selected_avg_quality >= 0.34:
        consensus_score += 0.10
    if not bool(getattr(evidence, "conflicting_sources", False)):
        consensus_score += 0.15
    if weak_selected > 0 and selected_avg_quality < 0.30:
        consensus_score -= 0.10
    if topical_filtered > 0 and usable_results <= 1:
        consensus_score -= 0.08
    if strict_numeric and numeric_selected > 0:
        consensus_score += min(0.20, 0.08 + (0.04 * numeric_selected))
    elif strict_numeric:
        consensus_score -= 0.20
    if conflict_severity >= 0.85:
        consensus_score -= 0.32
    elif conflict_severity >= 0.55:
        consensus_score -= 0.18
    freshness_ok = bool(
        (not bool(getattr(classification, "requires_freshness", False)))
        or usable_results > 0
    )
    if not freshness_ok:
        consensus_score *= 0.6
    score = max(0.0, min(1.0, consensus_score))
    evidence_strength = score
    if strict_numeric:
        evidence_strength = max(
            0.0,
            min(
                1.0,
                (0.26 * min(1.0, usable_results / 2.0))
                + (0.18 * min(1.0, unique_domains / 2.0))
                + (0.18 * min(1.0, trusted_count / 2.0))
                + (0.22 * selected_avg_quality)
                + (0.16 * min(1.0, numeric_selected / 2.0))
                - (0.24 * conflict_severity),
            ),
        )
    final_factual_confidence = evidence_strength
    if strict_numeric:
        if conflict_severity >= 0.85:
            final_factual_confidence *= 0.28
        elif conflict_severity >= 0.55:
            final_factual_confidence *= 0.52
        elif conflict_severity > 0.0:
            final_factual_confidence *= 0.74
        if type_mismatch_only:
            final_factual_confidence *= 0.82
        if numeric_selected == 0:
            final_factual_confidence *= 0.40
        if selected_avg_quality < 0.34:
            final_factual_confidence *= 0.72
    final_factual_confidence = max(0.0, min(1.0, final_factual_confidence))
    cautious_synthesis = bool(
        bool(getattr(evidence, "conflicting_sources", False))
        or usable_results == 0
        or selected_avg_quality < 0.32
        or weak_selected > 0
        or type_mismatch_only
        or (strict_numeric and (numeric_selected == 0 or conflict_severity >= 0.55 or final_factual_confidence < 0.64))
    )
    conflict_reason = ""
    if bool(getattr(evidence, "conflicting_sources", False)) and conflict_severity >= 0.55:
        conflict_reason = "true_numeric_conflict"
    elif type_mismatch_only:
        conflict_reason = "rate_type_mismatch"
    elif cautious_synthesis and strict_numeric:
        conflict_reason = "weak_numeric_evidence"
    factual_basis = dict(selection_summary.get("factual_basis") or {})
    return {
        "usable_results": int(usable_results),
        "unique_domains": int(unique_domains),
        "trusted_count": int(trusted_count),
        "consensus_score": round(float(consensus_score), 4),
        "freshness_ok": bool(freshness_ok),
        "score": round(float(score), 4),
        "has_conflict": bool(getattr(evidence, "conflicting_sources", False)),
        "selected_avg_quality": round(float(selected_avg_quality), 4),
        "weak_selected_sources": int(weak_selected),
        "topical_filtered_sources": int(topical_filtered),
        "numeric_profile": str(numeric_profile),
        "strict_numeric": bool(strict_numeric),
        "numeric_candidates_selected": int(numeric_selected),
        "numeric_candidates_rejected": int(numeric_rejected),
        "selected_rate_type": str(selection_summary.get("currency_selected_rate_type") or ""),
        "requested_rate_type": str(selection_summary.get("currency_requested_rate_type") or ""),
        "selected_page_types": [str(x or "").strip() for x in list(selection_summary.get("currency_page_types_selected") or []) if str(x or "").strip()],
        "selected_result_url": str(selection_summary.get("primary_selected_url") or ""),
        "selected_result_domain": str(selection_summary.get("primary_selected_domain") or ""),
        "selected_result_page_type": str(selection_summary.get("primary_selected_page_type") or ""),
        "selected_result_factual_page_type": str(selection_summary.get("primary_selected_factual_page_type") or ""),
        "conflict_severity": round(float(conflict_severity), 4),
        "conflict_reason": str(conflict_reason or ""),
        "true_conflict_notes": [str(x or "").strip() for x in list(selection_summary.get("true_conflict_notes") or []) if str(x or "").strip()][:6],
        "type_mismatch_notes": [str(x or "").strip() for x in list(selection_summary.get("type_mismatch_notes") or []) if str(x or "").strip()][:6],
        "evidence_strength": round(float(evidence_strength), 4),
        "final_factual_confidence": round(float(final_factual_confidence), 4),
        "cautious_synthesis": bool(cautious_synthesis),
        "temporal_risk": round(float(getattr(freshness, "temporal_risk", 0.0) or 0.0), 4),
        "factual_basis": factual_basis,
    }


def _conflict_severity(*, evidence, source_audit: list[dict[str, Any]]) -> float:
    max_audit = 0.0
    for audit in list(source_audit or []):
        try:
            max_audit = max(max_audit, float(audit.get("numeric_conflict_severity") or 0.0))
        except Exception:
            continue
    for note in list(getattr(evidence, "conflict_notes", []) or []):
        text = str(note or "").strip().lower()
        if "numeric_conflict_high" in text:
            max_audit = max(max_audit, 0.92)
        elif "numeric_conflict_medium" in text:
            max_audit = max(max_audit, 0.60)
    return max(0.0, min(1.0, max_audit))


def _merge_source_audit(
    *,
    pre_rank_audit: list[dict[str, Any]],
    pack_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}

    def _key(row: dict[str, Any]) -> str:
        item = _as_dict(row)
        url = str(item.get("url") or "").strip().lower()
        domain = str(item.get("domain") or "").strip().lower()
        title = str(item.get("title") or "").strip().lower()
        return url or f"{domain}|{title}"

    for row in list(pre_rank_audit or []):
        item = dict(row or {})
        key = _key(item)
        if not key:
            continue
        by_key[key] = item
        out.append(item)

    for row in list(pack_audit or []):
        item = dict(row or {})
        key = _key(item)
        if not key:
            continue
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = item
            out.append(item)
            continue
        merged = {**existing, **item}
        existing.clear()
        existing.update(merged)

    return out[:12]


def _needs_clarifying_question(
    *,
    quality: dict[str, Any],
    classification,
    min_score: float,
    min_usable: int,
    min_domains: int,
    min_trusted: int,
) -> bool:
    if not bool(getattr(classification, "is_external_fact_question", False)):
        return False
    score = float(quality.get("score") or 0.0)
    if score >= float(min_score):
        return False
    usable = int(quality.get("usable_results") or 0)
    domains = int(quality.get("unique_domains") or 0)
    trusted = int(quality.get("trusted_count") or 0)
    if usable >= int(min_usable) and domains >= int(min_domains) and trusted >= int(min_trusted):
        return False
    return True


def _build_clarifying_question(
    *,
    original_query: str,
    effective_query: str,
    compat_intent: str,
    geo_hint: str,
) -> str:
    loc = str(geo_hint or "").strip()
    if compat_intent == "weather":
        if loc:
            return f"Могу уточнить прогноз по {loc}. Нужны часы, 3 дня или 7 дней?"
        return "Могу уточнить прогноз, но нужна локация. Для какого города или страны?"
    if compat_intent == "fx_rate":
        if loc:
            return f"Уточню курс для {loc}. Нужна покупка/продажа и какой банк или межбанк?"
        return "Уточню курс, но нужна страна/рынок и валюта (например USD/UAH)."
    if compat_intent == "news_release":
        return "Чтобы дать точный ответ, уточни тему или источник: официальный релиз, СМИ или обзор?"
    preview = str(original_query or effective_query or "").strip()
    if len(preview) > 100:
        preview = preview[:97].rstrip() + "..."
    if preview:
        return f"Нужна небольшая конкретизация запроса: «{preview}». Что именно проверить в первую очередь?"
    return "Нужна небольшая конкретизация запроса, чтобы проверить данные по надёжным источникам."


def _apply_continuity_patch(ctx, patch: dict[str, Any]) -> None:
    if not isinstance(patch, dict) or not patch:
        return
    if isinstance(getattr(ctx, "state", None), dict):
        ctx.state.update(dict(patch))
    if isinstance(getattr(ctx, "memory_ops", None), list):
        ctx.memory_ops.append({"op": "state_patch", "value": dict(patch)})


def _task_id_for_query(query: str) -> str:
    text = str(query or "").strip().lower()
    if not text:
        return "task_empty"
    return f"task_{hashlib.sha1(text.encode('utf-8', errors='ignore')).hexdigest()[:12]}"


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
    return strip_service_command_prefix(text)


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


def _compat_query_intent(query: str, *, classification=None) -> str:
    low = str(query or "").strip().lower()
    category = str(getattr(classification, "primary_category", "") or "").strip().lower()
    if category == "finance":
        return "fx_rate"
    if any(token in low for token in ("usd", "uah", "eur", "forex", "курс", "exchange rate", "доллар", "евро", "гривн", "грн", "валют", "бакс")):
        return "fx_rate"
    if any(token in low for token in ("weather", "forecast", "погод", "температур")):
        return "weather"
    if any(token in low for token in ("latest", "release", "version", "news", "релиз", "версия", "новост")):
        return "news_release"
    return "generic"

def _resolve_intent_alignment(
    *,
    original_intent: str,
    web_intent: str,
    classification,
    decision,
) -> dict[str, Any]:
    original = str(original_intent or "").strip().lower()
    target = str(web_intent or "").strip().lower()
    category = str(getattr(classification, "primary_category", "") or "").strip().lower()
    query_type = str(getattr(classification, "query_type", "") or "").strip().lower()
    corrected = original or target
    reason = "no_web_intent_alignment"
    signal_score = 0.0
    signals: list[str] = []

    allowed_categories = set(_INTENT_ALIGNMENT_TARGET_CATEGORIES.get(target, set()))
    if target:
        signals.append(f"web_intent:{target}")
    if category and category in allowed_categories:
        signal_score += 1.2
        signals.append(f"category:{category}")
    if bool(getattr(classification, "is_external_fact_question", False)):
        signal_score += 0.9
        signals.append("external_fact")
    if query_type == "external_factual":
        signal_score += 0.9
        signals.append("query_type:external_factual")
    if bool(getattr(classification, "requires_freshness", False)):
        signal_score += 0.8
        signals.append("requires_freshness")
    if bool(getattr(classification, "explicit_search_intent", False)):
        signal_score += 0.6
        signals.append("explicit_search_intent")
    if bool(getattr(decision, "should_search", False)):
        signal_score += 0.6
        signals.append("policy_should_search")
    if float(_to_float(getattr(decision, "web_need_score", 0.0), 0.0) or 0.0) >= 0.30:
        signal_score += 0.4
        signals.append("web_need_score>=0.30")

    applied = False
    if target not in _INTENT_ALIGNMENT_TARGET_CATEGORIES:
        reason = "web_intent_not_actionable"
    elif original == target and target:
        corrected = target
        reason = "already_aligned"
    elif original in _PROTECTED_TOP_LEVEL_INTENTS:
        reason = "protected_original_intent"
    elif category not in allowed_categories and signal_score < 2.6:
        reason = "category_mismatch"
    elif original in _GENERIC_TOP_LEVEL_INTENTS and signal_score >= 2.2:
        corrected = target
        applied = True
        reason = "web_external_factual_alignment"
    else:
        reason = "insufficient_alignment_signals"

    if not corrected:
        corrected = target or original

    return {
        "original_intent": original,
        "corrected_intent": corrected,
        "final_intent": corrected,
        "web_intent": target,
        "classification_category": category,
        "query_type": query_type,
        "applied": bool(applied),
        "reason": str(reason),
        "signal_score": float(round(signal_score, 4)),
        "signals": list(signals),
    }


def _apply_intent_alignment(ctx, alignment: dict[str, Any]) -> None:
    row = dict(alignment or {})
    original_intent = str(row.get("original_intent") or "").strip().lower()
    final_intent = str(row.get("final_intent") or row.get("corrected_intent") or original_intent).strip().lower()
    applied = bool(row.get("applied"))

    tags = _as_dict(getattr(ctx, "tags", {}))
    if final_intent:
        tags["resolved_intent"] = final_intent
    if applied and final_intent:
        tags["intent"] = final_intent
    setattr(ctx, "tags", tags)

    if isinstance(getattr(ctx, "meta", None), dict):
        ctx.meta["resolved_intent"] = str(final_intent or original_intent)
        ctx.meta["intent_alignment"] = dict(row)

        summaries = dict(_as_dict(ctx.meta.get("turn_log_summaries")) or {})
        intent_summary = dict(_as_dict(summaries.get("intent_summary")) or {})
        if not intent_summary:
            intent_summary = {
                "intent": str(original_intent),
                "intent_confidence": float(_to_float(tags.get("intent_conf"), 0.0) or 0.0),
            }
        intent_summary["original_intent"] = str(original_intent)
        intent_summary["corrected_intent"] = str(final_intent or original_intent)
        intent_summary["intent_alignment_applied"] = bool(applied)
        intent_summary["intent_alignment_reason"] = str(row.get("reason") or "")
        intent_summary["web_intent"] = str(row.get("web_intent") or "")
        intent_summary["web_primary_category"] = str(row.get("classification_category") or "")
        intent_summary["signal_score"] = float(_to_float(row.get("signal_score"), 0.0) or 0.0)
        intent_summary["signals"] = list(row.get("signals") or [])
        if final_intent:
            intent_summary["intent"] = str(final_intent)
        summaries["intent_summary"] = intent_summary
        summaries["intent_alignment_summary"] = dict(row)
        ctx.meta["turn_log_summaries"] = summaries

    if applied:
        ctx.plan = _align_plan_for_intent(
            plan=str(getattr(ctx, "plan", "") or ""),
            original_intent=original_intent,
            final_intent=final_intent,
        )

    if str(row.get("web_intent") or "").strip():
        context = _stage_log_context(ctx)
        compact = (
            f"original={original_intent or '-'} corrected={final_intent or '-'} "
            f"applied={str(applied).lower()} reason={str(row.get('reason') or '-')}"
        )
        ctx.logs.append(
            "summary=intent_alignment_summary "
            f"trace={context.get('trace_id') or '-'} "
            f"request={context.get('request_id') or '-'} "
            f"turn={context.get('turn_id') or '-'} "
            f"conversation={context.get('conversation_id') or '-'} "
            f"{compact}"
        )
        log_json(LOGGER, "intent_alignment_summary", summary=compact, context=context, **row)


def _align_plan_for_intent(*, plan: str, original_intent: str, final_intent: str) -> str:
    src = str(plan or "").strip()
    target = str(final_intent or "").strip().lower()
    if target not in {"fx_rate", "weather", "news_release"}:
        return src
    if not src or "maintain conversational flow" in src or f"intent={original_intent}" in src:
        return f"intent={target}; provide concise factual answer grounded in current evidence"
    return src


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
    source_audit: list[dict[str, Any]] | None = None,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compact_citations = [str(x or "").strip() for x in list(getattr(evidence, "compact_citations", []) or []) if str(x or "").strip()]
    key_facts = [str(x or "").strip() for x in list(getattr(evidence, "key_facts", []) or []) if str(x or "").strip()]
    trust_hints = [str(x or "").strip() for x in list(getattr(evidence, "trust_hints", []) or []) if str(x or "").strip()]
    conflict_notes = [str(x or "").strip() for x in list(getattr(evidence, "conflict_notes", []) or []) if str(x or "").strip()]
    summary = str(getattr(evidence, "summary", "") or "").strip()
    freshness_summary = str(getattr(evidence, "freshness_summary", "") or "").strip()
    conflicting_sources = bool(getattr(evidence, "conflicting_sources", False))
    quality_row = _as_dict(quality)
    caution_reasons: list[str] = []
    if conflicting_sources:
        caution_reasons.append("source_conflict")
    if bool(quality_row.get("cautious_synthesis")):
        caution_reasons.append("weak_evidence")
    if float(quality_row.get("conflict_severity") or 0.0) >= 0.55:
        caution_reasons.append("numeric_conflict")
    if str(quality_row.get("conflict_reason") or "").strip() == "rate_type_mismatch":
        caution_reasons.append("rate_type_mismatch")
    if int(quality_row.get("topical_filtered_sources") or 0) > 0:
        caution_reasons.append("topical_filtering")

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
                "selected_for_evidence": True,
            }
        )
    audit_rows = [dict(x or {}) for x in list(source_audit or [])[:12]]

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
        "numeric_profile": str(quality_row.get("numeric_profile") or ""),
        "numeric_candidates_selected": int(quality_row.get("numeric_candidates_selected") or 0),
        "numeric_candidates_rejected": int(quality_row.get("numeric_candidates_rejected") or 0),
        "selected_rate_type": str(quality_row.get("selected_rate_type") or ""),
        "requested_rate_type": str(quality_row.get("requested_rate_type") or ""),
        "selected_result_url": str(quality_row.get("selected_result_url") or ""),
        "selected_result_domain": str(quality_row.get("selected_result_domain") or ""),
        "selected_result_page_type": str(quality_row.get("selected_result_page_type") or ""),
        "selected_result_factual_page_type": str(quality_row.get("selected_result_factual_page_type") or ""),
        "conflict_severity": float(_to_float(quality_row.get("conflict_severity"), 0.0) or 0.0),
        "conflict_reason": str(quality_row.get("conflict_reason") or ""),
        "true_conflict_notes": [str(x or "").strip() for x in list(quality_row.get("true_conflict_notes") or []) if str(x or "").strip()][:6],
        "type_mismatch_notes": [str(x or "").strip() for x in list(quality_row.get("type_mismatch_notes") or []) if str(x or "").strip()][:6],
        "evidence_strength": float(_to_float(quality_row.get("evidence_strength"), 0.0) or 0.0),
        "final_factual_confidence": float(_to_float(quality_row.get("final_factual_confidence"), 0.0) or 0.0),
        "cautious_synthesis": bool(quality_row.get("cautious_synthesis")),
        "caution_reasons": caution_reasons,
        "factual_basis": dict(quality_row.get("factual_basis") or {}),
        "sources": source_rows,
        "source_audit": audit_rows,
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
            numeric_profile=str(quality_row.get("numeric_profile") or ""),
            numeric_candidates_selected=int(quality_row.get("numeric_candidates_selected") or 0),
            selected_rate_type=str(quality_row.get("selected_rate_type") or ""),
            requested_rate_type=str(quality_row.get("requested_rate_type") or ""),
            selected_result_page_type=str(quality_row.get("selected_result_page_type") or ""),
            conflict_severity=float(_to_float(quality_row.get("conflict_severity"), 0.0) or 0.0),
            conflict_reason=str(quality_row.get("conflict_reason") or ""),
            final_factual_confidence=float(_to_float(quality_row.get("final_factual_confidence"), 0.0) or 0.0),
            cautious_synthesis=bool(quality_row.get("cautious_synthesis")),
            caution_reasons=caution_reasons,
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
    numeric_profile: str,
    numeric_candidates_selected: int,
    selected_rate_type: str,
    requested_rate_type: str,
    selected_result_page_type: str,
    conflict_severity: float,
    conflict_reason: str,
    final_factual_confidence: float,
    cautious_synthesis: bool,
    caution_reasons: list[str],
) -> str:
    lines: list[str] = []
    status_block = (
        "[WEB_TOOL_STATUS]\n"
        "- live_web_lookup: already_executed_for_this_turn\n"
        "- response_rule: do not say that you cannot browse/check the internet or access live data for this turn.\n"
        "- response_rule: answer from the WEB_EVIDENCE facts below."
    )
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
    if numeric_profile:
        lines.append(f"- numeric_profile: {numeric_profile}")
    if numeric_candidates_selected > 0:
        lines.append(f"- numeric_candidates_selected: {int(numeric_candidates_selected)}")
    if requested_rate_type:
        lines.append(f"- requested_rate_type: {requested_rate_type}")
    if selected_rate_type:
        lines.append(f"- selected_rate_type: {selected_rate_type}")
    if selected_result_page_type and selected_result_page_type != selected_rate_type:
        lines.append(f"- selected_page_type: {selected_result_page_type}")
    if conflict_severity > 0.0:
        lines.append(f"- conflict_severity: {float(conflict_severity):.3f}")
    if conflict_reason:
        lines.append(f"- conflict_reason: {conflict_reason}")
    if final_factual_confidence > 0.0:
        lines.append(f"- final_factual_confidence: {float(final_factual_confidence):.3f}")
    if conflicting_sources and conflict_notes:
        lines.append("- conflict_notes:")
        for note in list(conflict_notes)[:4]:
            lines.append(f"  - {note}")
    if any(str(note or "").startswith("rate_type_mismatch:") for note in list(conflict_notes or [])):
        lines.append("- synthesis_rule: do not mix cash, NBU, bank, index or historical rates as if they were the same value.")
    if cautious_synthesis:
        lines.append("- synthesis_guidance: cautious")
        lines.append("- synthesis_rule: if evidence is weak or conflicting, present uncertainty and avoid overconfident exact claims.")
        lines.append("- synthesis_rule: for numeric/date answers, do not assert an exact value unless WEB_EVIDENCE supports it consistently.")
        if caution_reasons:
            lines.append("- caution_reasons:")
            for reason in list(caution_reasons)[:4]:
                lines.append(f"  - {reason}")

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
    return "\n".join([status_block, "[WEB_EVIDENCE]\n" + "\n".join(lines).strip()]).strip()


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
    trace = _trace_token(ctx)
    log_json(LOGGER, str(event or "").strip(), trace=trace, **dict(payload or {}))


def _stage_log_context(ctx) -> dict[str, Any]:
    meta = _as_dict(getattr(ctx, "meta", {}))
    state = _as_dict(getattr(ctx, "state", {}))
    return {
        "trace_id": str(_pick(meta.get("trace_id"), state.get("conversation_id"), "-")).strip() or "-",
        "request_id": str(_pick(meta.get("request_id"), meta.get("trace_id"), "")).strip(),
        "turn_id": str(_pick(meta.get("turn_id"), state.get("turn_id"), "")).strip(),
        "conversation_id": str(_pick(meta.get("conversation_id"), state.get("conversation_id"), "")).strip(),
    }


def _emit_web_summary(ctx, *, summary: str, payload: dict[str, Any]) -> None:
    compact = str(summary or "").strip()
    context = _stage_log_context(ctx)
    if isinstance(getattr(ctx, "logs", None), list):
        ctx.logs.append(
            "summary=web_summary "
            f"trace={context.get('trace_id') or '-'} "
            f"request={context.get('request_id') or '-'} "
            f"turn={context.get('turn_id') or '-'} "
            f"conversation={context.get('conversation_id') or '-'} "
            f"{compact}"
        )
    if isinstance(getattr(ctx, "meta", None), dict):
        ctx.meta["web_summary"] = dict(payload or {})
        ctx.meta["web_trace_compact_summary"] = dict(payload or {})
        turn_summaries = dict(_as_dict(ctx.meta.get("turn_log_summaries")) or {})
        turn_summaries["web_summary"] = dict(payload or {})
        ctx.meta["turn_log_summaries"] = turn_summaries
    log_json(LOGGER, "web_summary", summary=compact, context=context, **dict(payload or {}))
    append_human_log("WEB", context=context, lines=_human_web_summary_lines(payload))
    _emit_trace_event(ctx, "web_summary", dict(payload or {}))


def _human_web_summary_lines(payload: dict[str, Any]) -> list[str]:
    row = _as_dict(payload)
    policy = _as_dict(row.get("policy"))
    sources = _as_dict(row.get("sources"))
    fetch = _as_dict(sources.get("fetch"))
    evidence = _as_dict(row.get("evidence"))
    issues = ", ".join(_issue_list(_as_list(row.get("issues")))) or "none"
    warnings = ", ".join(_issue_list(_as_list(row.get("warnings")))) or "none"
    return [
        f"query: {_clip(str(row.get('query') or ''), 220) or '-'}",
        (
            f"policy: used={'yes' if bool(row.get('web_used')) else 'no'} "
            f"mode={str(row.get('mode') or '-')} reason={str(policy.get('reason') or '-')} "
            f"status={str(row.get('status') or 'ok')}"
        ),
        (
            f"sources: scanned={int(sources.get('scanned') or 0)} "
            f"selected={int(sources.get('selected') or 0)} "
            f"trusted={int(sources.get('trusted_hits') or 0)} "
            f"preferred={int(sources.get('preferred_hits') or 0)} "
            f"risky={int(sources.get('risky_hits') or 0)} "
            f"blocked={int(sources.get('blocked_hits') or 0)} "
            f"fetch={int(fetch.get('attempted') or 0)}/{int(fetch.get('fetched') or 0)}"
        ),
        (
            f"evidence: count={int(evidence.get('count') or 0)} "
            f"quality={float(_to_float(evidence.get('quality_score'), 0.0) or 0.0):.3f} "
            f"conflicts={'yes' if bool(evidence.get('conflicting_sources')) else 'no'}"
        ),
        f"issues: {issues}",
        f"warnings: {warnings}",
    ]


def _compact_query_roles(query_roles: dict[str, Any], *, limit_per_role: int = 2) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, values in dict(query_roles or {}).items():
        role = str(key or "").strip()
        if not role:
            continue
        out[role] = [str(x or "").strip() for x in list(values or []) if str(x or "").strip()][: max(1, int(limit_per_role))]
    return out


def _issue_list(items: list[str]) -> list[str]:
    out: list[str] = []
    for row in list(items or []):
        text = str(row or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _compact_query_runs(rows: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list(rows or [])[: max(1, int(limit))]:
        item = _as_dict(row)
        out.append(
            {
                "query": str(item.get("query") or "").strip(),
                "stage": str(item.get("stage") or "").strip(),
                "attempts": int(_to_int(item.get("attempts"), 0)),
                "result_count": int(_to_int(item.get("result_count"), 0)),
                "raw_result_count": int(_to_int(item.get("raw_result_count"), 0)),
                "reported_result_count": _to_int(item.get("reported_result_count"), None),
                "usable_results_count": int(_to_int(item.get("usable_results_count"), 0)),
                "effective_success": bool(item.get("effective_success")),
                "effective_search_confidence": float(_to_float(item.get("effective_search_confidence"), 0.0) or 0.0),
                "success_reason": str(item.get("success_reason") or ""),
                "failed": bool(item.get("failed")),
                "engine_success_count": int(_to_int(item.get("engine_success_count"), 0)),
                "engine_failure_count": int(_to_int(item.get("engine_failure_count"), 0)),
                "query_locale": str(item.get("query_locale") or "").strip(),
                "region_bias": str(item.get("region_bias") or "").strip(),
                "top_domains": [str(x or "").strip().lower() for x in list(item.get("top_domains") or []) if str(x or "").strip()][:4],
            }
        )
    return out


def _build_compact_web_trace_summary(
    *,
    query: str,
    effective_query: str,
    plan,
    decision,
    classification,
    freshness,
    executed,
    evidence,
    quality: dict[str, Any],
    source_audit: list[dict[str, Any]],
    geo_hint: str,
    clarify_needed: bool,
    fresh_missing: bool,
    web_used: bool,
    compat_intent: str,
    intent_alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan_debug = _as_dict(getattr(plan, "debug", {}) or {})
    query_debug = _as_dict(plan_debug.get("query_text"))
    selection_summary = _as_dict(getattr(evidence, "selection_summary", {}) or {})
    fetch_summary = _as_dict(getattr(executed, "fetch_summary", {}) or {})
    search_debug = _as_dict(getattr(executed, "search_debug", {}) or {})
    query_runs = [dict(x or {}) for x in list(getattr(executed, "query_runs", []) or []) if isinstance(x, dict)]
    blocked = int(sum(1 for row in list(source_audit or []) if bool(_as_dict(row).get("blocked_hit"))))
    risky = int(sum(1 for row in list(source_audit or []) if bool(_as_dict(row).get("risky_hit"))))
    preferred = int(sum(1 for row in list(source_audit or []) if bool(_as_dict(row).get("preferred_hit"))))
    trusted = int(sum(1 for row in list(source_audit or []) if bool(_as_dict(row).get("trusted_hit"))))

    issues: list[str] = []
    warnings: list[str] = []
    if bool(clarify_needed):
        issues.append("low_evidence_quality")
    if bool(fresh_missing):
        issues.append("fresh_data_missing")
    if bool(getattr(evidence, "conflicting_sources", False)):
        issues.append("source_conflict")
    if bool(quality.get("cautious_synthesis")):
        warnings.append("cautious_synthesis")
    if bool(getattr(executed, "cooldown_applied", False)):
        warnings.append("cooldown_applied")
    warnings.extend([str(x or "").strip() for x in list(getattr(executed, "warnings", []) or []) if str(x or "").strip()])
    warnings.extend([str(x or "").strip() for x in list(getattr(evidence, "conflict_notes", []) or []) if str(x or "").strip()])

    return {
        "status": "ok",
        "query": str(query or "").strip(),
        "effective_query": str(effective_query or "").strip(),
        "search_core": str(query_debug.get("extracted_search_core") or query_debug.get("search_core") or "").strip(),
        "geo_hint": str(geo_hint or "").strip(),
        "resolved_intent": str(compat_intent or "").strip(),
        "intent_alignment": {
            "original_intent": str(_as_dict(intent_alignment).get("original_intent") or "").strip(),
            "corrected_intent": str(_as_dict(intent_alignment).get("corrected_intent") or "").strip(),
            "applied": bool(_as_dict(intent_alignment).get("applied")),
            "reason": str(_as_dict(intent_alignment).get("reason") or "").strip(),
        },
        "web_used": bool(web_used),
        "mode": str(getattr(decision.mode, "value", decision.mode) or "").strip(),
        "result_count": int(len(list(getattr(executed, "results", []) or []))),
        "fetched": int(fetch_summary.get("attempted") or 0),
        "quality_score": float(_to_float(quality.get("score"), 0.0) or 0.0),
        "search": {
            "raw_result_count": int(_to_int(search_debug.get("raw_result_count"), 0) or 0),
            "raw_results_count": int(_to_int(search_debug.get("raw_results_count"), 0) or 0),
            "reported_result_count": _to_int(search_debug.get("reported_result_count"), None),
            "reported_number_of_results": _to_int(search_debug.get("reported_number_of_results"), None),
            "usable_results_count": int(_to_int(search_debug.get("usable_results_count"), len(list(getattr(executed, "results", []) or []))) or len(list(getattr(executed, "results", []) or []))),
            "engine_success_count": int(_to_int(search_debug.get("engine_success_count"), 0) or 0),
            "engine_failure_count": int(_to_int(search_debug.get("engine_failure_count"), 0) or 0),
            "engine_health": dict(search_debug.get("engine_health") or {}),
            "effective_success": bool(search_debug.get("effective_success", bool(getattr(executed, "results", [])))),
            "effective_success_reason": str(search_debug.get("effective_success_reason") or ("results_present" if getattr(executed, "results", []) else "empty_results")),
            "effective_search_confidence": float(_to_float(search_debug.get("effective_search_confidence"), 0.0) or 0.0),
            "query_locale": str(search_debug.get("query_locale") or ""),
            "region_bias": str(search_debug.get("region_bias") or ""),
            "region_bias_reasons": [str(x or "").strip() for x in list(search_debug.get("region_bias_reasons") or []) if str(x or "").strip()][:4],
        },
        "policy": {
            "mode": str(getattr(decision.mode, "value", decision.mode) or "").strip(),
            "reason": str(getattr(decision, "reason", "") or "").strip(),
            "should_search": bool(getattr(decision, "should_search", False)),
            "decision_score": float(_to_float(getattr(decision, "web_need_score", 0.0), 0.0)),
            "query_type": str(getattr(classification, "query_type", "") or "").strip(),
            "primary_category": str(getattr(classification, "primary_category", "") or "").strip(),
            "planner_category": str(plan_debug.get("strategy") or "").strip(),
            "requires_freshness": bool(getattr(classification, "requires_freshness", False)),
            "fresh_missing": bool(fresh_missing),
            "clarify_needed": bool(clarify_needed),
        },
        "queries": {
            "planned_total": int(len(list(plan.all_queries() or []))),
            "used_total": int(len(list(getattr(executed, "queries_used", []) or []))),
            "used": [str(x or "").strip() for x in list(getattr(executed, "queries_used", []) or []) if str(x or "").strip()][:8],
            "roles": _compact_query_roles(getattr(plan, "query_roles", {}) or {}),
            "runs": _compact_query_runs(query_runs),
        },
        "budget": {
            "limit": {
                "max_queries": int(getattr(getattr(decision, "budget", None), "max_queries", 0) or 0),
                "max_sources": int(getattr(getattr(decision, "budget", None), "max_sources", 0) or 0),
                "max_fetches": int(_budget_fetches(decision)),
            },
            "used": {
                "queries": int(len(list(getattr(executed, "queries_used", []) or []))),
                "sources": int(selection_summary.get("sources_scanned") or 0),
                "fetches": int(fetch_summary.get("attempted") or len(list(getattr(executed, "fetched_pages", {}) or {}))),
            },
            "cooldown_applied": bool(getattr(executed, "cooldown_applied", False)),
            "retries_used": int(getattr(executed, "retries_used", 0) or 0),
        },
        "sources": {
            "scanned": int(selection_summary.get("sources_scanned") or 0),
            "unique_domains_scanned": int(selection_summary.get("unique_domains_scanned") or 0),
            "domains_scanned": list(selection_summary.get("domains_scanned") or [])[:8],
            "selected": int(selection_summary.get("selected_sources") or 0),
            "selected_domains": list(selection_summary.get("selected_domains") or [])[:8],
            "filtered": int(selection_summary.get("filtered_sources") or 0),
            "filtered_reasons": dict(selection_summary.get("filtered_reasons") or {}),
            "trusted_hits": int(trusted),
            "preferred_hits": int(preferred),
            "risky_hits": int(risky),
            "blocked_hits": int(blocked),
            "fetch": {
                "attempted": int(fetch_summary.get("attempted") or 0),
                "fetched": int(fetch_summary.get("fetched") or 0),
                "fallbacks": int(fetch_summary.get("snippet_fallback_count") or 0),
                "failed": int(fetch_summary.get("failed_fetches") or 0),
            },
        },
        "evidence": {
            "count": int(len(list(getattr(evidence, "items", []) or []))),
            "citations": int(len(list(getattr(evidence, "compact_citations", []) or []))),
            "key_facts": int(len(list(getattr(evidence, "key_facts", []) or []))),
            "summary": str(getattr(evidence, "summary", "") or "").strip(),
            "freshness_summary": str(getattr(evidence, "freshness_summary", "") or "").strip(),
            "conflicting_sources": bool(getattr(evidence, "conflicting_sources", False)),
            "quality_score": float(_to_float(quality.get("score"), 0.0) or 0.0),
            "selected_avg_quality": float(_to_float(quality.get("selected_avg_quality"), 0.0) or 0.0),
            "numeric_candidates_selected": int(quality.get("numeric_candidates_selected") or 0),
            "numeric_candidates_rejected": int(quality.get("numeric_candidates_rejected") or 0),
            "selected_rate_type": str(quality.get("selected_rate_type") or ""),
            "selected_result_page_type": str(quality.get("selected_result_page_type") or ""),
            "conflict_severity": float(_to_float(quality.get("conflict_severity"), 0.0) or 0.0),
            "conflict_reason": str(quality.get("conflict_reason") or ""),
            "evidence_strength": float(_to_float(quality.get("evidence_strength"), 0.0) or 0.0),
            "final_factual_confidence": float(_to_float(quality.get("final_factual_confidence"), 0.0) or 0.0),
            "cautious_synthesis": bool(quality.get("cautious_synthesis")),
        },
        "issues": _issue_list(issues),
        "warnings": _issue_list(warnings),
    }


def _build_skip_web_trace_summary(
    *,
    query: str,
    mode: str,
    reason: str,
    classification=None,
    decision_score: float = 0.0,
    error: str = "",
    compat_intent: str = "",
    intent_alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_query = str(query or "").strip()
    display_query = _trace_query_text(query=effective_query)
    issues = [str(reason or "").strip()] if str(reason or "").strip() else []
    if str(error or "").strip():
        issues.append(str(error or "").strip())
    return {
        "status": "skipped" if not error else "failed",
        "query": display_query,
        "effective_query": effective_query,
        "resolved_intent": str(compat_intent or "").strip(),
        "intent_alignment": {
            "original_intent": str(_as_dict(intent_alignment).get("original_intent") or "").strip(),
            "corrected_intent": str(_as_dict(intent_alignment).get("corrected_intent") or "").strip(),
            "applied": bool(_as_dict(intent_alignment).get("applied")),
            "reason": str(_as_dict(intent_alignment).get("reason") or "").strip(),
        },
        "web_used": False,
        "mode": str(mode or "").strip(),
        "result_count": 0,
        "fetched": 0,
        "quality_score": 0.0,
        "policy": {
            "mode": str(mode or "").strip(),
            "reason": str(reason or "").strip(),
            "should_search": False if not error else True,
            "decision_score": float(max(0.0, min(1.0, decision_score))),
            "query_type": str(getattr(classification, "query_type", "") or "").strip() if classification is not None else "",
            "primary_category": str(getattr(classification, "primary_category", "") or "").strip() if classification is not None else "",
        },
        "queries": {"planned_total": 0, "used_total": 0, "used": [], "roles": {}, "runs": []},
        "budget": {"limit": {}, "used": {"queries": 0, "sources": 0, "fetches": 0}, "cooldown_applied": False, "retries_used": 0},
        "sources": {
            "scanned": 0,
            "unique_domains_scanned": 0,
            "domains_scanned": [],
            "selected": 0,
            "selected_domains": [],
            "filtered": 0,
            "filtered_reasons": {},
            "trusted_hits": 0,
            "preferred_hits": 0,
            "risky_hits": 0,
            "blocked_hits": 0,
            "fetch": {"attempted": 0, "fetched": 0, "fallbacks": 0, "failed": 0},
        },
        "evidence": {
            "count": 0,
            "citations": 0,
            "key_facts": 0,
            "summary": "",
            "freshness_summary": "",
            "conflicting_sources": False,
            "quality_score": 0.0,
        },
        "issues": _issue_list(issues),
        "warnings": [],
    }


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


def _trace_token(ctx) -> str:
    meta = _as_dict(getattr(ctx, "meta", {}))
    state = _as_dict(getattr(ctx, "state", {}))
    return str(_pick(meta.get("trace_id"), meta.get("conversation_id"), state.get("conversation_id"), "-")).strip() or "-"


def _trace_query_text(*, query: str, plan=None) -> str:
    _ = plan
    cleaned = normalize_search_text(query)
    return str(cleaned or query or "").strip()


def _merge_domains(base: list[str], extra: list[str]) -> list[str]:
    out: list[str] = []
    for row in list(base or []) + list(extra or []):
        item = str(row or "").strip().lower()
        if not item:
            continue
        if item.startswith("www."):
            item = item[4:]
        if item not in out:
            out.append(item)
    return out


def _resolve_geo_hint(
    *,
    original_query: str,
    effective_query: str,
    state: dict[str, Any],
    memory_context: dict[str, Any],
    retrieved_memories: list[Any],
    classification,
) -> dict[str, Any]:
    scored: dict[str, dict[str, Any]] = {}
    category = str(getattr(classification, "primary_category", "") or "").strip().lower()
    query_locations = _extract_locations_from_text(effective_query)
    continuation_used = normalize_search_text(original_query) != normalize_search_text(effective_query)

    def _remember(raw_values: list[str], *, source: str, base_score: float) -> None:
        for index, raw in enumerate(list(raw_values or [])):
            norm = _canonical_geo(raw)
            if not norm:
                continue
            key = norm.lower()
            score = float(base_score) - min(index, 3) * 1.5
            existing = scored.get(key)
            if existing is None:
                scored[key] = {"value": norm, "score": score, "sources": [source]}
                continue
            existing["score"] = max(float(existing.get("score") or 0.0), score) + 2.0
            sources = list(existing.get("sources") or [])
            if source not in sources:
                sources.append(source)
            existing["sources"] = sources

    _remember(query_locations, source="query", base_score=100.0)

    active_task = _as_dict(state.get("web_active_task"))
    active_location = _canonical_geo(
        _pick(
            active_task.get("resolved_location"),
            active_task.get("location"),
            active_task.get("location_place"),
            active_task.get("city"),
            active_task.get("country"),
        )
    )
    if active_location:
        _remember([active_location], source="active_task", base_score=88.0 if continuation_used else 54.0)

    history = list(_as_list(state.get("history")))
    history_rank = 0
    for row in reversed(history[-12:]):
        row_map = _as_dict(row)
        if str(row_map.get("role") or "").strip().lower() != "user":
            continue
        text = _history_text(row_map)
        if not text:
            continue
        locations = _extract_locations_from_text(text)
        if not locations:
            continue
        base_score = max(60.0, 82.0 - history_rank * 8.0)
        _remember(locations, source=f"history:{history_rank + 1}", base_score=base_score)
        history_rank += 1

    context_tags = _as_dict(state.get("context_tags"))
    state_candidates: list[str] = []
    for key in ("dialogue_location", "location", "location_place", "city", "country", "local_region"):
        value = str(context_tags.get(key) or state.get(key) or "").strip()
        if value:
            state_candidates.append(value)
    _remember(state_candidates, source="state", base_score=44.0)

    profile_summary = _as_dict(state.get("profile_summary"))
    profile_user = _as_dict(profile_summary.get("user"))
    profile_candidates: list[str] = []
    for key in ("location_place", "location", "city", "country"):
        value = str(profile_user.get(key) or "").strip()
        if value:
            profile_candidates.append(value)
    _remember(profile_candidates, source="profile", base_score=28.0)

    blocks = _as_dict(memory_context.get("blocks"))
    memory_candidates: list[str] = []
    for key in ("session_summary", "working_memory", "retrieved_semantic"):
        memory_candidates.extend(_extract_locations_from_text(str(blocks.get(key) or "")))
    _remember(memory_candidates, source="memory_context", base_score=20.0)

    retrieved_candidates: list[str] = []
    for row in list(retrieved_memories or [])[:12]:
        row_map = _as_dict(row)
        text = str(row_map.get("text") or "")
        fact_match = _LOCATION_FACT_RE.search(text)
        if fact_match:
            retrieved_candidates.append(str(fact_match.group(1) or "").strip())
        metadata = _as_dict(row_map.get("metadata"))
        if str(metadata.get("key") or "").strip().lower() == "location_place":
            retrieved_candidates.append(str(metadata.get("value") or "").strip())
        for key in ("location", "location_place", "city", "country"):
            value = str(metadata.get(key) or "").strip()
            if value:
                retrieved_candidates.append(value)
        retrieved_candidates.extend(_extract_locations_from_text(text))
    _remember(retrieved_candidates, source="retrieved_memory", base_score=14.0)

    candidates = _top_geo_candidates(scored)
    if query_locations:
        best_query = next((row for row in candidates if "query" in list(row.get("sources") or [])), candidates[0] if candidates else {})
        return _geo_decision_payload(
            selected=best_query,
            applied=True,
            reason="explicit_query_location",
            category=category,
            continuation_used=continuation_used,
            candidates=candidates,
        )

    if (
        active_location
        and continuation_used
        and category in {"weather", "news"}
        and _active_task_matches_query(active_task=active_task, effective_query=effective_query)
    ):
        best_active = next((row for row in candidates if "active_task" in list(row.get("sources") or [])), candidates[0] if candidates else {})
        return _geo_decision_payload(
            selected=best_active,
            applied=True,
            reason="active_task_continuation_location",
            category=category,
            continuation_used=continuation_used,
            candidates=candidates,
        )

    best = candidates[0] if candidates else {}
    reject_reason = "no_geo_candidates"
    if best:
        if category == "finance":
            reject_reason = "implicit_geo_disabled_for_finance"
        else:
            reject_reason = "implicit_geo_disabled_without_explicit_location"
    return _geo_decision_payload(
        selected=best,
        applied=False,
        reason=reject_reason,
        category=category,
        continuation_used=continuation_used,
        candidates=candidates,
    )


def _geo_decision_payload(
    *,
    selected: dict[str, Any],
    applied: bool,
    reason: str,
    category: str,
    continuation_used: bool,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    picked = _as_dict(selected)
    value = str(picked.get("value") or "").strip()
    sources = [str(x or "").strip() for x in list(picked.get("sources") or []) if str(x or "").strip()]
    return {
        "value": value if applied else "",
        "candidate": value,
        "candidate_sources": sources,
        "candidate_score": float(picked.get("score") or 0.0),
        "applied": bool(applied),
        "reason": str(reason or "").strip(),
        "category": str(category or "").strip().lower(),
        "continuation_used": bool(continuation_used),
        "candidates": candidates[:4],
    }


def _top_geo_candidates(scored: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        [dict(value=row.get("value") or "", score=float(row.get("score") or 0.0), sources=list(row.get("sources") or [])) for row in scored.values()],
        key=lambda item: (
            float(item.get("score") or 0.0),
            len(list(item.get("sources") or [])),
            len(str(item.get("value") or "")),
        ),
        reverse=True,
    )
    return ranked[:6]


def _active_task_matches_query(*, active_task: dict[str, Any], effective_query: str) -> bool:
    base_query = normalize_search_text(
        _pick(
            active_task.get("base_query"),
            active_task.get("query"),
            active_task.get("latest_query"),
        )
    )
    current_query = normalize_search_text(effective_query)
    if not base_query or not current_query:
        return False
    base_tokens = {token.lower() for token in re.findall(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9]+", base_query) if len(token) >= 3}
    current_tokens = {token.lower() for token in re.findall(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9]+", current_query) if len(token) >= 3}
    if not base_tokens or not current_tokens:
        return False
    return bool(base_tokens.intersection(current_tokens))


def _history_text(row: dict[str, Any]) -> str:
    return str(
        _pick(
            row.get("content"),
            row.get("text"),
            row.get("message"),
            row.get("query"),
        )
        or ""
    ).strip()


def _extract_locations_from_text(text: str) -> list[str]:
    src = normalize_search_text(text)
    if not src:
        return []
    out: list[str] = []
    for match in _GEO_PATTERN.findall(src):
        token = str(match or "").strip(" .,!?;:()[]{}\"'").strip()
        if token:
            out.append(token)
    # Keep one-token country/city hints even without a preposition.
    low = src.lower()
    for token in (
        "ukraine",
        "украина",
        "україна",
        "kyiv",
        "kiev",
        "киев",
        "київ",
    ):
        if token in low:
            out.append(token)
    return out


def _canonical_geo(value: str) -> str:
    src = str(value or "").strip()
    if not src:
        return ""
    src = re.sub(r"\s*[\(\[].*$", "", src).strip()
    cleaned = re.sub(r"\s+", " ", src).strip(" .,!?;:()[]{}\"'")
    if not cleaned:
        return ""
    low = cleaned.lower()
    if low in _NON_GEO_HINTS:
        return ""
    mapping = {
        "киеве": "Kyiv",
        "киев": "Kyiv",
        "києві": "Kyiv",
        "київ": "Kyiv",
        "kiev": "Kyiv",
        "kyiv": "Kyiv",
        "украине": "Ukraine",
        "украина": "Ukraine",
        "україні": "Ukraine",
        "україна": "Ukraine",
        "ukraine": "Ukraine",
        "финляндия": "Finland",
        "финляндии": "Finland",
        "finland": "Finland",
    }
    if low in mapping:
        return mapping[low]
    if len(cleaned) <= 1:
        return ""
    return cleaned


def _geo_preferred_domains(geo_hint: str, query_category: str = "") -> list[str]:
    low = str(geo_hint or "").strip().lower()
    if not low:
        return []
    if any(token in low for token in _UA_GEO_TOKENS):
        category = _normalize_policy_geo_category(query_category)
        return list(_UA_PREFERRED_DOMAINS.get(category, []))
    return []


def _normalize_policy_geo_category(value: str) -> str:
    token = str(value or "").strip().lower()
    if token in {"version", "docs"}:
        return "docs"
    if token in {"finance"}:
        return "finance"
    if token in {"weather"}:
        return "weather"
    if token in {"news"}:
        return "news"
    return "generic"


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


def _extract_nlu_intents_from_query(
    *,
    query: str,
    segmenter: SemanticSegmenter | None,
    segment_classifier: SegmentClassifier | None,
    intent_resolver: IntentResolver | None,
    min_confidence: float = 0.55,
) -> list[str]:
    text = str(query or "").strip()
    if not text or segmenter is None or segment_classifier is None or intent_resolver is None:
        return []
    try:
        raw_segments = list(segmenter.split(text))
        if not raw_segments:
            return []
        classified = segment_classifier.classify(raw_segments)
        intents, _ = intent_resolver.resolve(classified)
    except Exception:
        return []

    out: list[str] = []
    threshold = max(0.0, min(1.0, float(min_confidence)))
    for row in list(intents or []):
        name = str(getattr(row, "name", "") or "").strip().lower()
        if not name:
            continue
        confidence = _to_float(getattr(row, "confidence", 0.0), 0.0)
        if confidence < threshold:
            continue
        if name not in out:
            out.append(name)
    return out


def _merge_lower_tokens(base: Any, extra: list[str]) -> list[str]:
    out: list[str] = []
    for row in list(_as_list(base)) + list(extra or []):
        item = str(row or "").strip().lower()
        if item and item not in out:
            out.append(item)
    return out


def _has_external_nlu_intent(intents: list[str]) -> bool:
    values = {str(x or "").strip().lower() for x in list(intents or []) if str(x or "").strip()}
    return bool(values.intersection({"weather_query", "search_query", "coding_question", "finance_query", "news_query"}))


def _is_smalltalk_query(query: str, *, ctx_tags: dict[str, Any]) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    intent = str(ctx_tags.get("intent") or "").strip().lower()
    if intent in {"chat", "smalltalk", "chatting"} and _SMALLTALK_RE.search(text):
        return True
    return False


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
