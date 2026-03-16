from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modules.internet.web.web_models import (
    ConfidenceAssessment,
    FreshnessAssessment,
    QueryClassification,
    SearchBudget,
    WebPolicyDecision,
    WebSearchMode,
)


@dataclass(frozen=True)
class PolicyThresholds:
    no_search_max: float = 0.26
    verify_max: float = 0.46
    soft_max: float = 0.66
    targeted_max: float = 0.84


@dataclass(frozen=True)
class PolicyWeights:
    base: float = 0.14
    external_fact: float = 0.42
    mixed_query: float = 0.24
    ambiguous_query: float = 0.22
    temporal_risk: float = 0.48
    confidence_penalty: float = 0.45
    freshness_required: float = 0.24
    stakes_medium: float = 0.08
    stakes_high: float = 0.22
    explicit_search_intent: float = 0.24
    force_keyword_boost: float = 0.34
    mode_on_boost: float = 0.06
    nlu_external_boost: float = 0.12
    correction_challenge_boost: float = 0.28
    local_scope_penalty: float = 0.50
    local_scope_depth_cap_threshold: float = 0.33
    category_penalty_default: float = 0.36


@dataclass(frozen=True)
class WebPolicyConfig:
    thresholds: PolicyThresholds = field(default_factory=PolicyThresholds)
    weights: PolicyWeights = field(default_factory=PolicyWeights)
    never_search_categories: list[str] = field(default_factory=list)
    category_penalties: dict[str, float] = field(default_factory=dict)
    force_search_keywords: list[str] = field(default_factory=list)
    budget_by_mode: dict[WebSearchMode, SearchBudget] = field(default_factory=dict)


def default_policy_config() -> WebPolicyConfig:
    return WebPolicyConfig(
        never_search_categories=["reasoning", "architecture", "rewrite"],
        category_penalties={
            "reasoning": 0.32,
            "architecture": 0.30,
            "refactor": 0.35,
            "local": 0.42,
        },
        force_search_keywords=[
            "сейчас",
            "latest",
            "актуаль",
            "последняя версия",
            "цена",
            "сколько стоит",
            "вышло ли",
            "news",
            "today",
            "что нового",
            "доступно ли",
            "2026",
        ],
        budget_by_mode={
            WebSearchMode.NO_SEARCH: SearchBudget(max_queries=0, max_sources=0, max_pages=0, max_fetches=0),
            WebSearchMode.VERIFY_ONLY: SearchBudget(max_queries=1, max_sources=2, max_pages=1, max_fetches=1),
            WebSearchMode.SOFT_SEARCH: SearchBudget(max_queries=2, max_sources=3, max_pages=2, max_fetches=2),
            WebSearchMode.TARGETED_SEARCH: SearchBudget(max_queries=3, max_sources=5, max_pages=3, max_fetches=3),
            WebSearchMode.DEEP_SEARCH: SearchBudget(max_queries=5, max_sources=8, max_pages=5, max_fetches=5),
        },
    )


class WebPolicyEngine:
    def __init__(self, cfg: WebPolicyConfig | None = None):
        self._cfg = cfg or default_policy_config()

    def decide(
        self,
        *,
        query: str,
        classification: QueryClassification,
        confidence: ConfidenceAssessment,
        freshness: FreshnessAssessment,
        web_mode: str,
        web_auto_profile: str | None = None,
        internet_enabled: bool,
        user_override: str | None = None,
        policy_context: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> WebPolicyDecision:
        _ = web_auto_profile
        mode_raw = str(web_mode or "auto").strip().lower()
        source = str(query or "").strip().lower()
        context_override = ""
        if isinstance(policy_context, dict):
            context_override = str(policy_context.get("web_override") or "").strip()
        override = _normalize_user_override(user_override or context_override)
        context = dict(policy_context or {})
        nlu_intents = _extract_nlu_intents(context)
        smalltalk_context = _is_smalltalk_context(
            context=context,
            classification=classification,
            nlu_intents=nlu_intents,
        )
        weights = self._cfg.weights
        thresholds = self._cfg.thresholds

        if not bool(internet_enabled):
            return self._hard_decision(mode=WebSearchMode.NO_SEARCH, reason="internet_disabled")
        if override == "no-web":
            return self._hard_decision(mode=WebSearchMode.NO_SEARCH, reason="user_override_no_web")
        if mode_raw == "off" and override != "web":
            return self._hard_decision(mode=WebSearchMode.NO_SEARCH, reason="web_mode_off")
        if (
            override != "web"
            and smalltalk_context
            and not classification.is_correction_challenge
            and not classification.explicit_search_intent
            and not classification.requires_freshness
            and not classification.is_external_fact_question
        ):
            return self._hard_decision(mode=WebSearchMode.NO_SEARCH, reason="smalltalk_context_hard_skip")

        force_keyword = _has_any(source, self._cfg.force_search_keywords)
        clear_fx_request = _is_clear_fx_request(classification)
        if (
            override != "web"
            and not classification.explicit_search_intent
            and not classification.is_correction_challenge
            and str(classification.primary_category or "").strip().lower() == "chitchat"
            and not classification.requires_freshness
            and not classification.is_external_fact_question
        ):
            return self._hard_decision(mode=WebSearchMode.NO_SEARCH, reason="chitchat_hard_skip")
        category_penalty = self._category_penalty_for(classification.primary_category)
        category_override_signal = bool(
            category_penalty > 0.0
            and (
                classification.requires_freshness
                or classification.is_temporal
                or freshness.temporal_risk >= 0.55
                or classification.explicit_search_intent
            )
        )

        breakdown = {
            "base": float(weights.base),
            "external_fact": float(weights.external_fact if classification.is_external_fact_question else 0.0),
            "mixed_query": float(weights.mixed_query if classification.query_type == "mixed" else 0.0),
            "ambiguous_query": float(weights.ambiguous_query if classification.is_ambiguous else 0.0),
            "temporal_risk": float(weights.temporal_risk * max(0.0, min(1.0, freshness.temporal_risk))),
            "confidence_penalty": float(weights.confidence_penalty * max(0.0, min(1.0, 1.0 - confidence.score))),
            "freshness_required": float(weights.freshness_required if classification.requires_freshness else 0.0),
            "stakes": float(weights.stakes_high if classification.stakes_level == "high" else (weights.stakes_medium if classification.stakes_level == "medium" else 0.0)),
            "explicit_search_intent": float(weights.explicit_search_intent if classification.explicit_search_intent else 0.0),
            "force_keyword_boost": float(weights.force_keyword_boost if force_keyword else 0.0),
            "mode_on_boost": float(weights.mode_on_boost if mode_raw == "on" else 0.0),
            "nlu_external_boost": float(weights.nlu_external_boost if _has_nlu_external_intent(nlu_intents) else 0.0),
            "correction_challenge_boost": float(
                weights.correction_challenge_boost if classification.is_correction_challenge else 0.0
            ),
            "clear_fx_request": 1.0 if clear_fx_request else 0.0,
            "category_penalty": float(-category_penalty),
            "category_penalty_overridden": 1.0 if category_override_signal else 0.0,
        }

        local_scope_cap_applied = False
        if (
            (
                classification.is_local_project_question
                or (
                    classification.query_type == "local_logical"
                    and not classification.explicit_search_intent
                    and not classification.is_temporal
                )
            )
            and not classification.requires_freshness
            and not classification.is_external_fact_question
        ):
            breakdown["local_scope_penalty"] = float(-weights.local_scope_penalty)
            local_scope_cap_applied = True
        else:
            breakdown["local_scope_penalty"] = 0.0

        score = 0.0
        for value in breakdown.values():
            score += float(value)
        score = max(0.0, min(1.0, score))

        mode = _mode_from_score(score=score, thresholds=thresholds)

        if local_scope_cap_applied and score <= float(weights.local_scope_depth_cap_threshold):
            mode = WebSearchMode.NO_SEARCH
        elif local_scope_cap_applied and mode in {WebSearchMode.TARGETED_SEARCH, WebSearchMode.DEEP_SEARCH}:
            mode = WebSearchMode.SOFT_SEARCH

        if classification.requires_freshness and mode == WebSearchMode.NO_SEARCH:
            mode = WebSearchMode.VERIFY_ONLY
        fx_floor_applied = False
        if clear_fx_request:
            min_mode = _minimum_currency_mode(classification)
            if _mode_rank(mode) < _mode_rank(min_mode):
                mode = min_mode
                fx_floor_applied = True
        if classification.is_correction_challenge:
            min_mode = _minimum_recheck_mode(classification)
            if _mode_rank(mode) < _mode_rank(min_mode):
                mode = min_mode
        if (
            mode_raw == "on"
            and mode == WebSearchMode.NO_SEARCH
            and (
                classification.is_external_fact_question
                or classification.requires_freshness
                or _has_nlu_external_intent(nlu_intents)
            )
        ):
            mode = WebSearchMode.VERIFY_ONLY

        reason = "score_routing"
        if override == "web":
            if mode == WebSearchMode.NO_SEARCH:
                mode = WebSearchMode.VERIFY_ONLY
            reason = "user_override_web"
        elif mode_raw == "on" and mode != WebSearchMode.NO_SEARCH:
            reason = "web_mode_on_prefer"
        elif force_keyword:
            reason = "force_keyword_boost"
        elif classification.is_correction_challenge:
            reason = "factual_challenge_recheck"
        elif fx_floor_applied:
            reason = "clear_fx_query_floor"
        elif smalltalk_context and mode == WebSearchMode.NO_SEARCH:
            reason = "smalltalk_context_hard_skip"
        elif classification.requires_freshness:
            reason = "freshness_required"
        elif local_scope_cap_applied and mode == WebSearchMode.NO_SEARCH:
            reason = "local_scope_cap"

        budget = self._cfg.budget_by_mode.get(mode) or SearchBudget()
        if override == "web":
            breakdown["user_override_web"] = 1.0
        category_penalty_overridden = bool(category_override_signal and mode != WebSearchMode.NO_SEARCH)
        return WebPolicyDecision(
            mode=mode,
            should_search=bool(mode != WebSearchMode.NO_SEARCH),
            web_need_score=float(score),
            reason=reason,
            budget=budget,
            decision_breakdown=breakdown,
            local_scope_cap_applied=bool(local_scope_cap_applied),
            category_penalty_applied=float(category_penalty),
            category_penalty_overridden=category_penalty_overridden,
        )

    def _hard_decision(self, *, mode: WebSearchMode, reason: str) -> WebPolicyDecision:
        budget = self._cfg.budget_by_mode.get(mode) or SearchBudget()
        return WebPolicyDecision(
            mode=mode,
            should_search=bool(mode != WebSearchMode.NO_SEARCH),
            web_need_score=0.0,
            reason=str(reason),
            budget=budget,
            decision_breakdown={"hard_gate": 1.0},
            local_scope_cap_applied=False,
            category_penalty_applied=0.0,
            category_penalty_overridden=False,
        )

    def upgrade_mode_once(self, decision: WebPolicyDecision) -> WebPolicyDecision:
        current = decision.mode
        next_mode = current
        if current == WebSearchMode.VERIFY_ONLY:
            next_mode = WebSearchMode.SOFT_SEARCH
        elif current == WebSearchMode.SOFT_SEARCH:
            next_mode = WebSearchMode.TARGETED_SEARCH
        elif current == WebSearchMode.TARGETED_SEARCH:
            next_mode = WebSearchMode.DEEP_SEARCH
        if next_mode == current:
            return decision
        budget = self._cfg.budget_by_mode.get(next_mode) or decision.budget
        breakdown = dict(decision.decision_breakdown or {})
        breakdown["quality_escalation"] = 1.0
        return WebPolicyDecision(
            mode=next_mode,
            should_search=True,
            web_need_score=max(float(decision.web_need_score), 0.5),
            reason="quality_escalation",
            budget=budget,
            decision_breakdown=breakdown,
            local_scope_cap_applied=bool(decision.local_scope_cap_applied),
            category_penalty_applied=float(decision.category_penalty_applied),
            category_penalty_overridden=bool(decision.category_penalty_overridden),
        )

    def _category_penalty_for(self, category: str) -> float:
        value = str(category or "").strip().lower()
        if not value:
            return 0.0
        if value not in {str(x).strip().lower() for x in list(self._cfg.never_search_categories or [])}:
            return 0.0
        raw = self._cfg.category_penalties.get(value)
        if raw is None:
            return max(0.0, float(self._cfg.weights.category_penalty_default))
        try:
            return max(0.0, float(raw))
        except Exception:
            return max(0.0, float(self._cfg.weights.category_penalty_default))


def config_from_dict(payload: dict[str, Any] | None) -> WebPolicyConfig:
    src = dict(payload or {})
    base = default_policy_config()

    thresholds_src = dict(src.get("thresholds") or {})
    weights_src = dict(src.get("weights") or {})
    budgets_src = dict(src.get("budgets") or {})

    thresholds = PolicyThresholds(
        no_search_max=_num(thresholds_src.get("no_search_max"), base.thresholds.no_search_max),
        verify_max=_num(thresholds_src.get("verify_max"), base.thresholds.verify_max),
        soft_max=_num(thresholds_src.get("soft_max"), base.thresholds.soft_max),
        targeted_max=_num(thresholds_src.get("targeted_max"), base.thresholds.targeted_max),
    )

    weights = PolicyWeights(
        base=_num(weights_src.get("base"), base.weights.base),
        external_fact=_num(weights_src.get("external_fact"), base.weights.external_fact),
        mixed_query=_num(weights_src.get("mixed_query"), base.weights.mixed_query),
        ambiguous_query=_num(weights_src.get("ambiguous_query"), base.weights.ambiguous_query),
        temporal_risk=_num(weights_src.get("temporal_risk"), base.weights.temporal_risk),
        confidence_penalty=_num(weights_src.get("confidence_penalty"), base.weights.confidence_penalty),
        freshness_required=_num(weights_src.get("freshness_required"), base.weights.freshness_required),
        stakes_medium=_num(weights_src.get("stakes_medium"), base.weights.stakes_medium),
        stakes_high=_num(weights_src.get("stakes_high"), base.weights.stakes_high),
        explicit_search_intent=_num(weights_src.get("explicit_search_intent"), base.weights.explicit_search_intent),
        force_keyword_boost=_num(weights_src.get("force_keyword_boost"), base.weights.force_keyword_boost),
        mode_on_boost=_num(weights_src.get("mode_on_boost"), base.weights.mode_on_boost),
        nlu_external_boost=_num(weights_src.get("nlu_external_boost"), base.weights.nlu_external_boost),
        correction_challenge_boost=_num(
            weights_src.get("correction_challenge_boost"),
            base.weights.correction_challenge_boost,
        ),
        local_scope_penalty=_num(weights_src.get("local_scope_penalty"), base.weights.local_scope_penalty),
        local_scope_depth_cap_threshold=_num(
            weights_src.get("local_scope_depth_cap_threshold"),
            base.weights.local_scope_depth_cap_threshold,
        ),
        category_penalty_default=_num(weights_src.get("category_penalty_default"), base.weights.category_penalty_default),
    )

    budget_by_mode = dict(base.budget_by_mode)
    for mode in list(WebSearchMode):
        key = mode.value.lower()
        row = dict(budgets_src.get(key) or budgets_src.get(mode.value) or {})
        if not row:
            continue
        budget_by_mode[mode] = SearchBudget(
            max_queries=max(0, int(_num(row.get("max_queries"), budget_by_mode[mode].max_queries))),
            max_sources=max(0, int(_num(row.get("max_sources"), budget_by_mode[mode].max_sources))),
            max_pages=max(0, int(_num(row.get("max_pages"), budget_by_mode[mode].max_pages))),
            max_fetches=max(
                0,
                int(
                    _num(
                        row.get("max_fetches"),
                        budget_by_mode[mode].effective_max_fetches()
                        if hasattr(budget_by_mode[mode], "effective_max_fetches")
                        else budget_by_mode[mode].max_pages,
                    )
                ),
            ),
        )

    raw_never = src.get("never_search_categories")
    never_categories: list[str]
    category_penalties = {
        str(k or "").strip().lower(): max(0.0, float(v))
        for k, v in dict(src.get("category_penalties") or base.category_penalties).items()
        if str(k or "").strip()
    }
    if isinstance(raw_never, dict):
        never_categories = [str(k or "").strip().lower() for k in list(raw_never.keys()) if str(k or "").strip()]
        for key, value in dict(raw_never).items():
            token = str(key or "").strip().lower()
            if not token:
                continue
            try:
                category_penalties[token] = max(0.0, float(value))
            except Exception:
                pass
    else:
        never_categories = [
            str(x or "").strip().lower()
            for x in list(raw_never or base.never_search_categories)
            if str(x or "").strip()
        ]
    force_keywords = [
        str(x or "").strip().lower()
        for x in list(src.get("force_search_keywords") or base.force_search_keywords)
        if str(x or "").strip()
    ]

    return WebPolicyConfig(
        thresholds=thresholds,
        weights=weights,
        never_search_categories=never_categories,
        category_penalties=category_penalties,
        force_search_keywords=force_keywords,
        budget_by_mode=budget_by_mode,
    )


def _mode_from_score(*, score: float, thresholds: PolicyThresholds) -> WebSearchMode:
    value = max(0.0, min(1.0, float(score)))
    if value <= thresholds.no_search_max:
        return WebSearchMode.NO_SEARCH
    if value <= thresholds.verify_max:
        return WebSearchMode.VERIFY_ONLY
    if value <= thresholds.soft_max:
        return WebSearchMode.SOFT_SEARCH
    if value <= thresholds.targeted_max:
        return WebSearchMode.TARGETED_SEARCH
    return WebSearchMode.DEEP_SEARCH


def _minimum_recheck_mode(classification: QueryClassification) -> WebSearchMode:
    category = str(classification.primary_category or "").strip().lower()
    if category in {"finance", "price", "weather", "news"}:
        return WebSearchMode.TARGETED_SEARCH
    if bool(getattr(classification, "requires_freshness", False)):
        return WebSearchMode.TARGETED_SEARCH
    hits = {str(x or "").strip().lower() for x in list(getattr(classification, "category_hits", []) or [])}
    if any("date" in hit or "\u0434\u0430\u0442" in hit for hit in hits):
        return WebSearchMode.TARGETED_SEARCH
    return WebSearchMode.VERIFY_ONLY


def _minimum_currency_mode(classification: QueryClassification) -> WebSearchMode:
    if bool(getattr(classification, "requires_freshness", False) or getattr(classification, "is_temporal", False)):
        return WebSearchMode.TARGETED_SEARCH
    return WebSearchMode.VERIFY_ONLY


def _is_clear_fx_request(classification: QueryClassification) -> bool:
    category = str(getattr(classification, "primary_category", "") or "").strip().lower()
    if category != "finance":
        return False
    hits = {
        str(x or "").strip().lower()
        for x in list(getattr(classification, "category_hits", []) or [])
        if str(x or "").strip()
    }
    if any(
        token in hits
        for token in {
            "usd",
            "eur",
            "uah",
            "доллар",
            "евро",
            "гривна",
            "грн",
            "валюта",
            "currency_pair",
            "colloquial_fx",
            "exchange rate",
        }
    ):
        return True
    return bool(getattr(classification, "is_external_fact_question", False))


def _mode_rank(mode: WebSearchMode) -> int:
    order = {
        WebSearchMode.NO_SEARCH: 0,
        WebSearchMode.VERIFY_ONLY: 1,
        WebSearchMode.SOFT_SEARCH: 2,
        WebSearchMode.TARGETED_SEARCH: 3,
        WebSearchMode.DEEP_SEARCH: 4,
    }
    return int(order.get(mode, 0))


def _has_any(text: str, items: list[str]) -> bool:
    low = str(text or "").strip().lower()
    if not low:
        return False
    return any(str(item or "").strip().lower() in low for item in list(items or []))


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_user_override(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw in {"web", "/web", "on", "force_web", "force-web"}:
        return "web"
    if raw in {"no-web", "/no-web", "off", "no_web", "disable_web", "disable-web"}:
        return "no-web"
    return ""


def _extract_nlu_intents(context: dict[str, Any]) -> list[str]:
    raw = context.get("nlu_intents")
    out: list[str] = []
    for row in list(raw or []):
        token = str(row or "").strip().lower()
        if token and token not in out:
            out.append(token)
    return out


def _normalize_meta_intent(context: dict[str, Any]) -> str:
    direct = str(context.get("intent") or "").strip().lower()
    if direct:
        return direct
    metadata = context.get("metadata")
    if isinstance(metadata, dict):
        intent = metadata.get("intent")
        if isinstance(intent, dict):
            label = str(intent.get("label") or "").strip().lower()
            if label:
                return label
        label = str(metadata.get("intent") or "").strip().lower()
        if label:
            return label
    return ""


def _is_smalltalk_context(
    *,
    context: dict[str, Any],
    classification: QueryClassification,
    nlu_intents: list[str],
) -> bool:
    if "smalltalk" in set(nlu_intents):
        return True
    meta_intent = _normalize_meta_intent(context)
    if meta_intent in {"chat", "smalltalk", "chitchat", "greeting"}:
        return True
    if str(classification.primary_category or "").strip().lower() == "chitchat":
        return True
    return False


def _has_nlu_external_intent(intents: list[str]) -> bool:
    values = {str(x or "").strip().lower() for x in list(intents or []) if str(x or "").strip()}
    return bool(values.intersection({"weather_query", "search_query", "coding_question", "finance_query", "news_query"}))
