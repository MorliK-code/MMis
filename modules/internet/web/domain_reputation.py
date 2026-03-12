from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from config.settings import load_config


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_]+")
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "today",
    "now",
    "latest",
    "official",
    "source",
    "news",
    "update",
    "about",
    "what",
    "where",
    "when",
    "there",
    "here",
    "это",
    "этот",
    "эта",
    "эти",
    "сегодня",
    "сейчас",
    "последний",
    "обновление",
    "новость",
    "по",
    "для",
}


_TRUST_POLICY_STATES = {"trusted", "preferred", "blocked", "risky", "degraded"}
_TRUST_POLICY_CATEGORIES = {"docs", "finance", "weather", "news", "generic"}
_TRUST_POLICY_CATEGORY_ALIASES = {
    "docs": "docs",
    "doc": "docs",
    "documentation": "docs",
    "version": "docs",
    "release": "docs",
    "finance": "finance",
    "fx_rate": "finance",
    "currency_rate": "finance",
    "weather": "weather",
    "forecast": "weather",
    "news": "news",
    "mentions": "news",
    "news_release": "news",
    "generic": "generic",
    "external": "generic",
    "mixed": "generic",
    "ambiguous": "generic",
}


def _normalize_domain_token(value: Any) -> str:
    src = str(value or "").strip().lower()
    if not src:
        return ""
    if "://" in src:
        src = src.split("://", 1)[1]
    if "/" in src:
        src = src.split("/", 1)[0]
    if ":" in src:
        src = src.split(":", 1)[0]
    if src.startswith("www."):
        src = src[4:]
    return src.strip(". ")


def _normalize_domain_list(value: Any) -> tuple[str, ...]:
    out: list[str] = []
    for row in list(value or []):
        item = _normalize_domain_token(row)
        if item and item not in out:
            out.append(item)
    return tuple(out)


def _merge_domain_lists(*values: Any) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        for item in _normalize_domain_list(value):
            if item not in out:
                out.append(item)
    return tuple(out)


def _domain_matches(domain: str, candidate: str) -> bool:
    src = _normalize_domain_token(domain)
    rule = _normalize_domain_token(candidate)
    if not src or not rule:
        return False
    return src == rule or src.endswith("." + rule)


def _normalize_manual_overrides(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    out: dict[str, str] = {}
    keys = {str(k or "").strip().lower() for k in value.keys()}
    if keys and keys.issubset(_TRUST_POLICY_STATES):
        for state, domains in value.items():
            token = str(state or "").strip().lower()
            if token not in _TRUST_POLICY_STATES:
                continue
            for domain in _normalize_domain_list(domains):
                out[domain] = token
        return out

    for domain, state in value.items():
        rule = _normalize_domain_token(domain)
        token = str(state or "").strip().lower()
        if rule and token in _TRUST_POLICY_STATES:
            out[rule] = token
    return out


def _normalize_policy_category(value: Any) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    return str(_TRUST_POLICY_CATEGORY_ALIASES.get(token, token if token in _TRUST_POLICY_CATEGORIES else "")).strip().lower()


def _normalize_category_domain_map(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for category, domains in value.items():
        key = _normalize_policy_category(category)
        if not key:
            continue
        merged = _merge_domain_lists(out.get(key, ()), domains)
        if merged:
            out[key] = merged
    return out


def _merge_category_domain_maps(*values: Any) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for value in values:
        for category, domains in _normalize_category_domain_map(value).items():
            merged = _merge_domain_lists(out.get(category, ()), domains)
            if merged:
                out[category] = merged
    return out


def _reputation_state(score: float) -> str:
    value = _clamp(float(score), -1.0, 1.0)
    if value >= 0.65:
        return "trusted"
    if value >= 0.35:
        return "positive"
    if value <= -0.65:
        return "degraded"
    if value <= -0.35:
        return "risky"
    return "neutral"


@dataclass(frozen=True)
class DomainTrustAssessment:
    domain: str
    policy_state: str = "neutral"
    manual_override: str = ""
    reputation_score: float = 0.0
    reputation_state: str = "neutral"
    hard_blocked: bool = False
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": str(self.domain or "").strip().lower(),
            "policy_state": str(self.policy_state or "neutral"),
            "manual_override": str(self.manual_override or ""),
            "reputation_score": float(self.reputation_score),
            "reputation_state": str(self.reputation_state or "neutral"),
            "hard_blocked": bool(self.hard_blocked),
            "reasons": [str(x or "").strip() for x in list(self.reasons or ()) if str(x or "").strip()],
        }


@dataclass(slots=True)
class DomainTrustPolicy:
    trusted_allowlist: tuple[str, ...] = ()
    preferred_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    risky_domains: tuple[str, ...] = ()
    degraded_domains: tuple[str, ...] = ()
    trusted_allowlist_by_category: dict[str, tuple[str, ...]] = field(default_factory=dict)
    preferred_domains_by_category: dict[str, tuple[str, ...]] = field(default_factory=dict)
    blocked_domains_by_category: dict[str, tuple[str, ...]] = field(default_factory=dict)
    risky_domains_by_category: dict[str, tuple[str, ...]] = field(default_factory=dict)
    degraded_domains_by_category: dict[str, tuple[str, ...]] = field(default_factory=dict)
    manual_overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, payload: dict[str, Any] | None = None) -> DomainTrustPolicy:
        cfg = dict(payload or {})
        trust_cfg = dict(cfg.get("trust_policy") or {})
        return cls(
            trusted_allowlist=_merge_domain_lists(cfg.get("trusted_allowlist"), trust_cfg.get("trusted_allowlist")),
            preferred_domains=_merge_domain_lists(cfg.get("preferred_domains"), trust_cfg.get("preferred_domains")),
            blocked_domains=_merge_domain_lists(cfg.get("blocked_domains"), trust_cfg.get("blocked_domains")),
            risky_domains=_merge_domain_lists(cfg.get("risky_domains"), trust_cfg.get("risky_domains")),
            degraded_domains=_merge_domain_lists(cfg.get("degraded_domains"), trust_cfg.get("degraded_domains")),
            trusted_allowlist_by_category=_merge_category_domain_maps(
                cfg.get("trusted_allowlist_by_category"),
                trust_cfg.get("trusted_allowlist_by_category"),
            ),
            preferred_domains_by_category=_merge_category_domain_maps(
                cfg.get("preferred_domains_by_category"),
                trust_cfg.get("preferred_domains_by_category"),
            ),
            blocked_domains_by_category=_merge_category_domain_maps(
                cfg.get("blocked_domains_by_category"),
                trust_cfg.get("blocked_domains_by_category"),
            ),
            risky_domains_by_category=_merge_category_domain_maps(
                cfg.get("risky_domains_by_category"),
                trust_cfg.get("risky_domains_by_category"),
            ),
            degraded_domains_by_category=_merge_category_domain_maps(
                cfg.get("degraded_domains_by_category"),
                trust_cfg.get("degraded_domains_by_category"),
            ),
            manual_overrides=_normalize_manual_overrides(
                trust_cfg.get("manual_overrides", cfg.get("manual_overrides"))
            ),
        )

    def with_runtime(
        self,
        *,
        preferred_domains: list[str] | tuple[str, ...] | None = None,
        blocked_domains: list[str] | tuple[str, ...] | None = None,
        trusted_allowlist: list[str] | tuple[str, ...] | None = None,
        risky_domains: list[str] | tuple[str, ...] | None = None,
        degraded_domains: list[str] | tuple[str, ...] | None = None,
        preferred_domains_by_category: dict[str, Any] | None = None,
        trusted_allowlist_by_category: dict[str, Any] | None = None,
        blocked_domains_by_category: dict[str, Any] | None = None,
        risky_domains_by_category: dict[str, Any] | None = None,
        degraded_domains_by_category: dict[str, Any] | None = None,
        manual_overrides: dict[str, Any] | None = None,
    ) -> DomainTrustPolicy:
        merged_manual = dict(self.manual_overrides or {})
        merged_manual.update(_normalize_manual_overrides(manual_overrides))
        return DomainTrustPolicy(
            trusted_allowlist=_merge_domain_lists(self.trusted_allowlist, trusted_allowlist),
            preferred_domains=_merge_domain_lists(self.preferred_domains, preferred_domains),
            blocked_domains=_merge_domain_lists(self.blocked_domains, blocked_domains),
            risky_domains=_merge_domain_lists(self.risky_domains, risky_domains),
            degraded_domains=_merge_domain_lists(self.degraded_domains, degraded_domains),
            trusted_allowlist_by_category=_merge_category_domain_maps(
                self.trusted_allowlist_by_category,
                trusted_allowlist_by_category,
            ),
            preferred_domains_by_category=_merge_category_domain_maps(
                self.preferred_domains_by_category,
                preferred_domains_by_category,
            ),
            blocked_domains_by_category=_merge_category_domain_maps(
                self.blocked_domains_by_category,
                blocked_domains_by_category,
            ),
            risky_domains_by_category=_merge_category_domain_maps(
                self.risky_domains_by_category,
                risky_domains_by_category,
            ),
            degraded_domains_by_category=_merge_category_domain_maps(
                self.degraded_domains_by_category,
                degraded_domains_by_category,
            ),
            manual_overrides=merged_manual,
        )

    def discovery_domains(self, *, query_category: str = "") -> list[str]:
        category = _normalize_policy_category(query_category)
        if not category:
            return list(_merge_domain_lists(self.trusted_allowlist_by_category.get("generic"), self.preferred_domains_by_category.get("generic")))
        return list(
            _merge_domain_lists(
                self.trusted_allowlist_by_category.get(category),
                self.preferred_domains_by_category.get(category),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        manual: dict[str, list[str]] = {key: [] for key in sorted(_TRUST_POLICY_STATES)}
        for domain, state in sorted(dict(self.manual_overrides or {}).items(), key=lambda item: item[0]):
            token = str(state or "").strip().lower()
            if token in manual:
                manual[token].append(str(domain))
        return {
            "trusted_allowlist": list(self.trusted_allowlist),
            "preferred_domains": list(self.preferred_domains),
            "blocked_domains": list(self.blocked_domains),
            "risky_domains": list(self.risky_domains),
            "degraded_domains": list(self.degraded_domains),
            "trusted_allowlist_by_category": {k: list(v) for k, v in sorted(self.trusted_allowlist_by_category.items()) if v},
            "preferred_domains_by_category": {k: list(v) for k, v in sorted(self.preferred_domains_by_category.items()) if v},
            "blocked_domains_by_category": {k: list(v) for k, v in sorted(self.blocked_domains_by_category.items()) if v},
            "risky_domains_by_category": {k: list(v) for k, v in sorted(self.risky_domains_by_category.items()) if v},
            "degraded_domains_by_category": {k: list(v) for k, v in sorted(self.degraded_domains_by_category.items()) if v},
            "manual_overrides": {k: v for k, v in manual.items() if v},
        }

    def evaluate(self, domain: str, *, reputation_score: float = 0.0, query_category: str = "") -> DomainTrustAssessment:
        src = _normalize_domain_token(domain)
        if not src:
            return DomainTrustAssessment(domain="", reputation_score=0.0)

        manual_override = self._manual_override_for(src)
        category = _normalize_policy_category(query_category)
        configured_state, configured_scope = self._configured_state_for(src, query_category=category)
        policy_state = manual_override or configured_state or "neutral"
        reputation_band = _reputation_state(reputation_score)
        reasons: list[str] = []
        if manual_override:
            reasons.append(f"manual:{manual_override}")
        elif configured_state:
            suffix = f"@{configured_scope}" if configured_scope else ""
            reasons.append(f"policy:{configured_state}{suffix}")
        if reputation_band != "neutral":
            reasons.append(f"reputation:{reputation_band}")

        return DomainTrustAssessment(
            domain=src,
            policy_state=policy_state,
            manual_override=manual_override,
            reputation_score=_clamp(float(reputation_score), -1.0, 1.0),
            reputation_state=reputation_band,
            hard_blocked=(policy_state == "blocked"),
            reasons=tuple(reasons),
        )

    def _manual_override_for(self, domain: str) -> str:
        src = _normalize_domain_token(domain)
        for candidate, state in dict(self.manual_overrides or {}).items():
            if _domain_matches(src, candidate):
                return str(state or "").strip().lower()
        return ""

    def _configured_state_for(self, domain: str, *, query_category: str = "") -> tuple[str, str]:
        src = _normalize_domain_token(domain)
        if any(_domain_matches(src, token) for token in self.blocked_domains):
            return ("blocked", "global")
        matched_category = self._matched_category(self.blocked_domains_by_category, src, query_category)
        if matched_category:
            return ("blocked", matched_category)
        matched_category = self._matched_category(self.trusted_allowlist_by_category, src, query_category)
        if matched_category:
            return ("trusted", matched_category)
        if any(_domain_matches(src, token) for token in self.trusted_allowlist):
            return ("trusted", "global")
        matched_category = self._matched_category(self.preferred_domains_by_category, src, query_category)
        if matched_category:
            return ("preferred", matched_category)
        if any(_domain_matches(src, token) for token in self.preferred_domains):
            return ("preferred", "global")
        matched_category = self._matched_category(self.risky_domains_by_category, src, query_category)
        if matched_category:
            return ("risky", matched_category)
        if any(_domain_matches(src, token) for token in self.risky_domains):
            return ("risky", "global")
        matched_category = self._matched_category(self.degraded_domains_by_category, src, query_category)
        if matched_category:
            return ("degraded", matched_category)
        if any(_domain_matches(src, token) for token in self.degraded_domains):
            return ("degraded", "global")
        return ("", "")

    def _matched_category(self, mapping: dict[str, tuple[str, ...]], domain: str, query_category: str) -> str:
        category = _normalize_policy_category(query_category)
        if not mapping:
            return ""
        if category and any(_domain_matches(domain, token) for token in mapping.get(category, ())):
            return category
        if any(_domain_matches(domain, token) for token in mapping.get("generic", ())):
            return "generic"
        return ""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class DomainReputationEntry:
    domain: str
    score: float = 0.0
    evidence_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": str(self.domain or "").strip().lower(),
            "score": float(self.score),
            "evidence_count": int(self.evidence_count),
            "positive_count": int(self.positive_count),
            "negative_count": int(self.negative_count),
            "last_updated": str(self.last_updated or ""),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> DomainReputationEntry | None:
        data = dict(row or {})
        domain = str(data.get("domain") or "").strip().lower()
        if not domain:
            return None
        try:
            score = float(data.get("score") or 0.0)
        except Exception:
            score = 0.0
        try:
            evidence_count = int(data.get("evidence_count") or 0)
        except Exception:
            evidence_count = 0
        try:
            positive_count = int(data.get("positive_count") or 0)
        except Exception:
            positive_count = 0
        try:
            negative_count = int(data.get("negative_count") or 0)
        except Exception:
            negative_count = 0
        return cls(
            domain=domain,
            score=_clamp(score, -1.0, 1.0),
            evidence_count=max(0, evidence_count),
            positive_count=max(0, positive_count),
            negative_count=max(0, negative_count),
            last_updated=str(data.get("last_updated") or ""),
        )


class DomainReputationStore:
    """
    Lightweight domain reputation memory.

    Reputation is updated from multi-source semantic agreement:
    - domains agreeing with other independent domains move up;
    - outliers move down.
    """

    def __init__(
        self,
        *,
        storage_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ):
        cfg = load_config()
        cfg_map = dict(config or {})
        self.enabled = bool(cfg_map.get("enabled", True))
        self.min_domains = max(2, int(cfg_map.get("min_domains") or 3))
        self.good_similarity = _clamp(float(cfg_map.get("good_similarity") or 0.58), 0.05, 0.99)
        self.bad_similarity = _clamp(float(cfg_map.get("bad_similarity") or 0.30), 0.0, 0.95)
        if self.bad_similarity >= self.good_similarity:
            self.bad_similarity = max(0.0, self.good_similarity - 0.15)
        self.alpha = _clamp(float(cfg_map.get("alpha") or 0.22), 0.01, 0.65)
        self.decay_days = max(1, int(cfg_map.get("decay_days") or 45))

        default_path = Path(cfg.memory_dir).expanduser().resolve() / "web_domain_reputation.json"
        raw_path = storage_path if storage_path is not None else cfg_map.get("storage_path")
        self._path = Path(raw_path).expanduser().resolve() if raw_path else default_path
        self._lock = RLock()
        self._entries: dict[str, DomainReputationEntry] = {}
        self._load()

    def snapshot_scores(self) -> dict[str, float]:
        with self._lock:
            return {str(k): float(v.score) for k, v in self._entries.items()}

    def snapshot_stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {str(k): v.to_dict() for k, v in self._entries.items()}

    def learn_from_evidence(self, items: list[Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "updated": 0, "domains": []}
        collapsed = self._collapse_by_domain(items)
        if len(collapsed) < self.min_domains:
            return {"enabled": True, "updated": 0, "domains": []}

        updates: list[dict[str, Any]] = []
        now_iso = _utc_now_iso()
        with self._lock:
            for domain, row in collapsed.items():
                mean_sim = self._mean_similarity(domain=domain, rows=collapsed)
                signal = self._signal_from_similarity(mean_sim)
                if signal == 0:
                    continue
                confidence = self._confidence_from_similarity(mean_sim, signal)
                quality = _clamp(float(row.get("quality") or 0.5), 0.0, 1.0)
                has_conflict = bool(row.get("has_conflict", False))
                weight = _clamp(0.35 + (0.65 * quality), 0.2, 1.0)
                if has_conflict:
                    weight *= 0.7
                delta = float(signal) * self.alpha * weight * (0.5 + 0.5 * confidence)

                entry = self._entries.get(domain) or DomainReputationEntry(domain=domain)
                self._apply_decay(entry)
                entry.score = _clamp(float(entry.score) + float(delta), -1.0, 1.0)
                entry.evidence_count += 1
                if signal > 0:
                    entry.positive_count += 1
                else:
                    entry.negative_count += 1
                entry.last_updated = now_iso
                self._entries[domain] = entry
                updates.append(
                    {
                        "domain": domain,
                        "similarity": round(float(mean_sim), 4),
                        "delta": round(float(delta), 4),
                        "score": round(float(entry.score), 4),
                        "signal": "positive" if signal > 0 else "negative",
                    }
                )

            if updates:
                self._save_locked()

        updates.sort(key=lambda x: abs(float(x.get("delta") or 0.0)), reverse=True)
        return {"enabled": True, "updated": len(updates), "domains": updates[:12]}

    def _collapse_by_domain(self, items: list[Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for raw in list(items or []):
            row = self._as_dict(raw)
            domain = self._normalize_domain(str(row.get("domain") or ""))
            if not domain:
                continue
            title = str(row.get("title") or "").strip()
            snippet = str(row.get("snippet") or "").strip()
            facts = [str(x or "").strip() for x in list(row.get("key_facts") or []) if str(x or "").strip()]
            text_source = " ".join(facts) or snippet or title
            tokens = self._semantic_tokens(text_source)
            if not tokens:
                continue
            quality = _clamp(float(row.get("quality_score") or 0.5), 0.0, 1.0)
            has_conflict = bool(list(row.get("conflict_flags") or []))
            numbers = self._extract_numbers(text_source)
            state = out.get(domain)
            if state is None:
                out[domain] = {
                    "tokens": set(tokens),
                    "numbers": list(numbers),
                    "quality": quality,
                    "count": 1,
                    "has_conflict": has_conflict,
                }
                continue
            state_tokens = set(state.get("tokens") or set())
            state_tokens.update(tokens)
            prev_q = float(state.get("quality") or 0.5)
            prev_count = int(state.get("count") or 1)
            state["tokens"] = state_tokens
            state_numbers = [float(x) for x in list(state.get("numbers") or [])]
            state_numbers.extend(numbers)
            state["numbers"] = state_numbers[:12]
            state["quality"] = ((prev_q * prev_count) + quality) / max(1, prev_count + 1)
            state["count"] = prev_count + 1
            state["has_conflict"] = bool(state.get("has_conflict", False) or has_conflict)
            out[domain] = state
        return out

    def _mean_similarity(self, *, domain: str, rows: dict[str, dict[str, Any]]) -> float:
        base = set(rows.get(domain, {}).get("tokens") or set())
        if not base:
            return 0.0
        sims: list[float] = []
        for other_domain, payload in rows.items():
            if other_domain == domain:
                continue
            other = set(payload.get("tokens") or set())
            if not other:
                continue
            lexical = self._jaccard(base, other)
            numeric = self._numeric_compatibility(
                list(rows.get(domain, {}).get("numbers") or []),
                list(payload.get("numbers") or []),
            )
            sims.append(lexical * numeric)
        if not sims:
            return 0.0
        sims.sort(reverse=True)
        top = sims[:2]
        return float(sum(top) / max(1, len(top)))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return float(len(a & b) / max(1, len(union)))

    def _signal_from_similarity(self, similarity: float) -> int:
        if similarity >= self.good_similarity:
            return 1
        if similarity <= self.bad_similarity:
            return -1
        return 0

    def _confidence_from_similarity(self, similarity: float, signal: int) -> float:
        if signal > 0:
            span = max(0.01, 1.0 - self.good_similarity)
            return _clamp((float(similarity) - self.good_similarity) / span, 0.0, 1.0)
        span = max(0.01, self.bad_similarity)
        return _clamp((self.bad_similarity - float(similarity)) / span, 0.0, 1.0)

    def _semantic_tokens(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for token in _WORD_RE.findall(str(text or "").lower()):
            word = str(token or "").strip().lower()
            if not word:
                continue
            if len(word) <= 2:
                continue
            if word in _STOPWORDS:
                continue
            tokens.add(word)
        return tokens

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        out: list[float] = []
        for token in _NUM_RE.findall(str(text or "")):
            value = str(token or "").strip().replace(",", ".")
            if not value:
                continue
            try:
                out.append(float(value))
            except Exception:
                continue
        return out[:8]

    @staticmethod
    def _numeric_compatibility(base: list[float], other: list[float]) -> float:
        left = [float(x) for x in list(base or []) if float(x) > 0]
        right = [float(x) for x in list(other or []) if float(x) > 0]
        if not left or not right:
            return 1.0
        best = 1.0
        for a in left:
            for b in right:
                den = max(1.0, abs(a), abs(b))
                rel = abs(a - b) / den
                if rel < best:
                    best = rel
        if best <= 0.08:
            return 1.0
        if best <= 0.18:
            return 0.6
        return 0.2

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if value is None:
            return {}
        try:
            return dict(vars(value))
        except Exception:
            return {}

    @staticmethod
    def _normalize_domain(value: str) -> str:
        return _normalize_domain_token(value)

    def _apply_decay(self, entry: DomainReputationEntry) -> None:
        last = str(entry.last_updated or "").strip()
        if not last:
            return
        try:
            last_ts = dt.datetime.fromisoformat(last.replace("Z", "+00:00"))
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=dt.timezone.utc)
        except Exception:
            return
        now = dt.datetime.now(dt.timezone.utc)
        delta_days = max(0.0, float((now - last_ts).total_seconds()) / 86400.0)
        if delta_days <= 0:
            return
        factor = math.exp(-delta_days / float(self.decay_days))
        entry.score = _clamp(float(entry.score) * factor, -1.0, 1.0)

    def _load(self) -> None:
        with self._lock:
            self._entries = {}
            path = self._path
            try:
                if not path.exists():
                    return
                payload = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
            except Exception:
                return
            domains = dict(payload.get("domains") or {}) if isinstance(payload, dict) else {}
            for domain, row in domains.items():
                data = dict(row or {})
                data["domain"] = str(data.get("domain") or domain)
                entry = DomainReputationEntry.from_dict(data)
                if entry is None:
                    continue
                self._entries[entry.domain] = entry

    def _save_locked(self) -> None:
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": _utc_now_iso(),
            "domains": {str(k): v.to_dict() for k, v in sorted(self._entries.items(), key=lambda x: str(x[0]))},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
