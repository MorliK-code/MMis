from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import urlparse


@dataclass(slots=True)
class EvidenceClaim:
    subject: str
    predicate: str
    value: str
    source_domain: str
    confidence: float
    published_at: str = ""


@dataclass(slots=True)
class ConsensusResult:
    key: str
    total_claims: int
    source_domains: list[str]
    consensus_score: float
    has_conflict: bool


@dataclass(slots=True)
class _ClaimCluster:
    representative_value: str
    claims: list[EvidenceClaim]
    domains: set[str]


class ConsensusEngine:
    """
    Lightweight consensus layer for web evidence.

    The engine groups claims by (subject, predicate), checks conflicts between values,
    and estimates a consensus score that rewards independent-domain agreement.
    """

    _NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

    def build_consensus(self, claims: list[EvidenceClaim]) -> list[ConsensusResult]:
        if not claims:
            return []

        grouped = self._group_claims(claims)
        results: list[ConsensusResult] = []

        for group_key, group_claims in grouped.items():
            clusters = self._cluster_group_claims(group_claims)
            has_conflict = self._claims_conflict(group_claims)
            domains = sorted({self._normalize_domain(c.source_domain) for c in group_claims if c.source_domain})
            consensus_score = self._compute_consensus_score(
                group=group_claims,
                clusters=clusters,
                has_conflict=has_conflict,
            )

            results.append(
                ConsensusResult(
                    key=group_key,
                    total_claims=len(group_claims),
                    source_domains=domains,
                    consensus_score=round(consensus_score, 4),
                    has_conflict=has_conflict,
                )
            )

        results.sort(key=lambda row: row.consensus_score, reverse=True)
        return results

    def _claims_conflict(self, group: list[EvidenceClaim]) -> bool:
        if len(group) <= 1:
            return False

        clusters = self._cluster_group_claims(group)
        if len(clusters) <= 1:
            return False

        total_domains = {self._normalize_domain(c.source_domain) for c in group if c.source_domain}
        if len(total_domains) <= 1:
            return False

        # Conflict is meaningful when at least two distinct value clusters
        # are backed by independent domains.
        supported_clusters = [cluster for cluster in clusters if len(cluster.domains) >= 1]
        return len(supported_clusters) >= 2

    def _group_claims(self, claims: list[EvidenceClaim]) -> dict[str, list[EvidenceClaim]]:
        grouped: dict[str, list[EvidenceClaim]] = {}

        for claim in claims:
            subject = self._normalize_token(claim.subject)
            predicate = self._normalize_token(claim.predicate)
            if not subject or not predicate:
                continue
            key = f"{subject}:{predicate}"
            grouped.setdefault(key, []).append(claim)

        return grouped

    def _cluster_group_claims(self, group: list[EvidenceClaim]) -> list[_ClaimCluster]:
        clusters: list[_ClaimCluster] = []
        predicate = self._normalize_token(group[0].predicate) if group else ""

        for claim in group:
            domain = self._normalize_domain(claim.source_domain)
            if not domain:
                continue

            placed = False
            for cluster in clusters:
                if self._values_equivalent(claim.value, cluster.representative_value, predicate=predicate):
                    cluster.claims.append(claim)
                    cluster.domains.add(domain)
                    placed = True
                    break

            if placed:
                continue

            clusters.append(
                _ClaimCluster(
                    representative_value=str(claim.value or "").strip(),
                    claims=[claim],
                    domains={domain},
                )
            )

        clusters.sort(key=lambda item: (len(item.domains), len(item.claims)), reverse=True)
        return clusters

    def _compute_consensus_score(
        self,
        *,
        group: list[EvidenceClaim],
        clusters: list[_ClaimCluster],
        has_conflict: bool,
    ) -> float:
        if not group:
            return 0.0
        if not clusters:
            return 0.0

        total_claims = len(group)
        total_domains = {self._normalize_domain(c.source_domain) for c in group if c.source_domain}
        total_domain_count = max(1, len(total_domains))

        top_cluster = clusters[0]
        top_claim_count = len(top_cluster.claims)
        top_domain_count = max(1, len(top_cluster.domains))
        avg_confidence = self._average_confidence(top_cluster.claims)

        claim_support_ratio = top_claim_count / max(1, total_claims)
        domain_support_ratio = top_domain_count / total_domain_count
        domain_coverage_bonus = min(1.0, total_domain_count / 4.0)

        score = (
            (0.34 * claim_support_ratio)
            + (0.38 * domain_support_ratio)
            + (0.20 * avg_confidence)
            + (0.08 * domain_coverage_bonus)
        )

        if has_conflict:
            score -= 0.22
        if total_domain_count == 1:
            score -= 0.10

        return self._clamp(score)

    def _values_equivalent(self, left: str, right: str, *, predicate: str) -> bool:
        left_text = self._normalize_value(left)
        right_text = self._normalize_value(right)
        if not left_text or not right_text:
            return False
        if left_text == right_text:
            return True

        left_num = self._extract_first_number(left_text)
        right_num = self._extract_first_number(right_text)
        if left_num is not None and right_num is not None:
            return self._numbers_close(left_num, right_num, predicate=predicate)

        if left_text in right_text or right_text in left_text:
            if min(len(left_text), len(right_text)) >= 4:
                return True

        ratio = SequenceMatcher(a=left_text, b=right_text).ratio()
        return ratio >= 0.88

    def _numbers_close(self, left: float, right: float, *, predicate: str) -> bool:
        pred = self._normalize_token(predicate)
        delta = abs(left - right)
        anchor = max(1.0, abs(left), abs(right))

        if "weather" in pred or "temperature" in pred or "temp" in pred:
            return delta <= 2.0
        if "currency" in pred or "rate" in pred or "fx" in pred:
            return delta <= 0.2 or (delta / anchor) <= 0.02
        if "news" in pred:
            return delta <= 0.0

        return delta <= 0.1 or (delta / anchor) <= 0.03

    def _extract_first_number(self, value: str) -> float | None:
        match = self._NUMBER_RE.search(str(value or ""))
        if not match:
            return None
        raw = match.group(0).replace(",", ".")
        try:
            return float(raw)
        except Exception:
            return None

    @staticmethod
    def _average_confidence(claims: list[EvidenceClaim]) -> float:
        if not claims:
            return 0.0
        return sum(max(0.0, min(1.0, float(c.confidence))) for c in claims) / len(claims)

    @staticmethod
    def _normalize_token(value: str) -> str:
        return str(value or "").strip().lower()

    def _normalize_value(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("ё", "е")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\"'`]+", "", text)
        return text.strip()

    @staticmethod
    def _normalize_domain(value: str) -> str:
        host = str(value or "").strip().lower()
        if not host:
            return ""
        if "://" in host:
            host = urlparse(host).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    @staticmethod
    def _clamp(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

