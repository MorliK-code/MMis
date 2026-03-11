from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .normalizer import normalize_text


@dataclass(slots=True)
class AliasEntry:
    alias: str
    canonical: str
    confidence: float
    evidence_count: int
    source: str  # builtin | learned | imported
    updated_at: str = ""


class AliasMemory:
    """Adaptive alias memory with staged promotion from candidates to active map."""

    _VALID_SOURCES = {"builtin", "learned", "imported"}
    _DEFAULT_BUILTIN_ALIASES: dict[str, str] = {
        "че": "что",
        "шо": "что",
        "погодка": "weather",
        "погодке": "weather",
        "видюха": "gpu",
        "видюху": "gpu",
        "долларчик": "currency_rate",
        "долларчику": "currency_rate",
        "эмбеддинг": "embeddings",
        "эмбеддинги": "embeddings",
    }

    def __init__(self, builtin_aliases: dict[str, str] | None = None) -> None:
        self._active_entries: dict[str, AliasEntry] = {}
        self._candidate_entries: dict[tuple[str, str], AliasEntry] = {}
        self._builtin_aliases: set[str] = set()

        source_aliases = builtin_aliases if builtin_aliases is not None else self._DEFAULT_BUILTIN_ALIASES
        self._load_builtin_aliases(source_aliases)

    def resolve(self, token: str) -> str:
        normalized_token = self._normalize_token(token)
        if not normalized_token:
            return ""
        entry = self._active_entries.get(normalized_token)
        if entry is None:
            return normalized_token
        return entry.canonical

    def get_alias_map(self) -> dict[str, str]:
        return {alias: entry.canonical for alias, entry in self._active_entries.items()}

    def register_candidate(self, alias: str, canonical: str, confidence: float) -> None:
        alias_norm = self._normalize_token(alias)
        canonical_norm = self._normalize_token(canonical)
        if not alias_norm or not canonical_norm:
            return
        if alias_norm == canonical_norm:
            return

        confidence_norm = self._clamp_confidence(confidence)
        active = self._active_entries.get(alias_norm)
        if active is not None and active.canonical == canonical_norm:
            self._merge_signal_into_entry(active, confidence_norm, evidence_increment=1)
            return

        key = (alias_norm, canonical_norm)
        candidate = self._candidate_entries.get(key)
        if candidate is None:
            self._candidate_entries[key] = AliasEntry(
                alias=alias_norm,
                canonical=canonical_norm,
                confidence=confidence_norm,
                evidence_count=1,
                source="learned",
                updated_at=self._now_iso(),
            )
            return

        self._merge_signal_into_entry(candidate, confidence_norm, evidence_increment=1)

    def promote_candidates(self, min_evidence: int = 3, min_confidence: float = 0.8) -> None:
        min_evidence = max(1, int(min_evidence))
        min_confidence = self._clamp_confidence(min_confidence)

        eligible_by_alias: dict[str, list[AliasEntry]] = {}
        for entry in self._candidate_entries.values():
            if entry.evidence_count < min_evidence:
                continue
            if entry.confidence < min_confidence:
                continue
            eligible_by_alias.setdefault(entry.alias, []).append(entry)

        promoted_aliases: set[str] = set()
        for alias, candidates in eligible_by_alias.items():
            best = max(candidates, key=self._candidate_rank)
            if self._try_promote(alias=alias, candidate=best):
                promoted_aliases.add(alias)

        if not promoted_aliases:
            return

        self._candidate_entries = {
            key: entry
            for key, entry in self._candidate_entries.items()
            if entry.alias not in promoted_aliases
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "active_entries": [asdict(entry) for entry in self._active_entries.values()],
            "candidate_entries": [asdict(entry) for entry in self._candidate_entries.values()],
        }

    def load_from_dict(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return

        active_entries: dict[str, AliasEntry] = {}
        builtin_aliases: set[str] = set()

        for item in self._extract_entry_rows(data, keys=("active_entries", "active", "aliases", "alias_map")):
            entry = self._entry_from_row(item, default_source="imported")
            if entry is None:
                continue
            active_entries[entry.alias] = entry
            if entry.source == "builtin":
                builtin_aliases.add(entry.alias)

        if active_entries:
            self._active_entries = active_entries
            self._builtin_aliases = builtin_aliases

        candidate_entries: dict[tuple[str, str], AliasEntry] = {}
        for item in self._extract_entry_rows(data, keys=("candidate_entries", "candidates")):
            entry = self._entry_from_row(item, default_source="learned")
            if entry is None:
                continue
            candidate_entries[(entry.alias, entry.canonical)] = entry

        self._candidate_entries = candidate_entries

    def _try_promote(self, *, alias: str, candidate: AliasEntry) -> bool:
        existing = self._active_entries.get(alias)
        if existing is None:
            self._active_entries[alias] = AliasEntry(
                alias=candidate.alias,
                canonical=candidate.canonical,
                confidence=candidate.confidence,
                evidence_count=candidate.evidence_count,
                source="learned",
                updated_at=self._now_iso(),
            )
            return True

        if existing.source == "builtin" and existing.canonical != candidate.canonical:
            return False

        if existing.canonical == candidate.canonical:
            self._merge_signal_into_entry(
                existing,
                incoming_confidence=candidate.confidence,
                evidence_increment=candidate.evidence_count,
            )
            return True

        if existing.source != "builtin" and self._candidate_rank(candidate) > self._candidate_rank(existing):
            self._active_entries[alias] = AliasEntry(
                alias=candidate.alias,
                canonical=candidate.canonical,
                confidence=candidate.confidence,
                evidence_count=candidate.evidence_count,
                source="learned",
                updated_at=self._now_iso(),
            )
            return True

        return False

    def _load_builtin_aliases(self, builtin_aliases: dict[str, str]) -> None:
        for raw_alias, raw_canonical in builtin_aliases.items():
            alias = self._normalize_token(raw_alias)
            canonical = self._normalize_token(raw_canonical)
            if not alias or not canonical:
                continue
            if alias == canonical:
                continue

            self._active_entries[alias] = AliasEntry(
                alias=alias,
                canonical=canonical,
                confidence=1.0,
                evidence_count=1,
                source="builtin",
                updated_at=self._now_iso(),
            )
            self._builtin_aliases.add(alias)

    @classmethod
    def _entry_from_row(cls, row: Any, default_source: str) -> AliasEntry | None:
        if isinstance(row, dict) and "alias" in row and "canonical" in row:
            alias = cls._normalize_token(str(row.get("alias") or ""))
            canonical = cls._normalize_token(str(row.get("canonical") or ""))
            if not alias or not canonical or alias == canonical:
                return None
            source = str(row.get("source") or default_source).strip().lower()
            if source not in cls._VALID_SOURCES:
                source = default_source
            confidence = cls._clamp_confidence(row.get("confidence", 0.85))
            evidence_count = cls._coerce_evidence(row.get("evidence_count", 1))
            updated_at = str(row.get("updated_at") or "")
            return AliasEntry(
                alias=alias,
                canonical=canonical,
                confidence=confidence,
                evidence_count=evidence_count,
                source=source,
                updated_at=updated_at,
            )
        return None

    @classmethod
    def _extract_entry_rows(cls, data: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in keys:
            if key not in data:
                continue
            payload = data.get(key)
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        rows.append(item)
            elif isinstance(payload, dict):
                for alias, canonical in payload.items():
                    rows.append({"alias": alias, "canonical": canonical})
        return rows

    @staticmethod
    def _candidate_rank(entry: AliasEntry) -> tuple[int, float]:
        return (int(entry.evidence_count), float(entry.confidence))

    @staticmethod
    def _merge_signal_into_entry(
        entry: AliasEntry,
        incoming_confidence: float,
        evidence_increment: int,
    ) -> None:
        old_evidence = max(1, int(entry.evidence_count))
        increment = max(1, int(evidence_increment))
        new_evidence = old_evidence + increment
        weighted_conf = ((entry.confidence * old_evidence) + (incoming_confidence * increment)) / new_evidence
        entry.evidence_count = new_evidence
        entry.confidence = round(AliasMemory._clamp_confidence(weighted_conf), 4)
        entry.updated_at = AliasMemory._now_iso()

    @staticmethod
    def _normalize_token(value: str) -> str:
        return normalize_text(str(value or ""))

    @staticmethod
    def _coerce_evidence(value: Any) -> int:
        try:
            return max(1, int(value))
        except Exception:
            return 1

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 0.0
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return round(score, 4)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

