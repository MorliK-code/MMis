"""Conflict governor for fact memory rows."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from memory.memory_lifecycle import MemoryLifecycleManager
from memory.memory_models import MemoryRecord, MemoryStatus, MemoryType
from memory.memory_policy import MemoryPolicy

FACT_GROUPS = {
    "identity_name": "identity.name",
    "identity_age_years": "identity.age",
    "environment_os": "environment.os",
    "environment_runtime_python": "environment.python",
    "environment_ram_gb": "environment.ram",
    "environment_memory_gb": "environment.ram",
    "environment_gpu_model": "environment.gpu_model",
    "environment_gpu_vram_gb": "environment.gpu_vram",
    "preferred_editor": "preferences.editor",
    "preferred_language": "preferences.language",
    "preferred_tool": "preferences.tool",
    "interest": "profile.interests",
    "project_name": "project.name",
}

GROUP_POLICY = {
    "identity.name": "singleton",
    "identity.age": "singleton",
    "environment.os": "singleton",
    "environment.python": "singleton",
    "environment.ram": "singleton",
    "environment.gpu_model": "singleton",
    "environment.gpu_vram": "singleton",
    "preferences.editor": "singleton",
    "preferences.language": "soft_singleton",
    "preferences.tool": "multi",
    "profile.interests": "multi",
    "project.name": "multi",
}


@dataclass(frozen=True)
class GovernorDecision:
    action: str  # keep_active | supersede_old | keep_parallel | archive_old | noop
    reason: str
    winner_record_id: str | None = None
    loser_record_id: str | None = None
    parallel_with_record_id: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GovernorProfileSnapshot:
    namespace: str
    active_facts: dict[str, dict[str, Any]]
    conflicts: list[dict[str, Any]]
    updated_at: float


class MemoryGovernor:
    """Govern conflict decisions for active fact memory.

    This is intentionally a thin foundation layer: it only reasons about fact
    rows that already exist and does not write to storage on its own.
    """

    def __init__(
        self,
        *,
        parallel_margin: float = 0.03,
        lifecycle: MemoryLifecycleManager | None = None,
    ):
        self._parallel_margin = max(0.0, float(parallel_margin))
        self._lifecycle = lifecycle or MemoryLifecycleManager(parallel_margin=self._parallel_margin)

    def decide_for_fact(
        self,
        *,
        new_record: MemoryRecord,
        active_candidates: list[MemoryRecord],
    ) -> GovernorDecision:
        if new_record.memory_type != MemoryType.FACT:
            return GovernorDecision(
                action="noop",
                reason="new_record_not_fact",
                debug={"memory_type": str(new_record.memory_type.value)},
            )

        candidates = [
            row for row in list(active_candidates or [])
            if row.memory_type == MemoryType.FACT and row.status == MemoryStatus.ACTIVE
        ]
        if not candidates:
            return GovernorDecision(
                action="keep_active",
                reason="no_active_conflict",
                winner_record_id=str(new_record.id or "") or None,
                debug={"candidate_count": 0},
            )

        new_fact = self._fact_view(new_record)
        comparable = [
            row for row in candidates
            if self._same_canonical(new_fact, self._fact_view(row))
            or self._same_group(new_fact, self._fact_view(row))
        ]
        if not comparable:
            incumbent = self._best_record(candidates)
            return GovernorDecision(
                action="keep_active",
                reason="different_canonical_or_group",
                winner_record_id=str(new_record.id or "") or None,
                parallel_with_record_id=str(incumbent.id or "") or None,
                debug={
                    "candidate_count": len(candidates),
                    "new_canonical_key": new_fact["canonical_key"],
                    "new_group": new_fact["group"],
                },
            )

        if new_fact["group_mode"] == "multi" and not any(
            self._same_canonical(new_fact, self._fact_view(row)) for row in comparable
        ):
            incumbent = self._best_record(comparable)
            return GovernorDecision(
                action="keep_parallel",
                reason="multi_value_group_allowed",
                winner_record_id=str(new_record.id or "") or None,
                parallel_with_record_id=str(incumbent.id or "") or None,
                debug={
                    "group": new_fact["group"],
                    "group_mode": new_fact["group_mode"],
                    "candidate_count": len(comparable),
                },
            )

        incumbent = self._best_record(comparable)
        incumbent_fact = self._fact_view(incumbent)
        legacy_decision = self._lifecycle.resolve_conflict(old=incumbent, new=new_record)
        new_conf = self._coerce_float(new_fact["confidence"] or new_record.confidence)
        incumbent_conf = self._coerce_float(incumbent_fact["confidence"] or incumbent.confidence)

        if new_fact["value_norm"] and new_fact["value_norm"] == incumbent_fact["value_norm"]:
            if new_conf > incumbent_conf:
                return GovernorDecision(
                    action="supersede_old",
                    reason="same_value_higher_confidence",
                    winner_record_id=str(new_record.id or "") or None,
                    loser_record_id=str(incumbent.id or "") or None,
                    debug={
                        "new_conf": new_conf,
                        "old_conf": incumbent_conf,
                        "group": new_fact["group"],
                        "group_mode": new_fact["group_mode"],
                        "legacy_conflict": self._conflict_to_debug(legacy_decision),
                    },
                )
            return GovernorDecision(
                action="noop",
                reason="same_value_existing_kept",
                winner_record_id=str(incumbent.id or "") or None,
                loser_record_id=str(new_record.id or "") or None,
                debug={
                    "new_conf": new_conf,
                    "old_conf": incumbent_conf,
                    "group": new_fact["group"],
                    "group_mode": new_fact["group_mode"],
                    "legacy_conflict": self._conflict_to_debug(legacy_decision),
                },
            )

        if new_fact["group_mode"] == "multi":
            return GovernorDecision(
                action="keep_parallel",
                reason="multi_value_group_allowed",
                winner_record_id=str(new_record.id or "") or None,
                parallel_with_record_id=str(incumbent.id or "") or None,
                debug={
                    "group": new_fact["group"],
                    "group_mode": new_fact["group_mode"],
                    "new_value": new_fact["value"],
                    "old_value": incumbent_fact["value"],
                    "legacy_conflict": self._conflict_to_debug(legacy_decision),
                },
            )

        if new_fact["group_mode"] in {"singleton", "soft_singleton"}:
            return self._decide_singleton_group_conflict(
                new_record=new_record,
                incumbent=incumbent,
                new_fact=new_fact,
                incumbent_fact=incumbent_fact,
                legacy_score_delta=float(legacy_decision.score_delta or 0.0),
            )

        return self._map_legacy_conflict(
            new_record=new_record,
            incumbent=incumbent,
            legacy_decision=legacy_decision,
            new_fact=new_fact,
        )

    def rebuild_profile_snapshot(
        self,
        *,
        namespace: str,
        active_fact_rows: list[MemoryRecord],
    ) -> GovernorProfileSnapshot:
        namespace_norm = str(namespace or "default")
        active_rows = [
            row for row in list(active_fact_rows or [])
            if (
                row.memory_type == MemoryType.FACT
                and row.status == MemoryStatus.ACTIVE
                and str(row.namespace or "default") == namespace_norm
            )
        ]
        active_facts: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[dict[str, Any]]] = {}

        for row in active_rows:
            fact = self._fact_view(row)
            key = str(fact["canonical_key"] or row.id or "").strip() or str(row.id or "")
            if key in active_facts:
                key = f"{key}#{row.id}"
            active_facts[key] = {
                "record_id": str(row.id or ""),
                "canonical_key": str(fact["canonical_key"] or ""),
                "group": str(fact["group"] or ""),
                "group_mode": str(fact["group_mode"] or ""),
                "predicate": str(fact["predicate"] or ""),
                "value": fact["value"],
                "confidence": self._coerce_float(fact["confidence"] or row.confidence),
                "status": str(row.status.value),
            }
            if str(fact["group"] or "").strip() and str(fact["group_mode"] or "") == "singleton":
                grouped.setdefault(str(fact["group"] or ""), []).append(active_facts[key])

        conflicts: list[dict[str, Any]] = []
        for group, rows in grouped.items():
            values = {
                str(item.get("value") or "").strip().lower()
                for item in list(rows or [])
                if str(item.get("value") or "").strip()
            }
            if len(rows) <= 1 or len(values) <= 1:
                continue
            conflicts.append(
                {
                    "group": str(group or ""),
                    "record_ids": [str(item.get("record_id") or "") for item in list(rows or [])],
                    "values": [item.get("value") for item in list(rows or [])],
                }
            )

        return GovernorProfileSnapshot(
            namespace=namespace_norm,
            active_facts=active_facts,
            conflicts=conflicts,
            updated_at=float(time.time()),
        )

    @staticmethod
    def _best_record(rows: list[MemoryRecord]) -> MemoryRecord:
        return max(
            list(rows or []),
            key=lambda row: (
                MemoryGovernor._coerce_float(row.confidence),
                MemoryGovernor._coerce_float(row.importance),
                MemoryGovernor._coerce_float(row.updated_at),
            ),
        )

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _same_canonical(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return bool(left["canonical_key"]) and left["canonical_key"] == right["canonical_key"]

    @staticmethod
    def _same_group(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return bool(left["group"]) and left["group"] == right["group"]

    @staticmethod
    def _fact_view(record: MemoryRecord) -> dict[str, Any]:
        meta = dict(record.metadata or {})
        fact = dict(meta.get("fact") or {})
        predicate = str(fact.get("predicate") or "").strip().lower()
        value = fact.get("value")
        canonical_key = str(meta.get("canonical_key") or fact.get("canonical_key") or "").strip().lower()
        group, group_mode = MemoryGovernor._group_profile(predicate)
        return {
            "canonical_key": canonical_key,
            "predicate": predicate,
            "group": group,
            "group_mode": group_mode,
            "value": value,
            "value_norm": str(value or "").strip().lower(),
            "confidence": fact.get("confidence"),
        }

    @classmethod
    def _group_profile(cls, predicate: str) -> tuple[str, str]:
        pred = str(predicate or "").strip().lower()
        group = str(FACT_GROUPS.get(pred) or "").strip()
        if not group:
            legacy_group = str(MemoryPolicy.fact_group(pred) or pred).strip()
            if MemoryPolicy.is_singleton_group(legacy_group):
                return legacy_group, "singleton"
            return legacy_group, "multi"
        mode = str(GROUP_POLICY.get(group) or "").strip().lower()
        if mode in {"singleton", "soft_singleton", "multi"}:
            return group, mode
        return group, "multi"

    def _decide_singleton_group_conflict(
        self,
        *,
        new_record: MemoryRecord,
        incumbent: MemoryRecord,
        new_fact: dict[str, Any],
        incumbent_fact: dict[str, Any],
        legacy_score_delta: float,
    ) -> GovernorDecision:
        if new_fact["group_mode"] == "soft_singleton" and abs(float(legacy_score_delta)) <= (self._parallel_margin * 1.5):
            return GovernorDecision(
                action="keep_parallel",
                reason="soft_singleton_scores_close_parallel",
                winner_record_id=str(new_record.id or "") or None,
                parallel_with_record_id=str(incumbent.id or "") or None,
                debug={
                    "group": new_fact["group"],
                    "group_mode": new_fact["group_mode"],
                    "legacy_score_delta": float(legacy_score_delta),
                    "new_value": new_fact["value"],
                    "old_value": incumbent_fact["value"],
                },
            )
        if float(legacy_score_delta) > 0.0:
            return GovernorDecision(
                action="supersede_old",
                reason="singleton_group_higher_score",
                winner_record_id=str(new_record.id or "") or None,
                loser_record_id=str(incumbent.id or "") or None,
                debug={
                    "group": new_fact["group"],
                    "group_mode": new_fact["group_mode"],
                    "legacy_score_delta": float(legacy_score_delta),
                    "new_value": new_fact["value"],
                    "old_value": incumbent_fact["value"],
                },
            )
        return GovernorDecision(
            action="noop",
            reason="singleton_group_existing_kept",
            winner_record_id=str(incumbent.id or "") or None,
            loser_record_id=str(new_record.id or "") or None,
            debug={
                "group": new_fact["group"],
                "group_mode": new_fact["group_mode"],
                "legacy_score_delta": float(legacy_score_delta),
                "new_value": new_fact["value"],
                "old_value": incumbent_fact["value"],
            },
        )

    def _map_legacy_conflict(
        self,
        *,
        new_record: MemoryRecord,
        incumbent: MemoryRecord,
        legacy_decision: Any,
        new_fact: dict[str, Any],
    ) -> GovernorDecision:
        action = str(getattr(legacy_decision, "action", "") or "").strip().lower()
        debug = {
            "group": new_fact["group"],
            "group_mode": new_fact["group_mode"],
            "legacy_conflict": self._conflict_to_debug(legacy_decision),
        }
        if action == "parallel":
            return GovernorDecision(
                action="keep_parallel",
                reason=str(getattr(legacy_decision, "reason", "") or "legacy_parallel"),
                winner_record_id=str(new_record.id or "") or None,
                parallel_with_record_id=(
                    str(getattr(legacy_decision, "parallel_with_record_id", "") or incumbent.id or "") or None
                ),
                debug=debug,
            )
        if action == "archive" and str(getattr(legacy_decision, "archive_record_id", "") or "") == str(incumbent.id or ""):
            return GovernorDecision(
                action="archive_old",
                reason=str(getattr(legacy_decision, "reason", "") or "legacy_archive_old"),
                winner_record_id=str(getattr(legacy_decision, "keep_record_id", "") or new_record.id or "") or None,
                loser_record_id=str(getattr(legacy_decision, "archive_record_id", "") or incumbent.id or "") or None,
                debug=debug,
            )
        if action == "supersede" and str(getattr(legacy_decision, "keep_record_id", "") or "") == str(new_record.id or ""):
            return GovernorDecision(
                action="supersede_old",
                reason=str(getattr(legacy_decision, "reason", "") or "legacy_supersede_old"),
                winner_record_id=str(new_record.id or "") or None,
                loser_record_id=str(getattr(legacy_decision, "superseded_record_id", "") or incumbent.id or "") or None,
                debug=debug,
            )
        return GovernorDecision(
            action="noop",
            reason=str(getattr(legacy_decision, "reason", "") or "legacy_existing_kept"),
            winner_record_id=str(incumbent.id or "") or None,
            loser_record_id=str(new_record.id or "") or None,
            debug=debug,
        )

    @staticmethod
    def _conflict_to_debug(conflict: Any) -> dict[str, Any]:
        return {
            "action": str(getattr(conflict, "action", "") or ""),
            "reason": str(getattr(conflict, "reason", "") or ""),
            "keep_record_id": str(getattr(conflict, "keep_record_id", "") or ""),
            "superseded_record_id": str(getattr(conflict, "superseded_record_id", "") or ""),
            "archive_record_id": str(getattr(conflict, "archive_record_id", "") or ""),
            "parallel_with_record_id": str(getattr(conflict, "parallel_with_record_id", "") or ""),
            "score_delta": float(getattr(conflict, "score_delta", 0.0) or 0.0),
        }
