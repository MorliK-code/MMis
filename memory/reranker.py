from __future__ import annotations

from dataclasses import dataclass

from memory.memory_models import MemoryLevel, MemoryScope, RetrievalCandidate, RetrievalQuery


def _level_bonus(level: MemoryLevel) -> float:
    if level == MemoryLevel.L0_WORKING:
        return 0.12
    if level == MemoryLevel.L1_SESSION:
        return 0.08
    if level == MemoryLevel.L3_SEMANTIC:
        return 0.06
    if level == MemoryLevel.L4_DOCUMENT:
        return 0.02
    return 0.0


def _scope_bonus(scope: MemoryScope) -> float:
    # Temporary scope is short-lived but highly relevant for immediate turns.
    if scope == MemoryScope.TEMPORARY:
        return 0.08
    if scope == MemoryScope.SESSION:
        return 0.04
    if scope == MemoryScope.CONVERSATION:
        return 0.03
    return 0.0


@dataclass
class HeuristicReranker:
    diversity_penalty: float = 0.07

    def rerank(self, query: RetrievalQuery, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        _ = query
        rows = list(candidates or [])
        if not rows:
            return []

        seen_prefix: set[str] = set()
        scored: list[tuple[float, RetrievalCandidate]] = []
        for item in rows:
            prefix = str(item.record.id or "").split(":", 1)[0]
            bonus = _level_bonus(item.record.level)
            bonus += _scope_bonus(item.record.scope)
            penalty = self.diversity_penalty if prefix in seen_prefix else 0.0
            score = float(item.final_score) + float(bonus) - float(penalty)
            scored.append((score, item))
            seen_prefix.add(prefix)

        scored.sort(key=lambda x: float(x[0]), reverse=True)
        return [item for _, item in scored[: max(1, int(top_k))]]
