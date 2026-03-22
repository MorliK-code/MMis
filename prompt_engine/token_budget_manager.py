from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenBudget:
    total: int = 2048
    reserve: int = 168
    system_core: int = 220
    personality: int = 180
    rules: int = 180
    user_profile: int = 120
    metadata: int = 80
    tools_state: int = 120
    memory_native_state: int = 280
    recent_chat: int = 600
    memory_retrieval: int = 400
    long_summary: int = 180
    output: int = 120

    @property
    def system(self) -> int:
        return self.system_core + self.personality + self.rules + self.user_profile + self.metadata + self.tools_state

    @property
    def memory(self) -> int:
        return self.memory_native_state + self.memory_retrieval

    @property
    def history(self) -> int:
        return self.recent_chat + self.long_summary

    @property
    def user(self) -> int:
        # Keep a hard floor for user message.
        return 240


@dataclass
class ContextBlock:
    id: str
    content: str
    bucket: str
    priority: int = 50
    max_tokens: int = 0
    min_tokens: int = 0
    shrink_strategy: str = "truncate_tail"
    required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenBudgetManager:
    """Token economy manager with context buckets and reserve budget for model output."""

    def __init__(self, budget: TokenBudget | None = None, chars_per_token: float = 4.0):
        self.budget = budget or TokenBudget()
        self.chars_per_token = max(1.0, float(chars_per_token))

    def count(self, text: str) -> int:
        src = str(text or "")
        if not src:
            return 0
        return max(1, int(len(src) / self.chars_per_token))

    def truncate(self, text: str, limit_tokens: int, strategy: str = "truncate_tail") -> str:
        src = str(text or "")
        limit = max(1, int(limit_tokens))
        if self.count(src) <= limit:
            return src

        mode = str(strategy or "truncate_tail").strip().lower()
        max_chars = max(8, int(limit * self.chars_per_token))
        if mode == "truncate_head":
            clipped = src[-max_chars:]
            return "... " + clipped.lstrip() if len(clipped) < len(src) else clipped
        if mode == "summarize":
            return self._summarize_text(src, limit)
        clipped = src[:max_chars]
        return clipped.rstrip() + " ..." if len(clipped) < len(src) else clipped

    def fit_blocks(self, blocks: dict[str, str], limits: dict[str, int] | None = None) -> dict[str, str]:
        """Backward-compatible path for old call sites."""
        mapped = [
            ContextBlock(id="system", content=str(blocks.get("system") or ""), bucket="system", priority=95, required=True),
            ContextBlock(
                id="memory",
                content=str(blocks.get("memory") or ""),
                bucket="memory",
                priority=72,
                shrink_strategy="summarize",
                min_tokens=max(24, int(self.budget.memory_retrieval * 0.25)),
            ),
            ContextBlock(id="history", content=str(blocks.get("history") or ""), bucket="history", priority=62, shrink_strategy="summarize"),
            ContextBlock(id="output", content=str(blocks.get("output") or ""), bucket="rules", priority=85),
            ContextBlock(id="user", content=str(blocks.get("user") or ""), bucket="user", priority=100, required=True),
        ]
        out, _stats = self.fit_context_blocks(mapped, limits=limits)
        return out

    def fit_context_blocks(
        self,
        blocks: list[ContextBlock],
        *,
        limits: dict[str, int] | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        budget = self._bucket_limits()
        if isinstance(limits, dict):
            for key, value in limits.items():
                try:
                    budget[str(key)] = max(1, int(value))
                except Exception:
                    continue

        usable_total = max(128, int(self.budget.total - self.budget.reserve))
        out: dict[str, str] = {}
        stats = {
            "usable_total": usable_total,
            "reserve": int(self.budget.reserve),
            "token_usage": {},
            "dropped": [],
            "trimmed": [],
        }

        working: list[dict[str, Any]] = []
        for block in list(blocks or []):
            if not isinstance(block, ContextBlock):
                continue
            content = str(block.content or "").strip()
            if not content:
                out[block.id] = ""
                continue
            bucket_limit = int(budget.get(block.bucket, max(16, usable_total // max(1, len(blocks) or 1))))
            hard_cap = int(block.max_tokens) if int(block.max_tokens or 0) > 0 else bucket_limit
            hard_cap = max(1, min(bucket_limit, hard_cap))
            trimmed = self.truncate(content, hard_cap, strategy=block.shrink_strategy)
            row = {
                "block": block,
                "content": trimmed,
                "tokens": self.count(trimmed),
                "min_tokens": max(0, int(block.min_tokens or 0)),
            }
            working.append(row)
            out[block.id] = trimmed
            stats["token_usage"][block.id] = row["tokens"]
            if trimmed != content:
                stats["trimmed"].append(block.id)

        def _total_tokens() -> int:
            return sum(int(x.get("tokens") or 0) for x in working)

        while _total_tokens() > usable_total:
            candidate = self._pick_shrink_candidate(working)
            if candidate is None:
                break
            idx = candidate
            row = working[idx]
            block: ContextBlock = row["block"]
            current = str(row["content"] or "")
            current_tokens = int(row["tokens"] or 0)
            if current_tokens <= int(row["min_tokens"]):
                # If cannot shrink below min, consider dropping if allowed.
                if block.required or str(block.shrink_strategy or "").strip().lower() != "drop":
                    working[idx]["tokens"] = current_tokens
                    break
                working[idx]["content"] = ""
                working[idx]["tokens"] = 0
                out[block.id] = ""
                stats["dropped"].append(block.id)
                stats["token_usage"][block.id] = 0
                continue

            target_tokens = max(int(row["min_tokens"]), int(current_tokens * 0.82))
            if target_tokens >= current_tokens:
                target_tokens = max(int(row["min_tokens"]), current_tokens - 8)
            if target_tokens <= 0:
                target_tokens = max(1, int(row["min_tokens"]))

            shrunk = self.truncate(current, target_tokens, strategy=block.shrink_strategy)
            new_tokens = self.count(shrunk)
            if new_tokens >= current_tokens:
                # Force drop for low-priority drop-strategy blocks.
                if not block.required and str(block.shrink_strategy or "").strip().lower() == "drop":
                    shrunk = ""
                    new_tokens = 0
                    stats["dropped"].append(block.id)
                else:
                    break

            working[idx]["content"] = shrunk
            working[idx]["tokens"] = new_tokens
            out[block.id] = shrunk
            stats["token_usage"][block.id] = new_tokens
            if block.id not in stats["trimmed"]:
                stats["trimmed"].append(block.id)

        stats["total_tokens"] = _total_tokens()
        return out, stats

    def apply_verbosity(self, verbosity_level: float) -> dict[str, int]:
        """Scale selected bucket limits by dialog verbosity (0..1)."""
        level = max(0.0, min(1.0, float(verbosity_level)))
        return {
            "user_profile": max(60, int(self.budget.user_profile * (0.8 + (0.6 * level)))),
            "history": max(120, int(self.budget.recent_chat * (0.62 + (0.78 * level)))),
            "long_summary": max(80, int(self.budget.long_summary * (0.7 + (0.7 * level)))),
            "memory_native_state": max(140, int(self.budget.memory_native_state * (0.5 + (0.5 * level)))),
            "output": max(70, int(self.budget.output * (0.7 + (1.0 * level)))),
        }

    def _bucket_limits(self) -> dict[str, int]:
        return {
            "system": int(self.budget.system_core),
            "personality": int(self.budget.personality),
            "rules": int(self.budget.rules),
            "user_profile": int(self.budget.user_profile),
            "metadata": int(self.budget.metadata),
            "tools": int(self.budget.tools_state),
            "memory": int(self.budget.memory_native_state + self.budget.memory_retrieval),
            "memory_native_state": int(self.budget.memory_native_state),
            "history": int(self.budget.recent_chat),
            "long_summary": int(self.budget.long_summary),
            "output": int(self.budget.output),
            "user": int(self.budget.user),
        }

    def _pick_shrink_candidate(self, working: list[dict[str, Any]]) -> int | None:
        ranked: list[tuple[float, int]] = []
        for idx, row in enumerate(working):
            block: ContextBlock = row["block"]
            tokens = int(row.get("tokens") or 0)
            if tokens <= 0:
                continue
            min_tokens = int(row.get("min_tokens") or 0)
            if tokens <= min_tokens and str(block.shrink_strategy or "").strip().lower() != "drop":
                continue
            priority = max(1, min(100, int(block.priority or 50)))
            if block.required:
                # Required blocks can still be trimmed, but with lower preference.
                priority = min(100, priority + 20)
            score = float(tokens) / float(priority)
            ranked.append((score, idx))

        if not ranked:
            return None
        ranked.sort(reverse=True)
        return ranked[0][1]

    def _summarize_text(self, text: str, limit_tokens: int) -> str:
        src = str(text or "").strip()
        if not src:
            return ""
        lines = [x.strip() for x in src.replace("\r", "\n").split("\n") if x.strip()]
        if not lines:
            return self.truncate(src, limit_tokens, strategy="truncate_tail")

        keep_head = lines[:4]
        keep_tail = lines[-2:] if len(lines) > 6 else []
        merged = keep_head + (["..."] if keep_tail else []) + keep_tail
        summary = "\n".join(merged)
        if self.count(summary) <= limit_tokens:
            return summary
        return self.truncate(summary, limit_tokens, strategy="truncate_tail")
