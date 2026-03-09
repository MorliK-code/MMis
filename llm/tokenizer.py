from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from llm.provider_base import Message


class Tokenizer(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        raise NotImplementedError

    def count_messages(self, messages: list[Message]) -> int:
        total = 0
        for msg in list(messages or []):
            total += self.count(str(msg.content or ""))
            if msg.name:
                total += self.count(str(msg.name))
            if msg.tool_call_id:
                total += self.count(str(msg.tool_call_id))
            total += 2
        return total

    @abstractmethod
    def truncate_text(self, text: str, max_tokens: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def truncate_messages(self, messages: list[Message], max_tokens: int, strategy: str = "drop_oldest") -> list[Message]:
        raise NotImplementedError


@dataclass(frozen=True)
class ApproxTokenizer(Tokenizer):
    chars_per_token: float = 3.8

    def count(self, text: str) -> int:
        src = str(text or "")
        if not src:
            return 0
        base = int(len(src) / max(1.0, float(self.chars_per_token)))
        lines_bonus = max(0, src.count("\n") // 4)
        return max(1, base + lines_bonus)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        src = str(text or "")
        limit = max(1, int(max_tokens))
        if self.count(src) <= limit:
            return src
        max_chars = max(8, int(limit * self.chars_per_token))
        piece = src[:max_chars]
        if len(piece) < len(src):
            split = max(piece.rfind("\n"), piece.rfind(". "), piece.rfind(" "))
            if split > 24:
                piece = piece[:split]
            piece = piece.rstrip() + " ..."
        return piece.strip()

    def truncate_messages(self, messages: list[Message], max_tokens: int, strategy: str = "drop_oldest") -> list[Message]:
        rows = list(messages or [])
        limit = max(16, int(max_tokens))
        if self.count_messages(rows) <= limit:
            return rows

        mode = str(strategy or "drop_oldest").strip().lower()
        if mode == "compress_middle":
            return self._truncate_compress_middle(rows, limit)
        if mode == "preserve_system":
            return self._truncate_preserve_system(rows, limit)
        return self._truncate_drop_oldest(rows, limit)

    def _truncate_drop_oldest(self, rows: list[Message], limit: int) -> list[Message]:
        out = list(rows)
        while len(out) > 1 and self.count_messages(out) > limit:
            out.pop(0)
        if out and self.count_messages(out) > limit:
            last = out[-1]
            out[-1] = Message(
                role=last.role,
                content=self.truncate_text(last.content, max(8, limit // 2)),
                name=last.name,
                tool_call_id=last.tool_call_id,
            )
        return out

    def _truncate_preserve_system(self, rows: list[Message], limit: int) -> list[Message]:
        out = list(rows)
        first_system_idx = -1
        for idx, msg in enumerate(out):
            if msg.role == "system":
                first_system_idx = idx
                break
        while len(out) > 1 and self.count_messages(out) > limit:
            drop_idx = 0
            if first_system_idx == 0 and len(out) > 1:
                drop_idx = 1
            out.pop(drop_idx)
        if out and self.count_messages(out) > limit:
            out[-1] = Message(role=out[-1].role, content=self.truncate_text(out[-1].content, max(8, limit // 2)))
        return out

    def _truncate_compress_middle(self, rows: list[Message], limit: int) -> list[Message]:
        out = list(rows)
        if len(out) <= 3:
            return self._truncate_drop_oldest(out, limit)
        while len(out) > 3 and self.count_messages(out) > limit:
            mid = max(1, len(out) // 2)
            dropped = out.pop(mid)
            if mid - 1 >= 0:
                prev = out[mid - 1]
                compressed = self.truncate_text(f"{prev.content}\n[...snip...]", max(12, self.count(prev.content) // 2))
                out[mid - 1] = Message(role=prev.role, content=compressed, name=prev.name, tool_call_id=prev.tool_call_id)
        if self.count_messages(out) > limit:
            return self._truncate_drop_oldest(out, limit)
        return out


@dataclass(frozen=True)
class BudgetBlock:
    block_id: str
    max_tokens: int
    min_tokens: int = 0
    priority: int = 50
    required: bool = False


def allocate_context_budgets(*, total_tokens: int, reserve_tokens: int, blocks: list[BudgetBlock]) -> dict[str, int]:
    usable = max(64, int(total_tokens) - max(0, int(reserve_tokens)))
    rows = [x for x in list(blocks or []) if isinstance(x, BudgetBlock)]
    if not rows:
        return {}

    allocated: dict[str, int] = {x.block_id: max(0, int(x.min_tokens)) for x in rows}
    used = sum(allocated.values())
    if used >= usable:
        return allocated

    ranked = sorted(rows, key=lambda x: (int(x.priority), int(x.max_tokens)), reverse=True)
    for block in ranked:
        if used >= usable:
            break
        current = int(allocated.get(block.block_id, 0))
        room = max(0, int(block.max_tokens) - current)
        if room <= 0:
            continue
        take = min(room, usable - used)
        allocated[block.block_id] = current + take
        used += take
    return allocated


def truncate_blocks(
    blocks: dict[str, str],
    *,
    block_budgets: dict[str, int],
    tokenizer: Tokenizer | None = None,
    overflow_strategy: str = "priority_drop",
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    tk = tokenizer or DEFAULT_TOKENIZER
    out: dict[str, str] = {}
    log: list[dict[str, Any]] = []

    for key, value in dict(blocks or {}).items():
        text = str(value or "")
        limit = max(0, int(block_budgets.get(key, 0)))
        if not text or limit <= 0:
            out[key] = ""
            continue
        clipped = tk.truncate_text(text, limit)
        if clipped != text:
            log.append({"block": key, "action": "truncate", "limit": limit})
        out[key] = clipped

    total = sum(tk.count(v) for v in out.values())
    if total <= sum(max(0, int(x)) for x in block_budgets.values()):
        return out, log

    mode = str(overflow_strategy or "priority_drop").strip().lower()
    if mode == "hard_truncate":
        for key in list(out.keys()):
            if not out[key]:
                continue
            out[key] = tk.truncate_text(out[key], max(8, tk.count(out[key]) // 2))
            log.append({"block": key, "action": "hard_truncate"})
    else:
        for key in sorted(out.keys(), key=lambda x: tk.count(out[x])):
            if total <= sum(max(0, int(x)) for x in block_budgets.values()):
                break
            if not out[key]:
                continue
            removed = tk.count(out[key])
            out[key] = ""
            total -= removed
            log.append({"block": key, "action": "drop_overflow", "removed_tokens": removed})

    return out, log


def create_tokenizer(*, chars_per_token: float | None = None) -> Tokenizer:
    return ApproxTokenizer(chars_per_token=float(chars_per_token or 3.8))


DEFAULT_TOKENIZER: Tokenizer = create_tokenizer()


def count(text: str) -> int:
    return DEFAULT_TOKENIZER.count(text)


def count_messages(messages: list[Message]) -> int:
    return DEFAULT_TOKENIZER.count_messages(messages)


def truncate_text(text: str, max_tokens: int) -> str:
    return DEFAULT_TOKENIZER.truncate_text(text, max_tokens)


def truncate_messages(messages: list[Message], max_tokens: int, strategy: str = "drop_oldest") -> list[Message]:
    return DEFAULT_TOKENIZER.truncate_messages(messages, max_tokens, strategy=strategy)


def estimate_tokens(text: str) -> int:
    return count(text)
