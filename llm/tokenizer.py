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
    chars_per_token: float = 4.0

    def count(self, text: str) -> int:
        clean = str(text or "")
        if not clean:
            return 0
        return max(1, int(len(clean) / max(1.0, float(self.chars_per_token))))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        content = str(text or "")
        limit = max(1, int(max_tokens))
        if self.count(content) <= limit:
            return content
        max_chars = max(8, int(limit * self.chars_per_token))
        clipped = content[:max_chars]
        if len(clipped) < len(content):
            cut = clipped.rsplit(" ", 1)[0].strip()
            clipped = (cut if cut else clipped.strip()) + " …"
        return clipped.strip()

    def truncate_messages(self, messages: list[Message], max_tokens: int, strategy: str = "drop_oldest") -> list[Message]:
        limit = max(16, int(max_tokens))
        rows = list(messages or [])
        if self.count_messages(rows) <= limit:
            return rows

        mode = str(strategy or "drop_oldest").strip().lower()
        if mode == "drop_oldest":
            return self._truncate_drop_oldest(rows, limit)
        if mode == "compress_summary":
            return self._truncate_compress_summary(rows, limit)
        if mode == "drop_low_priority_memories":
            return self._truncate_drop_oldest(rows, limit)
        return self._truncate_drop_oldest(rows, limit)

    def _truncate_drop_oldest(self, messages: list[Message], limit: int) -> list[Message]:
        rows = list(messages)
        while len(rows) > 1 and self.count_messages(rows) > limit:
            idx = self._oldest_non_system_index(rows)
            if idx < 0:
                rows = rows[1:]
            else:
                rows.pop(idx)
        if self.count_messages(rows) > limit and rows:
            last = rows[-1]
            rows[-1] = Message(
                role=last.role,
                content=self.truncate_text(last.content, max(1, limit // 2)),
                name=last.name,
                tool_call_id=last.tool_call_id,
            )
        return rows

    def _truncate_compress_summary(self, messages: list[Message], limit: int) -> list[Message]:
        rows = list(messages)
        for idx, msg in enumerate(rows):
            if msg.role == "system":
                rows[idx] = Message(
                    role=msg.role,
                    content=self.truncate_text(msg.content, max(24, limit // 5)),
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                )
                break
        return self._truncate_drop_oldest(rows, limit)

    @staticmethod
    def _oldest_non_system_index(rows: list[Message]) -> int:
        for i, msg in enumerate(rows):
            if msg.role != "system":
                return i
        return -1


def drop_low_priority_memories(
    memories: list[dict[str, Any]],
    *,
    tokenizer: Tokenizer | None = None,
    max_tokens: int = 400,
) -> list[dict[str, Any]]:
    tk = tokenizer or DEFAULT_TOKENIZER
    items = [dict(x) for x in list(memories or []) if isinstance(x, dict)]
    items.sort(
        key=lambda x: (
            float(x.get("priority", x.get("confidence", 0.5)) or 0.5),
            float(x.get("confidence", 0.5) or 0.5),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    used = 0
    limit = max(16, int(max_tokens))
    for row in items:
        text = str(row.get("text") or row.get("content") or row.get("fact") or "")
        tokens = tk.count(text)
        if used + tokens > limit:
            continue
        used += tokens
        selected.append(row)
    return selected


DEFAULT_TOKENIZER: Tokenizer = ApproxTokenizer()


def count(text: str) -> int:
    return DEFAULT_TOKENIZER.count(text)


def count_messages(messages: list[Message]) -> int:
    return DEFAULT_TOKENIZER.count_messages(messages)


def truncate_text(text: str, max_tokens: int) -> str:
    return DEFAULT_TOKENIZER.truncate_text(text, max_tokens)


def truncate_messages(messages: list[Message], max_tokens: int, strategy: str = "drop_oldest") -> list[Message]:
    return DEFAULT_TOKENIZER.truncate_messages(messages, max_tokens, strategy=strategy)


# Backward compatibility alias used in prompt_builder.
def estimate_tokens(text: str) -> int:
    return count(text)

