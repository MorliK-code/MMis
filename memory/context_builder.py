from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm.tokenizer import Tokenizer, create_tokenizer
from memory.memory_models import (
    ContextBuildRequest,
    ContextBuildResult,
    MemoryLevel,
    MemoryRecord,
    RetrievalCandidate,
)


@dataclass
class ContextBuilderV2:
    tokenizer: Tokenizer

    @classmethod
    def from_settings(cls) -> "ContextBuilderV2":
        return cls(tokenizer=create_tokenizer())

    def build(
        self,
        *,
        request: ContextBuildRequest,
        retrieved: list[RetrievalCandidate],
        private_runtime_state: dict[str, Any] | None = None,
    ) -> ContextBuildResult:
        private_runtime_state = dict(private_runtime_state or {})
        selected = list(retrieved or [])

        grouped = self._group_candidates(selected)
        working_block = self._render_working_memory(request.working_memory)
        session_block = str(request.session_summary or "").strip()
        semantic_block = self._render_candidates(grouped["semantic"])
        episodic_block = self._render_candidates(grouped["episodic"])
        docs_block = self._render_candidates(grouped["docs"])
        tools_block = self._render_tools(request.tool_state, private_runtime_state)
        unresolved_block = self._render_unresolved(request.unresolved_items)

        blocks: dict[str, str] = {
            "system_core": str(request.system_prompt or "").strip(),
            "user_message": str(request.user_message or "").strip(),
            "working_memory": working_block,
            "session_summary": session_block,
            "retrieved_semantic": semantic_block,
            "retrieved_episodic": episodic_block,
            "retrieved_docs": docs_block,
            "active_tool_state": tools_block,
            "unresolved_items": unresolved_block,
        }

        dropped: list[dict[str, Any]] = []
        log: list[dict[str, Any]] = []

        # Step 1: keep system and current user message as hard requirements.
        mandatory_tokens = self._count(blocks["system_core"]) + self._count(blocks["user_message"])
        usable = max(256, int(request.context_budget_total) - int(request.context_budget_response_reserve))

        # Step 2: enforce per-bucket soft limits before global cascade.
        blocks["working_memory"] = self._clip_to_budget(blocks["working_memory"], max(64, request.context_budget_memory // 4))
        blocks["session_summary"] = self._clip_to_budget(blocks["session_summary"], max(64, request.context_budget_memory // 4))
        blocks["retrieved_semantic"] = self._clip_to_budget(blocks["retrieved_semantic"], max(96, request.context_budget_memory // 3))
        blocks["retrieved_episodic"] = self._clip_to_budget(blocks["retrieved_episodic"], max(96, request.context_budget_memory // 3))
        blocks["retrieved_docs"] = self._clip_to_budget(blocks["retrieved_docs"], max(96, request.context_budget_docs))
        blocks["active_tool_state"] = self._clip_to_budget(blocks["active_tool_state"], max(48, request.context_budget_tools))
        blocks["unresolved_items"] = self._clip_to_budget(blocks["unresolved_items"], 160)

        # Step 3: cascade compression when overflow remains.
        while self._total(blocks) > usable and mandatory_tokens < usable:
            if blocks["session_summary"]:
                blocks["session_summary"] = self._micro_summary(blocks["session_summary"], limit_tokens=80)
                log.append({"step": "compress_session_summary", "total_tokens": self._total(blocks)})
                if self._total(blocks) <= usable:
                    break

            if blocks["retrieved_episodic"]:
                dropped.append({"block": "retrieved_episodic", "reason": "overflow_drop_low_priority"})
                blocks["retrieved_episodic"] = ""
                log.append({"step": "drop_low_priority_episodic", "total_tokens": self._total(blocks)})
                if self._total(blocks) <= usable:
                    break

            if blocks["retrieved_docs"]:
                dropped.append({"block": "retrieved_docs", "reason": "overflow_drop_low_priority"})
                blocks["retrieved_docs"] = self._micro_summary(blocks["retrieved_docs"], limit_tokens=120)
                log.append({"step": "drop_or_summarize_docs", "total_tokens": self._total(blocks)})
                if self._total(blocks) <= usable:
                    break

            if blocks["working_memory"]:
                blocks["working_memory"] = self._micro_summary(blocks["working_memory"], limit_tokens=80)
                log.append({"step": "micro_summary_working", "total_tokens": self._total(blocks)})
                if self._total(blocks) <= usable:
                    break

            # Final hard truncate of lowest-priority blocks.
            for key in ("retrieved_docs", "retrieved_episodic", "retrieved_semantic", "active_tool_state"):
                before = blocks.get(key, "")
                if not before:
                    continue
                blocks[key] = self._clip_to_budget(before, max(32, self._count(before) // 2))
                log.append({"step": f"hard_truncate_{key}", "total_tokens": self._total(blocks)})
                if self._total(blocks) <= usable:
                    break
            else:
                break

        return ContextBuildResult(
            blocks=blocks,
            selected=selected,
            dropped=dropped,
            score_breakdowns=[x.score_breakdown.to_dict() for x in selected],
            truncation_log=log,
        )

    def _group_candidates(self, rows: list[RetrievalCandidate]) -> dict[str, list[RetrievalCandidate]]:
        semantic: list[RetrievalCandidate] = []
        episodic: list[RetrievalCandidate] = []
        docs: list[RetrievalCandidate] = []
        for row in list(rows or []):
            level = row.record.level
            if level == MemoryLevel.L3_SEMANTIC:
                semantic.append(row)
            elif level == MemoryLevel.L2_EPISODIC:
                episodic.append(row)
            elif level == MemoryLevel.L4_DOCUMENT:
                docs.append(row)
            else:
                episodic.append(row)
        return {
            "semantic": semantic,
            "episodic": episodic,
            "docs": docs,
        }

    @staticmethod
    def _render_working_memory(rows: list[MemoryRecord]) -> str:
        items: list[str] = []
        for row in list(rows or []):
            text = str(row.text or "").strip()
            if not text:
                continue
            items.append(f"- ({row.memory_type.value}) {text}")
        return "\n".join(items)

    @staticmethod
    def _render_candidates(rows: list[RetrievalCandidate]) -> str:
        lines: list[str] = []
        for row in list(rows or []):
            text = str(row.record.text or "").strip()
            if not text:
                continue
            lines.append(
                f"- score={row.final_score:.3f} level={row.record.level.value} scope={row.record.scope.value} {text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_tools(tools: dict[str, Any], private_runtime: dict[str, Any]) -> str:
        payload = {
            "tool_state": dict(tools or {}),
            "private_runtime_state": dict(private_runtime or {}),
        }
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)

    @staticmethod
    def _render_unresolved(rows: list[str]) -> str:
        values = [str(x).strip() for x in list(rows or []) if str(x).strip()]
        return "\n".join([f"- {x}" for x in values])

    def _count(self, text: str) -> int:
        return int(self.tokenizer.count(str(text or "")))

    def _clip_to_budget(self, text: str, budget: int) -> str:
        src = str(text or "").strip()
        if not src:
            return ""
        return str(self.tokenizer.truncate_text(src, max(8, int(budget)))).strip()

    def _micro_summary(self, text: str, *, limit_tokens: int) -> str:
        src = str(text or "").strip()
        if not src:
            return ""
        lines = [x.strip() for x in src.splitlines() if x.strip()]
        if len(lines) <= 2:
            return self._clip_to_budget(src, limit_tokens)
        merged = "\n".join(lines[:2] + (["..."] if len(lines) > 3 else []) + lines[-1:])
        return self._clip_to_budget(merged, limit_tokens)

    def _total(self, blocks: dict[str, str]) -> int:
        total = 0
        for value in list(blocks.values()):
            total += self._count(str(value or ""))
        return total
