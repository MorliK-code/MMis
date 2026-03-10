"""Context assembly pipeline for Memory V2 prompt blocks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm.tokenizer import (
    BudgetBlock,
    Tokenizer,
    allocate_context_budgets,
    create_tokenizer,
    truncate_blocks,
)
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

        usable = max(256, int(request.context_budget_total) - int(request.context_budget_response_reserve))
        if usable <= 0:
            usable = 256

        # Phase 4: block-aware budgeting that preserves critical memory sections first.
        soft_memory = max(96, int(request.context_budget_memory))
        soft_working = max(72, int(soft_memory * 0.24))
        soft_session = max(72, int(soft_memory * 0.22))
        soft_semantic = max(96, int(soft_memory * 0.34))
        soft_episodic = max(64, int(soft_memory * 0.20))
        soft_docs = max(96, int(request.context_budget_docs))
        soft_tools = max(64, int(request.context_budget_tools))
        soft_unresolved = max(48, min(220, int(soft_memory * 0.20)))

        required_keys = {"system_core", "user_message"}
        priorities = {
            "system_core": 100,
            "user_message": 99,
            "working_memory": 92,
            "session_summary": 88,
            "retrieved_semantic": 90,
            "retrieved_episodic": 52,
            "retrieved_docs": 46,
            "active_tool_state": 87,
            "unresolved_items": 44,
        }
        min_tokens = {
            "system_core": 64 if blocks["system_core"] else 0,
            "user_message": 40 if blocks["user_message"] else 0,
            "working_memory": 32 if blocks["working_memory"] else 0,
            "session_summary": 24 if blocks["session_summary"] else 0,
            "retrieved_semantic": 72 if blocks["retrieved_semantic"] else 0,
            "retrieved_episodic": 0,
            "retrieved_docs": 0,
            "active_tool_state": 48 if blocks["active_tool_state"] else 0,
            "unresolved_items": 0,
        }
        budget_blocks = [
            BudgetBlock(
                block_id="system_core",
                max_tokens=max(min_tokens["system_core"], self._count(blocks["system_core"])),
                min_tokens=min_tokens["system_core"],
                priority=priorities["system_core"],
                required=True,
            ),
            BudgetBlock(
                block_id="user_message",
                max_tokens=max(min_tokens["user_message"], self._count(blocks["user_message"])),
                min_tokens=min_tokens["user_message"],
                priority=priorities["user_message"],
                required=True,
            ),
            BudgetBlock(
                block_id="working_memory",
                max_tokens=soft_working,
                min_tokens=min_tokens["working_memory"],
                priority=priorities["working_memory"],
            ),
            BudgetBlock(
                block_id="session_summary",
                max_tokens=soft_session,
                min_tokens=min_tokens["session_summary"],
                priority=priorities["session_summary"],
            ),
            BudgetBlock(
                block_id="retrieved_semantic",
                max_tokens=soft_semantic,
                min_tokens=min_tokens["retrieved_semantic"],
                priority=priorities["retrieved_semantic"],
            ),
            BudgetBlock(
                block_id="retrieved_episodic",
                max_tokens=soft_episodic,
                min_tokens=min_tokens["retrieved_episodic"],
                priority=priorities["retrieved_episodic"],
            ),
            BudgetBlock(
                block_id="retrieved_docs",
                max_tokens=soft_docs,
                min_tokens=min_tokens["retrieved_docs"],
                priority=priorities["retrieved_docs"],
            ),
            BudgetBlock(
                block_id="active_tool_state",
                max_tokens=soft_tools,
                min_tokens=min_tokens["active_tool_state"],
                priority=priorities["active_tool_state"],
            ),
            BudgetBlock(
                block_id="unresolved_items",
                max_tokens=soft_unresolved,
                min_tokens=min_tokens["unresolved_items"],
                priority=priorities["unresolved_items"],
            ),
        ]
        block_budgets = allocate_context_budgets(total_tokens=usable, reserve_tokens=0, blocks=budget_blocks)
        blocks, pre_log = truncate_blocks(
            blocks,
            block_budgets=block_budgets,
            tokenizer=self.tokenizer,
            overflow_strategy="priority_drop",
            block_priorities=priorities,
            required_blocks=required_keys,
            min_block_tokens=min_tokens,
        )
        for item in list(pre_log or []):
            log.append({"step": "pre_budget", **dict(item)})

        # Compression cascade order:
        # 1) compress session/history
        # 2) drop episodic
        # 3) drop/summarize docs
        # 4) replace low-priority blocks with micro-summaries
        # 5) final hard truncate low-priority blocks only
        self._apply_cascade(
            blocks=blocks,
            usable=usable,
            dropped=dropped,
            log=log,
        )

        log.append(
            {
                "step": "final_budget",
                "total_tokens": self._total(blocks),
                "usable_tokens": int(usable),
                "block_tokens": {k: self._count(v) for k, v in blocks.items()},
            }
        )

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

    def _apply_cascade(
        self,
        *,
        blocks: dict[str, str],
        usable: int,
        dropped: list[dict[str, Any]],
        log: list[dict[str, Any]],
    ) -> None:
        def _log_step(step: str) -> None:
            log.append({"step": step, "total_tokens": self._total(blocks)})

        def _drop_block(key: str, reason: str) -> None:
            if not str(blocks.get(key) or "").strip():
                return
            blocks[key] = ""
            dropped.append({"block": key, "reason": reason})

        if self._total(blocks) <= usable:
            return

        if blocks.get("session_summary"):
            blocks["session_summary"] = self._micro_summary(blocks["session_summary"], limit_tokens=72)
            _log_step("compress_session_summary")
        if self._total(blocks) <= usable:
            return

        if blocks.get("retrieved_episodic"):
            _drop_block("retrieved_episodic", "overflow_drop_low_priority")
            _log_step("drop_low_priority_episodic")
        if self._total(blocks) <= usable:
            return

        if blocks.get("retrieved_docs"):
            blocks["retrieved_docs"] = self._micro_summary(blocks["retrieved_docs"], limit_tokens=96)
            _log_step("compress_docs")
            if self._total(blocks) > usable and blocks.get("retrieved_docs"):
                _drop_block("retrieved_docs", "overflow_drop_low_priority")
                _log_step("drop_low_priority_docs")
        if self._total(blocks) <= usable:
            return

        if blocks.get("unresolved_items"):
            blocks["unresolved_items"] = self._micro_summary(blocks["unresolved_items"], limit_tokens=48)
            _log_step("micro_summary_unresolved")
            if self._total(blocks) > usable:
                _drop_block("unresolved_items", "overflow_drop_low_priority")
                _log_step("drop_low_priority_unresolved")
        if self._total(blocks) <= usable:
            return

        if blocks.get("working_memory"):
            blocks["working_memory"] = self._micro_summary(blocks["working_memory"], limit_tokens=80)
            _log_step("micro_summary_working")
        if self._total(blocks) <= usable:
            return

        for key in ("retrieved_docs", "retrieved_episodic", "unresolved_items", "session_summary", "working_memory"):
            value = str(blocks.get(key) or "").strip()
            if not value:
                continue
            blocks[key] = self._clip_to_budget(value, max(24, self._count(value) // 2))
            _log_step(f"hard_truncate_{key}")
            if self._total(blocks) <= usable:
                return

        # Last-resort: only after low-priority cascade, compress critical blocks.
        if blocks.get("retrieved_semantic") and self._total(blocks) > usable:
            blocks["retrieved_semantic"] = self._micro_summary(blocks["retrieved_semantic"], limit_tokens=96)
            _log_step("last_resort_compress_semantic")
        if blocks.get("active_tool_state") and self._total(blocks) > usable:
            blocks["active_tool_state"] = self._clip_to_budget(blocks["active_tool_state"], 72)
            _log_step("last_resort_compress_tool_state")

        # Absolute fallback: keep system + user bounded in extreme overflow.
        if self._total(blocks) > usable:
            system_target = max(48, usable // 2)
            blocks["system_core"] = self._clip_to_budget(str(blocks.get("system_core") or ""), system_target)
            user_target = max(32, usable - self._count(blocks["system_core"]))
            blocks["user_message"] = self._clip_to_budget(str(blocks.get("user_message") or ""), user_target)
            _log_step("last_resort_bound_mandatory")
