from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import shutil
import tempfile
import time
import unittest
from pathlib import Path

from core.character_runtime import PromptPack
from core.response_pipeline import _apply_memory_context_to_prompt_pack
from llm.tokenizer import ApproxTokenizer, BudgetBlock, allocate_context_budgets, truncate_blocks
from memory.memory_manager import MemoryManager
from memory.memory_models import (
    ContextBuildRequest,
    DebugRequest,
    MemoryEvent,
    MemoryLevel,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)


class MemoryPhase4ContextBuilderTests(unittest.TestCase):
    def test_context_builder_cascade_keeps_critical_blocks(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="mmis_phase4_context_")
        try:
            manager = MemoryManager(root_dir=Path(tmpdir))
            try:
                now_ts = float(time.time())
                manager._store.batch_upsert(
                    [
                        MemoryRecord(
                            id="fact:critical_1",
                            text="critical fact release deadline is friday",
                            memory_type=MemoryType.FACT,
                            level=MemoryLevel.L3_SEMANTIC,
                            scope=MemoryScope.CONVERSATION,
                            namespace="n1",
                            metadata={"canonical_key": "project.deadline"},
                            importance=0.92,
                            confidence=0.88,
                            created_at=now_ts,
                            updated_at=now_ts,
                            status=MemoryStatus.ACTIVE,
                        ),
                        MemoryRecord(
                            id="episode:low_1",
                            text="episodic note from old conversation with long details " * 16,
                            memory_type=MemoryType.EPISODE,
                            level=MemoryLevel.L2_EPISODIC,
                            scope=MemoryScope.CONVERSATION,
                            namespace="n1",
                            metadata={},
                            importance=0.40,
                            confidence=0.55,
                            created_at=now_ts,
                            updated_at=now_ts,
                            status=MemoryStatus.ACTIVE,
                        ),
                        MemoryRecord(
                            id="doc:chunk_1",
                            text="document chunk evidence for integration tests " * 22,
                            memory_type=MemoryType.DOCUMENT_CHUNK,
                            level=MemoryLevel.L4_DOCUMENT,
                            scope=MemoryScope.CONVERSATION,
                            namespace="n1",
                            metadata={"doc_id": "doc:1"},
                            importance=0.55,
                            confidence=0.70,
                            created_at=now_ts,
                            updated_at=now_ts,
                            status=MemoryStatus.ACTIVE,
                        ),
                    ]
                )
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="working memory seed for phase 4 context builder",
                        namespace="n1",
                        scope=MemoryScope.CONVERSATION,
                        memory_type=MemoryType.MESSAGE,
                    )
                )

                result = manager.build_context(
                    ContextBuildRequest(
                        system_prompt=("SYSTEM RULES " * 60).strip(),
                        user_message="need release deadline and evidence",
                        namespace="n1",
                        scopes=[MemoryScope.CONVERSATION],
                        top_k=8,
                        session_summary="session history summary " * 40,
                        tool_state={"active_task": "release", "status": "in_progress"},
                        unresolved_items=["confirm publish date", "validate integration tests"],
                        context_budget_total=360,
                        context_budget_memory=130,
                        context_budget_docs=90,
                        context_budget_tools=64,
                        context_budget_response_reserve=180,
                    )
                )

                self.assertTrue(str(result.blocks.get("system_core") or "").strip())
                self.assertTrue(str(result.blocks.get("user_message") or "").strip())
                self.assertTrue(str(result.blocks.get("active_tool_state") or "").strip())
                # Critical semantic block must survive longer than low-priority episodic/docs.
                self.assertTrue(str(result.blocks.get("retrieved_semantic") or "").strip())
                self.assertTrue(
                    (not str(result.blocks.get("retrieved_episodic") or "").strip())
                    or any("drop_low_priority_episodic" in str(x.get("step") or "") for x in result.truncation_log)
                )
                self.assertTrue(result.truncation_log)
            finally:
                manager.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_debug_snapshot_includes_selection_and_trimming_trace(self) -> None:
        tmpdir = tempfile.mkdtemp(prefix="mmis_phase4_debug_")
        try:
            manager = MemoryManager(root_dir=Path(tmpdir))
            try:
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="Remember this decision: use chromadb for vectors",
                        namespace="n2",
                        scope=MemoryScope.CONVERSATION,
                        memory_type=MemoryType.MESSAGE,
                    )
                )
                manager.build_context(
                    ContextBuildRequest(
                        system_prompt="sys",
                        user_message="what vector backend do we use?",
                        namespace="n2",
                        scopes=[MemoryScope.CONVERSATION],
                        context_budget_total=260,
                        context_budget_memory=96,
                        context_budget_docs=96,
                        context_budget_tools=64,
                        context_budget_response_reserve=140,
                    )
                )
                snap = manager.debug_snapshot(DebugRequest(namespace="n2", limit=60))
                trace = dict(snap.get("last_retrieval_trace") or {})
                self.assertIn("truncation_log", trace)
                self.assertIn("score_breakdowns", trace)
                self.assertIn("selected", trace)
                selected = list(trace.get("selected") or [])
                if selected:
                    self.assertIn("why_selected", dict(selected[0]))
                self.assertIn("inspection", snap)
                self.assertIn("status_counts", snap)
            finally:
                manager.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_prompt_pack_helper_injects_context_blocks(self) -> None:
        base = PromptPack(
            system_prompt="sys",
            user_message="user",
            full_prompt="full",
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
            blocks={
                "system_role": "legacy system",
                "user_message": "legacy user",
                "retrieved_memories": "legacy memory",
                "long_summary": "legacy summary",
                "conversation_tail": "legacy tail",
            },
            token_usage={},
            budgets={},
            cut_info={},
            context_tags={},
            selected_memories=[],
            conversation_tail=[],
        )
        memory_context = {
            "blocks": {
                "system_core": "strict system",
                "user_message": "actual user question",
                "working_memory": "- wm item",
                "session_summary": "session digest",
                "retrieved_semantic": "- semantic fact",
                "retrieved_episodic": "- episodic note",
                "retrieved_docs": "- doc evidence",
                "active_tool_state": "{\"task\": \"release\"}",
                "unresolved_items": "- unresolved x",
            },
            "selected": [{"id": "fact:1"}],
            "dropped": [{"block": "retrieved_episodic", "reason": "overflow_drop_low_priority"}],
            "truncation_log": [{"step": "compress_session_summary"}],
        }
        updated = _apply_memory_context_to_prompt_pack(
            base,
            memory_context=memory_context,
            selected_memories=[{"id": "fact:1"}],
        )
        self.assertIn("[SEMANTIC_FACTS]", str(updated.blocks.get("retrieved_memories") or ""))
        self.assertEqual(str(updated.blocks.get("conversation_tail") or ""), "")
        self.assertEqual(str(updated.blocks.get("system_role") or ""), "strict system")
        self.assertEqual(int(updated.cut_info.get("memory_context_compression_steps") or 0), 1)

    def test_tokenizer_budget_and_priority_truncation(self) -> None:
        budgets = allocate_context_budgets(
            total_tokens=220,
            reserve_tokens=80,
            blocks=[
                BudgetBlock(block_id="system", max_tokens=120, min_tokens=42, priority=100, required=True),
                BudgetBlock(block_id="user", max_tokens=80, min_tokens=28, priority=99, required=True),
                BudgetBlock(block_id="episodic", max_tokens=120, min_tokens=0, priority=20, required=False),
            ],
        )
        self.assertGreaterEqual(int(budgets.get("system") or 0), 42)
        self.assertGreaterEqual(int(budgets.get("user") or 0), 28)

        tk = ApproxTokenizer(chars_per_token=3.4)
        clipped, log = truncate_blocks(
            {
                "system": "S " * 200,
                "user": "U " * 160,
                "episodic": "E " * 480,
            },
            block_budgets={"system": 60, "user": 40, "episodic": 20},
            tokenizer=tk,
            overflow_strategy="priority_drop",
            block_priorities={"system": 100, "user": 99, "episodic": 5},
            required_blocks={"system", "user"},
            min_block_tokens={"system": 24, "user": 16, "episodic": 0},
        )
        self.assertTrue(str(clipped.get("system") or "").strip())
        self.assertTrue(str(clipped.get("user") or "").strip())
        self.assertIn("episodic", clipped)
        self.assertTrue(log)


if __name__ == "__main__":
    unittest.main()
