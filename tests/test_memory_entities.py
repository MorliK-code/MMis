from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from memory.memory_manager import MemoryManager
from memory.memory_models import ContextBuildRequest, MemoryEvent, MemoryScope, MemoryType, RetrievalQuery


class MemoryEntityTests(unittest.TestCase):
    def test_hybrid_retrieval_returns_relevant_candidate_with_score_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(root_dir=Path(tmpdir))
            try:
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="Docker compose fails in VS Code on Windows path C:\\Users\\dev",
                        namespace="n1",
                        scope=MemoryScope.CONVERSATION,
                        memory_type=MemoryType.MESSAGE,
                    )
                )
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="I like jazz music",
                        namespace="n1",
                        scope=MemoryScope.CONVERSATION,
                        memory_type=MemoryType.MESSAGE,
                    )
                )

                out = manager.retrieve(
                    RetrievalQuery(
                        query_text="docker vscode windows",
                        namespace="n1",
                        scopes=[MemoryScope.CONVERSATION],
                        top_k=5,
                    )
                )
                self.assertTrue(out.candidates)
                top = out.candidates[0]
                self.assertIn("docker", str(top.record.text).lower())
                self.assertGreater(float(top.score_breakdown.final_score), 0.0)
                self.assertTrue(
                    float(top.score_breakdown.lexical_score) > 0.0
                    or float(top.score_breakdown.semantic_similarity) > 0.0
                )
            finally:
                manager.close()

    def test_private_runtime_is_excluded_from_retrieval_but_available_in_context_tool_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = MemoryManager(root_dir=Path(tmpdir))
            try:
                manager.ingest_event(
                    MemoryEvent(
                        role="system",
                        text="tool transient cache",
                        namespace="n2",
                        scope=MemoryScope.PRIVATE_RUNTIME,
                        memory_type=MemoryType.RUNTIME_STATE,
                        metadata={
                            "runtime_key": "tool.last",
                            "runtime_value": {"ok": True},
                            "ttl_sec": 120,
                        },
                    )
                )
                manager.ingest_event(
                    MemoryEvent(
                        role="user",
                        text="temporary token abc123 for now",
                        namespace="n2",
                        scope=MemoryScope.TEMPORARY,
                        memory_type=MemoryType.MESSAGE,
                        metadata={"ttl_sec": 120},
                    )
                )

                retrieved = manager.retrieve(
                    RetrievalQuery(
                        query_text="abc123",
                        namespace="n2",
                        scopes=[MemoryScope.TEMPORARY, MemoryScope.CONVERSATION],
                        top_k=5,
                    )
                )
                scopes = [x.record.scope for x in list(retrieved.candidates or [])]
                self.assertNotIn(MemoryScope.PRIVATE_RUNTIME, scopes)
                self.assertIn(MemoryScope.TEMPORARY, scopes)

                context = manager.build_context(
                    ContextBuildRequest(
                        system_prompt="sys",
                        user_message="check runtime state",
                        namespace="n2",
                        scopes=[MemoryScope.TEMPORARY, MemoryScope.CONVERSATION],
                    )
                )
                self.assertIn("private_runtime_state", str(context.blocks.get("active_tool_state") or ""))
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
