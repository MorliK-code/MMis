import unittest

from llm.tokenizer import create_tokenizer
from memory.context_builder import ContextBuilderV2
from memory.memory_models import (
    ContextBuildRequest,
    MemoryLevel,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RetrievalCandidate,
    RetrievalResult,
    ScoreBreakdown,
)
from memory.tool_bridge import (
    build_memory_plan_queries,
    build_memory_tool_context_pack,
    normalize_memory_retrieval_plan,
)


def _candidate(
    record_id: str,
    text: str,
    *,
    memory_type: MemoryType,
    level: MemoryLevel,
    score: float,
    metadata: dict | None = None,
) -> RetrievalCandidate:
    record = MemoryRecord(
        id=record_id,
        text=text,
        memory_type=memory_type,
        level=level,
        scope=MemoryScope.CONVERSATION,
        namespace="conv-tool-bridge",
        metadata=dict(metadata or {}),
        status=MemoryStatus.ACTIVE,
        importance=0.7,
        confidence=0.9,
    )
    return RetrievalCandidate(
        record=record,
        score_breakdown=ScoreBreakdown(final_score=score),
        source="seed",
    )


class _Debugger:
    def __init__(self) -> None:
        self.last_trace = {}

    def record_retrieval_trace(self, **kwargs) -> None:
        self.last_trace = dict(kwargs or {})


class _Manager:
    def __init__(self) -> None:
        self.retrieve_calls = []
        self._context_builder = ContextBuilderV2(tokenizer=create_tokenizer())
        self._debugger = _Debugger()
        self._open_questions = ["How to keep retrieval-plan deterministic?"]
        self._current_decisions = ["Use fan-out retrieval."]
        self._fact = _candidate(
            "fact:name",
            "user.identity_name=Pasha",
            memory_type=MemoryType.FACT,
            level=MemoryLevel.L3_SEMANTIC,
            score=0.92,
            metadata={"fact": {"subject": "user", "predicate": "identity_name", "value": "Pasha"}},
        )
        self._episode = _candidate(
            "episode:name",
            "We previously discussed the user's name.",
            memory_type=MemoryType.EPISODE,
            level=MemoryLevel.L2_EPISODIC,
            score=0.71,
            metadata={"dialog_episode": {"id": "episode:name", "topic": "identity"}},
        )

    def retrieve(self, query):
        self.retrieve_calls.append(query)
        search = str(query.search_text or "").lower()
        rows = [self._fact, self._episode]
        if "profile" in search or "identity" in search:
            rows = [self._fact]
        elif "conversation" in search or "recent" in search or "session" in search:
            rows = [self._episode, self._fact]
        return RetrievalResult(query=query, candidates=rows, reindex_required=False)

    def _classify_memory_recall_mode(self, *, query_text: str) -> str:
        _ = query_text
        return "contextual_recall"

    def _build_fact_expectation_check(self, *, query_text: str, namespace: str) -> dict:
        _ = (query_text, namespace)
        return {
            "expected_predicates": ["identity_name"],
            "found_predicates": ["identity_name"],
            "found_facts": {
                "identity_name": [{"value": "Pasha", "record_id": "fact:name", "scope": "conversation"}]
            },
            "exact_fact_required": True,
            "should_answer_cautiously": False,
        }

    def _exact_fact_candidates_for_expectation(self, *, query, namespace: str, fact_expectation: dict):
        _ = (query, namespace, fact_expectation)
        return [self._fact]

    def _prioritize_retrieval_candidates_for_fact_expectation(self, *, query, candidates, exact_fact_candidates, fact_expectation):
        _ = (query, fact_expectation)
        return list(exact_fact_candidates or []) + [row for row in list(candidates or []) if row.record.id != "fact:name"]

    def _select_candidates_for_recall_mode(self, *, query_text: str, recall_mode: str, fallback_candidates, exact_fact_candidates, fact_expectation):
        _ = (query_text, fact_expectation)
        if recall_mode == "exact_fact_recall":
            return list(exact_fact_candidates or [])
        return list(fallback_candidates or [])

    def _private_runtime_for_namespace(self, namespace: str) -> dict:
        _ = namespace
        return {}

    def _working_for_namespace(self, *, namespace: str, scopes) -> list:
        _ = (namespace, scopes)
        return []

    def _render_relevant_claims(self, candidates) -> str:
        _ = candidates
        return ""

    def _retrieve_dialog_episode_hits(self, *, query, recall_mode: str):
        _ = (query, recall_mode)
        return []

    def _render_recalled_dialog(self, hits) -> str:
        _ = hits
        return ""

    def _retrieve_document_evidence_hits(self, *, query, recall_mode: str):
        _ = (query, recall_mode)
        return []

    def _render_document_evidence(self, hits) -> str:
        _ = hits
        return ""

    def _render_supporting_messages(self, *, selected_candidates, dialog_hits) -> str:
        _ = (selected_candidates, dialog_hits)
        return ""

    def _build_self_facts_context(self, *, query_text: str, selected_candidates, fact_expectation: dict) -> dict:
        _ = (query_text, selected_candidates, fact_expectation)
        return {
            "found_predicates": ["identity_name"],
            "found_facts": {"identity_name": [{"value": "Pasha"}]},
        }

    def _render_fact_expectation_check(self, row: dict) -> str:
        _ = row
        return "- expected_predicates: identity_name"

    def _render_self_facts(self, row: dict) -> str:
        _ = row
        return "- identity_name: Pasha"

    def _render_memory_recall_mode(self, mode: str) -> str:
        return f"- mode: {mode}"


class MemoryToolBridgeTests(unittest.TestCase):
    def test_build_memory_plan_queries_creates_deterministic_fanout(self) -> None:
        request = ContextBuildRequest(
            system_prompt="",
            user_message="What is my name?",
            namespace="conv-tool-bridge",
            scopes=[MemoryScope.CONVERSATION, MemoryScope.SESSION],
            top_k=4,
        )
        plan = normalize_memory_retrieval_plan(
            {
                "mode": "profile",
                "topic_hints": ["name", "identity"],
                "time_hint": "persistent",
                "sources": ["profile", "facts"],
                "top_k": 4,
            }
        )

        queries = build_memory_plan_queries(request=request, plan=plan)

        self.assertEqual([row.label for row in queries], ["raw_user_query", "topic_focused_query", "recent_context_query"])
        self.assertEqual(str(queries[0].query.search_text), "What is my name?")
        self.assertIn("identity", str(queries[1].search_text).lower())
        self.assertIn("persistent", str(queries[2].search_text).lower())

    def test_build_memory_tool_context_pack_uses_plan_and_fanout_queries(self) -> None:
        manager = _Manager()
        request = ContextBuildRequest(
            system_prompt="",
            user_message="What is my name?",
            namespace="conv-tool-bridge",
            scopes=[MemoryScope.CONVERSATION, MemoryScope.SESSION],
            top_k=3,
        )
        plan = normalize_memory_retrieval_plan(
            {
                "mode": "profile",
                "topic_hints": ["name", "identity"],
                "time_hint": "persistent",
                "sources": ["profile", "facts"],
                "top_k": 3,
            }
        )

        pack = build_memory_tool_context_pack(manager, request=request, plan=plan)

        self.assertEqual(len(manager.retrieve_calls), 3)
        self.assertEqual(dict(pack.get("tool_retrieval_plan") or {}).get("mode"), "profile")
        self.assertEqual(len(list(pack.get("fanout_queries") or [])), 3)
        self.assertEqual(str(pack.get("recall_mode") or ""), "exact_fact_recall")
        self.assertIn("self_facts", dict(pack.get("blocks") or {}))
        self.assertEqual(list(pack.get("selected") or [{}])[0].get("id"), "fact:name")
        self.assertEqual(str(dict(manager._debugger.last_trace).get("query") or ""), "What is my name?")


if __name__ == "__main__":
    unittest.main()
