from __future__ import annotations

import time

from memory.memory_models import MemoryLevel, MemoryRecord, MemoryScope, MemoryType, RetrievalQuery
from memory.retrieval import HybridRetriever
from memory.retrieval_projection import build_memory_views, ensure_memory_views, record_search_text
from memory.reranker import HeuristicReranker
from memory.memory_models import RetrievalCandidate, ScoreBreakdown
from memory.vector_store import SQLiteFTSBackend, VectorStore


def _record(
    record_id: str,
    text: str,
    *,
    metadata: dict | None = None,
    source_event_id: str = "",
    memory_type: MemoryType = MemoryType.MESSAGE,
    level: MemoryLevel = MemoryLevel.L1_SESSION,
) -> MemoryRecord:
    now = time.time()
    return MemoryRecord(
        id=record_id,
        text=text,
        memory_type=memory_type,
        level=level,
        scope=MemoryScope.CONVERSATION,
        namespace="default",
        metadata=dict(metadata or {}),
        importance=0.5,
        confidence=0.5,
        created_at=now,
        updated_at=now,
        source_event_id=source_event_id,
    )


class _StoreSpy:
    def __init__(self, *, semantic_hits=None, lexical_hits=None) -> None:
        self.reindex_required = False
        self.semantic_hits = list(semantic_hits or [])
        self.lexical_hits = list(lexical_hits or [])
        self.semantic_queries: list[str] = []
        self.lexical_queries: list[str] = []

    def semantic_search(self, *, query_text, top_k, namespace, scopes, include_stale, metadata_filters):
        _ = (top_k, namespace, scopes, include_stale)
        self.semantic_queries.append(str(query_text))
        return self._filter_hits(self.semantic_hits, metadata_filters)

    def lexical_search(self, *, query_text, top_k, namespace, scopes, include_stale, metadata_filters):
        _ = (top_k, namespace, scopes, include_stale)
        self.lexical_queries.append(str(query_text))
        return self._filter_hits(self.lexical_hits, metadata_filters)

    @staticmethod
    def _filter_hits(hits, metadata_filters):
        filters = dict(metadata_filters or {})
        allowed_types = {
            str(x).strip().lower()
            for x in list(filters.get("memory_type") or [])
            if str(x).strip()
        }
        if not allowed_types:
            return list(hits or [])
        out = []
        for record, score in list(hits or []):
            if str(record.memory_type.value).strip().lower() not in allowed_types:
                continue
            out.append((record, score))
        return out


class _EmbeddingSpy:
    model_name = "spy"
    embedding_version = "spy_v1"

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def model_fingerprint(self) -> str:
        return "spy-fingerprint"

    def embed(self, text: str) -> list[float]:
        self.inputs.append(str(text))
        return [1.0, 0.0]


def test_retrieval_uses_query_search_text_not_raw_text() -> None:
    store = _StoreSpy()
    retriever = HybridRetriever(store=store)

    retriever.retrieve(
        RetrievalQuery(
            query_text="какая у меня видяха",
            search_text="gpu",
            top_k=3,
        )
    )

    assert store.semantic_queries == ["gpu", "gpu"]
    assert store.lexical_queries == ["gpu", "gpu"]


def test_prepare_record_embeds_search_text_not_raw_text() -> None:
    embedding_provider = _EmbeddingSpy()
    store = VectorStore.__new__(VectorStore)
    store.embedding_provider = embedding_provider
    store.reindex_required = False

    record = _record("rec-1", "мне нравится моя видяха 3050 ti")
    prepared = VectorStore._prepare_record(store, record)

    assert embedding_provider.inputs == [record_search_text(prepared.text, metadata=prepared.metadata)]
    assert "gpu" in embedding_provider.inputs[0]


def test_retrieval_uses_search_text_not_raw_text(tmp_path) -> None:
    backend = SQLiteFTSBackend(tmp_path / "lexical.sqlite3")
    record = _record("rec-1", "мне нравится моя видяха 3050 ti")

    backend.batch_upsert([record])
    hits = backend.search(
        query="gpu",
        top_k=5,
        namespace="default",
        scopes=[MemoryScope.CONVERSATION],
        include_stale=False,
        metadata_filters={},
    )

    assert [item.id for item, _ in hits] == ["rec-1"]
    backend.close()


def test_retrieval_entity_keys_boost_correct_record() -> None:
    record_3050 = _record("gpu-3050", "ноутбук с rtx 3050 ti", metadata=ensure_memory_views("ноутбук с rtx 3050 ti"))
    record_3060 = _record("gpu-3060", "ноутбук с rtx 3060", metadata=ensure_memory_views("ноутбук с rtx 3060"))
    store = _StoreSpy(
        semantic_hits=[(record_3050, 0.6), (record_3060, 0.6)],
        lexical_hits=[(record_3050, 0.6), (record_3060, 0.6)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="rtx 3050 ti",
            search_text="gpu rtx 3050 ti",
            entity_keys=["gpu", "rtx_3050_ti"],
            top_k=2,
        )
    )

    assert [item.record.id for item in result.candidates] == ["gpu-3050", "gpu-3060"]
    assert result.candidates[0].final_score > result.candidates[1].final_score


def test_retrieval_includes_fact_channel_alongside_message_channel() -> None:
    fact_record = _record(
        "fact-gpu",
        "user.environment_gpu_model=RTX 3050 Ti",
        metadata=ensure_memory_views(
            "user.environment_gpu_model=RTX 3050 Ti",
            metadata={
                "fact": {"predicate": "environment_gpu_model", "subject": "user", "value": "RTX 3050 Ti"},
                "canonical_key": "user.environment_gpu_model",
            },
        ),
        memory_type=MemoryType.FACT,
        level=MemoryLevel.L3_SEMANTIC,
    )
    message_record = _record("msg-gpu", "у меня видеокарта rtx 3050 ti")
    store = _StoreSpy(
        semantic_hits=[(fact_record, 0.62), (message_record, 0.67)],
        lexical_hits=[(fact_record, 0.65), (message_record, 0.66)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="rtx 3050 ti",
            search_text="gpu rtx 3050 ti",
            entity_keys=["gpu", "rtx_3050_ti"],
            top_k=4,
        )
    )

    ids = [item.record.id for item in result.candidates]
    assert "fact-gpu" in ids
    assert "msg-gpu" in ids


def test_retrieval_self_like_query_boosts_expected_fact_hit() -> None:
    fact_record = _record(
        "fact-gpu",
        "user.environment_gpu_model=RTX 3050 Ti",
        metadata=ensure_memory_views(
            "user.environment_gpu_model=RTX 3050 Ti",
            metadata={
                "fact": {"predicate": "environment_gpu_model", "subject": "user", "value": "RTX 3050 Ti"},
                "canonical_key": "user.environment_gpu_model",
            },
        ),
        memory_type=MemoryType.FACT,
        level=MemoryLevel.L3_SEMANTIC,
    )
    message_record = _record("msg-gpu", "мы обсуждали мою видеокарту rtx 3050 ti", source_event_id="evt:gpu")
    store = _StoreSpy(
        semantic_hits=[(fact_record, 0.58), (message_record, 0.72)],
        lexical_hits=[(fact_record, 0.61), (message_record, 0.74)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="подскажи мою видеокарту",
            search_text="gpu rtx 3050 ti",
            entity_keys=["gpu", "rtx_3050_ti"],
            top_k=4,
        )
    )

    assert [item.record.id for item in result.candidates][:1] == ["fact-gpu"]


def test_retrieval_prefers_active_user_fact_over_assistant_fact_for_self_like_query() -> None:
    user_fact = _record(
        "fact-user-gpu",
        "user.environment_gpu_model=RTX 3050 Ti",
        metadata=ensure_memory_views(
            "user.environment_gpu_model=RTX 3050 Ti",
            metadata={
                "fact": {"predicate": "environment_gpu_model", "subject": "user", "value": "RTX 3050 Ti"},
                "canonical_key": "user.environment_gpu_model",
            },
        ),
        memory_type=MemoryType.FACT,
        level=MemoryLevel.L3_SEMANTIC,
    )
    assistant_fact = _record(
        "fact-assistant-gpu",
        "assistant.environment_gpu_model=RTX 3050 Ti",
        metadata=ensure_memory_views(
            "assistant.environment_gpu_model=RTX 3050 Ti",
            metadata={
                "fact": {"predicate": "environment_gpu_model", "subject": "assistant", "value": "RTX 3050 Ti"},
                "canonical_key": "assistant.environment_gpu_model",
            },
        ),
        memory_type=MemoryType.FACT,
        level=MemoryLevel.L3_SEMANTIC,
    )
    store = _StoreSpy(
        semantic_hits=[(user_fact, 0.56), (assistant_fact, 0.60)],
        lexical_hits=[(user_fact, 0.58), (assistant_fact, 0.61)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="какая у меня видеокарта",
            search_text="gpu rtx 3050 ti",
            entity_keys=["gpu", "rtx_3050_ti"],
            top_k=4,
        )
    )

    assert [item.record.id for item in result.candidates][:1] == ["fact-user-gpu"]


def test_retrieval_penalizes_assistant_memory_miss_help_reply_noise() -> None:
    assistant_noise = _record(
        "msg-assistant-noise",
        "Не помню твою видеокарту. Проверь сам через winver или python --version.",
        metadata=ensure_memory_views(
            "Не помню твою видеокарту. Проверь сам через winver или python --version.",
            metadata={"source_kind": "assistant_reply"},
        ),
        level=MemoryLevel.L2_EPISODIC,
    )
    user_message = _record(
        "msg-user-gpu",
        "У меня RTX 3050 Ti.",
        metadata=ensure_memory_views(
            "У меня RTX 3050 Ti.",
            metadata={"source_kind": "user"},
        ),
        level=MemoryLevel.L2_EPISODIC,
    )
    store = _StoreSpy(
        semantic_hits=[(assistant_noise, 0.76), (user_message, 0.70)],
        lexical_hits=[(assistant_noise, 0.78), (user_message, 0.69)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="какая у меня видеокарта",
            search_text="gpu rtx 3050 ti",
            entity_keys=["gpu", "rtx_3050_ti"],
            top_k=4,
        )
    )

    assert [item.record.id for item in result.candidates][:1] == ["msg-user-gpu"]


def test_reranker_prioritizes_fact_type_over_working_message_when_scores_are_close() -> None:
    fact_record = _record(
        "fact-gpu",
        "user.environment_gpu_model=RTX 3050 Ti",
        metadata=ensure_memory_views(
            "user.environment_gpu_model=RTX 3050 Ti",
            metadata={
                "fact": {"predicate": "environment_gpu_model", "subject": "user", "value": "RTX 3050 Ti"},
                "canonical_key": "user.environment_gpu_model",
            },
        ),
        memory_type=MemoryType.FACT,
        level=MemoryLevel.L3_SEMANTIC,
    )
    working_message = _record(
        "msg-working",
        "я сейчас обсуждаю свою видеокарту",
        memory_type=MemoryType.MESSAGE,
        level=MemoryLevel.L0_WORKING,
    )
    reranker = HeuristicReranker()

    ranked = reranker.rerank(
        RetrievalQuery(query_text="какая у меня видеокарта", top_k=2),
        candidates=[
            RetrievalCandidate(record=working_message, score_breakdown=ScoreBreakdown(final_score=0.72), source="message_channel"),
            RetrievalCandidate(record=fact_record, score_breakdown=ScoreBreakdown(final_score=0.68), source="fact_channel"),
        ],
        top_k=2,
    )

    assert [item.record.id for item in ranked] == ["fact-gpu", "msg-working"]


def test_retrieval_prefers_exact_numeric_match() -> None:
    record_4gb = _record("gpu-4", "видеокарта с 4 gb vram", metadata=ensure_memory_views("видеокарта с 4 gb vram"))
    record_6gb = _record("gpu-6", "видеокарта с 6 gb vram", metadata=ensure_memory_views("видеокарта с 6 gb vram"))
    store = _StoreSpy(
        semantic_hits=[(record_4gb, 0.6), (record_6gb, 0.6)],
        lexical_hits=[(record_4gb, 0.6), (record_6gb, 0.6)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="4 GB VRAM",
            search_text="vram 4gb",
            numeric_keys=["vram:4gb"],
            top_k=2,
        )
    )

    assert [item.record.id for item in result.candidates] == ["gpu-4", "gpu-6"]
    assert result.candidates[0].score_breakdown.numeric_overlap_score == 1.0
    assert result.candidates[0].final_score > result.candidates[1].final_score


def test_retrieval_collapses_duplicates_by_canonical_key() -> None:
    metadata = ensure_memory_views(
        "rtx 3050 ti",
        metadata={"canonical_key": "hardware.gpu_model"},
    )
    better = _record("gpu-a", "rtx 3050 ti", metadata=metadata)
    duplicate = _record("gpu-b", "rtx 3050 ti", metadata=metadata)
    store = _StoreSpy(
        semantic_hits=[(better, 0.7), (duplicate, 0.65)],
        lexical_hits=[(better, 0.7), (duplicate, 0.65)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="rtx 3050 ti",
            search_text="gpu rtx 3050 ti",
            entity_keys=["gpu", "rtx_3050_ti"],
            top_k=3,
        )
    )

    assert [item.record.id for item in result.candidates] == ["gpu-a"]


def test_retrieval_drops_candidates_below_min_candidate_score() -> None:
    weak = _record(
        "weak-1",
        "совсем не про это",
        metadata=ensure_memory_views("совсем не про это"),
    )
    weak = MemoryRecord(
        id=weak.id,
        text=weak.text,
        memory_type=weak.memory_type,
        level=weak.level,
        scope=weak.scope,
        namespace=weak.namespace,
        metadata=weak.metadata,
        importance=0.0,
        confidence=0.0,
        created_at=weak.created_at,
        updated_at=0.0,
        source_event_id=weak.source_event_id,
    )
    store = _StoreSpy(semantic_hits=[(weak, 0.05)], lexical_hits=[(weak, 0.05)])
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="gpu rtx 3050 ti",
            search_text="gpu rtx 3050 ti",
            entity_keys=["gpu", "rtx_3050_ti"],
            top_k=3,
        )
    )

    assert result.candidates == []


def test_build_memory_views_does_not_reinflate_existing_search_text() -> None:
    metadata = ensure_memory_views(
        "user.identity_name=Паша",
        metadata={
            "canonical_key": "user.identity_name",
            "fact": {
                "subject": "user",
                "predicate": "identity_name",
                "value": "Паша",
                "relation": "identity",
            },
        },
    )

    first = dict(metadata.get("memory_views") or {})
    rebuilt = build_memory_views("user.identity_name=Паша", metadata=metadata)
    rebuilt_twice = build_memory_views("user.identity_name=Паша", metadata={"memory_views": rebuilt, **metadata})

    assert rebuilt["search_text"] == first["search_text"]
    assert rebuilt_twice["search_text"] == rebuilt["search_text"]


def test_build_memory_views_does_not_leak_generic_source_into_search_text() -> None:
    views = build_memory_views(
        "привет. меня зовут Паша",
        metadata={"source": "api"},
    )

    assert views["canonical_text"] == ""
    assert views["search_text"] == "привет. меня зовут паша"


def test_build_memory_views_discards_old_low_signal_canonical_text() -> None:
    views = build_memory_views(
        "привет. меня зовут Паша",
        metadata={
            "source": "api",
            "memory_views": {
                "canonical_text": "api",
                "search_text": "привет. меня зовут паша api",
            },
        },
    )

    assert views["canonical_text"] == ""
    assert views["search_text"] == "привет. меня зовут паша"


def test_build_memory_views_uses_structured_metadata_keys_for_message_records() -> None:
    views = build_memory_views(
        "У меня видеокарта и Python",
        metadata={
            "memory_entities": [
                {"type": "gpu_model", "canonical": "RTX 3050 Ti"},
                {"type": "python_version", "canonical": "3.11"},
            ],
            "numeric_facts": [
                {"kind": "vram_gb", "value": 4, "unit": "gb"},
                {"kind": "python_version", "value": "3.11", "unit": ""},
            ],
        },
    )

    assert "gpu" in list(views.get("entity_keys") or [])
    assert "rtx_3050_ti" in list(views.get("entity_keys") or [])
    assert "python_3_11" in list(views.get("entity_keys") or [])
    assert "vram:4gb" in list(views.get("numeric_keys") or [])
    assert "version:3.11" in list(views.get("numeric_keys") or [])
    assert "value:3.11" not in list(views.get("numeric_keys") or [])


def test_build_memory_views_uses_fact_metadata_keys_for_fact_records() -> None:
    gpu_views = build_memory_views(
        "user.environment_gpu_model=RTX 3050 Ti",
        metadata={
            "fact": {
                "subject": "user",
                "predicate": "environment_gpu_model",
                "value": "RTX 3050 Ti",
            },
            "canonical_key": "user.environment_gpu_model",
        },
    )
    ram_views = build_memory_views(
        "user.environment_ram_gb=32",
        metadata={
            "fact": {
                "subject": "user",
                "predicate": "environment_ram_gb",
                "value": "32",
            },
            "canonical_key": "user.environment_ram_gb",
        },
    )

    assert "gpu" in list(gpu_views.get("entity_keys") or [])
    assert "rtx_3050_ti" in list(gpu_views.get("entity_keys") or [])
    assert "ram:32gb" in list(ram_views.get("numeric_keys") or [])
    assert "value:32gb" in list(ram_views.get("numeric_keys") or [])


def test_build_memory_views_uses_fact_metadata_keys_for_decision_and_task_records() -> None:
    decision_views = build_memory_views(
        "user.decision=store_search_text_and_canonical_text",
        metadata={
            "fact": {
                "subject": "user",
                "predicate": "decision",
                "value": "store_search_text_and_canonical_text",
            },
            "canonical_key": "user.decision",
        },
    )
    task_views = build_memory_views(
        "assistant.task=next_write_policy",
        metadata={
            "fact": {
                "subject": "assistant",
                "predicate": "task",
                "value": "next_write_policy",
            },
            "canonical_key": "assistant.task",
        },
    )

    assert "decision" in list(decision_views.get("entity_keys") or [])
    assert "store_search_text_and_canonical_text" in list(decision_views.get("entity_keys") or [])
    assert "task" in list(task_views.get("entity_keys") or [])
    assert "next_write_policy" in list(task_views.get("entity_keys") or [])
