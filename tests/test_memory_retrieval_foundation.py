from __future__ import annotations

import time

from memory.memory_models import MemoryLevel, MemoryRecord, MemoryScope, MemoryType, RetrievalQuery
from memory.retrieval import HybridRetriever
from memory.retrieval_projection import build_memory_views, ensure_memory_views, record_search_text
from memory.vector_store import SQLiteFTSBackend, VectorStore


def _record(
    record_id: str,
    text: str,
    *,
    metadata: dict | None = None,
    source_event_id: str = "",
) -> MemoryRecord:
    now = time.time()
    return MemoryRecord(
        id=record_id,
        text=text,
        memory_type=MemoryType.MESSAGE,
        level=MemoryLevel.L1_SESSION,
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
        _ = (top_k, namespace, scopes, include_stale, metadata_filters)
        self.semantic_queries.append(str(query_text))
        return list(self.semantic_hits)

    def lexical_search(self, *, query_text, top_k, namespace, scopes, include_stale, metadata_filters):
        _ = (top_k, namespace, scopes, include_stale, metadata_filters)
        self.lexical_queries.append(str(query_text))
        return list(self.lexical_hits)


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

    assert store.semantic_queries == ["gpu"]
    assert store.lexical_queries == ["gpu"]


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
