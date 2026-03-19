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

    assert store.semantic_queries == ["gpu", "gpu", "gpu"]
    assert store.lexical_queries == ["gpu", "gpu", "gpu"]


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


def test_claim_retrieval_prefers_matching_predicate_and_subject() -> None:
    likes_claim = _record(
        "claim-like-rose",
        "user.likes=смотреть в глаза розе",
        metadata=ensure_memory_views(
            "user.likes=смотреть в глаза розе",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "likes",
                    "obj": "смотреть в глаза розе",
                    "object_surface": "смотреть в глаза Розе",
                    "topic_keys": ["likes", "preference", "activity"],
                    "trigger_keys": ["likes", "розе", "смотреть"],
                    "recall_mode": "ambient",
                    "spontaneous_recall": True,
                },
                "canonical_key": "user.likes.rose_eyes",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    uses_claim = _record(
        "claim-use-vscode",
        "user.uses=vs code",
        metadata=ensure_memory_views(
            "user.uses=vs code",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "uses",
                    "obj": "vs code",
                    "object_surface": "VS Code",
                    "topic_keys": ["uses", "usage", "tool"],
                    "trigger_keys": ["uses", "vs", "code"],
                    "recall_mode": "contextual",
                    "spontaneous_recall": False,
                },
                "canonical_key": "user.uses.vs_code",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    store = _StoreSpy(
        semantic_hits=[(likes_claim, 0.60), (uses_claim, 0.63)],
        lexical_hits=[(likes_claim, 0.58), (uses_claim, 0.61)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="что мне нравится",
            search_text="что мне нравится",
            top_k=4,
        )
    )

    assert [item.record.id for item in result.candidates][:1] == ["claim-like-rose"]


def test_claim_retrieval_uses_topic_and_trigger_keys() -> None:
    rose_claim = _record(
        "claim-like-rose",
        "user.likes=смотреть в глаза розе",
        metadata=ensure_memory_views(
            "user.likes=смотреть в глаза розе",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "likes",
                    "obj": "смотреть в глаза розе",
                    "object_surface": "смотреть в глаза Розе",
                    "topic_keys": ["likes", "preference", "activity"],
                    "trigger_keys": ["likes", "розе", "смотреть"],
                    "recall_mode": "ambient",
                    "spontaneous_recall": True,
                },
                "canonical_key": "user.likes.rose_eyes",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    lotus_claim = _record(
        "claim-like-lotus",
        "user.likes=запах цветка лотоса",
        metadata=ensure_memory_views(
            "user.likes=запах цветка лотоса",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "likes",
                    "obj": "запах цветка лотоса",
                    "object_surface": "запах цветка лотоса",
                    "topic_keys": ["likes", "preference", "sensory_object"],
                    "trigger_keys": ["likes", "лотоса", "запах"],
                    "recall_mode": "contextual",
                    "spontaneous_recall": False,
                },
                "canonical_key": "user.likes.lotus_smell",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    store = _StoreSpy(
        semantic_hits=[(rose_claim, 0.57), (lotus_claim, 0.59)],
        lexical_hits=[(rose_claim, 0.58), (lotus_claim, 0.58)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="Роза",
            search_text="Роза",
            top_k=4,
        )
    )

    assert [item.record.id for item in result.candidates][:1] == ["claim-like-rose"]


def test_claim_retrieval_considers_recall_mode_and_spontaneous_recall() -> None:
    ambient_claim = _record(
        "claim-ambient",
        "user.likes=смотреть в глаза розе",
        metadata=ensure_memory_views(
            "user.likes=смотреть в глаза розе",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "likes",
                    "obj": "смотреть в глаза розе",
                    "object_surface": "смотреть в глаза Розе",
                    "topic_keys": ["likes", "preference", "activity"],
                    "trigger_keys": ["likes", "розе", "смотреть"],
                    "recall_mode": "ambient",
                    "spontaneous_recall": True,
                },
                "canonical_key": "user.likes.rose_eyes",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    contextual_claim = _record(
        "claim-contextual",
        "user.likes=запах цветка лотоса",
        metadata=ensure_memory_views(
            "user.likes=запах цветка лотоса",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "likes",
                    "obj": "запах цветка лотоса",
                    "object_surface": "запах цветка лотоса",
                    "topic_keys": ["likes", "preference", "sensory_object"],
                    "trigger_keys": ["likes", "лотоса", "запах"],
                    "recall_mode": "contextual",
                    "spontaneous_recall": False,
                },
                "canonical_key": "user.likes.lotus_smell",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    store = _StoreSpy(
        semantic_hits=[(ambient_claim, 0.58), (contextual_claim, 0.60)],
        lexical_hits=[(ambient_claim, 0.56), (contextual_claim, 0.59)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="что я люблю",
            search_text="что я люблю",
            top_k=4,
        )
    )

    assert [item.record.id for item in result.candidates][:1] == ["claim-ambient"]


def test_exact_self_fact_query_does_not_promote_unrelated_claim_noise() -> None:
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
    claim_record = _record(
        "claim-use-vscode",
        "user.uses=vs code",
        metadata=ensure_memory_views(
            "user.uses=vs code",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "uses",
                    "obj": "vs code",
                    "object_surface": "VS Code",
                    "topic_keys": ["uses", "tool"],
                    "trigger_keys": ["uses", "vs", "code"],
                    "recall_mode": "contextual",
                    "spontaneous_recall": False,
                },
                "canonical_key": "user.uses.vs_code",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    store = _StoreSpy(
        semantic_hits=[(claim_record, 0.72), (fact_record, 0.58)],
        lexical_hits=[(claim_record, 0.74), (fact_record, 0.61)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="what is my gpu?",
            search_text="gpu rtx 3050 ti",
            entity_keys=["gpu", "rtx_3050_ti"],
            top_k=4,
        )
    )

    assert [item.record.id for item in result.candidates][:1] == ["fact-gpu"]


def test_exact_claim_query_prefers_exact_ownership_claim_over_unrelated_flower_claim() -> None:
    owns_claim = _record(
        "claim-own-fridge",
        "user.owns=fridge samsung",
        metadata=ensure_memory_views(
            "user.owns=fridge samsung",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "owns",
                    "obj": "fridge samsung",
                    "object_surface": "Samsung fridge",
                    "topic_keys": ["owns", "ownership", "appliance"],
                    "trigger_keys": ["owns", "fridge", "samsung"],
                    "recall_mode": "exact",
                    "spontaneous_recall": False,
                },
                "canonical_key": "user.owns.fridge_samsung",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    flower_claim = _record(
        "claim-like-lotus",
        "user.likes=запах цветка лотоса",
        metadata=ensure_memory_views(
            "user.likes=запах цветка лотоса",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "likes",
                    "obj": "запах цветка лотоса",
                    "object_surface": "запах цветка лотоса",
                    "topic_keys": ["likes", "preference", "flower"],
                    "trigger_keys": ["likes", "запах", "лотоса"],
                    "recall_mode": "contextual",
                    "spontaneous_recall": False,
                },
                "canonical_key": "user.likes.lotus_smell",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    store = _StoreSpy(
        semantic_hits=[(flower_claim, 0.76), (owns_claim, 0.64)],
        lexical_hits=[(flower_claim, 0.74), (owns_claim, 0.67)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="what do i own?",
            search_text="what do i own",
            top_k=4,
        )
    )

    assert [item.record.id for item in result.candidates][:1] == ["claim-own-fridge"]


def test_exact_python_fact_query_demotes_old_dialog_episode_noise() -> None:
    fact_record = _record(
        "fact-python",
        "user.environment_runtime_python=3.11",
        metadata=ensure_memory_views(
            "user.environment_runtime_python=3.11",
            metadata={
                "fact": {"predicate": "environment_runtime_python", "subject": "user", "value": "3.11"},
                "canonical_key": "user.environment_runtime_python",
            },
        ),
        memory_type=MemoryType.FACT,
        level=MemoryLevel.L3_SEMANTIC,
    )
    episode_record = _record(
        "episode-python-discussion",
        "We discussed Python migration for memory ingestion.",
        metadata=ensure_memory_views(
            "We discussed Python migration for memory ingestion.",
            metadata={"episode": {"topic": "python migration"}},
        ),
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
    )
    store = _StoreSpy(
        semantic_hits=[(episode_record, 0.80), (fact_record, 0.58)],
        lexical_hits=[(episode_record, 0.77), (fact_record, 0.60)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="what is my python?",
            search_text="python 3.11",
            entity_keys=["python", "python_3_11"],
            top_k=4,
        )
    )

    assert [item.record.id for item in result.candidates][:1] == ["fact-python"]


def test_claim_retrieval_skips_document_level_claims_for_generic_claim_queries() -> None:
    live_claim = _record(
        "claim-use-vscode",
        "user.uses=vs code",
        metadata=ensure_memory_views(
            "user.uses=vs code",
            metadata={
                "claim": {
                    "subject": "user",
                    "predicate": "uses",
                    "obj": "vs code",
                    "object_surface": "VS Code",
                    "topic_keys": ["uses", "tool"],
                    "trigger_keys": ["uses", "vs", "code"],
                    "recall_mode": "contextual",
                    "spontaneous_recall": False,
                },
                "canonical_key": "user.uses.vs_code",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L3_SEMANTIC,
    )
    document_claim = _record(
        "claim-doc-ollama",
        "document.uses=ollama client",
        metadata=ensure_memory_views(
            "document.uses=ollama client",
            metadata={
                "claim": {
                    "subject": "document",
                    "predicate": "uses",
                    "obj": "ollama client",
                    "object_surface": "ollama client",
                    "topic_keys": ["uses", "tool"],
                    "trigger_keys": ["uses", "ollama", "client"],
                    "recall_mode": "document",
                    "spontaneous_recall": False,
                },
                "canonical_key": "document.uses.ollama_client",
                "document_id": "doc:ollama-guide",
            },
        ),
        memory_type=MemoryType.CLAIM,
        level=MemoryLevel.L4_DOCUMENT,
    )
    store = _StoreSpy(
        semantic_hits=[(document_claim, 0.80), (live_claim, 0.62)],
        lexical_hits=[(document_claim, 0.79), (live_claim, 0.64)],
    )
    retriever = HybridRetriever(store=store)

    result = retriever.retrieve(
        RetrievalQuery(
            query_text="what do i use?",
            search_text="what do i use",
            top_k=4,
        )
    )

    ids = [item.record.id for item in result.candidates]
    assert "claim-doc-ollama" not in ids
    assert ids[:1] == ["claim-use-vscode"]


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


def test_build_memory_views_returns_only_keys() -> None:
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

    # Keys may be empty if there are no entities/numeric facts
    assert "entity_keys" in rebuilt or len(rebuilt) == 0
    assert rebuilt.get("entity_keys") == first.get("entity_keys")
    assert rebuilt.get("numeric_keys") == first.get("numeric_keys")
    assert rebuilt_twice.get("entity_keys") == rebuilt.get("entity_keys")
    assert rebuilt_twice.get("numeric_keys") == rebuilt.get("numeric_keys")


def test_build_memory_views_does_not_leak_generic_source_into_search_text() -> None:
    search_text = record_search_text(
        "привет. меня зовут Паша",
        metadata={"source": "api"},
    )

    assert "привет" in search_text.lower()
    assert "паша" in search_text.lower()


def test_build_memory_views_does_not_leak_generic_source_into_keys() -> None:
    views = build_memory_views(
        "привет. меня зовут Паша",
        metadata={"source": "api"},
    )

    assert "api" not in list(views.get("entity_keys") or [])
    assert "api" not in list(views.get("numeric_keys") or [])


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
