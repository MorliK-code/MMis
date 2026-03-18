from __future__ import annotations

import time

from memory.document_retrieval import DocumentRetriever
from memory.memory_models import MemoryLevel, MemoryRecord, MemoryScope, MemoryType, RetrievalQuery


def _record(
    record_id: str,
    text: str,
    *,
    metadata: dict | None = None,
    memory_type: MemoryType = MemoryType.DOCUMENT_CHUNK,
    level: MemoryLevel = MemoryLevel.L4_DOCUMENT,
) -> MemoryRecord:
    now = time.time()
    return MemoryRecord(
        id=record_id,
        text=text,
        memory_type=memory_type,
        level=level,
        scope=MemoryScope.PROJECT,
        namespace="default",
        metadata=dict(metadata or {}),
        importance=0.6,
        confidence=0.7,
        created_at=now,
        updated_at=now,
    )


class _StoreSpy:
    def __init__(self, *, semantic_hits=None, lexical_hits=None, records=None) -> None:
        self.reindex_required = False
        self.semantic_hits = list(semantic_hits or [])
        self.lexical_hits = list(lexical_hits or [])
        self.records = list(records or [])

    def semantic_search(self, *, query_text, top_k, namespace, scopes, include_stale, metadata_filters):
        _ = (query_text, top_k, namespace, scopes, include_stale)
        return self._filter_hits(self.semantic_hits, metadata_filters)

    def lexical_search(self, *, query_text, top_k, namespace, scopes, include_stale, metadata_filters):
        _ = (query_text, top_k, namespace, scopes, include_stale)
        return self._filter_hits(self.lexical_hits, metadata_filters)

    def iter_records(self, *, namespace=None):
        _ = namespace
        return list(self.records or [])

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


def test_document_retrieval_returns_relevant_chunk_and_section_summary_for_chapter_query() -> None:
    document = _record(
        "doc:novel",
        "Novel outline",
        memory_type=MemoryType.DOCUMENT,
        metadata={"doc_id": "doc:novel", "title": "Novel"},
    )
    chapter_three_chunk = _record(
        "chunk:novel:3",
        "Chapter 3. The character dies in the river at the end of the chapter.",
        metadata={"document_id": "doc:novel", "doc_id": "doc:novel", "section_label": "Chapter 3"},
    )
    chapter_two_chunk = _record(
        "chunk:novel:2",
        "Chapter 2. The character travels to the city.",
        metadata={"document_id": "doc:novel", "doc_id": "doc:novel", "section_label": "Chapter 2"},
    )
    section_summary = _record(
        "docsum:novel:3",
        "Chapter 3: the character dies in the river.",
        memory_type=MemoryType.SUMMARY,
        metadata={
            "document_id": "doc:novel",
            "doc_id": "doc:novel",
            "summary_kind": "section",
            "section_label": "Chapter 3",
            "document_summary": {
                "id": "docsum:novel:3",
                "document_id": "doc:novel",
                "text": "Chapter 3: the character dies in the river.",
                "summary_kind": "section",
                "source_chunk_ids": ["chunk:novel:3"],
                "topic_keys": ["chapter", "story"],
                "entity_keys": [],
            },
        },
    )
    store = _StoreSpy(
        semantic_hits=[(chapter_three_chunk, 0.74), (chapter_two_chunk, 0.68), (section_summary, 0.72)],
        lexical_hits=[(chapter_three_chunk, 0.78), (chapter_two_chunk, 0.66), (section_summary, 0.81)],
        records=[document],
    )
    retriever = DocumentRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(query_text="что было в главе 3", search_text="chapter 3 story", top_k=3)
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.document_id == "doc:novel"
    assert row.section_summary is not None
    assert "Chapter 3" in row.section_summary.text
    assert any("dies in the river" in item.text for item in row.relevant_chunks)


def test_document_retrieval_finds_where_ollama_is_called() -> None:
    document = _record(
        "doc:code",
        "Codebase outline",
        memory_type=MemoryType.DOCUMENT,
        metadata={"doc_id": "doc:code", "title": "Codebase"},
    )
    ollama_chunk = _record(
        "chunk:code:1",
        "def call_ollama(prompt):\n    return ollama.chat(model='qwen', messages=[{'role': 'user', 'content': prompt}])",
        metadata={
            "document_id": "doc:code",
            "doc_id": "doc:code",
            "section_label": "def call_ollama",
            "language": "python",
            "path": "src/llm.py",
        },
    )
    other_chunk = _record(
        "chunk:code:2",
        "def load_config():\n    return {'mode': 'dev'}",
        metadata={
            "document_id": "doc:code",
            "doc_id": "doc:code",
            "section_label": "def load_config",
            "language": "python",
            "path": "src/config.py",
        },
    )
    section_summary = _record(
        "docsum:code:1",
        "def call_ollama: wraps ollama.chat for local LLM calls.",
        memory_type=MemoryType.SUMMARY,
        metadata={
            "document_id": "doc:code",
            "doc_id": "doc:code",
            "summary_kind": "section",
            "section_label": "def call_ollama",
            "document_summary": {
                "id": "docsum:code:1",
                "document_id": "doc:code",
                "text": "def call_ollama: wraps ollama.chat for local LLM calls.",
                "summary_kind": "section",
                "source_chunk_ids": ["chunk:code:1"],
                "topic_keys": ["ollama", "code", "python"],
                "entity_keys": ["ollama"],
            },
        },
    )
    store = _StoreSpy(
        semantic_hits=[(ollama_chunk, 0.69), (other_chunk, 0.71), (section_summary, 0.73)],
        lexical_hits=[(ollama_chunk, 0.86), (other_chunk, 0.58), (section_summary, 0.80)],
        records=[document],
    )
    retriever = DocumentRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(query_text="где в коде вызывается ollama", search_text="where code calls ollama", top_k=3)
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.document_id == "doc:code"
    assert row.relevant_chunks[0].id == "chunk:code:1"
    assert "ollama.chat" in row.relevant_chunks[0].text
    assert row.section_summary is not None
    assert "call_ollama" in row.section_summary.text


def test_document_retrieval_can_surface_document_claims_for_class_declaration_queries() -> None:
    document = _record(
        "doc:memory",
        "Memory module",
        memory_type=MemoryType.DOCUMENT,
        metadata={"doc_id": "doc:memory", "title": "memory_manager.py"},
    )
    class_chunk = _record(
        "chunk:memory:0",
        "class MemoryManager:\n    def __init__(self):\n        self.ready = True",
        metadata={
            "document_id": "doc:memory",
            "doc_id": "doc:memory",
            "section_label": "class MemoryManager",
            "language": "python",
            "path": "memory/memory_manager.py",
        },
    )
    class_claim = _record(
        "docclaim:memory:0:0",
        "code.declares=memorymanager",
        memory_type=MemoryType.CLAIM,
        metadata={
            "document_id": "doc:memory",
            "doc_id": "doc:memory",
            "chunk_id": "chunk:memory:0",
            "claim": {
                "subject": "code",
                "predicate": "declares",
                "obj": "memorymanager",
                "object_surface": "MemoryManager",
                "topic_keys": ["code", "class", "declaration"],
                "trigger_keys": ["memorymanager", "class"],
            },
            "document_claim": {
                "id": "docclaim:memory:0:0",
                "document_id": "doc:memory",
                "chunk_id": "chunk:memory:0",
                "subject": "code",
                "predicate": "declares",
                "obj": "memorymanager",
                "object_surface": "MemoryManager",
                "topic_keys": ["code", "class", "declaration"],
                "trigger_keys": ["memorymanager", "class"],
                "evidence_text": "class MemoryManager:",
            },
        },
    )
    store = _StoreSpy(
        semantic_hits=[(class_chunk, 0.70), (class_claim, 0.68)],
        lexical_hits=[(class_chunk, 0.88), (class_claim, 0.80)],
        records=[document],
    )
    retriever = DocumentRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(
            query_text="где объявляется класс MemoryManager",
            search_text="where class memorymanager is declared",
            top_k=3,
        )
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.document_id == "doc:memory"
    assert any("class MemoryManager" in item.text for item in row.relevant_chunks)
    assert row.document_claims
    assert row.document_claims[0].predicate == "declares"
    assert row.document_claims[0].object_surface == "MemoryManager"
