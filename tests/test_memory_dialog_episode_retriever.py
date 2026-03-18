from __future__ import annotations

import time

from memory.dialog_episode_retriever import DialogEpisodeRetriever
from memory.memory_models import MemoryLevel, MemoryRecord, MemoryScope, MemoryType, RetrievalQuery


def _record(
    record_id: str,
    text: str,
    *,
    metadata: dict | None = None,
    source_event_id: str = "",
    memory_type: MemoryType = MemoryType.MESSAGE,
    level: MemoryLevel = MemoryLevel.L2_EPISODIC,
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
        importance=0.6,
        confidence=0.7,
        created_at=now,
        updated_at=now,
        source_event_id=source_event_id,
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


def test_dialog_episode_retriever_returns_summary_reasoning_decisions_and_supporting_turns() -> None:
    episode_record = _record(
        "episode-memory",
        "Discussed memory architecture.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-memory",
                "topic": "memory",
                "turn_ids": ["turn-1", "turn-2"],
                "summary_short": "Discussed memory architecture.",
                "summary_reasoning": "We separated facts, claims, episodes, and docs to keep retrieval clean.",
                "decisions": ["Do not store assistant thoughts in long-term memory."],
                "open_questions": ["How to synthesize dialog episodes from turns?"],
                "participants": ["user", "assistant"],
                "salience": 0.86,
                "topic_keys": ["memory", "retrieval"],
                "entity_keys": ["memory_manager"],
                "created_at": 100.0,
                "updated_at": 120.0,
                "supporting_turns": [
                    {"turn_id": "turn-1", "role": "user", "text": "Что мы обсуждали про память?", "ts": 100.0},
                    {"turn_id": "turn-2", "role": "assistant", "text": "Разделяем facts, claims, episodes, docs.", "ts": 110.0},
                ],
            }
        },
    )
    store = _StoreSpy(semantic_hits=[(episode_record, 0.71)], lexical_hits=[(episode_record, 0.69)])
    retriever = DialogEpisodeRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(query_text="что мы обсуждали про память", search_text="memory discussion", top_k=3)
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.summary_short == "Discussed memory architecture."
    assert "keep retrieval clean" in row.summary_reasoning
    assert row.decisions == ["Do not store assistant thoughts in long-term memory."]
    assert len(row.supporting_turns) == 2
    assert row.supporting_turns[0].turn_id == "turn-1"


def test_dialog_episode_retriever_prefers_reasoning_episode_for_why_query() -> None:
    target_episode = _record(
        "episode-assistant-thoughts",
        "We refused assistant thoughts.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-assistant-thoughts",
                "topic": "memory",
                "turn_ids": ["evt:1", "evt:2"],
                "summary_short": "Rejected assistant-thought storage.",
                "summary_reasoning": "We refused assistant thoughts because they pollute retrieval and create self-looping noise.",
                "decisions": ["Do not store assistant thoughts."],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.9,
                "topic_keys": ["memory", "assistant_thoughts", "retrieval"],
                "entity_keys": ["assistant_thoughts"],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    distractor_episode = _record(
        "episode-web",
        "We tuned web verification.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-web",
                "topic": "web",
                "turn_ids": ["evt:3", "evt:4"],
                "summary_short": "Adjusted web verification.",
                "summary_reasoning": "We changed web isolation rules to avoid stale evidence bleed.",
                "decisions": ["Keep verify_only for unstable numeric turns."],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.82,
                "topic_keys": ["web", "verification"],
                "entity_keys": ["web"],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    store = _StoreSpy(
        semantic_hits=[(distractor_episode, 0.75), (target_episode, 0.70)],
        lexical_hits=[(distractor_episode, 0.70), (target_episode, 0.71)],
    )
    retriever = DialogEpisodeRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(
            query_text="почему мы отказались от мыслей ассистента",
            search_text="why rejected assistant thoughts",
            top_k=3,
        )
    )

    assert rows[0].record.id == "episode-assistant-thoughts"
    assert "pollute retrieval" in rows[0].summary_reasoning


def test_dialog_episode_retriever_resolves_supporting_turns_from_turn_ids() -> None:
    episode_record = _record(
        "episode-web-decisions",
        "Discussed web decisions.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-web-decisions",
                "topic": "web",
                "turn_ids": ["evt:w1", "evt:w2", "evt:w3", "evt:w4"],
                "summary_short": "Discussed web verification.",
                "summary_reasoning": "We aligned verify_only and memory isolation for self recall.",
                "decisions": ["Keep self facts stronger than web evidence."],
                "open_questions": ["How far to isolate WEB_TEMP?"],
                "participants": ["user", "assistant"],
                "salience": 0.88,
                "topic_keys": ["web", "verification", "self_recall"],
                "entity_keys": ["web_temp"],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    supporting_records = [
        _record(
            "msg-1",
            "Надо оставить SELF_FACTS сильнее WEB_EVIDENCE.",
            source_event_id="evt:w1",
            metadata={"source_kind": "user"},
        ),
        _record(
            "msg-2",
            "Согласна, веб не должен доминировать exact self recall.",
            source_event_id="evt:w2",
            metadata={"source_kind": "assistant_reply"},
        ),
        _record(
            "msg-3",
            "Что делаем с WEB_TEMP дальше?",
            source_event_id="evt:w3",
            metadata={"source_kind": "user"},
        ),
        _record(
            "msg-4",
            "Пока изолируем его только от self_memory_exact.",
            source_event_id="evt:w4",
            metadata={"source_kind": "assistant_reply"},
        ),
    ]
    store = _StoreSpy(
        semantic_hits=[(episode_record, 0.73)],
        lexical_hits=[(episode_record, 0.76)],
        records=[episode_record, *supporting_records],
    )
    retriever = DialogEpisodeRetriever(store=store, supporting_turn_limit=4)

    rows = retriever.retrieve(
        query=RetrievalQuery(
            query_text="что решили по вебу",
            search_text="web decisions",
            top_k=2,
        )
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.decisions == ["Keep self facts stronger than web evidence."]
    assert 2 <= len(row.supporting_turns) <= 4
    assert any("SELF_FACTS" in item.text for item in row.supporting_turns)
    assert any(item.role == "assistant" for item in row.supporting_turns)
