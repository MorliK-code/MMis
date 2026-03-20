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


def test_dialog_episode_retriever_finds_anchor_only_episode_when_summary_is_weak() -> None:
    target_episode = _record(
        "episode-memory-plan",
        "Discussed this.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-memory-plan",
                "topic": "general_dialog",
                "turn_ids": ["evt:m1", "evt:m2"],
                "summary_short": "Discussed this.",
                "summary_reasoning": "We aligned the memory roadmap around facts, claims, and dialog episodes.",
                "decisions": ["First stabilize memory, then return to web."],
                "open_questions": ["How should soft singleton groups work?"],
                "participants": ["user", "assistant"],
                "salience": 0.88,
                "topic_keys": ["memory", "plan"],
                "entity_keys": ["soft_singleton", "memory_governor"],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    distractor_episode = _record(
        "episode-greeting",
        "Discussed greetings.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-greeting",
                "topic": "general_dialog",
                "turn_ids": ["evt:g1", "evt:g2"],
                "summary_short": "Discussed greetings.",
                "summary_reasoning": "We exchanged short greetings.",
                "decisions": [],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.42,
                "topic_keys": ["greeting"],
                "entity_keys": [],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    store = _StoreSpy(records=[target_episode, distractor_episode])
    retriever = DialogEpisodeRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(
            query_text="What was our plan for memory?",
            search_text="memory plan",
            top_k=2,
        )
    )

    assert rows
    assert rows[0].record.id == "episode-memory-plan"
    assert rows[0].source == "episode_anchor_channel"
    assert "First stabilize memory, then return to web." in rows[0].decisions


def test_dialog_episode_retriever_anchor_score_can_beat_weaker_summary_match() -> None:
    target_episode = _record(
        "episode-python-policy",
        "Discussed this.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-python-policy",
                "topic": "general_dialog",
                "turn_ids": ["evt:p1", "evt:p2"],
                "summary_short": "Discussed this.",
                "summary_reasoning": "We decided singleton latest wins for environment.python to keep active facts canonical.",
                "decisions": ["Use singleton_latest_wins for environment.python."],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.9,
                "topic_keys": ["memory", "python"],
                "entity_keys": ["environment_python", "singleton_latest_wins"],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    distractor_episode = _record(
        "episode-memory-smalltalk",
        "Discussed memory setup in general.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-memory-smalltalk",
                "topic": "memory",
                "turn_ids": ["evt:s1", "evt:s2"],
                "summary_short": "Discussed memory setup in general.",
                "summary_reasoning": "We talked broadly about memory setup.",
                "decisions": [],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.6,
                "topic_keys": ["memory"],
                "entity_keys": [],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    store = _StoreSpy(
        semantic_hits=[(distractor_episode, 0.80), (target_episode, 0.60)],
        lexical_hits=[(distractor_episode, 0.78), (target_episode, 0.58)],
        records=[target_episode, distractor_episode],
    )
    retriever = DialogEpisodeRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(
            query_text="Why did we choose singleton latest wins for python?",
            search_text="python singleton plan",
            top_k=2,
        )
    )

    assert rows
    assert rows[0].record.id == "episode-python-policy"
    assert "singleton latest wins" in rows[0].summary_reasoning.lower()


def test_dialog_episode_retriever_uses_focus_keys_for_anchor_matching() -> None:
    target_episode = _record(
        "episode-environment-plan",
        "Dialog episode captured.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-environment-plan",
                "topic": "general_dialog",
                "turn_ids": ["evt:f1", "evt:f2"],
                "summary_short": "Dialog episode captured.",
                "summary_reasoning": "Context: We aligned the python environment on windows.",
                "decisions": ["Keep one python environment path on windows."],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.87,
                "topic_keys": [],
                "entity_keys": [],
                "focus_keys": ["python", "windows", "environment"],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    distractor_episode = _record(
        "episode-other-plan",
        "Dialog episode captured.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-other-plan",
                "topic": "general_dialog",
                "turn_ids": ["evt:o1", "evt:o2"],
                "summary_short": "Dialog episode captured.",
                "summary_reasoning": "Context: We discussed generic planning.",
                "decisions": ["Keep the roadmap visible."],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.7,
                "topic_keys": [],
                "entity_keys": [],
                "focus_keys": ["roadmap"],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    store = _StoreSpy(records=[target_episode, distractor_episode])
    retriever = DialogEpisodeRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(
            query_text="What did we decide about the python environment on windows?",
            search_text="python windows environment",
            top_k=2,
        )
    )

    assert rows
    assert rows[0].record.id == "episode-environment-plan"


def test_dialog_episode_retriever_uses_entity_keys_for_episode_lookup() -> None:
    target_episode = _record(
        "episode-python-env",
        "Dialog episode captured.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-python-env",
                "topic": "general_dialog",
                "turn_ids": ["evt:e1", "evt:e2"],
                "summary_short": "Dialog episode captured.",
                "summary_reasoning": "We aligned one python environment path on windows.",
                "decisions": ["Keep one python environment path on windows."],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.84,
                "topic_keys": [],
                "entity_keys": ["environment_python_path", "windows"],
                "focus_keys": [],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    distractor_episode = _record(
        "episode-memory-overview",
        "Discussed memory setup.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-memory-overview",
                "topic": "memory",
                "turn_ids": ["evt:m1", "evt:m2"],
                "summary_short": "Discussed memory setup.",
                "summary_reasoning": "We talked broadly about memory setup.",
                "decisions": [],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.7,
                "topic_keys": ["memory"],
                "entity_keys": [],
                "focus_keys": [],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    store = _StoreSpy(records=[target_episode, distractor_episode])
    retriever = DialogEpisodeRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(
            query_text="What did we decide about the python path on windows?",
            search_text="python path windows",
            top_k=2,
        )
    )

    assert rows
    assert rows[0].record.id == "episode-python-env"


def test_dialog_episode_retriever_uses_open_questions_for_continuity_queries() -> None:
    target_episode = _record(
        "episode-open-fusion",
        "Dialog episode captured.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-open-fusion",
                "topic": "memory",
                "turn_ids": ["evt:o1", "evt:o2"],
                "summary_short": "Dialog episode captured.",
                "summary_reasoning": "Current direction: memory fusion retrieval.",
                "decisions": [],
                "open_questions": ["How should fusion retrieval combine episodes and facts?"],
                "participants": ["user", "assistant"],
                "salience": 0.81,
                "topic_keys": ["memory"],
                "entity_keys": ["fusion_retrieval"],
                "focus_keys": ["memory", "fusion", "retrieval"],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    distractor_episode = _record(
        "episode-closed-web",
        "Adjusted web verification.",
        memory_type=MemoryType.EPISODE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={
            "dialog_episode": {
                "id": "episode-closed-web",
                "topic": "web",
                "turn_ids": ["evt:w1", "evt:w2"],
                "summary_short": "Adjusted web verification.",
                "summary_reasoning": "We isolated stale evidence bleed.",
                "decisions": ["Keep verify_only for unstable numeric turns."],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.76,
                "topic_keys": ["web"],
                "entity_keys": [],
                "focus_keys": ["verification"],
                "created_at": 100.0,
                "updated_at": 120.0,
            }
        },
    )
    store = _StoreSpy(records=[target_episode, distractor_episode])
    retriever = DialogEpisodeRetriever(store=store)

    rows = retriever.retrieve(
        query=RetrievalQuery(
            query_text="Что осталось открытым по fusion retrieval?",
            search_text="open question fusion retrieval",
            top_k=2,
        )
    )

    assert rows
    assert rows[0].record.id == "episode-open-fusion"
    assert rows[0].episode.open_questions == ["How should fusion retrieval combine episodes and facts?"]
