from __future__ import annotations

from memory.memory_policy import MemoryPolicy
from memory.memory_models import MemorySourceKind
from memory.storage_profile import compact_metadata_payload, prepare_storage_metadata, storage_schema_for_memory_type


def test_compact_storage_profile_strips_raw_analysis_and_nested_debug_payloads() -> None:
    compact = compact_metadata_payload(
        {
            "memory_analysis": {"anchors": [{"kind": "gpu"}]},
            "claim_candidates": [{"predicate": "uses"}],
            "search_text": "gpu rtx 3050 ti",
            "normalized_text": "gpu rtx 3050 ti",
            "canonical_text": "RTX 3050 Ti",
            "memory_views": {
                "entity_keys": ["gpu", "rtx_3050_ti"],
                "numeric_keys": ["vram:4gb"],
                "search_text": "gpu rtx 3050 ti",
            },
            "write_policy": {
                "action": "allow",
                "reason": "new_fact",
                "allow_write": True,
                "allow_supersede": False,
                "signals": {"old_conf": 0.3, "new_conf": 0.9},
            },
            "fact": {
                "subject": "user",
                "predicate": "environment_gpu_model",
                "value": "RTX 3050 Ti",
                "confidence": 0.92,
                "importance": 0.86,
                "evidence": "у меня rtx 3050 ti",
                "metadata": {"raw": True},
                "text": "user.environment_gpu_model=RTX 3050 Ti",
                "source_kind": "structured_fact",
            },
            "claim": {
                "subject": "user",
                "predicate": "uses",
                "obj": "vs code",
                "object_surface": "VS Code",
                "confidence": 0.88,
                "topic_keys": ["uses", "tool"],
                "trigger_keys": ["uses", "vs", "code"],
                "promotion_level": "strong_claim",
                "promotion_signals": {"specificity": 0.8},
                "evidence_text": "я использую VS Code",
            },
        }
    )

    assert "memory_analysis" not in compact
    assert "claim_candidates" not in compact
    assert "search_text" not in compact
    assert "normalized_text" not in compact
    assert "canonical_text" not in compact
    assert compact["memory_views"] == {
        "entity_keys": ["gpu", "rtx_3050_ti"],
        "numeric_keys": ["vram:4gb"],
    }
    assert compact["write_policy"] == {
        "action": "allow",
        "reason": "new_fact",
        "allow_write": True,
        "allow_supersede": False,
    }
    assert compact["fact"] == {
        "subject": "user",
        "predicate": "environment_gpu_model",
        "value": "RTX 3050 Ti",
        "confidence": 0.92,
        "importance": 0.86,
        "source_kind": "structured_fact",
    }
    assert compact["claim"] == {
        "subject": "user",
        "predicate": "uses",
        "obj": "vs code",
        "object_surface": "VS Code",
        "confidence": 0.88,
        "topic_keys": ["uses", "tool"],
        "trigger_keys": ["uses", "vs", "code"],
        "promotion_level": "strong_claim",
    }


def test_debug_storage_profile_keeps_raw_analysis_and_debug_payloads() -> None:
    sanitized = MemoryPolicy.sanitize_metadata_for_storage(
        metadata={
            "memory_analysis": {"anchors": [{"kind": "gpu"}]},
            "search_text": "gpu rtx 3050 ti",
            "memory_views": {
                "entity_keys": ["gpu", "rtx_3050_ti"],
                "search_text": "gpu rtx 3050 ti",
            },
            "write_policy": {
                "action": "allow",
                "reason": "new_fact",
                "signals": {"new_conf": 0.9},
            },
        },
        source_kind=MemorySourceKind.USER,
        storage_profile="debug",
    )

    assert "memory_analysis" in sanitized
    assert sanitized["search_text"] == "gpu rtx 3050 ti"
    assert dict(sanitized.get("memory_views") or {}).get("search_text") == "gpu rtx 3050 ti"
    assert dict(sanitized.get("write_policy") or {}).get("signals") == {"new_conf": 0.9}


def test_prepare_storage_metadata_is_single_compact_entrypoint_for_write_path() -> None:
    prepared = prepare_storage_metadata(
        metadata={
            "entities": {"software": ["Python"]},
            "runtime_entities": {"os": ["Windows"]},
            "tags": ["intent_chat", "topic_python"],
            "thinking": "hidden reasoning",
            "memory_analysis": {"anchors": [{"kind": "gpu"}]},
            "memory_views": {
                "entity_keys": ["python"],
                "numeric_keys": ["version:3.11"],
                "search_text": "python 3.11",
            },
        },
        storage_profile="compact",
        memory_type="message",
        source_kind=MemorySourceKind.ASSISTANT_REPLY,
        thinking="hidden reasoning",
    )

    assert "entities" not in prepared
    assert "runtime_entities" not in prepared
    assert "tags" not in prepared
    assert "thinking" not in prepared
    assert "memory_analysis" not in prepared
    assert prepared["source_kind"] == "assistant_reply"
    assert prepared["assistant_thinking_stripped"] is True
    assert prepared["memory_views"] == {
        "entity_keys": ["python"],
        "numeric_keys": ["version:3.11"],
    }


def test_storage_schema_for_fact_claim_episode_and_document_chunk_is_frozen() -> None:
    fact_schema = storage_schema_for_memory_type("fact")
    claim_schema = storage_schema_for_memory_type("claim")
    identity_core_schema = storage_schema_for_memory_type("identity_core")
    episode_schema = storage_schema_for_memory_type("episode")
    chunk_schema = storage_schema_for_memory_type("document_chunk")
    message_schema = storage_schema_for_memory_type("message")

    assert fact_schema["always"] == ["canonical_key", "fact", "governor_reason", "parallel_with", "relation", "write_policy"]
    assert claim_schema["always"] == [
        "canonical_key",
        "chunk_id",
        "claim",
        "doc_id",
        "document_claim",
        "document_id",
        "topic_keys",
        "trigger_keys",
        "write_policy",
    ]
    assert identity_core_schema["always"] == [
        "identity_core",
        "identity_core_key",
        "source_kind",
        "source_role",
        "write_policy",
    ]
    assert episode_schema["always"] == [
        "decisions",
        "dialog_episode",
        "entity_keys",
        "open_questions",
        "participants",
        "salience",
        "summary_reasoning",
        "summary_short",
        "topic",
        "topic_keys",
        "turn_ids",
    ]
    assert chunk_schema["always"] == [
        "chunk_id",
        "chunk_index",
        "claims",
        "doc_id",
        "document_id",
        "extension",
        "filename",
        "memory_entities",
        "memory_tags",
        "memory_views",
        "numeric_facts",
        "path",
        "section_label",
        "source_path",
        "stable_facts",
        "title",
    ]
    assert "assistant_noise_archived" in set(message_schema["always"])
    assert "lifecycle_reason" in set(message_schema["always"])
    assert "runtime_entities" not in set(message_schema["always"])
    assert "tags" not in set(message_schema["always"])
    assert "entities" not in set(message_schema["always"])
    assert "memory_analysis" in set(fact_schema["debug_only"])
    assert "claim_candidates" in set(claim_schema["debug_only"])


def test_compact_storage_profile_enforces_fact_episode_and_document_chunk_schema() -> None:
    fact = compact_metadata_payload(
        {
            "fact": {
                "subject": "user",
                "predicate": "environment_gpu_model",
                "value": "RTX 3050 Ti",
                "confidence": 0.92,
            },
            "canonical_key": "user.environment_gpu_model",
            "governor_reason": "singleton_group_higher_score",
            "relation": "environment",
            "write_policy": {"action": "allow", "reason": "new_fact", "signals": {"x": 1}},
            "memory_entities": [{"type": "gpu_model"}],
            "memory_analysis": {"anchors": [{"kind": "gpu"}]},
        },
        memory_type="fact",
    )
    episode = compact_metadata_payload(
        {
            "dialog_episode": {
                "id": "episode:1",
                "topic": "memory design",
                "turn_ids": ["t1", "t2"],
                "summary_short": "Discussed memory design.",
                "summary_reasoning": "Compared storage options.",
                "decisions": ["keep assistant thoughts debug-only"],
                "open_questions": [],
                "participants": ["user", "assistant"],
                "salience": 0.8,
                "topic_keys": ["memory"],
                "entity_keys": ["assistant_thoughts"],
                "metadata": {"raw": True},
            },
            "topic": "memory design",
            "summary_short": "Discussed memory design.",
            "summary_reasoning": "Compared storage options.",
            "turn_ids": ["t1", "t2"],
            "memory_views": {"entity_keys": ["should_drop"]},
            "memory_analysis": {"anchors": [{"kind": "topic"}]},
        },
        memory_type="episode",
    )
    chunk = compact_metadata_payload(
        {
            "document_id": "doc:1",
            "chunk_id": "chunk:1",
            "chunk_index": 0,
            "section_label": "LLM helpers",
            "memory_views": {"entity_keys": ["ollama"], "numeric_keys": ["version:3.11"], "search_text": "raw"},
            "claims": [{"subject": "document", "predicate": "uses", "obj": "ollama", "debug": True}],
            "memory_entities": [{"type": "tool_name", "canonical": "ollama"}],
            "numeric_facts": [{"kind": "python_version", "value": "3.11"}],
            "stable_facts": [{"kind": "tool", "value": "ollama"}],
            "memory_tags": ["topic_code"],
            "search_text": "raw chunk",
            "memory_analysis": {"anchors": [{"kind": "tool"}]},
            "weird_extra": True,
        },
        memory_type="document_chunk",
    )
    identity_core = compact_metadata_payload(
        {
            "identity_core": {
                "key": "addressing.canonical_name",
                "value": "Pasha",
                "confidence": 0.91,
                "source": "fact:self_identification",
                "requires_confirmation_to_override": True,
                "updated_at": 10.0,
                "debug": {"raw": True},
            },
            "identity_core_key": "addressing.canonical_name",
            "source_kind": "structured_fact",
            "source_role": "user",
            "write_policy": {"action": "allow", "reason": "canonical_name_strong_self_identification", "allow_write": True, "signals": {"x": 1}},
            "memory_analysis": {"anchors": [{"kind": "identity"}]},
        },
        memory_type="identity_core",
    )

    assert set(fact.keys()) == {"fact", "canonical_key", "governor_reason", "relation", "write_policy"}
    assert set(episode.keys()) == {"dialog_episode", "topic", "summary_short", "summary_reasoning", "turn_ids"}
    assert set(chunk.keys()) == {
        "document_id",
        "chunk_id",
        "chunk_index",
        "section_label",
        "memory_views",
        "claims",
        "memory_entities",
        "numeric_facts",
        "stable_facts",
        "memory_tags",
    }
    assert chunk["memory_views"] == {"entity_keys": ["ollama"], "numeric_keys": ["version:3.11"]}
    assert identity_core == {
        "identity_core": {
            "key": "addressing.canonical_name",
            "value": "Pasha",
            "confidence": 0.91,
            "source": "fact:self_identification",
            "requires_confirmation_to_override": True,
            "updated_at": 10.0,
        },
        "identity_core_key": "addressing.canonical_name",
        "source_kind": "structured_fact",
        "source_role": "user",
        "write_policy": {
            "action": "allow",
            "reason": "canonical_name_strong_self_identification",
            "allow_write": True,
        },
    }
