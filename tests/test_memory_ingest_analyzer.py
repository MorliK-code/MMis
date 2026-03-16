from __future__ import annotations

from memory.fact_extractor import FactExtractor
from memory.entity_resolver import resolve_entities
from memory.ingest_analyzer import analyze_message_for_memory
from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryScope
from memory.numeric_extractor import extract_numeric_facts


def test_analyze_message_for_memory_builds_structured_output() -> None:
    analysis = analyze_message_for_memory(
        "У меня RTX 3050 Ti, 4 GB VRAM, Python 3.11, проект MMis на Windows через Ollama."
    )

    entity_types = {item.type for item in analysis.entities}
    numeric_kinds = {item.kind for item in analysis.numeric_facts}
    stable_predicates = {item.predicate for item in analysis.stable_facts}
    tags = set(analysis.tags)

    assert analysis.search_text
    assert "gpu_model" in entity_types
    assert "python_version" in entity_types
    assert "project_name" in entity_types
    assert "os_name" in entity_types
    assert "tool_name" in entity_types
    assert "vram_gb" in numeric_kinds
    assert "python_version" in numeric_kinds
    assert "gpu_model" in stable_predicates
    assert "python_version" in stable_predicates
    assert "topic_hardware" in tags
    assert "entity_gpu_model_rtx_3050_ti" in tags
    assert "numeric_vram_gb" in tags
    assert analysis.emotion is not None


def test_fact_extractor_uses_ingest_analysis_structured_facts() -> None:
    text = "У меня RTX 3050 Ti с 4 GB VRAM и Python 3.11 на Windows."
    analysis = analyze_message_for_memory(text)
    rows = FactExtractor().extract_v2(
        text=text,
        metadata={"event_id": "evt:test", "namespace": "default"},
        speaker="user",
        scope=MemoryScope.CONVERSATION,
        mode="BALANCED",
        analysis=analysis,
    )

    pairs = {(row.predicate, str(row.value)) for row in rows}
    structured = [row for row in rows if bool(dict(row.metadata or {}).get("structured"))]

    assert ("environment_gpu_model", "RTX 3050 Ti") in pairs
    assert ("environment_gpu_vram_gb", "4") in pairs
    assert ("environment_runtime_python", "python 3.11") in pairs
    assert structured


def test_fact_extractor_uses_entities_and_numeric_before_self_cues() -> None:
    text = "Python 3.11, Windows, 32 GB RAM"
    analysis = analyze_message_for_memory(text)
    rows = FactExtractor().extract_v2(
        text=text,
        metadata={"event_id": "evt:structured", "namespace": "default"},
        speaker="user",
        scope=MemoryScope.CONVERSATION,
        mode="BALANCED",
        analysis=analysis,
    )

    pairs = {(row.predicate, str(row.value)) for row in rows}

    assert ("environment_runtime_python", "python 3.11") in pairs
    assert ("environment_os", "windows") in pairs
    assert ("environment_ram_gb", "32") in pairs


def test_fact_extractor_dedupes_structured_python_fact() -> None:
    text = "У меня Python 3.11"
    analysis = analyze_message_for_memory(text)
    rows = FactExtractor().extract_v2(
        text=text,
        metadata={"event_id": "evt:python", "namespace": "default"},
        speaker="user",
        scope=MemoryScope.CONVERSATION,
        mode="BALANCED",
        analysis=analysis,
    )

    python_rows = [row for row in rows if row.predicate == "environment_runtime_python" and str(row.value) == "python 3.11"]
    assert len(python_rows) == 1


def test_fact_extractor_does_not_treat_i_am_state_as_identity_name() -> None:
    rows = FactExtractor().extract_v2(
        text="I'm tired. I am frustrated. I am on Windows.",
        metadata={"event_id": "evt:no-name", "namespace": "default"},
        speaker="user",
        scope=MemoryScope.CONVERSATION,
        mode="BALANCED",
        analysis=None,
    )

    assert all(row.predicate != "identity_name" for row in rows)


def test_fact_extractor_does_not_store_assistant_environment_facts() -> None:
    text = "I use Windows, Python 3.11, RTX 3050 Ti with 4 GB VRAM."
    analysis = analyze_message_for_memory(text)
    rows = FactExtractor().extract_v2(
        text=text,
        metadata={"event_id": "evt:assistant-env", "namespace": "default"},
        speaker="assistant",
        scope=MemoryScope.CONVERSATION,
        mode="BALANCED",
        analysis=analysis,
    )

    assert all(not str(row.predicate or "").startswith("environment_") for row in rows)


def test_person_name_does_not_match_im_emotion_or_action() -> None:
    entities = resolve_entities("I'm Frustrated and I'm Working on MMis.")
    assert all(item.type != "person_name" for item in entities)


def test_numeric_extractor_skips_non_age_matches() -> None:
    facts = extract_numeric_facts("мне 32 гб RAM и i am 3 commits behind")
    assert all(item.kind != "age_years" for item in facts)


def test_gpu_short_match_does_not_invent_brand() -> None:
    entities = resolve_entities("nvidia 3050 ti в ноутбуке")
    gpu_entities = [item for item in entities if item.type == "gpu_model"]
    assert any(item.canonical == "3050 Ti" for item in gpu_entities)


def test_singleton_fact_rules_cover_structured_environment_predicates() -> None:
    assert MemoryManager._is_singleton_fact_canonical("user.environment_os")
    assert MemoryManager._is_singleton_fact_canonical("user.environment_runtime_python")
    assert MemoryManager._is_singleton_fact_canonical("user.environment_gpu_model")
