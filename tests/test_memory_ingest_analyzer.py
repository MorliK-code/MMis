from __future__ import annotations

from memory.fact_extractor import FactExtractor
from memory.contextual_resolver import resolve_anchor_context
from memory.entity_resolver import resolve_entities
from memory.ingest_analyzer import analyze_message_for_memory
from memory.memory_manager import MemoryManager
from memory.memory_models import MemoryScope
from memory.numeric_extractor import extract_numeric_facts


def test_analyze_message_for_memory_builds_structured_output() -> None:
    analysis = analyze_message_for_memory(
        "I use RTX 3050 Ti, 4 GB VRAM, Python 3.11, project MMis on Windows via Ollama."
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


def test_analyze_message_for_memory_builds_domain_anchors_before_facts() -> None:
    analysis = analyze_message_for_memory(
        "i am 21 years old and i have 32 gb ram, also 3050ti with 4gb memory"
    )

    anchor_kinds = {item.kind for item in analysis.anchors}

    assert "person_age_candidate" in anchor_kinds
    assert "ram_candidate" in anchor_kinds
    assert "gpu_model_candidate" in anchor_kinds
    assert "memory_size_candidate" in anchor_kinds
    assert any(item.kind == "gpu_model_candidate" and "3050" in str(item.normalized) for item in analysis.anchors)


def test_contextual_resolver_interprets_memory_size_as_vram_near_gpu() -> None:
    analysis = analyze_message_for_memory("3050ti with 4gb memory")

    resolution = dict(analysis.contextual_resolution or {})
    memory_kind_by_surface = dict(resolution.get("memory_kind_by_surface") or {})

    assert memory_kind_by_surface.get("4gb memory") == "vram"
    assert any(item.kind == "vram_gb" and str(item.value) == "4" for item in analysis.numeric_facts)


def test_contextual_resolver_marks_python_number_as_runtime_version() -> None:
    anchors = analyze_message_for_memory("python 3.11").anchors
    resolution = resolve_anchor_context("python 3.11", anchors=anchors)

    assert dict(resolution.version_kind_by_surface).get("python 3.11") == "runtime_python"


def test_contextual_resolver_marks_colloquial_python_and_os_mentions_present() -> None:
    analysis = analyze_message_for_memory("напомни мой пайтон и что у меня за система, версия винды")

    anchor_kinds = {item.kind for item in analysis.anchors}
    signals = dict(dict(analysis.contextual_resolution or {}).get("signals") or {})

    assert "python_present_candidate" in anchor_kinds
    assert "os_mention_candidate" in anchor_kinds
    assert signals.get("python_present") is True
    assert signals.get("os_present") is True


def test_ingest_analysis_ignores_runtime_metadata_as_truth_source() -> None:
    analysis = analyze_message_for_memory(
        "привет",
        metadata={
            "runtime_entities": {"software": ["Python"], "os": ["Windows"]},
            "entities": {"software": ["Python"], "os": ["Windows"]},
            "tags": ["topic_python", "intent_chat"],
            "topic": "python",
            "intent": "code_help",
            "project_name": "InjectedProject",
        },
    )

    assert list(analysis.entities or []) == []
    assert list(analysis.numeric_facts or []) == []
    assert list(analysis.stable_facts or []) == []
    assert list(analysis.claim_candidates or []) == []
    assert list(analysis.memory_views.get("entity_keys") or []) == []
    assert list(analysis.memory_views.get("numeric_keys") or []) == []


def test_contextual_resolver_tracks_current_and_past_os_values() -> None:
    analysis = analyze_message_for_memory("сейчас windows 11, до этого linux")

    resolution = dict(analysis.contextual_resolution or {})

    assert "Windows" in list(resolution.get("current_os_values") or [])
    assert "Linux" in list(resolution.get("past_os_values") or [])
    signals = dict(resolution.get("signals") or {})
    assert signals.get("current_os_present") is True
    assert signals.get("past_os_present") is True


def test_ingest_analysis_understands_udav_as_python_version() -> None:
    analysis = analyze_message_for_memory("сижу на удаве 3.11")

    assert any(item.type == "python_version" and str(item.canonical) == "3.11" for item in analysis.entities)
    assert any(item.kind == "python_version" and str(item.value) == "3.11" for item in analysis.numeric_facts)


def test_ingest_analysis_understands_udav_version_phrase_with_plain_number() -> None:
    analysis = analyze_message_for_memory("версия удава у меня 11")

    assert any(item.type == "python_version" and str(item.canonical) == "11" for item in analysis.entities)
    assert any(item.kind == "python_version" and str(item.value) == "11" for item in analysis.numeric_facts)


def test_fact_synthesis_builds_typed_stable_facts_after_anchor_and_context_passes() -> None:
    analysis = analyze_message_for_memory(
        "i am 21 years old, i use RTX 3050 Ti with 4 GB VRAM, 32 GB RAM, and Windows"
    )

    stable_pairs = {(item.subject, item.predicate, str(item.value)) for item in analysis.stable_facts}

    assert ("hardware", "gpu_model", "RTX 3050 Ti") in stable_pairs
    assert ("hardware", "gpu_vram_gb", "4") in stable_pairs
    assert ("hardware", "ram_gb", "32") in stable_pairs
    assert ("identity", "age_years", "21") in stable_pairs
    assert ("environment", "os_name", "Windows") in stable_pairs


def test_fact_synthesis_prefers_current_os_over_past_os() -> None:
    analysis = analyze_message_for_memory("у меня сейчас windows 11, до этого linux")

    stable_pairs = {(item.subject, item.predicate, str(item.value)) for item in analysis.stable_facts}

    assert ("environment", "os_name", "Windows") in stable_pairs
    assert ("environment", "os_name", "Linux") not in stable_pairs


def test_ingest_analysis_populates_memory_view_keys_from_entities_and_numeric_facts() -> None:
    analysis = analyze_message_for_memory("У меня RTX 3050 Ti с 4 GB VRAM и Python 3.11")

    views = dict(analysis.memory_views or {})
    entity_keys = set(str(x or "") for x in list(views.get("entity_keys") or []))
    numeric_keys = set(str(x or "") for x in list(views.get("numeric_keys") or []))

    assert "gpu" in entity_keys
    assert "rtx_3050_ti" in entity_keys
    assert "python" in entity_keys
    assert "python_3_11" in entity_keys
    assert "value:4gb" in numeric_keys
    assert "vram:4gb" in numeric_keys
    assert "version:3.11" in numeric_keys


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


def test_fact_extractor_keeps_only_current_os_from_structured_entities() -> None:
    text = "у меня сейчас windows 11, до этого linux"
    analysis = analyze_message_for_memory(text)
    rows = FactExtractor().extract_v2(
        text=text,
        metadata={"event_id": "evt:os-current", "namespace": "default"},
        speaker="user",
        scope=MemoryScope.CONVERSATION,
        mode="BALANCED",
        analysis=analysis,
    )

    os_rows = [row for row in rows if row.predicate == "environment_os"]

    assert [str(row.value) for row in os_rows] == ["windows"]


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



def test_anchor_context_helps_treat_4gb_memory_as_vram_near_gpu() -> None:
    analysis = analyze_message_for_memory("3050ti with 4gb memory")

    assert any(item.type == "gpu_model" and "3050" in str(item.canonical) for item in analysis.entities)
    assert any(item.kind == "vram_gb" and str(item.value) == "4" for item in analysis.numeric_facts)
    assert all(not (item.kind == "ram_gb" and str(item.value) == "4") for item in analysis.numeric_facts)


def test_ingest_analysis_understands_colloquial_hardware_and_age_forms() -> None:
    analysis = analyze_message_for_memory("оперативы 32, сижу на питоне 3.11, винда, мне уже 22")

    assert any(item.kind == "ram_gb" and str(item.value) == "32" for item in analysis.numeric_facts)
    assert any(item.type == "python_version" and str(item.canonical) == "3.11" for item in analysis.entities)
    assert any(item.type == "os_name" and str(item.canonical) == "Windows" for item in analysis.entities)
    assert any(item.kind == "age_years" and str(item.value) == "22" for item in analysis.numeric_facts)


def test_ingest_analysis_understands_gpu_short_form_with_memory_phrase() -> None:
    analysis = analyze_message_for_memory("3050ti с 4gb памяти")

    assert any(item.type == "gpu_model" and "3050" in str(item.canonical) for item in analysis.entities)
    assert any(item.kind == "vram_gb" and str(item.value) == "4" for item in analysis.numeric_facts)
    assert all(not (item.kind == "ram_gb" and str(item.value) == "4") for item in analysis.numeric_facts)

def test_fact_extractor_does_not_create_temporary_fact_from_long_narrative_with_plain_poka() -> None:
    text = (
        "Это длинная история. Я долго сомневался, пока друзья уговаривали меня сделать первый шаг. "
        "Потом стало пусто в голове и я наконец выспался."
    )
    analysis = analyze_message_for_memory(text)
    rows = FactExtractor().extract_v2(
        text=text,
        metadata={"event_id": "evt:narrative-poka", "namespace": "default"},
        speaker="user",
        scope=MemoryScope.CONVERSATION,
        mode="BALANCED",
        analysis=analysis,
    )

    assert all(row.predicate != "temporary_fact" for row in rows)


def test_singleton_fact_rules_cover_structured_environment_predicates() -> None:
    assert MemoryManager._is_singleton_fact_canonical("user.environment_os")
    assert MemoryManager._is_singleton_fact_canonical("user.environment_runtime_python")
    assert MemoryManager._is_singleton_fact_canonical("user.environment_gpu_model")
