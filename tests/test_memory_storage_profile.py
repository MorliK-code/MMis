from __future__ import annotations

from memory.memory_policy import MemoryPolicy
from memory.memory_models import MemorySourceKind
from memory.storage_profile import compact_metadata_payload


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
