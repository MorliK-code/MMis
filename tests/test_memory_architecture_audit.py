from __future__ import annotations

import memory.claim_extractor as legacy_claim_extractor
from memory.document_ingest import DOCUMENT_INGEST_ROLE
from memory.document_memory import DOCUMENT_MEMORY_ROLE
from memory.long_memory import LONG_MEMORY_ROLE
from memory import (
    ClaimCandidate,
    ClaimPromotionDecision,
    ClaimRecord,
    ClaimRetriever,
    DialogEpisode,
    DialogEpisodeBuilder,
    DialogEpisodeRetriever,
    DocumentChunk,
    DocumentChunker,
    DocumentClaim,
    DocumentIngestPipeline,
    DocumentRecord,
    DocumentRetriever,
    DocumentSummary,
    GovernorDecision,
    GovernorProfileSnapshot,
    IdentityCoreCandidate,
    MemoryGovernor,
    MemoryLevel,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    IdentityCoreWriteDecision,
    build_memory_debug_snapshot,
    analyze_message_for_memory,
)
from memory.claim_extractor import promote_claim_candidates as legacy_promote_claim_candidates
from memory.claim_promoter import promote_claim_candidates


def test_memory_layers_are_importable() -> None:
    assert ClaimCandidate is not None
    assert ClaimRecord is not None
    assert ClaimPromotionDecision is not None
    assert ClaimRetriever is not None
    assert MemoryGovernor is not None
    assert GovernorDecision is not None
    assert GovernorProfileSnapshot is not None
    assert IdentityCoreCandidate is not None
    assert IdentityCoreWriteDecision is not None

    assert DialogEpisode is not None
    assert DialogEpisodeBuilder is not None
    assert DialogEpisodeRetriever is not None

    assert DocumentRecord is not None
    assert DocumentChunk is not None
    assert DocumentSummary is not None
    assert DocumentClaim is not None
    assert DocumentChunker is not None
    assert DocumentIngestPipeline is not None
    assert DocumentRetriever is not None
    assert build_memory_debug_snapshot is not None


def test_claim_layer_is_connected_to_ingest() -> None:
    analysis = analyze_message_for_memory("я использую VS Code")

    assert len(list(analysis.claim_candidates or [])) >= 1
    assert any(
        str(item.predicate or "").strip().lower() == "uses"
        for item in list(analysis.claim_candidates or [])
    )


def test_claim_extractor_legacy_shim_points_to_claim_promoter() -> None:
    assert legacy_claim_extractor.LEGACY_COMPAT_ONLY is True
    assert legacy_claim_extractor.LEGACY_SHIM_NOTE == "Do not use this module in new code."
    assert legacy_promote_claim_candidates(
        [],
        event_id="evt:test",
        namespace="default",
    ) == promote_claim_candidates(
        [],
        event_id="evt:test",
        namespace="default",
    )


def test_document_layers_have_explicit_roles() -> None:
    assert DOCUMENT_MEMORY_ROLE == "low_level_document_storage"
    assert DOCUMENT_INGEST_ROLE == "high_level_document_pipeline"
    assert LONG_MEMORY_ROLE == "thin_document_facade"


def test_identity_core_memory_type_is_available_for_roundtrip() -> None:
    row = MemoryRecord.from_dict(
        {
            "id": "identity-core:asya",
            "text": "identity core snapshot",
            "memory_type": MemoryType.IDENTITY_CORE.value,
            "level": MemoryLevel.L3_SEMANTIC.value,
            "scope": MemoryScope.CHARACTER.value,
            "namespace": "asya",
        }
    )

    assert row.memory_type == MemoryType.IDENTITY_CORE
    assert row.to_dict()["memory_type"] == MemoryType.IDENTITY_CORE.value
