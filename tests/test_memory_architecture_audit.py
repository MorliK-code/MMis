from __future__ import annotations

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
    analyze_message_for_memory,
)


def test_memory_layers_are_importable() -> None:
    assert ClaimCandidate is not None
    assert ClaimRecord is not None
    assert ClaimPromotionDecision is not None
    assert ClaimRetriever is not None

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


def test_claim_layer_is_connected_to_ingest() -> None:
    analysis = analyze_message_for_memory("я использую VS Code")

    assert len(list(analysis.claim_candidates or [])) >= 1
    assert any(
        str(item.predicate or "").strip().lower() == "uses"
        for item in list(analysis.claim_candidates or [])
    )
