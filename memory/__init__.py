from __future__ import annotations

from memory.auto_migration import run_auto_migration
from memory.claim_models import ClaimCandidate, ClaimPromotionDecision, ClaimRecord
from memory.debug_snapshot import build_memory_debug_snapshot
from memory.claim_promoter import decide_claim_promotion, promote_claim_candidates
from memory.document_chunker import ChunkingConfig, DocumentChunker
from memory.claim_retrieval import ClaimRetriever
from memory.dialog_episode_builder import DialogEpisodeBoundary, DialogEpisodeBuilder, DialogTurn
from memory.dialog_episode_models import DialogEpisode
from memory.episode_planner import ActiveTaskState, EpisodePlanner
from memory.dialog_episode_retriever import DialogEpisodeHit, DialogEpisodeRetriever, DialogEpisodeQueryHints, DialogSupportingTurn
from memory.document_ingest import DocumentChunkAnalysis, DocumentIngestArtifacts, DocumentIngestPipeline, DocumentOutline
from memory.document_models import DocumentChunk, DocumentClaim, DocumentRecord, DocumentSummary
from memory.document_retrieval import DocumentRetrievalHit, DocumentRetriever, DocumentRetrievalHints
from memory.document_memory import DocumentMemory
from memory.embedding_provider import build_embedding_provider
from memory.fact_extractor import FactExtractor
from memory.governor import GovernorDecision, GovernorProfileSnapshot, MemoryGovernor
from memory.identity_core import (
    IdentityCoreCandidate,
    IdentityCoreManager,
    PROTECTED_IDENTITY_CORE_KEYS,
    IdentityCoreRecord,
    IdentityCoreSnapshot,
    IdentityCoreWriteDecision,
)
from memory.ingest_analyzer import (
    EntityItem,
    IngestAnalysis,
    IngestEmotion,
    NumericFact,
    StableFact,
    analyze_message_for_memory,
)
from memory.long_memory import LongMemoryV2
from memory.memory_manager import MemoryManager
from memory.memory_policy import MemoryPolicy
from memory.memory_models import (
    ChunkRecord,
    ContextCompressor,
    ContextBuildRequest,
    ContextBuildResult,
    DocumentIngestRequest,
    DocumentIngestResult,
    EmbeddingProvider,
    IngestResult,
    LexicalIndexBackend,
    MemoryDebugger,
    MemoryEvent,
    MemoryLifecycle,
    MemoryLevel,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    Reranker,
    RetrievalQuery,
    RetrievalResult,
    VectorIndexBackend,
)
from memory.vector_store import EMBED_VERSION, VectorStore, embed_text

__all__ = [
    "MemoryManager",
    "MemoryRecord",
    "MemoryEvent",
    "MemoryLevel",
    "MemoryScope",
    "MemoryStatus",
    "MemoryType",
    "EmbeddingProvider",
    "VectorIndexBackend",
    "LexicalIndexBackend",
    "Reranker",
    "ContextCompressor",
    "MemoryDebugger",
    "MemoryLifecycle",
    "ContextBuildRequest",
    "ContextBuildResult",
    "IngestResult",
    "RetrievalQuery",
    "RetrievalResult",
    "DocumentIngestRequest",
    "DocumentIngestResult",
    "DocumentRecord",
    "DocumentChunk",
    "DocumentSummary",
    "DocumentClaim",
    "DocumentRetrievalHints",
    "DocumentRetrievalHit",
    "DocumentRetriever",
    "ChunkRecord",
    "VectorStore",
    "EMBED_VERSION",
    "embed_text",
    "FactExtractor",
    "MemoryGovernor",
    "GovernorDecision",
    "GovernorProfileSnapshot",
    "IdentityCoreCandidate",
    "IdentityCoreRecord",
    "IdentityCoreSnapshot",
    "IdentityCoreManager",
    "IdentityCoreWriteDecision",
    "PROTECTED_IDENTITY_CORE_KEYS",
    "ClaimCandidate",
    "ClaimRecord",
    "ClaimPromotionDecision",
    "build_memory_debug_snapshot",
    "decide_claim_promotion",
    "promote_claim_candidates",
    "ClaimRetriever",
    "ChunkingConfig",
    "DialogTurn",
    "DialogEpisodeBoundary",
    "DialogEpisodeBuilder",
    "DialogEpisode",
    "ActiveTaskState",
    "EpisodePlanner",
    "DialogEpisodeQueryHints",
    "DialogSupportingTurn",
    "DialogEpisodeHit",
    "DialogEpisodeRetriever",
    "DocumentChunkAnalysis",
    "DocumentOutline",
    "DocumentIngestArtifacts",
    "DocumentIngestPipeline",
    "DocumentChunker",
    "DocumentMemory",
    "LongMemoryV2",
    "MemoryPolicy",
    "build_embedding_provider",
    "EntityItem",
    "NumericFact",
    "StableFact",
    "IngestEmotion",
    "IngestAnalysis",
    "analyze_message_for_memory",
    "run_auto_migration",
]
