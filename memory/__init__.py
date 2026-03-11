from __future__ import annotations

from memory.auto_migration import run_auto_migration
from memory.document_memory import DocumentMemory
from memory.embedding_provider import build_embedding_provider
from memory.fact_extractor import FactExtractor
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
    DocumentRecord,
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
    "ChunkRecord",
    "VectorStore",
    "EMBED_VERSION",
    "embed_text",
    "FactExtractor",
    "DocumentMemory",
    "LongMemoryV2",
    "MemoryPolicy",
    "build_embedding_provider",
    "run_auto_migration",
]
