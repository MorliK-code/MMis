from __future__ import annotations

from memory.auto_migration import run_auto_migration
from memory.document_memory import DocumentMemory
from memory.embedding_provider import build_embedding_provider
from memory.fact_extractor import Fact, FactExtractor
from memory.memory_manager import MemoryManager
from memory.memory_models import (
    ChunkRecord,
    ContextBuildRequest,
    ContextBuildResult,
    DocumentIngestRequest,
    DocumentIngestResult,
    DocumentRecord,
    IngestResult,
    MemoryEvent,
    MemoryLevel,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    RetrievalQuery,
    RetrievalResult,
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
    "Fact",
    "FactExtractor",
    "DocumentMemory",
    "build_embedding_provider",
    "run_auto_migration",
]
