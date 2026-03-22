"""
Processors layer для memory_core.
"""

from memory_core.processors.base import MemoryProcessor
from memory_core.processors.ingest_analyzer import IngestAnalyzer
from memory_core.processors.fact_processor import FactProcessor
from memory_core.processors.profile_processor import ProfileProcessor
from memory_core.processors.episode_processor import EpisodeProcessor
from memory_core.processors.task_processor import TaskProcessor
from memory_core.processors.document_processor import DocumentProcessor
from memory_core.processors.dedupe_processor import DedupeProcessor
from memory_core.processors.summary_processor import SummaryProcessor

__all__ = [
    "MemoryProcessor",
    "IngestAnalyzer",
    "FactProcessor",
    "ProfileProcessor",
    "EpisodeProcessor",
    "TaskProcessor",
    "DocumentProcessor",
    "DedupeProcessor",
    "SummaryProcessor",
]
