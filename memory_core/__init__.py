"""
Memory Core - новый внутренний слой памяти MMis.

Единый фасад для работы с памятью через MemoryService.
"""

from memory_core.schemas import (
    MemoryEnvelope,
    MemoryArtifact,
    MemoryQuery,
    MemoryQueryResult,
    DocumentIngestRequest,
    DocumentIngestResult,
    MemoryInspectRequest,
    MemoryTrace,
)
from memory_core.facade import MemoryService
from memory_core.errors import MemoryError, EventStoreError, ArtifactStoreError, RetrievalError
from memory_core.constants import DEFAULT_NAMESPACE, DEFAULT_WORKSPACE, DEFAULT_TOP_K
from memory_core.adapter import MemoryCoreAdapter, init_memory_core, get_memory_core_adapter
from memory_core.integration import (
    MemoryCoreContextPack,
    MemoryCoreRetrieveAdapter,
    MemoryCoreNativeStateAdapter,
    MemoryCoreProfileAdapter,
    MemoryCoreRetrievalHintsAdapter,
    MemoryCoreRecallPolicyAdapter,
    MemoryCoreFormatAdapter,
    MemoryCoreNormalizeAdapter,
    create_memory_core_adapters,
)
from memory_core.utils import (
    is_low_quality_session_summary,
    sanitize_session_summary_text,
    is_meaningful_summary_turn,
    clean_assistant_text_for_memory,
    sanitize_assistant_memory_text,
    contains_memory_service_sections,
    AssistantTextSanitizeResult,
)
from memory_core.processors.state_reducer import (
    reduce_state_for_turn,
    merge_state_updates,
    StateUpdate,
)
from memory_core.retrieval.history_tools import (
    history_tools_list,
    get_history_tools,
    HistoryReadResult,
    history_read_recent_tool_spec,
    history_read_range_tool_spec,
    history_search_tool_spec,
)

__all__ = [
    # Schemas
    "MemoryEnvelope",
    "MemoryArtifact",
    "MemoryQuery",
    "MemoryQueryResult",
    "DocumentIngestRequest",
    "DocumentIngestResult",
    "MemoryInspectRequest",
    "MemoryTrace",
    # Facade
    "MemoryService",
    # Adapter
    "MemoryCoreAdapter",
    "init_memory_core",
    "get_memory_core_adapter",
    # Integration
    "MemoryCoreContextPack",
    "MemoryCoreRetrieveAdapter",
    "MemoryCoreNativeStateAdapter",
    "MemoryCoreProfileAdapter",
    "MemoryCoreRetrievalHintsAdapter",
    "MemoryCoreRecallPolicyAdapter",
    "MemoryCoreFormatAdapter",
    "MemoryCoreNormalizeAdapter",
    "create_memory_core_adapters",
    # Utils
    "is_low_quality_session_summary",
    "sanitize_session_summary_text",
    "is_meaningful_summary_turn",
    "clean_assistant_text_for_memory",
    "sanitize_assistant_memory_text",
    "contains_memory_service_sections",
    "AssistantTextSanitizeResult",
    # State reducer
    "reduce_state_for_turn",
    "merge_state_updates",
    "StateUpdate",
    # History tools
    "history_tools_list",
    "get_history_tools",
    "HistoryReadResult",
    "history_read_recent_tool_spec",
    "history_read_range_tool_spec",
    "history_search_tool_spec",
    # Errors
    "MemoryError",
    "EventStoreError",
    "ArtifactStoreError",
    "RetrievalError",
    # Constants
    "DEFAULT_NAMESPACE",
    "DEFAULT_WORKSPACE",
    "DEFAULT_TOP_K",
]
