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

# Memory Agent v2 - новые компоненты
from memory_core.storage.job_queue_store import JobQueueStore, IngestJob
from memory_core.worker.background_worker import (
    BackgroundWorker,
    BackgroundWorkerPool,
    WorkerConfig,
    WorkerStats,
)
from memory_core.processors.memory_llm_processor import (
    MemoryLLMProcessor,
    ArtifactProposal,
    MemoryLLMResult,
    build_memory_llm_processor,
)
from memory_core.governor.governor import (
    Governor,
    GovernorDecision,
    GovernorResult,
    build_governor,
)
from memory_core.identity.identity_core import (
    IdentityCore,
    IdentityProfile,
    build_identity_core,
)
from memory_core.retrieval.persona_context_builder import (
    PersonaSnapshot,
    PersonaContextBuilder,
    build_persona_context_builder,
)
from memory_core.inspect.memory_inspector import (
    MemoryInspector as NewMemoryInspector,
    InspectorStats,
    build_memory_inspector,
)
from memory_core.planner.episode_planner import (
    EpisodePlanner,
    Episode,
    EpisodeContext,
    build_episode_planner,
)
from memory_core.topic import (
    RelatedTopicLink,
    TopicThread,
    TopicRouteDecision,
    TopicRelationshipLinker,
    TopicRouter,
    TopicStore,
    TopicSummaryBuilder,
    TopicSummarySnapshot,
    TopicToolService,
    topic_read_tool_spec,
    topic_related_tool_spec,
    topic_search_tool_spec,
    topic_tools_list,
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
    # Memory Agent v2
    "JobQueueStore",
    "IngestJob",
    "BackgroundWorker",
    "BackgroundWorkerPool",
    "WorkerConfig",
    "WorkerStats",
    "MemoryLLMProcessor",
    "ArtifactProposal",
    "MemoryLLMResult",
    "build_memory_llm_processor",
    "Governor",
    "GovernorDecision",
    "GovernorResult",
    "build_governor",
    "IdentityCore",
    "IdentityProfile",
    "build_identity_core",
    "PersonaSnapshot",
    "PersonaContextBuilder",
    "build_persona_context_builder",
    "NewMemoryInspector",
    "InspectorStats",
    "build_memory_inspector",
    "EpisodePlanner",
    "Episode",
    "EpisodeContext",
    "build_episode_planner",
    "RelatedTopicLink",
    "TopicThread",
    "TopicRouteDecision",
    "TopicRelationshipLinker",
    "TopicRouter",
    "TopicStore",
    "TopicSummaryBuilder",
    "TopicSummarySnapshot",
    "TopicToolService",
    "topic_read_tool_spec",
    "topic_related_tool_spec",
    "topic_search_tool_spec",
    "topic_tools_list",
]
