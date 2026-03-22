"""
Memory Core Integration - интеграция memory_core в response_pipeline.

Этот файл предоставляет адаптеры для замены старых интерфейсов памяти
на новые без переписывания всего response_pipeline.
"""

from typing import Any
from memory_core.schemas import MemoryQuery, MemoryEnvelope
from memory_core.facade import MemoryService
from memory_core.retrieval.query_models import ContextPack
from memory_core.retrieval.context_builder import ContextBuilder


class MemoryCoreContextPack:
    """
    Адаптер ContextPack для совместимости со старым кодом.
    """

    def __init__(self, context_pack: ContextPack):
        self.context_pack = context_pack

    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь для совместимости."""
        return {
            "profile_facts": self.context_pack.profile_facts,
            "active_tasks": self.context_pack.active_tasks,
            "recent_episodes": self.context_pack.recent_episodes,
            "relevant_facts": self.context_pack.relevant_facts,
            "document_chunks": self.context_pack.document_chunks,
            "workspace_info": self.context_pack.workspace_info,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение по ключу."""
        return getattr(self.context_pack, key, default)

    def __getitem__(self, key: str) -> Any:
        """Получает значение по ключу (для совместимости с dict)."""
        return getattr(self.context_pack, key, [])

    def __contains__(self, key: str) -> bool:
        """Проверяет наличие ключа."""
        return hasattr(self.context_pack, key)


class MemoryCoreRetrieveAdapter:
    """
    Адаптер для замены memory.tool_bridge.build_memory_tool_context_pack.
    """

    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    def build_context_pack(
        self,
        query_text: str,
        workspace_id: str = "global",
        namespace: str = "default",
        top_k: int = 8,
        include_citations: bool = True,
    ) -> MemoryCoreContextPack:
        """
        Строит контекст для LLM.

        Args:
            query_text: Текст запроса.
            workspace_id: ID workspace.
            namespace: Namespace.
            top_k: Количество результатов.
            include_citations: Включать ли цитаты.

        Returns:
            ContextPack в совместимом формате.
        """
        query = MemoryQuery(
            text=query_text,
            workspace_id=workspace_id,
            namespace=namespace,
            top_k=top_k,
            include_citations=include_citations,
        )

        result = self.memory_service.query(query)

        return MemoryCoreContextPack(
            ContextPack(
                profile_facts=[h.get("text", "") for h in result.hits if h.get("artifact_type") == "profile_fact"],
                active_tasks=[h.get("text", "") for h in result.hits if h.get("artifact_type") == "task"],
                recent_episodes=[h.get("text", "") for h in result.hits if h.get("artifact_type") == "episode"],
                relevant_facts=[h.get("text", "") for h in result.hits if h.get("artifact_type") == "fact"],
                document_chunks=[h.get("text", "") for h in result.hits if h.get("artifact_type") == "document_chunk"],
            )
        )


class MemoryCoreNativeStateAdapter:
    """
    Адаптер для замены memory.native_state.build_memory_native_state.
    """

    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    def build_native_state(
        self,
        workspace_id: str = "global",
    ) -> dict[str, Any]:
        """
        Строит нативное состояние памяти.

        Args:
            workspace_id: ID workspace.

        Returns:
            Словарь с состоянием.
        """
        stats = self.memory_service.get_stats()

        return {
            "events_count": stats.get("events_count", 0),
            "artifacts_count": stats.get("artifacts_count", 0),
            "workspaces_count": stats.get("workspaces_count", 0),
            "current_workspace": workspace_id,
        }


class MemoryCoreProfileAdapter:
    """
    Адаптер для замены memory.profile_evolution.flatten_governor_profile_snapshot.
    """

    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    def flatten_profile_snapshot(
        self,
        workspace_id: str = "global",
    ) -> dict[str, Any]:
        """
        Получает профиль пользователя.

        Args:
            workspace_id: ID workspace.

        Returns:
            Словарь с профилем.
        """
        from memory_core.schemas import MemoryInspectRequest

        # Получаем профильные факты
        request = MemoryInspectRequest(kind="profile", workspace_id=workspace_id, limit=50)
        result = self.memory_service.inspect(request)

        profile_facts = result.get("profile_facts", [])

        # Преобразуем в плоский формат
        flattened = {}
        for fact in profile_facts:
            metadata = fact.get("metadata", {})
            category = metadata.get("category", "general")
            value = metadata.get("value", fact.get("summary", fact.get("text", "")))

            if category not in flattened:
                flattened[category] = []
            flattened[category].append(value)

        return {
            "profile_facts": profile_facts,
            "flattened": flattened,
            "workspace_id": workspace_id,
        }


class MemoryCoreRetrievalHintsAdapter:
    """
    Адаптер для замены memory.project_terms.build_retrieval_hints.
    """

    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    def build_hints(
        self,
        query_text: str,
        workspace_id: str = "global",
    ) -> dict[str, Any]:
        """
        Строит подсказки для retrieval.

        Args:
            query_text: Текст запроса.
            workspace_id: ID workspace.

        Returns:
            Словарь с подсказками.
        """
        # Определяем возможные типы артефактов для запроса
        artifact_types = []

        query_lower = query_text.lower()

        if any(word in query_lower for word in ["кто", "что", "где", "когда", "почему", "как"]):
            artifact_types.append("fact")

        if any(word in query_lower for word in ["задача", "надо", "нужно", "план"]):
            artifact_types.append("task")

        if any(word in query_lower for word in ["прошлый раз", "недавно", "обсуждали"]):
            artifact_types.append("episode")

        if any(word in query_lower for word in ["документ", "файл", "код"]):
            artifact_types.append("document_chunk")

        return {
            "suggested_artifact_types": artifact_types or ["fact", "episode"],
            "workspace_id": workspace_id,
            "query_text": query_text,
        }


class MemoryCoreRecallPolicyAdapter:
    """
    Адаптер для замены memory.recall_policy.classify_query_recall_profile.
    """

    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service

    def classify_query(
        self,
        query_text: str,
        workspace_id: str = "global",
    ) -> dict[str, Any]:
        """
        Классифицирует запрос для retrieval.

        Args:
            query_text: Текст запроса.
            workspace_id: ID workspace.

        Returns:
            Словарь с классификацией.
        """
        # Простая эвристика для определения необходимости retrieval
        should_retrieve = len(query_text.strip()) > 5

        return {
            "should_retrieve": should_retrieve,
            "confidence": 0.8 if should_retrieve else 0.2,
            "reason": "query_length" if should_retrieve else "too_short",
            "workspace_id": workspace_id,
        }


class MemoryCoreFormatAdapter:
    """
    Адаптер для замены memory.tool_bridge.format_memory_result_for_llm.
    """

    @staticmethod
    def format_result_for_llm(
        context_pack: MemoryCoreContextPack,
        include_citations: bool = True,
    ) -> str:
        """
        Форматирует результат памяти для LLM.

        Args:
            context_pack: Контекст из памяти.
            include_citations: Включать ли цитаты.

        Returns:
            Отформатированный текст.
        """
        blocks = context_pack.context_pack.to_context_blocks()

        if not blocks:
            return ""

        return "\n\n".join(blocks)


class MemoryCoreNormalizeAdapter:
    """
    Адаптер для замены memory.tool_bridge.normalize_memory_retrieval_plan.
    """

    @staticmethod
    def normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
        """
        Нормализует план retrieval.

        Args:
            plan: Исходный план.

        Returns:
            Нормализованный план.
        """
        # Упрощённая нормализация
        return {
            "artifact_types": plan.get("artifact_types", ["fact", "episode"]),
            "top_k": plan.get("top_k", 8),
            "workspace_id": plan.get("workspace_id", "global"),
        }


def create_memory_core_adapters(memory_service: MemoryService) -> dict[str, Any]:
    """
    Создаёт все адаптеры для интеграции.

    Args:
        memory_service: Экземпляр MemoryService.

    Returns:
        Словарь с адаптерами.
    """
    return {
        "retrieve": MemoryCoreRetrieveAdapter(memory_service),
        "native_state": MemoryCoreNativeStateAdapter(memory_service),
        "profile": MemoryCoreProfileAdapter(memory_service),
        "hints": MemoryCoreRetrievalHintsAdapter(memory_service),
        "recall_policy": MemoryCoreRecallPolicyAdapter(memory_service),
        "format": MemoryCoreFormatAdapter(),
        "normalize": MemoryCoreNormalizeAdapter(),
    }
