"""
Context Builder - сборка итогового контекста для LLM.
"""

from memory_core.schemas import MemoryArtifact, MemoryQuery
from memory_core.retrieval.query_models import ContextPack, Citation
from memory_core.retrieval.prompt_adapter import distill_memory_artifacts, resolve_artifact_prompt_view


class ContextBuilder:
    """
    Сборщик контекста для LLM.
    
    Собирает итоговый контекст из:
    - краткий профиль пользователя
    - активные задачи
    - relevant facts
    - recent episodes
    - relevant document chunks
    - citations/debug info
    """
    
    def __init__(self, max_context_length: int = 4000):
        """
        Инициализирует Context Builder.
        
        Args:
            max_context_length: Максимальная длина контекста.
        """
        self.max_context_length = max_context_length
    
    def build(
        self,
        artifacts: list[MemoryArtifact],
        query: MemoryQuery,
    ) -> tuple[ContextPack, list[Citation]]:
        """
        Строит контекст из артефактов.

        Args:
            artifacts: Список артефактов.
            query: Исходный запрос.

        Returns:
            Кортеж (ContextPack, список Citation).
        """
        # Группируем артефакты по типам
        profile_facts = []
        active_tasks = []
        runtime_episodes = []  # episode_event — runtime continuity
        semantic_episodes = []  # episode — historical summaries
        facts = []
        emotional_signals = []
        document_chunks = []

        for artifact in artifacts:
            # Profile facts
            if artifact.artifact_type == "profile_fact":
                profile_facts.append(artifact)

            # Task states (новые + старые)
            elif artifact.artifact_type in {"task", "task_state"}:
                # Проверяем, активная ли задача
                if artifact.metadata.get("task_status") == "open" or artifact.status == "active":
                    active_tasks.append(artifact)

            # Runtime episodes (episode_event) — приоритет для continuity
            elif artifact.artifact_type == "episode_event":
                runtime_episodes.append(artifact)

            # Semantic episodes (episode) — historical summaries
            elif artifact.artifact_type == "episode":
                semantic_episodes.append(artifact)

            # Facts and preferences
            elif artifact.artifact_type in {"fact", "preference"}:
                facts.append(artifact)

            # Emotional signals should not go into generic facts path
            elif artifact.artifact_type == "emotional_state":
                emotional_signals.append(artifact)

            # Documents
            elif artifact.artifact_type in {"document_chunk", "document_summary"}:
                document_chunks.append(artifact)

        # Сортируем по важности и давности
        profile_facts = self._sort_by_relevance(profile_facts)[:5]
        active_tasks = self._sort_by_relevance(active_tasks)[:5]
        
        # Runtime episodes имеют приоритет над semantic
        runtime_episodes = self._sort_by_recency(runtime_episodes)[:2]
        semantic_episodes = self._sort_by_recency(semantic_episodes)[:1]
        
        # Объединяем: сначала runtime, потом semantic
        episodes = runtime_episodes + semantic_episodes
        
        facts = self._sort_by_relevance(facts)[:10]
        emotional_signals = self._sort_by_relevance(emotional_signals)[:4]
        document_chunks = self._sort_by_relevance(document_chunks)[:5]
        distilled = distill_memory_artifacts(
            [
                *profile_facts,
                *active_tasks,
                *episodes,
                *facts,
                *emotional_signals,
                *document_chunks,
            ],
            query_text=str(query.text or ""),
        )

        # Создаём ContextPack
        context_pack = ContextPack(
            profile_facts=distilled.legacy_lists.get("profile_facts", []) or [self._safe_memory_text(a) for a in profile_facts],
            active_tasks=distilled.legacy_lists.get("active_tasks", []) or [self._safe_memory_text(a) for a in active_tasks],
            recent_episodes=distilled.legacy_lists.get("recent_episodes", []) or [self._safe_memory_text(a) for a in episodes],
            relevant_facts=distilled.legacy_lists.get("relevant_facts", []) or [self._safe_memory_text(a) for a in facts],
            document_chunks=distilled.legacy_lists.get("document_chunks", []) or [self._safe_memory_text(a) for a in document_chunks],
            tone_hints=self._block_to_items(distilled.blocks.get("tone_hints")),
            continuity_hints=self._block_to_items(distilled.blocks.get("continuity_hints")),
            answer_support=self._block_to_items(distilled.blocks.get("answer_support")),
            exact_recall=self._block_to_items(distilled.blocks.get("exact_recall")),
            blocks=dict(distilled.blocks),
            selected_memories=list(distilled.selected),
            dropped_memories=list(distilled.dropped),
            recent_user_state=dict(distilled.recent_user_state),
            response_bias=dict(distilled.response_bias),
            debug=dict(distilled.debug),
        )

        # Создаём Citation
        citations = [
            Citation(
                artifact_id=a.artifact_id,
                artifact_type=a.artifact_type,
                source_event_id=a.source_event_id,
                text=(self._safe_memory_text(a) or a.text)[:200],
                metadata={
                    **dict(a.metadata or {}),
                    "prompt_view": resolve_artifact_prompt_view(
                        a.artifact_type,
                        text=a.text,
                        summary=a.summary,
                        metadata=a.metadata,
                    ),
                },
            )
            for a in artifacts[:20]  # Ограничиваем количество citation
        ]
        
        return context_pack, citations
    
    def build_from_context_pack(
        self,
        context_pack: ContextPack,
        include_citations: bool = True,
    ) -> list[str]:
        """
        Преобразует ContextPack в текстовые блоки.
        
        Args:
            context_pack: Пакет контекста.
            include_citations: Включать ли цитаты.
            
        Returns:
            Список текстовых блоков.
        """
        return context_pack.to_context_blocks()
    
    def _sort_by_relevance(
        self,
        artifacts: list[MemoryArtifact],
    ) -> list[MemoryArtifact]:
        """Сортирует артефакты по релевантности."""
        # Сортируем по confidence и recency
        import time
        
        now = time.time()
        
        def score(a: MemoryArtifact) -> float:
            confidence = a.metadata.get("confidence", 0.5)
            recency = 0.5 ** ((now - a.created_at) / (24 * 60 * 60))  # 1 день half-life
            return confidence * 0.6 + recency * 0.4
        
        return sorted(artifacts, key=score, reverse=True)
    
    def _sort_by_recency(
        self,
        artifacts: list[MemoryArtifact],
    ) -> list[MemoryArtifact]:
        """Сортирует артефакты по давности (новые первые)."""
        return sorted(artifacts, key=lambda a: a.created_at, reverse=True)
    
    def truncate_context(
        self,
        context_blocks: list[str],
        max_length: int | None = None,
    ) -> list[str]:
        """
        Обрезает контекст до максимальной длины.
        
        Args:
            context_blocks: Список блоков контекста.
            max_length: Максимальная длина.
            
        Returns:
            Обрезанный контекст.
        """
        if max_length is None:
            max_length = self.max_context_length
        
        result = []
        current_length = 0
        
        for block in context_blocks:
            if current_length + len(block) > max_length:
                # Обрезаем последний блок
                remaining = max_length - current_length
                if remaining > 0:
                    result.append(block[:remaining] + "...")
                break
            
            result.append(block)
            current_length += len(block)
        
        return result

    @staticmethod
    def _safe_memory_text(artifact: MemoryArtifact) -> str:
        return str(
            resolve_artifact_prompt_view(
                artifact.artifact_type,
                text=artifact.text,
                summary=artifact.summary,
                metadata=artifact.metadata,
            )
            or artifact.summary
            or ""
        ).strip()

    @staticmethod
    def _block_to_items(value: str) -> list[str]:
        rows: list[str] = []
        for line in str(value or "").splitlines():
            text = str(line or "").strip()
            if text.startswith("- "):
                text = text[2:].strip()
            if text:
                rows.append(text)
        return rows
