"""
Retrieval Service - сервис поиска и извлечения памяти.
"""

import time
from typing import Any
from memory_core.storage.sqlite_db import Database
from memory_core.storage.artifact_store import ArtifactStore
from memory_core.runtime_session_store import RuntimeSessionSnapshot, RuntimeSessionStore
from memory_core.schemas import MemoryQuery, MemoryArtifact
from memory_core.retrieval.query_models import RetrievalFilters, ContextPack, Citation
from memory_core.retrieval.context_builder import ContextBuilder
from memory_core.retrieval.reranker import Reranker
from memory_core.retrieval.filters import apply_filters, filter_by_text_similarity
from memory_core.indexing.vector_index import VectorIndex
from memory_core.config_manager import RetrievalConfig
from memory_core.errors import RetrievalError
from memory_core.topic import TopicStore, TopicToolService


class RetrievalService:
    """
    Сервис для hybrid retrieval.

    Ищет по:
    - mandatory memory (identity_core, profile_fact, task_state)
    - session/episode continuity
    - semantic search (vector index)
    - lexical search (LIKE fallback)
    """

    def __init__(
        self,
        db: Database,
        vector_index: VectorIndex | None = None,
        retrieval_config: RetrievalConfig | None = None,
        runtime_session_store: RuntimeSessionStore | None = None,
        include_pending_facts_in_retrieval: bool = False,
    ):
        """
        Инициализирует Retrieval Service.

        Args:
            db: Экземпляр Database.
            vector_index: Векторный индекс для semantic search.
            retrieval_config: Конфигурация retrieval правил.
        """
        self.db = db
        self.artifact_store = ArtifactStore(db)
        self.context_builder = ContextBuilder()
        self.reranker = Reranker()
        self.vector_index = vector_index
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.runtime_session_store = runtime_session_store
        self.include_pending_facts_in_retrieval = bool(include_pending_facts_in_retrieval)
        self.topic_store = TopicStore(self.artifact_store)
        self.topic_tools = TopicToolService(self.topic_store)
    
    def query(self, query: MemoryQuery) -> tuple[ContextPack, list[Citation]]:
        """
        Выполняет запрос к памяти.

        Args:
            query: Запрос к памяти.

        Returns:
            Кортеж (ContextPack, список Citation).
        """
        # 4 стадии retrieval
        runtime_session = self._get_runtime_session(query)
        topic_details = self._get_topic_details(query, runtime_session)
        runtime_artifacts = self._collect_runtime_artifacts(
            query,
            runtime_session=runtime_session,
            topic_details=topic_details,
        )
        mandatory = self._collect_mandatory_artifacts(query)
        continuity = self._collect_continuity_artifacts(query)
        semantic = self._collect_semantic_artifacts(query)
        lexical = self._collect_lexical_artifacts(query)

        # Объединяем
        merged = self._merge_artifact_lists(runtime_artifacts, mandatory, continuity, semantic, lexical)
        merged = self._apply_topic_boost(merged, query)

        # Применяем runtime rules
        filtered = self._apply_runtime_rules(merged, query)

        # Rerank
        ranked = self.reranker.rerank(
            filtered,
            query.text,
            top_k=query.top_k,
            session_id=query.session_id,
        )

        # Строим контекст
        context_pack, citations = self.context_builder.build(ranked, query)
        context_pack.dialog_episode_hits = self._build_dialog_episode_hits(
            runtime_session=runtime_session,
            topic_details=topic_details,
        )
        context_pack.task_continuity = self._build_task_continuity(runtime_session=runtime_session)
        context_pack.open_questions = self._build_open_questions(
            runtime_session=runtime_session,
            topic_details=topic_details,
        )
        context_pack.current_decisions = self._build_current_decisions(
            runtime_session=runtime_session,
            topic_details=topic_details,
        )
        context_pack.runtime_session = dict(runtime_session or {})
        context_pack.debug = {
            **dict(getattr(context_pack, "debug", {}) or {}),
            "runtime_session_present": bool(runtime_session),
            "runtime_session": dict(runtime_session or {}),
            "runtime_sources": self._summarize_retrieval_sources(ranked),
            "include_pending_facts_in_retrieval": bool(self.include_pending_facts_in_retrieval),
        }

        return context_pack, citations

    def _get_runtime_session(self, query: MemoryQuery) -> dict[str, Any]:
        if self.runtime_session_store is None or not str(query.session_id or "").strip():
            return {}
        snapshot = self.runtime_session_store.get_session(
            namespace=query.namespace,
            workspace_id=query.workspace_id,
            session_id=str(query.session_id or "").strip(),
        )
        row = snapshot.to_dict()
        if not any(
            row.get(key)
            for key in (
                "last_user_turn",
                "last_assistant_turn",
                "recent_turns",
                "active_topic",
                "active_task",
                "open_questions",
                "recent_decisions",
                "recent_user_state",
                "current_episode_id",
            )
        ):
            return {}
        return row

    def _get_topic_details(self, query: MemoryQuery, runtime_session: dict[str, Any]) -> dict[str, Any]:
        topic_thread_id = str(query.topic_thread_id or "").strip()
        if not topic_thread_id:
            topic_thread_id = str(dict(runtime_session.get("active_topic") or {}).get("thread_id") or "").strip()
        if not topic_thread_id:
            return {}
        try:
            return dict(
                self.topic_tools.read_topic(
                    topic_thread_id,
                    workspace_id=query.workspace_id,
                    limit=8,
                )
                or {}
            )
        except Exception:
            return {}

    def _collect_runtime_artifacts(
        self,
        query: MemoryQuery,
        *,
        runtime_session: dict[str, Any],
        topic_details: dict[str, Any],
    ) -> list[MemoryArtifact]:
        if self.runtime_session_store is None or not runtime_session:
            return []
        snapshot = RuntimeSessionSnapshot.from_dict(runtime_session)
        return self.runtime_session_store.to_retrieval_artifacts(
            snapshot,
            include_pending=bool(self.include_pending_facts_in_retrieval),
            topic_details=topic_details,
        )

    def _build_task_continuity(self, *, runtime_session: dict[str, Any]) -> dict[str, Any]:
        if self.runtime_session_store is None or not runtime_session:
            return {}
        snapshot = RuntimeSessionSnapshot.from_dict(runtime_session)
        return self.runtime_session_store.build_task_continuity(snapshot=snapshot)

    def _build_open_questions(
        self,
        *,
        runtime_session: dict[str, Any],
        topic_details: dict[str, Any],
    ) -> list[str]:
        return self._merge_strings(
            runtime_session.get("open_questions"),
            dict(topic_details or {}).get("open_questions"),
            limit=10,
        )

    def _build_current_decisions(
        self,
        *,
        runtime_session: dict[str, Any],
        topic_details: dict[str, Any],
    ) -> list[str]:
        return self._merge_strings(
            runtime_session.get("recent_decisions"),
            dict(topic_details or {}).get("current_decisions"),
            limit=10,
        )

    def _build_dialog_episode_hits(
        self,
        *,
        runtime_session: dict[str, Any],
        topic_details: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not runtime_session and not topic_details:
            return []
        active_topic = dict(runtime_session.get("active_topic") or {})
        thread = dict(dict(topic_details or {}).get("thread") or {})
        open_questions = self._build_open_questions(runtime_session=runtime_session, topic_details=topic_details)
        current_decisions = self._build_current_decisions(runtime_session=runtime_session, topic_details=topic_details)
        episode_id = str(
            runtime_session.get("current_episode_id")
            or active_topic.get("thread_id")
            or thread.get("thread_id")
            or ""
        ).strip()
        topic_title = str(
            active_topic.get("title")
            or thread.get("title")
            or active_topic.get("topic_key")
            or thread.get("topic_key")
            or ""
        ).strip()
        summary = str(dict(topic_details or {}).get("summary") or thread.get("summary") or "").strip()
        if not episode_id and not topic_title and not summary and not open_questions and not current_decisions:
            return []
        return [
            {
                "record_id": episode_id or topic_title or "runtime_episode",
                "summary_short": summary or topic_title,
                "summary_reasoning": str((current_decisions or [summary or topic_title or ""])[0] or "").strip(),
                "decisions": current_decisions,
                "episode": {
                    "id": episode_id or topic_title or "runtime_episode",
                    "topic": topic_title,
                    "summary_short": summary or topic_title,
                    "open_questions": open_questions,
                    "decisions": current_decisions,
                },
                "source": "runtime_session",
            }
        ]

    @staticmethod
    def _merge_strings(*values: Any, limit: int = 10) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            for item in list(value or []):
                text = str(item or "").strip()
                if not text:
                    continue
                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                result.append(text)
                if len(result) >= max(1, int(limit or 10)):
                    return result
        return result

    @staticmethod
    def _summarize_retrieval_sources(artifacts: list[MemoryArtifact]) -> dict[str, int]:
        summary: dict[str, int] = {}
        for artifact in list(artifacts or []):
            source = str(dict(artifact.metadata or {}).get("retrieval_source") or "unknown").strip() or "unknown"
            summary[source] = int(summary.get(source, 0) or 0) + 1
        return summary
    
    def _retrieve_artifacts(self, query: MemoryQuery) -> list[MemoryArtifact]:
        """
        Извлекает артефакты по запросу (legacy метод).

        Args:
            query: Запрос к памяти.

        Returns:
            Список артефактов.
        """
        # Создаём фильтры
        filters = RetrievalFilters(
            artifact_types=query.artifact_types if query.artifact_types else [],
            workspace_id=query.workspace_id,
            namespace=query.namespace,
        )

        # Получаем артефакты из хранилища
        all_artifacts = self.artifact_store.list_artifacts(
            workspace_id=query.workspace_id,
            namespace=query.namespace,
            limit=100,  # Берём с запасом для rerank
        )

        # Применяем фильтры
        filtered = apply_filters(all_artifacts, filters)

        # Если есть текст запроса, фильтруем по схожести
        if query.text:
            # Поиск по тексту
            text_matches = self.artifact_store.search_by_text(
                query=query.text,
                workspace_id=query.workspace_id,
                artifact_types=query.artifact_types if query.artifact_types else None,
                limit=50,
            )

            # Объединяем результаты
            artifact_ids = {a.artifact_id for a in filtered}
            for artifact in text_matches:
                if artifact.artifact_id not in artifact_ids:
                    filtered.append(artifact)

        # Rerank
        if query.text:
            filtered = self.reranker.rerank(filtered, query.text, top_k=query.top_k)

        return filtered

    def _collect_mandatory_artifacts(self, query: MemoryQuery) -> list[MemoryArtifact]:
        """
        Собирает обязательные артефакты (always_load).

        Args:
            query: Запрос к памяти.

        Returns:
            Список обязательных артефактов.
        """
        result: list[MemoryArtifact] = []

        for artifact_type in self.retrieval_config.always_load:
            artifacts = self.artifact_store.list_artifacts(
                artifact_type=artifact_type,
                workspace_id=query.workspace_id,
                namespace=query.namespace,
                status="active",
                limit=8,
            )
            result.extend(artifacts)

        # Для task_state фильтруем только открытые
        result = [
            a for a in result
            if a.artifact_type != "task_state" or a.metadata.get("task_status") == "open"
        ]

        return self._mark_retrieval_source(result, "mandatory")

    def _collect_continuity_artifacts(self, query: MemoryQuery) -> list[MemoryArtifact]:
        """
        Собирает артефакты continuity для текущей сессии.

        Args:
            query: Запрос к памяти.

        Returns:
            Список артефактов continuity.
        """
        if not query.session_id:
            return []

        candidates = self.artifact_store.list_artifacts(
            workspace_id=query.workspace_id,
            namespace=query.namespace,
            status="active",
            limit=80,
        )

        result: list[MemoryArtifact] = []
        same_topic: list[MemoryArtifact] = []
        related_topic: list[MemoryArtifact] = []
        session_fallback: list[MemoryArtifact] = []
        wanted_topic = str(query.topic_thread_id or "").strip()
        related_topics = {
            str(item).strip()
            for item in list(query.related_topic_ids or [])
            if str(item).strip()
        }
        for artifact in candidates:
            meta = artifact.metadata or {}
            if meta.get("session_id") != query.session_id:
                continue
            if artifact.artifact_type in {"episode_event", "emotional_state", "task_state", "task"}:
                artifact_topic = str(meta.get("topic_thread_id") or "").strip()
                if wanted_topic and artifact_topic == wanted_topic:
                    same_topic.append(artifact)
                elif artifact_topic and artifact_topic in related_topics:
                    related_topic.append(artifact)
                else:
                    session_fallback.append(artifact)

        result.extend(sorted(same_topic, key=lambda a: a.updated_at, reverse=True))
        result.extend(sorted(related_topic, key=lambda a: a.updated_at, reverse=True))
        result.extend(sorted(session_fallback, key=lambda a: a.updated_at, reverse=True))
        return self._mark_retrieval_source(result[:12], "continuity")

    def _collect_semantic_artifacts(self, query: MemoryQuery) -> list[MemoryArtifact]:
        """
        Собирает артефакты через semantic search (vector index).

        Args:
            query: Запрос к памяти.

        Returns:
            Список семантически релевантных артефактов.
        """
        if not query.text or self.vector_index is None:
            return []

        hits = self.vector_index.search(query.text, top_k=max(query.top_k * 3, 12))
        result: list[MemoryArtifact] = []
        seen: set[str] = set()

        for hit in hits:
            artifact_id = str(hit.get("artifact_id") or "").strip()
            if not artifact_id or artifact_id in seen:
                continue
            artifact = self.artifact_store.get_by_id(artifact_id)
            if artifact is None:
                continue
            seen.add(artifact_id)

            # Сохраняем score из semantic поиска
            artifact.metadata = dict(artifact.metadata or {})
            artifact.metadata["semantic_score"] = float(hit.get("score", 0.0))
            result.append(artifact)

        return self._mark_retrieval_source(result, "semantic")

    def _collect_lexical_artifacts(self, query: MemoryQuery) -> list[MemoryArtifact]:
        """
        Собирает артефакты через lexical search (LIKE fallback).

        Args:
            query: Запрос к памяти.

        Returns:
            Список артефактов с literal match.
        """
        if not query.text:
            return []

        return self._mark_retrieval_source(self.artifact_store.search_by_text(
            query=query.text,
            workspace_id=query.workspace_id,
            artifact_types=query.artifact_types if query.artifact_types else None,
            limit=max(query.top_k * 3, 12),
        ), "lexical")

    def _apply_topic_boost(
        self,
        artifacts: list[MemoryArtifact],
        query: MemoryQuery,
    ) -> list[MemoryArtifact]:
        topic_id = str(query.topic_thread_id or "").strip()
        related = {
            str(item).strip()
            for item in list(query.related_topic_ids or [])
            if str(item).strip()
        }
        if not topic_id and not related:
            return artifacts

        boosted: list[MemoryArtifact] = []
        for artifact in list(artifacts or []):
            meta = dict(artifact.metadata or {})
            artifact_topic_id = str(meta.get("topic_thread_id") or "").strip()
            score_boost = 0.0
            if topic_id and artifact_topic_id == topic_id:
                score_boost += 0.35
            elif artifact_topic_id and artifact_topic_id in related:
                score_boost += 0.15
            meta["_topic_boost"] = float(score_boost)
            meta["_topic_scope"] = (
                "same_topic"
                if topic_id and artifact_topic_id == topic_id
                else "related_topic"
                if artifact_topic_id and artifact_topic_id in related
                else "session_or_global"
            )
            artifact.metadata = meta
            boosted.append(artifact)
        return boosted

    def _merge_artifact_lists(
        self,
        runtime_artifacts: list[MemoryArtifact],
        mandatory: list[MemoryArtifact],
        continuity: list[MemoryArtifact],
        semantic: list[MemoryArtifact],
        lexical: list[MemoryArtifact],
    ) -> list[MemoryArtifact]:
        """
        Объединяет списки артефактов, удаляя дубликаты.

        Args:
            mandatory: Обязательные артефакты.
            continuity: Артефакты continuity.
            semantic: Семантические артефакты.
            lexical: Лексические артефакты.

        Returns:
            Объединённый список без дубликатов.
        """
        result: list[MemoryArtifact] = []
        seen: set[str] = set()

        for artifact in runtime_artifacts + mandatory + continuity + semantic + lexical:
            if artifact.artifact_id not in seen:
                seen.add(artifact.artifact_id)
                result.append(artifact)

        return result

    @staticmethod
    def _mark_retrieval_source(
        artifacts: list[MemoryArtifact],
        source: str,
    ) -> list[MemoryArtifact]:
        for artifact in list(artifacts or []):
            meta = dict(artifact.metadata or {})
            meta.setdefault("retrieval_source", source)
            artifact.metadata = meta
        return artifacts

    def _apply_runtime_rules(
        self,
        artifacts: list[MemoryArtifact],
        query: MemoryQuery,
    ) -> list[MemoryArtifact]:
        """
        Применяет runtime rules из конфига.

        Args:
            artifacts: Список артефактов.
            query: Запрос к памяти.

        Returns:
            Отфильтрованный список.
        """
        result: list[MemoryArtifact] = []
        seen: set[str] = set()

        for artifact in artifacts:
            if artifact.artifact_id in seen:
                continue

            meta = artifact.metadata or {}

            # workspace/namespace фильтры
            if artifact.workspace_id != query.workspace_id:
                continue
            if query.namespace and artifact.namespace != query.namespace:
                continue
            if query.artifact_types and artifact.artifact_type not in query.artifact_types:
                continue

            # never_load_if правила
            if artifact.status == self.retrieval_config.never_load_if_status:
                continue
            if str(meta.get("decay", "")).strip() == self.retrieval_config.never_load_if_decay:
                continue
            if float(meta.get("confidence", 0.5)) < self.retrieval_config.never_load_if_confidence_below:
                continue

            seen.add(artifact.artifact_id)
            result.append(artifact)

        return result
        
        # Rerank по релевантности
        if query.text:
            filtered = self.reranker.rerank(
                filtered,
                query.text,
                top_k=query.top_k,
            )
        else:
            # Если нет текста, просто берём top_k по recency
            filtered = sorted(
                filtered,
                key=lambda a: a.created_at,
                reverse=True,
            )[:query.top_k]
        
        return filtered
    
    def get_profile_facts(
        self,
        workspace_id: str = "global",
        limit: int = 20,
    ) -> list[MemoryArtifact]:
        """
        Получает профильные факты.
        
        Args:
            workspace_id: ID workspace.
            limit: Максимальное количество.
            
        Returns:
            Список профильных фактов.
        """
        return self.artifact_store.list_artifacts(
            artifact_type="profile_fact",
            workspace_id=workspace_id,
            status="active",
            limit=limit,
        )
    
    def get_active_tasks(
        self,
        workspace_id: str = "global",
        limit: int = 10,
    ) -> list[MemoryArtifact]:
        """
        Получает активные задачи.
        
        Args:
            workspace_id: ID workspace.
            limit: Максимальное количество.
            
        Returns:
            Список активных задач.
        """
        all_tasks = self.artifact_store.list_artifacts(
            artifact_type="task",
            workspace_id=workspace_id,
            status="active",
            limit=limit * 2,  # Берём с запасом
        )
        
        # Фильтруем только открытые
        open_tasks = [
            t for t in all_tasks
            if t.metadata.get("task_status") == "open"
        ]
        
        return open_tasks[:limit]
    
    def get_recent_episodes(
        self,
        workspace_id: str = "global",
        limit: int = 5,
    ) -> list[MemoryArtifact]:
        """
        Получает недавние эпизоды.
        
        Args:
            workspace_id: ID workspace.
            limit: Максимальное количество.
            
        Returns:
            Список недавних эпизодов.
        """
        return self.artifact_store.list_artifacts(
            artifact_type="episode",
            workspace_id=workspace_id,
            status="active",
            limit=limit,
        )
    
    def search_documents(
        self,
        query: str,
        workspace_id: str = "global",
        limit: int = 10,
    ) -> list[MemoryArtifact]:
        """
        Ищет по документам.
        
        Args:
            query: Поисковый запрос.
            workspace_id: ID workspace.
            limit: Максимальное количество.
            
        Returns:
            Список документ чанков.
        """
        return self.artifact_store.search_by_text(
            query=query,
            workspace_id=workspace_id,
            artifact_types=["document_chunk", "document_summary"],
            limit=limit,
        )
    
    def get_workspace_summary(self, workspace_id: str) -> str:
        """
        Получает краткую информацию о workspace.
        
        Args:
            workspace_id: ID workspace.
            
        Returns:
            Текстовое описание workspace.
        """
        from memory_core.storage.workspace_store import WorkspaceStore
        
        workspace_store = WorkspaceStore(self.db)
        workspace = workspace_store.get_workspace(workspace_id)
        
        if workspace:
            title = workspace.get("title", "Unknown")
            goal = workspace.get("goal", "")
            
            if goal:
                return f"Workspace: {title}\nGoal: {goal}"
            return f"Workspace: {title}"
        
        return f"Workspace: {workspace_id}"
