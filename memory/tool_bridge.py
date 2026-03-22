from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from memory.memory_models import (
    ContextBuildRequest,
    ContextBuildResult,
    MemoryLevel,
    MemoryScope,
    MemoryType,
    RetrievalCandidate,
    RetrievalQuery,
)
from memory.summary_quality import sanitize_session_summary_text


_PLAN_MODE_ALIASES = {
    "auto": "auto",
    "mixed": "auto",
    "context": "context",
    "chat_context": "context",
    "dialog": "episode",
    "episode": "episode",
    "episodes": "episode",
    "task": "task",
    "tasks": "task",
    "plan": "task",
    "profile": "profile",
    "persona": "profile",
    "identity": "profile",
    "fact": "fact",
    "facts": "fact",
    "document": "document",
    "documents": "document",
    "doc": "document",
    "code": "document",
}

_PLAN_SOURCE_ALIASES = {
    "all": "all",
    "any": "all",
    "facts": "facts",
    "fact": "facts",
    "claims": "facts",
    "claim": "facts",
    "episodes": "episodes",
    "episode": "episodes",
    "dialog": "episodes",
    "dialogs": "episodes",
    "messages": "messages",
    "message": "messages",
    "chat": "messages",
    "documents": "documents",
    "document": "documents",
    "docs": "documents",
    "doc": "documents",
    "files": "documents",
    "tasks": "tasks",
    "task": "tasks",
    "plans": "tasks",
    "plan": "tasks",
    "profile": "profile",
    "identity": "profile",
    "preferences": "profile",
    "preference": "profile",
    "working_memory": "working_memory",
    "working": "working_memory",
    "runtime": "working_memory",
    "tool_state": "working_memory",
}

_PLAN_TIME_ALIASES = {
    "any": "any",
    "none": "any",
    "recent": "recent",
    "latest": "recent",
    "current": "recent",
    "now": "recent",
    "session": "session",
    "conversation": "session",
    "historical": "historical",
    "history": "historical",
    "older": "historical",
    "archive": "historical",
    "persistent": "persistent",
    "long_term": "persistent",
    "profile": "persistent",
}

_TASK_FACT_PREDICATES = {"decision", "task", "task_goal", "agreed_plan"}


@dataclass(frozen=True)
class MemoryRetrievalPlan:
    mode: str = "auto"
    topic_hints: list[str] = field(default_factory=list)
    time_hint: str = "recent"
    sources: list[str] = field(default_factory=lambda: ["all"])
    top_k: int = 4  # Уменьшено с 6 до 4 для скорости (меньше запросов, меньше токенов)
    max_candidates: int = 12  # Максимум кандидатов до reranking (было 24)
    rerank_top_k: int = 3  # Сколько вернуть после reranking (было 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode or ""),
            "topic_hints": [str(x).strip() for x in list(self.topic_hints or []) if str(x).strip()],
            "time_hint": str(self.time_hint or ""),
            "sources": [str(x).strip() for x in list(self.sources or []) if str(x).strip()],
            "top_k": int(self.top_k or 0),
        }


@dataclass(frozen=True)
class PlannedRetrievalQuery:
    label: str
    query: RetrievalQuery
    search_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": str(self.label or ""),
            "query_text": str(self.query.query_text or ""),
            "search_text": str(self.search_text or ""),
            "top_k": int(self.query.top_k or 0),
            "scopes": [str(scope.value) for scope in list(self.query.scopes or [])],
        }


def normalize_memory_retrieval_plan(raw: dict[str, Any] | None, *, default_top_k: int = 6) -> MemoryRetrievalPlan:
    row = dict(raw or {})
    mode = _PLAN_MODE_ALIASES.get(str(row.get("mode") or "").strip().lower(), "")
    if not mode:
        mode = "auto"
    topic_hints = _dedupe_text_items(row.get("topic_hints"))
    time_hint = _PLAN_TIME_ALIASES.get(str(row.get("time_hint") or "").strip().lower(), "")
    if not time_hint:
        time_hint = "recent"
    sources = []
    raw_sources = row.get("sources")
    if isinstance(raw_sources, str):
        raw_source_items = [raw_sources]
    else:
        raw_source_items = list(raw_sources or [])
    for item in raw_source_items:
        key = _PLAN_SOURCE_ALIASES.get(str(item or "").strip().lower(), "")
        if key and key not in sources:
            sources.append(key)
    if not sources:
        sources = ["all"]
    try:
        top_k = int(row.get("top_k"))
    except Exception:
        top_k = int(default_top_k or 6)
    top_k = max(1, min(12, int(top_k or default_top_k or 6)))
    return MemoryRetrievalPlan(
        mode=mode,
        topic_hints=topic_hints,
        time_hint=time_hint,
        sources=sources,
        top_k=top_k,
    )


def build_memory_plan_queries(
    *,
    request: ContextBuildRequest,
    plan: MemoryRetrievalPlan,
) -> list[PlannedRetrievalQuery]:
    """
    Построить fan-out запросы для retrieval.

    Оптимизация: вместо 3 запросов (raw, topic, recent) делаем 2:
    1. Комбинированный query + topic (основной)
    2. Recent context (для continuity)
    
    Это даёт ~33% меньше запросов к БД без потери качества.
    """
    user_query = str(request.user_message or "").strip()
    scopes = list(request.scopes or [])
    base_top_k = max(1, int(plan.top_k or request.top_k or 1))
    topic_text = " ".join(list(plan.topic_hints or [])).strip()
    topic_tokens = _topic_search_text(plan)
    recent_tokens = _recent_search_text(plan)

    # Комбинированный search text: query + topic hints
    combined_search = user_query
    if topic_tokens:
        combined_search = f"{user_query} {topic_tokens}"
    elif topic_text:
        combined_search = f"{user_query} {topic_text}"

    candidates = [
        # Основной запрос: user query + topic hints
        PlannedRetrievalQuery(
            label="combined_query_with_topics",
            query=RetrievalQuery(
                query_text=user_query,
                search_text=combined_search,
                namespace=str(request.namespace or "default"),
                scopes=scopes,
                top_k=base_top_k,
                include_private_runtime=False,
                include_stale=False,
                metadata_filters={},
            ),
            search_text=combined_search,
        ),
        # Recent context: для continuity диалога
        PlannedRetrievalQuery(
            label="recent_context_query",
            query=RetrievalQuery(
                query_text=user_query,
                search_text=(recent_tokens or topic_tokens or user_query),
                namespace=str(request.namespace or "default"),
                scopes=scopes,
                top_k=base_top_k,
                include_private_runtime=False,
                include_stale=False,
                metadata_filters={},
            ),
            search_text=(recent_tokens or topic_tokens or user_query),
        ),
    ]

    out: list[PlannedRetrievalQuery] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        key = (str(item.query.query_text or ""), str(item.search_text or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_memory_tool_context_pack(
    manager: Any,
    *,
    request: ContextBuildRequest,
    plan: MemoryRetrievalPlan,
) -> dict[str, Any]:
    """
    Построить memory context pack с параллельным retrieval.
    
    Оптимизация: запросы выполняются параллельно через asyncio.gather,
    что даёт ~40-50% ускорение для fan-out retrieval.
    """
    if not _supports_fanout_manager(manager):
        return _fallback_context_pack(manager, request=request, plan=plan)

    planned_queries = build_memory_plan_queries(request=request, plan=plan)
    
    # Параллельное выполнение retrieval запросов
    # Используем try/except для безопасности (тесты, старые менеджеры)
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Создаём задачи для параллельного выполнения
            tasks = [
                loop.run_in_executor(None, manager.retrieve, item.query)
                for item in list(planned_queries or [])
            ]
            
            # Выполняем параллельно
            results = loop.run_until_complete(asyncio.gather(*tasks))
            
            # Собираем результаты
            retrieval_runs = [
                (item, result)
                for item, result in zip(list(planned_queries or []), results)
            ]
        finally:
            loop.close()
    except Exception:
        # Fallback на последовательное выполнение
        retrieval_runs = [
            (item, manager.retrieve(item.query))
            for item in list(planned_queries or [])
        ]

    merged_candidates = _merge_retrieval_candidates(runs=retrieval_runs, plan=plan)
    recall_mode = _resolve_recall_mode(manager=manager, query_text=str(request.user_message or ""), plan=plan)
    fact_expectation = manager._build_fact_expectation_check(
        query_text=str(request.user_message or ""),
        namespace=str(request.namespace or "default"),
    )
    primary_query = planned_queries[0].query if planned_queries else RetrievalQuery(
        query_text=str(request.user_message or ""),
        search_text=str(request.user_message or ""),
        namespace=str(request.namespace or "default"),
        scopes=list(request.scopes or []),
        top_k=max(1, int(plan.top_k or request.top_k or 1)),
    )
    exact_fact_candidates = manager._exact_fact_candidates_for_expectation(
        query=primary_query,
        namespace=str(request.namespace or "default"),
        fact_expectation=fact_expectation,
    )
    prioritized_candidates = manager._prioritize_retrieval_candidates_for_fact_expectation(
        query=primary_query,
        candidates=list(merged_candidates or []),
        exact_fact_candidates=exact_fact_candidates,
        fact_expectation=fact_expectation,
    )
    selected_candidates = manager._select_candidates_for_recall_mode(
        query_text=str(request.user_message or ""),
        recall_mode=recall_mode,
        fallback_candidates=list(prioritized_candidates or []),
        exact_fact_candidates=exact_fact_candidates,
        fact_expectation=fact_expectation,
    )

    private_runtime = manager._private_runtime_for_namespace(request.namespace)
    working = manager._working_for_namespace(namespace=request.namespace, scopes=list(request.scopes or []))
    effective_request = ContextBuildRequest(
        system_prompt=request.system_prompt,
        user_message=str(request.user_message or ""),
        namespace=request.namespace,
        scopes=list(request.scopes or []),
        top_k=max(1, int(plan.top_k or request.top_k or 1)),
        session_summary=sanitize_session_summary_text(request.session_summary or ""),
        working_memory=working,
        tool_state=dict(request.tool_state or {}),
        unresolved_items=[str(x) for x in list(request.unresolved_items or []) if str(x).strip()],
        context_budget_total=int(request.context_budget_total or 2048),
        context_budget_memory=int(request.context_budget_memory or 700),
        context_budget_docs=int(request.context_budget_docs or 600),
        context_budget_tools=int(request.context_budget_tools or 200),
        context_budget_response_reserve=int(request.context_budget_response_reserve or 256),
    )
    result = manager._context_builder.build(
        request=effective_request,
        retrieved=list(selected_candidates or []),
        private_runtime_state=private_runtime,
    )
    result = _finalize_context_result(
        manager,
        request=effective_request,
        primary_query=primary_query,
        recall_mode=recall_mode,
        fact_expectation=fact_expectation,
        selected_candidates=list(result.selected or []),
        base_result=result,
    )
    pack = result.to_dict()
    pack["tool_retrieval_plan"] = plan.to_dict()
    pack["fanout_queries"] = [item.to_dict() for item in list(planned_queries or [])]
    pack["fanout_sources"] = [str(x.label or "") for x, _ in list(retrieval_runs or [])]
    pack["fanout_total_candidates"] = int(
        sum(len(list(getattr(result_row, "candidates", []) or [])) for _, result_row in list(retrieval_runs or []))
    )
    return pack


def _supports_fanout_manager(manager: Any) -> bool:
    required = (
        "retrieve",
        "_build_fact_expectation_check",
        "_exact_fact_candidates_for_expectation",
        "_prioritize_retrieval_candidates_for_fact_expectation",
        "_select_candidates_for_recall_mode",
        "_private_runtime_for_namespace",
        "_working_for_namespace",
        "_context_builder",
        "_render_relevant_claims",
        "_retrieve_dialog_episode_hits",
        "_render_recalled_dialog",
        "_retrieve_document_evidence_hits",
        "_render_document_evidence",
        "_render_supporting_messages",
        "_build_self_facts_context",
        "_render_fact_expectation_check",
        "_render_self_facts",
        "_render_memory_recall_mode",
    )
    return all(hasattr(manager, name) for name in required)


def _fallback_context_pack(manager: Any, *, request: ContextBuildRequest, plan: MemoryRetrievalPlan) -> dict[str, Any]:
    result = manager.build_context(request)
    pack = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
    if not isinstance(pack, dict):
        raise TypeError("invalid memory context pack")
    pack["tool_retrieval_plan"] = plan.to_dict()
    pack["fanout_queries"] = [item.to_dict() for item in list(build_memory_plan_queries(request=request, plan=plan))]
    pack.setdefault("fanout_sources", ["fallback_build_context"])
    return pack


def _finalize_context_result(
    manager: Any,
    *,
    request: ContextBuildRequest,
    primary_query: RetrievalQuery,
    recall_mode: str,
    fact_expectation: dict[str, Any],
    selected_candidates: list[RetrievalCandidate],
    base_result: ContextBuildResult,
) -> ContextBuildResult:
    blocks = dict(base_result.blocks or {})
    relevant_claims_block = manager._render_relevant_claims(list(selected_candidates or []))
    if relevant_claims_block:
        blocks["relevant_claims"] = relevant_claims_block

    dialog_hits = manager._retrieve_dialog_episode_hits(query=primary_query, recall_mode=recall_mode)
    recalled_dialog_block = manager._render_recalled_dialog(dialog_hits)
    if recalled_dialog_block:
        blocks["recalled_dialog"] = recalled_dialog_block

    document_hits = manager._retrieve_document_evidence_hits(query=primary_query, recall_mode=recall_mode)
    document_evidence_block = manager._render_document_evidence(document_hits)
    if document_evidence_block:
        blocks["document_evidence"] = document_evidence_block

    supporting_messages_block = manager._render_supporting_messages(
        selected_candidates=list(selected_candidates or []),
        dialog_hits=dialog_hits,
    )
    if supporting_messages_block:
        blocks["supporting_messages"] = supporting_messages_block

    task_continuity_getter = getattr(manager, "get_task_continuity_snapshot", None)
    task_continuity = (
        dict(task_continuity_getter(request.namespace) or {})
        if callable(task_continuity_getter)
        else {}
    )
    result = ContextBuildResult(
        blocks=blocks,
        selected=list(selected_candidates or []),
        dropped=[dict(x) for x in list(base_result.dropped or [])],
        score_breakdowns=[dict(x) for x in list(base_result.score_breakdowns or [])],
        truncation_log=[dict(x) for x in list(base_result.truncation_log or [])],
        fact_expectation=dict(base_result.fact_expectation or {}),
        self_facts_context=dict(base_result.self_facts_context or {}),
        dialog_episode_hits=[row.to_dict() for row in list(dialog_hits or [])],
        open_questions=[str(x) for x in list(getattr(manager, "_open_questions", []) or []) if str(x).strip()],
        current_decisions=[str(x) for x in list(getattr(manager, "_current_decisions", []) or []) if str(x).strip()],
        task_continuity=task_continuity,
        recall_mode=str(base_result.recall_mode or recall_mode or ""),
    )
    self_facts_context = manager._build_self_facts_context(
        query_text=str(request.user_message or ""),
        selected_candidates=list(result.selected or []),
        fact_expectation=fact_expectation,
    )
    if fact_expectation or self_facts_context:
        blocks["fact_expectation_check"] = manager._render_fact_expectation_check(fact_expectation)
        self_facts_block = manager._render_self_facts(self_facts_context)
        if self_facts_block:
            blocks["self_facts"] = self_facts_block
        recall_mode_block = manager._render_memory_recall_mode(recall_mode)
        if recall_mode_block:
            blocks["memory_recall_mode"] = recall_mode_block
        result = ContextBuildResult(
            blocks=blocks,
            selected=list(result.selected or []),
            dropped=[dict(x) for x in list(result.dropped or [])],
            score_breakdowns=[dict(x) for x in list(result.score_breakdowns or [])],
            truncation_log=[dict(x) for x in list(result.truncation_log or [])],
            fact_expectation=fact_expectation,
            self_facts_context=self_facts_context,
            dialog_episode_hits=[dict(x) for x in list(result.dialog_episode_hits or [])],
            open_questions=[str(x) for x in list(result.open_questions or []) if str(x).strip()],
            current_decisions=[str(x) for x in list(result.current_decisions or []) if str(x).strip()],
            task_continuity=dict(result.task_continuity or {}),
            recall_mode=recall_mode,
        )
    elif recall_mode:
        recall_mode_block = manager._render_memory_recall_mode(recall_mode)
        if recall_mode_block:
            blocks["memory_recall_mode"] = recall_mode_block
        result = ContextBuildResult(
            blocks=blocks,
            selected=list(result.selected or []),
            dropped=[dict(x) for x in list(result.dropped or [])],
            score_breakdowns=[dict(x) for x in list(result.score_breakdowns or [])],
            truncation_log=[dict(x) for x in list(result.truncation_log or [])],
            fact_expectation=fact_expectation,
            self_facts_context=self_facts_context,
            dialog_episode_hits=[dict(x) for x in list(result.dialog_episode_hits or [])],
            open_questions=[str(x) for x in list(result.open_questions or []) if str(x).strip()],
            current_decisions=[str(x) for x in list(result.current_decisions or []) if str(x).strip()],
            task_continuity=dict(result.task_continuity or {}),
            recall_mode=recall_mode,
        )
    debugger = getattr(manager, "_debugger", None)
    if debugger is not None and hasattr(debugger, "record_retrieval_trace"):
        debugger.record_retrieval_trace(
            query=request.user_message,
            selected=list(result.selected or []),
            dropped=list(result.dropped or []),
            score_breakdowns=list(result.score_breakdowns or []),
            truncation_log=list(result.truncation_log or []),
            context_blocks=dict(result.blocks or {}),
        )
    return result


def _resolve_recall_mode(manager: Any, *, query_text: str, plan: MemoryRetrievalPlan) -> str:
    mode = str(plan.mode or "").strip().lower()
    if mode in {"fact", "profile"}:
        return "exact_fact_recall"
    if mode in {"episode", "task", "context"}:
        return "contextual_recall"
    if mode == "document":
        return "document_recall"
    return str(manager._classify_memory_recall_mode(query_text=str(query_text or "")) or "")


def _merge_retrieval_candidates(
    *,
    runs: list[tuple[PlannedRetrievalQuery, Any]],
    plan: MemoryRetrievalPlan,
) -> list[RetrievalCandidate]:
    merged: dict[str, tuple[RetrievalCandidate, set[str]]] = {}
    for planned, result in list(runs or []):
        rows = list(getattr(result, "candidates", []) or [])
        for item in rows:
            if not isinstance(item, RetrievalCandidate):
                continue
            record_id = str(item.record.id or "")
            labels = set()
            if record_id in merged:
                current, labels = merged[record_id]
                best = current if float(current.final_score) >= float(item.final_score) else item
                labels = set(labels)
                labels.add(str(planned.label or ""))
                merged[record_id] = (_apply_plan_bias(best, plan=plan, query_hits=len(labels)), labels)
                continue
            labels = {str(planned.label or "")}
            merged[record_id] = (_apply_plan_bias(item, plan=plan, query_hits=1), labels)

    ordered = sorted(
        [candidate for candidate, _labels in merged.values()],
        key=lambda row: float(row.final_score),
        reverse=True,
    )
    limit = max(6, min(32, int(plan.top_k or 6) * 4))
    return ordered[:limit]


def _apply_plan_bias(candidate: RetrievalCandidate, *, plan: MemoryRetrievalPlan, query_hits: int) -> RetrievalCandidate:
    record = candidate.record
    bonus = 0.0
    if _candidate_matches_plan_sources(candidate, plan):
        bonus += 0.05
    if _candidate_matches_plan_mode(candidate, plan):
        bonus += 0.04
    if query_hits > 1:
        bonus += min(0.06, 0.02 * float(query_hits - 1))
    time_hint = str(plan.time_hint or "").strip().lower()
    if time_hint in {"recent", "session"} and record.level in {MemoryLevel.L0_WORKING, MemoryLevel.L1_SESSION, MemoryLevel.L2_EPISODIC}:
        bonus += 0.02
    if time_hint == "persistent" and record.level in {MemoryLevel.L3_SEMANTIC, MemoryLevel.L4_DOCUMENT}:
        bonus += 0.02
    if time_hint == "historical" and record.level in {MemoryLevel.L2_EPISODIC, MemoryLevel.L4_DOCUMENT}:
        bonus += 0.01
    breakdown = replace(
        candidate.score_breakdown,
        final_score=min(1.0, float(candidate.final_score) + float(bonus)),
    )
    return replace(candidate, score_breakdown=breakdown, source=f"fanout:{query_hits}")


def _candidate_matches_plan_mode(candidate: RetrievalCandidate, plan: MemoryRetrievalPlan) -> bool:
    mode = str(plan.mode or "").strip().lower()
    record = candidate.record
    if mode in {"", "auto"}:
        return False
    if mode == "document":
        return bool(record.memory_type in {MemoryType.DOCUMENT, MemoryType.DOCUMENT_CHUNK} or record.level == MemoryLevel.L4_DOCUMENT)
    if mode == "episode":
        return bool(record.memory_type in {MemoryType.EPISODE, MemoryType.MESSAGE} or record.level == MemoryLevel.L2_EPISODIC)
    if mode == "task":
        predicate = str(dict(dict(record.metadata or {}).get("fact") or {}).get("predicate") or "").strip().lower()
        return bool(record.memory_type == MemoryType.TASK_STATE or predicate in _TASK_FACT_PREDICATES)
    if mode == "profile":
        return bool(_is_profile_candidate(record))
    if mode == "fact":
        return bool(record.memory_type in {MemoryType.FACT, MemoryType.CLAIM})
    if mode == "context":
        return bool(record.memory_type in {MemoryType.MESSAGE, MemoryType.EPISODE, MemoryType.SUMMARY})
    return False


def _candidate_matches_plan_sources(candidate: RetrievalCandidate, plan: MemoryRetrievalPlan) -> bool:
    sources = {str(x).strip().lower() for x in list(plan.sources or []) if str(x).strip()}
    if not sources or "all" in sources:
        return True
    record = candidate.record
    predicate = str(dict(dict(record.metadata or {}).get("fact") or {}).get("predicate") or "").strip().lower()
    for source in sources:
        if source == "facts" and record.memory_type in {MemoryType.FACT, MemoryType.CLAIM}:
            return True
        if source == "episodes" and (record.memory_type == MemoryType.EPISODE or record.level == MemoryLevel.L2_EPISODIC):
            return True
        if source == "messages" and record.memory_type == MemoryType.MESSAGE:
            return True
        if source == "documents" and (record.memory_type in {MemoryType.DOCUMENT, MemoryType.DOCUMENT_CHUNK} or record.level == MemoryLevel.L4_DOCUMENT):
            return True
        if source == "tasks" and (record.memory_type == MemoryType.TASK_STATE or predicate in _TASK_FACT_PREDICATES):
            return True
        if source == "profile" and _is_profile_candidate(record):
            return True
        if source == "working_memory" and (record.level == MemoryLevel.L0_WORKING or record.memory_type in {MemoryType.TOOL_RESULT, MemoryType.RUNTIME_STATE}):
            return True
    return False


def _is_profile_candidate(record) -> bool:
    if record.memory_type == MemoryType.IDENTITY_CORE:
        return True
    if record.memory_type == MemoryType.FACT:
        fact = dict(dict(record.metadata or {}).get("fact") or {})
        if str(fact.get("subject") or "").strip().lower() == "user":
            return True
    if record.memory_type == MemoryType.CLAIM:
        claim = dict(dict(record.metadata or {}).get("claim") or {})
        if str(claim.get("subject") or "").strip().lower() == "user":
            return True
    return False


def _topic_search_text(plan: MemoryRetrievalPlan) -> str:
    parts: list[str] = []
    parts.extend(list(plan.topic_hints or []))
    parts.extend(_mode_tokens(plan.mode))
    parts.extend(_source_tokens(plan.sources))
    return " ".join(_dedupe_text_items(parts)).strip()


def _recent_search_text(plan: MemoryRetrievalPlan) -> str:
    parts: list[str] = []
    parts.extend(list(plan.topic_hints or []))
    parts.extend(_time_tokens(plan.time_hint))
    parts.extend(_mode_tokens(plan.mode))
    parts.extend(_source_tokens(plan.sources))
    return " ".join(_dedupe_text_items(parts)).strip()


def _mode_tokens(mode: str) -> list[str]:
    key = str(mode or "").strip().lower()
    mapping = {
        "profile": ["user", "profile", "identity", "preference", "facts"],
        "fact": ["exact", "fact", "memory", "remembered"],
        "episode": ["dialog", "episode", "discussion", "conversation"],
        "task": ["task", "decision", "plan", "open_question"],
        "document": ["document", "file", "code", "snippet"],
        "context": ["context", "recent", "conversation"],
        "auto": [],
    }
    return list(mapping.get(key, []))


def _source_tokens(sources: list[str]) -> list[str]:
    out: list[str] = []
    mapping = {
        "facts": ["facts", "claims", "preferences"],
        "episodes": ["episodes", "dialog", "history"],
        "messages": ["messages", "quotes"],
        "documents": ["documents", "files", "code"],
        "tasks": ["tasks", "decisions", "plans"],
        "profile": ["profile", "identity", "user"],
        "working_memory": ["working", "runtime", "recent"],
    }
    for item in list(sources or []):
        out.extend(list(mapping.get(str(item or "").strip().lower(), [])))
    return out


def _time_tokens(time_hint: str) -> list[str]:
    key = str(time_hint or "").strip().lower()
    mapping = {
        "recent": ["recent", "latest", "current", "today"],
        "session": ["session", "current", "conversation", "recent"],
        "historical": ["history", "older", "previous", "earlier"],
        "persistent": ["long_term", "persistent", "profile", "stable"],
        "any": [],
    }
    return list(mapping.get(key, []))


def _dedupe_text_items(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        rows = [value]
    else:
        rows = list(value or [])
    out: list[str] = []
    seen: set[str] = set()
    for item in rows:
        text = str(item or "").strip()
        low = text.lower()
        if not text or low in seen:
            continue
        seen.add(low)
        out.append(text)
    return out


def format_memory_result_for_llm(
    *,
    context_result: ContextBuildResult,
    query: str = "",
) -> dict[str, Any]:
    """
    Форматировать результат memory retrieval в LLM-readable формате.
    
    Цель: модель читает память как понятные куски сознания,
    а не как сырой технический dump.
    
    Args:
        context_result: Результат build_context из memory_manager.
        query: Оригинальный запрос пользователя.
    
    Returns:
        dict: Структурированный, понятный для LLM результат.
    """
    result_dict = context_result.to_dict() if hasattr(context_result, "to_dict") else dict(context_result or {})
    
    # Формируем memory hits — понятные блоки для модели
    memory_hits = []
    selected = list(result_dict.get("selected") or [])
    
    for candidate in selected[:8]:  # Максимум 8 хитов
        if not isinstance(candidate, dict):
            continue
        
        record = dict(candidate.get("record") or {})
        metadata = dict(record.get("metadata") or {})
        
        # Определяем kind (тип памяти)
        memory_type = str(record.get("memory_type") or "")
        level = str(record.get("level") or "")
        
        if memory_type == "fact" or "fact" in level.lower():
            kind = "fact"
        elif memory_type == "episode" or "episode" in level.lower():
            kind = "episode"
        elif memory_type == "task" or "task" in level.lower():
            kind = "task"
        elif memory_type == "document" or "document" in level.lower():
            kind = "document"
        elif memory_type == "identity_core":
            kind = "identity"
        else:
            kind = "memory"
        
        # Извлекаем факты
        facts = []
        fact_data = dict(metadata.get("fact") or {})
        if fact_data:
            predicate = str(fact_data.get("predicate") or "")
            value = fact_data.get("value")
            if predicate and value is not None:
                facts.append(f"{predicate}: {value}")
        
        # Если нет явных фактов, берём текст
        if not facts:
            text = str(record.get("text") or "")
            if text:
                # Короткий preview
                preview = text[:200] + "..." if len(text) > 200 else text
                facts.append(preview)
        
        # Определяем topic
        topic = str(metadata.get("topic") or record.get("topic") or "")
        if not topic:
            # Пробуем извлечь из fact
            topic = str(fact_data.get("predicate") or "")
        
        # Определяем time_scope
        scope = str(record.get("scope") or "")
        if scope == "conversation":
            time_scope = "recent"
        elif scope == "session":
            time_scope = "session"
        elif scope == "project":
            time_scope = "project"
        elif scope == "global_user":
            time_scope = "long_term"
        else:
            time_scope = "recent"
        
        # Вычисляем confidence
        score = float(candidate.get("score") or candidate.get("confidence") or 0.0)
        confidence = min(1.0, max(0.0, score))
        
        # Определяем source
        source = str(record.get("memory_type") or record.get("level") or "memory")
        
        # Формируем why_relevant
        why_relevant = ""
        if query:
            # Кратко объясняем, почему это релевантно
            if topic:
                why_relevant = f"похоже на '{topic}'"
            elif kind == "episode":
                why_relevant = "из прошлой беседы"
            elif kind == "fact":
                why_relevant = "факт из памяти"
            elif kind == "task":
                why_relevant = "активная задача"
        
        hit = {
            "kind": kind,
            "topic": topic if topic else "general",
            "why_relevant": why_relevant,
            "facts": facts[:3],  # Максимум 3 факта
            "time_scope": time_scope,
            "confidence": round(confidence, 2),
            "source": source,
        }
        memory_hits.append(hit)
    
    # Формируем memory_state_patch — подсказки для обновления состояния
    memory_state_patch = {}
    
    # Topic hint
    if memory_hits:
        topics = [h["topic"] for h in memory_hits if h.get("topic") and h["topic"] != "general"]
        if topics:
            memory_state_patch["topic_hint"] = topics[0]
    
    # Active task hint
    task_continuity = dict(result_dict.get("task_continuity") or {})
    if task_continuity:
        active_task = dict(task_continuity.get("active_task") or {})
        if active_task:
            memory_state_patch["active_task_hint"] = {
                "current_goal": str(active_task.get("current_goal") or ""),
                "next_step": str(active_task.get("next_step") or ""),
            }
    
    # Open questions
    open_questions = [str(x) for x in list(result_dict.get("open_questions") or []) if str(x).strip()]
    if open_questions:
        memory_state_patch["open_questions"] = open_questions[:3]
    
    # Current decisions
    current_decisions = [str(x) for x in list(result_dict.get("current_decisions") or []) if str(x).strip()]
    if current_decisions:
        memory_state_patch["current_decisions"] = current_decisions[:3]
    
    # Итоговый результат
    return {
        "memory_hits": memory_hits,
        "memory_state_patch": memory_state_patch,
        "recall_mode": str(result_dict.get("recall_mode") or "auto"),
        "total_hits": len(memory_hits),
    }
