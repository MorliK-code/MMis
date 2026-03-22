"""
Project-aware topic extraction для memory retrieval.

Содержит словарь проектных терминов для улучшения извлечения тем.
Помогает LLM не обязана идеально формулировать topic_hints.
"""
from __future__ import annotations

from typing import Any


# Проектные термины для memory architecture
MEMORY_PROJECT_TERMS = {
    # Ядро памяти
    "memory_governor": ["governor", "memory governor", "governor policy", "conflict resolution"],
    "persona_snapshot": ["persona snapshot", "persona compiler", "compiled persona"],
    "identity_core": ["identity core", "identity facts", "user identity", "canonical name"],
    "episode_planner": ["episode planner", "task continuity", "active task"],
    "memory_native_state": ["native state", "memory native", "always-on memory", "working memory state"],
    "summary": ["summary", "dialog summary", "long summary", "session summary", "rolling summary"],
    "tool_bridge": ["tool bridge", "memory tool", "memory_retrieve", "retrieval plan"],
    "claim_promotion": ["claim promotion", "fact promotion", "claim validation"],
    "working_memory": ["working memory", "runtime state", "tool state", "private runtime"],
    
    # Уровни памяти
    "episodic_memory": ["episodic", "episodes", "dialog history", "conversation episodes"],
    "semantic_memory": ["semantic", "facts", "claims", "preferences", "identity facts"],
    "procedural_memory": ["procedural", "skills", "capabilities", "tools"],
    
    # Scopes
    "conversation_scope": ["conversation", "current dialog", "this conversation"],
    "session_scope": ["session", "current session", "today"],
    "project_scope": ["project", "codebase", "this project", "MMis"],
    "global_user_scope": ["global", "user profile", "long-term", "persistent"],
    "character_scope": ["character", "persona", "personality"],
    "temporary_scope": ["temporary", "transient", "short-term"],
    
    # Memory operations
    "memory_retrieve": ["retrieve", "recall", "fetch memory", "get memory", "remember"],
    "memory_write": ["write memory", "save memory", "store", "ingest"],
    "memory_consolidate": ["consolidate", "merge", "update memory"],
    "memory_decay": ["decay", "archive", "forget", "deprecate"],
    
    # Query types
    "fact_query": ["exact fact", "specific fact", "do you remember", "what is"],
    "episode_query": ["when did", "what did we discuss", "earlier conversation"],
    "task_query": ["what's the task", "current goal", "what are we doing", "active task"],
    "preference_query": ["what do I prefer", "my preference", "how I like"],
    "identity_query": ["what's my name", "how to call me", "who am I"],
}

# Алиасы для режимов retrieval
RETRIEVAL_MODE_ALIASES = {
    "auto": ["auto", "automatic", "default", "any"],
    "context": ["context", "chat context", "dialog context", "recent context"],
    "episode": ["episode", "dialog", "conversation", "discussion", "chat history"],
    "task": ["task", "tasks", "plan", "planning", "goal", "goals", "active task"],
    "profile": ["profile", "persona", "identity", "user profile", "preferences"],
    "fact": ["fact", "facts", "claim", "claims", "exact fact", "specific"],
    "document": ["document", "documents", "docs", "files", "code", "snippets"],
    "working": ["working", "runtime", "tool state", "recent state"],
}

# Алиасы для источников
RETRIEVAL_SOURCE_ALIASES = {
    "all": ["all", "any", "everything", "any source"],
    "facts": ["facts", "claims", "preferences", "identity facts", "semantic"],
    "episodes": ["episodes", "dialog", "history", "conversation", "chat history"],
    "messages": ["messages", "quotes", "exact messages", "chat messages"],
    "documents": ["documents", "files", "code", "snippets", "docs"],
    "tasks": ["tasks", "decisions", "plans", "goals", "active tasks"],
    "profile": ["profile", "identity", "user", "persona", "preferences"],
    "working_memory": ["working", "runtime", "recent", "tool state"],
}

# Алиасы для временных диапазонов
RETRIEVAL_TIME_ALIASES = {
    "any": ["any", "none", "all time", "whenever"],
    "recent": ["recent", "latest", "current", "now", "today", "just now"],
    "session": ["session", "conversation", "this session", "current session"],
    "historical": ["historical", "history", "older", "previous", "earlier", "past"],
    "persistent": ["persistent", "long_term", "profile", "stable", "permanent"],
}


def normalize_topic(topic: str) -> str:
    """
    Нормализовать тему, приводя к каноническому термину.
    
    Args:
        topic: Исходная тема (может быть неточной).
    
    Returns:
        str: Нормализованная тема или исходная, если нет совпадений.
    """
    topic_lower = str(topic or "").strip().lower()
    if not topic_lower:
        return ""
    
    # Ищем совпадения в проектных терминах
    for canonical, aliases in MEMORY_PROJECT_TERMS.items():
        # Проверяем точное совпадение
        if topic_lower == canonical.lower():
            return canonical
        
        # Проверяем совпадение с алиасами
        for alias in aliases:
            if topic_lower == alias.lower():
                return canonical
            # Проверяем частичное совпадение (если алиас содержится в теме)
            if alias.lower() in topic_lower and len(alias) > 4:
                return canonical
    
    # Нет совпадений — возвращаем исходную тему
    return topic


def extract_topics_from_query(query: str) -> list[str]:
    """
    Извлечь темы из запроса.
    
    Args:
        query: Текст запроса.
    
    Returns:
        list[str]: Список извлечённых тем (нормализованных).
    """
    query_lower = str(query or "").strip().lower()
    if not query_lower:
        return []
    
    topics = []
    seen = set()
    
    # Ищем совпадения с проектными терминами
    for canonical, aliases in MEMORY_PROJECT_TERMS.items():
        all_terms = [canonical] + list(aliases)
        for term in all_terms:
            term_lower = term.lower()
            if len(term_lower) < 4:
                continue
            if term_lower in query_lower and canonical not in seen:
                topics.append(canonical)
                seen.add(canonical)
                break
    
    return topics


def classify_query_mode(query: str) -> str:
    """
    Классифицировать режим retrieval для запроса.
    
    Args:
        query: Текст запроса.
    
    Returns:
        str: Режим retrieval (auto, context, episode, task, profile, fact, document, working).
    """
    query_lower = str(query or "").strip().lower()
    if not query_lower:
        return "auto"
    
    # Проверяем каждый режим
    for mode, aliases in RETRIEVAL_MODE_ALIASES.items():
        if mode == "auto":
            continue
        for alias in aliases:
            if len(alias) < 3:
                continue
            if alias.lower() in query_lower:
                return mode
    
    # По умолчанию auto
    return "auto"


def classify_query_sources(query: str) -> list[str]:
    """
    Классифицировать источники для retrieval.
    
    Args:
        query: Текст запроса.
    
    Returns:
        list[str]: Список источников.
    """
    query_lower = str(query or "").strip().lower()
    if not query_lower:
        return ["all"]
    
    sources = []
    
    # Проверяем каждый источник
    for source, aliases in RETRIEVAL_SOURCE_ALIASES.items():
        if source == "all":
            continue
        for alias in aliases:
            if len(alias) < 3:
                continue
            if alias.lower() in query_lower:
                sources.append(source)
                break
    
    return sources if sources else ["all"]


def classify_query_time(query: str) -> str:
    """
    Классифицировать временной диапазон для retrieval.
    
    Args:
        query: Текст запроса.
    
    Returns:
        str: Временной диапазон (any, recent, session, historical, persistent).
    """
    query_lower = str(query or "").strip().lower()
    if not query_lower:
        return "recent"
    
    # Проверяем каждый диапазон
    for time_range, aliases in RETRIEVAL_TIME_ALIASES.items():
        for alias in aliases:
            if len(alias) < 3:
                continue
            if alias.lower() in query_lower:
                return time_range
    
    # По умолчанию recent
    return "recent"


def build_retrieval_hints(query: str) -> dict[str, Any]:
    """
    Построить подсказки для retrieval.
    
    Args:
        query: Текст запроса.
    
    Returns:
        dict: Подсказки для retrieval (mode, sources, time, topics).
    """
    return {
        "mode": classify_query_mode(query),
        "sources": classify_query_sources(query),
        "time": classify_query_time(query),
        "topics": extract_topics_from_query(query),
    }
