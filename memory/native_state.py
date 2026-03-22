"""
Memory Native State — always-on memory layer.

Это компактный, структурированный слой памяти, который всегда доступен модели
как часть её внутреннего состояния (не через retrieval).

Цель: память ощущается как часть мышления модели, а не внешний сервис.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryNativeState:
    """
    Always-on memory state — рабочая память личности.
    
    Это не retrieval, а внутреннее состояние, которое всегда с моделью.
    """
    identity_core: dict[str, Any] = field(default_factory=dict)
    active_task: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    current_decisions: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    relation_state: dict[str, Any] = field(default_factory=dict)
    active_profile_facts: dict[str, Any] = field(default_factory=dict)
    continuity: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_core": dict(self.identity_core or {}),
            "active_task": dict(self.active_task or {}),
            "open_questions": list(self.open_questions or []),
            "current_decisions": list(self.current_decisions or []),
            "recent_topics": list(self.recent_topics or []),
            "relation_state": dict(self.relation_state or {}),
            "active_profile_facts": dict(self.active_profile_facts or {}),
            "continuity": dict(self.continuity or {}),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemoryNativeState":
        d = dict(data or {})
        return cls(
            identity_core=dict(d.get("identity_core") or {}),
            active_task=dict(d.get("active_task") or {}),
            open_questions=list(d.get("open_questions") or []),
            current_decisions=list(d.get("current_decisions") or []),
            recent_topics=list(d.get("recent_topics") or []),
            relation_state=dict(d.get("relation_state") or {}),
            active_profile_facts=dict(d.get("active_profile_facts") or {}),
            continuity=dict(d.get("continuity") or {}),
        )


class MemoryNativeStateBuilder:
    """
    Строитель always-on memory state.
    
    Собирает компактный, структурированный memory state из различных источников:
    - state manager
    - identity core
    - episode planner
    - memory retrieval
    """
    
    MAX_OPEN_QUESTIONS = 5
    MAX_CURRENT_DECISIONS = 5
    MAX_RECENT_TOPICS = 5
    MAX_PROFILE_FACTS = 7
    
    def build(
        self,
        *,
        state: dict[str, Any],
        memory_context: dict[str, Any] | None = None,
        identity_core: dict[str, Any] | None = None,
    ) -> MemoryNativeState:
        """
        Построить always-on memory state.
        
        Args:
            state: Текущее состояние диалога.
            memory_context: Контекст из memory retrieval (опционально).
            identity_core: Identity core данные (опционально).
        
        Returns:
            MemoryNativeState: Компактный, структурированный memory state.
        """
        state = dict(state or {})
        memory_context = dict(memory_context or {}) if memory_context else None
        identity_core_data = dict(identity_core or {}) if identity_core else None
        
        return MemoryNativeState(
            identity_core=self._build_identity_core(state, identity_core_data),
            active_task=self._build_active_task(state, memory_context),
            open_questions=self._build_open_questions(state, memory_context),
            current_decisions=self._build_current_decisions(state, memory_context),
            recent_topics=self._build_recent_topics(state, memory_context),
            relation_state=self._build_relation_state(state),
            active_profile_facts=self._build_active_profile_facts(state, memory_context),
            continuity=self._build_continuity(state, memory_context),
        )
    
    def _build_identity_core(
        self,
        state: dict[str, Any],
        identity_core_data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Построить identity core — ключевые факты о пользователе.
        
        Включает:
        - canonical_name: как пользователь представился
        - allowed_forms: предпочтительные формы обращения
        - forbidden_forms: запрещённые формы обращения
        - boundaries: установленные границы
        - interaction_preferences: предпочтения в общении
        """
        result: dict[str, Any] = {}
        
        # Из state
        addressing = dict(state.get("addressing") or state.get("user_addressing") or {})
        if addressing:
            if addressing.get("canonical_name"):
                result["canonical_name"] = str(addressing["canonical_name"])
            if addressing.get("allowed_forms"):
                forms = addressing["allowed_forms"]
                result["allowed_forms"] = list(forms) if isinstance(forms, list) else [forms]
            if addressing.get("forbidden_forms"):
                forms = addressing["forbidden_forms"]
                result["forbidden_forms"] = list(forms) if isinstance(forms, list) else [forms]
        
        boundaries = dict(state.get("boundaries") or {})
        if boundaries:
            boundary_flags = []
            if boundaries.get("avoid_baby_tone"):
                boundary_flags.append("no_baby_talk")
            if boundaries.get("avoid_repeating_question"):
                boundary_flags.append("no_repeat_questions")
            if boundaries.get("avoid_inventing_user_facts"):
                boundary_flags.append("no_invent_facts")
            if boundary_flags:
                result["boundaries"] = boundary_flags
        
        interaction = dict(state.get("interaction_style") or {})
        if interaction:
            if interaction.get("prefers_directness"):
                result["direct_style"] = True
            if interaction.get("prefers_short_answers"):
                result["short_answers"] = True
            if interaction.get("allows_light_teasing"):
                result["allows_teasing"] = True
        
        # Из explicit identity core data (если есть)
        if identity_core_data:
            for key, value in identity_core_data.items():
                if key not in result and value is not None:
                    result[key] = value
        
        return result
    
    def _build_active_task(
        self,
        state: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Построить active task — текущую активную задачу.
        
        Это сердце continuity: модель продолжает активную линию,
        а не ищет прошлое вообще.
        """
        result: dict[str, Any] = {}
        
        # Из state
        active_task = dict(state.get("active_task") or {})
        if active_task:
            result["current_goal"] = str(active_task.get("current_goal") or active_task.get("summary_short") or "")
            result["next_step"] = str(active_task.get("next_step") or "")
            result["status"] = str(active_task.get("status") or "active")
            result["topic"] = str(active_task.get("topic") or "")
        
        # Из memory context (если есть task continuity)
        if memory_context:
            task_continuity = dict(memory_context.get("task_continuity") or {})
            if task_continuity and not result:
                active_task = dict(task_continuity.get("active_task") or {})
                if active_task:
                    result["current_goal"] = str(active_task.get("current_goal") or active_task.get("summary_short") or "")
                    result["next_step"] = str(active_task.get("next_step") or "")
                    result["status"] = str(active_task.get("status") or "active")
                    result["topic"] = str(active_task.get("topic") or "")
        
        # Из state напрямую
        if not result.get("current_goal"):
            goal = str(state.get("active_goal") or state.get("current_task") or state.get("task") or "")
            if goal:
                result["current_goal"] = goal
        
        return result
    
    def _build_open_questions(
        self,
        state: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> list[str]:
        """
        Построить open questions — незавершённые вопросы.
        
        Помогает модели помнить о том, что ещё требует ответа.
        """
        questions: list[str] = []
        
        # Из state
        open_questions = state.get("open_questions") or state.get("unresolved_questions") or []
        if isinstance(open_questions, list):
            questions = [str(q).strip() for q in open_questions if str(q).strip()]
        elif isinstance(open_questions, str):
            questions = [open_questions.strip()] if open_questions.strip() else []
        
        # Из memory context
        if memory_context:
            mq = memory_context.get("open_questions") or memory_context.get("unresolved_items") or []
            if isinstance(mq, list):
                for q in mq:
                    q_str = str(q).strip()
                    if q_str and q_str not in questions:
                        questions.append(q_str)
        
        # Ограничиваем количество
        return questions[:self.MAX_OPEN_QUESTIONS]
    
    def _build_current_decisions(
        self,
        state: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> list[str]:
        """
        Построить current decisions — принятые решения.
        
        Помогает модели помнить о договорённостях.
        """
        decisions: list[str] = []
        
        # Из state
        current_decisions = state.get("current_decisions") or state.get("active_decisions") or []
        if isinstance(current_decisions, list):
            decisions = [str(d).strip() for d in current_decisions if str(d).strip()]
        elif isinstance(current_decisions, str):
            decisions = [current_decisions.strip()] if current_decisions.strip() else []
        
        # Из memory context
        if memory_context:
            task_continuity = dict(memory_context.get("task_continuity") or {})
            active_task = dict(task_continuity.get("active_task") or {})
            cds = active_task.get("decisions") or []
            if isinstance(cds, list):
                for d in cds:
                    d_str = str(d).strip()
                    if d_str and d_str not in decisions:
                        decisions.append(d_str)
        
        # Ограничиваем количество
        return decisions[:self.MAX_CURRENT_DECISIONS]
    
    def _build_recent_topics(
        self,
        state: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> list[str]:
        """
        Построить recent topics — последние обсуждавшиеся темы.
        
        Помогает модели поддерживать контекст беседы.
        """
        topics: list[str] = []
        
        # Из state
        recent_topics = state.get("recent_topics") or []
        if isinstance(recent_topics, list):
            topics = [str(t).strip() for t in recent_topics if str(t).strip()]
        
        # Из context_tags
        context_tags = dict(state.get("context_tags") or {})
        topic = context_tags.get("topic")
        if topic and str(topic).strip() and str(topic).strip() not in topics:
            topics.insert(0, str(topic).strip())
        
        # Из memory context
        if memory_context:
            mt = memory_context.get("topics") or memory_context.get("recent_topics") or []
            if isinstance(mt, list):
                for t in mt:
                    t_str = str(t).strip()
                    if t_str and t_str not in topics:
                        topics.append(t_str)
        
        # Ограничиваем количество
        return topics[:self.MAX_RECENT_TOPICS]
    
    def _build_relation_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Построить relation state — состояние отношений с пользователем.
        
        Включает:
        - rapport_level: уровень доверия/близости
        - last_interaction_tone: тон последнего взаимодействия
        - user_mood_pattern: паттерн настроения пользователя
        """
        result: dict[str, Any] = {}
        
        # Из state
        relation_state = dict(state.get("relation_state") or {})
        if relation_state:
            if relation_state.get("rapport_level"):
                result["rapport"] = str(relation_state["rapport_level"])
            if relation_state.get("last_interaction_tone"):
                result["last_tone"] = str(relation_state["last_interaction_tone"])
            if relation_state.get("user_mood_pattern"):
                result["mood_pattern"] = str(relation_state["user_mood_pattern"])
        
        # Из user_addressing
        user_addressing = dict(state.get("user_addressing") or {})
        if user_addressing:
            if user_addressing.get("rapport_level") and not result.get("rapport"):
                result["rapport"] = str(user_addressing["rapport_level"])
            if user_addressing.get("last_interaction_tone") and not result.get("last_tone"):
                result["last_tone"] = str(user_addressing["last_interaction_tone"])
        
        return result
    
    def _build_active_profile_facts(
        self,
        state: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Построить active profile facts — главные факты из профиля.
        
        Это 3-7 самых важных фактов о пользователе/проекте.
        """
        result: dict[str, Any] = {}
        
        # Из active profile snapshot
        profile_snapshot = dict(state.get("active_profile_snapshot") or {})
        active_facts = profile_snapshot.get("active_facts") or {}
        if active_facts:
            # Берём только топ-7 фактов
            for key, value in list(active_facts.items())[:self.MAX_PROFILE_FACTS]:
                if value is not None:
                    result[key] = value
        
        # Из memory context
        if memory_context:
            facts = memory_context.get("facts") or memory_context.get("identity_facts") or []
            if isinstance(facts, list):
                for fact in facts[:self.MAX_PROFILE_FACTS]:
                    if isinstance(fact, dict):
                        key = str(fact.get("key") or fact.get("predicate") or "")
                        value = fact.get("value")
                        if key and value is not None and key not in result:
                            result[key] = value
        
        return result
    
    def _build_continuity(
        self,
        state: dict[str, Any],
        memory_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Построить continuity metadata — информацию о непрерывности.
        
        Это служебные данные для отладки и policy.
        """
        result: dict[str, Any] = {
            "active": False,
            "source": "none",
            "confidence": 0.0,
        }
        
        # Из task continuity
        task_continuity = dict(state.get("task_continuity") or {})
        if task_continuity:
            result["active"] = True
            result["source"] = str(task_continuity.get("source") or "task_continuity")
            result["confidence"] = float(task_continuity.get("confidence") or 0.8)
        
        # Из memory context
        if memory_context:
            if memory_context.get("task_continuity"):
                result["active"] = True
                result["source"] = "memory_context"
                if not result.get("confidence") or result["confidence"] < 0.8:
                    result["confidence"] = 0.85
        
        # Из active task
        active_task = dict(state.get("active_task") or {})
        if active_task and active_task.get("current_goal"):
            result["active"] = True
            if not result.get("source") or result["source"] == "none":
                result["source"] = "active_task"
        
        return result


def build_memory_native_state(
    *,
    state: dict[str, Any],
    memory_context: dict[str, Any] | None = None,
    identity_core: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Построить always-on memory state.
    
    Convenience function для быстрого использования.
    
    Args:
        state: Текущее состояние диалога.
        memory_context: Контекст из memory retrieval (опционально).
        identity_core: Identity core данные (опционально).
    
    Returns:
        dict: Компактный, структурированный memory state.
    """
    builder = MemoryNativeStateBuilder()
    native_state = builder.build(
        state=state,
        memory_context=memory_context,
        identity_core=identity_core,
    )
    return native_state.to_dict()
