from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.personality_engine import PersonalityEngine, ensure_default_personality_files, ensure_default_prompt_files
from llm.provider_base import Message
from modules.character.engine import CharacterEngine
from prompt_engine.prompt_registry import PromptRegistry
from prompt_engine.token_budget_manager import ContextBlock, TokenBudgetManager


@dataclass(frozen=True)
class PromptEngineResult:
    messages: list[Message]
    sections: dict[str, str]


class PromptEngine:
    """Compose final LLM messages from prompt templates, personality, state, and token budgets."""

    def __init__(
        self,
        registry: PromptRegistry | None = None,
        budget_manager: TokenBudgetManager | None = None,
        personality_engine: PersonalityEngine | None = None,
        character_engine: CharacterEngine | None = None,
    ):
        ensure_default_prompt_files()
        ensure_default_personality_files()
        self.registry = registry or PromptRegistry()
        self.budget_manager = budget_manager or TokenBudgetManager()
        self.personality_engine = personality_engine or PersonalityEngine()
        self.character_engine = character_engine or CharacterEngine()

    def compose(
        self,
        *,
        prompt_pack,
        state: dict[str, Any] | None = None,
        traits: dict[str, Any] | None = None,
        policies: dict[str, Any] | None = None,
    ) -> PromptEngineResult:
        state_map = dict(state or {})
        traits_map = dict(traits or {})
        policies_map = dict(policies or {})
        blocks = dict(getattr(prompt_pack, "blocks", {}) or {})

        active_personality = _pick(
            state_map.get("active_personality_id"),
            traits_map.get("personality"),
            traits_map.get("persona"),
            policies_map.get("personality"),
            _resolve_persona_name(traits_map=traits_map, state_map=state_map, policies_map=policies_map),
            "default",
        ).lower()
        active_character = _pick(
            state_map.get("active_character_id"),
            traits_map.get("character"),
            policies_map.get("character"),
            active_personality,
        ).lower()
        personality = self.personality_engine.get_profile(active_personality)

        base_doc = self._safe_doc(key="system.base", fallback_text=blocks.get("system_role") or "")
        persona_doc = self._safe_doc(path=personality.system_prompt, fallback_text=blocks.get("persona") or "")
        style_doc = self._safe_doc(path=personality.style_prompt, fallback_text="")
        rules_doc = self._safe_doc(path=personality.rules_prompt, fallback_text="")
        safety_doc = self._safe_doc(key="response.safety_filter", fallback_text="")
        formatting_doc = self._safe_doc(key="response.formatting", fallback_text="")
        character_prompt_block = str(state_map.get("character_prompt_block") or "").strip()
        if not character_prompt_block and active_character:
            try:
                character_prompt_block = str(self.character_engine.build_prompt(active_character) or "").strip()
            except Exception:
                character_prompt_block = ""
        personality_core_text = character_prompt_block or _join_non_empty([persona_doc["text"], style_doc["text"]])

        blend = _as_dict(state_map.get("personality_blend"))
        blend_old_block = self._build_blend_block(blend)

        user_profile_block = self._build_user_profile_block(state_map)
        metadata_block = self._build_metadata_block(state_map=state_map, blocks=blocks)
        tools_state_block = self._build_tools_state_block(state_map)

        context_blocks = [
            ContextBlock(
                id="system_core",
                content=base_doc["text"],
                bucket="system",
                priority=100,
                required=True,
                max_tokens=self.budget_manager.budget.system_core,
            ),
            ContextBlock(
                id="personality_core",
                content=personality_core_text,
                bucket="personality",
                priority=92,
                required=True,
                max_tokens=self.budget_manager.budget.personality,
            ),
            ContextBlock(
                id="personality_blend",
                content=blend_old_block,
                bucket="personality",
                priority=58,
                required=False,
                shrink_strategy="summarize",
                max_tokens=max(64, self.budget_manager.budget.personality // 2),
            ),
            ContextBlock(
                id="rules",
                content=_join_non_empty([rules_doc["text"], safety_doc["text"], formatting_doc["text"]]),
                bucket="rules",
                priority=96,
                required=True,
                max_tokens=self.budget_manager.budget.rules,
            ),
            ContextBlock(
                id="user_profile",
                content=user_profile_block,
                bucket="user_profile",
                priority=74,
                shrink_strategy="summarize",
                max_tokens=self.budget_manager.budget.user_profile,
            ),
            ContextBlock(
                id="metadata",
                content=metadata_block,
                bucket="metadata",
                priority=80,
                max_tokens=self.budget_manager.budget.metadata,
            ),
            ContextBlock(
                id="tools_state",
                content=tools_state_block,
                bucket="tools",
                priority=66,
                shrink_strategy="truncate_head",
                max_tokens=self.budget_manager.budget.tools_state,
            ),
            ContextBlock(
                id="memory_retrieval",
                content=str(blocks.get("retrieved_memories") or ""),
                bucket="memory",
                priority=68,
                shrink_strategy="drop",
                max_tokens=self.budget_manager.budget.memory_retrieval,
            ),
            ContextBlock(
                id="recent_chat",
                content=str(blocks.get("conversation_tail") or ""),
                bucket="history",
                priority=72,
                shrink_strategy="summarize",
                max_tokens=self.budget_manager.budget.recent_chat,
            ),
            ContextBlock(
                id="long_summary",
                content=str(blocks.get("long_summary") or state_map.get("dialog_summary") or ""),
                bucket="long_summary",
                priority=70,
                shrink_strategy="summarize",
                max_tokens=self.budget_manager.budget.long_summary,
            ),
            ContextBlock(
                id="output_schema",
                content=str(blocks.get("output_schema") or ""),
                bucket="output",
                priority=88,
                max_tokens=self.budget_manager.budget.output,
            ),
            ContextBlock(
                id="user",
                content=str(blocks.get("user_message") or ""),
                bucket="user",
                priority=100,
                required=True,
                max_tokens=self.budget_manager.budget.user,
                min_tokens=32,
            ),
        ]

        fitted, budget_stats = self.budget_manager.fit_context_blocks(context_blocks)
        system_content = _join_sections(
            [
                ("SYSTEM_CORE", fitted.get("system_core", "")),
                ("PERSONALITY", fitted.get("personality_core", "")),
                ("PERSONALITY_BLEND", fitted.get("personality_blend", "")),
                ("RULES", fitted.get("rules", "")),
                ("USER_PROFILE", fitted.get("user_profile", "")),
                ("METADATA", fitted.get("metadata", "")),
                ("TOOLS_STATE", fitted.get("tools_state", "")),
                ("MEMORY", fitted.get("memory_retrieval", "")),
                ("RECENT_CHAT", fitted.get("recent_chat", "")),
                ("LONG_SUMMARY", fitted.get("long_summary", "")),
                ("OUTPUT_SCHEMA", fitted.get("output_schema", "")),
            ]
        )
        user_content = str(fitted.get("user") or "").strip()

        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]

        sections = {
            "active_personality_id": personality.id,
            "active_personality_name": personality.name,
            "active_personality_version": personality.version,
            "active_character_id": active_character,
            "base_prompt_id": base_doc["id"],
            "base_prompt_version": base_doc["version"],
            "persona_prompt_id": persona_doc["id"],
            "persona_prompt_version": persona_doc["version"],
            "style_prompt_id": style_doc["id"],
            "style_prompt_version": style_doc["version"],
            "rules_prompt_id": rules_doc["id"],
            "rules_prompt_version": rules_doc["version"],
            "system": system_content,
            "user": user_content,
        }
        sections["budget_total_tokens"] = str(budget_stats.get("total_tokens") or 0)
        sections["budget_dropped"] = ",".join([str(x) for x in list(budget_stats.get("dropped") or [])])
        sections["budget_trimmed"] = ",".join([str(x) for x in list(budget_stats.get("trimmed") or [])])

        return PromptEngineResult(messages=messages, sections=sections)

    def _safe_doc(self, *, key: str | None = None, path: str | None = None, fallback_text: str = "") -> dict[str, str]:
        if key:
            try:
                doc = self.registry.get_prompt(key)
                return {
                    "id": str(doc.id or key),
                    "version": str(doc.version or "0.0.0"),
                    "text": str(doc.text or "").strip(),
                }
            except Exception:
                pass
        if path:
            rel = str(path or "").replace("\\", "/").strip().strip("/")
            if rel:
                try:
                    doc = self.registry.get_by_path(rel)
                    return {
                        "id": str(doc.id or rel),
                        "version": str(doc.version or "0.0.0"),
                        "text": str(doc.text or "").strip(),
                    }
                except Exception:
                    pass
        return {
            "id": str(key or path or "fallback"),
            "version": "0.0.0",
            "text": str(fallback_text or "").strip(),
        }

    def _build_blend_block(self, blend: dict[str, Any]) -> str:
        row = dict(blend or {})
        if not bool(row.get("active")):
            return ""
        old_id = str(row.get("from") or "").strip().lower()
        new_id = str(row.get("to") or "").strip().lower()
        old_weight = float(row.get("old_weight") or 0.0)
        new_weight = float(row.get("new_weight") or 1.0)
        if not old_id or not new_id or old_id == new_id:
            return ""

        old_profile = self.personality_engine.get_profile(old_id)
        old_style = self._safe_doc(path=old_profile.style_prompt, fallback_text="")
        old_persona = self._safe_doc(path=old_profile.system_prompt, fallback_text="")
        old_text = _join_non_empty([old_persona["text"], old_style["text"]])
        old_short = _first_sentences(old_text, max_lines=4)
        return (
            "Temporary blend mode is active. "
            f"Use old persona '{old_id}' weight={old_weight:.2f} and new persona '{new_id}' weight={new_weight:.2f}.\n"
            f"Old persona short summary:\n{old_short}"
        ).strip()

    @staticmethod
    def _build_user_profile_block(state_map: dict[str, Any]) -> str:
        profile_summary = _as_dict(state_map.get("profile_summary"))
        user_profile = profile_summary.get("user")
        if isinstance(user_profile, str) and user_profile.strip():
            return user_profile.strip()
        if isinstance(user_profile, dict):
            lines = []
            for key, value in user_profile.items():
                if value is None:
                    continue
                lines.append(f"- {key}: {value}")
            if lines:
                return "\n".join(lines)
        return ""

    @staticmethod
    def _build_metadata_block(*, state_map: dict[str, Any], blocks: dict[str, str]) -> str:
        tags_block = str(blocks.get("context_tags") or "").strip()
        context = _as_dict(state_map.get("context_tags"))
        lines = []
        if tags_block:
            lines.append(tags_block)
        for key in ("lang", "intent", "mood", "topic"):
            value = str(context.get(key) or "").strip()
            if value:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines).strip()

    @staticmethod
    def _build_tools_state_block(state_map: dict[str, Any]) -> str:
        value = state_map.get("last_tool_result")
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            import json

            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)


def _resolve_persona_name(
    *,
    traits_map: dict[str, Any],
    state_map: dict[str, Any],
    policies_map: dict[str, Any],
) -> str:
    values = [
        traits_map.get("persona"),
        traits_map.get("personality"),
        state_map.get("active_personality_id"),
        state_map.get("persona"),
        state_map.get("personality"),
        policies_map.get("persona"),
        policies_map.get("personality"),
        traits_map.get("profile"),
        state_map.get("profile"),
        policies_map.get("profile"),
    ]
    raw = ""
    for value in values:
        text = str(value or "").strip().lower()
        if text:
            raw = text
            break

    if any(x in raw for x in ("flirty", "playful", "romantic")):
        return "flirty"
    if any(x in raw for x in ("strict", "formal")):
        return "strict"
    if "support" in raw:
        return "supportive"
    return "default"


def _join_sections(parts: list[tuple[str, str]]) -> str:
    out: list[str] = []
    for name, text in parts:
        block = str(text or "").strip()
        if not block or block == "- none":
            continue
        out.append(f"<<<{name}>>>\n{block}\n<<<END_{name}>>>")
    return "\n\n".join(out).strip()


def _as_dict(value) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _pick(*values) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _join_non_empty(items: list[str]) -> str:
    rows = [str(x or "").strip() for x in list(items or []) if str(x or "").strip()]
    return "\n\n".join(rows).strip()


def _first_sentences(text: str, *, max_lines: int = 4) -> str:
    lines = [x.strip() for x in str(text or "").splitlines() if x.strip()]
    if not lines:
        return ""
    return "\n".join(lines[:max_lines]).strip()
