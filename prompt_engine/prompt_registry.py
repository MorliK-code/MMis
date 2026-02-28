from __future__ import annotations

from dataclasses import dataclass

from prompt_engine.prompt_loader import PromptDocument, PromptLoader


@dataclass(frozen=True)
class PromptEntry:
    key: str
    rel_path: str
    description: str = ""


class PromptRegistry:
    """Maps logical keys to prompt files and returns parsed prompt documents."""

    DEFAULTS: dict[str, PromptEntry] = {
        "system.base": PromptEntry("system.base", "system/base_system.txt", "Base system rules"),
        "system.persona.default": PromptEntry("system.persona.default", "legacy/system/personality_default.txt"),
        "system.persona.flirty": PromptEntry("system.persona.flirty", "legacy/system/personality_flirty.txt"),
        "system.persona.strict": PromptEntry("system.persona.strict", "legacy/system/personality_strict.txt"),
        "system.style.default": PromptEntry("system.style.default", "legacy/system/style_default.txt"),
        "system.style.flirty": PromptEntry("system.style.flirty", "legacy/system/style_flirty.txt"),
        "system.style.strict": PromptEntry("system.style.strict", "legacy/system/style_strict.txt"),
        "system.style.supportive": PromptEntry("system.style.supportive", "legacy/system/style_supportive.txt"),
        "system.rules.default": PromptEntry("system.rules.default", "legacy/system/rules_default.txt"),
        "memory.fact_extraction": PromptEntry("memory.fact_extraction", "memory/fact_extraction.txt"),
        "memory.cleanup": PromptEntry("memory.cleanup", "memory/memory_cleanup.txt"),
        "memory.merge": PromptEntry("memory.merge", "memory/memory_merge.txt"),
        "metadata.emotion": PromptEntry("metadata.emotion", "metadata/emotion_detection.txt"),
        "metadata.intent": PromptEntry("metadata.intent", "metadata/intent_classification.txt"),
        "metadata.tagging": PromptEntry("metadata.tagging", "metadata/tagging.txt"),
        "automation.browser_action": PromptEntry("automation.browser_action", "automation/browser_action.txt"),
        "automation.os_action": PromptEntry("automation.os_action", "automation/os_action.txt"),
        "response.safety_filter": PromptEntry("response.safety_filter", "response/safety_filter.txt"),
        "response.formatting": PromptEntry("response.formatting", "response/formatting.txt"),
    }

    def __init__(self, loader: PromptLoader | None = None):
        self.loader = loader or PromptLoader()
        self._entries: dict[str, PromptEntry] = dict(self.DEFAULTS)

    def register(self, key: str, rel_path: str, description: str = "") -> None:
        item = PromptEntry(key=str(key), rel_path=str(rel_path), description=str(description or ""))
        self._entries[item.key] = item

    def get_entry(self, key: str) -> PromptEntry:
        name = str(key or "").strip()
        if not name:
            raise ValueError("prompt key is empty")
        if name not in self._entries:
            raise KeyError(f"unknown prompt key: {name}")
        return self._entries[name]

    def get_prompt(self, key: str, *, use_cache: bool = True, hot_reload: bool = True) -> PromptDocument:
        entry = self.get_entry(key)
        return self.loader.load_document(entry.rel_path, use_cache=use_cache, hot_reload=hot_reload)

    def get_text(self, key: str, *, use_cache: bool = True) -> str:
        return self.get_prompt(key, use_cache=use_cache).text

    def get_by_path(self, rel_path: str, *, use_cache: bool = True, hot_reload: bool = True) -> PromptDocument:
        return self.loader.load_document(rel_path, use_cache=use_cache, hot_reload=hot_reload)

    def get_text_by_path(self, rel_path: str, *, use_cache: bool = True) -> str:
        return self.get_by_path(rel_path, use_cache=use_cache).text

    def keys(self) -> list[str]:
        return sorted(self._entries.keys())
