from __future__ import annotations

from dataclasses import dataclass

from core.spec_registry import SpecRegistry, get_spec_registry
from prompt_engine.prompt_loader import PromptDocument


@dataclass(frozen=True)
class PromptEntry:
    key: str
    rel_path: str
    description: str = ""


class PromptRegistry:
    """Maps logical prompt keys to JSON specs and returns PromptDocument objects."""

    DEFAULTS: dict[str, PromptEntry] = {
        "system.base": PromptEntry("system.base", "system_spec.json#prompts.system.base", "Base system rules"),
        "memory.fact_extraction": PromptEntry("memory.fact_extraction", "memory_spec.json#prompts.memory.fact_extraction"),
        "memory.cleanup": PromptEntry("memory.cleanup", "memory_spec.json#prompts.memory.cleanup"),
        "memory.merge": PromptEntry("memory.merge", "memory_spec.json#prompts.memory.merge"),
        "metadata.emotion": PromptEntry("metadata.emotion", "metadata_spec.json#prompts.metadata.emotion"),
        "metadata.intent": PromptEntry("metadata.intent", "metadata_spec.json#prompts.metadata.intent"),
        "metadata.tagging": PromptEntry("metadata.tagging", "metadata_spec.json#prompts.metadata.tagging"),
        "response.safety_filter": PromptEntry("response.safety_filter", "system_spec.json#prompts.response.safety_filter"),
        "response.formatting": PromptEntry("response.formatting", "system_spec.json#prompts.response.formatting"),
    }

    def __init__(self, loader=None, spec_registry: SpecRegistry | None = None):
        _ = loader  # compatibility with previous constructor
        self._spec_registry = spec_registry or get_spec_registry()
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
        _ = (use_cache, hot_reload)  # kept for API compatibility
        entry = self.get_entry(key)
        return self._load_document(entry.key, rel_hint=entry.rel_path)

    def get_text(self, key: str, *, use_cache: bool = True) -> str:
        _ = use_cache
        return self.get_prompt(key).text

    def get_by_path(self, rel_path: str, *, use_cache: bool = True, hot_reload: bool = True) -> PromptDocument:
        _ = (use_cache, hot_reload)
        ref = str(rel_path or "").replace("\\", "/").strip().strip("/")
        if not ref:
            raise ValueError("rel_path is empty")
        key = ""
        # Allow looking up by exact rel_path from current registry entries.
        for item in self._entries.values():
            if item.rel_path == ref:
                key = item.key
                break
        if not key:
            raise KeyError(f"unknown prompt path: {ref}")
        return self._load_document(key, rel_hint=ref)

    def get_text_by_path(self, rel_path: str, *, use_cache: bool = True) -> str:
        _ = use_cache
        return self.get_by_path(rel_path).text

    def keys(self) -> list[str]:
        return sorted(self._entries.keys())

    def _load_document(self, key: str, *, rel_hint: str) -> PromptDocument:
        payload = self._spec_registry.load_prompt_document(key)
        return PromptDocument(
            rel_path=str(rel_hint or payload.get("rel_path") or key),
            source_path=str(payload.get("source_path") or ""),
            text=str(payload.get("text") or "").strip(),
            id=str(payload.get("id") or key),
            version=str(payload.get("version") or "0.0.0"),
            role=str(payload.get("role") or "system"),
            tags=[str(x).strip().lower() for x in list(payload.get("tags") or []) if str(x).strip()],
            min_ctx=max(0, int(payload.get("min_ctx") or 0)),
            meta={"spec_name": str(payload.get("spec_name") or "")},
        )
