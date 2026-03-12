from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from modules.character.composer import CharacterComposer
from modules.character.evaluator import RuleEvaluator
from modules.character.storage import CharacterStorage
from modules.character.trait_limits import normalize_trait_value

if TYPE_CHECKING:
    from core.character_runtime import CharacterRuntime, CharacterRuntimeResult
    CharacterUpdateResult = CharacterRuntimeResult
else:
    CharacterUpdateResult = Any


class CharacterEngine:
    """
    Thin compatibility wrapper over CharacterRuntime.

    Runtime owns trait loading, rule application, mood updates, learned deltas,
    and prompt compilation. Engine keeps the old API surface for callers that
    still import modules.character.engine.
    """

    runtime: "CharacterRuntime"

    def __init__(
        self,
        *,
        storage: CharacterStorage | None = None,
        evaluator: RuleEvaluator | None = None,
        composer: CharacterComposer | None = None,
    ):
        from core.character_runtime import CharacterRuntime

        resolved_storage = storage or CharacterStorage()
        state_path = (Path(resolved_storage.root) / "_character_engine_state.json").resolve()
        self.runtime = CharacterRuntime(
            storage=resolved_storage,
            evaluator=evaluator,
            composer=composer,
            state_path=state_path,
            autosave=False,
        )
        self.storage = self.runtime.storage
        self.evaluator = self.runtime.evaluator
        self.composer = self.runtime.composer

    def list_ids(self) -> list[str]:
        return self.runtime.list_ids()

    def get_manifest(self) -> dict[str, Any]:
        return self.runtime.get_manifest()

    def get_active_character_id(self, state: dict[str, Any] | None = None) -> str:
        return self.runtime.get_active_character_id(state)

    def set_active_character(self, character_id: str) -> str:
        return self.runtime.set_active_character(character_id)

    def list_traits(self, character_id: str) -> dict[str, Any]:
        return self.runtime.list_traits(character_id)

    def set_trait(
        self,
        character_id: str,
        trait_name: str,
        *,
        value: Any,
        confidence: float = 0.8,
        trait_type: str | None = None,
    ) -> dict[str, Any]:
        normalized_value = value
        normalized_type = str(trait_type or "").strip().lower()
        if normalized_type != "flag" and isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized_value = normalize_trait_value(trait_name, value)
        return self.runtime.set_trait(
            character_id,
            trait_name,
            value=normalized_value,
            confidence=confidence,
            trait_type=trait_type,
        )

    def remove_trait(self, character_id: str, trait_name: str) -> bool:
        return self.runtime.remove_trait(character_id, trait_name)

    def build_prompt(self, character_id: str) -> str:
        return self.runtime.build_personality_block(character_id)

    def update(
        self,
        *,
        text: str,
        meta: dict[str, Any] | None,
        active_character_id: str | None = None,
    ) -> CharacterUpdateResult:
        return self.runtime.evolve(
            text=text,
            meta=meta,
            active_character_id=active_character_id,
        )
