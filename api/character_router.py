"""
Character API - endpoints для управления персонажами MMis.
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from core.character_runtime import CharacterRuntime
from utils.logger import get_logger


LOGGER = get_logger(__name__)


# === Request/Response Models ===

class CharacterActiveSetRequest(BaseModel):
    """Запрос на смену активного персонажа."""
    id: str = Field(..., description="ID персонажа")


class CharacterCreateRequest(BaseModel):
    """Запрос на создание нового персонажа."""
    id: str = Field(..., description="Уникальный ID персонажа")
    name: str = Field(..., description="Отображаемое имя")
    llm_profile: str = Field(default="BALANCED", description="Профиль производительности LLM")
    default_mood: str = Field(default="thoughtful", description="Настроение по умолчанию")


class CharacterUpdateRequest(BaseModel):
    """Запрос на обновление существующего персонажа."""
    name: str | None = None
    llm_profile: str | None = None
    performance: str | None = None
    default_mood: str | None = None

    character: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    persona_state: dict[str, Any] | None = None
    persona_spec: dict[str, Any] | None = None
    emotion_state: dict[str, Any] | None = None
    user_addressing: dict[str, Any] | None = None
    identity_core: dict[str, Any] | None = None
    builtin_traits: dict[str, Any] | None = None
    learned_traits: dict[str, Any] | None = None
    rules: dict[str, Any] | None = None


def create_character_router(runtime: CharacterRuntime) -> APIRouter:
    """
    Создаёт API router для управления персонажами.
    """
    router = APIRouter(prefix="/characters", tags=["characters"])

    def _full_character_payload(character_id: str) -> dict[str, Any]:
        cid = str(character_id or "").strip().lower()
        runtime.storage.ensure_character_structure(cid)

        character = runtime.storage.load_character(cid)
        state = runtime.storage.load_state(cid)
        persona_state = runtime.storage.load_persona_state(cid)
        persona_spec = runtime.storage.load_persona_spec(cid)
        emotion_state = runtime.storage.load_emotion_state(cid)
        user_addressing = runtime.storage.load_user_addressing(cid)
        identity_core = runtime.storage.load_identity_core(cid)
        builtin_traits = runtime.storage.load_builtin_traits(cid)
        learned_traits = runtime.storage.load_learned_traits(cid)
        rules = runtime.storage.load_rules(cid)

        return {
            "id": cid,
            "character": character,
            "state": state,
            "persona_state": persona_state,
            "persona_spec": persona_spec,
            "emotion_state": emotion_state,
            "user_addressing": user_addressing,
            "identity_core": identity_core,
            "builtin_traits": builtin_traits,
            "learned_traits": learned_traits,
            "rules": rules,
        }

    @router.get("")
    async def list_characters():
        """Список всех доступных персонажей."""
        try:
            return runtime.get_manifest()
        except Exception as e:
            LOGGER.exception("Failed to list characters")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/active")
    async def get_active_character():
        """Получить информацию об активном персонаже."""
        try:
            active_id = runtime.get_active_character_id()
            meta = runtime.get_meta(active_id)
            return {
                "id": active_id,
                "meta": meta
            }
        except Exception as e:
            LOGGER.exception("Failed to get active character")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/active")
    async def set_active_character(req: CharacterActiveSetRequest):
        """Сменить активного персонажа."""
        try:
            runtime.set_active_character(req.id)
            return {"status": "ok", "active_id": req.id}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            LOGGER.exception("Failed to set active character")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/{character_id}")
    async def get_character(character_id: str):
        """Получить полный payload персонажа."""
        try:
            return _full_character_payload(character_id)
        except Exception as e:
            LOGGER.exception("Failed to get character")
            raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found: {e}")

    @router.post("")
    async def create_character(req: CharacterCreateRequest):
        """Создать нового персонажа."""
        try:
            cid = runtime.create_character(
                req.id, 
                req.name, 
                llm_profile=req.llm_profile, 
                default_mood=req.default_mood
            )
            return {"status": "ok", "id": cid}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            LOGGER.exception("Failed to create character")
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/{character_id}")
    async def update_character(character_id: str, req: CharacterUpdateRequest):
        """Обновить полный payload персонажа."""
        try:
            cid = str(character_id or "").strip().lower()
            runtime.storage.ensure_character_structure(cid)

            data = req.model_dump(exclude_none=True)

            # 1. Backward-compatible простые поля.
            simple_updates = {}
            for key in ("name", "llm_profile", "performance", "default_mood"):
                if key in data:
                    simple_updates[key] = data[key]

            if simple_updates:
                runtime.update_character(cid, simple_updates)

            # 2. Полные секции.
            if isinstance(data.get("character"), dict):
                runtime.storage.update_character(cid, data["character"])

            if isinstance(data.get("state"), dict):
                runtime.storage.save_state(cid, data["state"])

            if isinstance(data.get("persona_state"), dict):
                runtime.storage.save_persona_state(cid, data["persona_state"])

            if isinstance(data.get("emotion_state"), dict):
                runtime.storage.save_emotion_state(cid, data["emotion_state"])

            if isinstance(data.get("user_addressing"), dict):
                runtime.storage.save_user_addressing(cid, data["user_addressing"])

            if isinstance(data.get("identity_core"), dict):
                runtime.storage.save_identity_core(cid, data["identity_core"])

            if isinstance(data.get("builtin_traits"), dict):
                runtime.storage.save_builtin_traits(cid, data["builtin_traits"])

            if isinstance(data.get("learned_traits"), dict):
                runtime.storage.save_learned_traits(cid, data["learned_traits"])

            # persona_spec и rules — spec-файлы, поэтому пишем напрямую.
            if isinstance(data.get("persona_spec"), dict):
                from pathlib import Path
                import json
                path = runtime.storage.persona_spec_path(cid)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data["persona_spec"], ensure_ascii=False, indent=2), encoding="utf-8")

            if isinstance(data.get("rules"), dict):
                from pathlib import Path
                import json
                path = runtime.storage.character_spec_dir(cid) / "evolution_spec.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data["rules"], ensure_ascii=False, indent=2), encoding="utf-8")

            # 3. Сброс runtime caches.
            for cache_name in ("_meta_cache", "_meta_cache_revision", "_profiles_cache", "_profiles_cache_revision"):
                cache = getattr(runtime, cache_name, None)
                if isinstance(cache, dict):
                    cache.pop(cid, None)

            return {"status": "ok", "character": _full_character_payload(cid)}
        except Exception as e:
            LOGGER.exception("Failed to update character")
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/{character_id}")
    async def delete_character(character_id: str):
        """Удалить персонажа."""
        try:
            runtime.delete_character(character_id)
            return {"status": "ok"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            LOGGER.exception("Failed to delete character")
            raise HTTPException(status_code=500, detail=str(e))

    return router
