from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config.settings import load_config


_cfg = load_config()

MemoryStorageDir = _cfg.memory_dir
MMIS_MEMORY_DB_PATH = _cfg.db_path
MODEL_NAME = _cfg.model_name
SHORT_MEMORY_LIMIT = _cfg.short_memory_limit

MMIS_VOICE_TTS_VOICE = _cfg.voice_tts_voice
MMIS_VOICE_TTS_RATE = _cfg.voice_tts_rate
MMIS_VOICE_TTS_VOLUME = _cfg.voice_tts_volume
MMIS_VOICE_INPUT_DIR = _cfg.voice_input_dir
MMIS_VOICE_OUTPUT_DIR = _cfg.voice_output_dir

MMIS_CHAT_RECALL_RESULTS = _cfg.chat_recall_results
MMIS_CHAT_EVENTS_LIMIT = _cfg.chat_events_limit
MMIS_CHAT_PROOFREAD = _cfg.chat_proofread
MMIS_CHAT_PROOFREAD_STRICT = _cfg.chat_proofread_strict
MODEL_FALLBACKS = _cfg.model_fallbacks


def build_ollama_options(task_type: str) -> dict:
    _ = task_type
    return {"temperature": 0.7, "num_predict": 768}


# --- Model Profiles Merged from model_profiles.py ---

@dataclass(frozen=True)
class GenerationProfile:
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class OllamaProfile:
    num_thread: int = 6
    num_ctx: int = 8192
    num_gpu: int = 1
    num_batch: int = 128
    keep_alive: str = "5m"


@dataclass(frozen=True)
class OpenAIProfile:
    model: str = ""
    reasoning_effort: str = "medium"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    generation: GenerationProfile
    ollama: OllamaProfile
    openai: OpenAIProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generation": asdict(self.generation),
            "ollama": asdict(self.ollama),
            "openai": asdict(self.openai),
        }


PROFILES: dict[str, ModelProfile] = {
    "FAST": ModelProfile(
        name="FAST",
        generation=GenerationProfile(temperature=0.55, top_p=0.9, repeat_penalty=1.05, max_tokens=512),
        ollama=OllamaProfile(num_thread=8, num_ctx=4096, num_gpu=1, num_batch=64, keep_alive="2m"),
        openai=OpenAIProfile(reasoning_effort="low"),
    ),
    "BALANCED": ModelProfile(
        name="BALANCED",
        generation=GenerationProfile(temperature=0.7, top_p=0.92, repeat_penalty=1.1, max_tokens=1024),
        ollama=OllamaProfile(num_thread=6, num_ctx=8192, num_gpu=1, num_batch=128, keep_alive="5m"),
        openai=OpenAIProfile(reasoning_effort="medium"),
    ),
    "QUALITY": ModelProfile(
        name="QUALITY",
        generation=GenerationProfile(temperature=0.82, top_p=0.95, repeat_penalty=1.2, max_tokens=2048),
        ollama=OllamaProfile(num_thread=6, num_ctx=12288, num_gpu=1, num_batch=160, keep_alive="10m"),
        openai=OpenAIProfile(reasoning_effort="high"),
    ),
    "ECONOM": ModelProfile(
        name="ECONOM",
        generation=GenerationProfile(temperature=0.45, top_p=0.88, repeat_penalty=1.12, max_tokens=384),
        ollama=OllamaProfile(num_thread=4, num_ctx=3072, num_gpu=0, num_batch=48, keep_alive="1m"),
        openai=OpenAIProfile(reasoning_effort="low"),
    ),
    "ASYA": ModelProfile(
        name="ASYA",
        generation=GenerationProfile(temperature=0.7, top_p=0.9, repeat_penalty=1.2, max_tokens=-1),
        ollama=OllamaProfile(num_thread=6, num_ctx=8192, num_gpu=20, num_batch=128, keep_alive="5m"),
        openai=OpenAIProfile(reasoning_effort="medium"),
    ),
}


def get_profile(name: str | None) -> ModelProfile:
    key = str(name or "BALANCED").strip().upper()
    if key not in PROFILES:
        key = "BALANCED"
    profile = PROFILES[key]
    adjusted = _apply_hardware_guards(profile)
    return adjusted


def merge_profile(profile: ModelProfile | str, overrides: dict[str, Any] | None = None) -> ModelProfile:
    base = get_profile(profile if isinstance(profile, str) else profile.name)
    if isinstance(profile, ModelProfile):
        base = profile
    override_map = dict(overrides or {})
    if not override_map:
        return base

    payload = base.to_dict()
    _deep_merge(payload, override_map)
    generation = GenerationProfile(**payload.get("generation", {}))
    ollama = OllamaProfile(**payload.get("ollama", {}))
    openai = OpenAIProfile(**payload.get("openai", {}))
    name = str(payload.get("name") or base.name).upper()
    merged = ModelProfile(name=name, generation=generation, ollama=ollama, openai=openai)
    return _apply_hardware_guards(merged)


def _apply_hardware_guards(profile: ModelProfile) -> ModelProfile:
    from config.settings import load_config
    
    app_settings = load_config()
    vram = app_settings.gpu_vram_gb
    if vram is None:
        return profile
    ollama = profile.ollama
    if vram <= 4:
        ollama = OllamaProfile(
            num_thread=min(ollama.num_thread, 6),
            num_ctx=min(ollama.num_ctx, 4096),
            num_gpu=min(ollama.num_gpu, 1),
            num_batch=min(ollama.num_batch, 64),
            keep_alive=ollama.keep_alive,
        )
    elif vram <= 6:
        ollama = OllamaProfile(
            num_thread=min(ollama.num_thread, 8),
            num_ctx=min(ollama.num_ctx, 6144),
            num_gpu=min(ollama.num_gpu, 1),
            num_batch=min(ollama.num_batch, 96),
            keep_alive=ollama.keep_alive,
        )
    return ModelProfile(name=profile.name, generation=profile.generation, ollama=ollama, openai=profile.openai)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if key not in base:
            base[key] = copy.deepcopy(value)
            continue
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_merge(current, value)
            continue
        base[key] = copy.deepcopy(value)


# Backward compatibility with early config key names.
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "temperature": PROFILES["BALANCED"].generation.temperature,
        "description": "Balanced quality/speed profile.",
    },
    "metadata": {
        "temperature": 0.1,
        "description": "Deterministic extraction profile.",
    },
}
