from __future__ import annotations

import json

from core.character_runtime import CharacterRuntime
from core.response_pipeline import PROFILE_BALANCED, PipelineContext, PromptBuildStage
from memory_core.processors.memory_llm_processor import MemoryLLMProcessor
from memory_core.retrieval.context_builder import ContextBuilder
from memory_core.retrieval.prompt_adapter import distill_memory_artifacts
from memory_core.schemas import MemoryArtifact, MemoryQuery


def test_emotional_memory_not_spoken_literally() -> None:
    artifact = MemoryArtifact(
        artifact_id="emo-1",
        artifact_type="emotional_state",
        text="пользователь чувствует грусть",
        summary="грусть пользователя",
        metadata={
            "emotion": "sad",
            "prompt_view": "user currently low-bandwidth; keep tone gentle",
            "exposure_mode": "prompt_safe",
        },
    )
    context_pack, _ = ContextBuilder().build([artifact], MemoryQuery(text="да не знаю уже"))

    assert "пользователь чувствует грусть" not in "\n".join(context_pack.to_context_blocks())
    assert context_pack.relevant_facts == []
    assert context_pack.tone_hints == ["user currently low-bandwidth; keep tone gentle"]
    assert bool(dict(context_pack.recent_user_state).get("low_bandwidth")) is True


def test_latent_memory_hidden_from_prompt_block() -> None:
    runtime = CharacterRuntime(autosave=False)
    pack = runtime.build_prompt(
        user_msg="да не знаю уже",
        retrieved_memories=[
            {
                "artifact_id": "emo-hidden",
                "artifact_type": "emotional_state",
                "text": "пользователь чувствует грусть",
                "prompt_view": "keep answer shorter and steadier",
                "exposure_mode": "latent",
                "confidence": 0.91,
                "relevant": True,
            }
        ],
        traits={},
        policies={},
    )

    block = str(dict(pack.blocks or {}).get("retrieved_memories") or "")
    assert "пользователь чувствует грусть" not in block
    assert "keep answer shorter and steadier" not in block


def test_preference_memory_not_rendered_as_dossier() -> None:
    runtime = CharacterRuntime(autosave=False)
    pack = runtime.build_prompt(
        user_msg="покажи на моём коде",
        retrieved_memories=[
            {
                "artifact_id": "pref-1",
                "artifact_type": "preference",
                "text": "user prefers examples grounded in own code",
                "summary": "prefers own-code examples",
                "prompt_view": "Prefer examples grounded in the user's current code when useful.",
                "exposure_mode": "prompt_safe",
                "confidence": 0.88,
                "relevant": True,
            }
        ],
        traits={},
        policies={},
    )

    block = str(dict(pack.blocks or {}).get("retrieved_memories") or "")
    assert "Prefer examples grounded in the user's current code when useful." in block
    assert "user prefers examples grounded in own code" not in block


def test_self_recall_mode_can_surface_memory_without_internal_metadata() -> None:
    artifact = MemoryArtifact(
        artifact_id="prof-1",
        artifact_type="profile_fact",
        text="environment_gpu_model: RTX 3050 Ti",
        summary="RTX 3050 Ti",
        metadata={
            "prompt_view": "environment_gpu_model: RTX 3050 Ti",
            "exposure_mode": "exact_quote",
        },
    )

    pack = distill_memory_artifacts([artifact], query_text="что ты про меня помнишь?")

    assert "exact_recall" in pack.blocks
    assert "RTX 3050 Ti" in str(pack.blocks["exact_recall"])
    assert "exposure_mode" not in str(pack.blocks["exact_recall"])
    assert "metadata" not in str(pack.blocks["exact_recall"])


def test_prompt_build_adds_anti_verbatim_memory_policy_rules() -> None:
    ctx = PipelineContext(
        route="chat",
        user_msg="да не знаю уже",
        clean_user_msg="да не знаю уже",
        state={},
        meta={},
        tags={},
        retrieved_memories=[],
        traits={},
        policies={},
        profile=PROFILE_BALANCED,
        memory_context={
            "blocks": {
                "tone_hints": "- user currently low-bandwidth; keep tone gentle",
            },
            "selected": [
                {
                    "artifact_id": "emo-1",
                    "artifact_type": "emotional_state",
                    "text": "пользователь чувствует грусть",
                    "prompt_view": "user currently low-bandwidth; keep tone gentle",
                    "exposure_mode": "prompt_safe",
                }
            ],
            "recent_user_state": {"low_bandwidth": True, "sources": ["memory.emotional_state"]},
        },
    )

    ctx = PromptBuildStage(character_runtime=CharacterRuntime(autosave=False)).run(ctx)

    rules = "\n".join(str(x) for x in list(ctx.policies.get("rules") or []))
    assert "Retrieved memory is internal guidance" in rules
    assert "Convert emotional or profile memory into tone adaptation" in rules
    assert "Never mention internal artifact text" in rules


def test_memory_llm_processor_normalizes_emotional_state_into_internal_format() -> None:
    processor = MemoryLLMProcessor(task_router=None)
    result = processor._parse_response(
        "evt-1",
        json.dumps(
            {
                "event_id": "evt-1",
                "importance": 0.72,
                "should_process": True,
                "proposals": [
                    {
                        "artifact_type": "emotional_state",
                        "text": "пользователь чувствует грусть",
                        "summary": "грусть",
                        "confidence": 0.84,
                        "scope": "episode",
                        "decay": "fast",
                        "retrieve_when": ["emotion"],
                        "action": "create",
                        "metadata": {},
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )

    proposal = result.proposals[0]
    assert proposal.text.startswith("user emotion:")
    assert proposal.metadata["exposure_mode"] == "latent"
    assert str(proposal.metadata.get("prompt_view") or "").strip() != ""
