from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from core.brain import Brain, BrainResult
from core.character_runtime import CharacterRuntime
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, ModelInfo, ProviderHealth, Timings, Usage


class _ProviderStub(LLMProviderBase):
    def generate(self, req: LLMRequest) -> LLMResponse:
        _ = req
        return LLMResponse(
            text="ok",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            timings=Timings(latency_ms=1.0),
            model="stub",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="stub", detail="ok", model="stub")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="stub", model=model or "stub")

    def list_models(self) -> list[str]:
        return ["stub"]


class _MemoryManagerStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ingest_message(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))


class BrainPersistAnswerOnlyTests(unittest.TestCase):
    def test_assistant_ingest_uses_answer_only_text(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_brain_answer_only_") as tmpdir:
            state = CharacterRuntime(state_path=Path(tmpdir) / "brain_state.json", autosave=False)
            memory = _MemoryManagerStub()
            brain = Brain(provider=_ProviderStub(), state_manager=state, memory_manager=memory, throttle_sec=0.0)

            seen_meta_inputs: list[str] = []

            def _capture_metadata(*, text: str, state, last_messages):
                _ = (state, last_messages)
                seen_meta_inputs.append(str(text))
                return {"tags": []}

            brain._extract_turn_metadata = _capture_metadata  # type: ignore[method-assign]

            result = BrainResult(
                text="[PARAMETERS]\nmode=debugger\n\n[SUMMARY]\nshort\n\n[RESPONSE]\nanswer-only",
                route="chat",
                structured_output={"text": "answer-only"},
                stats={},
            )
            brain._persist_turns(
                route="chat",
                user_text="hello",
                result=result,
                meta={
                    "store_turn": True,
                    "source": "test",
                    "conversation_id": "c1",
                    "turn_id": 1,
                    "quality_profile": "BALANCED",
                },
                non_persistent_turn=False,
            )

            self.assertEqual(len(memory.calls), 2)
            assistant_call = dict(memory.calls[-1] or {})
            self.assertEqual(str(assistant_call.get("role") or ""), "assistant")
            self.assertEqual(str(assistant_call.get("text") or ""), "answer-only")
            self.assertTrue(seen_meta_inputs)
            self.assertEqual(str(seen_meta_inputs[-1] or ""), "answer-only")


if __name__ == "__main__":
    unittest.main()
