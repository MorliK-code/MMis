from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from core.response_pipeline import ResponsePipeline
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, ModelInfo, ProviderHealth, Timings, Usage


class _StubProvider(LLMProviderBase):
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


class CommandScopesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ResponsePipeline(provider=_StubProvider())

    def _run(self, route: str, msg: str, state: dict):
        return self.pipeline.run(
            route=route,
            user_msg=msg,
            state=state,
            meta={"source": "test", "store_turn": False, "conversation_id": "scope-test"},
            retrieved_memories=[],
            traits={},
            policies={},
        )

    def test_chat_scope_keeps_global_ops(self) -> None:
        result = self._run("command", "/web", {"history": [], "quality_profile": "BALANCED", "active_mode": "friend_chat"})
        ops = [x for x in result.memory_ops if str(x.get("op")) == "state_web_mode"]
        self.assertTrue(ops)
        self.assertEqual(str(ops[-1].get("value") or ""), "on")

    def test_studio_scope_uses_scoped_settings(self) -> None:
        state = {"history": [], "quality_profile": "BALANCED", "active_mode": "friend_chat"}
        start = self._run("command", "/studio start demo", state)
        state["studio_generator"] = dict(start.structured_output.get("studio_generator") or {})
        result = self._run("command", "/web", state)
        scoped_ops = [x for x in result.memory_ops if str(x.get("op")) == "state_scoped_settings"]
        self.assertTrue(scoped_ops)
        web_ops = [x for x in result.memory_ops if str(x.get("op")) == "state_web_mode"]
        self.assertFalse(web_ops)

    def test_studio_dialog_not_written_to_turn_memory(self) -> None:
        state = {"history": [], "quality_profile": "BALANCED", "active_mode": "friend_chat"}
        start = self._run("command", "/studio start demo", state)
        state["studio_generator"] = dict(start.structured_output.get("studio_generator") or {})

        step = self._run("chat", "demo", state)
        turn_ops = [x for x in list(step.memory_ops or []) if str(x.get("op")) in {"turn_user", "turn_assistant"}]
        self.assertFalse(turn_ops)


if __name__ == "__main__":
    unittest.main()
