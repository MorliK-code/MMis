from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import tempfile
import unittest
from pathlib import Path

from core.character_runtime import CharacterRuntime
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


class WebProfileGatingTests(unittest.TestCase):
    def _build_pipeline(self) -> ResponsePipeline:
        tmp = tempfile.TemporaryDirectory(prefix="mmis_web_profile_")
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        runtime = CharacterRuntime(
            state_path=root / "brain_state.json",
            state_store_dir=root / "brain_state_store",
            autosave=False,
        )
        return ResponsePipeline(provider=_StubProvider(), character_runtime=runtime)

    def test_web_stage_enabled_for_balanced_quality_asya(self) -> None:
        pipeline = self._build_pipeline()
        for profile in ("BALANCED", "QUALITY", "ASYA"):
            resolved = pipeline._resolve_profile(meta={}, state={"quality_profile": profile}, policies={})  # type: ignore[attr-defined]
            stage_profile = pipeline._resolve_stage_profile(resolved)  # type: ignore[attr-defined]
            stages = pipeline._resolve_stage_names(stage_profile, {}, {})  # type: ignore[attr-defined]
            self.assertIn("web_retrieve", stages, msg=f"profile={profile} should include web_retrieve")

    def test_web_stage_disabled_for_fast_econom_autonomous(self) -> None:
        pipeline = self._build_pipeline()
        for profile in ("FAST", "ECONOM", "AUTONOMOUS"):
            resolved = pipeline._resolve_profile(meta={}, state={"quality_profile": profile}, policies={})  # type: ignore[attr-defined]
            stage_profile = pipeline._resolve_stage_profile(resolved)  # type: ignore[attr-defined]
            stages = pipeline._resolve_stage_names(stage_profile, {}, {})  # type: ignore[attr-defined]
            self.assertNotIn("web_retrieve", stages, msg=f"profile={profile} should not include web_retrieve")


if __name__ == "__main__":
    unittest.main()
