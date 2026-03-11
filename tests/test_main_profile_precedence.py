from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest
from types import SimpleNamespace

from config.settings import get_profile
from main import _brain_meta


class _RuntimeStub:
    def get_active_character_id(self):
        return "asya"

    def get_meta(self, _cid=None):
        return SimpleNamespace(llm_profile="FAST")


class MainProfilePrecedenceTests(unittest.TestCase):
    def test_brain_meta_uses_character_profile_over_config(self) -> None:
        container = SimpleNamespace(
            settings=SimpleNamespace(
                active_profile="QUALITY",
                model_name="stub-model",
                thinking_enabled=True,
                llm_default_provider="ollama",
            ),
            character_runtime=_RuntimeStub(),
        )
        meta = _brain_meta(container, source="cli")
        fast_profile = get_profile("FAST")
        self.assertEqual(str(meta.get("quality_profile") or ""), "FAST")
        self.assertEqual(float(meta.get("temperature") or 0.0), float(fast_profile.generation.temperature))
        self.assertEqual(float(meta.get("top_p") or 0.0), float(fast_profile.generation.top_p))
        self.assertEqual(float(meta.get("repeat_penalty") or 0.0), float(fast_profile.generation.repeat_penalty))

    def test_brain_meta_preserves_asya_profile_name(self) -> None:
        class _AsyaRuntimeStub(_RuntimeStub):
            def get_meta(self, _cid=None):
                return SimpleNamespace(llm_profile="ASYA")

        container = SimpleNamespace(
            settings=SimpleNamespace(
                active_profile="QUALITY",
                model_name="stub-model",
                thinking_enabled=True,
                llm_default_provider="ollama",
            ),
            character_runtime=_AsyaRuntimeStub(),
        )
        meta = _brain_meta(container, source="cli")
        asya_profile = get_profile("ASYA")
        self.assertEqual(str(meta.get("quality_profile") or ""), "ASYA")
        self.assertEqual(float(meta.get("temperature") or 0.0), float(asya_profile.generation.temperature))
        self.assertEqual(float(meta.get("top_p") or 0.0), float(asya_profile.generation.top_p))
        self.assertEqual(float(meta.get("repeat_penalty") or 0.0), float(asya_profile.generation.repeat_penalty))


if __name__ == "__main__":
    unittest.main()
