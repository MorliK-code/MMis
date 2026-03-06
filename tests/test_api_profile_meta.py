from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import unittest

from fastapi.testclient import TestClient

from api import app as api_app_module
from config.settings import get_profile
from core.brain import BrainResult


class ApiProfileMetaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_app_module.app)
        self._orig_handle_message = api_app_module._runtime.brain.handle_message
        self._orig_active_profile = api_app_module._runtime.active_profile
        self._orig_quality_profile = api_app_module._runtime.quality_profile
        self._orig_provider_name = api_app_module._runtime.provider_name
        self._orig_model = api_app_module._runtime.model
        self._orig_get_profile = api_app_module.get_profile

    def tearDown(self) -> None:
        api_app_module._runtime.brain.handle_message = self._orig_handle_message
        api_app_module._runtime.active_profile = self._orig_active_profile
        api_app_module._runtime.quality_profile = self._orig_quality_profile
        api_app_module._runtime.provider_name = self._orig_provider_name
        api_app_module._runtime.model = self._orig_model
        api_app_module.get_profile = self._orig_get_profile

    def test_chat_meta_contains_profile_generation_params(self) -> None:
        quality_profile = get_profile("QUALITY")
        api_app_module._runtime.active_profile = "QUALITY"
        api_app_module._runtime.quality_profile = "QUALITY"
        api_app_module._runtime.provider_name = "ollama"
        api_app_module._runtime.model = "stub-model"
        captured_meta: dict = {}

        def _fake_handle_message(user_msg, meta=None):
            _ = user_msg
            captured_meta.clear()
            captured_meta.update(dict(meta or {}))
            return BrainResult(text="ok", route="chat", stats={"served_model": "stub"})

        api_app_module._runtime.brain.handle_message = _fake_handle_message
        response = self.client.post("/chat", json={"text": "hello", "store_turn": False})
        self.assertEqual(response.status_code, 200)

        self.assertEqual(str(captured_meta.get("quality_profile") or ""), "QUALITY")
        self.assertEqual(float(captured_meta.get("temperature")), float(quality_profile.generation.temperature))
        self.assertEqual(float(captured_meta.get("top_p")), float(quality_profile.generation.top_p))
        self.assertEqual(float(captured_meta.get("repeat_penalty")), float(quality_profile.generation.repeat_penalty))
        self.assertEqual(int(captured_meta.get("max_tokens")), int(quality_profile.generation.max_tokens or 0))
        self.assertEqual(int(captured_meta.get("num_ctx")), int(quality_profile.ollama.num_ctx))
        self.assertEqual(int(captured_meta.get("num_thread")), int(quality_profile.ollama.num_thread))
        self.assertEqual(int(captured_meta.get("num_batch")), int(quality_profile.ollama.num_batch))
        self.assertEqual(int(captured_meta.get("num_gpu")), int(quality_profile.ollama.num_gpu))
        self.assertEqual(str(captured_meta.get("keep_alive") or ""), str(quality_profile.ollama.keep_alive))

    def test_stream_meta_contains_profile_generation_params(self) -> None:
        fast_profile = get_profile("FAST")
        api_app_module._runtime.active_profile = "FAST"
        api_app_module._runtime.quality_profile = "FAST"
        api_app_module._runtime.provider_name = "ollama"
        captured_meta: dict = {}

        def _fake_handle_message(user_msg, meta=None):
            _ = user_msg
            captured_meta.clear()
            captured_meta.update(dict(meta or {}))
            on_answer = captured_meta.get("stream_on_answer_chunk")
            if callable(on_answer):
                on_answer("ok")
            return BrainResult(text="ok", route="chat", stats={"served_model": "stub"})

        api_app_module._runtime.brain.handle_message = _fake_handle_message
        with self.client.stream("POST", "/chat/stream", json={"text": "hello", "store_turn": False}) as response:
            self.assertEqual(response.status_code, 200)
            for line in response.iter_lines():
                raw = str(line or "").strip()
                if raw:
                    json.loads(raw)

        self.assertEqual(str(captured_meta.get("quality_profile") or ""), "FAST")
        self.assertEqual(float(captured_meta.get("temperature")), float(fast_profile.generation.temperature))
        self.assertEqual(float(captured_meta.get("top_p")), float(fast_profile.generation.top_p))
        self.assertEqual(float(captured_meta.get("repeat_penalty")), float(fast_profile.generation.repeat_penalty))
        self.assertEqual(int(captured_meta.get("max_tokens")), int(fast_profile.generation.max_tokens or 0))
        self.assertTrue(callable(captured_meta.get("stream_on_answer_chunk")))
        self.assertTrue(callable(captured_meta.get("stream_on_thinking_chunk")))

    def test_chat_meta_uses_get_profile_source_of_truth(self) -> None:
        custom_profile = get_profile("BALANCED")
        custom_profile = type(custom_profile)(
            name=custom_profile.name,
            generation=type(custom_profile.generation)(
                temperature=0.12,
                top_p=0.77,
                repeat_penalty=1.33,
                max_tokens=321,
                stop=(),
            ),
            ollama=type(custom_profile.ollama)(
                num_thread=3,
                num_ctx=3072,
                num_gpu=0,
                num_batch=42,
                keep_alive="3m",
            ),
            openai=custom_profile.openai,
        )
        api_app_module._runtime.active_profile = "BALANCED"
        api_app_module._runtime.quality_profile = "BALANCED"
        api_app_module._runtime.provider_name = "ollama"
        api_app_module.get_profile = lambda _name: custom_profile
        captured_meta: dict = {}

        def _fake_handle_message(user_msg, meta=None):
            _ = user_msg
            captured_meta.clear()
            captured_meta.update(dict(meta or {}))
            return BrainResult(text="ok", route="chat", stats={"served_model": "stub"})

        api_app_module._runtime.brain.handle_message = _fake_handle_message
        response = self.client.post("/chat", json={"text": "hello", "store_turn": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(float(captured_meta.get("temperature")), 0.12)
        self.assertEqual(float(captured_meta.get("top_p")), 0.77)
        self.assertEqual(float(captured_meta.get("repeat_penalty")), 1.33)
        self.assertEqual(int(captured_meta.get("max_tokens")), 321)
        self.assertEqual(int(captured_meta.get("num_ctx")), 3072)
        self.assertEqual(int(captured_meta.get("num_batch")), 42)


if __name__ == "__main__":
    unittest.main()
