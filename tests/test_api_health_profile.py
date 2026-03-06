from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from fastapi.testclient import TestClient

from api import app as api_app_module
from config.settings import get_profile


class ApiHealthProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_app_module.app)
        self._orig_active_profile = api_app_module._runtime.active_profile
        self._orig_quality_profile = api_app_module._runtime.quality_profile
        self._orig_get_profile = api_app_module.get_profile

    def tearDown(self) -> None:
        api_app_module._runtime.active_profile = self._orig_active_profile
        api_app_module._runtime.quality_profile = self._orig_quality_profile
        api_app_module.get_profile = self._orig_get_profile

    def test_health_exposes_profile_and_parameters(self) -> None:
        base = get_profile("BALANCED")
        custom = type(base)(
            name=base.name,
            generation=type(base.generation)(
                temperature=0.31,
                top_p=0.81,
                repeat_penalty=1.19,
                max_tokens=777,
                stop=(),
            ),
            ollama=type(base.ollama)(
                num_thread=7,
                num_ctx=7168,
                num_gpu=1,
                num_batch=111,
                keep_alive="7m",
            ),
            openai=base.openai,
        )
        api_app_module._runtime.active_profile = "BALANCED"
        api_app_module._runtime.quality_profile = "BALANCED"
        api_app_module.get_profile = lambda _name: custom

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(str(payload.get("active_profile") or ""), "BALANCED")
        self.assertEqual(str(payload.get("quality_profile") or ""), "BALANCED")
        params = dict(payload.get("profile_parameters") or {})
        gen = dict(params.get("generation") or {})
        ollama = dict(params.get("ollama") or {})
        self.assertEqual(float(gen.get("temperature")), 0.31)
        self.assertEqual(float(gen.get("top_p")), 0.81)
        self.assertEqual(float(gen.get("repeat_penalty")), 1.19)
        self.assertEqual(int(gen.get("max_tokens")), 777)
        self.assertEqual(int(ollama.get("num_ctx")), 7168)
        self.assertEqual(int(ollama.get("num_batch")), 111)


if __name__ == "__main__":
    unittest.main()
