from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import unittest

from fastapi.testclient import TestClient

from api import app as api_app_module
from core.brain import BrainResult


class ApiOutputFormatContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_app_module.app)
        self._orig_handle_message = api_app_module._runtime.brain.handle_message

    def tearDown(self) -> None:
        api_app_module._runtime.brain.handle_message = self._orig_handle_message

    def test_chat_response_exposes_parameters_and_summary(self) -> None:
        def _fake_handle_message(user_msg, meta=None):
            _ = (user_msg, meta)
            return BrainResult(
                text="[PARAMETERS]\nmode=debugger\n\n[SUMMARY]\nshort\n\n[RESPONSE]\nbody",
                route="chat",
                thinking="",
                structured_output={
                    "parameters": {"mode": "debugger"},
                    "summary": "short",
                    "text": "body",
                    "formatted": True,
                },
                stats={"served_model": "stub"},
            )

        api_app_module._runtime.brain.handle_message = _fake_handle_message
        response = self.client.post("/chat", json={"text": "hello", "store_turn": False})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("answer", payload)
        self.assertEqual(payload.get("summary"), "short")
        self.assertEqual(dict(payload.get("parameters") or {}).get("mode"), "debugger")

    def test_stream_final_payload_exposes_parameters_and_summary(self) -> None:
        def _fake_handle_message(user_msg, meta=None):
            _ = user_msg
            callbacks = dict(meta or {})
            on_answer = callbacks.get("stream_on_answer_chunk")
            if callable(on_answer):
                on_answer("[RESPONSE]\nbody")
            return BrainResult(
                text="[PARAMETERS]\nmode=debugger\n\n[SUMMARY]\nshort\n\n[RESPONSE]\nbody",
                route="chat",
                thinking="",
                structured_output={
                    "parameters": {"mode": "debugger"},
                    "summary": "short",
                    "text": "body",
                    "formatted": True,
                },
                stats={"served_model": "stub"},
            )

        api_app_module._runtime.brain.handle_message = _fake_handle_message
        with self.client.stream("POST", "/chat/stream", json={"text": "hello", "store_turn": False}) as response:
            self.assertEqual(response.status_code, 200)
            events = []
            for line in response.iter_lines():
                raw = str(line or "").strip()
                if not raw:
                    continue
                events.append(json.loads(raw))
        finals = [x for x in events if str(x.get("event")) == "final"]
        self.assertTrue(finals)
        payload = dict(finals[-1].get("data") or {})
        self.assertEqual(payload.get("summary"), "short")
        self.assertEqual(dict(payload.get("parameters") or {}).get("mode"), "debugger")


if __name__ == "__main__":
    unittest.main()
