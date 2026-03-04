from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from fastapi.testclient import TestClient

from api import app as api_app_module


class ApiFeedbackLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_app_module.app)

    def test_feedback_records_runtime_action(self) -> None:
        state_mgr = api_app_module._runtime.brain.state_manager
        before_actions = list(state_mgr.snapshot().last_actions or [])
        before_len = len(before_actions)

        response = self.client.post(
            "/feedback",
            json={
                "user_text": "не подкалывай",
                "assistant_text": "поняла",
                "feedback": -1,
                "penalty": 0.2,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(int(payload.get("feedback") or 0), -1)

        after_actions = list(state_mgr.snapshot().last_actions or [])
        tail = after_actions[before_len:]
        self.assertTrue(any(str(x.get("type") or "").upper() == "FEEDBACK_RECEIVED" for x in tail))


if __name__ == "__main__":
    unittest.main()
