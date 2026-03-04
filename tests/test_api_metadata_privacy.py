from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api import app as api_app_module
from core.brain import BrainResult


class ApiMetadataPrivacyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_app_module.app)
        self._orig_handle_message = api_app_module._runtime.brain.handle_message
        self._orig_meta_root = Path(api_app_module._runtime.meta_root)
        self._orig_message_seq = int(getattr(api_app_module._runtime, "_message_seq", 0) or 0)
        self._tmp = tempfile.TemporaryDirectory(prefix="mmis_api_metadata_privacy_")
        api_app_module._runtime.meta_root = Path(self._tmp.name).resolve() / "metadata"
        api_app_module._runtime.meta_root.mkdir(parents=True, exist_ok=True)
        api_app_module._runtime._message_seq = 0

    def tearDown(self) -> None:
        api_app_module._runtime.brain.handle_message = self._orig_handle_message
        api_app_module._runtime.meta_root = self._orig_meta_root
        api_app_module._runtime._message_seq = self._orig_message_seq
        self._tmp.cleanup()

    def _rows(self) -> list[dict]:
        path = api_app_module._metadata_model_dir(api_app_module._runtime.model) / "metadata_messages.jsonl"
        if not path.exists():
            return []
        out: list[dict] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            raw = str(line or "").strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    def test_chat_commands_do_not_store_metadata(self) -> None:
        def _fake_handle_message(user_msg, meta=None):
            _ = (user_msg, meta)
            return BrainResult(
                text="command-ok",
                route="command",
                stats={"served_model": "stub"},
            )

        api_app_module._runtime.brain.handle_message = _fake_handle_message

        resp_native = self.client.post("/chat", json={"text": "/help", "store_turn": True})
        self.assertEqual(resp_native.status_code, 200)
        resp_command = self.client.post("/chat", json={"text": "/mode debugger", "store_turn": True})
        self.assertEqual(resp_command.status_code, 200)

        self.assertEqual(len(self._rows()), 0)

    def test_chat_studio_payload_does_not_store_metadata(self) -> None:
        def _fake_handle_message(user_msg, meta=None):
            _ = (user_msg, meta)
            return BrainResult(
                text="studio-step",
                route="chat",
                structured_output={
                    "studio_generator": {"active": True, "phase": "clarify"},
                    "studio": {"phase": "clarify"},
                },
                stats={"served_model": "stub"},
            )

        api_app_module._runtime.brain.handle_message = _fake_handle_message

        response = self.client.post("/chat", json={"text": "обнови гарри", "store_turn": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self._rows()), 0)

    def test_stream_studio_payload_does_not_store_metadata(self) -> None:
        def _fake_handle_message(user_msg, meta=None):
            _ = (user_msg, meta)
            return BrainResult(
                text="studio-stream-step",
                route="chat",
                structured_output={
                    "studio_generator": {"active": True, "phase": "clarify"},
                    "studio": {"phase": "clarify"},
                },
                stats={"served_model": "stub"},
            )

        api_app_module._runtime.brain.handle_message = _fake_handle_message

        with self.client.stream("POST", "/chat/stream", json={"text": "обнови гарри", "store_turn": True}) as response:
            self.assertEqual(response.status_code, 200)
            for _line in response.iter_lines():
                pass

        self.assertEqual(len(self._rows()), 0)


if __name__ == "__main__":
    unittest.main()

