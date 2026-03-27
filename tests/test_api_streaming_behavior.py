from __future__ import annotations

import json
import sys
import unittest
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from fastapi.testclient import TestClient


def _reply(
    *,
    text: str,
    thinking: str = "",
    stats: dict | None = None,
    model: str = "stub-model",
    status: str = "ok",
):
    return SimpleNamespace(
        text=text,
        thinking=thinking,
        structured_output={},
        stats=dict(stats or {}),
        model=model,
        debug_trace={},
        memory_debug_snapshot={},
        status=status,
    )


class ApiStreamingBehaviorTests(unittest.TestCase):
    @staticmethod
    def _api_app():
        cached = sys.modules.get("api.app")
        if cached is not None:
            return cached
        with patch("llm.build_provider", return_value=SimpleNamespace()):
            return import_module("api.app")

    def _collect_events(self, client: TestClient, payload: dict) -> list[dict]:
        with client.stream("POST", "/chat/stream", json=payload) as resp:
            self.assertEqual(resp.status_code, 200)
            return [json.loads(line) for line in resp.iter_lines() if line]

    def test_chat_stream_does_not_fake_chunk_replay_when_no_live_stream(self) -> None:
        def fake_handle_message(_text: str, meta: dict | None = None):
            self.assertTrue(callable(dict(meta or {}).get("stream_on_answer_chunk")))
            return _reply(text="Hello world", stats={"served_model": "stub-model"})

        api_app = self._api_app()
        with TestClient(api_app.app) as client:
            with patch.object(api_app._runtime.brain, "handle_message", side_effect=fake_handle_message):
                events = self._collect_events(client, {"text": "hello", "store_turn": False})

        self.assertEqual([str(row.get("event") or "") for row in events], ["final"])
        payload = dict(events[0].get("data") or {})
        stats = dict(payload.get("stats") or {})
        self.assertEqual(str(payload.get("answer") or ""), "Hello world")
        self.assertFalse(bool(stats.get("streaming_live")))
        self.assertEqual(int(stats.get("streamed_answer_chars") or 0), 0)
        self.assertEqual(int(stats.get("streamed_thinking_chars") or 0), 0)

    def test_chat_stream_preserves_live_chunks_without_duplicate_replay(self) -> None:
        def fake_handle_message(_text: str, meta: dict | None = None):
            meta_map = dict(meta or {})
            meta_map["stream_on_answer_chunk"]("Hello ")
            meta_map["stream_on_answer_chunk"]("world")
            return _reply(text="Hello world", stats={"served_model": "stub-model", "streaming": True})

        api_app = self._api_app()
        with TestClient(api_app.app) as client:
            with patch.object(api_app._runtime.brain, "handle_message", side_effect=fake_handle_message):
                events = self._collect_events(client, {"text": "hello", "store_turn": False})

        self.assertEqual(
            [str(row.get("event") or "") for row in events],
            ["chunk", "chunk", "final"],
        )
        self.assertEqual("".join(str(row.get("data") or "") for row in events[:-1]), "Hello world")
        payload = dict(events[-1].get("data") or {})
        stats = dict(payload.get("stats") or {})
        self.assertEqual(str(payload.get("answer") or ""), "Hello world")
        self.assertTrue(bool(stats.get("streaming_live")))
        self.assertEqual(int(stats.get("streamed_answer_chars") or 0), len("Hello world"))

    def test_chat_does_not_store_assistant_metadata_for_error_result(self) -> None:
        api_app = self._api_app()
        with TestClient(api_app.app) as client:
            with patch.object(api_app._runtime.brain, "handle_message", return_value=_reply(text="Я затупила. Повтори, пожалуйста, еще раз.", status="error")):
                with patch.object(api_app, "_append_metadata_row") as append_row:
                    with patch.object(type(api_app.memory_core_adapter), "worker_pause_enabled", new_callable=PropertyMock, return_value=False):
                        resp = client.post("/chat", json={"text": "hello", "store_turn": True})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(append_row.call_args_list), 1)
        self.assertEqual(str(append_row.call_args.kwargs.get("role") or ""), "user")


    def test_build_chat_meta_disables_qwen3_thinking_hotfix(self) -> None:
        api_app = self._api_app()
        req = api_app.ChatRequest(text="hello", think=True, store_turn=False)

        with patch.object(api_app._runtime, "model", "qwen3:8b"):
            with patch.object(api_app._runtime, "thinking_enabled", True):
                qwen_meta = api_app._build_chat_meta(req=req, source="api")

        with patch.object(api_app._runtime, "model", "llama3.1:8b"):
            with patch.object(api_app._runtime, "thinking_enabled", True):
                other_meta = api_app._build_chat_meta(req=req, source="api")

        self.assertFalse(bool(qwen_meta.get("think")))
        self.assertTrue(bool(other_meta.get("think")))


if __name__ == "__main__":
    unittest.main()
