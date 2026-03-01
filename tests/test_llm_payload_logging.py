from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm.ollama_provider import OllamaProvider
from llm.openai_provider import OpenAIProvider
from llm.provider_base import LLMRequest, Message, ToolSpec


class _FakeOllamaClient:
    def __init__(self) -> None:
        self.last_kwargs = None

    def chat(self, **kwargs):
        self.last_kwargs = dict(kwargs)
        return {
            "model": str(kwargs.get("model") or ""),
            "message": {"content": "ok"},
            "prompt_eval_count": 1,
            "eval_count": 1,
            "total_duration": 1000,
        }


class _FakeOpenAICompletions:
    def __init__(self) -> None:
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = dict(kwargs)
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        msg = SimpleNamespace(content="ok", tool_calls=[])
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice], usage=usage, model=str(kwargs.get("model") or ""))


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeOpenAICompletions())


def _extract_payload_from_log_calls(mock_log, provider: str) -> dict:
    for call in mock_log.call_args_list:
        if len(call.args) < 2:
            continue
        event = call.args[1]
        if str(event) != "llm_request_payload":
            continue
        if str(call.kwargs.get("provider") or "") != provider:
            continue
        payload = call.kwargs.get("payload")
        if isinstance(payload, dict):
            return payload
    raise AssertionError(f"llm_request_payload for provider={provider} not found in logs")


def _extract_request_log_kwargs(mock_log, provider: str) -> dict:
    for call in mock_log.call_args_list:
        if len(call.args) < 2:
            continue
        event = call.args[1]
        if str(event) != "llm_request_payload":
            continue
        if str(call.kwargs.get("provider") or "") != provider:
            continue
        return dict(call.kwargs)
    raise AssertionError(f"llm_request_payload for provider={provider} not found in logs")


class PayloadLoggingTests(unittest.TestCase):
    def test_ollama_logs_exact_request_payload(self) -> None:
        provider = OllamaProvider(default_model="model-ollama-test")
        fake_client = _FakeOllamaClient()
        provider._client = fake_client

        req = LLMRequest(
            model="model-ollama-test",
            messages=[Message(role="system", content="s"), Message(role="user", content="u")],
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.05,
            seed=7,
            max_tokens=64,
            stop=["</s>"],
            json_mode=True,
            metadata={"think": True, "keep_alive": "10s", "num_ctx": 2048},
            tools=[
                ToolSpec(
                    name="tool_x",
                    description="d",
                    input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
                )
            ],
        )

        with patch("llm.ollama_provider.log_json") as mock_log:
            provider.generate(req)

        self.assertIsInstance(fake_client.last_kwargs, dict)
        logged_payload = _extract_payload_from_log_calls(mock_log, provider="ollama")
        self.assertEqual(logged_payload, fake_client.last_kwargs)
        row = _extract_request_log_kwargs(mock_log, provider="ollama")
        self.assertEqual(row.get("messages"), fake_client.last_kwargs.get("messages"))
        self.assertEqual(row.get("message_roles"), ["system", "user"])
        self.assertEqual(row.get("message_order"), ["0:system", "1:user"])
        self.assertEqual(row.get("system_message"), "s")

    def test_openai_logs_exact_request_payload(self) -> None:
        provider = OpenAIProvider(api_key="test-key", default_model="model-openai-test")
        fake_client = _FakeOpenAIClient()
        provider._client = fake_client

        req = LLMRequest(
            model="model-openai-test",
            messages=[Message(role="system", content="s"), Message(role="user", content="u")],
            temperature=0.3,
            top_p=0.8,
            seed=11,
            max_tokens=77,
            stop=["<END>"],
            json_mode=True,
            tools=[
                ToolSpec(
                    name="tool_y",
                    description="d2",
                    input_schema={"type": "object", "properties": {"y": {"type": "integer"}}},
                )
            ],
        )

        with patch("llm.openai_provider.log_json") as mock_log:
            provider.generate(req)

        sent_payload = fake_client.chat.completions.last_kwargs
        self.assertIsInstance(sent_payload, dict)
        logged_payload = _extract_payload_from_log_calls(mock_log, provider="openai")
        self.assertEqual(logged_payload, sent_payload)
        row = _extract_request_log_kwargs(mock_log, provider="openai")
        self.assertEqual(row.get("messages"), sent_payload.get("messages"))
        self.assertEqual(row.get("message_roles"), ["system", "user"])
        self.assertEqual(row.get("message_order"), ["0:system", "1:user"])
        self.assertEqual(row.get("system_message"), "s")


if __name__ == "__main__":
    unittest.main()

