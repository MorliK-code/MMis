from __future__ import annotations

from llm import ollama_provider as ollama_provider_module
from llm.ollama_provider import OllamaProvider
from llm.provider_base import LLMRequest, Message


class _PriorityManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def wait_for_turn(self, priority: int, timeout: float | None = None) -> bool:
        self.calls.append(("wait", priority))
        return True

    def release(self, priority: int) -> None:
        self.calls.append(("release", priority))


def test_ollama_stream_uses_thinking_field_as_raw_delta(monkeypatch) -> None:
    provider = OllamaProvider(default_model="main-model", timeout_sec=1.0)

    def _chat_with_retry(*, req, model, stream):
        assert stream is True
        return iter(
            [
                {
                    "message": {"thinking": "Long "},
                    "done": False,
                    "model": model,
                },
                {
                    "message": {"thinking": "chain", "content": "Answer"},
                    "done": True,
                    "model": model,
                },
            ]
        )

    provider._chat_with_retry = _chat_with_retry
    priority = _PriorityManager()
    monkeypatch.setattr(ollama_provider_module, "get_priority_manager", lambda: priority)

    chunks = list(
        provider.stream(
            LLMRequest(
                model="main-model",
                messages=[Message(role="user", content="hello")],
                metadata={"source": "api", "think": True},
            )
        )
    )

    assert [chunk.thinking_delta for chunk in chunks] == ["Long ", "chain"]
    assert "".join(chunk.thinking_delta for chunk in chunks) == "Long chain"
    assert "".join(chunk.text_delta for chunk in chunks) == "Answer"
    assert priority.calls == [
        ("wait", ollama_provider_module.LLMPriorityManager.PRIORITY_MAIN),
        ("release", ollama_provider_module.LLMPriorityManager.PRIORITY_MAIN),
    ]
