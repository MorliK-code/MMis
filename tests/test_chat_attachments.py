from __future__ import annotations

from api import app as api_app
from core.response_pipeline import _messages_with_attachment_prompt
from llm.ollama_provider import OllamaProvider, _message_log_fields
from llm.provider_base import LLMRequest, Message


class _FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"message": {"content": "ok"}, "model": kwargs.get("model") or "fake"}


def test_build_chat_meta_forwards_attachments() -> None:
    req = api_app.ChatRequest(
        text="describe this",
        attachments=[
            {
                "kind": "image",
                "name": "screen.png",
                "mime_type": "image/png",
                "size": 12,
                "data_base64": "abc123",
            }
        ],
    )

    meta = api_app._build_chat_meta(req=req, source="api")

    assert meta["attachments"][0]["kind"] == "image"
    assert meta["attachments"][0]["name"] == "screen.png"
    assert meta["attachments"][0]["data_base64"] == "abc123"


def test_chat_request_allows_attachment_only_message() -> None:
    req = api_app.ChatRequest(
        text="",
        attachments=[{"kind": "text", "name": "notes.txt", "text": "hello"}],
    )

    assert req.text == ""
    assert req.attachments[0].name == "notes.txt"


def test_attachment_prompt_adds_text_without_inlining_image_base64() -> None:
    messages = [Message(role="user", content="review the attachments")]
    attachments = [
        {"kind": "text", "name": "notes.md", "mime_type": "text/markdown", "size": 5, "text": "hello"},
        {"kind": "image", "name": "screen.png", "mime_type": "image/png", "size": 3, "data_base64": "abc123"},
    ]

    out = _messages_with_attachment_prompt(messages, attachments)
    content = out[-1].content

    assert "review the attachments" in content
    assert "notes.md" in content
    assert "hello" in content
    assert "screen.png" in content
    assert "abc123" not in content


def test_ollama_provider_attaches_images_to_last_user_message() -> None:
    fake_client = _FakeOllamaClient()
    provider = OllamaProvider(host="http://127.0.0.1:11434", default_model="fake", retries=0)
    provider._client = fake_client
    req = LLMRequest(
        model="fake",
        messages=[Message(role="system", content="sys"), Message(role="user", content="what is here?")],
        metadata={
            "attachments": [
                {
                    "kind": "image",
                    "name": "screen.png",
                    "mime_type": "image/png",
                    "data_base64": "abc123",
                }
            ]
        },
    )

    payload = provider._chat_once(req=req, model="fake", stream=False)

    assert payload["message"]["content"] == "ok"
    sent_messages = fake_client.calls[0]["messages"]
    assert sent_messages[-1]["role"] == "user"
    assert sent_messages[-1]["images"] == ["abc123"]


def test_ollama_payload_log_redacts_image_bytes() -> None:
    fields = _message_log_fields([{"role": "user", "content": "look", "images": ["abc123"]}])

    assert fields["messages"][0]["images"] == ["<image:6 chars>"]
