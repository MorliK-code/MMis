from __future__ import annotations

from core.character_runtime import CharacterRuntime
from core.response_pipeline import GenerateStage, PipelineContext
from llm.provider_base import LLMChunk, LLMRequest, Message, ToolCall, ToolSpec


class _FakeStreamProvider:
    def __init__(self, chunks: list[LLMChunk]) -> None:
        self._chunks = list(chunks)

    def stream(self, req: LLMRequest):
        for chunk in self._chunks:
            yield chunk

    def generate(self, req: LLMRequest):
        raise AssertionError("generate() should not be used in this test")


class _AssertingLiveStreamProvider:
    def __init__(self, streamed_sink: list[str]) -> None:
        self._streamed_sink = streamed_sink

    def stream(self, req: LLMRequest):
        yield LLMChunk(
            text_delta="This is a fairly long opening sentence that should stream immediately ",
            model="fake-model",
            done=False,
        )
        assert "".join(self._streamed_sink).startswith("This is a fairly long opening sentence")
        yield LLMChunk(
            text_delta="and continue without waiting for done.",
            model="fake-model",
            done=False,
        )
        assert "continue without waiting" in "".join(self._streamed_sink)
        yield LLMChunk(model="fake-model", done=True)

    def generate(self, req: LLMRequest):
        raise AssertionError("generate() should not be used in this test")


def _make_ctx() -> PipelineContext:
    return PipelineContext(
        route="chat",
        user_msg="hello",
        state={},
        meta={},
        retrieved_memories=[],
        traits={},
        policies={},
        profile="AUTONOMOUS",
    )


def test_generate_stage_streams_answer_immediately_when_enabled() -> None:
    provider = _FakeStreamProvider(
        [
            LLMChunk(text_delta="Draft ", model="fake-model", done=False),
            LLMChunk(text_delta="answer", model="fake-model", done=False),
            LLMChunk(model="fake-model", done=True),
        ]
    )
    stage = GenerateStage(provider=provider, character_runtime=CharacterRuntime())
    streamed: list[str] = []

    response = stage._generate_with_agent_loop_streaming(
        _make_ctx(),
        LLMRequest(
            messages=[Message(role="user", content="hello")],
            tools=[ToolSpec(name="memory_retrieve", description="...", input_schema={})],
        ),
        on_answer=lambda piece: streamed.append(piece),
        on_thinking=lambda piece: None,
        emit_answer_live=True,
    )

    assert "".join(streamed) == "Draft answer"
    assert response.text == "Draft answer"
    assert len(response.tool_calls) == 0


def test_generate_stage_can_suppress_provisional_answer_without_tool_call() -> None:
    provider = _FakeStreamProvider(
        [
            LLMChunk(text_delta="Provisional ", model="fake-model", done=False),
            LLMChunk(text_delta="answer", model="fake-model", done=True),
        ]
    )
    stage = GenerateStage(provider=provider, character_runtime=CharacterRuntime())
    streamed: list[str] = []

    response = stage._generate_with_agent_loop_streaming(
        _make_ctx(),
        LLMRequest(messages=[Message(role="user", content="hello")]),
        on_answer=lambda piece: streamed.append(piece),
        on_thinking=lambda piece: None,
        emit_answer_live=False,
    )

    assert streamed == []
    assert response.text == "Provisional answer"


def test_generate_stage_streams_long_answer_live_without_probe_pause() -> None:
    streamed: list[str] = []
    provider = _AssertingLiveStreamProvider(streamed)
    stage = GenerateStage(provider=provider, character_runtime=CharacterRuntime())

    response = stage._generate_with_agent_loop_streaming(
        _make_ctx(),
        LLMRequest(
            messages=[Message(role="user", content="hello")],
            tools=[ToolSpec(name="memory_retrieve", description="...", input_schema={})],
        ),
        on_answer=lambda piece: streamed.append(piece),
        on_thinking=lambda piece: None,
        emit_answer_live=True,
    )

    assert "".join(streamed) == response.text
    assert response.text.endswith("continue without waiting for done.")
