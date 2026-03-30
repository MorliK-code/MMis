"""
Тесты на streaming + tools в response_pipeline.

Проверяют, что:
- streaming чанки отдаются сразу (live)
- tool_calls_delta корректно собираются
- tools по-прежнему работают
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm.provider_base import LLMChunk, LLMRequest, Message, ToolCall, ToolSpec


class _FakeProvider:
    """Фейковый provider для тестов streaming."""

    def __init__(self, chunks: list[LLMChunk] | None = None):
        self.chunks = chunks or []
        self.generate_called = False
        self._stream_answer_chunks: list[str] = []
        self._stream_thinking_chunks: list[str] = []

    def stream(self, req: LLMRequest):
        for chunk in self.chunks:
            yield chunk

    def generate(self, req: LLMRequest):
        self.generate_called = True
        raise AssertionError("generate fallback should not be used in this test")


def _run_agent_loop_streaming(ctx, provider, req, on_answer, on_thinking):
    """
    Вспомогательная функция для тестирования streaming.
    Копия логики из _generate_with_agent_loop_streaming.
    """
    from core.response_pipeline import _ThinkStreamParser
    from llm.provider_base import LLMResponse, Usage, Timings

    parser = _ThinkStreamParser()
    answer_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    model_name = str(req.model or "")
    usage = Usage()
    timings = Timings()

    # В streaming-режиме не буферизуем первые видимые куски ответа.
    probe_tool_calls = False
    probe_answer_buffer: list[str] = []
    probe_thinking_buffer: list[str] = []
    probe_visible_pieces = 0
    probe_released = True

    def _handle_visible_answer(piece: str) -> None:
        if not piece:
            return
        if callable(on_answer):
            try:
                on_answer(piece)
            except Exception:
                pass

    def _handle_visible_thinking(piece: str) -> None:
        if not piece:
            return
        if callable(on_thinking):
            try:
                on_thinking(piece)
            except Exception:
                pass

    try:
        for chunk in provider.stream(req):
            chunk_model = str(getattr(chunk, "model", "") or "").strip()
            if chunk_model:
                model_name = chunk_model

            tool_calls_delta = list(getattr(chunk, "tool_calls_delta", []) or [])
            if tool_calls_delta:
                for call in tool_calls_delta:
                    if not isinstance(call, ToolCall):
                        continue
                    if any(str(existing.id or "") == str(call.id or "") for existing in tool_calls):
                        continue
                    tool_calls.append(call)

            # Thinking delta from provider
            thinking_delta = str(getattr(chunk, "thinking_delta", "") or "")
            if thinking_delta:
                thinking_parts.append(thinking_delta)
                _handle_visible_thinking(thinking_delta)

            # Answer delta with inline <think> support
            text_delta = str(getattr(chunk, "text_delta", "") or "")
            if text_delta:
                visible, thinking_from_text = parser.feed(text_delta)
                if thinking_from_text:
                    thinking_parts.append(thinking_from_text)
                    _handle_visible_thinking(thinking_from_text)
                if visible:
                    answer_parts.append(visible)
                    _handle_visible_answer(visible)

            if bool(getattr(chunk, "done", False)):
                usage = getattr(chunk, "usage", None) or usage
                timings = getattr(chunk, "timings", None) or timings

        visible, thinking_from_text = parser.flush()
        if thinking_from_text:
            thinking_parts.append(thinking_from_text)
            _handle_visible_thinking(thinking_from_text)
        if visible:
            answer_parts.append(visible)
            _handle_visible_answer(visible)
    except Exception as exc:
        ctx.logs.append(f"stage=generate agent_loop_stream_error={type(exc).__name__}")
        ctx.errors.append(f"agent_loop_stream:{type(exc).__name__}:{exc}")
        raise

    return LLMResponse(
        text="".join(answer_parts).strip(),
        tool_calls=list(tool_calls),
        thinking="".join(thinking_parts).strip(),
        model=model_name,
        usage=usage,
        timings=timings,
    )


def test_agent_loop_streaming_emits_chunks_live_and_preserves_tools():
    """
    Проверить сценарий:
    - provider.stream() отдаёт: answer chunk 1, answer chunk 2, tool_calls_delta, done
    - on_answer вызывается сразу на первых чанках
    - LLMResponse.tool_calls сохраняется
    """
    chunks = [
        LLMChunk(text_delta="При", thinking_delta="", tool_calls_delta=[], done=False),
        LLMChunk(text_delta="вет", thinking_delta="", tool_calls_delta=[], done=False),
        LLMChunk(
            text_delta="",
            thinking_delta="",
            tool_calls_delta=[
                ToolCall(id="call_1", name="memory_retrieve", arguments={"query": "test"})
            ],
            done=False,
        ),
        LLMChunk(text_delta="", thinking_delta="", tool_calls_delta=[], done=True),
    ]

    provider = _FakeProvider(chunks)

    answer_chunks: list[str] = []
    thinking_chunks: list[str] = []

    req = LLMRequest(
        messages=[Message(role="user", content="hello")],
        tools=[ToolSpec(name="memory_retrieve", description="...", input_schema={})],
        metadata={"agent_loop_iteration": 1},
    )

    ctx = SimpleNamespace(logs=[], errors=[])

    resp = _run_agent_loop_streaming(
        ctx,
        provider,
        req,
        on_answer=lambda piece: answer_chunks.append(piece),
        on_thinking=lambda piece: thinking_chunks.append(piece),
    )

    # Проверяем, что чанки пришли сразу
    assert "".join(answer_chunks) == "Привет"
    assert resp.text == "Привет"
    # Проверяем, что tool_calls сохранились
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "memory_retrieve"
    assert resp.tool_calls[0].id == "call_1"
    # Проверяем, что generate не вызывался (streaming работал)
    assert not provider.generate_called


def test_agent_loop_streaming_thinking_live():
    """
    Проверить, что thinking_delta тоже отдаётся сразу.
    """
    chunks = [
        LLMChunk(text_delta="", thinking_delta="Дума", tool_calls_delta=[], done=False),
        LLMChunk(text_delta="", thinking_delta="ю...", tool_calls_delta=[], done=False),
        LLMChunk(text_delta="Ответ", thinking_delta="", tool_calls_delta=[], done=False),
        LLMChunk(text_delta="", thinking_delta="", tool_calls_delta=[], done=True),
    ]

    provider = _FakeProvider(chunks)

    answer_chunks: list[str] = []
    thinking_chunks: list[str] = []

    req = LLMRequest(
        messages=[Message(role="user", content="hello")],
        tools=[],
        metadata={},
    )

    ctx = SimpleNamespace(logs=[], errors=[])

    resp = _run_agent_loop_streaming(
        ctx,
        provider,
        req,
        on_answer=lambda piece: answer_chunks.append(piece),
        on_thinking=lambda piece: thinking_chunks.append(piece),
    )

    # Thinking должен прийти сразу
    assert "".join(thinking_chunks) == "Думаю..."
    assert resp.thinking == "Думаю..."
    # Answer тоже
    assert "".join(answer_chunks) == "Ответ"
    assert resp.text == "Ответ"


def test_think_stream_parser_emits_inline_thinking_without_tag_tail_delay() -> None:
    from core.response_pipeline import _ThinkStreamParser

    parser = _ThinkStreamParser()

    visible, thinking = parser.feed("Hello <think>rea")
    assert visible == "Hello "
    assert thinking == "rea"

    visible, thinking = parser.feed("son")
    assert visible == ""
    assert thinking == "son"


def test_think_stream_parser_does_not_hold_recent_thinking_while_close_tag_is_incomplete() -> None:
    from core.response_pipeline import _ThinkStreamParser

    parser = _ThinkStreamParser()

    visible, thinking = parser.feed("<think>abc</thi")
    assert visible == ""
    assert thinking == "abc"

    visible, thinking = parser.feed("nk>done")
    assert visible == "done"
    assert thinking == ""


def test_agent_loop_streaming_no_probe_delay_with_tools():
    """
    Проверить:
    - при наличии req.tools
    - и agent_loop_iteration == 1
    - первые чанки сразу попадают в callback (без probe-задержки)
    """
    chunks = [
        LLMChunk(text_delta="Пер", thinking_delta="", tool_calls_delta=[], done=False),
        LLMChunk(text_delta="вый", thinking_delta="", tool_calls_delta=[], done=False),
        LLMChunk(
            text_delta="",
            thinking_delta="",
            tool_calls_delta=[
                ToolCall(id="call_2", name="history_read", arguments={"limit": 10})
            ],
            done=False,
        ),
        LLMChunk(text_delta="", thinking_delta="", tool_calls_delta=[], done=True),
    ]

    provider = _FakeProvider(chunks)

    answer_chunks: list[str] = []

    req = LLMRequest(
        messages=[Message(role="user", content="test")],
        tools=[
            ToolSpec(name="memory_retrieve", description="...", input_schema={}),
            ToolSpec(name="history_read", description="...", input_schema={}),
        ],
        metadata={"agent_loop_iteration": 1},
    )

    ctx = SimpleNamespace(logs=[], errors=[])

    resp = _run_agent_loop_streaming(
        ctx,
        provider,
        req,
        on_answer=lambda piece: answer_chunks.append(piece),
        on_thinking=lambda piece: None,
    )

    # Чанки должны прийти сразу, без задержки
    assert "".join(answer_chunks) == "Первый"
    assert resp.text == "Первый"
    # Tools должны сохраниться
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "history_read"


def test_agent_loop_streaming_multiple_tool_calls():
    """
    Проверить, что несколько tool_calls в stream корректно собираются.
    """
    chunks = [
        LLMChunk(text_delta="Тек", thinking_delta="", tool_calls_delta=[], done=False),
        LLMChunk(
            text_delta="",
            thinking_delta="",
            tool_calls_delta=[
                ToolCall(id="call_a", name="memory_retrieve", arguments={"query": "first"})
            ],
            done=False,
        ),
        LLMChunk(
            text_delta="",
            thinking_delta="",
            tool_calls_delta=[
                ToolCall(id="call_b", name="history_read", arguments={"limit": 5})
            ],
            done=False,
        ),
        LLMChunk(text_delta="ст", thinking_delta="", tool_calls_delta=[], done=False),
        LLMChunk(text_delta="", thinking_delta="", tool_calls_delta=[], done=True),
    ]

    provider = _FakeProvider(chunks)

    answer_chunks: list[str] = []

    req = LLMRequest(
        messages=[Message(role="user", content="test")],
        tools=[
            ToolSpec(name="memory_retrieve", description="...", input_schema={}),
            ToolSpec(name="history_read", description="...", input_schema={}),
        ],
        metadata={"agent_loop_iteration": 1},
    )

    ctx = SimpleNamespace(logs=[], errors=[])

    resp = _run_agent_loop_streaming(
        ctx,
        provider,
        req,
        on_answer=lambda piece: answer_chunks.append(piece),
        on_thinking=lambda piece: None,
    )

    # Текст должен прийти
    assert "".join(answer_chunks) == "Текст"
    assert resp.text == "Текст"
    # Оба tool_calls должны сохраниться
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].name == "memory_retrieve"
    assert resp.tool_calls[0].id == "call_a"
    assert resp.tool_calls[1].name == "history_read"
    assert resp.tool_calls[1].id == "call_b"


def test_agent_loop_streaming_empty_text_with_tools():
    """
    Проверить, что streaming работает даже если текст пустой, но есть tools.
    """
    chunks = [
        LLMChunk(
            text_delta="",
            thinking_delta="",
            tool_calls_delta=[
                ToolCall(id="call_only", name="memory_retrieve", arguments={"query": "only"})
            ],
            done=False,
        ),
        LLMChunk(text_delta="", thinking_delta="", tool_calls_delta=[], done=True),
    ]

    provider = _FakeProvider(chunks)

    answer_chunks: list[str] = []

    req = LLMRequest(
        messages=[Message(role="user", content="test")],
        tools=[ToolSpec(name="memory_retrieve", description="...", input_schema={})],
        metadata={"agent_loop_iteration": 1},
    )

    ctx = SimpleNamespace(logs=[], errors=[])

    resp = _run_agent_loop_streaming(
        ctx,
        provider,
        req,
        on_answer=lambda piece: answer_chunks.append(piece),
        on_thinking=lambda piece: None,
    )

    # Текст пустой
    assert "".join(answer_chunks) == ""
    assert resp.text == ""
    # Tool должен сохраниться
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "memory_retrieve"
    assert resp.tool_calls[0].id == "call_only"
