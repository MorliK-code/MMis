from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    text: str = Field(min_length=1)
    store_turn: bool = True
    think: bool | None = None


class ChatResponse(BaseModel):
    answer: str
    thinking: str = ""
    stats: dict = {}
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    thinking_enabled: bool


class ModelsResponse(BaseModel):
    runtime_model: str
    models: list[str]


class ModelSetRequest(BaseModel):
    model: str = Field(min_length=1)


class ThinkingRequest(BaseModel):
    enabled: bool


class FeedbackRequest(BaseModel):
    user_text: str = Field(min_length=1)
    assistant_text: str = Field(min_length=1)
    feedback: int
    penalty: float = 0.2


class MetadataResponse(BaseModel):
    model: str
    count: int
    items: list[dict]

