from __future__ import annotations

from pydantic import BaseModel, Field


class ChatAttachment(BaseModel):
    kind: str = "file"
    name: str = ""
    mime_type: str = ""
    size: int = 0
    path: str = ""
    text: str | None = None
    data_base64: str | None = None


class ChatRequest(BaseModel):
    text: str = ""
    store_turn: bool = True
    think: bool | None = None
    verbose: bool | None = None
    json_mode: bool | None = None
    attachments: list[ChatAttachment] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    thinking: str = ""
    stats: dict = {}
    model: str
    parameters: dict | None = None
    summary: str | None = None
    debug_trace: dict | None = None
    memory_debug_snapshot: dict | None = None


class MemoryInspectorResponse(BaseModel):
    conversation_id: str
    request_id: str = ""
    debug_trace: dict | None = None
    memory_debug_snapshot: dict | None = None
    memory_store_debug: dict | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    thinking_enabled: bool
    verbose_enabled: bool
    web_mode: str
    persona_name: str = ""
    json_mode_enabled: bool
    active_profile: str
    quality_profile: str
    profile_parameters: dict
    model_status: dict = Field(default_factory=dict)
    memory_status: dict = Field(default_factory=dict)
    server_resources: dict = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    runtime_model: str
    models: list[str]


class ModelSetRequest(BaseModel):
    model: str = Field(min_length=1)


class ThinkingRequest(BaseModel):
    enabled: bool

class VerboseRequest(BaseModel):
    enabled: bool

class WebModeRequest(BaseModel):
    mode: str

class JsonModeRequest(BaseModel):
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
