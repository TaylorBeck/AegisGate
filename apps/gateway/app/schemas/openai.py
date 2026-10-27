"""OpenAI-compatible chat completion request models."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ContentPart(BaseModel):
    """One element of an array-style message `content` field."""

    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    """Chat message. `content` is a string or a list of parts."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Chat completion body. Unknown OpenAI parameters pass through."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
