"""Versioned request and response contracts."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=256)
    max_new_tokens: int = Field(default=200, ge=50, le=500)
    temperature: float = Field(default=0.8, ge=0.5, le=1.5)


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    generated_characters: int
    elapsed_ms: int
    filtered_characters: list[str]


class AttentionRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=256)
    layer: int = Field(default=0, ge=0)
    head: int | None = Field(default=0, ge=0)


class AttentionResponse(BaseModel):
    prompt: str
    labels: list[str]
    layer: int
    head: int | None
    matrices: list[list[list[float]]]
    filtered_characters: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    device: str
    error: str | None = None
