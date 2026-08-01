"""Pydantic request/response models for the HTTP API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=128, gt=0, le=4096)
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
