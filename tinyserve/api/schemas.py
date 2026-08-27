"""Pydantic request/response models for the HTTP API."""

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=128, gt=0, le=4096)
    stream: bool = False
    priority: int = Field(default=1, ge=1)


class GenerateResponse(BaseModel):
    text: str


class PolicyRequest(BaseModel):
    policy: str = Field(min_length=1)


class ConfigResponse(BaseModel):
    """Live server configuration, surfaced for the demo page (GET /config)."""

    scheduling_policy: str
    n_seq_max: int
    n_ctx: int
    kv_block_size: int
    max_queue_depth: int
    prefix_cache_enabled: bool
    model_path: str
