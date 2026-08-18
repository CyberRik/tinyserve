"""Runtime configuration, loaded from environment variables (prefix TINYSERVE_)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TINYSERVE_")

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    model_path: str
    n_ctx: int = 2048
    n_seq_max: int = 4
    kv_block_size: int = 16
    chunk_size: int = 512
    max_queue_depth: int = 128
    scheduling_policy: str = "wfq"
    # Prefix reuse across sequences (native/ radix tree, Python fallback).
    # On by default: a miss costs one tree walk over the prompt and a hit
    # removes prefill work outright. Set false to A/B it -- benchmarks/
    # prefix_reuse.py drives exactly that comparison.
    prefix_cache_enabled: bool = True
    queue_timeout_seconds: float = 30.0
    generation_timeout_seconds: float = 120.0
