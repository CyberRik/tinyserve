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
    scheduling_policy: str = "wfq"
    queue_timeout_seconds: float = 30.0
    generation_timeout_seconds: float = 120.0
