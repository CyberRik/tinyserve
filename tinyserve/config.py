"""Runtime configuration, loaded from environment variables (prefix TINYSERVE_)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TINYSERVE_")

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
