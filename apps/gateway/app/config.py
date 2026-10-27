"""Process-wide settings loaded from environment / `.env`."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway configuration. Secret values are never logged."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    redis_url: str = "redis://127.0.0.1:6379/0"
    database_url: str = "postgresql://aegis:aegis@127.0.0.1:5432/aegis"
    openai_api_key: SecretStr = SecretStr("")
    openai_compat_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: SecretStr = SecretStr("")
    aegis_virtual_keys: SecretStr = SecretStr("sandbox:sk-aegis-sandbox")
    aegis_cors_origins: str = "http://localhost:3000"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_threads: int = 4
    embed_max_chars: int = 1024
    cache_ttl_seconds: int = 3600
    rate_limit_capacity: float = 10000.0
    rate_limit_refill_per_sec: float = 10000.0
    router_margin: float = 0.03
    aegis_catalog: str = ""
    fastembed_cache_path: str | None = None
    redis_hot_max_connections: int = 64
    redis_blocking_max_connections: int = 16
    pg_min_size: int = 2
    pg_max_size: int = 10
    http_max_keepalive_connections: int = Field(default=100)
    http_max_connections: int = Field(default=500)
