from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "DocPilot"
    app_version: str = "0.1.0"
    environment: str = "development"
    allowed_origins: list[str] = ["http://localhost:5173"]

    database_url: str = "postgresql+psycopg://docpilot:docpilot@localhost:5432/docpilot"
    primary_model: str = "claude-sonnet-4-20250514"
    fallback_model: str = "gpt-4o-mini"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
