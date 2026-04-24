from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "DocPilot"
    app_version: str = "0.1.0"
    environment: str = "development"
    allowed_origins: list[str] = ["http://localhost:5173"]

    database_url: str = "postgresql+psycopg://docpilot:docpilot@localhost:5432/docpilot"
    primary_model: str = "gemini/gemini-2.0-flash"
    fallback_model: str = "gemini/gemini-1.5-pro-latest"
    allowed_models: list[str] = [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-1.5-pro-latest",
        "claude-sonnet-4-20250514",
        "gpt-4o-mini",
    ]

    # LLM provider keys (loaded from environment/.env)
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
