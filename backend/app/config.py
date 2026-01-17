from __future__ import annotations

import os

try:  # pragma: no cover
    from pydantic_settings import BaseSettings, SettingsConfigDict
except Exception:  # pragma: no cover
    BaseSettings = None  # type: ignore[assignment]
    SettingsConfigDict = None  # type: ignore[assignment]


def _load_dotenv_if_available() -> None:
    try:  # pragma: no cover
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


_load_dotenv_if_available()


if BaseSettings is not None:

    class Settings(BaseSettings):
        mongodb_uri: str
        mongodb_db_name: str
        gemini_api_key: str
        voyage_api_key: str
        environment: str = "development"
        log_level: str = "INFO"

        model_config = SettingsConfigDict(env_file=".env", extra="ignore")

else:

    class Settings:  # minimal fallback (keeps tests/imports working without pydantic-settings)
        def __init__(self) -> None:
            self.mongodb_uri = os.getenv("MONGODB_URI", "")
            self.mongodb_db_name = os.getenv("MONGODB_DB_NAME", "tutor_agent")
            self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
            self.voyage_api_key = os.getenv("VOYAGE_API_KEY", "")
            self.environment = os.getenv("ENVIRONMENT", "development")
            self.log_level = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()

