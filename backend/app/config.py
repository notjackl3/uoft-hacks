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
        openai_api_key: str  # Changed from gemini_api_key
        voyage_api_key: str
        environment: str = "development"
        log_level: str = "INFO"
        
        # Rate limiting settings
        llm_min_delay: float = 0.2   # Min seconds between LLM calls (lower = faster, higher = safer)
        llm_max_retries: int = 3     # Max retries on rate limit
        llm_initial_backoff: float = 2.0   # Initial backoff seconds
        llm_max_backoff: float = 30.0      # Max backoff seconds

        model_config = SettingsConfigDict(env_file=".env", extra="ignore")

else:

    class Settings:  # minimal fallback (keeps tests/imports working without pydantic-settings)
        def __init__(self) -> None:
            self.mongodb_uri = os.getenv("MONGODB_URI", "")
            self.mongodb_db_name = os.getenv("MONGODB_DB_NAME", "tutor_agent")
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")  # Changed from GEMINI_API_KEY
            self.voyage_api_key = os.getenv("VOYAGE_API_KEY", "")
            self.environment = os.getenv("ENVIRONMENT", "development")
            self.log_level = os.getenv("LOG_LEVEL", "INFO")
            
            # Rate limiting settings
            self.llm_min_delay = float(os.getenv("LLM_MIN_DELAY", "0.2"))
            self.llm_max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
            self.llm_initial_backoff = float(os.getenv("LLM_INITIAL_BACKOFF", "2.0"))
            self.llm_max_backoff = float(os.getenv("LLM_MAX_BACKOFF", "30.0"))


settings = Settings()
