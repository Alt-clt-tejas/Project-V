# app/config/base.py
from pathlib import Path
from typing import Optional, List
from pydantic import SecretStr, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """
    Unified application settings using pydantic v2.
    Loaded from environment variables with validation.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Basic Application Configuration
    APP_NAME: str = "CreatorSearch Service"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API Keys & Secrets
    YOUTUBE_API_KEY: Optional[SecretStr] = None
    TWITTER_BEARER_TOKEN: Optional[SecretStr] = None

    # Instagram Configuration
    INSTAGRAM_USERNAME: Optional[SecretStr] = None
    INSTAGRAM_PASSWORD: Optional[SecretStr] = None
    INSTAGRAM_SESSION_PATH: str = "./sessions"
    INSTAGRAM_RATE_LIMIT_DELAY: float = 1.5
    INSTAGRAM_MAX_LOGIN_ATTEMPTS: int = 3
    INSTAGRAM_LOGIN_COOLDOWN: int = 300  # seconds

    # Security Settings
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    @validator("INSTAGRAM_SESSION_PATH")
    def ensure_session_directory(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @validator("INSTAGRAM_RATE_LIMIT_DELAY")
    def clamp_rate_limit_delay(cls, v: float) -> float:
        return max(0.5, min(v, 10.0))
