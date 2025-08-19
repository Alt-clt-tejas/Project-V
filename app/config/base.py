# app/config/base.py
import os
from enum import Enum
from typing import Optional

from pydantic import RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    """Enumeration for application environments."""
    PROD = "prod"
    DEV = "dev"
    TEST = "test"


class AppSettings(BaseSettings):
    """
    Core application settings, loaded from environment variables.
    Utilizes Pydantic's BaseSettings for robust validation and type-hinting.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # --- Application Core ---
    APP_ENV: AppEnv = AppEnv.DEV
    APP_NAME: str = "Project Oracle: Agent S5"
    DEBUG: bool = False
    
    # --- Logging & Monitoring ---
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: Optional[str] = None

    # --- Caching ---
    REDIS_URL: RedisDsn = "redis://localhost:6379/0"

    # --- API Keys & Secrets ---
    # Using SecretStr to prevent accidental exposure in logs or tracebacks.
    YOUTUBE_API_KEY: Optional[SecretStr] = None
    TWITTER_BEARER_TOKEN: Optional[SecretStr] = None
    
    # Add other API keys as needed
    # NEWS_API_KEY: Optional[SecretStr] = None
    # INSTAGRAM_USERNAME: Optional[SecretStr] = None
    # INSTAGRAM_PASSWORD: Optional[SecretStr] = None
    
    @property
    def is_dev(self) -> bool:
        return self.APP_ENV == AppEnv.DEV

    @property
    def is_prod(self) -> bool:
        return self.APP_ENV == AppEnv.PROD

    @property
    def is_test(self) -> bool:
        return self.APP_ENV == AppEnv.TEST


# Singleton instance of the settings
settings = AppSettings()