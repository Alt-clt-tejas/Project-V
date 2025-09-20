# app/config/base.py
import os
from pathlib import Path
from typing import Optional, List
from pydantic import SecretStr, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """
    Intelligent, environment-aware application settings.
    It first loads the main .env file to determine the APP_ENV,
    and then loads the specific .env.{APP_ENV} file.
    """
    # This first setting is loaded from the main .env file
    APP_ENV: str = "prod"

    model_config = SettingsConfigDict(
        # This tells pydantic-settings to look for files like .env.prod or .env.dev
        env_file=f".env.{os.getenv('APP_ENV', 'prod')}",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # All settings below will be loaded from the environment-specific file
    # (.env.dev or .env.prod)
    
    # Basic Application Configuration
    APP_NAME: str = "Project Oracle"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str

    # API Keys & Secrets
    YOUTUBE_API_KEY: Optional[SecretStr] = None
    # ... other settings ...

# The AppSettings class needs to be instantiated after we determine the APP_ENV
# We use a small helper function to manage this.
def get_settings() -> AppSettings:
    from dotenv import load_dotenv
    # Load the main .env file first to get the APP_ENV
    load_dotenv(".env")
    return AppSettings()

settings = get_settings()