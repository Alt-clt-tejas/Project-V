# app/config/base.py
import os
from pathlib import Path
from typing import Optional, List
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pydantic import field_validator

class AppSettings(BaseSettings):
    """
    Application settings with environment variable support.
    """
    # Basic Application Configuration
    APP_ENV: str = "prod"
    APP_NAME: str = "Project Oracle"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/project_oracle"

    # Social Media Credentials
    INSTAGRAM_USERNAME: Optional[str] = None
    INSTAGRAM_PASSWORD: Optional[str] = None
    TWITTER_API_KEY: Optional[SecretStr] = None
    TWITTER_API_SECRET: Optional[SecretStr] = None
    YOUTUBE_API_KEY: Optional[SecretStr] = None
    FACEBOOK_ACCESS_TOKEN: Optional[SecretStr] = None
    LINKEDIN_CLIENT_ID: Optional[SecretStr] = None
    LINKEDIN_CLIENT_SECRET: Optional[SecretStr] = None
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def validate_cors_origins(cls, v):
        # Handle string input (comma-separated)
        if isinstance(v, str):
            if v.startswith("["):
                # If it's a JSON string, parse it
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # Otherwise, split by comma
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    model_config = {
        "env_file": f".env.{os.getenv('APP_ENV', 'prod')}",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore"
    }
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

# Initialize settings
settings = AppSettings()