# app/domains/search/schemas.py
from enum import Enum
from typing import Optional

from pydantic import BaseModel, HttpUrl, Field


class Platform(str, Enum):
    """Enumeration of supported social media platforms."""
    YOUTUBE = "YouTube"
    TWITTER = "Twitter"
    INSTAGRAM = "Instagram"
    TIKTOK = "TikTok"
    NEWS = "News"
    WEB = "Web"


class CreatorProfile(BaseModel):
    """
    Standardized internal representation of a creator's profile
    from any platform. This is the pure domain model.
    """
    platform: Platform
    name: str
    handle: str
    profile_url: HttpUrl
    bio: Optional[str] = None
    followers_count: Optional[int] = Field(None, ge=0)
    is_verified: bool = False


class SearchResult(BaseModel):
    """
    Represents a creator profile found during a search, augmented with
    our internal scoring.
    """
    profile: CreatorProfile
    match_confidence: float = Field(..., ge=0.0, le=1.0)