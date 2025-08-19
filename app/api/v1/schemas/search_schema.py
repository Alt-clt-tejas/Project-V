# app/api/v1/schemas/search_schema.py
from typing import List, Optional

from pydantic import BaseModel, Field

from app.domains.search.schemas import Platform


class SearchRequest(BaseModel):
    """
    Defines the structure for an incoming search request to the API.
    """
    query: str = Field(..., min_length=1, max_length=100,
                       description="The search query (name, handle, or keywords).")
    platforms: List[Platform] = Field(
        default=[p for p in Platform if p not in [Platform.NEWS, Platform.WEB]],
        description="A list of platforms to search on."
    )


class SearchResultItemResponse(BaseModel):
    """
    Defines the structure for a single search result item in the API response.
    This is the public-facing data transfer object (DTO).
    """
    platform: Platform
    name: str
    handle: str
    match_confidence: float
    # We can rename fields for the public API, e.g., 'followers_count' -> 'subscribers_count'
    # For simplicity, we keep them the same for now, but this is where you'd do it.
    follower_count: Optional[int] = None
    profile_url: str

    class Config:
        # Pydantic v1: orm_mode = True
        # Pydantic v2:
        from_attributes = True


class SearchResponse(BaseModel):
    """
    Defines the final structure of the API search response.
    """
    query: str
    results: List[SearchResultItemResponse]