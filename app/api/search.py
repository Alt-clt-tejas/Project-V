# app/api/search.py
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, Body, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from app.agents.s5_search_agent import S5SearchAgent
from app.api.dependencies import get_agent
from app.domains.search.schemas import (
    SearchQuery,
    SearchResponse,
    Platform,
    SearchType,
    SearchFilter
)

# Use a distinct name for the router in this module
router = APIRouter(prefix="/search", tags=["Search"])

# --- Public-Facing API Models (DTOs) ---
class PublicSearchResult(BaseModel):
    """The lean, public representation of a search result."""
    platform: Platform
    name: str
    handle: str
    match_confidence: float
    follower_count: Optional[int] = None
    profile_url: str
    match_reasons: Optional[List[str]] = None

class PublicSearchResponse(BaseModel):
    """The lean, public representation of a full search response."""
    results: List[PublicSearchResult]
    total_count: int
    search_duration_ms: float
    suggestions: List[str]

# --- API Endpoints ---
@router.post(
    "/",
    response_model=PublicSearchResponse,
    summary="Perform a Comprehensive Creator Search",
)
async def comprehensive_search(
    query: SearchQuery = Body(...),
    # --- START OF FIX ---
    # Use the correct dependency injection function
    agent: S5SearchAgent = Depends(get_agent),
    # --- END OF FIX ---
):
    """
    Accepts a complex SearchQuery object for powerful filtering, sorting, and discovery.
    Returns a lean, public-facing response.
    """
    try:
        start_time = time.time()
        
        domain_response: SearchResponse = await agent.search(
            query=query.query,
            platforms=query.filters.platforms if query.filters and query.filters.platforms else list(Platform),
            search_type=query.search_type,
            filters=query.filters
        )
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        public_results = []
        for result in domain_response.results:
            profile = result.profile
            reasons = result.match_details.match_reasons if result.match_details else None
            follower_count = profile.social_metrics.followers_count if profile.social_metrics else None
            
            public_results.append(
                PublicSearchResult(
                    platform=profile.platform,
                    name=profile.name,
                    handle=profile.handle,
                    match_confidence=result.match_confidence,
                    follower_count=follower_count,
                    profile_url=str(profile.profile_url),
                    match_reasons=reasons
                )
            )

        return PublicSearchResponse(
            results=public_results,
            total_count=domain_response.total_count,
            search_duration_ms=duration_ms,
            suggestions=domain_response.suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred in the search agent: {e}")

@router.post(
    "/simple",
    response_model=PublicSearchResponse,
    summary="Simple Creator Search",
)
async def simple_search(
    query: str = Body(..., embed=True),
    platforms: Optional[List[Platform]] = Body(default=None),
    search_type: SearchType = Body(default=SearchType.CREATOR),
    min_followers: Optional[int] = Body(default=None),
    verified_only: bool = Body(default=False),
    agent: S5SearchAgent = Depends(get_agent),
):
    """
    A simplified search endpoint that accepts individual parameters.
    """
    try:
        search_filter = SearchFilter(
            platforms=platforms,
            min_followers=min_followers,
            verified_only=verified_only,
            active_only=True
        )
        
        domain_response: SearchResponse = await agent.search(
            query=query,
            platforms=platforms or list(Platform),
            search_type=search_type,
            filters=search_filter
        )
        
        public_results = []
        for result in domain_response.results:
            profile = result.profile
            reasons = result.match_details.match_reasons if result.match_details else None
            follower_count = profile.social_metrics.followers_count if profile.social_metrics else None
            
            public_results.append(
                PublicSearchResult(
                    platform=profile.platform,
                    name=profile.name,
                    handle=profile.handle,
                    match_confidence=result.match_confidence,
                    follower_count=follower_count,
                    profile_url=str(profile.profile_url),
                    match_reasons=reasons
                )
            )

        return PublicSearchResponse(
            results=public_results,
            total_count=domain_response.total_count,
            search_duration_ms=domain_response.search_duration_ms,
            suggestions=domain_response.suggestions
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")