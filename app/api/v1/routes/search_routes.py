# app/api/v1/routes/search_routes.py
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, Body, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from app.agents.s5_search_agent import S5SearchAgent
from app.api.dependencies import get_s5_agent
# Import the definitive models directly from the domain
from app.domains.search.schemas import (
    SearchQuery,
    SearchResponse,  # This is what we'll use instead of DomainSearchResponse
    SearchResult,
    Platform,
    SearchType,
    SearchFilter,
    ContentCategory
)

router = APIRouter()

# --- Public-Facing API Models (DTOs) ---
# This creates a stable public contract, independent of our internal domain models.

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
    suggestions: Optional[List[str]] = None

# --- API Endpoint ---

@router.post(
    "/",
    response_model=PublicSearchResponse,
    summary="Perform a Comprehensive Creator Search",
)
async def search_creators(
    query: SearchQuery = Body(...),
    agent: S5SearchAgent = Depends(get_s5_agent),
):
    """
    Accepts a complex SearchQuery object for powerful filtering, sorting, and discovery.
    Returns a lean, public-facing response.
    """
    try:
        start_time = time.time()
        
        # Fix: Call the agent.search method with the correct parameters
        # Based on your S5SearchAgent.search method signature:
        domain_response: SearchResponse = await agent.search(
            query=query.query,
            platforms=query.filters.platforms if query.filters and query.filters.platforms else [Platform.YOUTUBE],
            search_type=query.search_type,
            filters=query.filters,
            use_cache=True,
            timeout_override=None
        )
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Transform the rich internal SearchResult into the lean PublicSearchResult
        public_results = []
        for result in domain_response.results:
            # Fix: Access profile properties correctly
            profile = result.profile
            
            # Handle match reasons - check if match_details exists
            reasons = None
            if hasattr(result, 'match_details') and result.match_details:
                if hasattr(result.match_details, 'match_reasons'):
                    reasons = result.match_details.match_reasons
            
            # Access follower count from social_metrics
            follower_count = None
            if hasattr(profile, 'social_metrics') and profile.social_metrics:
                follower_count = profile.social_metrics.followers_count
            elif hasattr(profile, 'followers_count'):
                follower_count = profile.followers_count
            
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
        # Last-resort error handler
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"An unexpected error occurred in the search agent: {e}"
        )


# Alternative endpoint that accepts individual parameters instead of SearchQuery object
@router.post(
    "/simple",
    response_model=PublicSearchResponse,
    summary="Simple Creator Search",
)
async def simple_search(
    query: str = Body(..., embed=True),
    platforms: Optional[List[Platform]] = Body(default=[Platform.YOUTUBE]),
    search_type: SearchType = Body(default=SearchType.CREATOR),
    min_followers: Optional[int] = Body(default=None),
    verified_only: bool = Body(default=False),
    agent: S5SearchAgent = Depends(get_s5_agent),
):
    """
    Simple search endpoint with individual parameters.
    """
    try:
        # Create SearchFilter from individual parameters
        search_filter = SearchFilter(
            platforms=platforms,
            min_followers=min_followers,
            verified_only=verified_only,
            active_only=True
        )
        
        # Call agent search method
        domain_response: SearchResponse = await agent.search(
            query=query,
            platforms=platforms or [Platform.YOUTUBE],
            search_type=search_type,
            filters=search_filter,
            use_cache=True
        )
        
        # Transform results
        public_results = []
        for result in domain_response.results:
            profile = result.profile
            
            follower_count = None
            if hasattr(profile, 'social_metrics') and profile.social_metrics:
                follower_count = profile.social_metrics.followers_count
            
            public_results.append(
                PublicSearchResult(
                    platform=profile.platform,
                    name=profile.name,
                    handle=profile.handle,
                    match_confidence=result.match_confidence,
                    follower_count=follower_count,
                    profile_url=str(profile.profile_url),
                    match_reasons=None
                )
            )

        return PublicSearchResponse(
            results=public_results,
            total_count=domain_response.total_count,
            search_duration_ms=domain_response.search_duration_ms,
            suggestions=domain_response.suggestions
        )

    except Exception as e:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search failed: {e}"
        )