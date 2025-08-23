# app/api/v1/routes/search_routes.py
# FINAL CORRECT VERSION

import time
from typing import List, Optional, Dict

from fastapi import APIRouter, Depends, Body, HTTPException
from pydantic import BaseModel, Field
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from app.agents.s5_search_agent import S5SearchAgent
from app.api.dependencies import get_s5_agent

# WE ARE IMPORTING DIRECTLY FROM THE DOMAIN LAYER.
# THIS IS THE ONLY IMPORT NEEDED FOR OUR SCHEMAS.
from app.domains.search.schemas import (
    SearchQuery, 
    SearchResponse as DomainSearchResponse,
    Platform,
    ContentCategory
)

#==============================================================================
# We define the PUBLIC models for our API response directly in this file.
# This makes the route self-contained and breaks all circular dependencies.
#==============================================================================

class PublicSearchResult(BaseModel):
    """The public-facing representation of a single search result."""
    platform: Platform
    name: str
    handle: str
    profile_url: str
    bio: Optional[str] = None
    follower_count: Optional[int] = None
    is_verified: bool
    categories: List[ContentCategory] = Field(default_factory=list)
    match_confidence: float
    match_reasons: Optional[List[str]] = None

class PublicSearchResponse(BaseModel):
    """The public-facing top-level API response object."""
    query: SearchQuery
    search_duration_ms: float
    total_count: int
    results: List[PublicSearchResult]
    platform_breakdown: Dict[Platform, int] = Field(default_factory=dict)
    category_breakdown: Dict[ContentCategory, int] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)

#==============================================================================

router = APIRouter()

@router.post(
    "/",
    response_model=PublicSearchResponse,
    summary="Perform a Comprehensive Creator Search",
)
async def search_creators(
    query: SearchQuery = Body(...),
    agent: S5SearchAgent = Depends(get_s5_agent),
):
    start_time = time.time()
    
    try:
        domain_response: DomainSearchResponse = await agent.search(
            query=query.query,
            platforms=query.filters.platforms,
            search_type=query.search_type,
            filters=query.filters
        )
        
        public_results = []
        for res in domain_response.results:
            public_results.append(
                PublicSearchResult(
                    platform=res.profile.platform,
                    name=res.profile.name,
                    handle=res.profile.handle,
                    profile_url=str(res.profile.profile_url),
                    bio=res.profile.bio,
                    follower_count=res.profile.followers_count,
                    is_verified=res.profile.is_verified,
                    categories=res.profile.metadata.categories,
                    match_confidence=res.match_confidence,
                    match_reasons=res.match_details.match_reasons if res.match_details else None
                )
            )

        end_time = time.time()

        return PublicSearchResponse(
            query=query,
            search_duration_ms=(end_time - start_time) * 1000,
            total_count=domain_response.total_count,
            results=public_results,
            platform_breakdown=domain_response.platform_breakdown,
            category_breakdown=domain_response.category_breakdown,
            suggestions=domain_response.suggestions
        )

    except Exception as e:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"An unexpected error occurred: {e}"
        )