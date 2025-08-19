# app/api/v1/routes/search_routes.py
from typing import List

from fastapi import APIRouter, Depends

from app.agents.s5_search_agent import S5SearchAgent
from app.api.dependencies import get_s5_agent
from app.api.v1.schemas.search_schema import (
    SearchRequest, SearchResponse, SearchResultItemResponse
)
from app.domains.search.schemas import SearchResult


router = APIRouter()


def _transform_results_for_response(
    results: List[SearchResult]
) -> List[SearchResultItemResponse]:
    """Transforms internal SearchResult models to public API response models."""
    response_items = []
    for result in results:
        response_items.append(
            SearchResultItemResponse(
                platform=result.profile.platform,
                name=result.profile.name,
                handle=result.profile.handle,
                match_confidence=result.match_confidence,
                follower_count=result.profile.followers_count,
                profile_url=str(result.profile.profile_url)
            )
        )
    return response_items


@router.post(
    "/",
    response_model=SearchResponse,
    summary="Search for Creators",
    description="Performs a multi-platform search for creators by name or handle."
)
async def search_creators(
    request: SearchRequest,
    agent: S5SearchAgent = Depends(get_s5_agent),
):
    """
    The main search endpoint for Agent S5.
    """
    search_results = await agent.search(query=request.query, platforms=request.platforms)
    
    response_items = _transform_results_for_response(search_results)
    
    return SearchResponse(query=request.query, results=response_items)