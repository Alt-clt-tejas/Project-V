# app/api/search.py
import logging
import time
from typing import List
from fastapi import APIRouter, Depends, Body, HTTPException

from app.agents.s5_search_agent import S5SearchAgent
from app.api.dependencies import get_agent
# Import the necessary schemas
from app.domains.search.schemas import SearchQuery, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Note the trailing slash to prevent the 307 redirect
@router.post("/search/", response_model=SearchResponse)
async def search_creators(
    query: SearchQuery = Body(...),
    agent: S5SearchAgent = Depends(get_agent),
):
    """
    Performs a comprehensive creator search and returns a ranked list.
    
    The endpoint leverages the S5SearchAgent's built-in response builder
    to return complete SearchResponse objects with nested profile data.
    
    Args:
        query (SearchQuery): Contains search parameters including query string,
                           search type, filters, and limits
        agent (S5SearchAgent): Injected search agent dependency
    
    Returns:
        SearchResponse: Complete response object with nested creator profiles
        
    Raises:
        HTTPException: If search operation fails
    """
    try:
        start_time = time.time()

        # Execute search through the agent
        search_results = await agent.search(
            query=query.query,
            search_type=query.search_type,
            filters=query.filters,
            limit=query.limit
        )
        
        # Calculate search duration
        duration_ms = (time.time() - start_time) * 1000

        # Create platform and category breakdowns
        platform_counts = {}
        category_counts = {}
        
        # Process results and build breakdowns
        for result in search_results:
            # Update platform counts
            platform = result.profile.platform
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
            
            # Update category counts for each category in the profile
            for category in result.profile.metadata.categories:
                category_counts[category] = category_counts.get(category, 0) + 1

        # Construct the complete SearchResponse object
        response = SearchResponse(
            results=search_results,
            total_count=len(search_results),
            page_count=1,  # We'll implement pagination later
            current_page=1,
            query=query,
            search_duration_ms=duration_ms,
            platform_breakdown=platform_counts,
            category_breakdown=category_counts,
            suggestions=[],  # We'll implement suggestions later
            related_queries=[]  # We'll implement related queries later
        )
        
        return response
    except Exception as e:
        logger.error(f"Critical error in search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during search.")