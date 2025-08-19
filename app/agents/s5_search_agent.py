# app/agents/s5_search_agent.py
import asyncio
from typing import List, Dict

from app.connectors.base_connector import BaseConnector
from app.domains.search.schemas import Platform, SearchResult
from app.domains.search.service import SearchService


class S5SearchAgent:
    """
    Agent S5: The Search Agent.
    Orchestrates the search process across multiple platforms.
    """
    def __init__(
        self,
        connectors: Dict[Platform, BaseConnector],
        search_service: SearchService
    ):
        self.connectors = connectors
        self.search_service = search_service

    async def search(self, query: str, platforms: List[Platform]) -> List[SearchResult]:
        """
        Asynchronously searches for a query across the specified platforms.

        1. Checks cache (TODO).
        2. Gathers results from connectors in parallel.
        3. Scores and sorts results using the SearchService.
        4. Updates cache (TODO).

        Args:
            query: The search term.
            platforms: A list of platforms to search on.

        Returns:
            A list of scored and sorted search results.
        """
        # TODO: Implement caching layer check here.

        connectors_to_run = [
            self.connectors[p] for p in platforms if p in self.connectors
        ]
        
        if not connectors_to_run:
            return []

        # Create and run search tasks concurrently
        tasks = [conn.search(query) for conn in connectors_to_run]
        results_from_connectors = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten the list of lists and filter out any exceptions
        all_profiles = []
        for res in results_from_connectors:
            if isinstance(res, list):
                all_profiles.extend(res)
            else:
                # TODO: Log the exception properly
                print(f"An error occurred in a connector: {res}")
        
        # Pass the aggregated profiles to the service layer for scoring
        scored_results = self.search_service.score_and_sort_results(query, all_profiles)

        # TODO: Implement cache update here.
        
        return scored_results