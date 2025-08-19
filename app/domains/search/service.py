# app/domains/search/service.py
from typing import List

from rapidfuzz import fuzz

from app.domains.search.schemas import CreatorProfile, SearchResult


class SearchService:
    """
    Contains the core business logic for the search domain.
    This includes scoring, filtering, and aggregating results.
    """

    def score_and_sort_results(
        self, query: str, profiles: List[CreatorProfile]
    ) -> List[SearchResult]:
        """
        Scores creator profiles based on their relevance to the search query.

        Args:
            query: The original search query.
            profiles: A list of CreatorProfile objects from various connectors.

        Returns:
            A list of SearchResult objects, sorted by match_confidence.
        """
        if not profiles:
            return []

        scored_results = []
        for profile in profiles:
            # Use a weighted ratio to better handle word order and partial matches.
            name_score = fuzz.WRatio(query, profile.name)
            handle_score = fuzz.WRatio(query, profile.handle)
            
            # Take the higher of the two scores and normalize to a 0.0-1.0 scale.
            confidence = max(name_score, handle_score) / 100.0

            scored_results.append(
                SearchResult(profile=profile, match_confidence=confidence)
            )

        # Sort results descending by confidence score
        scored_results.sort(key=lambda r: r.match_confidence, reverse=True)
        return scored_results