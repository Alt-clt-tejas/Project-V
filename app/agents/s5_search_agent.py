# app/agents/s5_search_agent.py
import asyncio
import logging
import time
from typing import List, Dict, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from app.connectors.base_connector import BaseConnector
from app.domains.search.schemas import (
    Platform, SearchResult, SearchType, SearchQuery, SearchResponse, 
    SearchFilter, CreatorProfile, ContentCategory
)
from app.domains.search.service import SearchService


logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
    """Metrics for search performance monitoring."""
    query: str
    search_type: SearchType
    platforms_searched: List[Platform]
    total_duration_ms: float = 0.0
    connector_durations: Dict[Platform, float] = field(default_factory=dict)
    scoring_duration_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    total_profiles_found: int = 0
    profiles_per_platform: Dict[Platform, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: List[CreatorProfile]
    timestamp: datetime
    query: str
    search_type: SearchType
    platforms: Set[Platform]
    ttl_seconds: int = 300  # 5 minutes default


class SearchCache:
    """In-memory search cache with TTL support."""
    
    def __init__(self, default_ttl_seconds: int = 300, max_entries: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl_seconds
        self.max_entries = max_entries
        self._lock = asyncio.Lock()
    
    def _generate_cache_key(
        self, 
        query: str, 
        platforms: Set[Platform], 
        search_type: SearchType,
        filters: Optional[SearchFilter] = None
    ) -> str:
        """Generate a unique cache key for the search parameters."""
        platforms_str = "_".join(sorted(p.value for p in platforms))
        
        # Include relevant filter parameters in cache key
        filter_parts = []
        if filters:
            if filters.categories:
                filter_parts.append(f"cat_{'_'.join(c.value for c in filters.categories)}")
            if filters.min_followers:
                filter_parts.append(f"minf_{filters.min_followers}")
            if filters.max_followers:
                filter_parts.append(f"maxf_{filters.max_followers}")
            if filters.verified_only:
                filter_parts.append("verified")
        
        filter_str = "_".join(filter_parts)
        
        # Normalize query for consistent caching
        normalized_query = query.lower().strip()
        
        key_parts = [normalized_query, platforms_str, search_type.value]
        if filter_str:
            key_parts.append(filter_str)
        
        return "|".join(key_parts)
    
    async def get(
        self, 
        query: str, 
        platforms: Set[Platform], 
        search_type: SearchType,
        filters: Optional[SearchFilter] = None
    ) -> Optional[List[CreatorProfile]]:
        """Retrieve cached results if available and not expired."""
        cache_key = self._generate_cache_key(query, platforms, search_type, filters)
        
        async with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                return None
            
            # Check if entry has expired
            age = (datetime.now() - entry.timestamp).total_seconds()
            if age > entry.ttl_seconds:
                del self._cache[cache_key]
                return None
            
            logger.debug(f"Cache hit for query: {query} (age: {age:.1f}s)")
            return entry.data
    
    async def set(
        self, 
        query: str, 
        platforms: Set[Platform], 
        search_type: SearchType,
        data: List[CreatorProfile],
        ttl_seconds: Optional[int] = None,
        filters: Optional[SearchFilter] = None
    ) -> None:
        """Store results in cache."""
        cache_key = self._generate_cache_key(query, platforms, search_type, filters)
        ttl = ttl_seconds or self.default_ttl
        
        async with self._lock:
            # Clean up expired entries if cache is getting full
            if len(self._cache) >= self.max_entries:
                await self._cleanup_expired()
            
            # If still full, remove oldest entries
            if len(self._cache) >= self.max_entries:
                oldest_keys = sorted(
                    self._cache.keys(), 
                    key=lambda k: self._cache[k].timestamp
                )[:self.max_entries // 10]  # Remove 10% of entries
                
                for key in oldest_keys:
                    del self._cache[key]
            
            self._cache[cache_key] = CacheEntry(
                data=data,
                timestamp=datetime.now(),
                query=query,
                search_type=search_type,
                platforms=platforms,
                ttl_seconds=ttl
            )
            
            logger.debug(f"Cache set for query: {query} ({len(data)} profiles)")
    
    async def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self._cache.items():
            age = (now - entry.timestamp).total_seconds()
            if age > entry.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            logger.info("Search cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self._cache),
            "max_entries": self.max_entries,
            "default_ttl": self.default_ttl,
            "memory_usage_estimate": len(self._cache) * 1024  # Rough estimate
        }


class S5SearchAgent:
    """
    Enhanced Agent S5: The Search Agent.
    Orchestrates sophisticated search processes across multiple platforms
    with advanced caching, error handling, and result aggregation.
    """

    def __init__(
        self,
        connectors: Dict[Platform, BaseConnector],
        search_service: SearchService,
        cache_ttl_seconds: int = 300,
        max_cache_entries: int = 1000,
        timeout_seconds: float = 30.0
    ):
        self.connectors = connectors
        self.search_service = search_service
        self.cache = SearchCache(cache_ttl_seconds, max_cache_entries)
        self.timeout_seconds = timeout_seconds
        
        # Performance tracking
        self._search_count = 0
        self._total_search_time = 0.0
        self._platform_performance: Dict[Platform, List[float]] = defaultdict(list)

    async def search(
        self,
        query: str,
        platforms: List[Platform],
        search_type: SearchType = SearchType.CREATOR,
        filters: Optional[SearchFilter] = None,
        use_cache: bool = True,
        timeout_override: Optional[float] = None
    ) -> SearchResponse:
        """
        Comprehensive search across multiple platforms with advanced features.

        Args:
            query: The search term
            platforms: List of platforms to search
            search_type: Type of search to perform
            filters: Advanced filtering options
            use_cache: Whether to use caching
            timeout_override: Override default timeout

        Returns:
            Complete search response with results and metadata
        """
        start_time = time.time()
        search_metrics = SearchMetrics(
            query=query,
            search_type=search_type,
            platforms_searched=platforms
        )
        
        # Create SearchQuery object for consistent handling
        search_query = SearchQuery(
            query=query,
            search_type=search_type,
            filters=filters or SearchFilter()
        )
        
        try:
            # Step 1: Check cache
            cached_profiles = None
            if use_cache:
                platforms_set = set(platforms)
                cached_profiles = await self.cache.get(
                    query, platforms_set, search_type, filters
                )
                
                if cached_profiles is not None:
                    search_metrics.cache_hits = 1
                    logger.info(f"Cache hit for query: {query}")
                else:
                    search_metrics.cache_misses = 1

            # Step 2: Gather results from connectors if not cached
            all_profiles = []
            if cached_profiles is not None:
                all_profiles = cached_profiles
            else:
                all_profiles = await self._gather_connector_results(
                    query, platforms, search_metrics, timeout_override
                )
                
                # Cache the raw results
                if use_cache and all_profiles:
                    await self.cache.set(
                        query, set(platforms), search_type, 
                        all_profiles, filters=filters
                    )

            # Step 3: Apply filters
            filtered_profiles = self._apply_filters(all_profiles, filters)
            search_metrics.total_profiles_found = len(filtered_profiles)

            # Step 4: Score and rank results
            scoring_start = time.time()
            scored_results = await self._score_and_rank_results(
                query, filtered_profiles, search_type, search_metrics
            )
            search_metrics.scoring_duration_ms = (time.time() - scoring_start) * 1000

            # Step 5: Build response
            search_metrics.total_duration_ms = (time.time() - start_time) * 1000
            response = self._build_search_response(
                scored_results, search_query, search_metrics
            )

            # Update performance tracking
            self._update_performance_metrics(search_metrics)
            
            logger.info(
                f"Search completed: {query} -> {len(scored_results)} results "
                f"in {search_metrics.total_duration_ms:.1f}ms"
            )

            return response

        except Exception as e:
            search_metrics.errors.append(str(e))
            search_metrics.total_duration_ms = (time.time() - start_time) * 1000
            
            logger.error(f"Search failed for query '{query}': {e}")
            
            # Return empty response with error information
            return SearchResponse(
                results=[],
                total_count=0,
                page_count=0,
                current_page=1,
                query=search_query,
                search_duration_ms=search_metrics.total_duration_ms,
                suggestions=self.search_service.get_suggestions(query, [], max_suggestions=3)
            )

    async def _gather_connector_results(
        self,
        query: str,
        platforms: List[Platform],
        metrics: SearchMetrics,
        timeout_override: Optional[float]
    ) -> List[CreatorProfile]:
        """Gather results from all relevant connectors concurrently."""
        available_connectors = [
            (platform, self.connectors[platform])
            for platform in platforms 
            if platform in self.connectors
        ]

        if not available_connectors:
            logger.warning(f"No connectors available for platforms: {platforms}")
            return []

        # Create search tasks with timeout
        timeout = timeout_override or self.timeout_seconds
        tasks = []
        
        for platform, connector in available_connectors:
            task = asyncio.create_task(
                self._search_with_timeout(connector, query, platform, timeout)
            )
            tasks.append((platform, task))

        # Execute searches concurrently
        all_profiles = []
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Process results and track performance
        for (platform, _), result in zip(tasks, results):
            connector_start = time.time()
            
            if isinstance(result, Exception):
                error_msg = f"Connector {platform.value} failed: {result}"
                metrics.errors.append(error_msg)
                logger.error(error_msg)
            elif isinstance(result, tuple):  # (profiles, duration)
                profiles, duration = result
                all_profiles.extend(profiles)
                metrics.connector_durations[platform] = duration
                metrics.profiles_per_platform[platform] = len(profiles)
                
                logger.debug(
                    f"{platform.value}: {len(profiles)} profiles in {duration:.1f}ms"
                )
            else:
                logger.warning(f"Unexpected result type from {platform.value}: {type(result)}")

        return all_profiles

    async def _search_with_timeout(
        self,
        connector: BaseConnector,
        query: str,
        platform: Platform,
        timeout: float
    ) -> Tuple[List[CreatorProfile], float]:
        """Execute a connector search with timeout and performance tracking."""
        start_time = time.time()
        
        try:
            profiles = await asyncio.wait_for(
                connector.search(query),
                timeout=timeout
            )
            duration_ms = (time.time() - start_time) * 1000
            
            # Track platform performance
            self._platform_performance[platform].append(duration_ms)
            
            return profiles, duration_ms
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"{platform.value} connector timed out after {timeout}s")
            raise TimeoutError(f"{platform.value} search timed out")
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"{platform.value} connector error: {e}")
            raise

    def _apply_filters(
        self, 
        profiles: List[CreatorProfile], 
        filters: Optional[SearchFilter]
    ) -> List[CreatorProfile]:
        """Apply search filters to profiles."""
        if not filters:
            return profiles

        filtered_profiles = []
        
        for profile in profiles:
            # Platform filter
            if filters.platforms and profile.platform not in filters.platforms:
                continue
            
            # Category filter
            if filters.categories and profile.metadata.categories:
                if not any(cat in filters.categories for cat in profile.metadata.categories):
                    continue
            
            # Follower count filters
            follower_count = profile.social_metrics.followers_count
            if follower_count is not None:
                if filters.min_followers and follower_count < filters.min_followers:
                    continue
                if filters.max_followers and follower_count > filters.max_followers:
                    continue
            
            # Engagement rate filters
            engagement_rate = profile.engagement_metrics.engagement_rate
            if engagement_rate is not None:
                if filters.min_engagement_rate and engagement_rate < filters.min_engagement_rate:
                    continue
                if filters.max_engagement_rate and engagement_rate > filters.max_engagement_rate:
                    continue
            
            # Verification filter
            if filters.verified_only and not profile.is_verified:
                continue
            
            # Activity filter
            if filters.active_only and not profile.is_active:
                continue
            
            # Country filter
            if filters.countries and profile.metadata.country:
                if profile.metadata.country not in filters.countries:
                    continue
            
            # Language filter
            if filters.languages and profile.metadata.languages:
                if not any(lang in filters.languages for lang in profile.metadata.languages):
                    continue
            
            # Time-based filters
            if filters.created_after and profile.created_at:
                if profile.created_at < filters.created_after:
                    continue
            
            if filters.created_before and profile.created_at:
                if profile.created_at > filters.created_before:
                    continue
            
            if filters.last_active_after and profile.engagement_metrics.last_activity:
                if profile.engagement_metrics.last_activity < filters.last_active_after:
                    continue
            
            filtered_profiles.append(profile)

        logger.debug(f"Filtered {len(profiles)} -> {len(filtered_profiles)} profiles")
        return filtered_profiles

    async def _score_and_rank_results(
        self,
        query: str,
        profiles: List[CreatorProfile],
        search_type: SearchType,
        metrics: SearchMetrics
    ) -> List[SearchResult]:
        """Score and rank search results."""
        if not profiles:
            return []

        try:
            # Use the enhanced search service for scoring
            scored_results = self.search_service.score_results(query, profiles, search_type)
            
            # Add ranking positions
            for i, result in enumerate(scored_results, 1):
                result.ranking_position = i
                result.search_query = query
                result.search_type = search_type
            
            return scored_results
            
        except Exception as e:
            metrics.errors.append(f"Scoring failed: {e}")
            logger.error(f"Failed to score results: {e}")
            
            # Return unscored results as fallback
            return [
                SearchResult(
                    profile=profile,
                    match_confidence=0.5,  # Default confidence
                    search_query=query,
                    search_type=search_type,
                    ranking_position=i
                )
                for i, profile in enumerate(profiles, 1)
            ]

    def _build_search_response(
        self,
        results: List[SearchResult],
        query: SearchQuery,
        metrics: SearchMetrics
    ) -> SearchResponse:
        """Build comprehensive search response."""
        # Calculate platform and category breakdowns
        platform_breakdown = defaultdict(int)
        category_breakdown = defaultdict(int)
        
        for result in results:
            platform_breakdown[result.profile.platform] += 1
            
            for category in result.profile.metadata.categories:
                category_breakdown[category] += 1

        # Generate suggestions
        all_profiles = [result.profile for result in results]
        suggestions = self.search_service.get_suggestions(
            query.query, all_profiles, max_suggestions=5
        )

        return SearchResponse(
            results=results,
            total_count=len(results),
            page_count=max(1, (len(results) + query.limit - 1) // query.limit),
            current_page=max(1, query.offset // query.limit + 1),
            query=query,
            search_duration_ms=metrics.total_duration_ms,
            platform_breakdown=dict(platform_breakdown),
            category_breakdown=dict(category_breakdown),
            suggestions=suggestions
        )

    def _update_performance_metrics(self, metrics: SearchMetrics) -> None:
        """Update internal performance tracking."""
        self._search_count += 1
        self._total_search_time += metrics.total_duration_ms

    async def get_similar_creators(
        self,
        creator_profile: CreatorProfile,
        platforms: Optional[List[Platform]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """Find creators similar to the given profile."""
        # Build a query from the creator's metadata
        query_parts = []
        
        # Use categories as primary similarity signal
        if creator_profile.metadata.categories:
            query_parts.extend([cat.value for cat in creator_profile.metadata.categories])
        
        # Add keywords from bio
        if creator_profile.bio:
            bio_keywords = self.search_service.text_processor.get_important_terms(creator_profile.bio)
            query_parts.extend(list(bio_keywords)[:3])  # Limit to top 3 keywords
        
        if not query_parts:
            return []
        
        search_query = " ".join(query_parts)
        target_platforms = platforms or [creator_profile.platform]
        
        # Search with topic-based scoring
        response = await self.search(
            query=search_query,
            platforms=target_platforms,
            search_type=SearchType.SIMILAR,
            filters=SearchFilter(limit=limit * 2)  # Get more for better filtering
        )
        
        # Filter out the original creator and return top results
        similar_results = [
            result for result in response.results
            if result.profile.profile_url != creator_profile.profile_url
        ]
        
        return similar_results[:limit]

    async def get_trending_creators(
        self,
        platforms: List[Platform],
        category: Optional[ContentCategory] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """Get trending creators based on engagement metrics."""
        # This is a simplified implementation
        # In production, you might have a separate trending calculation service
        
        filters = SearchFilter(
            platforms=platforms,
            categories=[category] if category else [],
            verified_only=False,
            active_only=True,
            limit=limit * 3  # Get more for better ranking
        )
        
        # Use a broad query to get active creators
        response = await self.search(
            query="trending popular active",
            platforms=platforms,
            search_type=SearchType.TRENDING,
            filters=filters
        )
        
        # Sort by a combination of follower count and engagement
        def trending_score(result: SearchResult) -> float:
            profile = result.profile
            score = 0.0
            
            # Follower count component (logarithmic)
            if profile.social_metrics.followers_count:
                score += min(50, profile.social_metrics.followers_count ** 0.3)
            
            # Engagement rate component
            if profile.engagement_metrics.engagement_rate:
                score += profile.engagement_metrics.engagement_rate * 10
            
            # Verification bonus
            if profile.is_verified:
                score += 20
            
            # Recent activity bonus
            if profile.engagement_metrics.last_activity:
                days_since = (datetime.now() - profile.engagement_metrics.last_activity).days
                if days_since <= 7:
                    score += 15
            
            return score
        
        # Re-sort by trending score
        trending_results = sorted(
            response.results,
            key=trending_score,
            reverse=True
        )
        
        return trending_results[:limit]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the search agent."""
        avg_search_time = (
            self._total_search_time / self._search_count 
            if self._search_count > 0 else 0
        )
        
        platform_avg_times = {
            platform.value: sum(times) / len(times)
            for platform, times in self._platform_performance.items()
            if times
        }
        
        return {
            "total_searches": self._search_count,
            "average_search_time_ms": avg_search_time,
            "cache_stats": self.cache.get_stats(),
            "platform_performance": platform_avg_times,
            "available_connectors": list(self.connectors.keys())
        }

    async def clear_cache(self) -> None:
        """Clear the search cache."""
        await self.cache.clear()

    async def warmup_cache(
        self, 
        common_queries: List[str], 
        platforms: List[Platform]
    ) -> None:
        """Warmup cache with common queries."""
        logger.info(f"Warming up cache with {len(common_queries)} queries")
        
        warmup_tasks = []
        for query in common_queries:
            task = asyncio.create_task(
                self.search(
                    query=query,
                    platforms=platforms,
                    search_type=SearchType.CREATOR,
                    use_cache=False  # Don't use cache for warmup
                )
            )
            warmup_tasks.append(task)
        
        # Execute warmup queries
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
        
        logger.info("Cache warmup completed")

    async def shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        await self.clear_cache()
        logger.info("Search agent shutdown completed")