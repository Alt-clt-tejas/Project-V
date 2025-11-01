# app/agents/s5_search_agent.py
import asyncio
import logging
import time
import math
from typing import List, Dict, Optional, Set, Tuple, Any, cast
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from app.connectors.base_connector import BaseConnector
from app.domains.search.schemas import (
    Platform, SearchResult, SearchType, SearchQuery, SearchResponse, 
    SearchFilter, CreatorProfile, ContentCategory, SearchMatchDetails,
    EngagementMetrics, SocialMetrics
)
from app.domains.search.service import SearchService
from app.exceptions import ConnectorException

logger = logging.getLogger(__name__)


@dataclass
class SearchMetrics:
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
    """Thread-safe cache for search results with TTL."""
    
    def __init__(self, default_ttl_seconds: int = 300, max_entries: int = 1000):
        self.default_ttl = default_ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        
    async def close(self):
        """Clean up resources during shutdown."""
        async with self._lock:
            self._cache.clear()
            logger.info("Search cache cleared during shutdown")
        # Reset cache and lock to initial state
        self._cache = {}
    
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
            if hasattr(filters, 'content_categories') and filters.content_categories:
                filter_parts.append(f"cat_{'_'.join(c.value for c in filters.content_categories)}")
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
        search_service: Optional[SearchService] = None,
        cache_ttl_seconds: int = 300,
        max_cache_entries: int = 1000,
        timeout_seconds: float = 30.0
    ):
        """Initialize the search agent with the given connectors and settings."""
        self._connectors = connectors
        self._search_service = search_service
        self._timeout_seconds = timeout_seconds
        self._is_shutting_down = False
        self._cache = SearchCache(
            default_ttl_seconds=cache_ttl_seconds,
            max_entries=max_cache_entries
        )
        
        # Active search tracking
        self._active_searches = set()
        
        # Performance tracking
        self._search_count = 0
        self._total_search_time = 0.0
        self._platform_performance = defaultdict(list)
        self._cached_suggestions: Dict[str, List[str]] = {}
        self._max_cache_entries = max_cache_entries
        
    @property
    def connectors(self) -> Dict[Platform, BaseConnector]:
        return self._connectors
        
    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds
        
    def get_suggestions(
        self,
        query: str,
        profiles: List[CreatorProfile],
        max_suggestions: int = 5
    ) -> List[str]:
        """Generate search suggestions based on profiles and query context."""
        if not query or not profiles:
            return []
            
        query_lower = query.lower()
        
        # Try cache first
        if query_lower in self._cached_suggestions:
            return self._cached_suggestions[query_lower][:max_suggestions]
            
        suggestions = set()
        
        # Extract terms from profiles
        for profile in profiles:
            # Add relevant categories
            for category in (profile.metadata.categories if profile.metadata else []):
                if category.value.lower() not in query_lower:
                    suggestions.add(f"{query} {category.value}")
            
            # Add terms from bio
            if profile.bio:
                words = {w.strip().lower() for w in profile.bio.split() if len(w) > 3}
                for word in words:
                    if word not in query_lower:
                        suggestions.add(f"{query} {word}")
                        
        # Convert to list and sort by length (prefer shorter suggestions)
        suggestion_list = sorted(suggestions, key=len)[:max_suggestions]
        
        # Cache results
        self._cached_suggestions[query_lower] = suggestion_list
        if len(self._cached_suggestions) > self._max_cache_entries:
            # Remove oldest entries if cache is full
            oldest_query = min(self._cached_suggestions.keys())
            del self._cached_suggestions[oldest_query]
            
        return suggestion_list
        
    async def _execute_platform_search(
        self,
        connector: BaseConnector,
        query: str,
        search_type: SearchType,
        filters: Dict[str, Any],
        limit: int
    ) -> List[CreatorProfile]:
        """Execute a search on a single platform with proper error handling."""
        platform = connector.platform
        start_time = datetime.now()
        
        try:
            results = await connector.search(
                query=query,
                search_type=search_type.value,
                filters=filters,
                limit=limit
            )
            
            # Track performance
            search_time = (datetime.now() - start_time).total_seconds()
            self._platform_performance[platform].append(search_time)
            
            logger.info(
                f"Platform search completed",
                extra={
                    "platform": platform.value,
                    "results_count": len(results),
                    "duration": f"{search_time:.2f}s"
                }
            )
            
            return results
            
        except ConnectorException as e:
            logger.error(
                f"Platform search failed",
                extra={
                    "platform": platform.value,
                    "error": str(e),
                    "duration": f"{(datetime.now() - start_time).total_seconds():.2f}s"
                }
            )
            raise
            
    async def cleanup_resources(self):
        """Clean up resources during shutdown."""
        if self._is_shutting_down:
            return
            
        self._is_shutting_down = True
        try:
            # Clear cache
            if hasattr(self, '_cache'):
                await self._cache.close()
            
            # Close connectors
            for connector in self._connectors.values():
                await connector.close()
            
            logger.info("Search agent cleanup complete")
        except Exception as e:
            logger.error(f"Error during search agent cleanup: {e}")
        finally:
            self._is_shutting_down = False
    async def search(
        self,
        query: str,
        search_type: SearchType,
        filters: Optional[SearchFilter] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Execute a search across configured platforms with advanced error handling
        and telemetry tracking.
        """
        start_time = datetime.now()
        self._search_count += 1
        errors: List[str] = []
        
        try:
            # Normalize filters
            filters = filters or SearchFilter()
            platforms_to_search = filters.platforms or list(Platform)
            
            # Convert SearchFilter to connector-compatible dict
            connector_filters = {
                'min_followers': filters.min_followers,
                'max_followers': filters.max_followers,
                'verified_only': filters.verified_only,
                'min_video_count': filters.min_video_count,
                'max_video_count': filters.max_video_count,
                'min_avg_views': filters.min_avg_views,
                'max_avg_views': filters.max_avg_views,
                'created_after': filters.created_after,
                'created_before': filters.created_before,
                'last_active_after': filters.last_active_after,
                'active_only': filters.active_only,
                'countries': filters.countries,
                'languages': filters.languages
            }
            # Remove None values and empty lists
            connector_filters = {
                k: v for k, v in connector_filters.items() 
                if v is not None and (not isinstance(v, list) or v)
            }
            
            logger.info(
                f"Starting search across {len(platforms_to_search)} platforms",
                extra={
                    "query": query,
                    "search_type": search_type.value,
                    "platforms": [p.value for p in platforms_to_search],
                    "filters": connector_filters
                }
            )
            
            # Handle different search types
            if search_type == SearchType.NICHE:
                # For niche search, use semantic search across all platforms at once
                niche_results = await self._execute_niche_search(
                    query=query,
                    filters=filters,
                    limit=limit
                )
                return niche_results
            
            # For other search types, gather results concurrently from all platforms
            tasks = []
            for platform in platforms_to_search:
                if platform in self._connectors:
                    connector = self._connectors[platform]
                    
                    # Create the coroutine
                    coro = self._execute_platform_search(
                        connector=connector,
                        query=query,
                        search_type=search_type,
                        filters=connector_filters,
                        limit=limit
                    )
                    
                    # Explicitly wrap the coroutine in a Task
                    task = asyncio.create_task(coro)
                    tasks.append(task)
            
            if not tasks:
                logger.warning("No active connectors found for the requested platforms")
                return []
                
            # Wait for all search tasks with timeout
            results = []
            done, pending = await asyncio.wait(
                tasks,
                timeout=self._timeout_seconds,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.warning("Search task cancelled due to timeout")
                    errors.append("Search timeout")
            
            # Gather results and handle errors
            platform_results_map: Dict[Platform, int] = {}
            for task in done:
                try:
                    platform_results = await task
                    if platform_results:
                        filtered_results = self._apply_additional_filters(
                            platform_results, filters
                        )
                        results.extend(filtered_results)
                        # Update metrics
                        platform = platform_results[0].platform if platform_results else None
                        if platform:
                            platform_results_map[platform] = len(filtered_results)
                except ConnectorException as e:
                    logger.error(f"Platform search failed: {e}", exc_info=True)
                    errors.append(str(e))
                except Exception as e:
                    logger.error(f"Unexpected platform search error: {e}", exc_info=True)
                    errors.append(f"Unexpected error: {str(e)}")
            
            # Update performance metrics
            search_time = (datetime.now() - start_time).total_seconds()
            self._total_search_time += search_time
            
            # Create search results with proper scoring
            search_results = []
            for i, profile in enumerate(results[:limit]):
                quality_score = self._calculate_quality_score(profile)
                freshness_score = self._calculate_freshness_score(profile)
                match_confidence = self._calculate_match_confidence(query, profile)
                match_details = self._calculate_match_details(query, profile)
                
                search_result = SearchResult(
                    profile=profile,
                    search_query=query,
                    search_type=search_type,
                    match_confidence=match_confidence,
                    ranking_position=i + 1,
                    relevance_score=match_confidence,  # Use match confidence as relevance for now
                    match_details=match_details,
                    data_quality_score=quality_score,
                    freshness_score=freshness_score
                )
                search_results.append(search_result)
            
            logger.info(
                f"Search completed successfully",
                extra={
                    "duration_ms": search_time * 1000,
                    "results_count": len(search_results),
                    "platforms_count": len(platforms_to_search),
                    "errors_count": len(errors),
                    "platform_breakdown": platform_results_map
                }
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            errors.append(f"Critical error: {str(e)}")
            raise

    async def _execute_niche_search(
        self,
        query: str,
        filters: Optional[SearchFilter] = None,
        limit: int = 50
    ) -> List[SearchResult]:
        
        try:
            # Use the SearchService for niche-based search
            if not self._search_service:
                raise ValueError("SearchService is required for niche-based search")
                
            niche_results = await self._search_service.find_creators_by_niche(
                query=query,
                filters=filters if filters is not None else SearchFilter(),
                limit=limit
            )
            
            # Process and enrich the results
            enriched_results = []
            for idx, result in enumerate(niche_results, start=1):
                # Calculate quality metrics
                data_quality = self._calculate_data_quality(result.profile)
                freshness = self._calculate_freshness(result.profile)
                
                # Create a SearchResult with all required fields
                search_result = SearchResult(
                    profile=result.profile,
                    match_confidence=result.match_confidence,
                    relevance_score=result.relevance_score,
                    ranking_position=idx,
                    search_query=query,
                    search_type=SearchType.NICHE,
                    data_quality_score=data_quality,
                    freshness_score=freshness,
                    match_details=SearchMatchDetails(
                        semantic_similarity=result.match_details.semantic_similarity if result.match_details else None,
                        social_signals=result.match_details.social_signals if result.match_details else None,
                        name_similarity=0.0,
                        handle_similarity=0.0,
                        bio_relevance=0.0,
                        social_signals_boost=0.0,
                        match_reasons=result.match_details.match_reasons if result.match_details else []
                    )
                )
                enriched_results.append(search_result)
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Error during niche search: {str(e)}", exc_info=True)
            return []

    def _calculate_data_quality(self, profile: CreatorProfile) -> float:
        """Calculate a score representing the completeness and quality of profile data."""
        score = 0.0
        total_checks = 0

        # Check basic profile info
        if profile.name: score += 1
        if profile.handle: score += 1
        if profile.bio: score += 1
        if profile.profile_url: score += 1
        total_checks += 4

        # Check social metrics
        metrics = profile.social_metrics
        if metrics:
            if metrics.followers_count is not None: score += 1
            if metrics.total_views is not None: score += 1
            if metrics.total_content_count is not None: score += 1
            total_checks += 3

        # Check engagement metrics
        engagement = profile.engagement_metrics
        if engagement:
            if engagement.engagement_rate is not None: score += 1
            if engagement.avg_views_per_content is not None: score += 1
            if engagement.content_frequency is not None: score += 1
            total_checks += 3

        return score / total_checks if total_checks > 0 else 0.0

    def _calculate_freshness(self, profile: CreatorProfile) -> float:
        """Calculate how recent and active the profile is."""
        if not profile.engagement_metrics or not profile.engagement_metrics.last_activity:
            return 0.0

        # Calculate days since last activity
        days_since = (datetime.now() - profile.engagement_metrics.last_activity).days
        
        # Score decays exponentially with time
        # Score = 1.0 for same day, 0.9 for 1 day ago, etc.
        return math.exp(-days_since / 30)  # 30-day decay factor
            
    def _apply_additional_filters(
        self,
        profiles: List[CreatorProfile],
        filters: SearchFilter
    ) -> List[CreatorProfile]:
        """Apply filters that couldn't be applied at the connector level."""
        if not filters:
            return profiles
            
        filtered = []
        for profile in profiles:
            # Check engagement rate
            if (filters.min_engagement_rate and 
                profile.engagement_metrics and 
                profile.engagement_metrics.engagement_rate and
                profile.engagement_metrics.engagement_rate < filters.min_engagement_rate):
                continue
                
            if (filters.max_engagement_rate and 
                profile.engagement_metrics and 
                profile.engagement_metrics.engagement_rate and
                profile.engagement_metrics.engagement_rate > filters.max_engagement_rate):
                continue
            
            # Check content consistency
            if (filters.content_consistency_min and 
                profile.engagement_metrics and 
                profile.engagement_metrics.content_consistency_score and
                profile.engagement_metrics.content_consistency_score < filters.content_consistency_min):
                continue
                
            filtered.append(profile)
            
        return filtered

    def _calculate_match_confidence(self, query: str, profile: CreatorProfile) -> float:
        """Calculate a confidence score for how well the profile matches the query."""
        query_lower = query.lower()
        profile_name_lower = profile.name.lower()
        handle_lower = profile.handle.lower()
        bio_lower = profile.bio.lower() if profile.bio else ""
        
        # Direct matches have higher weight
        if query_lower == profile_name_lower or query_lower == handle_lower:
            return 1.0
            
        # Partial matches
        confidence = 0.0
        if query_lower in profile_name_lower:
            confidence += 0.5
        if query_lower in handle_lower:
            confidence += 0.3
        if query_lower in bio_lower:
            confidence += 0.2
            
        # Boost confidence based on verification and metrics
        if profile.is_verified:
            confidence = min(1.0, confidence + 0.2)
            
        return round(min(1.0, confidence), 2)
    
    def _calculate_match_details(self, query: str, profile: CreatorProfile) -> SearchMatchDetails:
        """Calculate detailed matching information."""
        query_terms = query.lower().split()
        name_score = 0.0
        bio_score = 0.0
        handle_score = 0.0
        exact_matches = []
        partial_matches = []
        matched_fields = []
        match_reasons = []
        
        # Calculate name match
        if profile.name:
            name_lower = profile.name.lower()
            if query.lower() == name_lower:
                name_score = 1.0
                exact_matches.append("name")
                matched_fields.append("name")
                match_reasons.append("Exact name match")
            else:
                for term in query_terms:
                    if term == name_lower:
                        name_score += 1.0
                        exact_matches.append("name")
                    elif term in name_lower:
                        name_score += 0.5 / len(query_terms)
                        partial_matches.append("name")
                if name_score > 0:
                    matched_fields.append("name")
                    match_reasons.append(f"Name similarity: {name_score:.0%}")
                    
        # Calculate handle match
        handle_lower = profile.handle.lower()
        if query.lower() == handle_lower:
            handle_score = 1.0
            exact_matches.append("handle")
            matched_fields.append("handle")
            match_reasons.append("Exact handle match")
        else:
            for term in query_terms:
                if term == handle_lower:
                    handle_score += 1.0
                    exact_matches.append("handle")
                elif term in handle_lower:
                    handle_score += 0.5 / len(query_terms)
                    partial_matches.append("handle")
            if handle_score > 0:
                matched_fields.append("handle")
                match_reasons.append(f"Handle similarity: {handle_score:.0%}")
                
        # Calculate bio match
        if profile.bio:
            bio_lower = profile.bio.lower()
            for term in query_terms:
                if term in bio_lower:
                    bio_score += 1.0 / len(query_terms)
                    partial_matches.append("bio")
            if bio_score > 0:
                matched_fields.append("bio")
                match_reasons.append(f"Bio relevance: {bio_score:.0%}")
                
        # Calculate social signals boost
        social_boost = 0.0
        if profile.social_metrics:
            if profile.social_metrics.followers_count and profile.social_metrics.followers_count > 10000:
                social_boost += 0.2
                match_reasons.append(f"Popular creator with {profile.social_metrics.followers_count:,} followers")
            if profile.engagement_metrics and profile.engagement_metrics.engagement_rate and profile.engagement_metrics.engagement_rate > 5.0:
                social_boost += 0.2
                match_reasons.append(f"High engagement rate: {profile.engagement_metrics.engagement_rate:.1f}%")
                
        if profile.is_verified:
            social_boost += 0.1
            match_reasons.append("Verified creator")
                
        return SearchMatchDetails(
            name_similarity=round(name_score, 2),
            handle_similarity=round(handle_score, 2),
            bio_relevance=round(bio_score, 2),
            keyword_matches=len(exact_matches) + len(partial_matches),
            exact_matches=list(set(exact_matches)),
            partial_matches=list(set(partial_matches)),
            social_signals_boost=round(social_boost, 2),
            semantic_similarity=None,
            social_signals=None,
            match_reasons=match_reasons,
            matched_fields=list(set(matched_fields))
        )
    
    def _calculate_quality_score(self, profile: CreatorProfile) -> float:
        """Calculate a data quality score based on profile completeness."""
        score = 0.5  # Base score
        
        # Add points for complete fields
        if profile.bio:
            score += 0.1
        if profile.avatar_url:
            score += 0.1
        if profile.is_verified:
            score += 0.2
        if profile.metadata.profile_completeness:
            score = min(1.0, score + profile.metadata.profile_completeness)
            
        return round(score, 2)
    
    def _calculate_freshness_score(self, profile: CreatorProfile) -> float:
        """Calculate how fresh/recent the profile data is."""
        if not profile.scraped_at:
            return 0.5
            
        current_time = datetime.now()
        if not profile.scraped_at.tzinfo:
            profile_time = profile.scraped_at.replace(tzinfo=None)
        else:
            profile_time = profile.scraped_at.astimezone().replace(tzinfo=None)
            
        age_hours = (current_time - profile_time).total_seconds() / 3600
        
        if age_hours < 1:
            return 1.0
        elif age_hours < 24:
            return 0.9
        elif age_hours < 72:
            return 0.7
        elif age_hours < 168:  # 1 week
            return 0.5
        else:
            return 0.3

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
            if hasattr(filters, 'content_categories') and filters.content_categories and profile.metadata.categories:
                if not any(cat in filters.content_categories for cat in profile.metadata.categories):
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
            if self._search_service:
                scored_results = self._search_service.score_results(query, profiles, search_type)
            else:
                # Return unscored results if no search service
                scored_results = []
            
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
                    ranking_position=i,
                    relevance_score=0.5,  # Default relevance
                    match_details=SearchMatchDetails(
                        name_similarity=0.0,
                        handle_similarity=0.0,
                        bio_relevance=0.0,
                        social_signals_boost=0.0,
                        semantic_similarity=None,
                        social_signals=None,
                        keyword_matches=0,
                        exact_matches=[],
                        partial_matches=[],
                        match_reasons=[],
                        matched_fields=[]
                    ),
                    data_quality_score=0.5,  # Default quality
                    freshness_score=0.5  # Default freshness
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

        # Generate suggestions if search service available
        suggestions = []
        if self._search_service:
            all_profiles = [result.profile for result in results]
            suggestions = self._search_service.get_suggestions(
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
            # Extract bio keywords if search service available
            bio_keywords = []
            if self._search_service and self._search_service.text_processor:
                bio_keywords = self._search_service.text_processor.get_important_terms(creator_profile.bio)
            query_parts.extend(list(bio_keywords)[:3])  # Limit to top 3 keywords
        
        if not query_parts:
            return []
        
        search_query_str = " ".join(query_parts)
        target_platforms = platforms or [creator_profile.platform]
        
        # Create SearchQuery object for the new search method
        search_query = SearchQuery(
            query=search_query_str,
            search_type=SearchType.SIMILAR,
            filters=SearchFilter(
                platforms=target_platforms,

            )
        )
        
        # Search with topic-based scoring
        # Note: This now calls the modified 'search' method
        response_results = await self.search(
            query=search_query.query,
            search_type=search_query.search_type,
            filters=search_query.filters
        )
        
        # Filter out the original creator and return top results
        similar_results = [
            result for result in response_results
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

            verified_only=False,
            active_only=True,

        )
        
        # Use a broad query to get active creators
        search_query = SearchQuery(
            query="trending popular active",
            search_type=SearchType.TRENDING,
            filters=filters
        )
        
        # Note: This now calls the modified 'search' method
        response_results = await self.search(
            query=search_query.query,
            search_type=search_query.search_type,
            filters=search_query.filters
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
            response_results,
            key=trending_score,
            reverse=True
        )
        
        return trending_results[:limit]

    def _update_performance_tracking(
        self,
        metrics: SearchMetrics
    ) -> None:
        """Update internal performance tracking metrics."""
        self._search_count += 1
        self._total_search_time += metrics.total_duration_ms
        
        for platform, duration in metrics.connector_durations.items():
            self._platform_performance[platform].append(duration)
            
            # Keep only last 100 measurements per platform
            if len(self._platform_performance[platform]) > 100:
                self._platform_performance[platform] = self._platform_performance[platform][-100:]

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
            "cache_stats": self._cache.get_stats(),
            "platform_performance": platform_avg_times,
            "available_connectors": list(self._connectors.keys()),
            "active_searches": len(self._active_searches) if hasattr(self, '_active_searches') else 0
        }

    async def clear_cache(self) -> None:
        """Clear the search cache."""
        await self._cache.clear()

    async def warmup_cache(
        self, 
        common_queries: List[str], 
        platforms: List[Platform]
    ) -> None:
        """Warmup cache with common queries."""
        logger.info(f"Warming up cache with {len(common_queries)} queries")
        
        warmup_tasks = []
        for query_str in common_queries:
            
            task = asyncio.create_task(
                self.search(
                    query=query_str,
                    search_type=SearchType.CREATOR,
                    filters=SearchFilter(platforms=platforms)
                )
            )
            warmup_tasks.append(task)
        
        # Execute warmup queries
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
        
        logger.info("Cache warmup completed")

    async def _cleanup(self) -> None:
        """Clean up resources during shutdown."""
        if hasattr(self, '_is_shutting_down') and self._is_shutting_down:
            return
            
        self._is_shutting_down = True
        try:
            # Close connectors
            for connector in self._connectors.values():
                await connector.close()
                
            # Clear caches
            await self.clear_cache()
            
            logger.info("Search agent cleanup complete")
        except Exception as e:
            logger.error(f"Error during search agent cleanup: {e}")
        finally:
            self._is_shutting_down = False
            
    async def shutdown(self) -> None:
        """Clean up and shut down the search agent."""
        await self._cleanup()