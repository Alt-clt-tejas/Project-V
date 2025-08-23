# app/connectors/youtube_connector.py
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import httpx
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.base import AppSettings
from app.connectors.base_connector import BaseConnector
from app.domains.search.schemas import CreatorProfile, Platform


logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for YouTube search parameters."""
    max_results: int = 25
    order: str = "relevance"  # relevance, viewCount, date, rating, title, videoCount
    published_after: Optional[datetime] = None
    region_code: Optional[str] = None
    relevance_language: Optional[str] = None


@dataclass
class YouTubeMetrics:
    """Enhanced metrics for YouTube channels."""
    subscriber_count: Optional[int] = None
    view_count: Optional[int] = None
    video_count: Optional[int] = None
    engagement_rate: Optional[float] = None
    avg_views_per_video: Optional[float] = None
    last_upload_date: Optional[datetime] = None
    upload_frequency: Optional[str] = None


class YouTubeConnector(BaseConnector):
    """
    Enhanced connector for the YouTube Data API v3 with improved error handling,
    caching, rate limiting, and comprehensive metrics.
    """

    def __init__(self, settings: AppSettings, client: httpx.AsyncClient):
        super().__init__(settings, client)
        
        if not self.settings.YOUTUBE_API_KEY:
            raise ValueError("YOUTUBE_API_KEY is not set in the configuration.")

        self.youtube_service = build(
            "youtube", 
            "v3", 
            developerKey=self.settings.YOUTUBE_API_KEY.get_secret_value(),
            cache_discovery=False  # Disable discovery cache for better performance
        )
        
        # Rate limiting and caching
        self._request_semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(minutes=15)

    @property
    def platform(self) -> Platform:
        return Platform.YOUTUBE

    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key from method and parameters."""
        params_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{method}_{params_str}"

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache if not expired."""
        if cache_key in self._cache:
            timestamp, data = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                logger.debug(f"Cache hit for key: {cache_key}")
                return data
        return None

    def _set_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._cache[cache_key] = (datetime.now(), data)
        
        # Clean old cache entries (simple cleanup)
        if len(self._cache) > 1000:
            expired_keys = [
                key for key, (timestamp, _) in self._cache.items()
                if datetime.now() - timestamp > self._cache_ttl
            ]
            for key in expired_keys:
                del self._cache[key]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(HttpError)
    )
    async def _make_youtube_request(self, request_func, cache_key: str = None) -> Any:
        """Make YouTube API request with retry logic and caching."""
        # Check cache first
        if cache_key:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

        async with self._request_semaphore:
            try:
                result = await asyncio.to_thread(request_func)
                
                # Cache the result
                if cache_key:
                    self._set_cache(cache_key, result)
                
                return result
                
            except HttpError as e:
                if e.resp.status == 429:  # Rate limit exceeded
                    logger.warning("YouTube API rate limit exceeded, retrying...")
                    raise
                elif e.resp.status in [400, 403]:  # Bad request or forbidden
                    logger.error(f"YouTube API error {e.resp.status}: {e.content}")
                    return {}
                else:
                    logger.error(f"YouTube API HTTP error {e.resp.status}: {e.content}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in YouTube API request: {e}")
                raise

    async def search(
        self, 
        query: str, 
        config: Optional[SearchConfig] = None
    ) -> List[CreatorProfile]:
        """
        Enhanced search with configurable parameters and comprehensive error handling.
        
        Args:
            query: Search query string
            config: Search configuration parameters
            
        Returns:
            List of CreatorProfile objects with enhanced metrics
        """
        if not query.strip():
            logger.warning("Empty search query provided")
            return []

        config = config or SearchConfig()
        
        try:
            # Step 1: Search for channels
            search_results = await self._search_channels(query, config)
            
            if not search_results:
                logger.info(f"No channels found for query: {query}")
                return []

            # Step 2: Enrich with detailed statistics
            channel_ids = [item["snippet"]["channelId"] for item in search_results]
            enriched_data = await self._get_channel_statistics(channel_ids)
            
            # Step 3: Get recent video data for engagement metrics
            video_data = await self._get_recent_videos(channel_ids[:10])  # Limit to top 10
            
            # Step 4: Parse and merge all data
            profiles = self._parse_and_merge_response(
                search_results, enriched_data, video_data
            )
            
            logger.info(f"Successfully found {len(profiles)} channels for query: {query}")
            return profiles
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []

    async def _search_channels(
        self, 
        query: str, 
        config: SearchConfig
    ) -> List[Dict[str, Any]]:
        """Search for channels with enhanced parameters."""
        search_params = {
            "q": query,
            "part": "snippet",
            "type": "channel",
            "maxResults": min(config.max_results, 50),  # API limit
            "order": config.order,
        }
        
        if config.published_after:
            search_params["publishedAfter"] = config.published_after.isoformat() + "Z"
        if config.region_code:
            search_params["regionCode"] = config.region_code
        if config.relevance_language:
            search_params["relevanceLanguage"] = config.relevance_language

        cache_key = self._get_cache_key("search", **search_params)
        
        search_response = await self._make_youtube_request(
            lambda: self.youtube_service.search().list(**search_params).execute(),
            cache_key
        )
        
        return search_response.get("items", [])

    async def _get_channel_statistics(self, channel_ids: List[str]) -> Dict[str, Any]:
        """Fetch detailed statistics for channels in batches."""
        if not channel_ids:
            return {}

        all_data = {}
        
        # Process in batches of 50 (API limit)
        for i in range(0, len(channel_ids), 50):
            batch_ids = channel_ids[i:i + 50]
            cache_key = self._get_cache_key("channels", ids=",".join(batch_ids))
            
            try:
                response = await self._make_youtube_request(
                    lambda: self.youtube_service.channels().list(
                        part="snippet,statistics,status,brandingSettings",
                        id=",".join(batch_ids),
                    ).execute(),
                    cache_key
                )
                
                batch_data = {
                    item["id"]: item 
                    for item in response.get("items", [])
                }
                all_data.update(batch_data)
                
            except Exception as e:
                logger.error(f"Failed to get statistics for batch {i//50 + 1}: {e}")
                continue

        return all_data

    async def _get_recent_videos(self, channel_ids: List[str]) -> Dict[str, List[Dict]]:
        """Get recent videos for engagement calculations."""
        if not channel_ids:
            return {}

        video_data = {}
        
        for channel_id in channel_ids:
            try:
                cache_key = self._get_cache_key("channel_videos", channel_id=channel_id)
                
                response = await self._make_youtube_request(
                    lambda: self.youtube_service.search().list(
                        channelId=channel_id,
                        part="snippet",
                        type="video",
                        order="date",
                        maxResults=10,
                    ).execute(),
                    cache_key
                )
                
                videos = response.get("items", [])
                
                if videos:
                    # Get video statistics
                    video_ids = [video["id"]["videoId"] for video in videos]
                    stats_response = await self._make_youtube_request(
                        lambda: self.youtube_service.videos().list(
                            part="statistics,snippet",
                            id=",".join(video_ids),
                        ).execute(),
                        self._get_cache_key("video_stats", ids=",".join(video_ids))
                    )
                    
                    video_data[channel_id] = stats_response.get("items", [])
                    
            except Exception as e:
                logger.warning(f"Failed to get recent videos for channel {channel_id}: {e}")
                continue

        return video_data

    def _calculate_engagement_metrics(
        self, 
        channel_stats: Dict[str, Any], 
        recent_videos: List[Dict]
    ) -> YouTubeMetrics:
        """Calculate enhanced engagement metrics."""
        stats = channel_stats.get("statistics", {})
        
        def to_int(value: Optional[str]) -> Optional[int]:
            try:
                return int(value) if value else None
            except (ValueError, TypeError):
                return None

        def to_float(value: Optional[str]) -> Optional[float]:
            try:
                return float(value) if value else None
            except (ValueError, TypeError):
                return None

        metrics = YouTubeMetrics(
            subscriber_count=to_int(stats.get("subscriberCount")),
            view_count=to_int(stats.get("viewCount")),
            video_count=to_int(stats.get("videoCount")),
        )

        # Calculate average views per video
        if metrics.view_count and metrics.video_count and metrics.video_count > 0:
            metrics.avg_views_per_video = metrics.view_count / metrics.video_count

        # Calculate engagement rate from recent videos
        if recent_videos and metrics.subscriber_count and metrics.subscriber_count > 0:
            total_engagement = 0
            valid_videos = 0
            
            for video in recent_videos:
                video_stats = video.get("statistics", {})
                views = to_int(video_stats.get("viewCount"))
                likes = to_int(video_stats.get("likeCount"))
                comments = to_int(video_stats.get("commentCount"))
                
                if views and views > 0:
                    engagement = ((likes or 0) + (comments or 0)) / views
                    total_engagement += engagement
                    valid_videos += 1

            if valid_videos > 0:
                metrics.engagement_rate = (total_engagement / valid_videos) * 100

        # Get last upload date
        if recent_videos:
            try:
                latest_video = recent_videos[0]
                published_at = latest_video.get("snippet", {}).get("publishedAt")
                if published_at:
                    metrics.last_upload_date = datetime.fromisoformat(
                        published_at.replace("Z", "+00:00")
                    )
            except Exception as e:
                logger.warning(f"Failed to parse last upload date: {e}")

        # Estimate upload frequency
        if len(recent_videos) >= 2:
            try:
                dates = []
                for video in recent_videos[:5]:  # Use last 5 videos
                    published_at = video.get("snippet", {}).get("publishedAt")
                    if published_at:
                        dates.append(datetime.fromisoformat(published_at.replace("Z", "+00:00")))
                
                if len(dates) >= 2:
                    dates.sort(reverse=True)
                    time_diff = dates[0] - dates[-1]
                    avg_days = time_diff.days / (len(dates) - 1)
                    
                    if avg_days <= 1:
                        metrics.upload_frequency = "Daily"
                    elif avg_days <= 7:
                        metrics.upload_frequency = "Weekly"
                    elif avg_days <= 30:
                        metrics.upload_frequency = "Monthly"
                    else:
                        metrics.upload_frequency = "Irregular"
                        
            except Exception as e:
                logger.warning(f"Failed to calculate upload frequency: {e}")

        return metrics

    def _is_verified_channel(self, channel_data: Dict[str, Any]) -> bool:
        """Determine if a channel is verified based on available indicators."""
        # Check if channel has custom URL (usually indicates verification)
        snippet = channel_data.get("snippet", {})
        if snippet.get("customUrl"):
            return True
            
        # Check subscriber count threshold (100k+ often indicates verification eligibility)
        stats = channel_data.get("statistics", {})
        subscriber_count = stats.get("subscriberCount")
        try:
            if subscriber_count and int(subscriber_count) >= 100000:
                return True
        except (ValueError, TypeError):
            pass
            
        return False

    def _parse_and_merge_response(
        self,
        search_items: List[Dict[str, Any]],
        enriched_data: Dict[str, Any],
        video_data: Dict[str, List[Dict]]
    ) -> List[CreatorProfile]:
        """Parse and merge all collected data into CreatorProfile objects."""
        profiles = []
        
        for item in search_items:
            try:
                snippet = item.get("snippet", {})
                channel_id = snippet.get("channelId")

                if not channel_id or channel_id not in enriched_data:
                    continue

                enriched_item = enriched_data[channel_id]
                recent_videos = video_data.get(channel_id, [])
                
                # Calculate enhanced metrics
                metrics = self._calculate_engagement_metrics(enriched_item, recent_videos)
                
                # Build profile URL with custom URL if available
                custom_url = enriched_item.get("snippet", {}).get("customUrl")
                if custom_url:
                    profile_url = f"https://www.youtube.com/{custom_url}"
                else:
                    profile_url = f"https://www.youtube.com/channel/{channel_id}"

                profile = CreatorProfile(
                    platform=self.platform,
                    name=snippet.get("title", "").strip(),
                    handle=custom_url or snippet.get("title", "").strip(),
                    profile_url=profile_url,
                    bio=snippet.get("description", "").strip() or None,
                    followers_count=metrics.subscriber_count,
                    view_count=metrics.view_count,
                    video_count=metrics.video_count,
                    is_verified=self._is_verified_channel(enriched_item),
                    # Additional metadata can be stored in a metadata field if available
                    # metadata={
                    #     "engagement_rate": metrics.engagement_rate,
                    #     "avg_views_per_video": metrics.avg_views_per_video,
                    #     "last_upload_date": metrics.last_upload_date.isoformat() if metrics.last_upload_date else None,
                    #     "upload_frequency": metrics.upload_frequency,
                    # }
                )
                
                profiles.append(profile)
                
            except Exception as e:
                logger.warning(f"Failed to parse channel data: {e}")
                continue

        return profiles

    async def get_channel_by_id(self, channel_id: str) -> Optional[CreatorProfile]:
        """Get a specific channel by its ID."""
        try:
            enriched_data = await self._get_channel_statistics([channel_id])
            if not enriched_data:
                return None
                
            channel_data = enriched_data.get(channel_id)
            if not channel_data:
                return None
                
            video_data = await self._get_recent_videos([channel_id])
            
            # Create a mock search item for consistency with existing parsing
            mock_search_item = {
                "snippet": channel_data.get("snippet", {})
            }
            
            profiles = self._parse_and_merge_response(
                [mock_search_item], enriched_data, video_data
            )
            
            return profiles[0] if profiles else None
            
        except Exception as e:
            logger.error(f"Failed to get channel by ID {channel_id}: {e}")
            return None

    async def cleanup(self) -> None:
        """Cleanup resources and clear cache."""
        self._cache.clear()
        logger.info("YouTube connector cleanup completed")