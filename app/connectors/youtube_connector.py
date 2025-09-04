# app/connectors/youtube_connector.py
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import re

import httpx
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.base import AppSettings
from app.connectors.base_connector import BaseConnector
from app.domains.search.schemas import (
    CreatorProfile, Platform, SocialMetrics, EngagementMetrics, ProfileMetadata,
    VerificationStatus, ContentCategory
)
from app.exceptions import ConnectorException

logger = logging.getLogger(__name__)

# Enhanced YouTube category mapping with more comprehensive coverage
YOUTUBE_CATEGORY_MAP = {
    "1": ContentCategory.ENTERTAINMENT,  # Film & Animation
    "2": ContentCategory.OTHER,  # Autos & Vehicles
    "10": ContentCategory.MUSIC,
    "15": ContentCategory.OTHER,  # Pets & Animals
    "17": ContentCategory.SPORTS,
    "19": ContentCategory.TRAVEL,  # Travel & Events
    "20": ContentCategory.GAMING,
    "22": ContentCategory.LIFESTYLE,  # People & Blogs
    "23": ContentCategory.COMEDY,
    "24": ContentCategory.ENTERTAINMENT,
    "25": ContentCategory.NEWS,  # News & Politics
    "26": ContentCategory.LIFESTYLE,  # Howto & Style
    "27": ContentCategory.EDUCATION,
    "28": ContentCategory.TECHNOLOGY,  # Science & Technology
    "29": ContentCategory.OTHER,  # Nonprofits & Activism
}

# YouTube topic categories to our content categories
YOUTUBE_TOPIC_MAP = {
    "/m/04rlf": ContentCategory.MUSIC,
    "/m/02vxn": ContentCategory.MUSIC,  # Music (general)
    "/m/0bzvm2": ContentCategory.GAMING,
    "/m/025zzc": ContentCategory.ENTERTAINMENT,  # Entertainment
    "/m/02jjt": ContentCategory.ENTERTAINMENT,  # Entertainment (general)
    "/m/09s1f": ContentCategory.COMEDY,
    "/m/02vx4": ContentCategory.FILM,  # Film
    "/m/05qjc": ContentCategory.SPORTS,
    "/m/06ntj": ContentCategory.SPORTS,  # Sports (general)
    "/m/01k8wb": ContentCategory.LIFESTYLE,  # Knowledge
    "/m/05rwpb": ContentCategory.LIFESTYLE,  # Lifestyle
    "/m/019_rr": ContentCategory.LIFESTYLE,  # Lifestyle (general)
    "/m/032tl": ContentCategory.FOOD,
    "/m/07c1v": ContentCategory.TECHNOLOGY,
    "/m/07bxq": ContentCategory.HEALTH_FITNESS,  # Physical fitness
    "/m/027x7n": ContentCategory.FASHION,
    "/m/02wbm": ContentCategory.FOOD,  # Food (general)
    "/m/01h6rj": ContentCategory.TRAVEL,
    "/m/0kt51": ContentCategory.HEALTH_FITNESS,  # Health
}


@dataclass
class YouTubeSearchConfig:
    """Configuration for YouTube search operations."""
    max_results: int = 25
    order: str = "relevance"  # relevance, viewCount, date, rating, title, videoCount
    region_code: Optional[str] = None
    relevance_language: Optional[str] = None
    published_after: Optional[datetime] = None
    video_definition: Optional[str] = None  # any, high, standard
    video_duration: Optional[str] = None  # any, long, medium, short


@dataclass
class YouTubeMetrics:
    """Detailed YouTube-specific metrics."""
    subscriber_count: Optional[int] = None
    total_views: Optional[int] = None
    video_count: Optional[int] = None
    avg_views_per_video: Optional[float] = None
    engagement_rate: Optional[float] = None
    upload_frequency: Optional[str] = None
    last_upload: Optional[datetime] = None
    most_popular_video_views: Optional[int] = None
    channel_age_days: Optional[int] = None
    content_consistency_score: Optional[float] = None


class YouTubeConnector(BaseConnector):
    """
    Enterprise-grade YouTube Connector with comprehensive data extraction,
    advanced error handling, caching, and rich metrics calculation.
    """
    
    def __init__(self, settings: AppSettings, client: httpx.AsyncClient):
        super().__init__(settings, client)
        
        if not settings.YOUTUBE_API_KEY:
            raise ValueError("YOUTUBE_API_KEY must be configured.")
        
        self.youtube_service = build(
            'youtube', 'v3',
            developerKey=settings.YOUTUBE_API_KEY.get_secret_value(),
            cache_discovery=False
        )
        
        # Rate limiting and performance tracking
        self._request_semaphore = asyncio.Semaphore(10)  # Concurrent request limit
        self._api_call_count = 0
        self._quota_usage = 0
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(minutes=10)

    @property
    def platform(self) -> Platform:
        return Platform.YOUTUBE

    async def search(
        self, 
        query: str, 
        config: Optional[YouTubeSearchConfig] = None
    ) -> List[CreatorProfile]:
        """
        Enhanced search with configurable parameters and comprehensive error handling.
        """
        if not query.strip():
            logger.warning("Empty search query provided to YouTube connector")
            return []

        config = config or YouTubeSearchConfig()
        
        try:
            logger.info(f"Starting YouTube search for query: '{query}' with config: {config}")
            
            # Step 1: Search for channels
            search_items = await self._search_channels(query, config)
            if not search_items:
                logger.info(f"No channels found for query: {query}")
                return []

            # Extract channel IDs
            channel_ids = [item['id']['channelId'] for item in search_items if 'channelId' in item['id']]
            if not channel_ids:
                logger.warning("No valid channel IDs found in search results")
                return []

            logger.debug(f"Found {len(channel_ids)} channel IDs")

            # Step 2: Get detailed channel information
            channel_details = await self._get_channel_details(channel_ids)
            if not channel_details:
                logger.warning("No channel details retrieved")
                return []

            # Step 3: Get recent videos for engagement metrics (top 10 for performance)
            video_details_map = await self._get_recent_videos_for_channels(channel_ids[:10])

            # Step 4: Get additional insights for top channels
            channel_insights = await self._get_channel_insights(channel_ids[:5])

            # Step 5: Build comprehensive creator profiles
            profiles = await self._build_creator_profiles(
                channel_details, video_details_map, channel_insights
            )

            logger.info(f"Successfully created {len(profiles)} creator profiles")
            return profiles
            
        except HttpError as e:
            error_msg = f"YouTube API HTTP error {e.resp.status}: {e.content}"
            logger.error(error_msg, extra={"platform": self.platform.value})
            
            # Handle specific API errors gracefully
            if e.resp.status == 403:
                raise ConnectorException("YouTube API quota exceeded or access denied", self.platform)
            elif e.resp.status == 400:
                raise ConnectorException(f"Invalid YouTube API request: {e.content}", self.platform)
            else:
                raise ConnectorException(f"YouTube API error: {e.reason}", self.platform)
                
        except Exception as e:
            logger.error(
                f"Unexpected error in YouTube connector: {e}", 
                extra={"platform": self.platform.value}, 
                exc_info=True
            )
            raise ConnectorException(f"YouTube connector failed: {str(e)}", self.platform)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(HttpError)
    )
    async def _api_call(
        self, 
        func, 
        cache_key: Optional[str] = None, 
        quota_cost: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Enhanced API call wrapper with caching, retry logic, and quota tracking.
        """
        # Check cache first
        if cache_key:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

        async with self._request_semaphore:
            try:
                self._api_call_count += 1
                self._quota_usage += quota_cost
                
                response = await asyncio.to_thread(func(**kwargs).execute)
                result = response.get('items', [])
                
                # Cache the result
                if cache_key:
                    self._set_cache(cache_key, result)
                
                logger.debug(f"YouTube API call successful. Quota used: {quota_cost}")
                return result
                
            except HttpError as e:
                if e.resp.status == 429:  # Rate limit
                    logger.warning("YouTube API rate limit hit, retrying...")
                    raise
                elif e.resp.status in [400, 403]:
                    logger.error(f"YouTube API error {e.resp.status}: {e.content}")
                    return []
                else:
                    logger.error(f"YouTube API HTTP error {e.resp.status}: {e.content}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in YouTube API call: {e}")
                raise

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache if not expired."""
        if cache_key in self._cache:
            timestamp, data = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return data
            else:
                del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._cache[cache_key] = (datetime.now(), data)
        
        # Simple cache cleanup
        if len(self._cache) > 500:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]

    async def _search_channels(
        self, 
        query: str, 
        config: YouTubeSearchConfig
    ) -> List[Dict[str, Any]]:
        """Enhanced channel search with configurable parameters."""
        search_params = {
            'q': query,
            'part': 'snippet',
            'type': 'channel',
            'maxResults': min(config.max_results, 50),
            'order': config.order,
        }
        
        # Add optional parameters
        if config.region_code:
            search_params['regionCode'] = config.region_code
        if config.relevance_language:
            search_params['relevanceLanguage'] = config.relevance_language
        if config.published_after:
            search_params['publishedAfter'] = config.published_after.isoformat() + 'Z'

        cache_key = f"search_{hash(str(sorted(search_params.items())))}"
        
        return await self._api_call(
            self.youtube_service.search().list,
            cache_key=cache_key,
            quota_cost=100,  # Search costs 100 quota units
            **search_params
        )

    async def _get_channel_details(self, channel_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch comprehensive channel details in optimized batches."""
        all_details = []
        
        # Process in batches of 50 (API limit)
        for i in range(0, len(channel_ids), 50):
            batch_ids = channel_ids[i:i + 50]
            
            try:
                cache_key = f"channels_{hash(','.join(sorted(batch_ids)))}"
                
                details = await self._api_call(
                    self.youtube_service.channels().list,
                    cache_key=cache_key,
                    quota_cost=1,  # Channel details cost 1 quota unit per call
                    part='snippet,statistics,brandingSettings,topicDetails,status,contentDetails',
                    id=','.join(batch_ids)
                )
                
                all_details.extend(details)
                logger.debug(f"Retrieved details for batch of {len(details)} channels")
                
            except Exception as e:
                logger.error(f"Failed to get channel details for batch {i//50 + 1}: {e}")
                continue

        return all_details

    async def _get_recent_videos_for_channels(
        self, 
        channel_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent videos with enhanced error handling and performance optimization."""
        video_map = {}
        
        # Process channels concurrently but with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent video fetches
        
        async def fetch_channel_videos(channel_id: str) -> Tuple[str, List[Dict]]:
            async with semaphore:
                try:
                    # Get recent video IDs
                    cache_key = f"channel_videos_{channel_id}"
                    
                    search_items = await self._api_call(
                        self.youtube_service.search().list,
                        cache_key=cache_key,
                        quota_cost=100,
                        channelId=channel_id,
                        part='id,snippet',
                        type='video',
                        order='date',
                        maxResults=10
                    )
                    
                    if not search_items:
                        return channel_id, []
                    
                    # Extract video IDs
                    video_ids = [item['id']['videoId'] for item in search_items if 'videoId' in item['id']]
                    if not video_ids:
                        return channel_id, []
                    
                    # Get video statistics
                    video_details = await self._api_call(
                        self.youtube_service.videos().list,
                        cache_key=f"video_stats_{hash(','.join(video_ids))}",
                        quota_cost=1,
                        part='snippet,statistics,contentDetails',
                        id=','.join(video_ids)
                    )
                    
                    return channel_id, video_details
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch videos for channel {channel_id}: {e}")
                    return channel_id, []
        
        # Execute video fetching tasks
        tasks = [fetch_channel_videos(channel_id) for channel_id in channel_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Video fetch task failed: {result}")
                continue
            
            channel_id, videos = result
            if videos:
                video_map[channel_id] = videos

        logger.debug(f"Retrieved video data for {len(video_map)} channels")
        return video_map

    async def _get_channel_insights(self, channel_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get additional insights for top-performing channels."""
        insights = {}
        
        for channel_id in channel_ids:
            try:
                # Get channel's playlists for content organization insights
                playlists = await self._api_call(
                    self.youtube_service.playlists().list,
                    cache_key=f"playlists_{channel_id}",
                    quota_cost=1,
                    part='snippet,contentDetails',
                    channelId=channel_id,
                    maxResults=10
                )
                
                insights[channel_id] = {
                    'playlists': playlists,
                    # Could add more insights here (comments, community posts, etc.)
                }
                
            except Exception as e:
                logger.warning(f"Failed to get insights for channel {channel_id}: {e}")
                continue
        
        return insights

    async def _build_creator_profiles(
        self,
        channel_details: List[Dict],
        video_details_map: Dict[str, List[Dict]],
        channel_insights: Dict[str, Dict]
    ) -> List[CreatorProfile]:
        """Build comprehensive creator profiles with enhanced metrics."""
        profiles = []
        
        for data in channel_details:
            try:
                channel_id = data.get('id')
                if not channel_id:
                    continue
                
                snippet = data.get('snippet', {})
                stats = data.get('statistics', {})
                branding = data.get('brandingSettings', {}).get('channel', {})
                status = data.get('status', {})
                content_details = data.get('contentDetails', {})
                
                recent_videos = video_details_map.get(channel_id, [])
                insights = channel_insights.get(channel_id, {})
                
                # Calculate comprehensive metrics
                youtube_metrics = self._calculate_youtube_metrics(stats, recent_videos, data, insights)
                social_metrics = self._map_social_metrics(stats, youtube_metrics)
                engagement_metrics = self._map_engagement_metrics(youtube_metrics, recent_videos)
                metadata = self._map_metadata(snippet, branding, data.get('topicDetails', {}), insights)
                
                # Enhanced verification detection
                verification_status = self._determine_verification_status(
                    snippet, stats, status, youtube_metrics
                )
                
                # Build profile URL
                custom_url = snippet.get('customUrl')
                if custom_url:
                    if custom_url.startswith('@'):
                        profile_url = f"https://www.youtube.com/{custom_url}"
                    else:
                        profile_url = f"https://www.youtube.com/c/{custom_url}"
                else:
                    profile_url = f"https://www.youtube.com/channel/{channel_id}"
                
                # Parse creation date
                created_at = None
                if snippet.get('publishedAt'):
                    try:
                        created_at = datetime.fromisoformat(
                            snippet['publishedAt'].replace('Z', '+00:00')
                        )
                    except ValueError:
                        logger.warning(f"Could not parse creation date for channel {channel_id}")

                profile = CreatorProfile(
                    platform=self.platform,
                    platform_id=channel_id,
                    name=snippet.get('title', '').strip(),
                    handle=custom_url or f"channel/{channel_id}",
                    profile_url=profile_url,
                    bio=self._clean_description(snippet.get('description')),
                    avatar_url=self._get_best_thumbnail_url(snippet.get('thumbnails', {})),
                    banner_url=self._get_banner_url(branding),
                    is_verified=verification_status != VerificationStatus.UNVERIFIED,
                    verification_status=verification_status,
                    account_type=self._determine_account_type(branding, stats),
                    is_active=self._is_channel_active(recent_videos, stats),
                    created_at=created_at,
                    updated_at=datetime.now(timezone.utc),
                    social_metrics=social_metrics,
                    engagement_metrics=engagement_metrics,
                    metadata=metadata,
                    scraped_at=datetime.now(timezone.utc)
                )
                
                profiles.append(profile)
                logger.debug(f"Created profile for channel: {profile.name}")
                
            except Exception as e:
                logger.warning(f"Failed to build profile for channel {data.get('id', 'unknown')}: {e}")
                continue
        
        return profiles

    def _calculate_youtube_metrics(
        self,
        stats: Dict,
        recent_videos: List[Dict],
        channel_data: Dict,
        insights: Dict
    ) -> YouTubeMetrics:
        """Calculate comprehensive YouTube-specific metrics."""
        metrics = YouTubeMetrics()
        
        # Basic stats
        metrics.subscriber_count = self._safe_int(stats.get('subscriberCount'))
        metrics.total_views = self._safe_int(stats.get('viewCount'))
        metrics.video_count = self._safe_int(stats.get('videoCount'))
        
        # Calculate averages
        if metrics.total_views and metrics.video_count and metrics.video_count > 0:
            metrics.avg_views_per_video = metrics.total_views / metrics.video_count
        
        # Channel age
        created_at = channel_data.get('snippet', {}).get('publishedAt')
        if created_at:
            try:
                creation_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                metrics.channel_age_days = (datetime.now(timezone.utc) - creation_date).days
            except ValueError:
                pass
        
        # Recent video analysis
        if recent_videos:
            self._analyze_recent_videos(metrics, recent_videos)
        
        # Content consistency score based on upload patterns
        metrics.content_consistency_score = self._calculate_consistency_score(recent_videos)
        
        return metrics

    def _analyze_recent_videos(self, metrics: YouTubeMetrics, recent_videos: List[Dict]) -> None:
        """Analyze recent videos for engagement and upload patterns."""
        if not recent_videos:
            return
        
        # Last upload
        try:
            latest_video = max(
                recent_videos,
                key=lambda v: v.get('snippet', {}).get('publishedAt', '')
            )
            published_at = latest_video.get('snippet', {}).get('publishedAt')
            if published_at:
                metrics.last_upload = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        except (ValueError, KeyError):
            pass
        
        # Engagement analysis
        total_engagement = 0
        valid_videos = 0
        max_views = 0
        
        for video in recent_videos:
            video_stats = video.get('statistics', {})
            views = self._safe_int(video_stats.get('viewCount', 0))
            likes = self._safe_int(video_stats.get('likeCount', 0))
            comments = self._safe_int(video_stats.get('commentCount', 0))
            
            if views and views > 0:
                engagement_rate = ((likes or 0) + (comments or 0)) / views
                total_engagement += engagement_rate
                valid_videos += 1
                max_views = max(max_views, views)
        
        if valid_videos > 0:
            metrics.engagement_rate = (total_engagement / valid_videos) * 100
        
        metrics.most_popular_video_views = max_views if max_views > 0 else None
        
        # Upload frequency estimation
        metrics.upload_frequency = self._estimate_upload_frequency(recent_videos)

    def _estimate_upload_frequency(self, recent_videos: List[Dict]) -> Optional[str]:
        """Estimate upload frequency based on recent videos."""
        if len(recent_videos) < 2:
            return None
        
        try:
            # Sort videos by date
            dated_videos = []
            for video in recent_videos:
                published_at = video.get('snippet', {}).get('publishedAt')
                if published_at:
                    date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    dated_videos.append(date)
            
            if len(dated_videos) < 2:
                return None
            
            dated_videos.sort(reverse=True)
            
            # Calculate average days between uploads
            total_days = 0
            intervals = 0
            
            for i in range(len(dated_videos) - 1):
                days_diff = (dated_videos[i] - dated_videos[i + 1]).days
                if days_diff > 0:  # Only count positive intervals
                    total_days += days_diff
                    intervals += 1
            
            if intervals == 0:
                return None
            
            avg_days = total_days / intervals
            
            if avg_days <= 1:
                return "Daily"
            elif avg_days <= 3:
                return "Multiple times per week"
            elif avg_days <= 7:
                return "Weekly"
            elif avg_days <= 14:
                return "Bi-weekly"
            elif avg_days <= 30:
                return "Monthly"
            else:
                return "Irregular"
                
        except Exception as e:
            logger.debug(f"Failed to estimate upload frequency: {e}")
            return None

    def _calculate_consistency_score(self, recent_videos: List[Dict]) -> Optional[float]:
        """Calculate content consistency score based on upload patterns."""
        if len(recent_videos) < 3:
            return None
        
        try:
            # Get upload dates
            dates = []
            for video in recent_videos:
                published_at = video.get('snippet', {}).get('publishedAt')
                if published_at:
                    date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    dates.append(date)
            
            if len(dates) < 3:
                return None
            
            dates.sort(reverse=True)
            
            # Calculate intervals between uploads
            intervals = []
            for i in range(len(dates) - 1):
                days_diff = (dates[i] - dates[i + 1]).days
                if days_diff > 0:
                    intervals.append(days_diff)
            
            if len(intervals) < 2:
                return None
            
            # Calculate coefficient of variation (lower = more consistent)
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            std_dev = variance ** 0.5
            
            if avg_interval == 0:
                return None
            
            cv = std_dev / avg_interval
            
            # Convert to consistency score (1.0 = perfectly consistent, 0.0 = very inconsistent)
            consistency_score = max(0.0, 1.0 - min(cv, 1.0))
            return consistency_score
            
        except Exception as e:
            logger.debug(f"Failed to calculate consistency score: {e}")
            return None

    def _map_social_metrics(self, stats: Dict, youtube_metrics: YouTubeMetrics) -> SocialMetrics:
        """Map YouTube statistics to social metrics schema."""
        return SocialMetrics(
            followers_count=youtube_metrics.subscriber_count,
            total_views=youtube_metrics.total_views,
            video_count=youtube_metrics.video_count,
            total_content_count=youtube_metrics.video_count,
            # Additional YouTube-specific metrics could be added here
        )

    def _map_engagement_metrics(
        self, 
        youtube_metrics: YouTubeMetrics, 
        recent_videos: List[Dict]
    ) -> EngagementMetrics:
        """Map YouTube metrics to engagement metrics schema."""
        return EngagementMetrics(
            engagement_rate=youtube_metrics.engagement_rate,
            avg_views_per_content=youtube_metrics.avg_views_per_video,
            content_frequency=youtube_metrics.upload_frequency,
            last_activity=youtube_metrics.last_upload
        )

    def _map_metadata(
        self, 
        snippet: Dict, 
        branding: Dict, 
        topic_details: Dict,
        insights: Dict
    ) -> ProfileMetadata:
        """Map YouTube data to metadata schema with enhanced categorization."""
        # Extract categories from topic details
        categories = set()
        
        # Map topic categories
        for topic_url in topic_details.get('topicCategories', []):
            topic_id = topic_url.split('/')[-1] if '/' in topic_url else topic_url
            if topic_id in YOUTUBE_TOPIC_MAP:
                categories.add(YOUTUBE_TOPIC_MAP[topic_id])
        
        # Extract keywords/tags
        keywords = []
        keyword_string = branding.get('keywords', '')
        if keyword_string:
            # Split by common delimiters
            keywords = [
                kw.strip().lower()
                for kw in re.split(r'[,;|\n]', keyword_string)
                if kw.strip()
            ]
        
        # Get playlist count for content organization insight
        custom_fields = {}
        if 'playlists' in insights:
            custom_fields['playlist_count'] = len(insights['playlists'])
        
        return ProfileMetadata(
            categories=list(categories),
            tags=keywords[:20],  # Limit to first 20 tags
            country=snippet.get('country'),
            custom_fields=custom_fields
        )

    def _determine_verification_status(
        self, 
        snippet: Dict, 
        stats: Dict, 
        status: Dict,
        metrics: YouTubeMetrics
    ) -> VerificationStatus:
        """Determine verification status using multiple signals."""
        # Check for custom URL (strong indicator)
        if snippet.get('customUrl'):
            return VerificationStatus.VERIFIED
        
        # High subscriber count threshold
        if metrics.subscriber_count and metrics.subscriber_count >= 100000:
            return VerificationStatus.VERIFIED
        
        # Check for partner status or other indicators in channel status
        if status.get('madeForKids') is not None:  # Indicates API access level
            return VerificationStatus.PARTNER
        
        return VerificationStatus.UNVERIFIED

    def _determine_account_type(self, branding: Dict, stats: Dict) -> Optional[str]:
        """Determine account type based on available signals."""
        # This is simplified - you might have more sophisticated logic
        subscriber_count = self._safe_int(stats.get('subscriberCount', 0))
        
        if subscriber_count and subscriber_count > 1000000:
            return "enterprise"
        elif subscriber_count and subscriber_count > 100000:
            return "business"
        else:
            return "personal"

    def _is_channel_active(self, recent_videos: List[Dict], stats: Dict) -> bool:
        """Determine if channel is actively posting content."""
        if not recent_videos:
            return False
        
        # Check if there's been activity in the last 6 months
        try:
            latest_video = recent_videos[0]
            published_at = latest_video.get('snippet', {}).get('publishedAt')
            if published_at:
                last_upload = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
                return last_upload > six_months_ago
        except (ValueError, KeyError, IndexError):
            pass
        
        # Fallback: check if subscriber count is not hidden (active channels usually show stats)
        return not stats.get('hiddenSubscriberCount', False)

    def _clean_description(self, description: Optional[str]) -> Optional[str]:
        """Clean and truncate channel description."""
        if not description:
            return None
        
        # Remove excessive whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', description.strip())
        
        # Truncate if too long (keeping within reasonable limits)
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + "..."
        
        return cleaned if cleaned else None

    def _get_best_thumbnail_url(self, thumbnails: Dict) -> Optional[str]:
        """Get the highest quality thumbnail URL available."""
        if not thumbnails:
            return None
        
        # Priority order: maxres > high > medium > default
        for quality in ['maxres', 'high', 'medium', 'default']:
            if quality in thumbnails:
                return thumbnails[quality].get('url')
        
        return None

    def _get_banner_url(self, branding: Dict) -> Optional[str]:
        """Extract banner/cover image URL from branding settings."""
        if not branding:
            return None
        
        image_settings = branding.get('image', {})
        return image_settings.get('bannerExternalUrl')

    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert value to integer."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    async def get_channel_by_id(self, channel_id: str) -> Optional[CreatorProfile]:
        """Get a specific channel by its YouTube channel ID."""
        try:
            logger.info(f"Fetching YouTube channel by ID: {channel_id}")
            
            # Get channel details
            channel_details = await self._get_channel_details([channel_id])
            if not channel_details:
                logger.warning(f"No details found for channel ID: {channel_id}")
                return None
            
            # Get recent videos for engagement metrics
            video_details_map = await self._get_recent_videos_for_channels([channel_id])
            
            # Get additional insights
            channel_insights = await self._get_channel_insights([channel_id])
            
            # Build profile
            profiles = await self._build_creator_profiles(
                channel_details, video_details_map, channel_insights
            )
            
            if profiles:
                logger.info(f"Successfully retrieved channel: {profiles[0].name}")
                return profiles[0]
            else:
                logger.warning(f"Failed to build profile for channel ID: {channel_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get channel by ID {channel_id}: {e}")
            raise ConnectorException(f"Failed to retrieve YouTube channel: {str(e)}", self.platform)

    async def get_channel_by_handle(self, handle: str) -> Optional[CreatorProfile]:
        """Get a channel by its custom handle/username."""
        try:
            # Clean handle (remove @ if present)
            clean_handle = handle.lstrip('@')
            
            logger.info(f"Fetching YouTube channel by handle: {clean_handle}")
            
            # Search for the channel by exact handle match
            search_results = await self.search(f'"{clean_handle}"', YouTubeSearchConfig(max_results=5))
            
            # Look for exact handle match
            for profile in search_results:
                if (profile.handle.lower() == clean_handle.lower() or 
                    profile.handle.lower() == f"@{clean_handle.lower()}"):
                    return profile
            
            # If no exact match, try direct API call with forUsername (deprecated but sometimes works)
            try:
                channel_details = await self._api_call(
                    self.youtube_service.channels().list,
                    cache_key=f"channel_by_username_{clean_handle}",
                    quota_cost=1,
                    part='snippet,statistics,brandingSettings,topicDetails,status',
                    forUsername=clean_handle
                )
                
                if channel_details:
                    video_details_map = await self._get_recent_videos_for_channels([channel_details[0]['id']])
                    channel_insights = await self._get_channel_insights([channel_details[0]['id']])
                    
                    profiles = await self._build_creator_profiles(
                        channel_details, video_details_map, channel_insights
                    )
                    
                    if profiles:
                        return profiles[0]
                        
            except Exception as e:
                logger.debug(f"forUsername API call failed for {clean_handle}: {e}")
            
            logger.warning(f"No channel found for handle: {handle}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get channel by handle {handle}: {e}")
            raise ConnectorException(f"Failed to retrieve YouTube channel by handle: {str(e)}", self.platform)

    async def get_trending_channels(
        self, 
        category: Optional[ContentCategory] = None,
        region_code: Optional[str] = None,
        max_results: int = 20
    ) -> List[CreatorProfile]:
        """Get trending YouTube channels based on recent performance."""
        try:
            logger.info(f"Fetching trending channels - category: {category}, region: {region_code}")
            
            # Build search query for trending content
            if category:
                # Map our category back to YouTube search terms
                category_queries = {
                    ContentCategory.GAMING: "gaming trending",
                    ContentCategory.MUSIC: "music trending popular",
                    ContentCategory.EDUCATION: "educational trending",
                    ContentCategory.ENTERTAINMENT: "entertainment trending",
                    ContentCategory.TECHNOLOGY: "tech trending",
                    ContentCategory.SPORTS: "sports trending",
                    ContentCategory.COMEDY: "comedy trending funny",
                    ContentCategory.LIFESTYLE: "lifestyle trending",
                    ContentCategory.FOOD: "food cooking trending",
                    ContentCategory.TRAVEL: "travel trending",
                    ContentCategory.FASHION: "fashion style trending",
                    ContentCategory.HEALTH_FITNESS: "fitness health trending"
                }
                query = category_queries.get(category, "trending popular")
            else:
                query = "trending popular viral"
            
            config = YouTubeSearchConfig(
                max_results=max_results,
                order="viewCount",  # Sort by view count for trending
                region_code=region_code
            )
            
            profiles = await self.search(query, config)
            
            # Sort by a combination of recent activity and popularity
            def trending_score(profile: CreatorProfile) -> float:
                score = 0.0
                
                # Subscriber count (logarithmic scaling)
                if profile.social_metrics.followers_count:
                    score += min(100, profile.social_metrics.followers_count ** 0.2)
                
                # Recent activity bonus
                if profile.engagement_metrics.last_activity:
                    days_since = (datetime.now(timezone.utc) - profile.engagement_metrics.last_activity).days
                    if days_since <= 7:
                        score += 50
                    elif days_since <= 30:
                        score += 25
                
                # Engagement rate bonus
                if profile.engagement_metrics.engagement_rate:
                    score += profile.engagement_metrics.engagement_rate * 5
                
                # Verification bonus
                if profile.is_verified:
                    score += 20
                
                return score
            
            trending_profiles = sorted(profiles, key=trending_score, reverse=True)
            
            logger.info(f"Found {len(trending_profiles)} trending channels")
            return trending_profiles
            
        except Exception as e:
            logger.error(f"Failed to get trending channels: {e}")
            raise ConnectorException(f"Failed to retrieve trending YouTube channels: {str(e)}", self.platform)

    def get_quota_usage(self) -> Dict[str, Any]:
        """Get current API quota usage statistics."""
        return {
            "api_calls_made": self._api_call_count,
            "estimated_quota_used": self._quota_usage,
            "cache_entries": len(self._cache),
            "cache_hit_potential": len(self._cache) / max(1, self._api_call_count)
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the YouTube connector."""
        try:
            # Make a simple API call to test connectivity
            test_result = await self._api_call(
                self.youtube_service.channels().list,
                quota_cost=1,
                part='snippet',
                mine=False,
                id='UC_x5XG1OV2P6uZZ5FSM9Ttw'  # Google Developers channel
            )
            
            status = "healthy" if test_result is not None else "degraded"
            
            return {
                "status": status,
                "platform": self.platform.value,
                "api_accessible": status == "healthy",
                "quota_usage": self.get_quota_usage(),
                "last_check": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"YouTube connector health check failed: {e}")
            return {
                "status": "unhealthy",
                "platform": self.platform.value,
                "api_accessible": False,
                "error": str(e),
                "quota_usage": self.get_quota_usage(),
                "last_check": datetime.now(timezone.utc).isoformat()
            }

    async def cleanup(self) -> None:
        """Cleanup connector resources."""
        self._cache.clear()
        self._api_call_count = 0
        self._quota_usage = 0
        logger.info("YouTube connector cleanup completed")

    def __repr__(self) -> str:
        return f"YouTubeConnector(calls={self._api_call_count}, quota={self._quota_usage}, cache={len(self._cache)})"