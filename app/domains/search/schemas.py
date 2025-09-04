# app/domains/search/schemas.py
from enum import Enum
from typing import Optional, Dict, Any, List, Self
from datetime import datetime

from pydantic import BaseModel, HttpUrl, Field, validator, model_validator


class Platform(str, Enum):
    """Enumeration of supported social media platforms."""
    YOUTUBE = "YouTube"
    TWITTER = "Twitter"
    INSTAGRAM = "Instagram"
    TIKTOK = "TikTok"
    LINKEDIN = "LinkedIn"
    TWITCH = "Twitch"
    FACEBOOK = "Facebook"
    NEWS = "News"
    WEB = "Web"
    PODCAST = "Podcast"


class VerificationStatus(str, Enum):
    """Enhanced verification status indicators."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    UNKNOWN = "unknown"
    PARTNER = "partner"  # Platform-specific partnership status


class ContentCategory(str, Enum):
    """Content categories for better classification."""
    ENTERTAINMENT = "entertainment"
    EDUCATION = "education"
    TECHNOLOGY = "technology"
    GAMING = "gaming"
    LIFESTYLE = "lifestyle"
    BUSINESS = "business"
    HEALTH_FITNESS = "health_fitness"
    TRAVEL = "travel"
    FOOD = "food"
    FASHION = "fashion"
    MUSIC = "music"
    SPORTS = "sports"
    NEWS = "news"
    SCIENCE = "science"
    ART = "art"
    COMEDY = "comedy"
    FILM = "film"  # Added for YouTube film content
    OTHER = "other"


class EngagementMetrics(BaseModel):
    """Detailed engagement metrics for creators."""
    engagement_rate: Optional[float] = Field(None, ge=0.0, le=100.0, description="Engagement rate as percentage")
    avg_views_per_content: Optional[float] = Field(None, ge=0.0, description="Average views per piece of content")
    avg_likes_per_content: Optional[float] = Field(None, ge=0.0, description="Average likes per piece of content")
    avg_comments_per_content: Optional[float] = Field(None, ge=0.0, description="Average comments per piece of content")
    content_frequency: Optional[str] = Field(None, description="Upload/posting frequency (e.g., 'Daily', 'Weekly')")
    last_activity: Optional[datetime] = Field(None, description="Last content upload/post date")
    
    # Enhanced YouTube-specific engagement metrics
    most_popular_content_views: Optional[int] = Field(None, ge=0, description="Views on most popular content")
    content_consistency_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Content consistency score")
    upload_frequency_days: Optional[float] = Field(None, ge=0.0, description="Average days between uploads")


class SocialMetrics(BaseModel):
    """Comprehensive social media metrics."""
    followers_count: Optional[int] = Field(None, ge=0, description="Number of followers/subscribers")
    following_count: Optional[int] = Field(None, ge=0, description="Number of accounts following")
    total_views: Optional[int] = Field(None, ge=0, description="Total profile/channel views")
    total_content_count: Optional[int] = Field(None, ge=0, description="Total number of posts/videos")
    likes_count: Optional[int] = Field(None, ge=0, description="Total likes received")
    comments_count: Optional[int] = Field(None, ge=0, description="Total comments received")
    shares_count: Optional[int] = Field(None, ge=0, description="Total shares/reposts")
    
    # Platform-specific metrics
    video_count: Optional[int] = Field(None, ge=0, description="Number of videos (YouTube)")
    playlist_count: Optional[int] = Field(None, ge=0, description="Number of playlists (YouTube)")
    tweets_count: Optional[int] = Field(None, ge=0, description="Number of tweets (Twitter)")
    posts_count: Optional[int] = Field(None, ge=0, description="Number of posts (Instagram)")
    
    # Enhanced YouTube metrics
    avg_views_per_video: Optional[float] = Field(None, ge=0.0, description="Average views per video (YouTube)")
    channel_age_days: Optional[int] = Field(None, ge=0, description="Channel age in days (YouTube)")
    subscriber_growth_rate: Optional[float] = Field(None, description="Estimated subscriber growth rate")


class ProfileMetadata(BaseModel):
    """Extended metadata for creator profiles."""
    categories: List[ContentCategory] = Field(default_factory=list, description="Content categories")
    tags: List[str] = Field(default_factory=list, description="Profile tags/keywords")
    languages: List[str] = Field(default_factory=list, description="Content languages (ISO codes)")
    country: Optional[str] = Field(None, description="Creator's country (ISO code)")
    city: Optional[str] = Field(None, description="Creator's city")
    website: Optional[HttpUrl] = Field(None, description="Creator's website")
    business_email: Optional[str] = Field(None, description="Business contact email")
    
    # Platform-specific data
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Platform-specific additional data")
    
    # Enhanced YouTube-specific fields
    topic_categories: List[str] = Field(default_factory=list, description="YouTube topic categories")
    keywords: List[str] = Field(default_factory=list, description="Channel keywords/tags")
    
    # Data quality indicators
    profile_completeness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Profile completeness score")
    data_freshness: Optional[datetime] = Field(None, description="When the data was last updated")
    reliability_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data reliability score")


class CreatorProfile(BaseModel):
    """
    Enhanced standardized representation of a creator's profile
    from any platform with comprehensive metrics and metadata.
    """
    # Core identification
    platform: Platform
    platform_id: Optional[str] = Field(None, description="Platform-specific unique identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Display name")
    handle: str = Field(..., min_length=1, max_length=100, description="Username/handle")
    profile_url: HttpUrl = Field(..., description="Direct link to profile")
    
    # Profile information
    bio: Optional[str] = Field(None, max_length=5000, description="Profile description/bio")
    avatar_url: Optional[HttpUrl] = Field(None, description="Profile picture URL")
    banner_url: Optional[HttpUrl] = Field(None, description="Profile banner/cover image URL")
    
    # Verification and status
    is_verified: bool = Field(default=False, description="Platform verification status")
    verification_status: VerificationStatus = Field(default=VerificationStatus.UNKNOWN)
    account_type: Optional[str] = Field(None, description="Account type (personal, business, enterprise, etc.)")
    is_active: bool = Field(default=True, description="Whether the account appears to be active")
    
    # Comprehensive metrics
    social_metrics: SocialMetrics = Field(default_factory=SocialMetrics)
    engagement_metrics: EngagementMetrics = Field(default_factory=EngagementMetrics)
    
    # Enhanced metadata
    metadata: ProfileMetadata = Field(default_factory=ProfileMetadata)
    
    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Account creation date")
    updated_at: Optional[datetime] = Field(None, description="Last profile update")
    scraped_at: Optional[datetime] = Field(default_factory=datetime.now, description="When this data was collected")
    
    # Backward compatibility properties
    @property
    def followers_count(self) -> Optional[int]:
        return self.social_metrics.followers_count
    
    @property
    def view_count(self) -> Optional[int]:
        return self.social_metrics.total_views
    
    @property
    def video_count(self) -> Optional[int]:
        return self.social_metrics.video_count
    
    # Enhanced properties for YouTube connector compatibility
    @property
    def subscriber_count(self) -> Optional[int]:
        """Alias for followers_count for YouTube compatibility."""
        return self.social_metrics.followers_count
    
    @property
    def avg_views_per_video(self) -> Optional[float]:
        """YouTube-specific average views per video."""
        return self.social_metrics.avg_views_per_video or self.engagement_metrics.avg_views_per_content
    
    @property
    def channel_age_days(self) -> Optional[int]:
        """YouTube channel age in days."""
        return self.social_metrics.channel_age_days
    
    @validator('name', 'handle', 'bio', pre=True)
    def strip_and_prepare_strings(cls, v):
        if isinstance(v, str):
            stripped = v.strip()
            # For bio, an empty string should become None
            if not stripped:
                return None
            return stripped
        return v

    @validator('name', 'handle')
    def ensure_not_none(cls, v):
        # This validator runs after the one above.
        # If the string was stripped to nothing (and became None), this will catch it.
        if v is None:
            raise ValueError('must not be empty')
        return v
    
    @model_validator(mode='after')
    def validate_profile_consistency(self) -> Self:
        """Ensure profile data is consistent."""
        if self.platform == Platform.YOUTUBE and self.profile_url:
            url_str = str(self.profile_url)
            if 'youtube.com' not in url_str and 'youtu.be' not in url_str:
                raise ValueError('YouTube profiles must have youtube.com URLs')
        return self
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class SearchMatchDetails(BaseModel):
    """Detailed information about how a profile matched the search query."""
    name_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Name similarity score")
    handle_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Handle similarity score")
    bio_relevance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Bio relevance score")
    keyword_matches: int = Field(default=0, ge=0, description="Number of keyword matches")
    exact_matches: List[str] = Field(default_factory=list, description="Exact keyword matches found")
    partial_matches: List[str] = Field(default_factory=list, description="Partial keyword matches found")
    social_signals_boost: Optional[float] = Field(None, ge=0.0, le=1.0, description="Social signals contribution")
    match_reasons: List[str] = Field(default_factory=list, description="Human-readable match explanations")
    matched_fields: List[str] = Field(default_factory=list, description="Fields where matches were found")


class SearchResult(BaseModel):
    """
    Enhanced search result with detailed matching information
    and ranking factors.
    """
    profile: CreatorProfile
    match_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall match confidence score")
    relevance_score: Optional[float] = Field(None, ge=0.0, description="Raw relevance score before normalization")
    ranking_position: Optional[int] = Field(None, ge=1, description="Position in search results")
    
    # Detailed matching information
    match_details: Optional[SearchMatchDetails] = Field(None, description="Detailed match analysis")
    
    # Search context
    search_query: Optional[str] = Field(None, description="Original search query")
    search_type: Optional['SearchType'] = Field(None, description="Type of search performed")
    
    # Quality indicators
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality of profile data")
    freshness_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="How recent the data is")
    
    class Config:
        use_enum_values = True


class SearchType(str, Enum):
    """Enhanced enumeration for different types of searches."""
    CREATOR = "creator"
    TOPIC = "topic"
    HASHTAG = "hashtag"
    CATEGORY = "category"
    LOCATION = "location"
    SIMILAR = "similar"
    TRENDING = "trending"


class SearchFilter(BaseModel):
    """Advanced filtering options for search queries."""
    platforms: List[Platform] = Field(default_factory=list)
    categories: List[ContentCategory] = Field(default_factory=list)
    countries: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    
    min_followers: Optional[int] = Field(None, ge=0)
    max_followers: Optional[int] = Field(None, ge=0)
    min_engagement_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    max_engagement_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Enhanced YouTube-specific filters
    min_video_count: Optional[int] = Field(None, ge=0)
    max_video_count: Optional[int] = Field(None, ge=0)
    min_avg_views: Optional[int] = Field(None, ge=0)
    max_avg_views: Optional[int] = Field(None, ge=0)
    min_channel_age_days: Optional[int] = Field(None, ge=0)
    max_channel_age_days: Optional[int] = Field(None, ge=0)
    
    verified_only: bool = Field(default=False)
    active_only: bool = Field(default=True)
    
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    last_active_after: Optional[datetime] = None
    
    # YouTube-specific filters
    upload_frequency: Optional[str] = Field(None, description="Filter by upload frequency")
    content_consistency_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def validate_ranges(self) -> Self:
        if self.min_followers is not None and self.max_followers is not None and self.max_followers < self.min_followers:
            raise ValueError('max_followers must be greater than or equal to min_followers')
        if self.min_engagement_rate is not None and self.max_engagement_rate is not None and self.max_engagement_rate < self.min_engagement_rate:
            raise ValueError('max_engagement_rate must be greater than or equal to min_engagement_rate')
        if self.min_video_count is not None and self.max_video_count is not None and self.max_video_count < self.min_video_count:
            raise ValueError('max_video_count must be greater than or equal to min_video_count')
        if self.min_avg_views is not None and self.max_avg_views is not None and self.max_avg_views < self.min_avg_views:
            raise ValueError('max_avg_views must be greater than or equal to min_avg_views')
        if self.min_channel_age_days is not None and self.max_channel_age_days is not None and self.max_channel_age_days < self.min_channel_age_days:
            raise ValueError('max_channel_age_days must be greater than or equal to min_channel_age_days')
        return self


class SearchQuery(BaseModel):
    """Comprehensive search query with all parameters."""
    query: str = Field(..., min_length=1, max_length=500)
    search_type: SearchType = Field(default=SearchType.TOPIC)
    filters: SearchFilter = Field(default_factory=SearchFilter)
    
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc", pattern=r"^(asc|desc)$")
    
    include_inactive: bool = Field(default=False)
    boost_verified: bool = Field(default=True)
    boost_popular: bool = Field(default=True)

    @validator('query', pre=True)
    def validate_query(cls, v):
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError('Search query cannot be empty')
            return stripped
        return v


class SearchResponse(BaseModel):
    """Complete search response with results and metadata."""
    results: List[SearchResult]
    total_count: int = Field(ge=0, description="Total number of matching results")
    page_count: int = Field(ge=0, description="Total number of pages")
    current_page: int = Field(ge=1, description="Current page number")
    
    query: SearchQuery
    search_duration_ms: Optional[float] = Field(None, ge=0)
    
    platform_breakdown: Dict[Platform, int] = Field(default_factory=dict)
    category_breakdown: Dict[ContentCategory, int] = Field(default_factory=dict)
    
    suggestions: List[str] = Field(default_factory=list)
    related_queries: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


# YouTube-specific configuration and response models for the connector
class YouTubeSearchConfig(BaseModel):
    """Configuration for YouTube search operations."""
    max_results: int = Field(default=25, ge=1, le=50)
    order: str = Field(default="relevance", pattern=r"^(relevance|viewCount|date|rating|title|videoCount)$")
    region_code: Optional[str] = Field(None, min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code")
    relevance_language: Optional[str] = Field(None, min_length=2, max_length=5, description="ISO 639-1 language code")
    published_after: Optional[datetime] = Field(None, description="Only return channels created after this date")
    video_definition: Optional[str] = Field(None, pattern=r"^(any|high|standard)$")
    video_duration: Optional[str] = Field(None, pattern=r"^(any|long|medium|short)$")


class YouTubeMetrics(BaseModel):
    """Detailed YouTube-specific metrics."""
    subscriber_count: Optional[int] = Field(None, ge=0)
    total_views: Optional[int] = Field(None, ge=0)
    video_count: Optional[int] = Field(None, ge=0)
    avg_views_per_video: Optional[float] = Field(None, ge=0.0)
    engagement_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    upload_frequency: Optional[str] = None
    last_upload: Optional[datetime] = None
    most_popular_video_views: Optional[int] = Field(None, ge=0)
    channel_age_days: Optional[int] = Field(None, ge=0)
    content_consistency_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class YouTubeConnectorStats(BaseModel):
    """Statistics and health information for the YouTube connector."""
    api_calls_made: int = Field(ge=0)
    estimated_quota_used: int = Field(ge=0)
    cache_entries: int = Field(ge=0)
    cache_hit_potential: float = Field(ge=0.0, le=1.0)
    status: str = Field(pattern=r"^(healthy|degraded|unhealthy)$")
    last_check: datetime
    api_accessible: bool = Field(default=True)