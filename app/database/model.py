# app/database/model.py
import uuid
from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, DateTime, Float, BigInteger,
    ARRAY, Boolean, JSON, Index
)
from sqlalchemy import String, Text
from sqlalchemy.types import TypeDecorator
import uuid

# Custom UUID type that works with both PostgreSQL and SQLite
class UniversalUUID(TypeDecorator):
    impl = String
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            from sqlalchemy.dialects.postgresql import UUID
            return dialect.type_descriptor(UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(String(36))
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            return str(value)
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            return uuid.UUID(value)

# Use Text for VECTOR in both cases for now
VECTOR = Text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from app.domains.search.schemas import Platform, ContentCategory, VerificationStatus # Use our Pydantic enums

# All of our models will inherit from this Base
Base = declarative_base()

class Creator(Base):
    __tablename__ = 'creators'
    
    # Core Columns
    id = Column(UniversalUUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    handle = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    profiles = relationship("CreatorPlatformProfile", back_populates="creator", cascade="all, delete-orphan")
    aggregated_profile = relationship("CreatorAggregatedProfile", back_populates="creator", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Creator(id={self.id}, name='{self.name}')>"

class CreatorPlatformProfile(Base):
    __tablename__ = 'creator_platform_profiles'

    # Core Columns
    id = Column(UniversalUUID(), primary_key=True, default=uuid.uuid4)
    creator_id = Column(UniversalUUID(), ForeignKey('creators.id'), nullable=False)
    platform = Column(String(50), nullable=False) # Not using Enum here for db flexibility
    platform_user_id = Column(String(255), nullable=False, index=True)
    
    # Profile Data (from our Pydantic schema)
    profile_url = Column(String(512), nullable=False)
    bio = Column(Text)
    avatar_url = Column(String(512))
    banner_url = Column(String(512))
    is_verified = Column(Boolean, default=False)
    verification_status = Column(String(50), default=VerificationStatus.UNKNOWN.value)
    account_type = Column(String(100))
    is_active = Column(Boolean, default=True)

    # Social Metrics
    followers_count = Column(BigInteger)
    following_count = Column(Integer)
    total_views = Column(BigInteger)
    total_content_count = Column(Integer)
    
    # Timestamps
    account_created_at = Column(DateTime(timezone=True))
    data_scraped_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relationships
    creator = relationship("Creator", back_populates="profiles")
    videos = relationship("Video", back_populates="platform_profile", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_platform_user_id', 'platform', 'platform_user_id', unique=True),
    )

class Video(Base):
    __tablename__ = 'videos'

    id = Column(UniversalUUID(), primary_key=True, default=uuid.uuid4)
    platform_profile_id = Column(UniversalUUID(), ForeignKey('creator_platform_profiles.id'), nullable=False)
    platform_video_id = Column(String(255), nullable=False, index=True)
    
    title = Column(Text, nullable=False)
    description = Column(Text)
    published_at = Column(DateTime(timezone=True))
    
    # Metrics
    views = Column(BigInteger)
    likes = Column(BigInteger)
    comments = Column(Integer)
    duration_seconds = Column(Integer)
    
    # Metadata
    thumbnail_url = Column(String(512))
    tags = Column(JSON)  # Store as JSON instead of ARRAY for SQLite compatibility
    category = Column(String(100))
    raw_api_response = Column(JSON) # Store the original blob for re-processing
    
    # Timestamps
    data_scraped_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # Relationships
    platform_profile = relationship("CreatorPlatformProfile", back_populates="videos")
    enrichment = relationship("VideoEnrichment", back_populates="video", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_platform_video_id', 'platform_profile_id', 'platform_video_id', unique=True),
    )

class VideoEnrichment(Base):
    __tablename__ = 'video_enrichment'

    id = Column(UniversalUUID(), primary_key=True, default=uuid.uuid4)
    video_id = Column(UniversalUUID(), ForeignKey('videos.id'), nullable=False, unique=True, index=True)
    
    # NLP Insights
    language = Column(String(10))
    sentiment_label = Column(String(50))
    sentiment_score = Column(Float)
    topics = Column(JSON)  # Store as JSON instead of ARRAY for SQLite compatibility
    keywords = Column(JSON)  # Store as JSON instead of ARRAY for SQLite compatibility
    tone_label = Column(String(50))
    
    # Vision Insights
    thumbnail_text = Column(Text)
    thumbnail_tags = Column(JSON) # e.g., {"face_present": true, "text_density": 0.4}
    
    # AI/ML
    text_embedding = Column(VECTOR(384)) # all-MiniLM-L6-v2 uses 384 dimensions
    thumbnail_embedding = Column(VECTOR(512)) # CLIP uses 512 dimensions
    
    # Transcript
    transcript_path = Column(String(512)) # Path to file on local FS or S3
    
    # Status Tracking
    status = Column(String(50), default='pending', nullable=False, index=True)
    error_message = Column(Text)
    enriched_at = Column(DateTime(timezone=True))

    # Relationships
    video = relationship("Video", back_populates="enrichment")

class CreatorAggregatedProfile(Base):
    __tablename__ = 'creator_aggregated_profiles'

    id = Column(UniversalUUID(), primary_key=True, default=uuid.uuid4)
    creator_id = Column(UniversalUUID(), ForeignKey('creators.id'), nullable=False, unique=True, index=True)
    
    # Aggregated Metrics
    avg_views_per_video = Column(Float)
    avg_engagement_rate = Column(Float)
    content_frequency_days = Column(Float)
    
    # Aggregated NLP/Vision Insights
    dominant_language = Column(String(10))
    dominant_tone = Column(String(50))
    dominant_categories = Column(JSON)  # Store as JSON instead of ARRAY for SQLite compatibility
    common_topics = Column(JSON)  # Store as JSON instead of ARRAY for SQLite compatibility
    
    # Aggregated AI/ML
    profile_embedding = Column(VECTOR(384)) # Average of video text embeddings
    
    # Timestamps
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    creator = relationship("Creator", back_populates="aggregated_profile")
