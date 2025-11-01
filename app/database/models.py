# app/database/models.py
from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, DateTime, Float, BigInteger,
    ARRAY, JSON
)
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import VECTOR  # CORRECT: Import VECTOR from pgvector.sqlalchemy
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

# --- Base Configuration ---
Base = declarative_base()
Base.metadata.schema = "public"  # Ensure all tables are in the 'public' schema

# --- ORM Model Definitions ---

class Creator(Base):
    """Represents a unique content creator entity."""
    __tablename__ = 'creators'
    id = Column(Integer, primary_key=True)
    youtube_channel_id = Column(Text, unique=True)
    name = Column(Text)
    description = Column(Text)
    country = Column(Text)
    language = Column(Text)
    subscribers = Column(Integer)
    total_views = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    videos = relationship("Video", back_populates="creator", cascade="all, delete-orphan")
    profile = relationship("CreatorProfileDB", back_populates="creator", uselist=False, cascade="all, delete-orphan")

class Video(Base):
    """
    Represents a single video. This model is an exact mirror of the SQL
    CREATE TABLE script, resolving the previous 'UndefinedColumnError'.
    """
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True)
    youtube_video_id = Column(Text, unique=True)
    creator_id = Column(Integer, ForeignKey('public.creators.id', ondelete="CASCADE"))
    title = Column(Text)
    description = Column(Text)
    published_at = Column(DateTime(timezone=True))
    views = Column(Integer)
    likes = Column(Integer)
    comments = Column(Integer)
    category = Column(Text)
    duration_seconds = Column(Integer)
    thumbnail_url = Column(Text)

    creator = relationship("Creator", back_populates="videos")
    enrichment = relationship("VideoEnrichment", back_populates="video", uselist=False, cascade="all, delete-orphan")

class VideoEnrichment(Base):
    """
    Stores all derived NLP/AI insights for a video.
    This model is an exact mirror of its corresponding CREATE TABLE script.
    """
    __tablename__ = 'video_enrichment'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey('public.videos.id', ondelete="CASCADE"), unique=True)
    
    # Timestamps
    enriched_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Core NLP Fields
    sentiment = Column(Text)
    language = Column(Text)
    content_type = Column(Text)
    
    # Array/JSON Fields
    topics = Column(ARRAY(Text))
    keywords = Column(ARRAY(Text))
    confidence_scores = Column(JSONB)
    
    # CORRECT: The Python attribute is 'enrichment_metadata' to avoid conflict.
    # The 'name' argument maps it to the 'metadata' column in the database.
    enrichment_metadata = Column(JSONB, name="metadata")
    
    # Calculated Scores
    quality_score = Column(Float)
    engagement_prediction = Column(Float)
    
    # Vector Embedding
    embedding = Column(VECTOR(384)) # Correct dimension for all-MiniLM-L6-v2

    video = relationship("Video", back_populates="enrichment")

class CreatorProfileDB(Base):
    """Stores aggregated, pre-computed stats for a creator."""
    __tablename__ = 'creator_profiles'
    
    id = Column(Integer, primary_key=True)
    creator_id = Column(Integer, ForeignKey('public.creators.id', ondelete="CASCADE"), unique=True)
    avg_views = Column(Float)
    avg_engagement_rate = Column(Float)
    embedding = Column(VECTOR(384))  # Text embedding for semantic search
    dominant_language = Column(Text)
    dominant_tone = Column(Text)
    topics = Column(ARRAY(Text))
    profile_embedding = Column(VECTOR(384))
    avg_community_sentiment = Column(Float)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    creator = relationship("Creator", back_populates="profile")