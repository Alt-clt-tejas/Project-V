# app/api/collect.py
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.connectors.youtube_connector import YouTubeConnector
from app.api.dependencies import get_youtube_connector
from app.database.session import get_db_session
from app.database.models import Creator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collect", tags=["Collection"])

class YouTubeChannelCollectRequest(BaseModel):
    channel_id: str
    include_detailed_stats: bool = True
    max_videos: Optional[int] = 50

@router.post("/youtube-channel")
async def collect_youtube_channel(
    request: YouTubeChannelCollectRequest,
    youtube_connector: YouTubeConnector = Depends(get_youtube_connector),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Collect data for a specific YouTube channel and save it to the database.
    Maps CreatorProfile fields to your database schema.
    """
    try:
        # Step 1: Fetch from YouTube API
        logger.info(f"Fetching channel data for: {request.channel_id}")
        profile = await youtube_connector.get_channel_by_id(request.channel_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Channel not found on YouTube")
        
        # Step 2: Check if creator already exists
        result = await db.execute(
            select(Creator).where(Creator.youtube_channel_id == request.channel_id)
        )
        existing_creator = result.scalar_one_or_none()
        
        if existing_creator:
            # Update existing creator
            existing_creator.name = profile.name
            existing_creator.description = profile.bio
            existing_creator.subscribers = profile.social_metrics.followers_count
            existing_creator.total_views = profile.social_metrics.total_views
            existing_creator.country = profile.metadata.country
            existing_creator.language = profile.metadata.languages[0] if profile.metadata.languages else None
            
            creator = existing_creator
            action = "updated"
        else:
            # Create new creator - map CreatorProfile fields to database columns
            creator = Creator(
                youtube_channel_id=request.channel_id,
                name=profile.name,
                description=profile.bio,
                country=profile.metadata.country,
                language=profile.metadata.languages[0] if profile.metadata.languages else None,
                subscribers=profile.social_metrics.followers_count,
                total_views=profile.social_metrics.total_views
            )
            db.add(creator)
            action = "created"
        
        # Step 3: Commit to database
        await db.commit()
        await db.refresh(creator)
        
        logger.info(f"Successfully {action} creator: {creator.name} (ID: {creator.id})")
        
        return {
            "status": "success",
            "action": action,
            "message": f"Successfully {action} channel: {profile.name}",
            "creator_id": creator.id,
            "database_record": {
                "id": creator.id,
                "youtube_channel_id": creator.youtube_channel_id,
                "name": creator.name,
                "subscribers": creator.subscribers,
                "total_views": creator.total_views
            },
            "profile": profile
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error collecting YouTube channel data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to collect channel: {str(e)}")


@router.post("/youtube-channels-batch")
async def collect_multiple_channels(
    channel_ids: list[str],
    youtube_connector: YouTubeConnector = Depends(get_youtube_connector),
    db: AsyncSession = Depends(get_db_session)
):
    """Collect multiple YouTube channels in one request."""
    results = []
    successful = 0
    failed = 0
    
    for channel_id in channel_ids:
        try:
            # Fetch from YouTube
            profile = await youtube_connector.get_channel_by_id(channel_id)
            
            if not profile:
                results.append({
                    "channel_id": channel_id,
                    "status": "failed",
                    "error": "Channel not found"
                })
                failed += 1
                continue
            
            # Check if exists
            result = await db.execute(
                select(Creator).where(Creator.youtube_channel_id == channel_id)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                existing.name = profile.name
                existing.subscribers = profile.social_metrics.followers_count
                existing.total_views = profile.social_metrics.total_views
                creator_id = existing.id
                action = "updated"
            else:
                creator = Creator(
                    youtube_channel_id=channel_id,
                    name=profile.name,
                    description=profile.bio,
                    country=profile.metadata.country,
                    language=profile.metadata.languages[0] if profile.metadata.languages else None,
                    subscribers=profile.social_metrics.followers_count,
                    total_views=profile.social_metrics.total_views
                )
                db.add(creator)
                await db.flush()
                creator_id = creator.id
                action = "created"
            
            await db.commit()
            
            results.append({
                "channel_id": channel_id,
                "status": "success",
                "action": action,
                "creator_id": creator_id,
                "name": profile.name,
                "subscribers": profile.social_metrics.followers_count
            })
            successful += 1
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to collect channel {channel_id}: {e}")
            results.append({
                "channel_id": channel_id,
                "status": "failed",
                "error": str(e)
            })
            failed += 1
    
    return {
        "status": "completed",
        "total": len(channel_ids),
        "successful": successful,
        "failed": failed,
        "results": results
    }

@router.get("/creators/count")
async def count_creators(db: AsyncSession = Depends(get_db_session)):
    """Debug endpoint to count total creators in database"""
    from sqlalchemy import func
    
    result = await db.execute(select(func.count(Creator.id)))
    count = result.scalar()
    
    # Also get a sample
    sample_result = await db.execute(select(Creator).limit(5))
    samples = sample_result.scalars().all()
    
    return {
        "total_creators": count,
        "sample_creators": [
            {
                "id": c.id, 
                "name": c.name, 
                "youtube_channel_id": c.youtube_channel_id,
                "subscribers": c.subscribers,
                "total_views": c.total_views
            }
            for c in samples
        ]
    }