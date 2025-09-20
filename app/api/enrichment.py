# app/api/enrichment.py
import logging
from typing import List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.session import get_db_session
from app.services.enrichment_service import EnrichmentService, EnrichmentResult, EnrichmentStatus
from app.database.models import VideoEnrichment, Video

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enrich", tags=["Enrichment"])

# Response Models
class EnrichmentResponse(BaseModel):
    status: str
    message: str
    video_id: int
    enrichment_id: Optional[int] = None
    processing_time: Optional[float] = None
    cached: bool = False

class EnrichmentDetails(BaseModel):
    video_id: int
    language: str
    sentiment: dict
    keywords: List[str]
    topics: List[str]
    content_type: str
    quality_score: float
    engagement_prediction: float
    processing_time: float
    confidence_scores: dict
    metadata: dict
    cached: bool = False

class BatchEnrichmentRequest(BaseModel):
    video_ids: List[int] = Field(..., min_items=1, max_items=50, description="List of video IDs to enrich")
    force_refresh: bool = Field(False, description="Force refresh even if cached results exist")

class BatchEnrichmentResponse(BaseModel):
    status: str
    message: str
    total_videos: int
    successful: int
    failed: int
    results: List[EnrichmentResponse]
    processing_time: float

# Dependency for Redis (if available)
async def get_redis_client() -> Optional[Any]:
    """Get Redis client if available, otherwise return None"""
    try:
        # This should be configured in your app startup
        # For now, we'll return None to disable caching
        return None
    except Exception:
        return None

@router.post("/{video_id}", response_model=EnrichmentResponse)
async def trigger_video_enrichment(
    video_id: int,
    force_refresh: bool = Query(False, description="Force refresh even if cached"),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db_session),
    redis_client: Optional[Any] = Depends(get_redis_client)
):
    """
    Manually triggers the enrichment pipeline for a specific video ID.
    
    - **video_id**: The ID of the video to enrich
    - **force_refresh**: Bypass cache and force re-processing
    
    Returns enrichment results with processing details.
    """
    start_time = datetime.now()
    
    try:
        # Verify video exists first
        result = await db.execute(select(Video).where(Video.id == video_id))
        video = result.scalar_one_or_none()
        
        if not video:
            raise HTTPException(
                status_code=404, 
                detail=f"Video with ID {video_id} not found"
            )
        
        # Initialize service
        service = EnrichmentService(db, redis_client)
        
        # Run enrichment
        logger.info(f"Starting enrichment for video_id: {video_id}")
        enrichment_result = await service.enrich_video(video_id, force_refresh=force_refresh)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Check if result was cached
        was_cached = enrichment_result.metadata.get('status') == EnrichmentStatus.CACHED.value
        
        return EnrichmentResponse(
            status="success",
            message=f"Enrichment complete for video {video_id}" + (" (cached)" if was_cached else ""),
            video_id=video_id,
            processing_time=processing_time,
            cached=was_cached
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error for video_id {video_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Enrichment endpoint failed for video_id {video_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred during enrichment: {str(e)}"
        )

@router.get("/{video_id}/details", response_model=EnrichmentDetails)
async def get_enrichment_details(
    video_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get detailed enrichment results for a specific video.
    
    - **video_id**: The ID of the video to get enrichment details for
    
    Returns comprehensive enrichment analysis results.
    """
    try:
        # Get the most recent enrichment for this video
        result = await db.execute(
            select(VideoEnrichment)
            .where(VideoEnrichment.video_id == video_id)
            .order_by(VideoEnrichment.created_at.desc())
            .limit(1)
        )
        enrichment = result.scalar_one_or_none()
        
        if not enrichment:
            raise HTTPException(
                status_code=404,
                detail=f"No enrichment data found for video {video_id}"
            )
        
        # Convert to response model
        return EnrichmentDetails(
            video_id=enrichment.video_id,
            language=getattr(enrichment, 'language', 'unknown'),
            sentiment=enrichment.sentiment if isinstance(enrichment.sentiment, dict) else {"label": enrichment.sentiment},
            keywords=getattr(enrichment, 'keywords', []),
            topics=enrichment.topics or [],
            content_type=getattr(enrichment, 'content_type', 'other'),
            quality_score=getattr(enrichment, 'quality_score', 0.0),
            engagement_prediction=getattr(enrichment, 'engagement_prediction', 0.0),
            processing_time=0.0,  # Historical data doesn't have this
            confidence_scores=getattr(enrichment, 'confidence_scores', {}),
            metadata=getattr(enrichment, 'metadata', {}),
            cached=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get enrichment details for video_id {video_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while retrieving enrichment details."
        )

@router.post("/batch", response_model=BatchEnrichmentResponse)
async def batch_enrich_videos(
    request: BatchEnrichmentRequest,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db_session),
    redis_client: Optional[Any] = Depends(get_redis_client)
):
    """
    Enrich multiple videos in a single request.
    
    - **video_ids**: List of video IDs to enrich (max 50)
    - **force_refresh**: Force refresh for all videos
    
    Processes videos concurrently for better performance.
    """
    start_time = datetime.now()
    
    try:
        # Validate all video IDs exist
        result = await db.execute(
            select(Video.id).where(Video.id.in_(request.video_ids))
        )
        existing_ids = {row[0] for row in result.fetchall()}
        missing_ids = set(request.video_ids) - existing_ids
        
        if missing_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Videos not found: {sorted(missing_ids)}"
            )
        
        # Initialize service
        service = EnrichmentService(db, redis_client)
        
        # Process videos concurrently
        logger.info(f"Starting batch enrichment for {len(request.video_ids)} videos")
        
        # For batch processing, we'll process each video individually to get detailed results
        results = []
        successful = 0
        failed = 0
        
        for video_id in request.video_ids:
            try:
                enrichment_result = await service.enrich_video(
                    video_id, 
                    force_refresh=request.force_refresh
                )
                
                was_cached = enrichment_result.metadata.get('status') == EnrichmentStatus.CACHED.value
                
                results.append(EnrichmentResponse(
                    status="success",
                    message=f"Enrichment complete for video {video_id}",
                    video_id=video_id,
                    processing_time=enrichment_result.processing_time,
                    cached=was_cached
                ))
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to enrich video {video_id}: {e}")
                results.append(EnrichmentResponse(
                    status="error",
                    message=f"Failed to enrich video {video_id}: {str(e)}",
                    video_id=video_id,
                    processing_time=0.0,
                    cached=False
                ))
                failed += 1
        
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchEnrichmentResponse(
            status="completed",
            message=f"Batch enrichment completed: {successful} successful, {failed} failed",
            total_videos=len(request.video_ids),
            successful=successful,
            failed=failed,
            results=results,
            processing_time=total_processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Batch enrichment failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch enrichment failed: {str(e)}"
        )

@router.get("/status/{video_id}")
async def get_enrichment_status(
    video_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Check if a video has been enriched and get basic status information.
    
    - **video_id**: The ID of the video to check
    
    Returns enrichment status and basic metadata.
    """
    try:
        # Check if video exists
        video_result = await db.execute(select(Video).where(Video.id == video_id))
        video = video_result.scalar_one_or_none()
        
        if not video:
            raise HTTPException(
                status_code=404,
                detail=f"Video with ID {video_id} not found"
            )
        
        # Check for existing enrichment
        enrichment_result = await db.execute(
            select(VideoEnrichment)
            .where(VideoEnrichment.video_id == video_id)
            .order_by(VideoEnrichment.created_at.desc())
            .limit(1)
        )
        enrichment = enrichment_result.scalar_one_or_none()
        
        if not enrichment:
            return {
                "video_id": video_id,
                "enriched": False,
                "status": "not_processed",
                "message": "Video has not been enriched yet"
            }
        
        return {
            "video_id": video_id,
            "enriched": True,
            "status": "completed",
            "enriched_at": enrichment.created_at.isoformat(),
            "sentiment": enrichment.sentiment,
            "topics_count": len(enrichment.topics) if enrichment.topics else 0,
            "keywords_count": len(getattr(enrichment, 'keywords', [])),
            "quality_score": getattr(enrichment, 'quality_score', None),
            "message": "Video has been successfully enriched"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get enrichment status for video_id {video_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while checking enrichment status."
        )

@router.delete("/{video_id}/enrichment")
async def clear_video_enrichment(
    video_id: int,
    db: AsyncSession = Depends(get_db_session),
    redis_client: Optional[Any] = Depends(get_redis_client)
):
    """
    Clear enrichment data for a specific video (useful for testing).
    
    - **video_id**: The ID of the video to clear enrichment data for
    
    Removes both database records and cache entries.
    """
    try:
        # Delete from database
        result = await db.execute(
            select(VideoEnrichment).where(VideoEnrichment.video_id == video_id)
        )
        enrichments = result.scalars().all()
        
        if not enrichments:
            raise HTTPException(
                status_code=404,
                detail=f"No enrichment data found for video {video_id}"
            )
        
        for enrichment in enrichments:
            await db.delete(enrichment)
        
        await db.commit()
        
        # Clear cache if Redis is available
        if redis_client:
            try:
                # Clear all cache entries for this video
                cache_pattern = f"enrichment:{video_id}:*"
                keys = await redis_client.keys(cache_pattern)
                if keys:
                    await redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries for video {video_id}")
            except Exception as e:
                logger.warning(f"Failed to clear cache for video {video_id}: {e}")
        
        return {
            "status": "success",
            "message": f"Cleared enrichment data for video {video_id}",
            "video_id": video_id,
            "deleted_records": len(enrichments)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to clear enrichment for video_id {video_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while clearing enrichment data."
        )

@router.get("/health")
async def enrichment_health_check():
    """
    Health check endpoint for the enrichment service.
    
    Returns status of various components and models.
    """
    try:
        from app.services.enrichment_service import model_manager
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        # Check model availability (without loading them)
        model_names = ['yake_extractor', 'embedding_model']
        
        for model_name in model_names:
            try:
                # Just check if model can be loaded (this might load it)
                model_manager.get_model(model_name)
                health_status["models"][model_name] = "available"
            except Exception as e:
                health_status["models"][model_name] = f"error: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }