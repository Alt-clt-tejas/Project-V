# test_enrichment.py
import asyncio
import logging
from pprint import pprint
import time

# Setup logging
from app.utils.logging_config import setup_logging
setup_logging(log_level="INFO")

from app.database.session import AsyncSessionFactory
from app.services.enrichment_service import (
    EnrichmentService,
    model_manager
)

# Configuration
VIDEO_IDS_TO_TEST = [1, 2, 3, 4]
MAX_CONCURRENT_TASKS = 4 # How many videos to process in parallel

async def enrich_single_video(video_id: int):
    """
    This function represents a single, isolated unit of work.
    It creates its own database session and enrichment service instance.
    """
    # Each task gets its own session from the global session factory.
    # This is the key to preventing connection conflicts.
    async with AsyncSessionFactory() as session:
        try:
            # The service is now scoped to this single task and session.
            service = EnrichmentService(db_session=session)
            
            result = await service.enrich_video(
                video_id=video_id,
                force_refresh=True
            )
            return result
        except Exception as e:
            logging.error(f"Failed to enrich video {video_id}", exc_info=True)
            # In case of failure, return the exception to be handled by the main orchestrator.
            return e

async def main():
    """
    Orchestrates the batch enrichment process, managing concurrency.
    """
    start_time = time.time()
    print("--- Starting Professional Batch Enrichment Test ---")
    print(f"Targeting Video IDs: {VIDEO_IDS_TO_TEST}")

    # Preload models once, before starting any concurrent work.
    print("Preloading NLP models...")
    model_manager.preload_all_models()
    print("Models preloaded successfully.")

    # Create and run tasks concurrently using a semaphore for control.
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    tasks = []

    async def run_with_semaphore(video_id: int):
        async with semaphore:
            return await enrich_single_video(video_id)

    for video_id in VIDEO_IDS_TO_TEST:
        tasks.append(run_with_semaphore(video_id))
    
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"\n--- Batch Enrichment Complete ---")
    print(f"Successfully processed {len(results)} videos in {end_time - start_time:.2f} seconds.")
    
    print("\n--- Detailed Enrichment Results ---")
    for i, result in enumerate(results):
        video_id = VIDEO_IDS_TO_TEST[i]
        print(f"\n----- Result for Video ID: {video_id} -----")
        
        # Check if the result is an exception or a successful EnrichmentResult object
        if isinstance(result, Exception):
            print(f"!! ENRICHMENT FAILED FOR THIS VIDEO !!")
            print(f"Error Type: {type(result).__name__}")
            print(f"Error Details: {result}")
        else:
            # Use the .to_dict() method for clean printing
            pprint(result.to_dict())

if __name__ == "__main__":
    asyncio.run(main())