import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from services.batch_processor import batch_processor
from config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_batch():
    logger.info("Starting Batch Verification...")
    
    # Initialize DALI (loads metadata from .gz files)
    try:
        from utils.dali_loader import dali_loader
        dali_loader.download_dataset_metadata()
    except Exception as e:
        logger.error(f"Failed to initialize DALI: {e}")
        return

    # Find songs
    songs = dali_loader.list_songs()
    if not songs:
        logger.error("No songs found in DALI dataset. Run create_mock_dali.py first.")
        return
        
    song_id = songs[0]['id']
    logger.info(f"Testing with song: {song_id}")
    
    # Create job
    job = batch_processor.create_job(
        song_ids=[song_id],
        models=['dtw'], # Test DTW first as it had the MFCC config issue
        use_source_separation=False # Faster for testing
    )
    
    logger.info(f"Created Job: {job.id}")
    
    # Start job
    success = await batch_processor.start_job(job.id)
    if not success:
        logger.error("Failed to start job")
        return
        
    # Wait for completion
    while job.status == "running":
        await asyncio.sleep(1)
        logger.info(f"Job Status: {job.status}, Progress: {job.progress}/{job.total}")
        
    logger.info(f"Job Finished with status: {job.status}")
    
    # Inspect results
    if job.status == "completed":
        result = job.results[0]
        logger.info(f"Song Result Status: {result.status}")
        if result.error:
             logger.error(f"Song Error: {result.error}")
        
        # Check metrics
        metrics = result.metrics.get('dtw', {})
        logger.info(f"DTW Metrics: {metrics}")
        
        if 'mae' in metrics and 'accuracy' in metrics:
            logger.info("SUCCESS: Metrics calculated!")
        else:
            logger.error("FAILURE: Metrics missing!")
            
        # Check words
        words = result.results.get('dtw', {}).get('words', [])
        logger.info(f"Aligned Words: {len(words)}")
    else:
        logger.error(f"Job Failed: {job.error}")

if __name__ == "__main__":
    asyncio.run(verify_batch())
