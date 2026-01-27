"""
Script to verify DALI dev set availability.
Scans the local DALI_v1.0 folder (via DaliService) and checks if corresponding audio is available.
Downloads audio for a few entries if missing.
"""
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from services.dali_service import dali_service
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing DALI readiness check...")
    
    # 1. Check annotations
    all_ids = dali_service.get_all_ids()
    if not all_ids:
        logger.error(f"No .gz files found in {settings.DALI_ROOT}. Please ensure DALI v1.0 dataset is placed there.")
        return
    
    logger.info(f"Found {len(all_ids)} annotation files in {settings.DALI_ROOT}.")
    
    # 2. Check audio for a subset (dev set)
    limit = 5
    success_count = 0
    
    # Try to pick 5 random ones or specific ones if we had a list.
    # For now, just take first 5 from the list
    dev_ids = all_ids[:limit]
    
    for entry_id in dev_ids:
        logger.info(f"Checking audio for {entry_id}...")
        
        # This will check existence or download
        audio_path = dali_service.get_audio_path(entry_id, download_if_missing=True)
        
        if audio_path:
            logger.info(f"Audio ready: {audio_path.name}")
            success_count += 1
            
            # Verify we can read metadata
            meta = dali_service.get_metadata(entry_id)
            logger.info(f"  - Artist: {meta.get('artist')}")
            logger.info(f"  - Title: {meta.get('title')}")
        else:
            logger.warning(f"Audio missing/download failed for {entry_id}")

    logger.info(f"Readiness Check Complete. {success_count}/{limit} dev set songs ready.")
    if success_count == 0:
        logger.warning("No audio files are ready. Ensure yt-dlp is installed and internet connection is active, or manually place audio files in data/dali_audio.")

if __name__ == "__main__":
    main()
