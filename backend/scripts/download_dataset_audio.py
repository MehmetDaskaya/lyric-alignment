"""
Bulk Download Audio for DALI
Iterates through DALI entries, filters for English, and downloads audio via yt-dlp.
"""
import logging
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from services.dali_service import dali_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Download DALI audio")
    parser.add_argument("--limit", type=int, default=20, help="Number of songs to download")
    args = parser.parse_args()
    
    logger.info(f"Starting bulk download. Target: {args.limit} songs...")
    
    all_ids = dali_service.get_all_ids()
    downloaded_count = 0
    
    # Check existing first
    for entry_id in all_ids:
        if dali_service.get_audio_path(entry_id, download_if_missing=False):
            downloaded_count += 1
            
    logger.info(f"Initially found {downloaded_count} audio files locally.")
    if downloaded_count >= args.limit:
        logger.info("Target already met.")
        return

    # Download Loop
    pbar = tqdm(total=args.limit, initial=downloaded_count)
    
    for entry_id in all_ids:
        if downloaded_count >= args.limit:
            break
            
        # Check if already exists
        if dali_service.get_audio_path(entry_id, download_if_missing=False):
            continue
            
        # Filter English
        meta = dali_service.get_metadata(entry_id)
        lang = str(meta.get('language', '')).lower()
        if 'en' not in lang and lang != 'english':
            continue
            
        # Attempt Download
        try:
            path = dali_service.get_audio_path(entry_id, download_if_missing=True)
            if path:
                downloaded_count += 1
                pbar.update(1)
        except Exception as e:
            logger.error(f"Error downloading {entry_id}: {e}")
            
    pbar.close()
    logger.info(f"Finished. Total audio files: {downloaded_count}")

if __name__ == "__main__":
    main()
