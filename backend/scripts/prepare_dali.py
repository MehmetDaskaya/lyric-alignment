"""
Script to prepare DALI dev set
Downloads a few songs and their annotations for development
"""
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dali_loader import dali_loader
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Setting up DALI dev set...")
    
    # Initialize metadata
    try:
        dali_loader.download_dataset_metadata()
    except ImportError:
        logger.error("DALI library not found. Please install: pip install DALI-dataset yt-dlp")
        return
    
    # Get list of songs (filtered for English preferably if available in metadata, otherwise just first 5)
    # DALI 1.0 has many songs. Let's try to get a few specific ones if we knew IDs, 
    # but for now just first 5 safe ones.
    songs = dali_loader.list_songs(limit=5)
    
    if not songs:
        logger.warning("No songs found in DALI metadata. Trying to update metadata...")
        dali_loader.download_dataset_metadata()
        songs = dali_loader.list_songs(limit=5)

    if not songs:
        logger.error("Still no songs found. Check internet connection or DALI repo access.")
        return

    success_count = 0
    for song in songs:
        logger.info(f"Processing {song['artist']} - {song['title']} ({song['id']})...")
        
        try:
            # Download audio
            audio_path = dali_loader.download_audio(song['id'])
            if not audio_path:
                logger.warning(f"Skipping {song['id']} due to download failure")
                continue

            # Export ground truth
            gt = dali_loader.get_ground_truth_alignment(song['id'])
            
            # Save GT json
            dali_dir = settings.DATA_DIR / "dali"
            dali_dir.mkdir(parents=True, exist_ok=True)
            output_path = dali_dir / f"{song['id']}_gt.json"
            
            import json
            with open(output_path, "w") as f:
                json.dump(gt, f, indent=2)
                
            logger.info(f"Successfully prepared {song['id']}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {song['id']}: {e}")
            
    logger.info(f"Preparation complete. Successfully prepared {success_count}/{len(songs)} songs.")

if __name__ == "__main__":
    main()
