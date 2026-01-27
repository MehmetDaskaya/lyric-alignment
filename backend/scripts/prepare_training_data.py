"""
Prepare Training Data from DALI v1.0
Filters dataset for English songs, checks audio availability,
and generates a training manifest.
"""
import logging
import sys
import json
import csv
from pathlib import Path
from tqdm import tqdm

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from services.dali_service import dali_service
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Training Data Preparation...")
    
    output_dir = settings.DATA_DIR / "training_sets"
    output_dir.mkdir(exist_ok=True)
    
    manifest_path = output_dir / "dali_train_manifest.jsonl"
    
    # 1. Get all IDs
    all_ids = dali_service.get_all_ids()
    logger.info(f"Total DALI entries: {len(all_ids)}")
    
    valid_entries = 0
    skipped_no_audio = 0
    skipped_non_english = 0
    
    with open(manifest_path, 'w', encoding='utf-8') as f_out:
        for entry_id in tqdm(all_ids):
            # 2. Filter by Language (English only)
            meta = dali_service.get_metadata(entry_id)
            if meta.get('language') != 'english' and meta.get('language') != 'en':
                # DALI metadata language format might vary, checking both
                # Actually, DALI 1.0 often uses ISO codes or full names.
                # Let's inspect one if possible, but safe to skip non-explicit English for now.
                # Or if language metadata is missing, maybe keep it?
                # Let's be strict for high quality.
                lang = meta.get('language', '').lower()
                if 'en' not in lang:
                    skipped_non_english += 1
                    continue

            # 3. Check Audio
            # For training prep, we don't necessarily download EVERYTHING right now 
            # if it takes too long, but we need to know where it WOULD be.
            # However, to be useful, we usually need the audio. 
            # Let's check if it exists locally.
            audio_path = dali_service.get_audio_path(entry_id, download_if_missing=False)
            if not audio_path:
                skipped_no_audio += 1
                continue
                
            # 4. Get Data
            lyrics = dali_service.get_lyrics(entry_id)
            alignment = dali_service.get_ground_truth_alignment(entry_id)
            
            if not lyrics or not alignment:
                continue
                
            # 5. Write to Manifest
            entry = {
                'id': entry_id,
                'audio_filepath': str(audio_path),
                'text': lyrics,
                'duration': alignment[-1]['end'] if alignment else 0,
                'alignment': alignment
            }
            
            f_out.write(json.dumps(entry) + '\n')
            valid_entries += 1
            
    logger.info(f"Manifest generation complete.")
    logger.info(f"Valid English Songs with Audio: {valid_entries}")
    logger.info(f"Skipped (Non-English): {skipped_non_english}")
    logger.info(f"Skipped (No Audio Locally): {skipped_no_audio}")
    logger.info(f"Manifest saved to: {manifest_path}")
    
    if valid_entries < 10:
        logger.warning("Very few audio files found. You may need to run 'prepare_dali.py' or a bulk downloader to fetch more audio.")

if __name__ == "__main__":
    main()
