
import json
import logging
from pathlib import Path
import sys
import numpy as np

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from services.dali_service import DaliService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_segmented_manifest(
    output_path: str, 
    target_duration: float = 15.0, 
    min_duration: float = 2.0,
    silence_buffer: float = 0.2
):
    service = DaliService()
    entry_ids = service.get_all_ids()
    
    # Filter for english like before, or just process all present audio
    # Ideally reuse logic from prepare_training_data.py to filter, but let's just check valid audio
    
    valid_segments = 0
    total_songs = 0
    
    # Ensure output dir exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry_id in entry_ids:
            # Check audio existence
            audio_path = service.get_audio_path(entry_id, download_if_missing=False)
            if not audio_path:
                continue
                
            # Filter English (DALI metadata usually has language info)
            meta = service.get_metadata(entry_id)
            if meta.get('language') != 'english':
                continue
                
            # Get alignments
            alignment = service.get_ground_truth_alignment(entry_id)
            if not alignment:
                continue
                
            total_songs += 1
            
            # Segment the song
            # Algorithm: Accumulate words until duration >= target_duration
            # Then cut.
            
            current_chunk_words = []
            chunk_start_time = alignment[0]['start']
            
            for i, word in enumerate(alignment):
                current_chunk_words.append(word)
                
                current_duration = word['end'] - chunk_start_time
                
                # Check if we should split
                # Conditions: exist significant pause OR max duration reached
                # DALI words have text, start, end.
                
                is_last_word = (i == len(alignment) - 1)
                
                next_start = alignment[i+1]['start'] if not is_last_word else word['end']
                gap_to_next = next_start - word['end']
                
                should_split = False
                if current_duration >= target_duration:
                    should_split = True
                elif gap_to_next > 1.0 and current_duration >= min_duration:
                    # Natural pause break if we have at least min_duration
                    should_split = True
                elif is_last_word:
                    should_split = True
                    
                if should_split:
                    # Finalize chunk
                    chunk_text = " ".join([w['text'] for w in current_chunk_words])
                    chunk_end_time = word['end']
                    
                    # Add buffer
                    actual_start = max(0.0, chunk_start_time - silence_buffer)
                    # For end, we don't want to overlap too much into next word if gap is small
                    # but gap_to_next tracks that.
                    buffer_avail = min(silence_buffer, gap_to_next)
                    actual_end = chunk_end_time + buffer_avail
                    
                    final_duration = actual_end - actual_start
                    
                    if final_duration >= min_duration:
                        manifest_entry = {
                            "id": f"{entry_id}_{i}", # unique id for segment
                            "audio_filepath": str(audio_path),
                            "text": chunk_text,
                            "offset": actual_start,
                            "duration": final_duration,
                            "orig_entry_id": entry_id
                        }
                        f_out.write(json.dumps(manifest_entry) + '\n')
                        valid_segments += 1
                    
                    # Reset for next chunk
                    if not is_last_word:
                        chunk_start_time = next_start
                        current_chunk_words = []

    logger.info(f"Processed {total_songs} songs.")
    logger.info(f"Generated {valid_segments} training segments.")
    logger.info(f"Saved to {output_path}")

if __name__ == "__main__":
    output_file = Path("backend/data/training_sets/dali_segmented_train.jsonl")
    create_segmented_manifest(output_file)
