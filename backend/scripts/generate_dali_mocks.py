import sys
import shutil
import gzip
import pickle
import librosa
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Add DALI/code to path
dali_code_path = backend_dir / "DALI" / "code"
sys.path.append(str(dali_code_path))

import DALI
from config import settings

def generate_mocks():
    print("Generating 5 Mock DALI Songs...")
    
    # Setup directories (using project root data/audio as source, backend/data/dali as dest?)
    # Wait, in previous step I set dali_loader to use settings.DATA_DIR / "dali"
    # settings.DATA_DIR usually points to backend/../data = root/data.
    # So dest is root/data/dali.
    
    source_audio_dir = settings.DATA_DIR / "audio"
    source_lyrics_dir = settings.DATA_DIR / "lyrics"
    
    dali_dir = settings.DATA_DIR / "dali"
    dali_audio_out = dali_dir / "audio"
    dali_dir.mkdir(parents=True, exist_ok=True)
    dali_audio_out.mkdir(parents=True, exist_ok=True)
    
    songs_to_mock = [
        ("aha_take_on_me", "a-ha - Take On Me"),
        ("paramore_all_i_wanted", "Paramore - All I Wanted"),
        ("fleetwood_mac_the_chain", "Fleetwood Mac - The Chain"),
        ("scorpions_still_loving_you", "Scorpions - Still Loving You"),
        ("teoman_istanbulda_sonbahar", "Teoman - İstanbul'da Sonbahar") # Check special char
    ] # Map ID -> Filename Stem

    for dali_id, filename_stem in songs_to_mock:
        print(f"Processing {dali_id}...")
        
        # 1. Locate Source Files
        # Try to find file with exact name, handling potential unicode issues
        try:
            mp3_path = next(source_audio_dir.glob(f"{filename_stem}.mp3"))
            # For lyrics, might differ slightly on unicode?
            # Try approximate match if exact fails?
            try:
                txt_path = next(source_lyrics_dir.glob(f"{filename_stem}.txt"))
            except StopIteration:
                 # Fallback for Teoman normalization potentially
                 txt_path = list(source_lyrics_dir.glob("Teoman*.txt"))[0]

        except StopIteration:
            print(f"Skipping {dali_id}: Source files not found for '{filename_stem}'")
            continue
            
        # 2. Get Duration
        y, sr = librosa.load(str(mp3_path), sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 3. Read Lyrics
        with open(txt_path, 'r', encoding='utf-8') as f:
            lyrics_text = f.read()
            
        # 4. Create Mock Alignment
        # Split into lines first to preserve structure
        lines_text = [l.strip() for l in lyrics_text.split('\n') if l.strip()]
        
        all_words = []
        lines_data = []
        
        # Calculate total words to enable time distribution
        total_words = sum(len(l.split()) for l in lines_text)
        if total_words == 0:
            total_words = 1
            lines_text = ["Instrumental"]

        start_buffer = 5.0
        end_buffer = 5.0
        available_time = max(1.0, duration - start_buffer - end_buffer)
        word_duration = available_time / total_words
        
        current_time = start_buffer
        
        for line_idx, line_str in enumerate(lines_text):
            line_words_text = line_str.split()
            if not line_words_text:
                continue
                
            line_start_time = current_time
            line_notes = []
            
            for word in line_words_text:
                start = current_time
                end = current_time + (word_duration * 0.9)
                
                note_dict = {
                    'text': word,
                    'time': [start, end],
                    'freq': [0, 0],
                    'index': 0 # In real DALI this maps to line index usually
                }
                
                all_words.append(note_dict)
                line_notes.append(note_dict)
                current_time += word_duration
            
            line_end_time = line_notes[-1]['time'][1]
            
            lines_data.append({
                'text': line_str,
                'time': [line_start_time, line_end_time],
                'freq': [0, 0],
                'index': line_idx
            })

        # 5. Create DALI objects
        destination_mp3 = dali_audio_out / f"{dali_id}.mp3"
        shutil.copy2(mp3_path, destination_mp3)
        
        entry = DALI.Annotations()
        entry.info = {
            'id': dali_id,
            'artist': filename_stem.split(' - ')[0],
            'title': filename_stem.split(' - ')[1],
            'dataset_version': 1.0,
            'ground-truth': True,
            'audio': {
                'url': 'http://mock.local',
                'path': str(destination_mp3),
                'working': True
            }
        }
        
        entry.annotations = {
            'type': 'horizontal',
            'annot': {
                'notes': all_words,
                'words': [], 
                'lines': lines_data,
                'paragraphs': []
            },
            'annot_param': {'fr': 0.01, 'offset': 0}
        }
        
        # 6. Save .gz
        gz_path = dali_dir / f"{dali_id}.gz"
        with gzip.open(gz_path, 'wb') as f:
            pickle.dump(entry, f, protocol=2)
            
        print(f"Generated {dali_id} -> {gz_path}")

if __name__ == "__main__":
    generate_mocks()
