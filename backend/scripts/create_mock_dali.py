import sys
import shutil
import gzip
import pickle
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Add DALI/code to path
dali_code_path = backend_dir / "DALI" / "code"
sys.path.append(str(dali_code_path))

import DALI
from config import settings

def create_mock():
    # Setup directories
    dali_dir = settings.DATA_DIR / "dali"
    dali_audio = dali_dir / "audio"
    dali_dir.mkdir(parents=True, exist_ok=True)
    dali_audio.mkdir(parents=True, exist_ok=True)
    
    # Source audio
    src_audio = settings.DATA_DIR / "audio" / "a-ha - Take On Me.mp3"
    if not src_audio.exists():
        print("Source audio not found in data/audio")
        return

    # Mock Entry
    mock_id = "aha_take_on_me"
    
    # 1. Copy audio
    dst_audio = dali_audio / f"{mock_id}.mp3"
    shutil.copy(src_audio, dst_audio)
    print(f"Copied audio to {dst_audio}")
    
    # 2. Create Annotation object
    entry = DALI.Annotations()
    entry.info = {
        'id': mock_id,
        'artist': 'a-ha',
        'title': 'Take On Me',
        'dataset_version': 1.0,
        'ground-truth': True,
        'audio': {
            'url': 'http://mock.url',
            'path': str(dst_audio),
            'working': True
        }
    }
    
    # Create dummy notes
    # "Take on me"
    # Just dummy timing for testing metrics
    notes = [
        {'text': 'Take', 'time': [10.0, 10.5], 'freq': [0, 0], 'index': 0},
        {'text': 'on', 'time': [10.5, 10.7], 'freq': [0, 0], 'index': 0},
        {'text': 'me', 'time': [10.7, 11.2], 'freq': [0, 0], 'index': 0},
        {'text': 'Take', 'time': [11.5, 12.0], 'freq': [0, 0], 'index': 0},
        {'text': 'me', 'time': [12.0, 12.2], 'freq': [0, 0], 'index': 0},
        {'text': 'on', 'time': [12.2, 12.8], 'freq': [0, 0], 'index': 0},
    ]
    
    entry.annotations = {
        'type': 'horizontal',
        'annot': {
            'notes': notes,
            'lines': [], # Required structure might need this
            'words': [], 
            'paragraphs': []
        },
        'annot_param': {'fr': 0.01, 'offset': 0}
    }

    # Save as .gz
    save_path = dali_dir / f"{mock_id}.gz"
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(entry, f, protocol=2)
        
    print(f"Created mock annotation at {save_path}")

if __name__ == "__main__":
    create_mock()
