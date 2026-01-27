
import sys
import torch
import torchaudio
from transformers import Wav2Vec2Processor
from pathlib import Path

# Add backend to path to import model
sys.path.append(str(Path(__file__).parent.parent))

from training.model import LyricAlignmentModel

def test_checkpoint(checkpoint_path, audio_path):
    # Imports for safe globals
    from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor as W2V2Proc
    from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
    from transformers.models.wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
    from transformers.tokenization_utils import Trie
    from tokenizers import AddedToken
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    try:
        # Try-except block for safe_globals
        try:
            with torch.serialization.safe_globals([W2V2Proc, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Trie, AddedToken]):
               model = LyricAlignmentModel.load_from_checkpoint(checkpoint_path, processor=processor, map_location="cpu")
        except AttributeError:
             model = LyricAlignmentModel.load_from_checkpoint(checkpoint_path, processor=processor, map_location="cpu")
             
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Load audio
    print(f"Loading audio: {audio_path}")
    try:
        waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile") # Try soundfile backend explicitly
    except Exception as e:
        print(f"Torchaudio failed ({e}), trying librosa...")
        import librosa
        import numpy as np
        # librosa loads as mono (T,) by default
        y, sr = librosa.load(audio_path, sr=None) # Load native SR first
        waveform = torch.from_numpy(y)
        sample_rate = sr
        # Ensure float32
        if waveform.dtype != torch.float32:
            waveform = waveform.float()
        # Add channel dim if missing for compatibility below
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
    
    # Resample to 16k
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
        
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    print(f"\n--- Result ---")
    print(f"Logits shape: {logits.shape}")
    print(f"Predicted IDs max: {predicted_ids.max()}, min: {predicted_ids.min()}")
    print(f"Transcription: '{transcription}'")
    
    if not transcription.strip():
        print("WARNING: Transcription is empty!")
    else:
        print("Success: Generated text.")

if __name__ == "__main__":
    # Use a specific checkpoint
    ckpt_path = "/Users/mehmetdaskaya/Documents/projects/ai-lyric-alignment/backend/outputs/checkpoints/lyric-align-epoch=01-val_loss=0.00.ckpt"
    
    # Use a sample audio we know exists (from verify_batch or similar)
    # We verify_batch used something... let's check what audio is available
    # We can use the first file in dali_audio
    import os
    audio_dir = Path("/Users/mehmetdaskaya/Documents/projects/ai-lyric-alignment/data/dali_audio")
    try:
        audio_files = list(audio_dir.glob("*.mp3"))
        if not audio_files:
            print("No audio files found in data/dali_audio")
            sys.exit(1)
        test_audio = audio_files[0]
    except Exception as e:
        print(f"Error finding audio: {e}")
        sys.exit(1)

    test_checkpoint(ckpt_path, test_audio)
