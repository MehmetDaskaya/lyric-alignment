import torchaudio
import torch
import sys
import os

print(f"TorchAudio Version: {torchaudio.__version__}")
try:
    print(f"List Backends: {torchaudio.list_audio_backends()}")
except:
    pass

try:
    print(f"Backend info: {torchaudio.info}")
except:
    pass

# Find an mp3 file
dali_audio_dir = "data/dali_audio"
mp3_files = [f for f in os.listdir(dali_audio_dir) if f.endswith('.mp3')]

if not mp3_files:
    print("No MP3 files found to test.")
    sys.exit(0)

test_file = os.path.join(dali_audio_dir, mp3_files[0])
print(f"Testing with: {test_file}")

# Test 1: Default load
print("\n--- Test 1: Default Load ---")
try:
    waveform, sample_rate = torchaudio.load(test_file)
    print(f"Success! Shape: {waveform.shape}, SR: {sample_rate}")
except Exception as e:
    print(f"Failed: {e}")

# Test 2: Soundfile backend
print("\n--- Test 2: Soundfile Backend ---")
try:
    waveform, sample_rate = torchaudio.load(test_file, backend="soundfile")
    print(f"Success! Shape: {waveform.shape}, SR: {sample_rate}")
except Exception as e:
    print(f"Failed: {e}")

# Test 3: FFmpeg backend (deprecated but might work via string)
print("\n--- Test 3: FFmpeg Backend ---")
try:
    # Set backend if possible
    # torchaudio.set_audio_backend("ffmpeg") 
    waveform, sample_rate = torchaudio.load(test_file, backend="ffmpeg")
    print(f"Success! Shape: {waveform.shape}, SR: {sample_rate}")
except Exception as e:
    print(f"Failed: {e}")
