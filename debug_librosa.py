import librosa
import torch
import sys
import os
import numpy as np

# Find an mp3 file
dali_audio_dir = "data/dali_audio"
try:
    mp3_files = [f for f in os.listdir(dali_audio_dir) if f.endswith('.mp3')]
except FileNotFoundError:
    print("Data dir not found from CWD")
    sys.exit(1)

if not mp3_files:
    print("No MP3 files found to test.")
    sys.exit(0)

test_file = os.path.join(dali_audio_dir, mp3_files[0])
print(f"Testing with: {test_file}")

try:
    # Load with librosa at 16k
    y, sr = librosa.load(test_file, sr=16000)
    print(f"Success! Shape: {y.shape}, SR: {sr}")
    
    # Convert to tensor
    tensor = torch.from_numpy(y)
    print(f"Tensor shape: {tensor.shape}, Type: {tensor.dtype}")
except Exception as e:
    print(f"Failed: {e}")
