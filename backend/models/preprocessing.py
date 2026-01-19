"""
Audio Preprocessing Module
Handles audio loading, resampling, and source separation
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import librosa
import logging

from config import settings

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing with Demucs source separation"""
    
    def __init__(self):
        self._demucs_model = None
        
    @property
    def demucs_model(self):
        """Lazy load Demucs model"""
        if self._demucs_model is None:
            logger.info("Loading Demucs model...")
            try:
                from demucs.pretrained import get_model
                from demucs.apply import apply_model
                self._demucs_model = get_model('htdemucs')
                self._demucs_model.eval()
                if torch.cuda.is_available():
                    self._demucs_model.cuda()
            except Exception as e:
                logger.error(f"Failed to load Demucs: {e}")
                raise
        return self._demucs_model
    
    def load_audio(
        self, 
        audio_path: str, 
        target_sr: int = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load and resample audio to target sample rate (mono)
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (default: 16kHz)
            
        Returns:
            Tuple of (audio array, sample rate)
        """
        target_sr = target_sr or settings.SAMPLE_RATE
        
        # Load with librosa (handles various formats)
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        logger.info(f"Loaded audio: {Path(audio_path).name}, "
                   f"duration: {len(audio)/sr:.2f}s, sr: {sr}")
        
        return audio, sr
    
    def separate_vocals(
        self, 
        audio: np.ndarray, 
        sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate vocals from instrumental using Hybrid Demucs
        
        Args:
            audio: Audio array (mono)
            sr: Sample rate
            
        Returns:
            Tuple of (vocals, instrumental)
        """
        from demucs.apply import apply_model
        
        # Demucs expects stereo at 44.1kHz
        # Convert mono to stereo
        audio_stereo = np.stack([audio, audio])
        
        # Resample to 44.1kHz if needed
        if sr != 44100:
            audio_stereo = librosa.resample(
                audio_stereo, 
                orig_sr=sr, 
                target_sr=44100
            )
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio_stereo, dtype=torch.float32).unsqueeze(0)
        
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()
        
        # Apply model
        logger.info("Applying Demucs source separation...")
        with torch.no_grad():
            sources = apply_model(
                self.demucs_model, 
                audio_tensor,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        # sources shape: (batch, sources, channels, samples)
        # Demucs sources: drums, bass, other, vocals
        sources = sources.cpu().numpy()[0]
        
        vocals = sources[3].mean(axis=0)  # Average stereo to mono
        instrumental = sources[:3].sum(axis=0).mean(axis=0)
        
        # Resample back to target sr
        if sr != 44100:
            vocals = librosa.resample(vocals, orig_sr=44100, target_sr=sr)
            instrumental = librosa.resample(instrumental, orig_sr=44100, target_sr=sr)
        
        logger.info("Source separation complete")
        return vocals, instrumental
    
    def extract_mfcc(
        self, 
        audio: np.ndarray, 
        sr: int,
        n_mfcc: int = None,
        hop_length: int = None,
        win_length: int = None
    ) -> np.ndarray:
        """
        Extract MFCC features (39-dim with deltas)
        
        Args:
            audio: Audio array
            sr: Sample rate
            n_mfcc: Number of MFCCs (default: 13, expanded to 39 with deltas)
            hop_length: Hop length in samples
            win_length: Window length in samples
            
        Returns:
            MFCC features array (n_frames, 39)
        """
        n_mfcc = n_mfcc or 13
        hop_length = hop_length or settings.HOP_LENGTH
        win_length = win_length or settings.WIN_LENGTH
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            hop_length=hop_length,
            win_length=win_length
        )
        
        # Add delta and delta-delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate: (39, n_frames)
        mfcc_full = np.vstack([mfcc, delta, delta2])
        
        # Transpose to (n_frames, 39)
        return mfcc_full.T
    
    def get_frame_times(
        self, 
        n_frames: int, 
        sr: int, 
        hop_length: int = None
    ) -> np.ndarray:
        """Get time stamps for each frame"""
        hop_length = hop_length or settings.HOP_LENGTH
        return librosa.frames_to_time(
            np.arange(n_frames), 
            sr=sr, 
            hop_length=hop_length
        )
    def detect_vocal_onset(
        self, 
        audio: np.ndarray, 
        sr: int,
        threshold_db: float = -40,
        min_duration: float = 0.5
    ) -> float:
        """
        Detect when vocals/speech actually starts in the audio.
        Uses RMS energy and onset detection to find where singing begins.
        
        Args:
            audio: Audio array (ideally vocals-separated)
            sr: Sample rate
            threshold_db: RMS threshold in dB for considering as "active"
            min_duration: Minimum duration (seconds) of activity to confirm onset
            
        Returns:
            Onset time in seconds (0.0 if vocals start immediately)
        """
        hop_length = settings.HOP_LENGTH
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find frames above threshold
        active_frames = rms_db > threshold_db
        
        # Find first sustained active region
        min_frames = int(min_duration * sr / hop_length)
        onset_frame = 0
        
        for i in range(len(active_frames)):
            if active_frames[i]:
                # Check if this is sustained
                end_check = min(i + min_frames, len(active_frames))
                sustained = np.sum(active_frames[i:end_check]) >= (min_frames * 0.7)
                if sustained:
                    onset_frame = i
                    break
        
        onset_time = librosa.frames_to_time(onset_frame, sr=sr, hop_length=hop_length)
        logger.info(f"Detected vocal onset at {onset_time:.2f}s")
        
        return onset_time
    
    def trim_to_vocal_onset(
        self,
        audio: np.ndarray,
        sr: int,
        onset_time: float
    ) -> np.ndarray:
        """
        Trim audio to start from vocal onset.
        
        Args:
            audio: Full audio array
            sr: Sample rate
            onset_time: Onset time in seconds
            
        Returns:
            Trimmed audio starting from onset
        """
        onset_sample = int(onset_time * sr)
        return audio[onset_sample:]


# Global instance
preprocessor = AudioPreprocessor()
