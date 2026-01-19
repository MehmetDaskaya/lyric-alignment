"""
DTW-Based Lyric Alignment (Baseline Model)
Uses Text-to-Speech synthesis and Dynamic Time Warping
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import tempfile
import logging
from scipy.spatial.distance import cdist
from dataclasses import dataclass

from gtts import gTTS
import librosa

from .preprocessing import preprocessor
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class WordAlignment:
    """Alignment result for a single word"""
    text: str
    start: float
    end: float


class DTWAligner:
    """
    DTW-based lyric alignment using TTS synthesis
    
    Pipeline:
    1. Synthesize lyrics using TTS
    2. Extract MFCCs from both real vocals and synthetic audio
    3. Apply DTW with Sakoe-Chiba band constraint
    4. Map alignment path to word boundaries
    """
    
    def __init__(self, band_radius: float = 0.1):
        """
        Args:
            band_radius: Sakoe-Chiba band radius as fraction of sequence length
        """
        self.band_radius = band_radius
    
    def synthesize_lyrics(
        self, 
        lyrics: str, 
        output_path: str = None
    ) -> Tuple[np.ndarray, int, List[Dict]]:
        """
        Synthesize lyrics using gTTS
        
        Args:
            lyrics: Lyrics text
            output_path: Optional output path for synthesized audio
            
        Returns:
            Tuple of (audio array, sample rate, word info list)
        """
        # Use temp file if no output path
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            output_path = temp_file.name
        
        # Synthesize with gTTS
        tts = gTTS(text=lyrics, lang='en', slow=False)
        tts.save(output_path)
        
        # Load the synthesized audio
        audio, sr = librosa.load(output_path, sr=settings.SAMPLE_RATE, mono=True)
        
        # Parse words (simple tokenization)
        words = lyrics.split()
        
        # Estimate word boundaries (uniform distribution for TTS)
        duration = len(audio) / sr
        word_duration = duration / len(words)
        
        word_info = []
        for i, word in enumerate(words):
            word_info.append({
                'text': word,
                'start': i * word_duration,
                'end': (i + 1) * word_duration,
                'index': i
            })
        
        return audio, sr, word_info
    
    def compute_dtw(
        self, 
        mfcc_real: np.ndarray, 
        mfcc_synth: np.ndarray,
        band_radius: float = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute DTW alignment with Sakoe-Chiba band constraint
        
        Args:
            mfcc_real: MFCC features from real audio (n_frames_real, n_features)
            mfcc_synth: MFCC features from synthetic audio (n_frames_synth, n_features)
            band_radius: Band constraint radius
            
        Returns:
            Tuple of (alignment path, total cost)
        """
        band_radius = band_radius or self.band_radius
        
        n, m = len(mfcc_real), len(mfcc_synth)
        
        # Compute pairwise distances
        cost_matrix = cdist(mfcc_real, mfcc_synth, metric='euclidean')
        
        # Initialize DTW matrix with infinity
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0
        
        # Compute band size
        band = int(max(n, m) * band_radius)
        
        # Fill DTW matrix with Sakoe-Chiba constraint
        for i in range(1, n + 1):
            # Band constraint
            j_min = max(1, int((i * m) / n) - band)
            j_max = min(m, int((i * m) / n) + band)
            
            for j in range(j_min, j_max + 1):
                cost = cost_matrix[i - 1, j - 1]
                dtw[i, j] = cost + min(
                    dtw[i - 1, j],      # insertion
                    dtw[i, j - 1],      # deletion
                    dtw[i - 1, j - 1]   # match
                )
        
        # Backtrack to find optimal path
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                candidates = [
                    (dtw[i - 1, j - 1], i - 1, j - 1),
                    (dtw[i - 1, j], i - 1, j),
                    (dtw[i, j - 1], i, j - 1)
                ]
                _, i, j = min(candidates, key=lambda x: x[0])
        
        path.reverse()
        
        return np.array(path), dtw[n, m]
    
    def _map_path_to_words(
        self, 
        path: np.ndarray, 
        words: List[str], 
        n_frames_audio: int, 
        n_frames_lyrics: int,
        sr_audio: int,
        sr_lyrics: int
    ) -> List[WordAlignment]:
        """
        Maps the DTW path to word boundaries.
        This is a simplified approach for the baseline, assuming uniform distribution
        of words in the synthetic audio.
        """
        mfcc_hop_length = settings.HOP_LENGTH
        
        # Calculate frame times for both audio and lyrics MFCCs
        frame_times_audio = np.arange(n_frames_audio) * mfcc_hop_length / sr_audio
        frame_times_lyrics = np.arange(n_frames_lyrics) * mfcc_hop_length / sr_lyrics
        
        # Estimate word boundaries in synthetic audio (uniform distribution)
        # This is a simplification; a real TTS engine might provide these.
        synth_duration = frame_times_lyrics[-1] if len(frame_times_lyrics) > 0 else 0
        word_duration_synth = synth_duration / len(words) if len(words) > 0 else 0
        
        word_synth_boundaries = []
        for i, word in enumerate(words):
            word_synth_boundaries.append({
                'text': word,
                'start_time': i * word_duration_synth,
                'end_time': (i + 1) * word_duration_synth
            })
            
        aligned_words = []
        
        for word_info in word_synth_boundaries:
            synth_start_frame = np.searchsorted(frame_times_lyrics, word_info['start_time'])
            synth_end_frame = np.searchsorted(frame_times_lyrics, word_info['end_time'])
            
            real_frames_for_word = []
            for real_idx, synth_idx in path:
                if synth_start_frame <= synth_idx < synth_end_frame:
                    real_frames_for_word.append(real_idx)
            
            if real_frames_for_word:
                real_start_frame = min(real_frames_for_word)
                real_end_frame = max(real_frames_for_word)
                
                start_time = frame_times_audio[real_start_frame]
                end_time = frame_times_audio[min(real_end_frame, len(frame_times_audio) - 1)]
                
                aligned_words.append(WordAlignment(
                    text=word_info['text'],
                    start=round(start_time, 3),
                    end=round(end_time, 3)
                ))
        return aligned_words

    def align(
        self, 
        audio: np.ndarray, 
        sr: int, 
        lyrics: str,
        use_source_separation: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Align lyrics to audio using DTW
        
        Args:
            audio: Audio array
            sr: Sample rate
            lyrics: Lyrics text
            use_source_separation: Whether to use Demucs for vocal separation
            progress_callback: Optional status callback
            
        Returns:
            Alignment result dictionary
        """
        logger.info("Starting DTW alignment...")
        if progress_callback:
            progress_callback("Initializing DTW...")
        
        # Optionally separate vocals
        if use_source_separation:
            logger.info("Extracting MFCCs from vocals...")
            if progress_callback:
                progress_callback("Separating vocals (Demucs)...")
            vocals, _ = preprocessor.separate_vocals(audio, sr)
        else:
            vocals = audio
        
        # Detect vocal onset (when singing actually starts)
        if progress_callback:
            progress_callback("Detecting vocal onset...")
        vocal_onset_time = preprocessor.detect_vocal_onset(vocals, sr)
        
        # Trim audio to vocal onset for alignment
        trimmed_vocals = preprocessor.trim_to_vocal_onset(vocals, sr, vocal_onset_time)
        logger.info(f"Trimmed {vocal_onset_time:.2f}s of intro silence/instrumental")
            
        # 1. Audio Features from trimmed vocals
        if progress_callback:
            progress_callback("Extracting audio features...")
        mfcc_audio = preprocessor.extract_mfcc(trimmed_vocals, sr)
        
        # 2. Text Features (TTS)
        logger.info("Synthesizing lyrics with TTS...")
        if progress_callback:
            progress_callback("Synthesizing reference audio (TTS)...")
        synth_audio, synth_sr, synth_word_info = self.synthesize_lyrics(lyrics)
        mfcc_lyrics = preprocessor.extract_mfcc(synth_audio, synth_sr)
        
        # 3. DTW Alignment
        logger.info("Computing DTW alignment...")
        if progress_callback:
            progress_callback("Computing Dynamic Time Warping...")
        path, cost = self.compute_dtw(mfcc_audio, mfcc_lyrics)
        
        # 4. Map back to words
        logger.info("Mapping word boundaries...")
        if progress_callback:
            progress_callback("Mapping boundaries to words...")
        
        words = lyrics.split()
        word_alignments = self._map_path_to_words(
            path, 
            words, 
            len(mfcc_audio), 
            len(mfcc_lyrics),
            sr,
            synth_sr
        )
        
        # Add vocal onset offset to all timestamps
        aligned_words_with_offset = []
        for wa in word_alignments:
            aligned_words_with_offset.append({
                'text': wa.text,
                'start': round(wa.start + vocal_onset_time, 3),
                'end': round(wa.end + vocal_onset_time, 3)
            })
        
        logger.info(f"DTW alignment complete. Aligned {len(aligned_words_with_offset)} words.")
        
        return {
            'model': 'dtw',
            'words': aligned_words_with_offset,
            'metadata': {
                'dtw_cost': float(cost),
                'source_separation': use_source_separation,
                'vocal_onset_time': round(vocal_onset_time, 3)
            }
        }


# Global instance
dtw_aligner = DTWAligner()
