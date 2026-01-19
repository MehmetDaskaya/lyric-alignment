"""
Deep Learning CTC-Based Lyric Alignment
Uses wav2vec2 with CTC forced alignment via TorchAudio
"""
import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
import numpy as np
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

from .preprocessing import preprocessor
from config import settings

logger = logging.getLogger(__name__)


@dataclass 
class CTCSegment:
    """Segment from CTC alignment"""
    label: str
    start: float
    end: float
    score: float


class CTCAligner:
    """
    CTC-based lyric alignment using wav2vec2
    
    Uses TorchAudio's forced alignment functionality with wav2vec2
    to align transcript to audio without frame-level labels.
    """
    
    def __init__(self):
        self._model = None
        self._labels = None
        self._device = None
    
    @property
    def device(self):
        """Get compute device"""
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    def _load_model(self):
        """Lazy load wav2vec2 model"""
        if self._model is None:
            logger.info("Loading wav2vec2 model...")
            
            bundle = WAV2VEC2_ASR_BASE_960H
            self._model = bundle.get_model().to(self.device)
            self._model.eval()
            self._labels = bundle.get_labels()
            
            logger.info(f"Model loaded on {self.device}")
    
    def _get_trellis(
        self, 
        emission: torch.Tensor, 
        tokens: List[int]
    ) -> torch.Tensor:
        """
        Build trellis matrix for forced alignment (Vectorized)
        
        Args:
            emission: Log probabilities from model (n_frames, n_classes)
            tokens: Token indices for transcript
            
        Returns:
            Trellis matrix (n_frames, n_tokens)
        """
        n_frames = emission.size(0)
        n_tokens = len(tokens)
        
        # Initialize trellis with -inf
        trellis = torch.full((n_frames, n_tokens), -float('inf'), device=emission.device)
        
        # Convert tokens to tensor for broadcasting
        token_indices = torch.tensor(tokens, device=emission.device)
        
        # First frame
        trellis[0, 0] = emission[0, tokens[0]]
        
        # Pre-compute emissions for all tokens at all frames
        # Shape: (n_frames, n_tokens)
        token_emissions = emission[:, token_indices]
        
        # Forward pass
        # We iterate over time. The state space (tokens) update is vectorized.
        for t in range(1, n_frames):
            emit_t = token_emissions[t]
            prev = trellis[t-1]
            
            # 1. Stay in current state (token j comes from token j)
            score_stay = prev + emit_t
            
            # 2. Transition from previous state (token j comes from token j-1)
            # Shift prev right by 1. 
            # Note: For j=0, it comes only from stay (or init), handled by boundary
            prev_shifted = torch.roll(prev, 1)
            prev_shifted[0] = -float('inf')
            score_trans = prev_shifted + emit_t
            
            # Take max
            trellis[t] = torch.max(score_stay, score_trans)
            
        return trellis
    
    def _backtrack(
        self, 
        trellis: torch.Tensor, 
        emission: torch.Tensor, 
        tokens: List[int]
    ) -> List[CTCSegment]:
        """
        Backtrack through trellis to find optimal path
        
        Returns:
            List of CTCSegment with token boundaries
        """
        n_frames = trellis.size(0)
        n_tokens = trellis.size(1)
        
        # Start from the end
        t = n_frames - 1
        j = n_tokens - 1
        
        path = []
        
        while t >= 0 and j >= 0:
            path.append((t, j, trellis[t, j].item()))
            
            if j == 0:
                t -= 1
            elif t == 0:
                j -= 1
            else:
                # Check if we should stay or transition
                stay = trellis[t - 1, j]
                transition = trellis[t - 1, j - 1]
                
                if stay > transition:
                    t -= 1
                else:
                    t -= 1
                    j -= 1
        
        path.reverse()
        
        # Convert path to segments
        segments = []
        prev_j = -1
        start_t = 0
        
        for t, j, score in path:
            if j != prev_j:
                if prev_j >= 0:
                    segments.append(CTCSegment(
                        label=self._labels[tokens[prev_j]],
                        start=start_t,
                        end=t,
                        score=score
                    ))
                start_t = t
                prev_j = j
        
        # Add last segment
        if prev_j >= 0:
            segments.append(CTCSegment(
                label=self._labels[tokens[prev_j]],
                start=start_t,
                end=n_frames,
                score=path[-1][2] if path else 0.0
            ))
        
        return segments
    
    def _merge_segments_to_words(
        self, 
        segments: List[CTCSegment], 
        words: List[str],
        frame_rate: float
    ) -> List[Dict]:
        """
        Merge character segments into word boundaries
        
        Args:
            segments: List of character segments
            words: Original word list
            frame_rate: Frames per second
            
        Returns:
            List of word alignment dictionaries
        """
        # Build character sequence from segments (excluding blanks and |)
        char_segments = [s for s in segments if s.label not in ('|', '-', '<s>', '</s>')]
        
        # Reconstruct text and map to words
        word_alignments = []
        char_idx = 0
        
        for word in words:
            word_upper = word.upper()
            word_chars = [c for c in word_upper if c.isalpha()]
            
            if not word_chars:
                continue
            
            # Find matching segments for this word
            word_start = None
            word_end = None
            
            matched = 0
            for i in range(char_idx, len(char_segments)):
                seg = char_segments[i]
                if matched < len(word_chars):
                    # Check if this segment matches expected character
                    expected = word_chars[matched]
                    if seg.label == expected:
                        if word_start is None:
                            word_start = seg.start
                        word_end = seg.end
                        matched += 1
                        if matched == len(word_chars):
                            char_idx = i + 1
                            break
                    else:
                        # Mismatch - try to continue
                        if word_start is not None:
                            word_end = seg.end
            
            if word_start is not None and word_end is not None:
                word_alignments.append({
                    'text': word,
                    'start': round(word_start / frame_rate, 3),
                    'end': round(word_end / frame_rate, 3)
                })
        
        return word_alignments
    
    def align(
        self, 
        audio: np.ndarray, 
        sr: int, 
        lyrics: str,
        use_source_separation: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Align lyrics to audio using CTC forced alignment
        
        Args:
            audio: Audio array
            sr: Sample rate
            lyrics: Lyrics text
            use_source_separation: Whether to use Demucs for vocal separation
            progress_callback: Optional status callback
            
        Returns:
            Alignment result dictionary
        """
        logger.info("Starting CTC alignment...")
        if progress_callback:
            progress_callback("Initializing CTC model...")
        
        # Load model if not already loaded
        self._load_model()
        
        # Optionally separate vocals
        if use_source_separation:
            logger.info("Separating vocals...")
            if progress_callback:
                progress_callback("Separating vocals (this may take a while)...")
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
        
        # Resample to 16kHz if needed (wav2vec2 requirement)
        if sr != 16000:
            if progress_callback:
                progress_callback("Resampling audio...")
            trimmed_vocals = torchaudio.functional.resample(
                torch.tensor(trimmed_vocals).unsqueeze(0), sr, 16000
            ).squeeze().numpy()
            sr = 16000
        
        # Convert to tensor
        waveform = torch.tensor(trimmed_vocals, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get emissions from model
        logger.info("Getting model emissions...")
        if progress_callback:
            progress_callback("Computing acoustic model emissions...")
        with torch.no_grad():
            emissions, _ = self._model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)
        
        emission = emissions[0].cpu()
        
        # Prepare transcript
        words = lyrics.split()
        transcript = '|'.join(words).upper()  # | as word separator
        
        # Convert transcript to tokens
        tokens = []
        for char in transcript:
            if char in self._labels:
                tokens.append(self._labels.index(char))
            elif char == ' ':
                if '|' in self._labels:
                    tokens.append(self._labels.index('|'))
        
        if not tokens:
            logger.warning("No valid tokens found in transcript")
            return {'model': 'ctc', 'words': [], 'metadata': {}}
        
        # Build trellis and align
        logger.info("Building alignment trellis...")
        if progress_callback:
            progress_callback("Aligning sequence (Viterbi path)...")
        trellis = self._get_trellis(emission, tokens)
        
        logger.info("Backtracking to find alignment...")
        segments = self._backtrack(trellis, emission, tokens)
        
        # Calculate frame rate using trimmed audio duration
        audio_duration = len(trimmed_vocals) / sr
        n_frames = emission.size(0)
        frame_rate = n_frames / audio_duration
        
        # Merge to words
        logger.info("Merging segments to words...")
        if progress_callback:
            progress_callback("Finalizing word boundaries...")
        word_alignments = self._merge_segments_to_words(segments, words, frame_rate)
        
        # Add vocal onset offset to all timestamps
        for word in word_alignments:
            word['start'] = round(word['start'] + vocal_onset_time, 3)
            word['end'] = round(word['end'] + vocal_onset_time, 3)
        
        logger.info(f"CTC alignment complete. Aligned {len(word_alignments)} words.")
        
        return {
            'model': 'ctc',
            'words': word_alignments,
            'metadata': {
                'n_frames': n_frames,
                'frame_rate': round(frame_rate, 2),
                'source_separation': use_source_separation,
                'vocal_onset_time': round(vocal_onset_time, 3)
            }
        }


# Global instance
ctc_aligner = CTCAligner()
