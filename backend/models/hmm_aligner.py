"""
HMM-Based Lyric Alignment
Uses phoneme-level acoustic models with Viterbi forced alignment
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import re

import pronouncing
from g2p_en import G2p

from .preprocessing import preprocessor
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class PhonemeHMM:
    """Simple 3-state HMM for a phoneme"""
    phoneme: str
    n_states: int = 3
    
    def __post_init__(self):
        # HMM parameters (simplified)
        # Transition probabilities: mostly self-loop, some probability to next state
        self.trans_prob = np.array([
            [0.6, 0.4, 0.0],  # State 0: 60% stay, 40% -> state 1
            [0.0, 0.6, 0.4],  # State 1: 60% stay, 40% -> state 2
            [0.0, 0.0, 1.0],  # State 2: absorbing exit state
        ])


class CMUDictionary:
    """CMU Pronunciation Dictionary wrapper"""
    
    def __init__(self):
        self.g2p = G2p()
        self._cache = {}
    
    def get_phonemes(self, word: str) -> List[str]:
        """
        Get phonemes for a word using CMU dict or G2P fallback
        
        Args:
            word: Input word
            
        Returns:
            List of phoneme strings
        """
        word_lower = word.lower()
        
        # Check cache
        if word_lower in self._cache:
            return self._cache[word_lower]
        
        # Try CMU dictionary first
        phonemes_list = pronouncing.phones_for_word(word_lower)
        
        if phonemes_list:
            # Use first pronunciation, remove stress markers
            phonemes = [re.sub(r'\d', '', p) for p in phonemes_list[0].split()]
        else:
            # Fallback to G2P
            logger.debug(f"Using G2P for OOV word: {word}")
            phonemes = self.g2p(word_lower)
            # Filter out non-phoneme tokens
            phonemes = [p for p in phonemes if p.isalpha()]
        
        self._cache[word_lower] = phonemes
        return phonemes


class HMMAligner:
    """
    HMM-based lyric alignment using Viterbi forced alignment
    
    Pipeline:
    1. Convert lyrics to phoneme sequence using CMU dict / G2P
    2. Build composite HMM (concatenated word HMMs)
    3. Extract acoustic features (MFCCs)
    4. Run Viterbi algorithm for forced alignment
    5. Convert phoneme boundaries back to word boundaries
    """
    
    def __init__(self):
        self.cmu_dict = CMUDictionary()
        self.phoneme_hmms = {}
    
    def _get_phoneme_hmm(self, phoneme: str) -> PhonemeHMM:
        """Get or create HMM for a phoneme"""
        if phoneme not in self.phoneme_hmms:
            self.phoneme_hmms[phoneme] = PhonemeHMM(phoneme)
        return self.phoneme_hmms[phoneme]
    
    def _words_to_phonemes(
        self, 
        words: List[str]
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Convert word list to phoneme sequence with word boundaries
        
        Returns:
            Tuple of (phoneme list, word boundary indices)
        """
        phonemes = []
        word_boundaries = []  # (start_idx, end_idx) for each word
        
        for word in words:
            # Clean word
            word_clean = re.sub(r'[^\w\s]', '', word)
            if not word_clean:
                continue
            
            start_idx = len(phonemes)
            word_phonemes = self.cmu_dict.get_phonemes(word_clean)
            phonemes.extend(word_phonemes)
            end_idx = len(phonemes)
            
            word_boundaries.append((start_idx, end_idx, word))
        
        return phonemes, word_boundaries
    
    def _compute_emission_prob(
        self, 
        mfcc: np.ndarray, 
        phoneme: str
    ) -> np.ndarray:
        """
        Compute emission probabilities for frames given phoneme
        
        This is a simplified version - in production, you'd use
        trained Gaussian Mixture Models for each phoneme state
        
        Args:
            mfcc: MFCC features (n_frames, n_features)
            phoneme: Phoneme string
            
        Returns:
            Emission log-probabilities (n_frames, n_states)
        """
        n_frames = len(mfcc)
        n_states = 3
        
        # Simplified emission model: use MFCC energy as proxy
        # In production: train GMMs on labeled data
        energy = np.sum(mfcc ** 2, axis=1)
        energy_norm = (energy - energy.mean()) / (energy.std() + 1e-8)
        
        # Simple state-dependent emission
        emissions = np.zeros((n_frames, n_states))
        emissions[:, 0] = -0.5 * energy_norm  # Beginning of phoneme
        emissions[:, 1] = 0.0  # Middle (steady state)
        emissions[:, 2] = 0.5 * energy_norm  # End of phoneme
        
        return emissions
    
    def _viterbi_alignment(
        self, 
        mfcc: np.ndarray, 
        phonemes: List[str]
    ) -> List[Tuple[int, int, str]]:
        """
        Run Viterbi forced alignment (Vectorized)
        
        Args:
            mfcc: MFCC features (n_frames, n_features)
            phonemes: List of phonemes
            
        Returns:
            List of (start_frame, end_frame, phoneme) tuples
        """
        n_frames = len(mfcc)
        n_phonemes = len(phonemes)
        
        if n_phonemes == 0 or n_frames == 0:
            return []
        
        # Build state sequence: 3 states per phoneme
        n_states_per_phoneme = 3
        total_states = n_phonemes * n_states_per_phoneme
        
        # Pre-compute emission probabilities for all states
        # Shape: (n_frames, total_states)
        emissions = np.zeros((n_frames, total_states))
        
        # We can optimize this loop too by computing unique phonemes emissions once
        unique_phonemes = list(set(phonemes))
        phoneme_emissions = {}
        for p in unique_phonemes:
            phoneme_emissions[p] = self._compute_emission_prob(mfcc, p)
            
        for i, p in enumerate(phonemes):
            emissions[:, i*3:(i+1)*3] = phoneme_emissions[p]
            
        # Initialize Viterbi matrix in log domain
        # viterbi[t, s] = max log prob of path ending at state s at time t
        viterbi = np.full((n_frames, total_states), -1e9) # -inf equivalent
        # Backpointer to store the previous state that maximized the prob
        # backpointer[t, s] = prev_state index
        backpointer = np.zeros((n_frames, total_states), dtype=np.int32)
        
        # Transition probabilities (in log domain)
        # We assume simplified HMM:
        # State 0: self(0.6), next(0.4)
        # State 1: self(0.6), next(0.4)
        # State 2 (exit): self(ignored/1.0), next_phoneme(0.4)
        
        log_self = np.log(0.6 + 1e-10)
        log_next = np.log(0.4 + 1e-10)
        
        # Initial state probabilities (force start at state 0)
        viterbi[0, 0] = emissions[0, 0]
        
        # Forward pass (Vectorized)
        for t in range(1, n_frames):
            emit_t = emissions[t]
            prev_runs = viterbi[t-1]
            
            # 1. Self transitions (s -> s)
            # score = prev[s] + log(self) + emit[s]
            score_self = prev_runs + log_self + emit_t
            
            # 2. Next state transitions (s-1 -> s)
            # Shift prev array right by 1
            # score = prev[s-1] + log(next) + emit[s]
            prev_shifted = np.roll(prev_runs, 1)
            prev_shifted[0] = -1e9 # No transition to 0 from -1
            
            # Handle phoneme boundaries (state 2 of p -> state 0 of p+1)
            # The regular shift handles state 0->1 and 1->2 correctly with log_next
            # But 2->0 (next phoneme) also uses 0.4 prob in our simple model
            score_trans = prev_shifted + log_next + emit_t
            
            # We need to construct the winner for each state
            # Compare self vs incoming transition
            
            # Create a mask where transform is valid
            # Transitions are:
            # 0->0, 0->1
            # 1->1, 1->2
            # 2->2 (self), 2->0_next (cross boundary)
            
            # Currently score_trans covers (i-1 -> i) for all i. 
            # This is correct for 0->1 and 1->2.
            # For 2->0 (boundary), i is usually 0 mod 3.
            # index i is start of new phoneme. i-1 is end of prev phoneme.
            # So simple shift works for boundaries too!
            
            # Compare and pick max
            # Note: We prioritize self loop if equal? Doesn't matter much.
            
            # Use np.where to pick simpler
            # But we need indices for backpointer.
            
            # Let's stack them: shape (2, total_states)
            # 0: from self (s)
            # 1: from prev (s-1)
            candidates = np.stack([score_self, score_trans])
            
            # Max over axis 0
            best_idx = np.argmax(candidates, axis=0) # 0 or 1
            viterbi[t] = np.choose(best_idx, candidates)
            
            # Calculate backpointer
            # if best is 0 (self), prev is s
            # if best is 1 (trans), prev is s-1
            # so prev = s - best_idx
            state_indices = np.arange(total_states)
            backpointer[t] = state_indices - best_idx
            
        # Backtrace
        # Start from the last state of the last phoneme
        final_state = total_states - 1
        
        # If last state not reached/probable, try searching for best end state? 
        # Usually forced alignment enforces end state.
        curr_state = final_state
        
        path = [curr_state]
        for t in range(n_frames - 1, 0, -1):
            curr_state = backpointer[t, curr_state]
            path.append(curr_state)
            
        path.reverse()
        
        # Convert state sequence to phoneme boundaries
        phoneme_boundaries = []
        current_phoneme = -1
        start_frame = 0
        
        for t, state in enumerate(path):
            phoneme_idx = state // 3
            
            if phoneme_idx != current_phoneme:
                if current_phoneme >= 0 and current_phoneme < len(phonemes):
                    phoneme_boundaries.append(
                        (start_frame, t, phonemes[current_phoneme])
                    )
                current_phoneme = phoneme_idx
                start_frame = t
        
        # Add last phoneme
        if current_phoneme >= 0 and current_phoneme < len(phonemes):
            phoneme_boundaries.append(
                (start_frame, n_frames, phonemes[current_phoneme])
            )
        
        return phoneme_boundaries
    
    def align(
        self, 
        audio: np.ndarray, 
        sr: int, 
        lyrics: str,
        use_source_separation: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Align lyrics to audio using HMM forced alignment
        
        Args:
            audio: Audio array
            sr: Sample rate
            lyrics: Lyrics text
            use_source_separation: Whether to use Demucs for vocal separation
            progress_callback: Optional status callback
            
        Returns:
            Alignment result dictionary
        """
        logger.info("Starting HMM alignment...")
        if progress_callback:
            progress_callback("Initializing HMM...")
        
        # Optionally separate vocals
        if use_source_separation:
            logger.info("Separating vocals...")
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
        # We'll add this offset back to the final timestamps
        trimmed_vocals = preprocessor.trim_to_vocal_onset(vocals, sr, vocal_onset_time)
        logger.info(f"Trimmed {vocal_onset_time:.2f}s of intro silence/instrumental")
        
        # Extract MFCCs from trimmed audio
        logger.info("Extracting MFCCs...")
        if progress_callback:
            progress_callback("Extracting MFCC features...")
        mfcc = preprocessor.extract_mfcc(trimmed_vocals, sr)
        frame_times = preprocessor.get_frame_times(len(mfcc), sr)
        
        # Convert lyrics to phonemes
        words = lyrics.split()
        logger.info(f"Converting {len(words)} words to phonemes...")
        if progress_callback:
            progress_callback("Converting lyrics to phonemes (G2P)...")
        phonemes, word_boundaries = self._words_to_phonemes(words)
        
        logger.info(f"Total phonemes: {len(phonemes)}")
        
        # Run Viterbi alignment
        logger.info("Running Viterbi forced alignment...")
        if progress_callback:
            progress_callback("Running Viterbi forced alignment...")
        phoneme_boundaries = self._viterbi_alignment(mfcc, phonemes)
        
        # Map phoneme boundaries back to word boundaries
        logger.info("Mapping phoneme boundaries to words...")
        if progress_callback:
            progress_callback("Finalizing word boundaries...")
        aligned_words = []
        
        for start_ph_idx, end_ph_idx, word_text in word_boundaries:
            # Find frames for phonemes in this word
            word_frames = []
            
            for frame_start, frame_end, phoneme in phoneme_boundaries:
                ph_idx = phonemes.index(phoneme) if phoneme in phonemes else -1
                if start_ph_idx <= ph_idx < end_ph_idx:
                    word_frames.extend(range(frame_start, frame_end))
            
            if word_frames:
                start_time = frame_times[min(word_frames)]
                end_time = frame_times[min(max(word_frames), len(frame_times) - 1)]
            else:
                # Fallback: estimate based on phoneme count
                if phoneme_boundaries:
                    total_frames = phoneme_boundaries[-1][1]
                    frames_per_phoneme = total_frames / max(len(phonemes), 1)
                    start_time = frame_times[int(start_ph_idx * frames_per_phoneme)]
                    end_time = frame_times[int(min(end_ph_idx * frames_per_phoneme, len(frame_times) - 1))]
                else:
                    continue
            
            # Add vocal onset offset to get absolute time in original audio
            aligned_words.append({
                'text': word_text,
                'start': round(start_time + vocal_onset_time, 3),
                'end': round(end_time + vocal_onset_time, 3)
            })
        
        logger.info(f"HMM alignment complete. Aligned {len(aligned_words)} words.")
        
        return {
            'model': 'hmm',
            'words': aligned_words,
            'metadata': {
                'n_phonemes': len(phonemes),
                'source_separation': use_source_separation,
                'vocal_onset_time': round(vocal_onset_time, 3)
            }
        }


# Global instance
hmm_aligner = HMMAligner()
