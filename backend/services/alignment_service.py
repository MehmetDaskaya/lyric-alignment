"""
Alignment Service
Business logic for audio-lyric alignment
"""
import logging
from typing import Dict, Optional
import time

from models.preprocessing import preprocessor
from models.dtw_aligner import dtw_aligner
from models.hmm_aligner import hmm_aligner
from models.ctc_aligner import ctc_aligner

logger = logging.getLogger(__name__)


class AlignmentService:
    """Service for orchestrating alignment operations"""
    
    def __init__(self):
        self.aligners = {
            'dtw': dtw_aligner,
            'hmm': hmm_aligner,
            'ctc': ctc_aligner
        }
    
    def process_alignment(
        self,
        audio_path: str,
        lyrics: str,
        model: str = 'ctc',
        use_source_separation: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Process alignment with selected model
        
        Args:
            audio_path: Path to audio file
            lyrics: Lyrics text
            model: Model to use (dtw, hmm, ctc, all)
            use_source_separation: Whether to use Demucs
            progress_callback: Optional callback for status updates
            
        Returns:
            Alignment results
        """
        if progress_callback:
            progress_callback(f"Starting alignment with model: {model.upper()}")
            
        logger.info(f"Processing alignment with model: {model}")
        
        # Load audio
        if progress_callback:
            progress_callback("Loading audio file...")
        audio, sr = preprocessor.load_audio(audio_path)
        
        if model == 'all':
            # Run all models
            results = {}
            for name, aligner in self.aligners.items():
                if progress_callback:
                    progress_callback(f"Running {name.upper()} model...")
                    
                logger.info(f"Running {name} aligner...")
                start_time = time.time()
                
                result = aligner.align(
                    audio=audio,
                    sr=sr,
                    lyrics=lyrics,
                    use_source_separation=use_source_separation,
                    progress_callback=progress_callback
                )
                
                elapsed = time.time() - start_time
                result['metadata']['processing_time'] = round(elapsed, 3)
                result['metadata']['rtf'] = round(elapsed / (len(audio) / sr), 4)
                
                results[name] = result
            
            return results
        
        else:
            # Run single model
            if model not in self.aligners:
                raise ValueError(f"Unknown model: {model}")
            
            aligner = self.aligners[model]
            
            start_time = time.time()
            result = aligner.align(
                audio=audio,
                sr=sr,
                lyrics=lyrics,
                use_source_separation=use_source_separation,
                progress_callback=progress_callback
            )
            elapsed = time.time() - start_time
            
            result['metadata']['processing_time'] = round(elapsed, 3)
            result['metadata']['rtf'] = round(elapsed / (len(audio) / sr), 4)
            
            return result
    
    def get_supported_models(self):
        """Get list of supported models"""
        return list(self.aligners.keys())


# Global instance
alignment_service = AlignmentService()
