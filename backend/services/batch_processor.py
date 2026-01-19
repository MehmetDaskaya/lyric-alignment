"""
Batch processing service for running alignment on multiple songs.
"""
import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import settings
from utils.dali_loader import dali_loader

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SongResult:
    song_id: str
    song_name: str
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class BatchJob:
    id: str
    song_ids: List[str]
    models: List[str]
    use_source_separation: bool
    status: JobStatus = JobStatus.PENDING
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0
    total: int = 0
    current_song: str = ""
    current_model: str = ""
    results: List[SongResult] = field(default_factory=list)
    error: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.total = len(self.song_ids) * len(self.models)


class BatchProcessor:
    """Manages batch processing jobs for lyric alignment."""
    
    def __init__(self):
        self.jobs: Dict[str, BatchJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    def calculate_metrics(self, predicted_words: List[Dict], ground_truth_words: List[Dict]) -> Dict[str, float]:
        """
        Calculate alignment metrics (MAE, Accuracy)
        
        Args:
            predicted_words: List of predicted word alignments
            ground_truth_words: List of ground truth word alignments
            
        Returns:
            Dictionary of metrics
        """
        if not predicted_words or not ground_truth_words:
            return {'mae': 0.0, 'accuracy': 0.0, 'error': 'No predictions or ground truth'}
            
        # Align sequences based on text (simple matching assuming same text content)
        # In reality, might need Needleman-Wunsch if texts slightly differ, 
        # but here we assume clean DALI text.
        
        # Filter out words that might not match exact count if tokenization differs
        # For this baseline, we assume 1:1 mapping as we used DALI lyrics to generate alignment
        
        # Create map of GT words by index to handle potential mismatches if needed?
        # Let's assume ideal case first:
        
        errors = []
        correct_count = 0
        tolerance = 0.5 # 500ms tolerance
        
        # Use the shorter length to avoid index out of bounds
        n = min(len(predicted_words), len(ground_truth_words))
        
        for i in range(n):
            pred = predicted_words[i]
            # Find matching GT word (simple index match for now, could be improved)
            gt = ground_truth_words[i]
            
            # MAE of start times
            error = abs(pred['start'] - gt['start'])
            errors.append(error)
            
            # Accuracy (start time within tolerance)
            if error <= tolerance:
                correct_count += 1
                
        mae = sum(errors) / len(errors) if errors else 0.0
        accuracy = (correct_count / n) * 100 if n > 0 else 0.0
        
        return {
            'mae': round(mae, 3),
            'accuracy': round(accuracy, 2),
            'matched_words': n
        }

    def create_job(
        self,
        song_ids: List[str],
        models: List[str],
        use_source_separation: bool = False
    ) -> BatchJob:
        """Create a new batch job."""
        job_id = str(uuid.uuid4())
        job = BatchJob(
            id=job_id,
            song_ids=song_ids,
            models=models,
            use_source_separation=use_source_separation,
            results=[
                SongResult(song_id=sid, song_name=sid)
                for sid in song_ids
            ]
        )
        self.jobs[job_id] = job
        logger.info(f"Created batch job {job_id} with {len(song_ids)} songs and {len(models)} models")
        return job
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[BatchJob]:
        """Get all jobs."""
        return list(self.jobs.values())
    
    async def start_job(self, job_id: str) -> bool:
        """Start processing a batch job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status == JobStatus.RUNNING:
            return False
        
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        
        # Start processing in background
        task = asyncio.create_task(self._process_job(job))
        self._running_tasks[job_id] = task
        
        return True
    
    async def _process_job(self, job: BatchJob):
        """Process all songs in the job."""
        try:
            from models.dtw_aligner import DTWAligner
            from models.hmm_aligner import HMMAligner
            from models.ctc_aligner import CTCAligner
            
            aligners = {
                'dtw': DTWAligner(),
                'hmm': HMMAligner(),
                'ctc': CTCAligner()
            }
            
            audio_dir = settings.DATA_DIR / "audio"
            lyrics_dir = settings.DATA_DIR / "lyrics"
            
            processed = 0
            
            for song_result in job.results:
                song_id = song_result.song_id
                job.current_song = song_id
                
                # Get info from DALI loader
                try:
                    song_info = dali_loader.get_song_info(song_id)
                    lyrics_text = " ".join([note['text'] for note in song_info['lyrics']])
                    
                    # Ensure audio exists
                    audio_path = dali_loader.download_audio(song_id)
                    if not audio_path:
                         raise FileNotFoundError(f"Could not download/find audio for {song_id}")
                         
                except Exception as e:
                    song_result.status = "failed"
                    song_result.error = f"Failed to load DALI data for {song_id}: {e}"
                    continue

                lyrics = lyrics_text # Use DALI text
                
                # Load audio
                try:
                    import librosa
                    audio, sr = librosa.load(str(audio_path), sr=settings.SAMPLE_RATE)
                except Exception as e:
                    song_result.status = "failed"
                    song_result.error = f"Failed to load audio: {e}"
                    continue
                
                song_result.status = "processing"
                start_time = datetime.now()
                
                for model_name in job.models:
                    job.current_model = model_name
                    
                    try:
                        aligner = aligners.get(model_name)
                        if not aligner:
                            continue
                        
                        # Run alignment with correct signature: (audio_array, sr, lyrics, ...)
                        def run_align(a, audio_arr, sample_rate, ly, ss):
                            return a.align(audio_arr, sample_rate, ly, use_source_separation=ss)
                        
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            run_align,
                            aligner,
                            audio,
                            sr,
                            lyrics,
                            job.use_source_separation
                        )
                        
                        song_result.results[model_name] = {
                            'words': result.get('words', []),
                            'metadata': result.get('metadata', {})
                        }
                        
                        # Extract metrics
                        metadata = result.get('metadata', {})
                        song_result.metrics[model_name] = {
                            'processing_time': metadata.get('processing_time', 0),
                            'rtf': metadata.get('rtf', 0),
                            'word_count': len(result.get('words', []))
                        }
                        
                        # Calculate accuracy metrics against Ground Truth
                        try:
                            gt_data = dali_loader.get_ground_truth_alignment(song_id)
                            metrics = self.calculate_metrics(
                                result.get('words', []),
                                gt_data.get('words', [])
                            )
                            song_result.metrics[model_name].update(metrics)
                        except Exception as e:
                            logger.error(f"Failed to calculate metrics for {song_id}: {e}")
                            song_result.metrics[model_name]['error'] = f"Metric calc failed: {e}"
                        
                    except Exception as e:
                        logger.error(f"Error processing {song_id} with {model_name}: {e}")
                        song_result.results[model_name] = {'error': str(e)}
                    
                    processed += 1
                    job.progress = processed
                
                song_result.status = "completed"
                song_result.processing_time = (datetime.now() - start_time).total_seconds()
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()
            
            # Save results to file
            self._save_results(job)
            
        except Exception as e:
            logger.error(f"Batch job {job.id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
    
    def _save_results(self, job: BatchJob):
        """Save job results to a JSON file."""
        output_dir = settings.OUTPUT_DIR / "batch_results"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{job.id}.json"
        
        # Convert to dict
        job_dict = {
            'id': job.id,
            'status': job.status.value,
            'created_at': job.created_at,
            'completed_at': job.completed_at,
            'models': job.models,
            'use_source_separation': job.use_source_separation,
            'results': [
                {
                    'song_id': r.song_id,
                    'song_name': r.song_name,
                    'status': r.status,
                    'processing_time': r.processing_time,
                    'metrics': r.metrics,
                    'results': r.results
                }
                for r in job.results
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(job_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved batch results to {output_file}")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status != JobStatus.RUNNING:
            return False
        
        task = self._running_tasks.get(job_id)
        if task:
            task.cancel()
        
        job.status = JobStatus.CANCELLED
        return True


# Singleton instance
batch_processor = BatchProcessor()
