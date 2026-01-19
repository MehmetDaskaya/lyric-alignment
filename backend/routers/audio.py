"""
Audio Router
API endpoints for audio streaming and processing
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import logging

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/audio/{task_id}")
async def get_audio(task_id: str, separated: bool = False):
    """
    Stream audio file for a task
    
    Args:
        task_id: Task ID
        separated: If true, return separated vocals
    """
    # Look for audio file
    audio_files = list(settings.UPLOAD_DIR.glob(f"{task_id}_*"))
    
    if not audio_files:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_path = audio_files[0]
    
    if separated:
        # Check for separated vocals
        vocals_path = settings.OUTPUT_DIR / f"{task_id}_vocals.wav"
        if vocals_path.exists():
            audio_path = vocals_path
    
    # Determine media type
    suffix = audio_path.suffix.lower()
    media_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac'
    }
    media_type = media_types.get(suffix, 'audio/mpeg')
    
    return FileResponse(
        audio_path,
        media_type=media_type,
        filename=audio_path.name
    )


@router.get("/waveform/{task_id}")
async def get_waveform_data(task_id: str):
    """
    Get waveform data for visualization
    
    Returns JSON with normalized amplitude data
    """
    import numpy as np
    from models.preprocessing import preprocessor
    
    # Find audio file
    audio_files = list(settings.UPLOAD_DIR.glob(f"{task_id}_*"))
    
    if not audio_files:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio_path = str(audio_files[0])
    
    try:
        # Load audio
        audio, sr = preprocessor.load_audio(audio_path)
        
        # Downsample for visualization (max 1000 points)
        target_points = 1000
        hop = max(1, len(audio) // target_points)
        
        # Get peak amplitude for each segment
        waveform_data = []
        for i in range(0, len(audio), hop):
            segment = audio[i:i + hop]
            peak = float(np.max(np.abs(segment)))
            waveform_data.append(peak)
        
        duration = len(audio) / sr
        
        return {
            "task_id": task_id,
            "duration": round(duration, 3),
            "sample_rate": sr,
            "points": len(waveform_data),
            "waveform": waveform_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get waveform: {e}")
        raise HTTPException(status_code=500, detail=str(e))
