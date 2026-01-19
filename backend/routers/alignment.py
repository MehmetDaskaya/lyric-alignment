"""
Alignment Router
API endpoints for lyric-audio alignment
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
import uuid
import logging
from pathlib import Path
import json
import shutil

from config import settings
from services.alignment_service import alignment_service

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory task storage (use Redis in production)
tasks = {}


from models.preprocessing import AudioPreprocessor

@router.post("/process_dali/{song_id}")
async def process_dali_song(song_id: str):
    """
    Process a song entirely from DALI dataset.
    """
    try:
        from utils.dali_loader import dali_loader
        
        # 1. Ensure audio is downloaded/available
        audio_path = dali_loader.download_audio(song_id)
        if not audio_path or not Path(audio_path).exists():
             raise HTTPException(status_code=404, detail="DALI audio not found or download failed")

        # 2. Get Lyrics
        lyrics = dali_loader.get_lyrics_text(song_id)
        if not lyrics:
             raise HTTPException(status_code=404, detail="DALI lyrics not found")

        # Create a task ID 
        task_id = str(uuid.uuid4())
        
        # Define destination paths (in uploads dir) - we behave like the old system for compatibility
        dest_audio = settings.UPLOAD_DIR / f"{task_id}_{song_id}.mp3" 
        
        # Copy DALI audio to upload dir (mimicking "upload" for the pipeline)
        shutil.copy2(audio_path, dest_audio)
        
        # Store task info 
        tasks[task_id] = {
            "id": task_id,
            "status": "uploaded",
            "audio_path": str(dest_audio),
            "lyrics": lyrics,
            "dali_id": song_id, # Track DALI ID for GT fetch later?
            "results": {}
        }
        
        logger.info(f"Initialized DALI task {task_id} for song {song_id}")

        return {
            "task_id": task_id,
            "status": "processing_started", 
            "message": "DALI song initialized successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to init DALI task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Removed: @router.post("/upload") - No longer supported strictly


def background_alignment_task(
    task_id: str,
    model: str,
    use_source_separation: bool
):
    """
    Background task wrapper for alignment process
    """
    try:
        task = tasks[task_id]
        
        def progress_callback(status_msg):
            task["status"] = "processing"
            task["progress"] = status_msg
            logger.info(f"Task {task_id} progress: {status_msg}")
            
        progress_callback(f"Starting alignment with {model.upper()}...")
        
        result = alignment_service.process_alignment(
            audio_path=task["audio_path"],
            lyrics=task["lyrics"],
            model=model,
            use_source_separation=use_source_separation,
            progress_callback=progress_callback
        )
        
        task["results"][model] = result
        task["status"] = "completed"
        task["progress"] = "Completed successfully"
        
        # Save results to file
        output_path = settings.OUTPUT_DIR / f"{task_id}_{model}.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
            
    except Exception as e:
        logger.error(f"Background task failed: {e}")
        task["status"] = "error"
        task["error"] = str(e)
        task["progress"] = "Failed"

@router.post("/process/{task_id}")
async def process_alignment(
    task_id: str,
    model: str = Form(...),
    use_source_separation: bool = Form(True),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Start alignment processing with selected model (Async)
    
    Args:
        task_id: Task ID from upload
        model: Model to use (dtw, hmm, ctc)
        use_source_separation: Whether to use Demucs
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    valid_models = ['dtw', 'hmm', 'ctc', 'all']
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {valid_models}"
        )
    
    task = tasks[task_id]
    task["status"] = "queued"
    task["progress"] = "Added to queue..."
    task["model"] = model
    
    # Enqueue background task
    background_tasks.add_task(
        background_alignment_task,
        task_id=task_id,
        model=model,
        use_source_separation=use_source_separation
    )
    
    return {"task_id": task_id, "status": "queued", "message": "Alignment started in background"}

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the current status and progress of a task.
    Used for polling by the frontend.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task = tasks[task_id]
    return {
        "id": task["id"],
        "status": task["status"],
        "progress": task.get("progress", ""),
        "error": task.get("error", None),
        "results": list(task.get("results", {}).keys())
    }

@router.get("/results/{task_id}")
async def get_results(task_id: str, model: Optional[str] = None):
    """
    Get alignment results for a task
    
    Args:
        task_id: Task ID
        model: Optional specific model results (dtw, hmm, ctc)
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if model:
        if model not in task["results"]:
            raise HTTPException(
                status_code=404, 
                detail=f"No results for model: {model}"
            )
        return task["results"][model]
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "results": task["results"]
    }


@router.get("/tasks")
async def list_tasks():
    """List all tasks (for debugging)"""
    return {
        "tasks": [
            {
                "id": t["id"],
                "status": t["status"],
                "models": list(t.get("results", {}).keys())
            }
            for t in tasks.values()
        ]
    }


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its files"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    # Delete audio file
    audio_path = Path(task["audio_path"])
    if audio_path.exists():
        audio_path.unlink()
    
    # Delete result files
    for model in task.get("results", {}).keys():
        result_path = settings.OUTPUT_DIR / f"{task_id}_{model}.json"
        if result_path.exists():
            result_path.unlink()
    
    del tasks[task_id]
    
    return {"message": f"Task {task_id} deleted"}
