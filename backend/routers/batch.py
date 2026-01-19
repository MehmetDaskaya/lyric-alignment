"""
Batch processing API router.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from services.batch_processor import batch_processor, JobStatus

router = APIRouter(
    prefix="/batch",
    tags=["batch"],
    responses={404: {"description": "Not found"}},
)


class BatchJobRequest(BaseModel):
    song_ids: List[str]
    models: List[str] = ["dtw", "hmm", "ctc"]
    use_source_separation: bool = False


class BatchJobResponse(BaseModel):
    id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    progress: int
    total: int
    current_song: str
    current_model: str
    error: Optional[str]
    results: Optional[List[Dict[str, Any]]]


@router.post("/start", response_model=BatchJobResponse)
async def start_batch_job(request: BatchJobRequest):
    """
    Start a new batch processing job.
    """
    if not request.song_ids:
        raise HTTPException(status_code=400, detail="No songs specified")
    
    if not request.models:
        raise HTTPException(status_code=400, detail="No models specified")
    
    # Validate models
    valid_models = {"dtw", "hmm", "ctc"}
    for model in request.models:
        if model not in valid_models:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model}")
    
    # Create and start job
    job = batch_processor.create_job(
        song_ids=request.song_ids,
        models=request.models,
        use_source_separation=request.use_source_separation
    )
    
    await batch_processor.start_job(job.id)
    
    return BatchJobResponse(
        id=job.id,
        status=job.status.value,
        message=f"Batch job started with {len(request.song_ids)} songs and {len(request.models)} models"
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a batch job.
    """
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Prepare results summary
    results_summary = []
    for r in job.results:
        results_summary.append({
            "song_id": r.song_id,
            "song_name": r.song_name,
            "status": r.status,
            "processing_time": r.processing_time,
            "metrics": r.metrics
        })
    
    return JobStatusResponse(
        id=job.id,
        status=job.status.value,
        progress=job.progress,
        total=job.total,
        current_song=job.current_song,
        current_model=job.current_model,
        error=job.error,
        results=results_summary if job.status == JobStatus.COMPLETED else None
    )


@router.get("/{job_id}/results")
async def get_job_results(job_id: str):
    """
    Get full results of a completed batch job.
    """
    job = batch_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job.status.value}"
        )
    
    return {
        "id": job.id,
        "status": job.status.value,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "models": job.models,
        "results": [
            {
                "song_id": r.song_id,
                "song_name": r.song_name,
                "status": r.status,
                "processing_time": r.processing_time,
                "metrics": r.metrics,
                "alignments": r.results
            }
            for r in job.results
        ]
    }


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a running batch job.
    """
    success = batch_processor.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not cancel job")
    
    return {"message": "Job cancelled"}


@router.get("/")
async def list_jobs():
    """
    List all batch jobs.
    """
    jobs = batch_processor.get_all_jobs()
    return {
        "jobs": [
            {
                "id": j.id,
                "status": j.status.value,
                "progress": j.progress,
                "total": j.total,
                "created_at": j.created_at,
                "completed_at": j.completed_at
            }
            for j in jobs
        ]
    }
