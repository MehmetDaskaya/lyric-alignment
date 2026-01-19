from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List, Dict
import os
from config import settings

router = APIRouter(
    prefix="/data",
    tags=["data"],
    responses={404: {"description": "Not found"}},
)

@router.get("/songs")
async def list_songs() -> List[Dict[str, str]]:
    """
    List available songs in the DALI dataset.
    """
    try:
        # Import inside function to avoid circular imports if any, or just strictly use the loader
        from utils.dali_loader import dali_loader
        
        # Ensure metadata is available (might trigger download if not present)
        if not dali_loader.dali_data:
             try:
                 dali_loader.download_dataset_metadata()
             except Exception:
                 pass # If it fails, we might still have some data or return empty
        
        dali_songs = dali_loader.list_songs(limit=100) # List up to 100 for now
        
        return [
            {
                "id": s['id'],
                "name": f"{s['artist']} - {s['title']}",
                "filename": f"{s['id']}.mp3" # Virtual filename
            }
            for s in dali_songs
        ]
    except Exception as e:
        # Fallback or empty if DALI not ready
        return []
