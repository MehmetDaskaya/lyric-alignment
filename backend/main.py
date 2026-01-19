"""
Lyric-Audio Alignment Backend
FastAPI server for audio processing and alignment
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from routers import alignment, audio, evaluation, data, batch
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Lyric-Audio Alignment API...")
    
    # Ensure NLTK resources are available
    try:
        import nltk
        resources = ['averaged_perceptron_tagger_eng', 'cmudict', 'punkt'] 
        for res in resources:
            try:
                nltk.data.find(f'*/{res}')
            except LookupError:
                logger.info(f"Downloading missing NLTK resource: {res}")
                nltk.download(res, quiet=True)
    except Exception as e:
        logger.warning(f"Failed to verify NLTK resources: {e}")

    logger.info(f"Models will be loaded on first use")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Lyric-Audio Alignment API",
    description="API for comparing DTW, HMM, and Deep Learning alignment methods",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(alignment.router, prefix="/api", tags=["alignment"])
app.include_router(audio.router, prefix="/api", tags=["audio"])
app.include_router(evaluation.router, prefix="/api", tags=["evaluation"])
app.include_router(data.router, prefix="/api", tags=["data"])
app.include_router(batch.router, prefix="/api", tags=["batch"])


@app.get("/")
async def root():
    return {
        "message": "Lyric-Audio Alignment API",
        "docs": "/docs",
        "models": ["dtw", "hmm", "ctc"]
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
