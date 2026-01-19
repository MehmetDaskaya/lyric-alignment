"""
Application configuration
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    MODEL_CACHE_DIR: Path = BASE_DIR / "model_cache"
    
    # Dataset Paths
    DATA_DIR: Path = BASE_DIR.parent / "data"
    DALI_AUDIO_PATH: Path = DATA_DIR / "audio"
    
    # Audio settings
    SAMPLE_RATE: int = 16000
    HOP_LENGTH: int = 160  # 10ms at 16kHz
    MFCC_HOP_LENGTH: int = 512  # Hop length for MFCC extraction
    WIN_LENGTH: int = 400  # 25ms at 16kHz
    N_MFCC: int = 39
    
    # Model settings
    WAV2VEC2_MODEL: str = "facebook/wav2vec2-base-960h"
    
    # API settings
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    class Config:
        env_file = ".env"


settings = Settings()

# Create directories
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.OUTPUT_DIR.mkdir(exist_ok=True)
settings.MODEL_CACHE_DIR.mkdir(exist_ok=True)
settings.DATA_DIR.mkdir(exist_ok=True)
