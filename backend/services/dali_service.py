import os
import sys
import gzip
import pickle
import logging
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import subprocess

from config import settings

logger = logging.getLogger(__name__)

# Ensure DALI library is importable
# We maintain the existing project structure's way of including DALI
dali_code_path = settings.BASE_DIR / "DALI" / "code"
if dali_code_path.exists() and str(dali_code_path) not in sys.path:
    sys.path.append(str(dali_code_path))

try:
    import DALI
except ImportError:
    logger.warning("DALI module not found. Some functionality might be limited.")
    DALI = None

class DaliService:
    """
    Unified service for interacting with the local DALI v1.0 dataset.
    Reads annotations directly from the DALI_v1.0 folder config.
    """

    def __init__(self):
        self.root_path = settings.DALI_ROOT
        self.audio_path = settings.DALI_AUDIO_PATH
        self._entries_cache: List[Path] = []
        self._id_map: Dict[str, Path] = {}
        
        # Ensure audio path exists
        self.audio_path.mkdir(parents=True, exist_ok=True)
        
        self._scan_dataset()

    def _scan_dataset(self):
        """Scans the DALI root directory for .gz annotation files."""
        if not self.root_path.exists():
            logger.error(f"DALI root path not found: {self.root_path}")
            return

        self._entries_cache = list(self.root_path.glob("*.gz"))
        self._id_map = {p.name.split('.')[0]: p for p in self._entries_cache}
        logger.info(f"DALI Service: Found {len(self._entries_cache)} entries in {self.root_path}")

    def get_all_ids(self) -> List[str]:
        return list(self._id_map.keys())

    def get_random_id(self) -> Optional[str]:
        if not self._entries_cache:
            return None
        return random.choice(list(self._id_map.keys()))

    def load_entry_data(self, entry_id: str) -> Optional[Any]:
        """Loads the raw pickle data for a given ID."""
        if entry_id not in self._id_map:
            logger.error(f"Entry ID {entry_id} not found.")
            return None

        file_path = self._id_map[entry_id]
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load entry {entry_id}: {e}")
            return None

    def get_metadata(self, entry_id: str) -> Dict[str, Any]:
        """Returns simplified metadata for an entry."""
        data = self.load_entry_data(entry_id)
        if not data:
            return {}
        
        info = getattr(data, 'info', {})
        if isinstance(info, dict):
            return {
                'id': entry_id,
                'artist': info.get('artist', 'Unknown'),
                'title': info.get('title', 'Unknown'),
                'language': info.get('metadata', {}).get('language', 'en'),
                'url': info.get('audio', {}).get('url', None)
            }
        return {'id': entry_id}

    def get_lyrics(self, entry_id: str) -> str:
        """Extracts plain lyrics text."""
        data = self.load_entry_data(entry_id)
        if not data:
            return ""
        
        try:
            # Assuming standard DALI 1.0 structure
            # annotations['annot']['words'] is a list of dicts with 'text'
            words = data.annotations['annot']['words']
            return " ".join([w['text'] for w in words])
        except Exception as e:
            logger.error(f"Error extracting lyrics for {entry_id}: {e}")
            return ""

    def get_ground_truth_alignment(self, entry_id: str) -> List[Dict]:
        """Returns list of {text, start, end}."""
        data = self.load_entry_data(entry_id)
        if not data:
            return []
            
        alignment = []
        try:
            words = data.annotations['annot']['words']
            for w in words:
                alignment.append({
                    'text': w['text'],
                    'start': w['time'][0],
                    'end': w['time'][1]
                })
        except Exception as e:
            logger.error(f"Error extracting alignment for {entry_id}: {e}")
            
        return alignment

    def get_audio_path(self, entry_id: str, download_if_missing: bool = True) -> Optional[Path]:
        """
        Returns the path to the audio file. 
        Downloads it via yt-dlp config if missing and requested.
        """
        # Search for existing file with common extensions
        for ext in ['.mp3', '.wav', '.m4a', '.flac']:
            path = self.audio_path / f"{entry_id}{ext}"
            if path.exists():
                return path
        
        if not download_if_missing:
            return None

        # Attempt download
        metadata = self.get_metadata(entry_id)
        url = metadata.get('url')
        if not url:
            logger.warning(f"No audio URL found for {entry_id}")
            return None

        return self._download_audio(entry_id, url)

    def _download_audio(self, entry_id: str, url: str) -> Optional[Path]:
        """Downloads audio using yt-dlp."""
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt_dlp not installed. Cannot download audio.")
            return None
        
        output_template = str(self.audio_path / entry_id) # yt-dlp adds extension
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True
        }
        
        logger.info(f"Downloading audio for {entry_id} from {url}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Verify file exists (yt-dlp adds .mp3)
            expected_path = self.audio_path / f"{entry_id}.mp3"
            if expected_path.exists():
                return expected_path
            return None
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

# Global instance
dali_service = DaliService()
