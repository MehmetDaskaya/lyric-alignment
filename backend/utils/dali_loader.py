"""
DALI Dataset Loader
Tools for downloading and interacting with the DALI dataset
Repo: https://github.com/gabolsgabs/DALI
"""
import os
import logging
from pathlib import Path
import sys

# Add DALI/code to path to import DALI package correctly
# Assuming structure: backend/DALI/code/DALI
dali_code_path = Path(__file__).parent.parent / "DALI" / "code"
if dali_code_path.exists() and str(dali_code_path) not in sys.path:
    sys.path.append(str(dali_code_path))

import DALI as dali_code
from typing import List, Dict, Optional, Tuple
import requests
import json

from config import settings

logger = logging.getLogger(__name__)

class DaliLoader:
    """Loader for DALI dataset"""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or str(settings.DATA_DIR / "dali")
        self.audio_path = str(Path(self.dataset_path) / "audio")
        self.dali_data = None
        
        # Ensure directories exist
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
        Path(self.audio_path).mkdir(parents=True, exist_ok=True)
    
    def download_dataset_metadata(self):
        """Download DALI metadata/annotations"""
        logger.info("Initializing DALI dataset...")
        
        try:
            # Initialize DALI dataset (this downloads metadata if not present)
            # You might need to point to specific path depending on DALI library setup
            self.dali_data = dali_code.get_the_DALI_dataset(self.dataset_path, skip=["audio"])
            logger.info(f"Loaded DALI metadata. Total songs: {len(self.dali_data)}")
            
        except Exception as e:
            logger.error(f"Failed to load DALI metadata: {e}")
            logger.info("Please follow instructions at https://github.com/gabolsgabs/DALI for manual setup if needed")
            raise

    def get_song_info(self, dali_id: str) -> Dict:
        """Get info for a specific song"""
        if not self.dali_data:
            self.download_dataset_metadata()
            
        entry = self.dali_data[dali_id]
        return {
            'id': dali_id,
            'artist': entry.info['artist'],
            'title': entry.info['title'],
            'audio_url': entry.info['audio']['url'],  # Usually YouTube URL
            'visual_url': entry.info.get('visual', {}).get('url', ''),
            'lyrics': [
                {
                    'text': note['text'],
                    'start': note['time'][0],
                    'end': note['time'][1]
                }
                for note in entry.annotations['annot']['notes']
            ]
        }
    
    def download_audio(self, dali_id: str):
        """
        Download audio for a DALI entry
        Note: DALI audio usually requires youtube-dl/yt-dlp
        """
        import yt_dlp
        
        info = self.get_song_info(dali_id)
        url = info['audio_url']
        
        output_path = Path(self.audio_path) / f"{dali_id}.mp3"
        
        if output_path.exists():
            logger.info(f"Audio already exists: {output_path}")
            return str(output_path)
            
        logger.info(f"Downloading audio for {dali_id} from {url}...")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(Path(self.audio_path) / f"{dali_id}"),
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        return str(output_path)

    def get_ground_truth_alignment(self, dali_id: str) -> Dict:
        """Get ground truth alignment in our standard format"""
        info = self.get_song_info(dali_id)
        
        words = []
        for note in info['lyrics']:
            words.append({
                'text': note['text'],
                'start': round(note['start'], 3),
                'end': round(note['end'], 3)
            })
            
        return {
            'model': 'ground_truth',
            'words': words,
            'metadata': {
                'dali_id': dali_id,
                'artist': info['artist'],
                'title': info['title']
            }
        }
        
    def list_songs(self, limit: int = 10) -> List[Dict]:
        """List available songs in dataset"""
        if not self.dali_data:
            self.download_dataset_metadata()
            
        songs = []
        for dali_id in list(self.dali_data.keys())[:limit]:
            info = self.dali_data[dali_id].info
            songs.append({
                'id': dali_id,
                'artist': info['artist'],
                'title': info['title']
            })
        return songs

    def get_lyrics_text(self, dali_id: str) -> str:
        """Get full lyrics text for a song"""
        info = self.get_song_info(dali_id)
        words = [w['text'] for w in info['lyrics']]
        return " ".join(words)

    def get_ground_truth_lines(self, dali_id: str) -> List[Dict]:
        """Get ground truth line alignment"""
        info = self.get_song_info(dali_id)
        if not self.dali_data:
             self.download_dataset_metadata()
             
        entry = self.dali_data[dali_id]
        lines = []
        try:
            # Check if lines exist in annotations
            if 'lines' in entry.annotations['annot']:
                for line in entry.annotations['annot']['lines']:
                    lines.append({
                        'text': line['text'],
                        'start': round(line['time'][0], 3),
                        'end': round(line['time'][1], 3)
                    })
        except Exception:
            pass
            
        return lines

# Global instance
dali_loader = DaliLoader()
