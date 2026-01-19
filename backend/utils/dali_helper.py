import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import DALI
import numpy as np

logger = logging.getLogger(__name__)

class DALIHelper:
    """
    Helper class for interacting with the DALI dataset.
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.dali_data = None
        
        if dataset_path and os.path.exists(dataset_path):
            try:
                # DALI.get_the_DALI_dataset returns a list of DALI entry objects
                self.dali_data = DALI.get_the_DALI_dataset(dataset_path, skip=[], keep=[])
                logger.info(f"Loaded DALI dataset with {len(self.dali_data)} entries")
            except Exception as e:
                logger.error(f"Failed to load DALI dataset: {e}")
    
    def get_entry(self, entry_id: str):
        """Get a specific DALI entry by ID"""
        if not self.dali_data:
            return None
        
        # This is inefficient but DALI 1.0 might be a list. 
        # In a real app, map to a dict.
        for entry in self.dali_data:
            if entry.info['id'] == entry_id:
                return entry
        return None

    def get_lyrics(self, entry_id: str) -> Optional[str]:
        """Extract lyrics text from a DALI entry"""
        entry = self.get_entry(entry_id)
        if not entry:
            return None
        
        # Extract plain text lyrics from annotations
        # DALI annotations have 'text' field for words/lines
        try:
            # annotations['type'] could be 'words', 'lines', etc.
            # let's assume we want word level
            words = [ann['text'] for ann in entry.annotations['annot']['words']]
            return " ".join(words)
        except Exception as e:
            logger.error(f"Failed to extract lyrics: {e}")
            return None

    def get_ground_truth_alignment(self, entry_id: str) -> List[Dict]:
        """Get ground truth word alignment"""
        entry = self.get_entry(entry_id)
        if not entry:
            return []
            
        try:
            alignment = []
            for ann in entry.annotations['annot']['words']:
                alignment.append({
                    'text': ann['text'],
                    'start': ann['time'][0],
                    'end': ann['time'][1]
                })
            return alignment
        except Exception as e:
            logger.error(f"Failed to extract alignment: {e}")
            return []

# Singleton instance (empty for now)
dali_helper = DALIHelper()
