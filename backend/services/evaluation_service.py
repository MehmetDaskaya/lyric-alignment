"""
Evaluation Service
Metrics computation for lyric alignment evaluation
"""
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for computing alignment evaluation metrics"""
    
    def compute_metrics(
        self,
        prediction: Dict,
        ground_truth: Dict,
        tolerance: float = 0.3
    ) -> Dict:
        """
        Compute evaluation metrics
        
        Args:
            prediction: Predicted alignment with 'words' list
            ground_truth: Ground truth alignment with 'words' list
            tolerance: Tolerance for PC metric (seconds)
            
        Returns:
            Dictionary with metrics
        """
        pred_words = prediction.get('words', [])
        gt_words = ground_truth.get('words', [])
        
        if not pred_words or not gt_words:
            return {
                'mae': None,
                'pc': None,
                'error': 'Empty alignment'
            }
        
        # Match words by text and order
        matched_pairs = self._match_words(pred_words, gt_words)
        
        if not matched_pairs:
            return {
                'mae': None,
                'pc': None,
                'error': 'No matching words found'
            }
        
        # Calculate metrics
        errors = []
        correct = 0
        
        for pred, gt in matched_pairs:
            # Absolute error between start times
            error = abs(pred['start'] - gt['start'])
            errors.append(error)
            
            # Check if within tolerance
            if error <= tolerance:
                correct += 1
        
        mae = np.mean(errors)
        pc = (correct / len(matched_pairs)) * 100
        
        # End time metrics
        end_errors = [
            abs(pred['end'] - gt['end']) 
            for pred, gt in matched_pairs
        ]
        mae_end = np.mean(end_errors)
        
        return {
            'mae': round(mae, 4),
            'mae_end': round(mae_end, 4),
            'pc': round(pc, 2),
            'tolerance': tolerance,
            'matched_words': len(matched_pairs),
            'total_pred': len(pred_words),
            'total_gt': len(gt_words),
            'error_std': round(np.std(errors), 4),
            'median_error': round(np.median(errors), 4),
            'max_error': round(max(errors), 4)
        }
    
    def _match_words(
        self, 
        pred_words: List[Dict], 
        gt_words: List[Dict]
    ) -> List[tuple]:
        """
        Match predicted words to ground truth words
        
        Uses text matching with order consideration
        """
        matched = []
        gt_idx = 0
        
        for pred in pred_words:
            pred_text = pred['text'].lower().strip()
            
            # Look for matching word in remaining GT words
            for i in range(gt_idx, len(gt_words)):
                gt_text = gt_words[i]['text'].lower().strip()
                
                if pred_text == gt_text:
                    matched.append((pred, gt_words[i]))
                    gt_idx = i + 1
                    break
        
        return matched
    
    def compute_rtf(
        self,
        processing_time: float,
        audio_duration: float
    ) -> float:
        """
        Compute Real-Time Factor
        
        RTF < 1 means faster than real-time
        RTF > 1 means slower than real-time
        """
        if audio_duration == 0:
            return float('inf')
        return processing_time / audio_duration
    
    def compare_models(
        self,
        predictions: Dict[str, Dict],
        ground_truth: Dict,
        tolerance: float = 0.3
    ) -> Dict:
        """
        Compare multiple model predictions
        
        Args:
            predictions: Dict mapping model_name -> prediction
            ground_truth: Ground truth alignment
            tolerance: Tolerance for PC metric
            
        Returns:
            Comparison results with rankings
        """
        results = {}
        
        for model_name, prediction in predictions.items():
            metrics = self.compute_metrics(
                prediction=prediction,
                ground_truth=ground_truth,
                tolerance=tolerance
            )
            results[model_name] = metrics
        
        # Compute rankings
        if results:
            # Rank by MAE (lower is better)
            mae_sorted = sorted(
                [(k, v.get('mae', float('inf'))) for k, v in results.items()],
                key=lambda x: x[1]
            )
            for rank, (model, _) in enumerate(mae_sorted, 1):
                results[model]['mae_rank'] = rank
            
            # Rank by PC (higher is better)
            pc_sorted = sorted(
                [(k, v.get('pc', 0)) for k, v in results.items()],
                key=lambda x: x[1],
                reverse=True
            )
            for rank, (model, _) in enumerate(pc_sorted, 1):
                results[model]['pc_rank'] = rank
        
        return results


# Global instance
evaluation_service = EvaluationService()
