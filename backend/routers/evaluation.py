"""
Evaluation Router
API endpoints for model evaluation and metrics
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
import logging

from services.evaluation_service import evaluation_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/evaluate")
async def evaluate_alignment(
    prediction: dict,
    ground_truth: dict,
    tolerance: float = 0.3
):
    """
    Evaluate alignment prediction against ground truth
    
    Args:
        prediction: Predicted alignment (from model)
        ground_truth: Ground truth alignment (from dataset)
        tolerance: Tolerance threshold for PC metric (seconds)
        
    Returns:
        Evaluation metrics (MAE, PC, etc.)
    """
    try:
        metrics = evaluation_service.compute_metrics(
            prediction=prediction,
            ground_truth=ground_truth,
            tolerance=tolerance
        )
        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_models(
    predictions: dict,
    ground_truth: dict,
    tolerance: float = 0.3
):
    """
    Compare multiple model predictions
    
    Args:
        predictions: Dict of model_name -> alignment results
        ground_truth: Ground truth alignment
        tolerance: Tolerance threshold
        
    Returns:
        Comparison metrics for all models
    """
    try:
        comparison = {}
        
        for model_name, prediction in predictions.items():
            metrics = evaluation_service.compute_metrics(
                prediction=prediction,
                ground_truth=ground_truth,
                tolerance=tolerance
            )
            comparison[model_name] = metrics
        
        # Add ranking
        if comparison:
            mae_ranking = sorted(
                comparison.items(), 
                key=lambda x: x[1].get('mae', float('inf'))
            )
            for rank, (model, _) in enumerate(mae_ranking, 1):
                comparison[model]['mae_rank'] = rank
        
        return {
            "comparison": comparison,
            "best_model": mae_ranking[0][0] if mae_ranking else None
        }
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmarks")
async def list_benchmarks():
    """
    List available benchmark datasets
    """
    return {
        "benchmarks": [
            {
                "name": "Hansen",
                "songs": 9,
                "description": "Hansen benchmark dataset"
            },
            {
                "name": "Mauch",
                "songs": 8,
                "description": "Mauch benchmark dataset (segments)"
            },
            {
                "name": "Jamendo",
                "songs": 20,
                "description": "Jamendo dataset (10 genres)"
            },
            {
                "name": "DALI",
                "songs": "5000+",
                "description": "DALI dataset (development/testing)"
            }
        ]
    }
