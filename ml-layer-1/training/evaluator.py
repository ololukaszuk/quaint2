"""
Evaluator Module

Evaluates model performance and computes metrics.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

from .config import NUM_HORIZONS, PREDICTION_HORIZONS

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a model."""
    mse: float
    rmse: float
    mae: float
    direction_accuracy: float
    r2: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'direction_accuracy': self.direction_accuracy,
            'r2': self.r2,
        }


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> EvaluationMetrics:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted values (N,) or (N, horizons)
        targets: Target values (N,) or (N, horizons)
        
    Returns:
        EvaluationMetrics
    """
    # Flatten if multi-horizon
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Remove NaN/Inf
    mask = np.isfinite(predictions) & np.isfinite(targets)
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(predictions) == 0:
        return EvaluationMetrics(
            mse=float('nan'),
            rmse=float('nan'),
            mae=float('nan'),
            direction_accuracy=0.0,
            r2=float('nan'),
        )
    
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # Direction accuracy
    pred_direction = np.sign(predictions)
    target_direction = np.sign(targets)
    direction_accuracy = np.mean(pred_direction == target_direction)
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return EvaluationMetrics(
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        direction_accuracy=float(direction_accuracy),
        r2=float(r2),
    )


def compute_horizon_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    horizons: List[int] = PREDICTION_HORIZONS,
) -> Dict[int, EvaluationMetrics]:
    """
    Compute metrics for each prediction horizon.
    
    Args:
        predictions: (N, num_horizons)
        targets: (N, num_horizons)
        horizons: List of horizon values
        
    Returns:
        Dict mapping horizon to metrics
    """
    metrics = {}
    
    for i, h in enumerate(horizons):
        if i < predictions.shape[1]:
            metrics[h] = compute_metrics(predictions[:, i], targets[:, i])
    
    return metrics


class ModelEvaluator:
    """Evaluator for trained models."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {}
    
    def evaluate(
        self,
        model_name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions.
        
        Args:
            model_name: Name of the model
            predictions: (N, num_horizons)
            targets: (N, num_horizons)
            times: Optional timestamps
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Overall metrics
        overall = compute_metrics(predictions, targets)
        
        # Per-horizon metrics
        horizon_metrics = compute_horizon_metrics(predictions, targets)
        
        results = {
            'model_name': model_name,
            'overall': overall.to_dict(),
            'per_horizon': {h: m.to_dict() for h, m in horizon_metrics.items()},
            'sample_count': len(predictions),
        }
        
        self.results[model_name] = results
        
        # Log summary
        logger.info(f"{model_name} Results:")
        logger.info(f"  Overall - RMSE: {overall.rmse:.6f}, Direction Acc: {overall.direction_accuracy:.4f}")
        for h, m in horizon_metrics.items():
            logger.info(f"  Horizon {h}m - RMSE: {m.rmse:.6f}, Direction Acc: {m.direction_accuracy:.4f}")
        
        return results
    
    def compare_models(
        self,
        model_a: str,
        model_b: str,
    ) -> Dict[str, Any]:
        """
        Compare two models.
        
        Args:
            model_a: First model name
            model_b: Second model name
            
        Returns:
            Comparison results
        """
        if model_a not in self.results or model_b not in self.results:
            raise ValueError(f"Both models must be evaluated first")
        
        a = self.results[model_a]
        b = self.results[model_b]
        
        comparison = {
            'model_a': model_a,
            'model_b': model_b,
            'rmse_diff': a['overall']['rmse'] - b['overall']['rmse'],
            'direction_acc_diff': a['overall']['direction_accuracy'] - b['overall']['direction_accuracy'],
            'better_model': model_a if a['overall']['direction_accuracy'] > b['overall']['direction_accuracy'] else model_b,
        }
        
        # Per-horizon comparison
        horizon_comparison = {}
        for h in a['per_horizon'].keys():
            if h in b['per_horizon']:
                horizon_comparison[h] = {
                    'rmse_diff': a['per_horizon'][h]['rmse'] - b['per_horizon'][h]['rmse'],
                    'direction_acc_diff': a['per_horizon'][h]['direction_accuracy'] - b['per_horizon'][h]['direction_accuracy'],
                }
        
        comparison['per_horizon'] = horizon_comparison
        
        return comparison
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        summary = {}
        
        for model_name, results in self.results.items():
            summary[model_name] = {
                'rmse': results['overall']['rmse'],
                'mae': results['overall']['mae'],
                'direction_accuracy': results['overall']['direction_accuracy'],
                'r2': results['overall']['r2'],
            }
        
        return summary


def evaluate_ensemble(
    mamba_predictions: np.ndarray,
    lgbm_predictions: np.ndarray,
    targets: np.ndarray,
    mamba_weight: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Evaluate ensemble predictions with different weights.
    
    Args:
        mamba_predictions: (N, num_horizons)
        lgbm_predictions: (N, num_horizons)
        targets: (N, num_horizons)
        mamba_weight: Weight for Mamba model
        
    Returns:
        (ensemble_predictions, evaluation_results)
    """
    lgbm_weight = 1.0 - mamba_weight
    ensemble_preds = mamba_weight * mamba_predictions + lgbm_weight * lgbm_predictions
    
    evaluator = ModelEvaluator()
    
    # Evaluate individual models
    evaluator.evaluate('mamba', mamba_predictions, targets)
    evaluator.evaluate('lgbm', lgbm_predictions, targets)
    evaluator.evaluate('ensemble', ensemble_preds, targets)
    
    return ensemble_preds, evaluator.get_summary()


def find_optimal_weights(
    mamba_predictions: np.ndarray,
    lgbm_predictions: np.ndarray,
    targets: np.ndarray,
    metric: str = 'direction_accuracy',
) -> Tuple[float, float]:
    """
    Find optimal ensemble weights.
    
    Args:
        mamba_predictions: (N, num_horizons)
        lgbm_predictions: (N, num_horizons)
        targets: (N, num_horizons)
        metric: Metric to optimize ('direction_accuracy' or 'rmse')
        
    Returns:
        (best_mamba_weight, best_metric_value)
    """
    best_weight = 0.5
    best_value = -float('inf') if metric == 'direction_accuracy' else float('inf')
    
    for weight in np.arange(0.0, 1.05, 0.05):
        ensemble = weight * mamba_predictions + (1 - weight) * lgbm_predictions
        metrics = compute_metrics(ensemble, targets)
        
        value = metrics.direction_accuracy if metric == 'direction_accuracy' else -metrics.rmse
        
        if value > best_value:
            best_value = value
            best_weight = weight
    
    if metric == 'rmse':
        best_value = -best_value
    
    logger.info(f"Optimal Mamba weight: {best_weight:.2f} (metric: {best_value:.4f})")
    
    return best_weight, best_value
