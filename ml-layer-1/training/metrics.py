"""
Metrics Module

Utility functions for computing and tracking metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricsTracker:
    """Tracks metrics over time."""
    
    name: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def record(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """Record a metrics snapshot."""
        entry = {
            'timestamp': (timestamp or datetime.utcnow()).isoformat(),
            **metrics,
        }
        self.history.append(entry)
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get most recent metrics."""
        return self.history[-1] if self.history else None
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric."""
        entries = self.history[-last_n:] if last_n else self.history
        values = [e[key] for e in entries if key in e]
        return np.mean(values) if values else 0.0
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            'name': self.name,
            'history': self.history,
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MetricsTracker':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        tracker = cls(name=data['name'])
        tracker.history = data['history']
        return tracker


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict with mse, rmse, mae, r2
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'mse': np.nan, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification metrics (for direction prediction).
    
    Args:
        y_true: True directions (-1, 0, 1)
        y_pred: Predicted directions
        
    Returns:
        Dict with accuracy, precision, recall, f1
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.sign(y_true[mask])
    y_pred = np.sign(y_pred[mask])
    
    if len(y_true) == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    accuracy = np.mean(y_true == y_pred)
    
    true_positives = np.sum((y_true > 0) & (y_pred > 0))
    false_positives = np.sum((y_true <= 0) & (y_pred > 0))
    false_negatives = np.sum((y_true > 0) & (y_pred <= 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def compute_trading_metrics(
    returns: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute trading-specific metrics.
    
    Args:
        returns: Actual returns
        predictions: Predicted returns
        threshold: Threshold for taking position
        
    Returns:
        Dict with trading metrics
    """
    mask = np.isfinite(returns) & np.isfinite(predictions)
    returns = returns[mask]
    predictions = predictions[mask]
    
    if len(returns) == 0:
        return {
            'hit_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'sharpe_ratio': 0.0,
        }
    
    positions = np.sign(predictions) * (np.abs(predictions) > threshold)
    strategy_returns = positions * returns
    
    trades = positions != 0
    if np.sum(trades) > 0:
        profitable = (strategy_returns > 0) & trades
        hit_rate = np.sum(profitable) / np.sum(trades)
    else:
        hit_rate = 0.0
    
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    
    total_wins = np.sum(wins) if len(wins) > 0 else 0.0
    total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
    
    if np.std(strategy_returns) > 0:
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(525600)
    else:
        sharpe_ratio = 0.0
    
    return {
        'hit_rate': float(hit_rate),
        'profit_factor': float(profit_factor),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'sharpe_ratio': float(sharpe_ratio),
    }


class PerformanceMonitor:
    """Monitors model performance over time."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.predictions: List[float] = []
        self.actuals: List[float] = []
        self.timestamps: List[datetime] = []
        self.metrics_tracker = MetricsTracker(name=f"{model_name}_performance")
    
    def record_prediction(
        self,
        prediction: float,
        actual: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a prediction (and optional actual value)."""
        self.predictions.append(prediction)
        if actual is not None:
            self.actuals.append(actual)
        self.timestamps.append(timestamp or datetime.utcnow())
    
    def update_actual(self, actual: float) -> None:
        """Update the actual value for the oldest unmatched prediction."""
        if len(self.actuals) < len(self.predictions):
            self.actuals.append(actual)
    
    def compute_current_metrics(self) -> Dict[str, float]:
        """Compute metrics for recorded predictions."""
        if len(self.actuals) == 0:
            return {}
        
        n = min(len(self.predictions), len(self.actuals))
        preds = np.array(self.predictions[:n])
        acts = np.array(self.actuals[:n])
        
        regression = compute_regression_metrics(acts, preds)
        classification = compute_classification_metrics(acts, preds)
        
        metrics = {**regression, **classification}
        self.metrics_tracker.record(metrics)
        
        return metrics
    
    def get_rolling_accuracy(self, window: int = 100) -> float:
        """Get rolling direction accuracy."""
        n = min(len(self.predictions), len(self.actuals))
        if n < window:
            return 0.0
        
        preds = np.array(self.predictions[-window:])
        acts = np.array(self.actuals[-window:])
        
        return float(np.mean(np.sign(preds) == np.sign(acts)))
