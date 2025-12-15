"""
Predictor Module

Handles real-time prediction using trained models.
Used during A/B testing before Rust inference takes over.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from datetime import datetime

from .config import TrainingConfig, NUM_FEATURES, SEQUENCE_LENGTH, NUM_HORIZONS

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a prediction."""
    timestamp: datetime
    predictions: np.ndarray  # (5,) for 5 horizons
    confidences: np.ndarray  # (5,) confidence scores
    mamba_predictions: Optional[np.ndarray] = None
    lgbm_predictions: Optional[np.ndarray] = None
    latency_ms: float = 0.0
    ensemble_version_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'predictions': self.predictions.tolist(),
            'confidences': self.confidences.tolist(),
            'mamba_predictions': self.mamba_predictions.tolist() if self.mamba_predictions is not None else None,
            'lgbm_predictions': self.lgbm_predictions.tolist() if self.lgbm_predictions is not None else None,
            'latency_ms': self.latency_ms,
            'ensemble_version_id': self.ensemble_version_id,
        }


class Predictor:
    """Real-time predictor using trained models."""
    
    def __init__(
        self,
        models_dir: str,
        device: str = "cpu",
        mamba_weight: float = 0.5,
    ):
        """
        Initialize predictor.
        
        Args:
            models_dir: Directory with exported models
            device: Inference device (cpu/cuda)
            mamba_weight: Weight for Mamba in ensemble
        """
        self.models_dir = Path(models_dir)
        self.device = torch.device(device)
        self.mamba_weight = mamba_weight
        self.lgbm_weight = 1.0 - mamba_weight
        
        self.mamba_model = None
        self.lgbm_sessions = None
        self.norm_params = None
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all models."""
        import time
        start = time.time()
        
        # Load Mamba TorchScript
        mamba_path = self.models_dir / "mamba.pt"
        if mamba_path.exists():
            self.mamba_model = torch.jit.load(str(mamba_path), map_location=self.device)
            self.mamba_model.eval()
            logger.info(f"Loaded Mamba model from {mamba_path}")
        else:
            logger.warning(f"Mamba model not found at {mamba_path}")
        
        # Load LightGBM ONNX
        try:
            import onnxruntime as ort
            self.lgbm_sessions = []
            for h in range(1, NUM_HORIZONS + 1):
                onnx_path = self.models_dir / f"lgbm_horizon_{h}.onnx"
                if onnx_path.exists():
                    session = ort.InferenceSession(str(onnx_path))
                    self.lgbm_sessions.append(session)
                else:
                    logger.warning(f"LightGBM model not found: {onnx_path}")
                    self.lgbm_sessions.append(None)
            logger.info(f"Loaded {len([s for s in self.lgbm_sessions if s])} LightGBM models")
        except ImportError:
            logger.warning("onnxruntime not installed, LightGBM inference disabled")
            self.lgbm_sessions = None
        
        # Load normalization params
        norm_path = self.models_dir / "normalization_params_v1.json"
        if norm_path.exists():
            with open(norm_path) as f:
                self.norm_params = json.load(f)
            logger.info(f"Loaded normalization params from {norm_path}")
        else:
            logger.warning(f"Normalization params not found at {norm_path}")
        
        load_time = time.time() - start
        logger.info(f"Models loaded in {load_time:.2f}s")
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization.
        
        Args:
            features: Raw features (seq_len, num_features)
            
        Returns:
            Normalized features
        """
        if self.norm_params is None:
            return features
        
        mean = np.array(self.norm_params['mean'])
        std = np.array(self.norm_params['std'])
        
        normalized = (features - mean) / std
        return np.clip(normalized, -10, 10)
    
    def predict_mamba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Run Mamba inference.
        
        Args:
            sequence: Normalized sequence (1, seq_len, num_features)
            
        Returns:
            Predictions (num_horizons,)
        """
        if self.mamba_model is None:
            return np.zeros(NUM_HORIZONS)
        
        with torch.no_grad():
            x = torch.FloatTensor(sequence).to(self.device)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            output = self.mamba_model(x)
            return output.cpu().numpy().flatten()
    
    def predict_lgbm(self, sequence: np.ndarray) -> np.ndarray:
        """
        Run LightGBM inference.
        
        Args:
            sequence: Normalized sequence (seq_len, num_features)
            
        Returns:
            Predictions (num_horizons,)
        """
        if self.lgbm_sessions is None:
            return np.zeros(NUM_HORIZONS)
        
        # Flatten sequence
        x = sequence.flatten().reshape(1, -1).astype(np.float32)
        
        predictions = np.zeros(NUM_HORIZONS)
        for i, session in enumerate(self.lgbm_sessions):
            if session is not None:
                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: x})
                predictions[i] = output[0].flatten()[0]
        
        return predictions
    
    def predict(
        self,
        features: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> PredictionResult:
        """
        Generate ensemble prediction.
        
        Args:
            features: Raw features (seq_len, num_features)
            timestamp: Prediction timestamp
            
        Returns:
            PredictionResult
        """
        import time
        start = time.time()
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Normalize
        normalized = self.normalize(features)
        
        # Run individual models
        mamba_preds = self.predict_mamba(normalized)
        lgbm_preds = self.predict_lgbm(normalized)
        
        # Combine
        ensemble_preds = self.mamba_weight * mamba_preds + self.lgbm_weight * lgbm_preds
        
        # Compute confidence (based on agreement)
        agreement = 1.0 - np.abs(mamba_preds - lgbm_preds) / (np.abs(mamba_preds) + np.abs(lgbm_preds) + 1e-8)
        confidences = np.clip(agreement, 0, 1)
        
        latency_ms = (time.time() - start) * 1000
        
        return PredictionResult(
            timestamp=timestamp,
            predictions=ensemble_preds,
            confidences=confidences,
            mamba_predictions=mamba_preds,
            lgbm_predictions=lgbm_preds,
            latency_ms=latency_ms,
        )
    
    def reload_models(self) -> None:
        """Reload models (for hot-reload support)."""
        logger.info("Reloading models...")
        self._load_models()


def create_predictor(
    models_dir: str,
    device: str = "cpu",
    mamba_weight: float = 0.5,
) -> Predictor:
    """Create a predictor instance."""
    return Predictor(models_dir, device, mamba_weight)
