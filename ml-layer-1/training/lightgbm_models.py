"""
LightGBM Models Module

Implements LightGBM models for each prediction horizon.
Exports to ONNX format for Rust inference.
"""

import numpy as np
import lightgbm as lgb
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json
import logging

from .config import LightGBMConfig, NUM_HORIZONS, SEQUENCE_LENGTH, NUM_FEATURES

logger = logging.getLogger(__name__)


class LightGBMHorizonModel:
    """LightGBM model for a single prediction horizon."""
    
    def __init__(self, horizon: int, config: LightGBMConfig):
        """
        Initialize model for specific horizon.
        
        Args:
            horizon: Prediction horizon (1-5 minutes)
            config: Model configuration
        """
        self.horizon = horizon
        self.config = config
        self.model: Optional[lgb.Booster] = None
        
        self.params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'num_leaves': config.num_leaves,
            'min_child_samples': config.min_child_samples,
            'subsample': config.subsample,
            'colsample_bytree': config.colsample_bytree,
            'reg_alpha': config.reg_alpha,
            'reg_lambda': config.reg_lambda,
            'verbose': config.verbose,
            'force_col_wise': True,  # Better for wide data
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features (N, input_size)
            y_train: Training targets (N,)
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training history
        """
        logger.info(f"Training LightGBM for horizon {self.horizon}m...")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train with early stopping
        callbacks = [
            lgb.early_stopping(self.config.early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks,
        )
        
        # Get best iteration
        best_iter = self.model.best_iteration
        
        logger.info(f"Horizon {self.horizon}m training complete. Best iteration: {best_iter}")
        
        return {
            'best_iteration': best_iter,
            'train_loss': self.model.best_score['train']['l2'],
            'val_loss': self.model.best_score['val']['l2'],
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features (N, input_size)
            
        Returns:
            Predictions (N,)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.save_model(path)
        logger.info(f"Saved LightGBM model to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)
        logger.info(f"Loaded LightGBM model from {path}")
    
    def export_onnx(self, output_path: str) -> None:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path for output .onnx file
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            import onnxmltools
            from onnxmltools.convert import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            # Define input shape
            input_size = SEQUENCE_LENGTH * NUM_FEATURES
            initial_type = [('input', FloatTensorType([None, input_size]))]
            
            # Convert to ONNX
            onnx_model = convert_lightgbm(
                self.model,
                initial_types=initial_type,
                target_opset=12,
            )
            
            # Save
            onnxmltools.utils.save_model(onnx_model, output_path)
            logger.info(f"Exported ONNX model to {output_path}")
            
            # Verify
            self._verify_onnx_export(output_path)
            
        except ImportError:
            logger.warning("onnxmltools not installed, skipping ONNX export")
            raise
    
    def _verify_onnx_export(self, onnx_path: str) -> None:
        """Verify ONNX export produces same outputs."""
        try:
            import onnxruntime as ort
            
            # Create session
            session = ort.InferenceSession(onnx_path)
            
            # Test input
            test_input = np.random.randn(10, SEQUENCE_LENGTH * NUM_FEATURES).astype(np.float32)
            
            # Original prediction
            original_pred = self.model.predict(test_input)
            
            # ONNX prediction
            input_name = session.get_inputs()[0].name
            onnx_pred = session.run(None, {input_name: test_input})[0].flatten()
            
            # Compare
            max_diff = np.max(np.abs(original_pred - onnx_pred))
            assert max_diff < 1e-4, f"ONNX export verification failed: max diff = {max_diff}"
            
            logger.info(f"ONNX export verified (max diff: {max_diff:.6f})")
            
        except ImportError:
            logger.warning("onnxruntime not installed, skipping verification")


class LightGBMEnsemble:
    """Ensemble of LightGBM models for all prediction horizons."""
    
    def __init__(self, config: Optional[LightGBMConfig] = None):
        """
        Initialize ensemble.
        
        Args:
            config: Model configuration
        """
        self.config = config or LightGBMConfig()
        self.models: Dict[int, LightGBMHorizonModel] = {}
        
        for h in range(1, NUM_HORIZONS + 1):
            self.models[h] = LightGBMHorizonModel(h, self.config)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Train all horizon models.
        
        Args:
            X_train: Training sequences (N, seq_len, num_features)
            y_train: Training targets (N, num_horizons)
            X_val: Validation sequences
            y_val: Validation targets
            
        Returns:
            Training history for each horizon
        """
        # Flatten sequences for LightGBM
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        logger.info(f"Training LightGBM ensemble on {X_train_flat.shape[0]} samples, "
                   f"{X_train_flat.shape[1]} features")
        
        histories = {}
        for h in range(1, NUM_HORIZONS + 1):
            history = self.models[h].train(
                X_train_flat,
                y_train[:, h - 1],
                X_val_flat,
                y_val[:, h - 1],
            )
            histories[h] = history
        
        return histories
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for all horizons.
        
        Args:
            X: Sequences (N, seq_len, num_features) or flattened (N, seq_len * num_features)
            
        Returns:
            Predictions (N, num_horizons)
        """
        # Flatten if needed
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        predictions = np.zeros((X.shape[0], NUM_HORIZONS))
        for h in range(1, NUM_HORIZONS + 1):
            predictions[:, h - 1] = self.models[h].predict(X)
        
        return predictions
    
    def save(self, output_dir: str) -> None:
        """Save all models to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for h, model in self.models.items():
            model_path = output_path / f"lgbm_horizon_{h}.txt"
            model.save(str(model_path))
    
    def load(self, model_dir: str) -> None:
        """Load all models from directory."""
        model_path = Path(model_dir)
        
        for h in range(1, NUM_HORIZONS + 1):
            path = model_path / f"lgbm_horizon_{h}.txt"
            if path.exists():
                self.models[h].load(str(path))
            else:
                raise FileNotFoundError(f"Model file not found: {path}")
    
    def export_onnx(self, output_dir: str) -> None:
        """
        Export all models to ONNX format.
        
        Args:
            output_dir: Directory for output .onnx files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for h, model in self.models.items():
            onnx_path = output_path / f"lgbm_horizon_{h}.onnx"
            model.export_onnx(str(onnx_path))
        
        logger.info(f"Exported all ONNX models to {output_dir}")


def create_lgbm_ensemble(config: Optional[LightGBMConfig] = None) -> LightGBMEnsemble:
    """Create a new LightGBM ensemble."""
    return LightGBMEnsemble(config)


def load_lgbm_ensemble(model_dir: str, config: Optional[LightGBMConfig] = None) -> LightGBMEnsemble:
    """Load a trained LightGBM ensemble."""
    ensemble = LightGBMEnsemble(config)
    ensemble.load(model_dir)
    return ensemble
