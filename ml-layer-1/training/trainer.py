"""
Trainer Module

Handles training orchestration for Mamba and LightGBM models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import time

from .config import TrainingConfig, MambaConfig, LightGBMConfig
from .mamba_model import MambaModel, create_mamba_model
from .lightgbm_models import LightGBMEnsemble, create_lgbm_ensemble

logger = logging.getLogger(__name__)


class MambaTrainer:
    """Trainer for Mamba model."""
    
    def __init__(self, config: MambaConfig, device: str = "cpu"):
        """
        Initialize trainer.
        
        Args:
            config: Model configuration
            device: Training device (cpu/cuda)
        """
        self.config = config
        self.device = torch.device(device)
        self.model = create_mamba_model(config).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )
        
        self.criterion = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler() if 'cuda' in str(self.device) else None
    
    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: {'sequences': (N, 60, 27), 'targets': (N, 5)}
            val_data: Same format as train_data
            
        Returns:
            Training history
        """
        # Create data loaders
        train_loader = self._create_dataloader(train_data, shuffle=True)
        val_loader = self._create_dataloader(val_data, shuffle=False)
        
        logger.info(f"Training Mamba model on {self.device}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': [],
        }
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(current_lr)
            history['epoch_times'].append(epoch_time)
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, "
                f"lr: {current_lr:.6f}, time: {epoch_time:.1f}s"
            )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        history['best_val_loss'] = best_val_loss
        history['epochs_trained'] = len(history['train_loss'])
        
        return history
    
    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for sequences, targets in dataloader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for sequences, targets in dataloader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _create_dataloader(self, data: Dict[str, np.ndarray], shuffle: bool) -> DataLoader:
        """Create PyTorch DataLoader from numpy data."""
        sequences = torch.FloatTensor(data['sequences'])
        targets = torch.FloatTensor(data['targets'])
        
        dataset = TensorDataset(sequences, targets)
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if 'cuda' in str(self.device) else False,
        )
    
    def save_checkpoint(self, path: str, history: Dict[str, Any] = None) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'history': history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('history', {})
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(sequences).to(self.device)
            predictions = self.model(x).cpu().numpy()
        
        return predictions


class LightGBMTrainer:
    """Trainer for LightGBM ensemble."""
    
    def __init__(self, config: LightGBMConfig):
        """
        Initialize trainer.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.ensemble = create_lgbm_ensemble(config)
    
    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Train the ensemble.
        
        Args:
            train_data: {'sequences': (N, 60, 27), 'targets': (N, 5)}
            val_data: Same format
            
        Returns:
            Training history for each horizon
        """
        return self.ensemble.train(
            train_data['sequences'],
            train_data['targets'],
            val_data['sequences'],
            val_data['targets'],
        )
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.ensemble.predict(sequences)
    
    def save(self, output_dir: str) -> None:
        """Save ensemble."""
        self.ensemble.save(output_dir)
    
    def load(self, model_dir: str) -> None:
        """Load ensemble."""
        self.ensemble.load(model_dir)
    
    def export_onnx(self, output_dir: str) -> None:
        """Export to ONNX."""
        self.ensemble.export_onnx(output_dir)


class EnsembleTrainer:
    """Combined trainer for Mamba + LightGBM ensemble."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainers.
        
        Args:
            config: Complete training configuration
        """
        self.config = config
        self.mamba_trainer = MambaTrainer(config.mamba, config.device)
        self.lgbm_trainer = LightGBMTrainer(config.lgbm)
    
    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Train both models.
        
        Args:
            train_data: {'sequences': (N, 60, 27), 'targets': (N, 5)}
            val_data: Same format
            
        Returns:
            Combined training history
        """
        logger.info("=" * 60)
        logger.info("Training Mamba Model")
        logger.info("=" * 60)
        mamba_history = self.mamba_trainer.train(train_data, val_data)
        
        logger.info("=" * 60)
        logger.info("Training LightGBM Ensemble")
        logger.info("=" * 60)
        lgbm_history = self.lgbm_trainer.train(train_data, val_data)
        
        return {
            'mamba': mamba_history,
            'lgbm': lgbm_history,
        }
    
    def predict(
        self,
        sequences: np.ndarray,
        mamba_weight: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate ensemble predictions.
        
        Args:
            sequences: Input sequences (N, 60, 27)
            mamba_weight: Weight for Mamba (default from config)
            
        Returns:
            (ensemble_predictions, mamba_predictions, lgbm_predictions)
        """
        if mamba_weight is None:
            mamba_weight = self.config.ensemble.mamba_weight
        
        lgbm_weight = 1.0 - mamba_weight
        
        mamba_preds = self.mamba_trainer.predict(sequences)
        lgbm_preds = self.lgbm_trainer.predict(sequences)
        
        ensemble_preds = mamba_weight * mamba_preds + lgbm_weight * lgbm_preds
        
        return ensemble_preds, mamba_preds, lgbm_preds
    
    def save(self, output_dir: str, history: Dict[str, Any] = None) -> None:
        """Save all models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save Mamba
        mamba_path = output_path / "mamba_checkpoint.pt"
        self.mamba_trainer.save_checkpoint(str(mamba_path), history.get('mamba') if history else None)
        
        # Save LightGBM
        lgbm_path = output_path / "lgbm"
        self.lgbm_trainer.save(str(lgbm_path))
        
        logger.info(f"Saved all models to {output_dir}")
    
    def export(self, output_dir: str) -> None:
        """Export models for inference."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export Mamba TorchScript
        from .mamba_model import export_mamba_torchscript
        mamba_pt_path = output_path / "mamba.pt"
        export_mamba_torchscript(self.mamba_trainer.model, str(mamba_pt_path))
        
        # Export LightGBM ONNX
        lgbm_onnx_path = output_path
        self.lgbm_trainer.export_onnx(str(lgbm_onnx_path))
        
        logger.info(f"Exported all models to {output_dir}")
