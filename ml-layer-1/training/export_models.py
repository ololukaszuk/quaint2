"""
Export Models Module

Exports trained models to production formats:
- Mamba → TorchScript (.pt)
- LightGBM → ONNX (.onnx)
- Normalization params → JSON
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .config import TrainingConfig, NUM_FEATURES, SEQUENCE_LENGTH, NUM_HORIZONS
from .mamba_model import MambaModel, load_mamba_model, export_mamba_torchscript
from .lightgbm_models import LightGBMEnsemble, load_lgbm_ensemble

logger = logging.getLogger(__name__)


def export_normalization_params(
    norm_params: Dict[str, Any],
    output_path: str,
    version: int = 1,
) -> None:
    """
    Export normalization parameters to JSON.
    
    Args:
        norm_params: Dict with 'mean' and 'std' arrays
        output_path: Output file path
        version: Version number
    """
    export_data = {
        'version': version,
        'num_features': NUM_FEATURES,
        'sequence_length': SEQUENCE_LENGTH,
        'mean': norm_params['mean'],
        'std': norm_params['std'],
        'feature_names': [
            # Raw features
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", "number_of_trades",
            "spread_bps", "taker_buy_ratio",
            # Derived features
            "log_return_1m", "log_return_5m", "log_return_15m",
            "volatility_5m", "volatility_15m", "volatility_30m",
            "sma_5_norm", "sma_15_norm", "sma_30_norm",
            "ema_5_norm", "ema_15_norm", "ema_30_norm",
            "rsi_14", "volume_sma_ratio", "vwap_deviation", "price_position",
        ],
    }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Exported normalization params to {output_path}")


def export_mamba_model(
    checkpoint_path: str,
    output_path: str,
    config: Optional[TrainingConfig] = None,
) -> None:
    """
    Export Mamba model to TorchScript.
    
    Args:
        checkpoint_path: Path to training checkpoint
        output_path: Output .pt file path
        config: Training configuration
    """
    if config is None:
        config = TrainingConfig()
    
    logger.info(f"Loading Mamba checkpoint from {checkpoint_path}")
    model = load_mamba_model(checkpoint_path, config.mamba)
    
    logger.info(f"Exporting TorchScript to {output_path}")
    export_mamba_torchscript(model, output_path)


def export_lgbm_models(
    model_dir: str,
    output_dir: str,
    config: Optional[TrainingConfig] = None,
) -> None:
    """
    Export LightGBM models to ONNX.
    
    Args:
        model_dir: Directory with trained LightGBM models
        output_dir: Output directory for ONNX files
        config: Training configuration
    """
    if config is None:
        config = TrainingConfig()
    
    logger.info(f"Loading LightGBM ensemble from {model_dir}")
    ensemble = load_lgbm_ensemble(model_dir, config.lgbm)
    
    logger.info(f"Exporting ONNX models to {output_dir}")
    ensemble.export_onnx(output_dir)


def export_all_models(
    mamba_checkpoint: str,
    lgbm_dir: str,
    norm_params: Dict[str, Any],
    output_dir: str,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, str]:
    """
    Export all models for production inference.
    
    Args:
        mamba_checkpoint: Path to Mamba checkpoint
        lgbm_dir: Directory with LightGBM models
        norm_params: Normalization parameters
        output_dir: Output directory
        config: Training configuration
        
    Returns:
        Dict with paths to exported files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exported = {}
    
    # Export Mamba
    mamba_output = str(output_path / "mamba.pt")
    export_mamba_model(mamba_checkpoint, mamba_output, config)
    exported['mamba'] = mamba_output
    
    # Export LightGBM
    export_lgbm_models(lgbm_dir, str(output_path), config)
    for h in range(1, NUM_HORIZONS + 1):
        exported[f'lgbm_horizon_{h}'] = str(output_path / f"lgbm_horizon_{h}.onnx")
    
    # Export normalization params
    norm_output = str(output_path / "normalization_params_v1.json")
    export_normalization_params(norm_params, norm_output)
    exported['normalization_params'] = norm_output
    
    logger.info(f"Exported all models to {output_dir}")
    logger.info(f"Files: {list(exported.keys())}")
    
    return exported


def verify_exports(output_dir: str) -> bool:
    """
    Verify all exported files exist and are valid.
    
    Args:
        output_dir: Directory with exported files
        
    Returns:
        True if all files valid
    """
    output_path = Path(output_dir)
    
    required_files = [
        "mamba.pt",
        "lgbm_horizon_1.onnx",
        "lgbm_horizon_2.onnx",
        "lgbm_horizon_3.onnx",
        "lgbm_horizon_4.onnx",
        "lgbm_horizon_5.onnx",
        "normalization_params_v1.json",
    ]
    
    all_valid = True
    
    for filename in required_files:
        filepath = output_path / filename
        
        if not filepath.exists():
            logger.error(f"Missing file: {filepath}")
            all_valid = False
            continue
        
        # Verify file contents
        try:
            if filename.endswith('.pt'):
                model = torch.jit.load(str(filepath))
                # Test forward pass
                test_input = torch.randn(1, SEQUENCE_LENGTH, NUM_FEATURES)
                output = model(test_input)
                assert output.shape == (1, NUM_HORIZONS), f"Unexpected output shape: {output.shape}"
                logger.info(f"✓ {filename} - valid TorchScript")
                
            elif filename.endswith('.onnx'):
                import onnxruntime as ort
                session = ort.InferenceSession(str(filepath))
                input_name = session.get_inputs()[0].name
                test_input = np.random.randn(1, SEQUENCE_LENGTH * NUM_FEATURES).astype(np.float32)
                output = session.run(None, {input_name: test_input})
                logger.info(f"✓ {filename} - valid ONNX")
                
            elif filename.endswith('.json'):
                with open(filepath) as f:
                    data = json.load(f)
                assert 'mean' in data and 'std' in data
                assert len(data['mean']) == NUM_FEATURES
                assert len(data['std']) == NUM_FEATURES
                logger.info(f"✓ {filename} - valid JSON")
                
        except Exception as e:
            logger.error(f"Invalid file {filename}: {e}")
            all_valid = False
    
    return all_valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export trained models")
    parser.add_argument("--mamba-checkpoint", required=True, help="Mamba checkpoint path")
    parser.add_argument("--lgbm-dir", required=True, help="LightGBM models directory")
    parser.add_argument("--norm-params", required=True, help="Normalization params JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--verify", action="store_true", help="Verify exports after creation")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load norm params
    with open(args.norm_params) as f:
        norm_params = json.load(f)
    
    # Export
    export_all_models(
        args.mamba_checkpoint,
        args.lgbm_dir,
        norm_params,
        args.output_dir,
    )
    
    # Verify
    if args.verify:
        if verify_exports(args.output_dir):
            logger.info("All exports verified successfully!")
        else:
            logger.error("Export verification failed!")
            exit(1)
