#!/usr/bin/env python3
"""
Daily Retrain Script

Runs daily at 03:00 UTC to retrain models with latest data.

Workflow:
1. Load 13 months of data from TimescaleDB
2. Compute 27 features
3. Train Mamba + LightGBM models
4. Evaluate on test set
5. Export to TorchScript + ONNX
6. Register in database
7. Start A/B test or auto-promote
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import json
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import get_config, TrainingConfig
from training.data_loader import load_training_data
from training.trainer import EnsembleTrainer
from training.evaluator import ModelEvaluator, evaluate_ensemble, find_optimal_weights
from training.export_models import export_all_models, verify_exports

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


async def register_model_version(
    config: TrainingConfig,
    model_type: str,
    version: int,
    metrics: Dict[str, float],
    model_path: str,
) -> int:
    """
    Register a trained model version in the database.
    
    Args:
        config: Training configuration
        model_type: 'mamba' or 'lgbm'
        version: Version number
        metrics: Evaluation metrics
        model_path: Path to model file
        
    Returns:
        Model version ID
    """
    import asyncpg
    
    conn = await asyncpg.connect(
        host=config.database.host,
        port=config.database.port,
        database=config.database.name,
        user=config.database.user,
        password=config.database.password,
    )
    
    try:
        row = await conn.fetchrow("""
            INSERT INTO model_versions (
                model_type, version, model_path,
                accuracy_1m, accuracy_2m, accuracy_3m, accuracy_4m, accuracy_5m,
                rmse, mae, trained_at, is_active
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, false)
            RETURNING id
        """,
            model_type,
            version,
            model_path,
            metrics.get('accuracy_1m', 0),
            metrics.get('accuracy_2m', 0),
            metrics.get('accuracy_3m', 0),
            metrics.get('accuracy_4m', 0),
            metrics.get('accuracy_5m', 0),
            metrics.get('rmse', 0),
            metrics.get('mae', 0),
            datetime.utcnow(),
        )
        return row['id']
    finally:
        await conn.close()


async def register_ensemble(
    config: TrainingConfig,
    mamba_version_id: int,
    lgbm_version_id: int,
    mamba_weight: float,
    metrics: Dict[str, float],
) -> int:
    """
    Register a new ensemble configuration.
    
    Args:
        config: Training configuration
        mamba_version_id: Mamba model version ID
        lgbm_version_id: LightGBM model version ID
        mamba_weight: Weight for Mamba in ensemble
        metrics: Ensemble metrics
        
    Returns:
        Ensemble version ID
    """
    import asyncpg
    
    conn = await asyncpg.connect(
        host=config.database.host,
        port=config.database.port,
        database=config.database.name,
        user=config.database.user,
        password=config.database.password,
    )
    
    try:
        row = await conn.fetchrow("""
            INSERT INTO active_ensembles (
                mamba_version_id, lgbm_version_id, mamba_weight,
                accuracy_1m, accuracy_5m, rmse,
                is_active, is_test, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, false, true, $7)
            RETURNING id
        """,
            mamba_version_id,
            lgbm_version_id,
            mamba_weight,
            metrics.get('accuracy_1m', 0),
            metrics.get('accuracy_5m', 0),
            metrics.get('rmse', 0),
            datetime.utcnow(),
        )
        return row['id']
    finally:
        await conn.close()


async def get_next_version(config: TrainingConfig, model_type: str) -> int:
    """Get next version number for a model type."""
    import asyncpg
    
    conn = await asyncpg.connect(
        host=config.database.host,
        port=config.database.port,
        database=config.database.name,
        user=config.database.user,
        password=config.database.password,
    )
    
    try:
        row = await conn.fetchrow(
            "SELECT COALESCE(MAX(version), 0) + 1 as next_version FROM model_versions WHERE model_type = $1",
            model_type,
        )
        return row['next_version']
    finally:
        await conn.close()


async def run_daily_retrain(config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
    """
    Run the complete daily retraining pipeline.
    
    Args:
        config: Training configuration (loads from env if None)
        
    Returns:
        Training results summary
    """
    if config is None:
        config = get_config()
    
    results = {
        'started_at': datetime.utcnow().isoformat(),
        'status': 'running',
        'errors': [],
    }
    
    try:
        # Step 1: Load data
        logger.info("=" * 60)
        logger.info("Step 1: Loading training data")
        logger.info("=" * 60)
        
        train_data, val_data, test_data, norm_params = await load_training_data(
            config.database,
            months=config.data_months,
        )
        
        results['data'] = {
            'train_samples': len(train_data['sequences']),
            'val_samples': len(val_data['sequences']),
            'test_samples': len(test_data['sequences']),
        }
        
        # Step 2: Train models
        logger.info("=" * 60)
        logger.info("Step 2: Training models")
        logger.info("=" * 60)
        
        trainer = EnsembleTrainer(config)
        training_history = trainer.train(train_data, val_data)
        
        results['training'] = {
            'mamba_epochs': training_history['mamba']['epochs_trained'],
            'mamba_best_val_loss': training_history['mamba']['best_val_loss'],
            'lgbm_horizons_trained': len(training_history['lgbm']),
        }
        
        # Step 3: Evaluate on test set
        logger.info("=" * 60)
        logger.info("Step 3: Evaluating models")
        logger.info("=" * 60)
        
        ensemble_preds, mamba_preds, lgbm_preds = trainer.predict(test_data['sequences'])
        
        ensemble_results, summary = evaluate_ensemble(
            mamba_preds,
            lgbm_preds,
            test_data['targets'],
            mamba_weight=config.ensemble.mamba_weight,
        )
        
        results['evaluation'] = summary
        
        # Find optimal weights
        optimal_weight, optimal_acc = find_optimal_weights(
            mamba_preds,
            lgbm_preds,
            test_data['targets'],
        )
        
        results['optimal_weight'] = optimal_weight
        results['optimal_accuracy'] = optimal_acc
        
        # Step 4: Save models
        logger.info("=" * 60)
        logger.info("Step 4: Saving models")
        logger.info("=" * 60)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = config.models_dir / f"checkpoints_{timestamp}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.save(str(checkpoint_dir), training_history)
        
        # Save normalization params
        norm_params_path = checkpoint_dir / "norm_params.json"
        with open(norm_params_path, 'w') as f:
            json.dump(norm_params, f, indent=2)
        
        # Step 5: Export for inference
        logger.info("=" * 60)
        logger.info("Step 5: Exporting models")
        logger.info("=" * 60)
        
        export_dir = config.models_dir
        exported = export_all_models(
            mamba_checkpoint=str(checkpoint_dir / "mamba_checkpoint.pt"),
            lgbm_dir=str(checkpoint_dir / "lgbm"),
            norm_params=norm_params,
            output_dir=str(export_dir),
            config=config,
        )
        
        results['exported_files'] = list(exported.keys())
        
        # Verify exports
        if not verify_exports(str(export_dir)):
            raise RuntimeError("Export verification failed")
        
        # Step 6: Register in database
        logger.info("=" * 60)
        logger.info("Step 6: Registering models in database")
        logger.info("=" * 60)
        
        mamba_version = await get_next_version(config, 'mamba')
        lgbm_version = await get_next_version(config, 'lgbm')
        
        mamba_metrics = {
            'accuracy_1m': summary['mamba']['direction_accuracy'],
            'rmse': summary['mamba']['rmse'],
            'mae': summary['mamba']['mae'],
        }
        
        lgbm_metrics = {
            'accuracy_1m': summary['lgbm']['direction_accuracy'],
            'rmse': summary['lgbm']['rmse'],
            'mae': summary['lgbm']['mae'],
        }
        
        mamba_version_id = await register_model_version(
            config, 'mamba', mamba_version, mamba_metrics, str(export_dir / "mamba.pt")
        )
        
        lgbm_version_id = await register_model_version(
            config, 'lgbm', lgbm_version, lgbm_metrics, str(export_dir)
        )
        
        ensemble_metrics = {
            'accuracy_1m': summary['ensemble']['direction_accuracy'],
            'accuracy_5m': summary['ensemble']['direction_accuracy'],  # Simplified
            'rmse': summary['ensemble']['rmse'],
        }
        
        ensemble_id = await register_ensemble(
            config,
            mamba_version_id,
            lgbm_version_id,
            config.ensemble.mamba_weight,
            ensemble_metrics,
        )
        
        results['registered'] = {
            'mamba_version_id': mamba_version_id,
            'lgbm_version_id': lgbm_version_id,
            'ensemble_id': ensemble_id,
        }
        
        results['status'] = 'completed'
        results['completed_at'] = datetime.utcnow().isoformat()
        
        logger.info("=" * 60)
        logger.info("Daily retrain completed successfully!")
        logger.info(f"Ensemble accuracy: {summary['ensemble']['direction_accuracy']:.4f}")
        logger.info("=" * 60)
        
    except Exception as e:
        results['status'] = 'failed'
        results['errors'].append(str(e))
        results['traceback'] = traceback.format_exc()
        logger.error(f"Daily retrain failed: {e}")
        logger.error(traceback.format_exc())
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily model retraining")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output", help="Path to save results JSON")
    
    args = parser.parse_args()
    
    # Run async pipeline
    results = asyncio.run(run_daily_retrain())
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Exit with appropriate code
    if results['status'] == 'completed':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
