"""
ML Layer 1 Training Module

Python training infrastructure for Mamba + LightGBM ensemble.
"""

from .config import (
    TrainingConfig,
    MambaConfig,
    LightGBMConfig,
    EnsembleConfig,
    DatabaseConfig,
    get_config,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
    NUM_HORIZONS,
    RAW_FEATURES,
    DERIVED_FEATURES,
    ALL_FEATURES,
)

from .feature_engineering import (
    compute_extended_features,
    normalize_features,
    create_sequences,
    compute_targets,
)

from .data_loader import DataLoader, load_training_data

from .mamba_model import (
    MambaModel,
    MambaBlock,
    create_mamba_model,
    load_mamba_model,
    export_mamba_torchscript,
)

from .lightgbm_models import (
    LightGBMHorizonModel,
    LightGBMEnsemble,
    create_lgbm_ensemble,
    load_lgbm_ensemble,
)

from .trainer import (
    MambaTrainer,
    LightGBMTrainer,
    EnsembleTrainer,
)

from .evaluator import (
    ModelEvaluator,
    EvaluationMetrics,
    compute_metrics,
    compute_horizon_metrics,
    evaluate_ensemble,
    find_optimal_weights,
)

from .predictor import Predictor, PredictionResult, create_predictor

from .export_models import (
    export_all_models,
    export_mamba_model,
    export_lgbm_models,
    export_normalization_params,
    verify_exports,
)

__all__ = [
    # Config
    'TrainingConfig',
    'MambaConfig',
    'LightGBMConfig',
    'EnsembleConfig',
    'DatabaseConfig',
    'get_config',
    'NUM_FEATURES',
    'SEQUENCE_LENGTH',
    'NUM_HORIZONS',
    
    # Feature engineering
    'compute_extended_features',
    'normalize_features',
    'create_sequences',
    'compute_targets',
    
    # Data loading
    'DataLoader',
    'load_training_data',
    
    # Models
    'MambaModel',
    'MambaBlock',
    'create_mamba_model',
    'load_mamba_model',
    'LightGBMHorizonModel',
    'LightGBMEnsemble',
    'create_lgbm_ensemble',
    'load_lgbm_ensemble',
    
    # Training
    'MambaTrainer',
    'LightGBMTrainer',
    'EnsembleTrainer',
    
    # Evaluation
    'ModelEvaluator',
    'EvaluationMetrics',
    'compute_metrics',
    'evaluate_ensemble',
    'find_optimal_weights',
    
    # Prediction
    'Predictor',
    'PredictionResult',
    'create_predictor',
    
    # Export
    'export_all_models',
    'verify_exports',
]
