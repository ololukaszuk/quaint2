"""
ML Layer 1 Configuration Module

Central configuration for all training parameters, feature definitions,
and model hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# =============================================================================
# FEATURE DEFINITIONS - CRITICAL: Must match Rust features.rs exactly
# =============================================================================

# 11 Raw features from candles_1m table
RAW_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "number_of_trades",
    "spread_bps",
    "taker_buy_ratio",
]

# 16 Derived features computed from raw
DERIVED_FEATURES = [
    "log_return_1m",        # ln(close[t] / close[t-1])
    "log_return_5m",        # ln(close[t] / close[t-5])
    "log_return_15m",       # ln(close[t] / close[t-15])
    "volatility_5m",        # std(log_returns, 5)
    "volatility_15m",       # std(log_returns, 15)
    "volatility_30m",       # std(log_returns, 30)
    "sma_5_norm",           # (close - SMA5) / close
    "sma_15_norm",          # (close - SMA15) / close
    "sma_30_norm",          # (close - SMA30) / close
    "ema_5_norm",           # (close - EMA5) / close
    "ema_15_norm",          # (close - EMA15) / close
    "ema_30_norm",          # (close - EMA30) / close
    "rsi_14",               # RSI(14) / 100 (normalized 0-1)
    "volume_sma_ratio",     # volume / SMA(volume, 20)
    "vwap_deviation",       # (close - VWAP) / close
    "price_position",       # (close - low) / (high - low)
]

# All 27 features in order
ALL_FEATURES = RAW_FEATURES + DERIVED_FEATURES
NUM_FEATURES = len(ALL_FEATURES)  # 27

# Prediction horizons (minutes)
PREDICTION_HORIZONS = [1, 2, 3, 4, 5]
NUM_HORIZONS = len(PREDICTION_HORIZONS)

# Sequence length for model input
SEQUENCE_LENGTH = 60


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "timescaledb"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "btc_ml_production"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "mltrader"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class MambaConfig:
    """Mamba model hyperparameters."""
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 4
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Input/Output
    input_size: int = NUM_FEATURES  # 27
    sequence_length: int = SEQUENCE_LENGTH  # 60
    output_size: int = NUM_HORIZONS  # 5


@dataclass
class LightGBMConfig:
    """LightGBM model hyperparameters."""
    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    
    # Training
    early_stopping_rounds: int = 50
    verbose: int = -1
    
    # Input shape: flattened sequence
    input_size: int = SEQUENCE_LENGTH * NUM_FEATURES  # 60 * 27 = 1620


@dataclass
class EnsembleConfig:
    """Ensemble configuration."""
    mamba_weight: float = field(default_factory=lambda: float(os.getenv("ML_MAMBA_WEIGHT", "0.5")))
    lgbm_weight: float = field(default=None)
    
    # A/B Testing
    ab_test_hours: int = field(default_factory=lambda: int(os.getenv("ML_AB_TEST_HOURS", "4")))
    min_accuracy_improvement: float = field(default_factory=lambda: float(os.getenv("ML_MIN_ACCURACY_IMPROVEMENT", "0.02")))
    
    def __post_init__(self):
        if self.lgbm_weight is None:
            self.lgbm_weight = 1.0 - self.mamba_weight


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Sub-configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    lgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # Paths
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("ML_MODELS_DIR", "/app/models")))
    
    # Data
    data_months: int = field(default_factory=lambda: int(os.getenv("ML_DATA_MONTHS", "13")))
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    
    # Device
    device: str = field(default_factory=lambda: os.getenv("ML_DEVICE", "cpu"))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


def get_config() -> TrainingConfig:
    """Get training configuration from environment."""
    return TrainingConfig()


# Export for convenience
Config = TrainingConfig
