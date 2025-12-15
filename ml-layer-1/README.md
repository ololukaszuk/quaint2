# ML Layer 1: Price Prediction Module

Real-time Bitcoin price prediction using Mamba SSM + LightGBM ensemble.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ML Layer 1                              │
├─────────────────────────┬───────────────────────────────────┤
│   Training (Python)     │        Inference (Rust)           │
│                         │                                    │
│  • Data loading         │  • Feature computation (27)        │
│  • Feature engineering  │  • TorchScript inference           │
│  • Mamba training       │  • ONNX Runtime inference          │
│  • LightGBM training    │  • Ensemble combining              │
│  • Model evaluation     │  • Hot-reload support              │
│  • Export to artifacts  │  • Database integration            │
└─────────────────────────┴───────────────────────────────────┘
           │                              │
           ▼                              ▼
    ┌─────────────┐               ┌─────────────┐
    │   models/   │               │ data-feeder │
    │ mamba.pt    │◄──────────────│   (uses)    │
    │ lgbm_*.onnx │               └─────────────┘
    │ norm_*.json │
    └─────────────┘
```

## Features

- **27 Features** computed from 11 raw OHLCV fields
- **Mamba SSM**: State Space Model for sequence modeling (TorchScript)
- **LightGBM**: Gradient boosted trees for each horizon (ONNX)
- **5 Horizons**: 1, 2, 3, 4, 5 minute predictions
- **60 Minute Sequences**: Rolling window of recent data
- **Hot-Reload**: Automatic model updates from database
- **A/B Testing**: Multiple ensemble versions with auto-promotion

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Direction Accuracy | 56-58% | TBD |
| Inference Latency | <50ms | TBD |
| Memory Usage | <100MB | TBD |

## Directory Structure

```
ml-layer-1/
├── training/           # Python training code
│   ├── config.py       # Configuration
│   ├── feature_engineering.py  # 27 features (REFERENCE)
│   ├── data_loader.py  # TimescaleDB loading
│   ├── mamba_model.py  # Mamba implementation
│   ├── lightgbm_models.py  # LightGBM ensemble
│   ├── trainer.py      # Training orchestration
│   ├── evaluator.py    # Model evaluation
│   ├── predictor.py    # Python predictor
│   ├── export_models.py  # Export to artifacts
│   └── scripts/
│       └── daily_retrain.py  # 03:00 UTC cron job
│
├── inference/          # Rust inference code
│   ├── src/
│   │   ├── lib.rs      # Main entry + run_inference_pipeline()
│   │   ├── features.rs # 27 features (MUST MATCH PYTHON)
│   │   ├── candle_loader.rs  # Database queries
│   │   ├── config.rs   # Configuration loading
│   │   ├── models.rs   # TorchScript + ONNX loading
│   │   ├── ensemble.rs # Prediction combining
│   │   └── db.rs       # Database helpers
│   ├── Cargo.toml
│   └── README.md
│
└── models/             # Exported artifacts
    ├── mamba.pt        # TorchScript model
    ├── lgbm_horizon_1.onnx
    ├── lgbm_horizon_2.onnx
    ├── lgbm_horizon_3.onnx
    ├── lgbm_horizon_4.onnx
    ├── lgbm_horizon_5.onnx
    └── normalization_params_v1.json
```

## Feature List (27 Total)

### Raw Features (11)
From `candles_1m` table:
1. `open` - Opening price
2. `high` - High price
3. `low` - Low price
4. `close` - Closing price
5. `volume` - Base asset volume
6. `quote_asset_volume` - Quote asset volume
7. `taker_buy_base_asset_volume` - Taker buy volume
8. `taker_buy_quote_asset_volume` - Taker buy quote volume
9. `number_of_trades` - Number of trades
10. `spread_bps` - Spread in basis points
11. `taker_buy_ratio` - Ratio of taker buys

### Derived Features (16)
Computed from raw:
12. `log_return_1m` - ln(close[t] / close[t-1])
13. `log_return_5m` - ln(close[t] / close[t-5])
14. `log_return_15m` - ln(close[t] / close[t-15])
15. `volatility_5m` - Rolling std of 1m returns
16. `volatility_15m` - Rolling std of 1m returns
17. `volatility_30m` - Rolling std of 1m returns
18. `sma_5_norm` - (close - SMA5) / close
19. `sma_15_norm` - (close - SMA15) / close
20. `sma_30_norm` - (close - SMA30) / close
21. `ema_5_norm` - (close - EMA5) / close
22. `ema_15_norm` - (close - EMA15) / close
23. `ema_30_norm` - (close - EMA30) / close
24. `rsi_14` - Relative Strength Index (normalized 0-1)
25. `volume_sma_ratio` - volume / SMA20(volume)
26. `vwap_deviation` - (close - VWAP) / close
27. `price_position` - (close - low) / (high - low)

## Quick Start

### Training (Python)

```bash
cd ml-layer-1/training

# Install dependencies
pip install -r requirements.txt

# Run daily retraining
python scripts/daily_retrain.py
```

### Inference (Rust)

```bash
cd ml-layer-1/inference

# Build with ML support
cargo build --release --features ml-inference

# Run standalone (for testing)
cargo run --release --features ml-inference
```

### Integration with data-feeder

Add to `data-feeder/Cargo.toml`:
```toml
[dependencies]
ml-layer-1-inference = { path = "../ml-layer-1/inference", features = ["ml-inference"] }
```

Use in code:
```rust
use ml_layer_1_inference::{initialize, run_inference_pipeline};

// Initialize on startup
let (config, model_manager) = initialize(Some(&db_client), None).await?;

// Run inference every minute
let prediction = run_inference_pipeline(db_client.clone(), &model_manager).await?;
```

## Database Schema

### Required Tables

```sql
-- Active ensemble configuration
CREATE TABLE active_ensembles (
    id SERIAL PRIMARY KEY,
    mamba_version_id INTEGER,
    lgbm_version_id INTEGER,
    mamba_weight NUMERIC(5,4) DEFAULT 0.5,
    is_active BOOLEAN DEFAULT false,
    is_test BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model versions
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(20), -- 'mamba' or 'lgbm'
    version INTEGER,
    model_path TEXT,
    accuracy_1m NUMERIC(5,4),
    rmse NUMERIC(20,8),
    trained_at TIMESTAMPTZ
);
```

### Predictions Table

Already exists in `init.sql`. Predictions are stored with:
- `model_name` = 'mamba_lgbm_ensemble'
- `horizon` = 1-5 minutes
- `predicted_close` = log return prediction
- `confidence` = model agreement score

## Configuration

### Environment Variables

```bash
# Enable ML inference
ML_ENABLED=true

# Model paths
ML_MAMBA_MODEL_PATH=/app/models/mamba.pt
ML_LGBM_MODELS_DIR=/app/models
ML_NORM_PARAMS_PATH=/app/models/normalization_params_v1.json

# Ensemble config
ML_MAMBA_WEIGHT=0.5

# Device
ML_DEVICE=cpu  # or cuda
```

### Database Configuration

Primary source - queries `active_ensembles` table for current configuration.

## Daily Retraining (03:00 UTC)

Scheduled via cron or Kubernetes CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: ml-daily-retrain
spec:
  schedule: "0 3 * * *"
  jobTemplate:
    spec:
      containers:
      - name: trainer
        image: ml-layer-1-training:latest
        command: ["python", "scripts/daily_retrain.py"]
```

## A/B Testing

1. New model trained and exported
2. Registered as `is_test=true` in database
3. Runs alongside production model
4. After 24h, accuracy compared
5. Auto-promoted if accuracy > production + 0.5%

## Monitoring

### Health Endpoints

```bash
# From data-feeder
curl http://localhost:8080/health

# Response includes:
# - ml_inference_enabled: true/false
# - ml_model_version: current ensemble ID
# - ml_last_prediction: timestamp
```

### Metrics (Prometheus)

```
ml_inference_latency_ms
ml_inference_accuracy_24h
ml_model_version_current
ml_predictions_total
```

## Troubleshooting

### Inference disabled
Check `ML_ENABLED=true` and models exist in `models/` directory.

### Low accuracy
- Verify features match between Python and Rust
- Check normalization params are up to date
- Review recent candle data quality

### High latency
- Check CPU usage during inference
- Consider reducing sequence length
- Enable batch inference if multiple predictions needed

## Development

### Feature Parity Testing

CRITICAL: Rust features.rs must produce identical output to Python feature_engineering.py.

```bash
# Test feature computation
cd ml-layer-1/inference
cargo test --features ml-inference
```

### Adding New Features

1. Add to Python `feature_engineering.py`
2. Add to Rust `features.rs`
3. Update `NUM_FEATURES` constant
4. Retrain models with new features
5. Update normalization params

## License

MIT
