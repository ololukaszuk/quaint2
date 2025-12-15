# ML Layer 1 Inference (Rust)

High-performance Rust inference module for real-time Bitcoin price prediction.

## Overview

This crate provides:
- Feature computation (27 features matching Python implementation)
- TorchScript model loading (Mamba SSM)
- ONNX Runtime inference (LightGBM)
- Ensemble prediction combining
- Database integration
- Hot-reload support

## Building

### Without ML (for data-feeder compilation)
```bash
cargo build --release
```

### With ML Inference
```bash
cargo build --release --features ml-inference
```

## Dependencies

### Core (always included)
- `tokio` - Async runtime
- `tokio-postgres` - Database client
- `serde` / `serde_json` - Serialization
- `ndarray` - Numerical computation
- `chrono` - Date/time handling

### ML (optional, via `ml-inference` feature)
- `tch` - PyTorch bindings (TorchScript)
- `ort` - ONNX Runtime bindings

## Usage

### As a Library (recommended)

```rust
use ml_layer_1_inference::{
    initialize,
    run_inference_pipeline,
    check_hot_reload,
};
use std::sync::Arc;

// Initialize on startup
let (config, model_manager) = initialize(
    Some(&db_client),
    None,  // Optional config file path
).await?;

// Run inference
if model_manager.is_ready() {
    let prediction = run_inference_pipeline(
        db_client.clone(),
        &model_manager,
    ).await?;
    
    println!("Predictions: {:?}", prediction.predictions);
    println!("Confidences: {:?}", prediction.confidences);
    println!("Latency: {:.2}ms", prediction.latency_ms);
}

// Periodically check for model updates
check_hot_reload(&db_client, &model_manager).await?;
```

### As a Standalone Binary

```bash
# Set environment variables
export DB_HOST=localhost
export DB_PASSWORD=your_password
export ML_ENABLED=true
export ML_MAMBA_MODEL_PATH=/path/to/mamba.pt
export ML_LGBM_MODELS_DIR=/path/to/models

# Run
cargo run --release --features ml-inference
```

## Module Structure

```
src/
├── lib.rs           # Main entry, run_inference_pipeline()
├── features.rs      # 27 feature computation (CRITICAL)
├── candle_loader.rs # Database queries for candles
├── config.rs        # Configuration loading
├── models.rs        # Model loading + hot-reload
├── ensemble.rs      # Prediction combining
├── db.rs            # Database helpers
└── main.rs          # Standalone binary
```

## Feature Computation

CRITICAL: `features.rs` must compute the exact same 27 features as Python.

```rust
use ml_layer_1_inference::FeatureComputer;

let features = FeatureComputer::compute_extended_features(
    &opens, &highs, &lows, &closes, &volumes,
    &quote_asset_volumes, &taker_buy_base, &taker_buy_quote,
    &number_of_trades, &spread_bps, &taker_buy_ratio,
);

// features: Vec<Vec<f64>> with shape (n, 27)
```

## Configuration Priority

1. **Database** (primary) - Queries `active_ensembles` table
2. **Environment variables** - Fallback
3. **TOML config file** - Last resort
4. **Default** - Disabled

## Environment Variables

```bash
# Enable inference
ML_ENABLED=true

# Model paths
ML_MAMBA_MODEL_PATH=/app/models/mamba.pt
ML_LGBM_MODELS_DIR=/app/models
ML_NORM_PARAMS_PATH=/app/models/normalization_params_v1.json

# Ensemble configuration
ML_MAMBA_WEIGHT=0.5
ML_ENSEMBLE_VERSION=1

# Device
ML_DEVICE=cpu  # or cuda

# Database (same as data-feeder)
DB_HOST=timescaledb
DB_PORT=5432
DB_NAME=btc_ml_production
DB_USER=mltrader
DB_PASSWORD=secret
```

## Hot-Reload

Models are automatically reloaded when the database version changes:

```rust
// Check every 5 minutes
if check_hot_reload(&client, &model_manager).await? {
    println!("Models reloaded!");
}
```

The hot-reload mechanism:
1. Queries `active_ensembles.id` from database
2. Compares with current loaded version
3. If different, reloads config and models
4. Updates internal version counter

## Testing

```bash
# Unit tests
cargo test

# With ML feature
cargo test --features ml-inference

# Integration tests (requires database)
cargo test --features ml-inference -- --ignored
```

## Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| Feature computation | <5ms | Pure Rust, no ML |
| Mamba inference | <20ms | TorchScript |
| LightGBM inference | <10ms | ONNX Runtime |
| Total pipeline | <50ms | Including DB queries |

## Troubleshooting

### "tch" build fails
Install libtorch:
```bash
# Linux
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch*.zip
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

### "ort" build fails
Install ONNX Runtime:
```bash
# Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar xzf onnxruntime*.tgz
export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so
```

### Models not loading
1. Check file paths exist
2. Verify `ML_ENABLED=true`
3. Check database has active ensemble
4. Review logs for specific errors

### Feature mismatch
Compare outputs between Python and Rust:
```python
# Python
features_py = compute_extended_features(df)
print(features_py.iloc[-1].values)
```
```rust
// Rust
let features_rs = FeatureComputer::compute_extended_features(...);
println!("{:?}", features_rs.last());
```

## Integration with data-feeder

Add to `data-feeder/Cargo.toml`:
```toml
[dependencies.ml-layer-1-inference]
path = "../ml-layer-1/inference"
optional = true

[features]
ml-inference = ["ml-layer-1-inference/ml-inference"]
```

Use in data-feeder:
```rust
#[cfg(feature = "ml-inference")]
use ml_layer_1_inference::{initialize, run_inference_pipeline};

#[cfg(feature = "ml-inference")]
async fn run_ml_inference(client: Arc<Client>) {
    let (_, manager) = initialize(Some(&client), None).await.unwrap();
    
    if manager.is_ready() {
        let pred = run_inference_pipeline(client, &manager).await.unwrap();
        // Use prediction...
    }
}
```

## License

MIT
