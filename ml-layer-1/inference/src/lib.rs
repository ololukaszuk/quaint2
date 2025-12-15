//! ML Layer 1 Inference Module
//!
//! Real-time ML inference for Bitcoin price prediction using
//! Mamba (TorchScript) + LightGBM (ONNX) ensemble.
//!
//! ## Features
//! - 27 features computed from candles_1m (matches Python training)
//! - TorchScript inference for Mamba SSM
//! - ONNX Runtime inference for LightGBM
//! - Hot-reload on database version changes
//! - Async/await compatible
//!
//! ## Usage
//! ```rust,ignore
//! use ml_layer_1_inference::{run_inference_pipeline, InferenceConfig};
//!
//! let config = InferenceConfig::default();
//! let result = run_inference_pipeline(&db_client, &config).await?;
//! ```

pub mod candle_loader;
pub mod config;
pub mod db;
pub mod ensemble;
pub mod features;
pub mod models;

pub use candle_loader::{Candle, CandleLoader};
pub use config::{ConfigLoader, DatabaseConfig, InferenceConfig};
pub use db::{AccuracyStats, ActiveEnsemble, InferenceDb};
pub use ensemble::{EnsembleMetrics, EnsemblePrediction, HORIZON_MINUTES};
pub use features::{FeatureComputer, NormalizationParams, NUM_FEATURES, NUM_HORIZONS, SEQUENCE_LENGTH};
pub use models::{Model, ModelError, ModelManager, ModelPrediction};

use std::sync::Arc;
use thiserror::Error;
use tokio_postgres::Client;
use tracing::{debug, error, info, warn};

/// Inference pipeline errors
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("Database error: {0}")]
    Database(#[from] tokio_postgres::Error),

    #[error("Candle loader error: {0}")]
    CandleLoader(#[from] candle_loader::CandleLoaderError),

    #[error("Config error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Not enough candles: need {needed}, got {got}")]
    NotEnoughCandles { needed: usize, got: usize },

    #[error("Inference disabled")]
    Disabled,
}

/// Result type for inference operations
pub type Result<T> = std::result::Result<T, InferenceError>;

/// Run the complete inference pipeline
///
/// Steps:
/// 1. Load latest 60 candles from database
/// 2. Compute 27 features
/// 3. Normalize using saved parameters
/// 4. Run Mamba + LightGBM inference
/// 5. Combine predictions with ensemble weights
/// 6. Store predictions in database
///
/// # Arguments
/// * `client` - Database client
/// * `model_manager` - Loaded models
///
/// # Returns
/// * `Ok(EnsemblePrediction)` - Combined prediction
/// * `Err(InferenceError)` - If any step fails
pub async fn run_inference_pipeline(
    client: Arc<Client>,
    model_manager: &ModelManager,
) -> Result<EnsemblePrediction> {
    let start = std::time::Instant::now();

    // Check if models are ready
    if !model_manager.is_ready() {
        return Err(InferenceError::Disabled);
    }

    // Step 1: Load latest candles
    let candle_loader = CandleLoader::new(client.clone());
    let candles = candle_loader.fetch_latest_candles(SEQUENCE_LENGTH).await?;

    if candles.len() < SEQUENCE_LENGTH {
        return Err(InferenceError::NotEnoughCandles {
            needed: SEQUENCE_LENGTH,
            got: candles.len(),
        });
    }

    // Step 2: Extract arrays from candles
    let arrays = candle_loader::extract_arrays(&candles);

    // Step 3: Compute 27 features
    let raw_features = FeatureComputer::compute_extended_features(
        &arrays.opens,
        &arrays.highs,
        &arrays.lows,
        &arrays.closes,
        &arrays.volumes,
        &arrays.quote_asset_volumes,
        &arrays.taker_buy_base_asset_volumes,
        &arrays.taker_buy_quote_asset_volumes,
        &arrays.number_of_trades,
        &arrays.spread_bps,
        &arrays.taker_buy_ratio,
    );

    // Step 4: Normalize
    let normalized = match model_manager.get_norm_params() {
        Some(params) => params.normalize(&raw_features),
        None => raw_features, // Use raw if no params
    };

    // Step 5: Run inference
    let mamba_pred = model_manager.predict_mamba(&normalized)?;
    let lgbm_pred = model_manager.predict_lgbm(&normalized)?;

    // Step 6: Combine predictions
    let ensemble = EnsemblePrediction::combine(
        &mamba_pred,
        &lgbm_pred,
        model_manager.get_mamba_weight(),
        model_manager.current_version(),
    );

    // Step 7: Store predictions
    let db = InferenceDb::new(client.clone());
    if let Err(e) = db.insert_prediction(&ensemble).await {
        error!("Failed to store prediction: {}", e);
    }

    let total_latency = start.elapsed().as_secs_f64() * 1000.0;
    debug!(
        "Inference pipeline completed in {:.2}ms (mamba: {:.2}ms, lgbm: {:.2}ms)",
        total_latency, mamba_pred.latency_ms, lgbm_pred.latency_ms
    );

    Ok(ensemble)
}

/// Initialize the inference system
///
/// # Arguments
/// * `client` - Database client
/// * `config_file` - Optional path to config file
///
/// # Returns
/// * `Ok((InferenceConfig, ModelManager))` - Loaded config and models
pub async fn initialize(
    client: Option<&Client>,
    config_file: Option<&str>,
) -> Result<(InferenceConfig, ModelManager)> {
    // Load configuration
    let config = ConfigLoader::load(client, config_file).await?;

    if !config.enabled {
        info!("ML inference is disabled");
        return Ok((config.clone(), ModelManager::new(config)));
    }

    // Create model manager and load models
    let manager = ModelManager::new(config.clone());
    
    if let Err(e) = manager.load_models() {
        error!("Failed to load models: {}", e);
        // Continue with disabled inference
    }

    info!(
        "ML inference initialized: enabled={}, mamba_weight={}",
        manager.is_ready(),
        config.mamba_weight
    );

    Ok((config, manager))
}

/// Hot-reload check - call periodically to check for model updates
///
/// # Arguments
/// * `client` - Database client
/// * `model_manager` - Model manager to reload
///
/// # Returns
/// * `Ok(true)` - Models were reloaded
/// * `Ok(false)` - No changes
pub async fn check_hot_reload(
    client: &Client,
    model_manager: &ModelManager,
) -> Result<bool> {
    match model_manager.check_and_reload(client).await {
        Ok(reloaded) => {
            if reloaded {
                info!("Models hot-reloaded successfully");
            }
            Ok(reloaded)
        }
        Err(e) => {
            warn!("Hot-reload check failed: {}", e);
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count() {
        assert_eq!(NUM_FEATURES, 27);
        assert_eq!(SEQUENCE_LENGTH, 60);
        assert_eq!(NUM_HORIZONS, 5);
    }

    #[test]
    fn test_horizon_minutes() {
        assert_eq!(HORIZON_MINUTES, [1, 2, 3, 4, 5]);
    }
}
