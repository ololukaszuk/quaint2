//! Model Loading Module
//!
//! Handles loading TorchScript (Mamba) and ONNX (LightGBM) models
//! with hot-reload support based on database version changes.

use parking_lot::RwLock;
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tokio_postgres::Client;
use tracing::{debug, error, info, warn};

use crate::config::InferenceConfig;
use crate::features::{NormalizationParams, NUM_FEATURES, NUM_HORIZONS, SEQUENCE_LENGTH};

/// Model loading errors
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model file not found: {0}")]
    FileNotFound(String),

    #[error("Model load error: {0}")]
    LoadError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Invalid input shape: expected {expected}, got {got}")]
    InvalidShape { expected: String, got: String },

    #[error("Database error: {0}")]
    Database(#[from] tokio_postgres::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Prediction output from a single model
#[derive(Clone, Debug)]
pub struct ModelPrediction {
    /// Predictions for each horizon (5 values)
    pub predictions: Vec<f64>,
    /// Latency in milliseconds
    pub latency_ms: f64,
}

/// Trait for ML models
pub trait Model: Send + Sync {
    /// Run inference on input sequence
    fn predict(&self, input: &[Vec<f64>]) -> Result<ModelPrediction, ModelError>;

    /// Check if model is loaded
    fn is_loaded(&self) -> bool;

    /// Get model name
    fn name(&self) -> &str;
}

/// Mamba model wrapper (TorchScript)
/// Conditional compilation: only available with `ml-inference` feature
#[cfg(feature = "ml-inference")]
pub struct MambaModel {
    model: tch::CModule,
    device: tch::Device,
}

#[cfg(feature = "ml-inference")]
impl MambaModel {
    /// Load from TorchScript file
    pub fn load(path: &str, device: &str) -> Result<Self, ModelError> {
        if !Path::new(path).exists() {
            return Err(ModelError::FileNotFound(path.to_string()));
        }

        let device = match device {
            "cuda" | "gpu" => tch::Device::Cuda(0),
            _ => tch::Device::Cpu,
        };

        let model = tch::CModule::load_on_device(path, device)
            .map_err(|e| ModelError::LoadError(e.to_string()))?;

        info!("Loaded Mamba model from {} on {:?}", path, device);

        Ok(Self { model, device })
    }
}

#[cfg(feature = "ml-inference")]
impl Model for MambaModel {
    fn predict(&self, input: &[Vec<f64>]) -> Result<ModelPrediction, ModelError> {
        let start = std::time::Instant::now();

        // Convert input to tensor: (1, seq_len, num_features)
        let seq_len = input.len();
        let num_features = if seq_len > 0 { input[0].len() } else { 0 };

        // Flatten and create tensor
        let flat: Vec<f32> = input
            .iter()
            .flat_map(|row| row.iter().map(|&x| x as f32))
            .collect();

        let tensor = tch::Tensor::of_slice(&flat)
            .view([1, seq_len as i64, num_features as i64])
            .to_device(self.device);

        // Run inference
        let output = self
            .model
            .forward_ts(&[tensor])
            .map_err(|e| ModelError::InferenceError(e.to_string()))?;

        // Extract predictions
        let predictions: Vec<f64> = Vec::<f32>::from(&output)
            .iter()
            .map(|&x| x as f64)
            .collect();

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(ModelPrediction {
            predictions,
            latency_ms,
        })
    }

    fn is_loaded(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "mamba"
    }
}

/// LightGBM model wrapper (ONNX Runtime)
#[cfg(feature = "ml-inference")]
pub struct LightGBMModel {
    sessions: Vec<ort::Session>,
    input_size: usize,
}

#[cfg(feature = "ml-inference")]
impl LightGBMModel {
    /// Load all horizon models from ONNX files
    pub fn load(models_dir: &str) -> Result<Self, ModelError> {
        let input_size = SEQUENCE_LENGTH * NUM_FEATURES;
        let mut sessions = Vec::with_capacity(NUM_HORIZONS);

        for h in 1..=NUM_HORIZONS {
            let path = format!("{}/lgbm_horizon_{}.onnx", models_dir, h);

            if !Path::new(&path).exists() {
                return Err(ModelError::FileNotFound(path));
            }

            let session = ort::Session::builder()
                .map_err(|e| ModelError::LoadError(e.to_string()))?
                .with_intra_threads(1)
                .map_err(|e| ModelError::LoadError(e.to_string()))?
                .commit_from_file(&path)
                .map_err(|e| ModelError::LoadError(e.to_string()))?;

            sessions.push(session);
        }

        info!("Loaded {} LightGBM ONNX models from {}", NUM_HORIZONS, models_dir);

        Ok(Self {
            sessions,
            input_size,
        })
    }
}

#[cfg(feature = "ml-inference")]
impl Model for LightGBMModel {
    fn predict(&self, input: &[Vec<f64>]) -> Result<ModelPrediction, ModelError> {
        let start = std::time::Instant::now();

        // Flatten input: (seq_len * num_features,)
        let flat: Vec<f32> = input
            .iter()
            .flat_map(|row| row.iter().map(|&x| x as f32))
            .collect();

        if flat.len() != self.input_size {
            return Err(ModelError::InvalidShape {
                expected: format!("{}", self.input_size),
                got: format!("{}", flat.len()),
            });
        }

        let mut predictions = Vec::with_capacity(NUM_HORIZONS);

        for session in &self.sessions {
            let input_array = ndarray::Array::from_shape_vec((1, self.input_size), flat.clone())
                .map_err(|e| ModelError::InferenceError(e.to_string()))?;

            let outputs = session
                .run(ort::inputs!["input" => input_array.view()].unwrap())
                .map_err(|e| ModelError::InferenceError(e.to_string()))?;

            let output = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| ModelError::InferenceError(e.to_string()))?;

            predictions.push(output.view().iter().next().copied().unwrap_or(0.0) as f64);
        }

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(ModelPrediction {
            predictions,
            latency_ms,
        })
    }

    fn is_loaded(&self) -> bool {
        !self.sessions.is_empty()
    }

    fn name(&self) -> &str {
        "lgbm"
    }
}

/// Stub model for when ML inference is disabled
pub struct StubModel {
    name: String,
}

impl StubModel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

impl Model for StubModel {
    fn predict(&self, _input: &[Vec<f64>]) -> Result<ModelPrediction, ModelError> {
        Ok(ModelPrediction {
            predictions: vec![0.0; NUM_HORIZONS],
            latency_ms: 0.0,
        })
    }

    fn is_loaded(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Model manager with hot-reload support
pub struct ModelManager {
    #[cfg(feature = "ml-inference")]
    mamba: RwLock<Option<MambaModel>>,
    #[cfg(feature = "ml-inference")]
    lgbm: RwLock<Option<LightGBMModel>>,
    
    #[cfg(not(feature = "ml-inference"))]
    mamba: RwLock<Option<StubModel>>,
    #[cfg(not(feature = "ml-inference"))]
    lgbm: RwLock<Option<StubModel>>,
    
    norm_params: RwLock<Option<NormalizationParams>>,
    current_version: AtomicI32,
    config: RwLock<InferenceConfig>,
}

impl ModelManager {
    /// Create new model manager
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            mamba: RwLock::new(None),
            lgbm: RwLock::new(None),
            norm_params: RwLock::new(None),
            current_version: AtomicI32::new(config.ensemble_version_id),
            config: RwLock::new(config),
        }
    }

    /// Load all models
    pub fn load_models(&self) -> Result<(), ModelError> {
        let config = self.config.read().clone();

        if !config.enabled {
            info!("ML inference disabled, skipping model loading");
            return Ok(());
        }

        // Load normalization params
        if Path::new(&config.norm_params_path).exists() {
            let content = std::fs::read_to_string(&config.norm_params_path)?;
            let params = NormalizationParams::from_json(&content)
                .map_err(|e| ModelError::LoadError(e.to_string()))?;
            *self.norm_params.write() = Some(params);
            info!("Loaded normalization params from {}", config.norm_params_path);
        } else {
            warn!("Normalization params not found at {}", config.norm_params_path);
        }

        // Load Mamba model
        #[cfg(feature = "ml-inference")]
        {
            if Path::new(&config.mamba_model_path).exists() {
                match MambaModel::load(&config.mamba_model_path, &config.device) {
                    Ok(model) => {
                        *self.mamba.write() = Some(model);
                    }
                    Err(e) => {
                        error!("Failed to load Mamba model: {}", e);
                    }
                }
            }

            // Load LightGBM models
            match LightGBMModel::load(&config.lgbm_models_dir) {
                Ok(model) => {
                    *self.lgbm.write() = Some(model);
                }
                Err(e) => {
                    error!("Failed to load LightGBM models: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Check if models are loaded
    pub fn is_ready(&self) -> bool {
        let config = self.config.read();
        if !config.enabled {
            return false;
        }

        #[cfg(feature = "ml-inference")]
        {
            self.mamba.read().is_some() && self.lgbm.read().is_some()
        }
        
        #[cfg(not(feature = "ml-inference"))]
        {
            false
        }
    }

    /// Get current ensemble version
    pub fn current_version(&self) -> i32 {
        self.current_version.load(Ordering::SeqCst)
    }

    /// Check database for version changes and reload if needed
    pub async fn check_and_reload(&self, client: &Client) -> Result<bool, ModelError> {
        let query = r#"
            SELECT id FROM active_ensembles
            WHERE is_active = true
            LIMIT 1
        "#;

        let row = client.query_opt(query, &[]).await?;

        if let Some(row) = row {
            let db_version: i32 = row.get("id");
            let current = self.current_version.load(Ordering::SeqCst);

            if db_version != current {
                info!(
                    "Model version changed: {} -> {}, reloading...",
                    current, db_version
                );

                // Reload config from database
                if let Ok(new_config) = crate::config::ConfigLoader::from_database(client).await {
                    *self.config.write() = new_config;
                    self.current_version.store(db_version, Ordering::SeqCst);
                    self.load_models()?;
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get normalization params
    pub fn get_norm_params(&self) -> Option<NormalizationParams> {
        self.norm_params.read().clone()
    }

    /// Get Mamba weight from config
    pub fn get_mamba_weight(&self) -> f64 {
        self.config.read().mamba_weight
    }

    /// Run Mamba prediction
    #[cfg(feature = "ml-inference")]
    pub fn predict_mamba(&self, input: &[Vec<f64>]) -> Result<ModelPrediction, ModelError> {
        let guard = self.mamba.read();
        match guard.as_ref() {
            Some(model) => model.predict(input),
            None => Ok(ModelPrediction {
                predictions: vec![0.0; NUM_HORIZONS],
                latency_ms: 0.0,
            }),
        }
    }

    /// Run LightGBM prediction
    #[cfg(feature = "ml-inference")]
    pub fn predict_lgbm(&self, input: &[Vec<f64>]) -> Result<ModelPrediction, ModelError> {
        let guard = self.lgbm.read();
        match guard.as_ref() {
            Some(model) => model.predict(input),
            None => Ok(ModelPrediction {
                predictions: vec![0.0; NUM_HORIZONS],
                latency_ms: 0.0,
            }),
        }
    }

    /// Stub predictions when ml-inference is disabled
    #[cfg(not(feature = "ml-inference"))]
    pub fn predict_mamba(&self, _input: &[Vec<f64>]) -> Result<ModelPrediction, ModelError> {
        Ok(ModelPrediction {
            predictions: vec![0.0; NUM_HORIZONS],
            latency_ms: 0.0,
        })
    }

    #[cfg(not(feature = "ml-inference"))]
    pub fn predict_lgbm(&self, _input: &[Vec<f64>]) -> Result<ModelPrediction, ModelError> {
        Ok(ModelPrediction {
            predictions: vec![0.0; NUM_HORIZONS],
            latency_ms: 0.0,
        })
    }
}
