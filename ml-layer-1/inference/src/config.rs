//! Configuration Module
//!
//! Manages inference configuration from:
//! 1. Database (active_ensembles table) - Primary
//! 2. Environment variables - Fallback
//! 3. TOML config file - Last resort

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tokio_postgres::Client;
use tracing::{debug, info, warn};

/// Configuration errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Database error: {0}")]
    Database(#[from] tokio_postgres::Error),

    #[error("Environment variable error: {0}")]
    EnvVar(String),

    #[error("File error: {0}")]
    File(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("No active ensemble found")]
    NoActiveEnsemble,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Path to Mamba TorchScript model
    pub mamba_model_path: String,

    /// Directory containing LightGBM ONNX models
    pub lgbm_models_dir: String,

    /// Path to normalization parameters JSON
    pub norm_params_path: String,

    /// Weight for Mamba in ensemble (0.0 to 1.0)
    pub mamba_weight: f64,

    /// Inference device ("cpu" or "cuda")
    pub device: String,

    /// Active ensemble version ID from database
    pub ensemble_version_id: i32,

    /// Whether inference is enabled
    pub enabled: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            mamba_model_path: "/app/models/mamba.pt".to_string(),
            lgbm_models_dir: "/app/models".to_string(),
            norm_params_path: "/app/models/normalization_params_v1.json".to_string(),
            mamba_weight: 0.5,
            device: "cpu".to_string(),
            ensemble_version_id: 0,
            enabled: false,
        }
    }
}

/// Configuration loader
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from database (primary source)
    ///
    /// Queries active_ensembles table for active ensemble configuration.
    pub async fn from_database(client: &Client) -> Result<InferenceConfig, ConfigError> {
        let query = r#"
            SELECT 
                ae.id,
                ae.mamba_version_id,
                ae.lgbm_version_id,
                ae.mamba_weight,
                mv_mamba.model_path as mamba_path,
                mv_lgbm.model_path as lgbm_path
            FROM active_ensembles ae
            LEFT JOIN model_versions mv_mamba ON ae.mamba_version_id = mv_mamba.id
            LEFT JOIN model_versions mv_lgbm ON ae.lgbm_version_id = mv_lgbm.id
            WHERE ae.is_active = true
            LIMIT 1
        "#;

        let row = client.query_opt(query, &[]).await?;

        match row {
            Some(row) => {
                let ensemble_id: i32 = row.get("id");
                let mamba_weight: f64 = row
                    .get::<_, Option<rust_decimal::Decimal>>("mamba_weight")
                    .and_then(|d| d.to_string().parse().ok())
                    .unwrap_or(0.5);
                let mamba_path: String = row
                    .get::<_, Option<String>>("mamba_path")
                    .unwrap_or_else(|| "/app/models/mamba.pt".to_string());
                let lgbm_path: String = row
                    .get::<_, Option<String>>("lgbm_path")
                    .unwrap_or_else(|| "/app/models".to_string());

                info!(
                    "Loaded config from database: ensemble_id={}, mamba_weight={}",
                    ensemble_id, mamba_weight
                );

                Ok(InferenceConfig {
                    mamba_model_path: mamba_path,
                    lgbm_models_dir: lgbm_path,
                    norm_params_path: "/app/models/normalization_params_v1.json".to_string(),
                    mamba_weight,
                    device: std::env::var("ML_DEVICE").unwrap_or_else(|_| "cpu".to_string()),
                    ensemble_version_id: ensemble_id,
                    enabled: true,
                })
            }
            None => {
                warn!("No active ensemble found in database");
                Err(ConfigError::NoActiveEnsemble)
            }
        }
    }

    /// Load configuration from environment variables (fallback)
    pub fn from_env() -> Result<InferenceConfig, ConfigError> {
        let enabled = std::env::var("ML_ENABLED")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        if !enabled {
            debug!("ML inference disabled via ML_ENABLED env var");
            return Ok(InferenceConfig {
                enabled: false,
                ..Default::default()
            });
        }

        let mamba_model_path = std::env::var("ML_MAMBA_MODEL_PATH")
            .unwrap_or_else(|_| "/app/models/mamba.pt".to_string());

        let lgbm_models_dir = std::env::var("ML_LGBM_MODELS_DIR")
            .unwrap_or_else(|_| "/app/models".to_string());

        let norm_params_path = std::env::var("ML_NORM_PARAMS_PATH")
            .unwrap_or_else(|_| "/app/models/normalization_params_v1.json".to_string());

        let mamba_weight: f64 = std::env::var("ML_MAMBA_WEIGHT")
            .unwrap_or_else(|_| "0.5".to_string())
            .parse()
            .unwrap_or(0.5);

        let device = std::env::var("ML_DEVICE").unwrap_or_else(|_| "cpu".to_string());

        let ensemble_version_id: i32 = std::env::var("ML_ENSEMBLE_VERSION")
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .unwrap_or(0);

        info!("Loaded config from environment variables");

        Ok(InferenceConfig {
            mamba_model_path,
            lgbm_models_dir,
            norm_params_path,
            mamba_weight,
            device,
            ensemble_version_id,
            enabled,
        })
    }

    /// Load configuration from TOML file (last resort)
    pub fn from_file(path: &str) -> Result<InferenceConfig, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: InferenceConfig =
            toml::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?;

        info!("Loaded config from file: {}", path);

        Ok(config)
    }

    /// Load configuration with fallback chain:
    /// 1. Database (if client provided)
    /// 2. Environment variables
    /// 3. Config file (if path provided)
    /// 4. Default
    pub async fn load(
        client: Option<&Client>,
        config_file: Option<&str>,
    ) -> Result<InferenceConfig, ConfigError> {
        // Try database first
        if let Some(client) = client {
            match Self::from_database(client).await {
                Ok(config) => return Ok(config),
                Err(e) => {
                    warn!("Failed to load config from database: {}", e);
                }
            }
        }

        // Try environment variables
        match Self::from_env() {
            Ok(config) if config.enabled => return Ok(config),
            Ok(_) => {
                debug!("Config from env has enabled=false, continuing fallback");
            }
            Err(e) => {
                warn!("Failed to load config from environment: {}", e);
            }
        }

        // Try config file
        if let Some(path) = config_file {
            if Path::new(path).exists() {
                match Self::from_file(path) {
                    Ok(config) => return Ok(config),
                    Err(e) => {
                        warn!("Failed to load config from file {}: {}", path, e);
                    }
                }
            }
        }

        // Return default (disabled)
        warn!("Using default config (inference disabled)");
        Ok(InferenceConfig::default())
    }
}

/// Database configuration for connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub name: String,
    pub user: String,
    pub password: String,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            host: "timescaledb".to_string(),
            port: 5432,
            name: "btc_ml_production".to_string(),
            user: "mltrader".to_string(),
            password: String::new(),
        }
    }
}

impl DatabaseConfig {
    /// Load from environment variables
    pub fn from_env() -> Self {
        Self {
            host: std::env::var("DB_HOST").unwrap_or_else(|_| "timescaledb".to_string()),
            port: std::env::var("DB_PORT")
                .unwrap_or_else(|_| "5432".to_string())
                .parse()
                .unwrap_or(5432),
            name: std::env::var("DB_NAME").unwrap_or_else(|_| "btc_ml_production".to_string()),
            user: std::env::var("DB_USER").unwrap_or_else(|_| "mltrader".to_string()),
            password: std::env::var("DB_PASSWORD").unwrap_or_default(),
        }
    }

    /// Build connection string
    pub fn connection_string(&self) -> String {
        format!(
            "host={} port={} dbname={} user={} password={}",
            self.host, self.port, self.name, self.user, self.password
        )
    }
}
