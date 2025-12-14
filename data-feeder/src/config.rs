//! Configuration module for the data feeder service.
//!
//! Loads configuration from environment variables with sensible defaults.

use crate::errors::{DataFeederError, Result};
use std::env;
use std::time::Duration;

/// Main configuration struct for the data feeder.
#[derive(Debug, Clone)]
pub struct Config {
    /// Binance WebSocket stream URL
    pub binance_stream_url: String,

    /// Database configuration
    pub database: DatabaseConfig,

    /// Feature computation settings
    pub feature_update_interval: Duration,

    /// Gap detection threshold in seconds
    pub gap_detection_threshold: Duration,

    /// Gap handler service URL
    pub gap_handler_url: String,

    /// Health check HTTP server port
    pub health_check_port: u16,

    /// Log level
    pub log_level: String,

    /// Trading symbol (default: btcusdt)
    pub symbol: String,
}

/// Database connection configuration.
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub name: String,
    pub user: String,
    pub password: String,
    pub pool_min: usize,
    pub pool_max: usize,
    pub idle_timeout: Duration,
}

impl Config {
    /// Load configuration from environment variables.
    pub fn from_env() -> Result<Self> {
        // Load .env file if present (ignore errors if not found)
        let _ = dotenvy::dotenv();

        Ok(Self {
            binance_stream_url: env::var("BINANCE_STREAM_URL")
                .unwrap_or_else(|_| "wss://stream.binance.com:9443/ws".to_string()),

            database: DatabaseConfig {
                host: env::var("DB_HOST").unwrap_or_else(|_| "timescaledb".to_string()),
                port: env::var("DB_PORT")
                    .unwrap_or_else(|_| "5432".to_string())
                    .parse()
                    .map_err(|_| DataFeederError::config("Invalid DB_PORT"))?,
                name: env::var("DB_NAME").unwrap_or_else(|_| "btc_ml_production".to_string()),
                user: env::var("DB_USER").unwrap_or_else(|_| "mltrader".to_string()),
                password: env::var("DB_PASSWORD")
                    .map_err(|_| DataFeederError::config("DB_PASSWORD is required"))?,
                pool_min: env::var("DB_POOL_MIN")
                    .unwrap_or_else(|_| "2".to_string())
                    .parse()
                    .unwrap_or(2),
                pool_max: env::var("DB_POOL_MAX")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .unwrap_or(5),
                idle_timeout: Duration::from_secs(
                    env::var("DB_IDLE_TIMEOUT")
                        .unwrap_or_else(|_| "60".to_string())
                        .parse()
                        .unwrap_or(60),
                ),
            },

            feature_update_interval: Duration::from_secs(
                env::var("FEATURE_UPDATE_INTERVAL")
                    .unwrap_or_else(|_| "60".to_string())
                    .parse()
                    .unwrap_or(60),
            ),

            gap_detection_threshold: Duration::from_secs(
                env::var("GAP_DETECTION_THRESHOLD")
                    .unwrap_or_else(|_| "70".to_string())
                    .parse()
                    .unwrap_or(70),
            ),

            gap_handler_url: env::var("GAP_HANDLER_URL")
                .unwrap_or_else(|_| "http://gap-handler:9000/backfill".to_string()),

            health_check_port: env::var("HEALTH_CHECK_PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .map_err(|_| DataFeederError::config("Invalid HEALTH_CHECK_PORT"))?,

            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),

            symbol: env::var("TRADING_SYMBOL").unwrap_or_else(|_| "btcusdt".to_string()),
        })
    }

    /// Build the WebSocket URL for combined streams.
    /// Uses the /stream endpoint with streams parameter for combined stream format.
    pub fn websocket_url(&self) -> String {
        // Combined streams use /stream?streams= format which wraps messages with {"stream":..., "data":...}
        format!(
            "{}/stream?streams={}@kline_1m/{}@bookTicker",
            self.binance_stream_url.trim_end_matches("/ws"),
            self.symbol.to_lowercase(),
            self.symbol.to_lowercase()
        )
    }

    /// Build the database connection string.
    pub fn database_url(&self) -> String {
        format!(
            "host={} port={} dbname={} user={} password={}",
            self.database.host,
            self.database.port,
            self.database.name,
            self.database.user,
            self.database.password
        )
    }
}

impl DatabaseConfig {
    /// Create a deadpool configuration.
    pub fn to_pool_config(&self) -> deadpool_postgres::Config {
        let mut cfg = deadpool_postgres::Config::new();
        cfg.host = Some(self.host.clone());
        cfg.port = Some(self.port);
        cfg.dbname = Some(self.name.clone());
        cfg.user = Some(self.user.clone());
        cfg.password = Some(self.password.clone());
        cfg
    }
}
