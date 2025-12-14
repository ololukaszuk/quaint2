//! Error types for the data feeder service.
//!
//! This module defines all error types used throughout the application,
//! providing structured error handling with context.

use thiserror::Error;

/// Main error type for the data feeder service.
#[derive(Error, Debug)]
pub enum DataFeederError {
    /// WebSocket connection errors
    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),

    /// Database connection and query errors
    #[error("Database error: {0}")]
    Database(#[from] tokio_postgres::Error),

    /// Database pool errors
    #[error("Database pool error: {0}")]
    Pool(#[from] deadpool_postgres::PoolError),

    /// JSON parsing errors
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Channel communication errors
    #[error("Channel error: {0}")]
    Channel(String),

    /// Gap detection errors
    #[error("Gap detection error: {0}")]
    GapDetection(String),

    /// Feature computation errors
    #[error("Feature computation error: {0}")]
    FeatureComputation(String),

    /// HTTP client errors (for gap handler)
    #[error("HTTP error: {0}")]
    Http(String),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// URL parsing errors
    #[error("URL error: {0}")]
    Url(#[from] url::ParseError),

    /// Generic errors with context
    #[error("{context}: {source}")]
    WithContext {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl DataFeederError {
    /// Create a configuration error with a message.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a channel error with a message.
    pub fn channel(msg: impl Into<String>) -> Self {
        Self::Channel(msg.into())
    }

    /// Create a gap detection error with a message.
    pub fn gap_detection(msg: impl Into<String>) -> Self {
        Self::GapDetection(msg.into())
    }

    /// Create a feature computation error with a message.
    pub fn feature_computation(msg: impl Into<String>) -> Self {
        Self::FeatureComputation(msg.into())
    }

    /// Create an HTTP error with a message.
    pub fn http(msg: impl Into<String>) -> Self {
        Self::Http(msg.into())
    }

    /// Add context to an error.
    pub fn with_context<E>(context: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::WithContext {
            context: context.into(),
            source: Box::new(source),
        }
    }
}

/// Result type alias using DataFeederError.
pub type Result<T> = std::result::Result<T, DataFeederError>;

/// Represents the result of a database write operation.
#[derive(Debug, Clone)]
pub enum WriteResult {
    /// Successfully wrote the specified number of candles.
    Success { count: usize, duration_ms: u64 },
    /// Partial success - some candles written.
    Partial { written: usize, failed: usize },
    /// Complete failure.
    Failed { error: String, buffered: usize },
}

impl WriteResult {
    /// Check if the write was fully successful.
    pub fn is_success(&self) -> bool {
        matches!(self, WriteResult::Success { .. })
    }

    /// Get the number of successfully written candles.
    pub fn written_count(&self) -> usize {
        match self {
            WriteResult::Success { count, .. } => *count,
            WriteResult::Partial { written, .. } => *written,
            WriteResult::Failed { .. } => 0,
        }
    }
}
