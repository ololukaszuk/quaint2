//! Binance Data Feeder Library
//!
//! This crate provides components for streaming cryptocurrency data
//! from Binance to TimescaleDB.

pub mod binance;
pub mod config;
pub mod database;
pub mod errors;
pub mod features;
pub mod gap_detector;
pub mod health;

pub use binance::{BinanceWebSocketClient, CandleData};
pub use config::Config;
pub use database::DatabaseWriter;
pub use errors::{DataFeederError, Result, WriteResult};
pub use features::{ComputedFeatures, FeatureComputer};
pub use gap_detector::{GapDetector, GapEvent, SimpleGapDetector};
pub use health::{HealthResponse, HealthServer, HealthState, HealthStatus};
