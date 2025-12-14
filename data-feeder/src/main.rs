//! Binance Data Feeder Service
//!
//! A production-ready service that streams cryptocurrency price data
//! from Binance and stores it in TimescaleDB.
//!
//! ## Features
//! - Dual WebSocket streams (kline_1m, bookTicker)
//! - Automatic reconnection with exponential backoff
//! - Batch database writes with retry logic
//! - Gap detection and backfill triggering
//! - Feature computation for ML models
//! - Health check HTTP endpoint
//! - Graceful shutdown on SIGTERM

mod binance;
mod config;
mod database;
mod errors;
mod features;
mod gap_detector;
mod health;

use crate::binance::{BinanceWebSocketClient, CandleData};
use crate::config::Config;
use crate::database::DatabaseWriter;
use crate::errors::{DataFeederError, Result, WriteResult};
use crate::features::FeatureComputer;
use crate::gap_detector::SimpleGapDetector;
use crate::health::{HealthServer, HealthState};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tokio::sync::mpsc;
use tracing::{error, info, warn, Level};
use tracing_subscriber::EnvFilter;

/// Channel buffer sizes
const CANDLE_CHANNEL_SIZE: usize = 1000;
const RESULT_CHANNEL_SIZE: usize = 100;

/// Application state shared between tasks.
struct AppState {
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Health state for metrics
    health: HealthState,
    /// Configuration
    config: Config,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    init_tracing();

    info!("Starting Binance Data Feeder Service");

    // Load configuration
    let config = Config::from_env()?;
    info!("Configuration loaded:");
    info!("  WebSocket URL: {}", config.websocket_url());
    info!("  Database: {}:{}/{}", 
        config.database.host, 
        config.database.port, 
        config.database.name
    );
    info!("  Health check port: {}", config.health_check_port);
    info!("  Gap detection threshold: {:?}", config.gap_detection_threshold);

    // Create shared state
    let shutdown = Arc::new(AtomicBool::new(false));
    let health_server = HealthServer::new(config.health_check_port);
    let health_state = health_server.state();

    let app_state = Arc::new(AppState {
        shutdown: shutdown.clone(),
        health: health_state.clone(),
        config: config.clone(),
    });

    // Create channels
    let (ws_candle_tx, ws_candle_rx) = mpsc::channel::<CandleData>(CANDLE_CHANNEL_SIZE);
    let (db_candle_tx, db_candle_rx) = mpsc::channel::<CandleData>(CANDLE_CHANNEL_SIZE);
    let (result_tx, result_rx) = mpsc::channel::<WriteResult>(RESULT_CHANNEL_SIZE);

    // Initialize database writer
    info!("Connecting to database...");
    let db_writer = match DatabaseWriter::new(&config.database, result_tx).await {
        Ok(writer) => {
            info!("Database connection established");
            health_state.set_db_connected(true);
            Arc::new(writer)
        }
        Err(e) => {
            error!("Failed to connect to database: {}", e);
            return Err(e);
        }
    };

    // Get last candle time from database for gap detection
    let last_candle_time = db_writer.get_latest_candle_time().await.ok().flatten();
    if let Some(time) = last_candle_time {
        info!("Last candle in database: {}", time);
    }

    // Initialize components
    let ws_client = Arc::new(BinanceWebSocketClient::new(
        config.websocket_url(),
        ws_candle_tx,
    ));

    let gap_detector = Arc::new(SimpleGapDetector::new(
        config.gap_detection_threshold.as_secs(),
    ));
    if let Some(time) = last_candle_time {
        gap_detector.set_last_candle_time(time);
    }

    let feature_computer = Arc::new(parking_lot::Mutex::new(FeatureComputer::new(60)));

    // Spawn health check server
    let health_handle = tokio::spawn({
        let state = health_state.clone();
        async move {
            if let Err(e) = health::run_health_server(config.health_check_port, state).await {
                error!("Health server error: {}", e);
            }
        }
    });

    // Spawn WebSocket client
    let ws_handle = tokio::spawn({
        let client = ws_client.clone();
        let shutdown = shutdown.clone();
        let health = health_state.clone();
        async move {
            loop {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                
                health.set_ws_connected(client.is_connected());
                
                if let Err(e) = client.run().await {
                    error!("WebSocket client error: {}", e);
                    health.increment_errors();
                }
                
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                
                // Brief pause before reconnect attempt
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            info!("WebSocket client task ended");
        }
    });

    // Spawn candle processor (gap detection + feature computation + forwarding)
    let processor_handle = tokio::spawn({
        let gap_detector = gap_detector.clone();
        let feature_computer = feature_computer.clone();
        let health = health_state.clone();
        let shutdown = shutdown.clone();
        let mut rx = ws_candle_rx;
        let tx = db_candle_tx;
        
        async move {
            while let Some(candle) = rx.recv().await {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                
                // Update health metrics
                health.set_last_candle_time(candle.time);
                
                // Check for gaps
                if let Some(gap) = gap_detector.update(&candle) {
                    warn!(
                        "Gap detected: {} minutes missing from {} to {}",
                        gap.candles_missing, gap.gap_start, gap.gap_end
                    );
                    health.increment_errors();
                }
                
                // Compute features
                {
                    let mut computer = feature_computer.lock();
                    if let Err(e) = computer.add_candle(&candle) {
                        warn!("Feature computation error: {}", e);
                    }
                    health.set_features_warmed_up(computer.is_warmed_up());
                }
                
                // Forward to database writer
                if tx.send(candle).await.is_err() {
                    error!("Failed to send candle to database writer");
                    break;
                }
            }
            info!("Candle processor task ended");
        }
    });

    // Spawn database writer
    let db_handle = tokio::spawn({
        let writer = db_writer.clone();
        let shutdown = shutdown.clone();
        let health = health_state.clone();
        
        async move {
            let writer_ref = writer.as_ref();
            if let Err(e) = writer_ref.run(db_candle_rx).await {
                error!("Database writer error: {}", e);
            }
            info!("Database writer task ended");
        }
    });

    // Spawn result processor
    let result_handle = tokio::spawn({
        let health = health_state.clone();
        let db_writer = db_writer.clone();
        let mut rx = result_rx;
        
        async move {
            while let Some(result) = rx.recv().await {
                match &result {
                    WriteResult::Success { count, duration_ms } => {
                        health.set_candles_written(db_writer.candles_written());
                        health.set_db_connected(true);
                        info!(
                            "Wrote {} candles in {}ms (total: {})",
                            count, duration_ms, db_writer.candles_written()
                        );
                    }
                    WriteResult::Partial { written, failed } => {
                        health.set_candles_written(db_writer.candles_written());
                        health.increment_errors();
                        warn!("Partial write: {} written, {} failed", written, failed);
                    }
                    WriteResult::Failed { error, buffered } => {
                        health.set_db_connected(false);
                        health.increment_errors();
                        error!("Write failed: {} ({} candles buffered)", error, buffered);
                    }
                }
                
                health.set_buffer_size(db_writer.buffered_count());
            }
            info!("Result processor task ended");
        }
    });

    // Spawn metrics updater
    let metrics_handle = tokio::spawn({
        let ws_client = ws_client.clone();
        let health = health_state.clone();
        let shutdown = shutdown.clone();
        
        async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                
                health.set_candles_processed(ws_client.candles_processed());
                health.set_ws_connected(ws_client.is_connected());
            }
            info!("Metrics updater task ended");
        }
    });

    // Wait for shutdown signal
    info!("Service started, waiting for shutdown signal...");
    wait_for_shutdown().await;

    // Initiate graceful shutdown
    info!("Shutdown signal received, initiating graceful shutdown...");
    shutdown.store(true, Ordering::SeqCst);
    ws_client.shutdown();
    db_writer.shutdown();

    // Wait for tasks to complete with timeout
    let shutdown_timeout = Duration::from_secs(10);
    
    tokio::select! {
        _ = async {
            let _ = ws_handle.await;
            let _ = processor_handle.await;
            let _ = db_handle.await;
            let _ = result_handle.await;
            let _ = metrics_handle.await;
        } => {
            info!("All tasks completed gracefully");
        }
        _ = tokio::time::sleep(shutdown_timeout) => {
            warn!("Shutdown timeout reached, forcing exit");
        }
    }

    // Abort health server (it doesn't have graceful shutdown)
    health_handle.abort();

    info!("Binance Data Feeder Service stopped");
    Ok(())
}

/// Initialize tracing subscriber.
fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();
}

/// Wait for shutdown signal (SIGTERM or SIGINT).
async fn wait_for_shutdown() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C");
        }
        _ = terminate => {
            info!("Received SIGTERM");
        }
    }
}
