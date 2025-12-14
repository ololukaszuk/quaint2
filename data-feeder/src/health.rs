//! Health check HTTP server module.
//!
//! Provides a /health endpoint that returns comprehensive system status:
//! - Connection states (WebSocket, Database)
//! - Processing statistics
//! - Error counts
//! - Memory usage

use axum::{extract::State, http::StatusCode, response::Json, routing::get, Router};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

/// Health check response payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Overall status: healthy, degraded, or unhealthy
    pub status: HealthStatus,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Total candles processed by WebSocket client
    pub candles_processed: u64,
    /// Total candles written to database
    pub candles_in_db: u64,
    /// Timestamp of last candle received
    pub last_candle_time: Option<DateTime<Utc>>,
    /// Database connection status
    pub database_connected: bool,
    /// WebSocket connection status
    pub websocket_connected: bool,
    /// Number of pending backfill operations
    pub pending_backfills: u32,
    /// Number of errors in the last 24 hours
    pub errors_24h: u64,
    /// Current memory usage in MB
    pub memory_mb: f64,
    /// Current timestamp
    pub timestamp: DateTime<Utc>,
    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<HealthDetails>,
}

/// Detailed health information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDetails {
    /// Write buffer size
    pub buffer_size: usize,
    /// Database pool stats
    pub db_pool_size: Option<usize>,
    /// Feature computation status
    pub features_warmed_up: bool,
    /// Gap detection status
    pub gap_check_last: Option<DateTime<Utc>>,
}

/// Health status enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

/// Shared state for health checks.
#[derive(Clone)]
pub struct HealthState {
    inner: Arc<HealthStateInner>,
}

struct HealthStateInner {
    /// Service start time
    start_time: Instant,
    /// Candles processed counter
    candles_processed: AtomicU64,
    /// Candles written counter
    candles_written: AtomicU64,
    /// Last candle time
    last_candle_time: RwLock<Option<DateTime<Utc>>>,
    /// Database connected flag
    db_connected: RwLock<bool>,
    /// WebSocket connected flag
    ws_connected: RwLock<bool>,
    /// Pending backfills counter
    pending_backfills: AtomicU64,
    /// Errors in last 24h
    errors_24h: AtomicU64,
    /// Buffer size
    buffer_size: AtomicU64,
    /// Features warmed up flag
    features_warmed_up: RwLock<bool>,
    /// Memory limit in MB for healthy status
    memory_limit_mb: f64,
}

impl HealthState {
    /// Create a new health state.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(HealthStateInner {
                start_time: Instant::now(),
                candles_processed: AtomicU64::new(0),
                candles_written: AtomicU64::new(0),
                last_candle_time: RwLock::new(None),
                db_connected: RwLock::new(false),
                ws_connected: RwLock::new(false),
                pending_backfills: AtomicU64::new(0),
                errors_24h: AtomicU64::new(0),
                buffer_size: AtomicU64::new(0),
                features_warmed_up: RwLock::new(false),
                memory_limit_mb: 100.0,
            }),
        }
    }

    /// Update candles processed count.
    pub fn set_candles_processed(&self, count: u64) {
        self.inner.candles_processed.store(count, Ordering::SeqCst);
    }

    /// Update candles written count.
    pub fn set_candles_written(&self, count: u64) {
        self.inner.candles_written.store(count, Ordering::SeqCst);
    }

    /// Update last candle time.
    pub fn set_last_candle_time(&self, time: DateTime<Utc>) {
        *self.inner.last_candle_time.write() = Some(time);
    }

    /// Update database connection status.
    pub fn set_db_connected(&self, connected: bool) {
        *self.inner.db_connected.write() = connected;
    }

    /// Update WebSocket connection status.
    pub fn set_ws_connected(&self, connected: bool) {
        *self.inner.ws_connected.write() = connected;
    }

    /// Update pending backfills count.
    pub fn set_pending_backfills(&self, count: u32) {
        self.inner.pending_backfills.store(count as u64, Ordering::SeqCst);
    }

    /// Increment error count.
    pub fn increment_errors(&self) {
        self.inner.errors_24h.fetch_add(1, Ordering::SeqCst);
    }

    /// Reset error count (call daily).
    pub fn reset_errors(&self) {
        self.inner.errors_24h.store(0, Ordering::SeqCst);
    }

    /// Update buffer size.
    pub fn set_buffer_size(&self, size: usize) {
        self.inner.buffer_size.store(size as u64, Ordering::SeqCst);
    }

    /// Update features warmed up status.
    pub fn set_features_warmed_up(&self, warmed: bool) {
        *self.inner.features_warmed_up.write() = warmed;
    }

    /// Build health response.
    pub fn build_response(&self) -> HealthResponse {
        let uptime = self.inner.start_time.elapsed().as_secs();
        let candles_processed = self.inner.candles_processed.load(Ordering::SeqCst);
        let candles_written = self.inner.candles_written.load(Ordering::SeqCst);
        let last_candle_time = *self.inner.last_candle_time.read();
        let db_connected = *self.inner.db_connected.read();
        let ws_connected = *self.inner.ws_connected.read();
        let pending_backfills = self.inner.pending_backfills.load(Ordering::SeqCst) as u32;
        let errors_24h = self.inner.errors_24h.load(Ordering::SeqCst);
        let buffer_size = self.inner.buffer_size.load(Ordering::SeqCst) as usize;
        let features_warmed_up = *self.inner.features_warmed_up.read();

        // Get memory usage
        let memory_mb = get_memory_usage_mb();

        // Determine status
        let status = self.determine_status(
            db_connected,
            ws_connected,
            errors_24h,
            memory_mb,
            last_candle_time,
        );

        HealthResponse {
            status,
            uptime_seconds: uptime,
            candles_processed,
            candles_in_db: candles_written,
            last_candle_time,
            database_connected: db_connected,
            websocket_connected: ws_connected,
            pending_backfills,
            errors_24h,
            memory_mb,
            timestamp: Utc::now(),
            details: Some(HealthDetails {
                buffer_size,
                db_pool_size: None,
                features_warmed_up,
                gap_check_last: last_candle_time,
            }),
        }
    }

    /// Determine overall health status.
    fn determine_status(
        &self,
        db_connected: bool,
        ws_connected: bool,
        errors_24h: u64,
        memory_mb: f64,
        last_candle_time: Option<DateTime<Utc>>,
    ) -> HealthStatus {
        // Check for unhealthy conditions
        if !db_connected || !ws_connected {
            return HealthStatus::Unhealthy;
        }

        // Check for stale data (no candles in 2+ minutes)
        if let Some(last) = last_candle_time {
            let elapsed = Utc::now() - last;
            if elapsed.num_seconds() > 120 {
                return HealthStatus::Unhealthy;
            }
        }

        // Check for degraded conditions
        if errors_24h > 10 {
            return HealthStatus::Degraded;
        }

        if memory_mb > self.inner.memory_limit_mb {
            return HealthStatus::Degraded;
        }

        // Check for gap in last candle time
        if let Some(last) = last_candle_time {
            let elapsed = Utc::now() - last;
            if elapsed.num_seconds() > 60 {
                return HealthStatus::Degraded;
            }
        }

        HealthStatus::Healthy
    }
}

impl Default for HealthState {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current memory usage in MB.
fn get_memory_usage_mb() -> f64 {
    // On Linux, read from /proc/self/statm
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = content.split_whitespace().nth(1) {
                if let Ok(pages) = rss_pages.parse::<u64>() {
                    let page_size = 4096;
                    return (pages * page_size) as f64 / (1024.0 * 1024.0);
                }
            }
        }
    }
    0.0
}

/// Health check handler.
async fn health_handler(State(state): State<HealthState>) -> (StatusCode, Json<HealthResponse>) {
    let response = state.build_response();

    let status_code = match response.status {
        HealthStatus::Healthy => StatusCode::OK,
        HealthStatus::Degraded => StatusCode::OK,
        HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };

    (status_code, Json(response))
}

/// Readiness check handler.
async fn ready_handler(State(state): State<HealthState>) -> StatusCode {
    let db_connected = *state.inner.db_connected.read();
    let ws_connected = *state.inner.ws_connected.read();

    if db_connected && ws_connected {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Liveness check handler.
async fn live_handler() -> StatusCode {
    StatusCode::OK
}

/// Create the health check router.
pub fn create_health_router(state: HealthState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/live", get(live_handler))
        .with_state(state)
}

/// Run the health check server.
pub async fn run_health_server(port: u16, state: HealthState) -> std::io::Result<()> {
    let app = create_health_router(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    info!("Health check server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check server handle.
pub struct HealthServer {
    state: HealthState,
    port: u16,
}

impl HealthServer {
    /// Create a new health server.
    pub fn new(port: u16) -> Self {
        Self {
            state: HealthState::new(),
            port,
        }
    }

    /// Get a clone of the state for updating.
    pub fn state(&self) -> HealthState {
        self.state.clone()
    }

    /// Run the server (blocking).
    pub async fn run(&self) -> std::io::Result<()> {
        run_health_server(self.port, self.state.clone()).await
    }

    /// Spawn the server as a background task.
    pub fn spawn(self) -> tokio::task::JoinHandle<std::io::Result<()>> {
        tokio::spawn(async move { self.run().await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_serialization() {
        assert_eq!(
            serde_json::to_string(&HealthStatus::Healthy).unwrap(),
            "\"healthy\""
        );
        assert_eq!(
            serde_json::to_string(&HealthStatus::Degraded).unwrap(),
            "\"degraded\""
        );
        assert_eq!(
            serde_json::to_string(&HealthStatus::Unhealthy).unwrap(),
            "\"unhealthy\""
        );
    }

    #[test]
    fn test_health_state() {
        let state = HealthState::new();

        let response = state.build_response();
        assert_eq!(response.candles_processed, 0);
        assert_eq!(response.candles_in_db, 0);
        assert!(response.last_candle_time.is_none());

        state.set_candles_processed(100);
        state.set_candles_written(95);
        state.set_db_connected(true);
        state.set_ws_connected(true);
        state.set_last_candle_time(Utc::now());

        let response = state.build_response();
        assert_eq!(response.candles_processed, 100);
        assert_eq!(response.candles_in_db, 95);
        assert!(response.database_connected);
        assert!(response.websocket_connected);
        assert!(response.last_candle_time.is_some());
        assert_eq!(response.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_unhealthy_status() {
        let state = HealthState::new();

        state.set_db_connected(false);
        state.set_ws_connected(true);
        let response = state.build_response();
        assert_eq!(response.status, HealthStatus::Unhealthy);

        state.set_db_connected(true);
        state.set_ws_connected(false);
        let response = state.build_response();
        assert_eq!(response.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_degraded_status() {
        let state = HealthState::new();
        state.set_db_connected(true);
        state.set_ws_connected(true);
        state.set_last_candle_time(Utc::now());

        for _ in 0..15 {
            state.increment_errors();
        }

        let response = state.build_response();
        assert_eq!(response.status, HealthStatus::Degraded);
    }
}
