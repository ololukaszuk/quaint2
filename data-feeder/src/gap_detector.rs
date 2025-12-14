//! Gap detector module for monitoring data continuity.
//!
//! Monitors incoming candle timestamps and detects gaps in data.
//! When a gap is detected:
//! 1. Logs to data_quality_logs table
//! 2. Triggers gap_handler service via HTTP
//! 3. Waits for backfill completion

use crate::binance::CandleData;
use crate::errors::{DataFeederError, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Gap detector for monitoring data continuity.
pub struct GapDetector {
    /// Gap detection threshold
    threshold: Duration,
    /// Gap handler service URL
    gap_handler_url: String,
    /// Last candle timestamp
    last_candle_time: Arc<RwLock<Option<DateTime<Utc>>>>,
    /// Number of pending backfills
    pending_backfills: Arc<AtomicU32>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// HTTP client for gap handler requests
    http_client: reqwest::Client,
}

/// Gap event for logging.
#[derive(Debug, Clone)]
pub struct GapEvent {
    pub gap_start: DateTime<Utc>,
    pub gap_end: DateTime<Utc>,
    pub candles_missing: i64,
    pub detected_at: DateTime<Utc>,
}

impl GapDetector {
    /// Create a new gap detector.
    pub fn new(threshold: Duration, gap_handler_url: String) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();

        Self {
            threshold,
            gap_handler_url,
            last_candle_time: Arc::new(RwLock::new(None)),
            pending_backfills: Arc::new(AtomicU32::new(0)),
            shutdown: Arc::new(AtomicBool::new(false)),
            http_client,
        }
    }

    /// Get the last candle timestamp.
    pub fn last_candle_time(&self) -> Option<DateTime<Utc>> {
        *self.last_candle_time.read()
    }

    /// Get the number of pending backfills.
    pub fn pending_backfills(&self) -> u32 {
        self.pending_backfills.load(Ordering::SeqCst)
    }

    /// Signal the detector to shutdown.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Update the last candle time and check for gaps.
    pub fn update(&self, candle: &CandleData) -> Option<GapEvent> {
        let new_time = candle.time;
        let gap_event = {
            let last_time = *self.last_candle_time.read();

            if let Some(last) = last_time {
                // Check if there's a gap
                let expected_next = last + ChronoDuration::minutes(1);
                if new_time > expected_next + ChronoDuration::seconds(10) {
                    // Gap detected (allowing 10 second tolerance)
                    let gap_minutes = (new_time - expected_next).num_minutes();
                    Some(GapEvent {
                        gap_start: expected_next,
                        gap_end: new_time,
                        candles_missing: gap_minutes,
                        detected_at: Utc::now(),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        };

        // Update last candle time
        *self.last_candle_time.write() = Some(new_time);

        gap_event
    }

    /// Run the gap detector, monitoring candle timestamps.
    pub async fn run(
        &self,
        mut candle_rx: mpsc::Receiver<CandleData>,
        candle_tx: mpsc::Sender<CandleData>,
    ) -> Result<()> {
        let check_interval = Duration::from_secs(10);
        let mut last_check = std::time::Instant::now();

        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                info!("Gap detector shutting down");
                break;
            }

            tokio::select! {
                candle = candle_rx.recv() => {
                    match candle {
                        Some(c) => {
                            // Check for gap before forwarding
                            if let Some(gap) = self.update(&c) {
                                warn!(
                                    "Gap detected: {} minutes missing between {} and {}",
                                    gap.candles_missing, gap.gap_start, gap.gap_end
                                );
                                self.handle_gap(gap).await;
                            }

                            // Forward candle to database writer
                            let _ = candle_tx.send(c).await;
                        }
                        None => {
                            info!("Gap detector channel closed");
                            break;
                        }
                    }
                }
                _ = tokio::time::sleep(check_interval) => {
                    // Periodic check for stale data
                    if last_check.elapsed() >= check_interval {
                        self.check_for_stale_data().await;
                        last_check = std::time::Instant::now();
                    }
                }
            }
        }

        Ok(())
    }

    /// Run the gap detector in pass-through mode (no separate task).
    pub async fn run_inline(&self, candle: &CandleData) {
        if let Some(gap) = self.update(candle) {
            warn!(
                "Gap detected: {} minutes missing between {} and {}",
                gap.candles_missing, gap.gap_start, gap.gap_end
            );
            self.handle_gap(gap).await;
        }
    }

    /// Handle a detected gap.
    async fn handle_gap(&self, gap: GapEvent) {
        self.pending_backfills.fetch_add(1, Ordering::SeqCst);

        // Trigger backfill via HTTP
        match self.trigger_backfill(&gap).await {
            Ok(_) => {
                info!(
                    "Backfill triggered for gap: {} to {}",
                    gap.gap_start, gap.gap_end
                );
            }
            Err(e) => {
                error!("Failed to trigger backfill: {}", e);
            }
        }

        self.pending_backfills.fetch_sub(1, Ordering::SeqCst);
    }

    /// Trigger the gap handler service.
    async fn trigger_backfill(&self, gap: &GapEvent) -> Result<()> {
        let payload = serde_json::json!({
            "gap_start": gap.gap_start.to_rfc3339(),
            "gap_end": gap.gap_end.to_rfc3339(),
            "candles_missing": gap.candles_missing,
            "detected_at": gap.detected_at.to_rfc3339(),
        });

        debug!("Sending backfill request to {}", self.gap_handler_url);

        let response = self
            .http_client
            .post(&self.gap_handler_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| DataFeederError::http(format!("Failed to send backfill request: {}", e)))?;

        if !response.status().is_success() {
            return Err(DataFeederError::http(format!(
                "Backfill request failed with status: {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Check for stale data (no candles for threshold duration).
    async fn check_for_stale_data(&self) {
        let last_time = *self.last_candle_time.read();

        if let Some(last) = last_time {
            let elapsed = Utc::now() - last;
            let threshold_chrono = ChronoDuration::from_std(self.threshold).unwrap_or_default();

            if elapsed > threshold_chrono {
                warn!(
                    "Stale data detected: no candles for {:?} (threshold: {:?})",
                    elapsed, self.threshold
                );
            }
        }
    }

    /// Set the last candle time (for initialization from database).
    pub fn set_last_candle_time(&self, time: DateTime<Utc>) {
        *self.last_candle_time.write() = Some(time);
    }
}

/// Simplified gap detector that doesn't require HTTP client.
/// Used when gap_handler is not available.
pub struct SimpleGapDetector {
    /// Gap detection threshold in seconds
    threshold_secs: u64,
    /// Last candle timestamp
    last_candle_time: Arc<RwLock<Option<DateTime<Utc>>>>,
    /// Detected gaps
    gaps: Arc<RwLock<Vec<GapEvent>>>,
}

impl SimpleGapDetector {
    /// Create a new simple gap detector.
    pub fn new(threshold_secs: u64) -> Self {
        Self {
            threshold_secs,
            last_candle_time: Arc::new(RwLock::new(None)),
            gaps: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Update with new candle and check for gaps.
    pub fn update(&self, candle: &CandleData) -> Option<GapEvent> {
        let new_time = candle.time;
        let gap_event = {
            let last_time = *self.last_candle_time.read();

            if let Some(last) = last_time {
                let expected_next = last + ChronoDuration::minutes(1);
                let tolerance = ChronoDuration::seconds(10);

                if new_time > expected_next + tolerance {
                    let gap_minutes = (new_time - expected_next).num_minutes();
                    let event = GapEvent {
                        gap_start: expected_next,
                        gap_end: new_time,
                        candles_missing: gap_minutes,
                        detected_at: Utc::now(),
                    };

                    // Store gap
                    self.gaps.write().push(event.clone());

                    Some(event)
                } else {
                    None
                }
            } else {
                None
            }
        };

        *self.last_candle_time.write() = Some(new_time);
        gap_event
    }

    /// Get all detected gaps.
    pub fn get_gaps(&self) -> Vec<GapEvent> {
        self.gaps.read().clone()
    }

    /// Get the last candle time.
    pub fn last_candle_time(&self) -> Option<DateTime<Utc>> {
        *self.last_candle_time.read()
    }

    /// Check if data is stale.
    pub fn is_stale(&self) -> bool {
        if let Some(last) = *self.last_candle_time.read() {
            let elapsed = Utc::now() - last;
            elapsed.num_seconds() > self.threshold_secs as i64
        } else {
            false
        }
    }

    /// Set the last candle time.
    pub fn set_last_candle_time(&self, time: DateTime<Utc>) {
        *self.last_candle_time.write() = Some(time);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candle(time: DateTime<Utc>) -> CandleData {
        CandleData {
            time,
            open: 100000.0,
            high: 100500.0,
            low: 99500.0,
            close: 100200.0,
            volume: 1000.0,
            quote_asset_volume: 100000000.0,
            taker_buy_base_asset_volume: 500.0,
            taker_buy_quote_asset_volume: 50000000.0,
            number_of_trades: 5000,
            spread_bps: Some(5.0),
            best_bid: Some(100199.0),
            best_ask: Some(100201.0),
        }
    }

    #[test]
    fn test_gap_detection() {
        let detector = SimpleGapDetector::new(70);

        // First candle - no gap
        let t1 = Utc::now();
        let c1 = create_test_candle(t1);
        assert!(detector.update(&c1).is_none());

        // Second candle - 1 minute later, no gap
        let t2 = t1 + ChronoDuration::minutes(1);
        let c2 = create_test_candle(t2);
        assert!(detector.update(&c2).is_none());

        // Third candle - 5 minutes later, should detect gap
        let t3 = t2 + ChronoDuration::minutes(5);
        let c3 = create_test_candle(t3);
        let gap = detector.update(&c3);
        assert!(gap.is_some());

        let gap = gap.unwrap();
        assert_eq!(gap.candles_missing, 4); // 5 min gap = 4 missing candles
    }

    #[test]
    fn test_stale_detection() {
        let detector = SimpleGapDetector::new(70);

        // Set old timestamp
        let old_time = Utc::now() - ChronoDuration::seconds(100);
        detector.set_last_candle_time(old_time);

        assert!(detector.is_stale());

        // Set recent timestamp
        detector.set_last_candle_time(Utc::now());
        assert!(!detector.is_stale());
    }
}
