//! Database writer module for TimescaleDB operations.
//!
//! Handles:
//! - Connection pooling (min 2, max 5)
//! - Batch writing with buffering
//! - Retry logic with exponential backoff
//! - Prepared statements for efficiency

use crate::binance::CandleData;
use crate::config::DatabaseConfig;
use crate::errors::{DataFeederError, Result, WriteResult};
use chrono::{DateTime, Utc};
use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod};
use parking_lot::Mutex;
use rust_decimal::Decimal;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_postgres::{types::ToSql, NoTls};
use tracing::{debug, error, info, warn};

/// Maximum batch size before forcing a write.
const BATCH_SIZE: usize = 10;

/// Maximum time to hold candles before writing.
const BATCH_TIMEOUT: Duration = Duration::from_secs(1);

/// Maximum retry attempts for failed writes.
const MAX_RETRIES: u32 = 3;

/// Base delay for exponential backoff.
const RETRY_BASE_DELAY: Duration = Duration::from_millis(100);

/// Prepared statement for inserting candles.
const INSERT_CANDLE_SQL: &str = r#"
    INSERT INTO candles_1m (
        time, open, high, low, close, volume,
        quote_asset_volume, taker_buy_base_asset_volume,
        taker_buy_quote_asset_volume, number_of_trades
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    ON CONFLICT (time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        quote_asset_volume = EXCLUDED.quote_asset_volume,
        taker_buy_base_asset_volume = EXCLUDED.taker_buy_base_asset_volume,
        taker_buy_quote_asset_volume = EXCLUDED.taker_buy_quote_asset_volume,
        number_of_trades = EXCLUDED.number_of_trades
    WHERE
        candles_1m.open IS DISTINCT FROM EXCLUDED.open OR
        candles_1m.high IS DISTINCT FROM EXCLUDED.high OR
        candles_1m.low IS DISTINCT FROM EXCLUDED.low OR
        candles_1m.close IS DISTINCT FROM EXCLUDED.close OR
        candles_1m.volume IS DISTINCT FROM EXCLUDED.volume
"#;

/// SQL for logging data quality events.
const LOG_ERROR_SQL: &str = r#"
    INSERT INTO data_quality_logs (
        event_type, source, error_message, resolved
    ) VALUES ('error', 'system', $1, false)
"#;

/// Database writer for TimescaleDB.
pub struct DatabaseWriter {
    /// Connection pool
    pool: Pool,
    /// Write buffer
    buffer: Arc<Mutex<Vec<CandleData>>>,
    /// Channel for receiving write results
    result_tx: mpsc::Sender<WriteResult>,
    /// Connection status
    is_connected: Arc<AtomicBool>,
    /// Total candles written
    candles_written: Arc<AtomicU64>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Local backup buffer for DB failures
    backup_buffer: Arc<Mutex<Vec<CandleData>>>,
}

impl DatabaseWriter {
    /// Create a new database writer with connection pool.
    pub async fn new(
        config: &DatabaseConfig,
        result_tx: mpsc::Sender<WriteResult>,
    ) -> Result<Self> {
        let pg_config = config.to_pool_config();

        let mgr_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };

        let mgr = Manager::from_config(
            pg_config.get_pg_config()
                .map_err(|e| DataFeederError::config(format!("Invalid PG config: {}", e)))?,
            NoTls,
            mgr_config,
        );

        let pool = Pool::builder(mgr)
            .max_size(config.pool_max)
            .wait_timeout(Some(Duration::from_secs(10)))
            .create_timeout(Some(Duration::from_secs(10)))
            .recycle_timeout(Some(Duration::from_secs(10)))
            .runtime(deadpool::Runtime::Tokio1)
            .build()
            .map_err(|e| DataFeederError::config(format!("Failed to create pool: {}", e)))?;

        // Test connection
        let client = pool.get().await?;
        let _ = client.simple_query("SELECT 1").await?;
        info!("Database connection pool established");

        Ok(Self {
            pool,
            buffer: Arc::new(Mutex::new(Vec::with_capacity(BATCH_SIZE * 2))),
            result_tx,
            is_connected: Arc::new(AtomicBool::new(true)),
            candles_written: Arc::new(AtomicU64::new(0)),
            shutdown: Arc::new(AtomicBool::new(false)),
            backup_buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Check if the database is connected.
    pub fn is_connected(&self) -> bool {
        self.is_connected.load(Ordering::SeqCst)
    }

    /// Get the total number of candles written.
    pub fn candles_written(&self) -> u64 {
        self.candles_written.load(Ordering::SeqCst)
    }

    /// Signal the writer to shutdown.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Get the number of candles currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.buffer.lock().len() + self.backup_buffer.lock().len()
    }

    /// Run the database writer, consuming candles from the channel.
    pub async fn run(&self, mut candle_rx: mpsc::Receiver<CandleData>) -> Result<()> {
        let mut last_flush = Instant::now();

        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                // Final flush on shutdown
                self.flush_buffer().await;
                info!("Database writer shutting down");
                break;
            }

            // Try to receive with timeout
            let timeout = BATCH_TIMEOUT.saturating_sub(last_flush.elapsed());

            tokio::select! {
                candle = candle_rx.recv() => {
                    match candle {
                        Some(c) => {
                            self.buffer.lock().push(c);

                            // Check if we should flush
                            let should_flush = {
                                let buffer = self.buffer.lock();
                                buffer.len() >= BATCH_SIZE
                            };

                            if should_flush {
                                self.flush_buffer().await;
                                last_flush = Instant::now();
                            }
                        }
                        None => {
                            // Channel closed
                            self.flush_buffer().await;
                            info!("Candle channel closed");
                            break;
                        }
                    }
                }
                _ = tokio::time::sleep(timeout) => {
                    // Timeout - flush if we have any buffered candles
                    let has_candles = !self.buffer.lock().is_empty();
                    if has_candles {
                        self.flush_buffer().await;
                    }
                    last_flush = Instant::now();
                }
            }
        }

        Ok(())
    }

    /// Flush the buffer to the database.
    async fn flush_buffer(&self) {
        // Get candles from main buffer and backup buffer
        let mut candles: Vec<CandleData> = {
            let mut buffer = self.buffer.lock();
            let mut backup = self.backup_buffer.lock();

            let mut all_candles = Vec::with_capacity(buffer.len() + backup.len());
            all_candles.append(&mut *backup);
            all_candles.append(&mut *buffer);
            all_candles
        };

        if candles.is_empty() {
            return;
        }

        let count = candles.len();
        let start = Instant::now();

        // Try to write with retries
        let result = self.write_with_retry(&candles).await;

        match result {
            Ok(written) => {
                let duration = start.elapsed();
                self.candles_written.fetch_add(written as u64, Ordering::SeqCst);
                self.is_connected.store(true, Ordering::SeqCst);

                let result = WriteResult::Success {
                    count: written,
                    duration_ms: duration.as_millis() as u64,
                };

                debug!(
                    "Wrote {} candles in {:?}",
                    written, duration
                );

                let _ = self.result_tx.try_send(result);
            }
            Err(e) => {
                error!("Failed to write candles after retries: {}", e);
                self.is_connected.store(false, Ordering::SeqCst);

                // Move candles to backup buffer
                self.backup_buffer.lock().append(&mut candles);

                let result = WriteResult::Failed {
                    error: e.to_string(),
                    buffered: count,
                };

                let _ = self.result_tx.try_send(result);

                // Log error to database if possible
                let _ = self.log_error(&e.to_string()).await;
            }
        }
    }

    /// Write candles with retry logic.
    async fn write_with_retry(&self, candles: &[CandleData]) -> Result<usize> {
        let mut last_error = None;

        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                let delay = RETRY_BASE_DELAY * 2u32.pow(attempt - 1);
                warn!(
                    "Retry attempt {} after {:?} delay",
                    attempt + 1, delay
                );
                tokio::time::sleep(delay).await;
            }

            match self.write_batch(candles).await {
                Ok(count) => return Ok(count),
                Err(e) => {
                    warn!(
                        "Write attempt {} failed: {}",
                        attempt + 1, e
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| DataFeederError::Database(
            tokio_postgres::Error::__private_api_timeout()
        )))
    }

    /// Write a batch of candles to the database.
    async fn write_batch(&self, candles: &[CandleData]) -> Result<usize> {
        let client = self.pool.get().await?;
        let statement = client.prepare(INSERT_CANDLE_SQL).await?;

        let mut written = 0;

        for candle in candles {
            // Convert f64 to Decimal for database
            let open = Decimal::from_str(&format!("{:.8}", candle.open)).unwrap_or_default();
            let high = Decimal::from_str(&format!("{:.8}", candle.high)).unwrap_or_default();
            let low = Decimal::from_str(&format!("{:.8}", candle.low)).unwrap_or_default();
            let close = Decimal::from_str(&format!("{:.8}", candle.close)).unwrap_or_default();
            let volume = Decimal::from_str(&format!("{:.8}", candle.volume)).unwrap_or_default();
            
            let quote_vol = if candle.quote_asset_volume > 0.0 {
                Some(Decimal::from_str(&format!("{:.8}", candle.quote_asset_volume)).unwrap_or_default())
            } else {
                None
            };
            
            let taker_buy_base = if candle.taker_buy_base_asset_volume > 0.0 {
                Some(Decimal::from_str(&format!("{:.8}", candle.taker_buy_base_asset_volume)).unwrap_or_default())
            } else {
                None
            };
            
            let taker_buy_quote = if candle.taker_buy_quote_asset_volume > 0.0 {
                Some(Decimal::from_str(&format!("{:.8}", candle.taker_buy_quote_asset_volume)).unwrap_or_default())
            } else {
                None
            };

            let params: &[&(dyn ToSql + Sync)] = &[
                &candle.time,
                &open,
                &high,
                &low,
                &close,
                &volume,
                &quote_vol,
                &taker_buy_base,
                &taker_buy_quote,
                &candle.number_of_trades,
            ];

            match client.execute(&statement, params).await {
                Ok(_) => written += 1,
                Err(e) => {
                    warn!("Failed to insert candle at {}: {}", candle.time, e);
                }
            }
        }

        Ok(written)
    }

    /// Log an error to the data_quality_logs table.
    async fn log_error(&self, message: &str) -> Result<()> {
        match self.pool.get().await {
            Ok(client) => {
                let _ = client.execute(LOG_ERROR_SQL, &[&message]).await;
            }
            Err(_) => {
                // Can't connect to log, just skip
            }
        }
        Ok(())
    }

    /// Ping the database to check connection.
    pub async fn ping(&self) -> bool {
        match self.pool.get().await {
            Ok(client) => {
                client.simple_query("SELECT 1").await.is_ok()
            }
            Err(_) => false,
        }
    }

    /// Get the latest candle time from the database.
    pub async fn get_latest_candle_time(&self) -> Result<Option<DateTime<Utc>>> {
        let client = self.pool.get().await?;
        let row = client
            .query_opt("SELECT time FROM candles_1m ORDER BY time DESC LIMIT 1", &[])
            .await?;

        Ok(row.map(|r| r.get(0)))
    }

    /// Count total candles in the database.
    pub async fn count_candles(&self) -> Result<i64> {
        let client = self.pool.get().await?;
        let row = client
            .query_one(
                "SELECT COUNT(*) FROM candles_1m WHERE time > NOW() - INTERVAL '24 hours'",
                &[],
            )
            .await?;

        Ok(row.get(0))
    }
}

/// Extension trait to convert f64 to Option for nullable fields.
trait F64Ext {
    fn to_opt(self) -> Option<f64>;
}

impl F64Ext for f64 {
    fn to_opt(self) -> Option<f64> {
        if self > 0.0 {
            Some(self)
        } else {
            None
        }
    }
}

// Note: We need to add rust_decimal to handle NUMERIC types properly
// For now, the code assumes the database accepts f64 values

impl CandleData {
    /// Convert quote_asset_volume to Option for database.
    fn quote_volume_opt(&self) -> Option<f64> {
        if self.quote_asset_volume > 0.0 {
            Some(self.quote_asset_volume)
        } else {
            None
        }
    }

    /// Convert taker_buy_base_asset_volume to Option for database.
    fn taker_buy_base_opt(&self) -> Option<f64> {
        if self.taker_buy_base_asset_volume > 0.0 {
            Some(self.taker_buy_base_asset_volume)
        } else {
            None
        }
    }

    /// Convert taker_buy_quote_asset_volume to Option for database.
    fn taker_buy_quote_opt(&self) -> Option<f64> {
        if self.taker_buy_quote_asset_volume > 0.0 {
            Some(self.taker_buy_quote_asset_volume)
        } else {
            None
        }
    }
}
