//! Candle Loader Module
//!
//! Queries candles_1m table from TimescaleDB.

use chrono::{DateTime, Utc};
use rust_decimal::prelude::*;
use std::sync::Arc;
use thiserror::Error;
use tokio_postgres::Client;
use tracing::{debug, error};

/// Candle loader errors
#[derive(Error, Debug)]
pub enum CandleLoaderError {
    #[error("Database error: {0}")]
    Database(#[from] tokio_postgres::Error),

    #[error("Not enough candles: need {needed}, got {got}")]
    NotEnoughCandles { needed: usize, got: usize },

    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Single candle record from candles_1m table
#[derive(Clone, Debug)]
pub struct Candle {
    pub time: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_asset_volume: f64,
    pub taker_buy_base_asset_volume: f64,
    pub taker_buy_quote_asset_volume: f64,
    pub number_of_trades: f64,
    pub spread_bps: f64,
    pub taker_buy_ratio: f64,
    pub mid_price: f64,
}

impl Candle {
    /// Convert Decimal to f64 safely
    fn decimal_to_f64(decimal: Option<Decimal>) -> f64 {
        decimal.and_then(|d| d.to_f64()).unwrap_or(0.0)
    }

    /// Create from database row
    pub fn from_row(row: &tokio_postgres::Row) -> Result<Self, CandleLoaderError> {
        let time: DateTime<Utc> = row.get("time");

        let open = Self::decimal_to_f64(row.get::<_, Option<Decimal>>("open"));
        let high = Self::decimal_to_f64(row.get::<_, Option<Decimal>>("high"));
        let low = Self::decimal_to_f64(row.get::<_, Option<Decimal>>("low"));
        let close = Self::decimal_to_f64(row.get::<_, Option<Decimal>>("close"));
        let volume = Self::decimal_to_f64(row.get::<_, Option<Decimal>>("volume"));

        let quote_asset_volume =
            Self::decimal_to_f64(row.get::<_, Option<Decimal>>("quote_asset_volume"));
        let taker_buy_base_asset_volume =
            Self::decimal_to_f64(row.get::<_, Option<Decimal>>("taker_buy_base_asset_volume"));
        let taker_buy_quote_asset_volume =
            Self::decimal_to_f64(row.get::<_, Option<Decimal>>("taker_buy_quote_asset_volume"));

        let number_of_trades: i64 = row.get::<_, Option<i64>>("number_of_trades").unwrap_or(0);

        let spread_bps = Self::decimal_to_f64(row.get::<_, Option<Decimal>>("spread_bps"));
        let taker_buy_ratio =
            Self::decimal_to_f64(row.get::<_, Option<Decimal>>("taker_buy_ratio"));
        let mid_price = Self::decimal_to_f64(row.get::<_, Option<Decimal>>("mid_price"));

        Ok(Self {
            time,
            open,
            high,
            low,
            close,
            volume,
            quote_asset_volume,
            taker_buy_base_asset_volume,
            taker_buy_quote_asset_volume,
            number_of_trades: number_of_trades as f64,
            spread_bps,
            taker_buy_ratio,
            mid_price,
        })
    }
}

/// Loads candles from database
pub struct CandleLoader {
    client: Arc<Client>,
}

impl CandleLoader {
    /// Create new candle loader
    pub fn new(client: Arc<Client>) -> Self {
        Self { client }
    }

    /// Fetch latest N candles, returned in chronological order (oldest first)
    pub async fn fetch_latest_candles(
        &self,
        count: usize,
    ) -> Result<Vec<Candle>, CandleLoaderError> {
        let query = r#"
            SELECT 
                time,
                open,
                high,
                low,
                close,
                volume,
                COALESCE(quote_asset_volume, volume * close) as quote_asset_volume,
                COALESCE(taker_buy_base_asset_volume, volume * 0.5) as taker_buy_base_asset_volume,
                COALESCE(taker_buy_quote_asset_volume, volume * close * 0.5) as taker_buy_quote_asset_volume,
                COALESCE(number_of_trades, 0) as number_of_trades,
                COALESCE(spread_bps, 0) as spread_bps,
                COALESCE(taker_buy_ratio, 0.5) as taker_buy_ratio,
                COALESCE(mid_price, (high + low) / 2) as mid_price
            FROM candles_1m
            ORDER BY time DESC
            LIMIT $1
        "#;

        let rows = self.client.query(query, &[&(count as i64)]).await?;

        if rows.len() < count {
            debug!(
                "Fetched {} candles (requested {})",
                rows.len(),
                count
            );
        }

        // Convert rows to Candles and reverse to chronological order
        let mut candles: Vec<Candle> = rows
            .iter()
            .filter_map(|row| Candle::from_row(row).ok())
            .collect();

        candles.reverse(); // Oldest first

        Ok(candles)
    }

    /// Fetch candles since a specific timestamp
    pub async fn fetch_candles_since(
        &self,
        since: DateTime<Utc>,
    ) -> Result<Vec<Candle>, CandleLoaderError> {
        let query = r#"
            SELECT 
                time,
                open,
                high,
                low,
                close,
                volume,
                COALESCE(quote_asset_volume, volume * close) as quote_asset_volume,
                COALESCE(taker_buy_base_asset_volume, volume * 0.5) as taker_buy_base_asset_volume,
                COALESCE(taker_buy_quote_asset_volume, volume * close * 0.5) as taker_buy_quote_asset_volume,
                COALESCE(number_of_trades, 0) as number_of_trades,
                COALESCE(spread_bps, 0) as spread_bps,
                COALESCE(taker_buy_ratio, 0.5) as taker_buy_ratio,
                COALESCE(mid_price, (high + low) / 2) as mid_price
            FROM candles_1m
            WHERE time >= $1
            ORDER BY time ASC
        "#;

        let rows = self.client.query(query, &[&since]).await?;

        let candles: Vec<Candle> = rows
            .iter()
            .filter_map(|row| Candle::from_row(row).ok())
            .collect();

        Ok(candles)
    }

    /// Fetch candles in a time range
    pub async fn fetch_candles_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<Candle>, CandleLoaderError> {
        let query = r#"
            SELECT 
                time,
                open,
                high,
                low,
                close,
                volume,
                COALESCE(quote_asset_volume, volume * close) as quote_asset_volume,
                COALESCE(taker_buy_base_asset_volume, volume * 0.5) as taker_buy_base_asset_volume,
                COALESCE(taker_buy_quote_asset_volume, volume * close * 0.5) as taker_buy_quote_asset_volume,
                COALESCE(number_of_trades, 0) as number_of_trades,
                COALESCE(spread_bps, 0) as spread_bps,
                COALESCE(taker_buy_ratio, 0.5) as taker_buy_ratio,
                COALESCE(mid_price, (high + low) / 2) as mid_price
            FROM candles_1m
            WHERE time >= $1 AND time < $2
            ORDER BY time ASC
        "#;

        let rows = self.client.query(query, &[&start, &end]).await?;

        let candles: Vec<Candle> = rows
            .iter()
            .filter_map(|row| Candle::from_row(row).ok())
            .collect();

        Ok(candles)
    }

    /// Get the latest candle timestamp
    pub async fn get_latest_time(&self) -> Result<Option<DateTime<Utc>>, CandleLoaderError> {
        let row = self
            .client
            .query_opt("SELECT MAX(time) as latest FROM candles_1m", &[])
            .await?;

        Ok(row.and_then(|r| r.get::<_, Option<DateTime<Utc>>>("latest")))
    }

    /// Count total candles in database
    pub async fn count_candles(&self) -> Result<i64, CandleLoaderError> {
        let row = self
            .client
            .query_one("SELECT COUNT(*) as count FROM candles_1m", &[])
            .await?;

        Ok(row.get::<_, i64>("count"))
    }
}

/// Extract arrays from candles for feature computation
pub fn extract_arrays(candles: &[Candle]) -> CandleArrays {
    let n = candles.len();
    
    CandleArrays {
        times: candles.iter().map(|c| c.time).collect(),
        opens: candles.iter().map(|c| c.open).collect(),
        highs: candles.iter().map(|c| c.high).collect(),
        lows: candles.iter().map(|c| c.low).collect(),
        closes: candles.iter().map(|c| c.close).collect(),
        volumes: candles.iter().map(|c| c.volume).collect(),
        quote_asset_volumes: candles.iter().map(|c| c.quote_asset_volume).collect(),
        taker_buy_base_asset_volumes: candles.iter().map(|c| c.taker_buy_base_asset_volume).collect(),
        taker_buy_quote_asset_volumes: candles.iter().map(|c| c.taker_buy_quote_asset_volume).collect(),
        number_of_trades: candles.iter().map(|c| c.number_of_trades).collect(),
        spread_bps: candles.iter().map(|c| c.spread_bps).collect(),
        taker_buy_ratio: candles.iter().map(|c| c.taker_buy_ratio).collect(),
    }
}

/// Arrays extracted from candles for feature computation
pub struct CandleArrays {
    pub times: Vec<DateTime<Utc>>,
    pub opens: Vec<f64>,
    pub highs: Vec<f64>,
    pub lows: Vec<f64>,
    pub closes: Vec<f64>,
    pub volumes: Vec<f64>,
    pub quote_asset_volumes: Vec<f64>,
    pub taker_buy_base_asset_volumes: Vec<f64>,
    pub taker_buy_quote_asset_volumes: Vec<f64>,
    pub number_of_trades: Vec<f64>,
    pub spread_bps: Vec<f64>,
    pub taker_buy_ratio: Vec<f64>,
}
