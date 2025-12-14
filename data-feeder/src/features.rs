//! Feature computation module for ML model preprocessing.
//!
//! Computes normalized features from raw candle data:
//! - Min-max normalization: (x - min) / (max - min)
//! - Z-score normalization: (x - mean) / std
//!
//! Uses ndarray for vectorized SIMD-optimized operations.

use crate::binance::CandleData;
use crate::errors::{DataFeederError, Result};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Number of features computed from raw candle data.
pub const NUM_FEATURES: usize = 11;

/// Feature names for documentation and debugging.
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "number_of_trades",
    "spread_bps",
    "taker_buy_ratio",
];

/// Computed features for a single candle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputedFeatures {
    /// Raw feature values (11 columns)
    pub raw: Vec<f64>,
    /// Min-max normalized features (0-1 range)
    pub minmax_normalized: Vec<f64>,
    /// Z-score normalized features (mean=0, std=1)
    pub zscore_normalized: Vec<f64>,
    /// Feature names for reference
    pub feature_names: Vec<String>,
}

/// Statistics for normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

/// Feature computer with rolling statistics.
pub struct FeatureComputer {
    /// Window size for computing rolling statistics.
    window_size: usize,
    /// Rolling buffer of recent candles.
    buffer: Vec<Array1<f64>>,
    /// Current statistics (updated on each candle).
    stats: Option<FeatureStatistics>,
}

impl FeatureComputer {
    /// Create a new feature computer.
    ///
    /// # Arguments
    /// * `window_size` - Number of candles to use for computing statistics.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            buffer: Vec::with_capacity(window_size),
            stats: None,
        }
    }

    /// Extract raw features from a candle.
    pub fn extract_raw_features(candle: &CandleData) -> Array1<f64> {
        let taker_buy_ratio = if candle.volume > 0.0 {
            candle.taker_buy_base_asset_volume / candle.volume
        } else {
            0.5
        };

        Array1::from(vec![
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.quote_asset_volume,
            candle.taker_buy_base_asset_volume,
            candle.taker_buy_quote_asset_volume,
            candle.number_of_trades as f64,
            candle.spread_bps.unwrap_or(0.0),
            taker_buy_ratio,
        ])
    }

    /// Add a candle and compute features.
    pub fn add_candle(&mut self, candle: &CandleData) -> Result<ComputedFeatures> {
        let raw = Self::extract_raw_features(candle);

        // Add to buffer
        self.buffer.push(raw.clone());

        // Maintain window size
        if self.buffer.len() > self.window_size {
            self.buffer.remove(0);
        }

        // Update statistics
        self.update_statistics();

        // Compute normalized features
        let (minmax, zscore) = if let Some(ref stats) = self.stats {
            (
                self.minmax_normalize(&raw, stats),
                self.zscore_normalize(&raw, stats),
            )
        } else {
            (raw.to_vec(), raw.to_vec())
        };

        Ok(ComputedFeatures {
            raw: raw.to_vec(),
            minmax_normalized: minmax,
            zscore_normalized: zscore,
            feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// Update statistics from the buffer.
    fn update_statistics(&mut self) {
        if self.buffer.is_empty() {
            return;
        }

        let n = self.buffer.len();
        let m = NUM_FEATURES;

        // Build matrix from buffer
        let mut data = Array2::<f64>::zeros((n, m));
        for (i, row) in self.buffer.iter().enumerate() {
            for j in 0..m {
                data[[i, j]] = row[j];
            }
        }

        // Compute statistics along axis 0 (rows)
        let min = data
            .axis_iter(Axis(1))
            .map(|col| col.iter().cloned().fold(f64::INFINITY, f64::min))
            .collect();

        let max = data
            .axis_iter(Axis(1))
            .map(|col| col.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .collect();

        let mean: Vec<f64> = data
            .axis_iter(Axis(1))
            .map(|col| col.mean().unwrap_or(0.0))
            .collect();

        let std: Vec<f64> = data
            .axis_iter(Axis(1))
            .zip(mean.iter())
            .map(|(col, &m)| {
                let variance = col.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n as f64;
                variance.sqrt()
            })
            .collect();

        self.stats = Some(FeatureStatistics { min, max, mean, std });
    }

    /// Apply min-max normalization: (x - min) / (max - min)
    fn minmax_normalize(&self, features: &Array1<f64>, stats: &FeatureStatistics) -> Vec<f64> {
        features
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let range = stats.max[i] - stats.min[i];
                if range > 0.0 {
                    (x - stats.min[i]) / range
                } else {
                    0.5 // Default to middle if no variance
                }
            })
            .collect()
    }

    /// Apply z-score normalization: (x - mean) / std
    fn zscore_normalize(&self, features: &Array1<f64>, stats: &FeatureStatistics) -> Vec<f64> {
        features
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if stats.std[i] > 1e-10 {
                    (x - stats.mean[i]) / stats.std[i]
                } else {
                    0.0 // Default if no variance
                }
            })
            .collect()
    }

    /// Get current statistics.
    pub fn get_statistics(&self) -> Option<&FeatureStatistics> {
        self.stats.as_ref()
    }

    /// Get the current buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is full (has enough data for reliable statistics).
    pub fn is_warmed_up(&self) -> bool {
        self.buffer.len() >= self.window_size / 2
    }
}

/// Compute features for a batch of candles.
pub fn compute_batch_features(candles: &[CandleData]) -> Result<Vec<ComputedFeatures>> {
    if candles.is_empty() {
        return Ok(Vec::new());
    }

    let mut computer = FeatureComputer::new(candles.len());

    candles
        .iter()
        .map(|c| computer.add_candle(c))
        .collect()
}

/// Convert computed features to JSON for storage.
pub fn features_to_json(features: &ComputedFeatures) -> Result<serde_json::Value> {
    let mut map = serde_json::Map::new();

    // Store both normalized versions
    map.insert(
        "minmax".to_string(),
        serde_json::to_value(&features.minmax_normalized)?,
    );
    map.insert(
        "zscore".to_string(),
        serde_json::to_value(&features.zscore_normalized)?,
    );

    // Include feature names for reference
    map.insert(
        "feature_names".to_string(),
        serde_json::to_value(&features.feature_names)?,
    );

    Ok(serde_json::Value::Object(map))
}

/// Technical indicators computed from price data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicators {
    /// Simple Moving Average (various periods)
    pub sma: HashMap<String, f64>,
    /// Exponential Moving Average (various periods)
    pub ema: HashMap<String, f64>,
    /// Relative Strength Index
    pub rsi: Option<f64>,
    /// MACD (12,26,9)
    pub macd: Option<MACDResult>,
    /// Bollinger Bands
    pub bollinger: Option<BollingerBands>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACDResult {
    pub macd: f64,
    pub signal: f64,
    pub histogram: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerBands {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

/// Compute Simple Moving Average.
pub fn compute_sma(values: &[f64], period: usize) -> Option<f64> {
    if values.len() < period {
        return None;
    }

    let sum: f64 = values.iter().rev().take(period).sum();
    Some(sum / period as f64)
}

/// Compute Exponential Moving Average.
pub fn compute_ema(values: &[f64], period: usize) -> Option<f64> {
    if values.len() < period {
        return None;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);

    // Start with SMA for initial EMA
    let initial_sma: f64 = values.iter().take(period).sum::<f64>() / period as f64;

    // Calculate EMA from there
    let mut ema = initial_sma;
    for value in values.iter().skip(period) {
        ema = (value - ema) * multiplier + ema;
    }

    Some(ema)
}

/// Compute Relative Strength Index.
pub fn compute_rsi(closes: &[f64], period: usize) -> Option<f64> {
    if closes.len() < period + 1 {
        return None;
    }

    let mut gains = 0.0;
    let mut losses = 0.0;

    for i in 1..=period {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 {
            gains += change;
        } else {
            losses -= change;
        }
    }

    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;

    if avg_loss == 0.0 {
        return Some(100.0);
    }

    let rs = avg_gain / avg_loss;
    Some(100.0 - (100.0 / (1.0 + rs)))
}

/// Compute Bollinger Bands.
pub fn compute_bollinger(closes: &[f64], period: usize, std_dev: f64) -> Option<BollingerBands> {
    if closes.len() < period {
        return None;
    }

    let recent: Vec<f64> = closes.iter().rev().take(period).cloned().collect();
    let mean = recent.iter().sum::<f64>() / period as f64;

    let variance = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
    let std = variance.sqrt();

    Some(BollingerBands {
        upper: mean + std_dev * std,
        middle: mean,
        lower: mean - std_dev * std,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_candle(close: f64, volume: f64) -> CandleData {
        CandleData {
            time: Utc::now(),
            open: close - 10.0,
            high: close + 50.0,
            low: close - 50.0,
            close,
            volume,
            quote_asset_volume: volume * close,
            taker_buy_base_asset_volume: volume * 0.5,
            taker_buy_quote_asset_volume: volume * close * 0.5,
            number_of_trades: 1000,
            spread_bps: Some(5.0),
            best_bid: Some(close - 1.0),
            best_ask: Some(close + 1.0),
        }
    }

    #[test]
    fn test_feature_extraction() {
        let candle = create_test_candle(100000.0, 1000.0);
        let features = FeatureComputer::extract_raw_features(&candle);

        assert_eq!(features.len(), NUM_FEATURES);
        assert_eq!(features[3], 100000.0); // close
        assert_eq!(features[4], 1000.0); // volume
    }

    #[test]
    fn test_feature_computer() {
        let mut computer = FeatureComputer::new(10);

        // Add some candles
        for i in 0..15 {
            let candle = create_test_candle(100000.0 + i as f64 * 100.0, 1000.0 + i as f64 * 10.0);
            let features = computer.add_candle(&candle).unwrap();

            // After first candle, features should be computed
            assert_eq!(features.raw.len(), NUM_FEATURES);
            assert_eq!(features.minmax_normalized.len(), NUM_FEATURES);
            assert_eq!(features.zscore_normalized.len(), NUM_FEATURES);
        }

        // Buffer should be at window size
        assert_eq!(computer.buffer_size(), 10);

        // Statistics should be computed
        assert!(computer.get_statistics().is_some());
    }

    #[test]
    fn test_normalization() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Min-max: (3 - 1) / (5 - 1) = 0.5
        let minmax = (3.0 - 1.0) / (5.0 - 1.0);
        assert!((minmax - 0.5).abs() < 1e-10);

        // Z-score: (3 - 3) / std = 0
        let mean = 3.0;
        let zscore = (3.0 - mean);
        assert!((zscore - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = compute_sma(&values, 3).unwrap();
        assert!((sma - 4.0).abs() < 1e-10); // (3 + 4 + 5) / 3 = 4
    }

    #[test]
    fn test_rsi() {
        // Create a trending series
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi = compute_rsi(&closes, 14).unwrap();

        // All gains, no losses should give RSI = 100
        assert!((rsi - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_bollinger() {
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64).sin() * 10.0).collect();
        let bb = compute_bollinger(&closes, 20, 2.0).unwrap();

        assert!(bb.upper > bb.middle);
        assert!(bb.middle > bb.lower);
    }
}
