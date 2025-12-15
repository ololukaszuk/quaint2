//! Feature Computation Module
//!
//! CRITICAL: This module computes EXACTLY the same 27 features as the Python
//! implementation in ml-layer-1/training/feature_engineering.py.
//!
//! Any changes here MUST be synchronized with the Python version.
//!
//! Features (27 total):
//! - 11 Raw: from candles_1m table
//! - 16 Derived: computed from raw

use ndarray::{Array1, Array2, Axis};
use std::f64;

/// Number of features computed
pub const NUM_FEATURES: usize = 27;

/// Sequence length for model input
pub const SEQUENCE_LENGTH: usize = 60;

/// Number of prediction horizons
pub const NUM_HORIZONS: usize = 5;

/// Feature names for documentation
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    // Raw features (0-10)
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "number_of_trades",
    "spread_bps",
    "taker_buy_ratio",
    // Derived features (11-26)
    "log_return_1m",
    "log_return_5m",
    "log_return_15m",
    "volatility_5m",
    "volatility_15m",
    "volatility_30m",
    "sma_5_norm",
    "sma_15_norm",
    "sma_30_norm",
    "ema_5_norm",
    "ema_15_norm",
    "ema_30_norm",
    "rsi_14",
    "volume_sma_ratio",
    "vwap_deviation",
    "price_position",
];

/// Feature computer for computing all 27 features from raw OHLCV data.
pub struct FeatureComputer;

impl FeatureComputer {
    /// Compute all 27 features from 11 raw OHLCV fields.
    ///
    /// CRITICAL: Must exactly match Python's compute_extended_features()
    ///
    /// # Arguments
    /// * `opens` - Open prices (N,)
    /// * `highs` - High prices (N,)
    /// * `lows` - Low prices (N,)
    /// * `closes` - Close prices (N,)
    /// * `volumes` - Volumes (N,)
    /// * `quote_asset_volumes` - Quote asset volumes (N,)
    /// * `taker_buy_base_asset_volumes` - Taker buy base volumes (N,)
    /// * `taker_buy_quote_asset_volumes` - Taker buy quote volumes (N,)
    /// * `number_of_trades` - Number of trades (N,)
    /// * `spread_bps` - Spread in basis points (N,)
    /// * `taker_buy_ratio` - Taker buy ratio (N,)
    ///
    /// # Returns
    /// Feature matrix (N, 27)
    pub fn compute_extended_features(
        opens: &[f64],
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
        volumes: &[f64],
        quote_asset_volumes: &[f64],
        taker_buy_base_asset_volumes: &[f64],
        taker_buy_quote_asset_volumes: &[f64],
        number_of_trades: &[f64],
        spread_bps: &[f64],
        taker_buy_ratio: &[f64],
    ) -> Vec<Vec<f64>> {
        let n = closes.len();
        let mut features = vec![vec![0.0; NUM_FEATURES]; n];

        // ====================================================================
        // RAW FEATURES (0-10): Direct from input
        // ====================================================================
        for i in 0..n {
            features[i][0] = opens[i];
            features[i][1] = highs[i];
            features[i][2] = lows[i];
            features[i][3] = closes[i];
            features[i][4] = volumes[i];
            features[i][5] = quote_asset_volumes[i];
            features[i][6] = taker_buy_base_asset_volumes[i];
            features[i][7] = taker_buy_quote_asset_volumes[i];
            features[i][8] = number_of_trades[i];
            features[i][9] = spread_bps[i];
            features[i][10] = taker_buy_ratio[i];
        }

        // ====================================================================
        // DERIVED FEATURES (11-26): Computed from raw
        // ====================================================================

        // Log returns (indices 11-13)
        let log_return_1m = Self::compute_log_returns(closes, 1);
        let log_return_5m = Self::compute_log_returns(closes, 5);
        let log_return_15m = Self::compute_log_returns(closes, 15);

        for i in 0..n {
            features[i][11] = log_return_1m[i];
            features[i][12] = log_return_5m[i];
            features[i][13] = log_return_15m[i];
        }

        // Volatility (indices 14-16)
        let volatility_5m = Self::compute_volatility(&log_return_1m, 5);
        let volatility_15m = Self::compute_volatility(&log_return_1m, 15);
        let volatility_30m = Self::compute_volatility(&log_return_1m, 30);

        for i in 0..n {
            features[i][14] = volatility_5m[i];
            features[i][15] = volatility_15m[i];
            features[i][16] = volatility_30m[i];
        }

        // SMA normalized (indices 17-19)
        let sma_5 = Self::compute_sma(closes, 5);
        let sma_15 = Self::compute_sma(closes, 15);
        let sma_30 = Self::compute_sma(closes, 30);

        for i in 0..n {
            features[i][17] = if closes[i] > 0.0 {
                (closes[i] - sma_5[i]) / closes[i]
            } else {
                0.0
            };
            features[i][18] = if closes[i] > 0.0 {
                (closes[i] - sma_15[i]) / closes[i]
            } else {
                0.0
            };
            features[i][19] = if closes[i] > 0.0 {
                (closes[i] - sma_30[i]) / closes[i]
            } else {
                0.0
            };
        }

        // EMA normalized (indices 20-22)
        let ema_5 = Self::compute_ema(closes, 5);
        let ema_15 = Self::compute_ema(closes, 15);
        let ema_30 = Self::compute_ema(closes, 30);

        for i in 0..n {
            features[i][20] = if closes[i] > 0.0 {
                (closes[i] - ema_5[i]) / closes[i]
            } else {
                0.0
            };
            features[i][21] = if closes[i] > 0.0 {
                (closes[i] - ema_15[i]) / closes[i]
            } else {
                0.0
            };
            features[i][22] = if closes[i] > 0.0 {
                (closes[i] - ema_30[i]) / closes[i]
            } else {
                0.0
            };
        }

        // RSI (index 23)
        let rsi = Self::compute_rsi(closes, 14);
        for i in 0..n {
            features[i][23] = rsi[i];
        }

        // Volume SMA ratio (index 24)
        let volume_sma_20 = Self::compute_sma(volumes, 20);
        for i in 0..n {
            features[i][24] = if volume_sma_20[i] > 0.0 {
                volumes[i] / volume_sma_20[i]
            } else {
                1.0
            };
        }

        // VWAP deviation (index 25)
        let vwap = Self::compute_vwap(highs, lows, closes, volumes);
        for i in 0..n {
            features[i][25] = if closes[i] > 0.0 {
                (closes[i] - vwap[i]) / closes[i]
            } else {
                0.0
            };
        }

        // Price position (index 26)
        for i in 0..n {
            let range = highs[i] - lows[i];
            features[i][26] = if range > 0.0 {
                (closes[i] - lows[i]) / range
            } else {
                0.5
            };
        }

        features
    }

    /// Compute log returns: ln(close[t] / close[t-period])
    pub fn compute_log_returns(closes: &[f64], period: usize) -> Vec<f64> {
        let n = closes.len();
        let mut result = vec![0.0; n];

        if n > period {
            for i in period..n {
                if closes[i - period] > 0.0 {
                    result[i] = (closes[i] / closes[i - period]).ln();
                }
            }
        }

        result
    }

    /// Compute rolling volatility (standard deviation of returns)
    pub fn compute_volatility(returns: &[f64], period: usize) -> Vec<f64> {
        let n = returns.len();
        let mut result = vec![0.0; n];

        if n >= period {
            for i in (period - 1)..n {
                let window = &returns[(i - period + 1)..=i];
                result[i] = Self::std_dev(window);
            }
        }

        result
    }

    /// Compute Simple Moving Average
    pub fn compute_sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        // Compute cumulative sum
        let mut cumsum = vec![0.0; n];
        cumsum[0] = data[0];
        for i in 1..n {
            cumsum[i] = cumsum[i - 1] + data[i];
        }

        // First period-1 values: use expanding window
        for i in 0..std::cmp::min(period - 1, n) {
            result[i] = cumsum[i] / (i + 1) as f64;
        }

        // Rest: proper rolling window
        if n >= period {
            for i in (period - 1)..n {
                let sum = if i >= period {
                    cumsum[i] - cumsum[i - period]
                } else {
                    cumsum[i]
                };
                result[i] = sum / period as f64;
            }
        }

        result
    }

    /// Compute Exponential Moving Average
    pub fn compute_ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut result = vec![0.0; n];

        result[0] = data[0];
        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Compute Relative Strength Index (normalized to 0-1)
    pub fn compute_rsi(closes: &[f64], period: usize) -> Vec<f64> {
        let n = closes.len();
        let mut result = vec![0.5; n]; // Default neutral

        if n < period + 1 {
            return result;
        }

        // Calculate price changes
        let mut gains = vec![0.0; n - 1];
        let mut losses = vec![0.0; n - 1];

        for i in 0..(n - 1) {
            let delta = closes[i + 1] - closes[i];
            if delta > 0.0 {
                gains[i] = delta;
            } else {
                losses[i] = -delta;
            }
        }

        // Calculate average gain/loss using smoothed average
        let mut avg_gain = vec![0.0; n - 1];
        let mut avg_loss = vec![0.0; n - 1];

        // Initial average
        if period - 1 < gains.len() {
            let initial_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
            let initial_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;
            avg_gain[period - 1] = initial_gain;
            avg_loss[period - 1] = initial_loss;
        }

        // Smoothed averages
        for i in period..gains.len() {
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) as f64 + losses[i]) / period as f64;
        }

        // Calculate RSI
        for i in (period - 1)..gains.len() {
            if avg_loss[i] == 0.0 {
                result[i + 1] = 1.0; // All gains
            } else {
                let rs = avg_gain[i] / avg_loss[i];
                result[i + 1] = 1.0 - (1.0 / (1.0 + rs)); // Normalized to 0-1
            }
        }

        result
    }

    /// Compute Volume Weighted Average Price
    pub fn compute_vwap(highs: &[f64], lows: &[f64], closes: &[f64], volumes: &[f64]) -> Vec<f64> {
        let n = closes.len();
        let mut result = vec![0.0; n];

        let mut cum_tp_vol = 0.0;
        let mut cum_vol = 0.0;

        for i in 0..n {
            let typical_price = (highs[i] + lows[i] + closes[i]) / 3.0;
            cum_tp_vol += typical_price * volumes[i];
            cum_vol += volumes[i];

            result[i] = if cum_vol > 0.0 {
                cum_tp_vol / cum_vol
            } else {
                closes[i]
            };
        }

        result
    }

    /// Compute standard deviation
    fn std_dev(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    }
}

/// Normalization parameters loaded from JSON
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl NormalizationParams {
    /// Load from JSON file
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        let data: serde_json::Value = serde_json::from_str(json_str)?;

        let mean: Vec<f64> = data["mean"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        let std: Vec<f64> = data["std"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        Ok(Self { mean, std })
    }

    /// Normalize features using Z-score
    pub fn normalize(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let std = if i < self.std.len() && self.std[i] > 1e-10 {
                            self.std[i]
                        } else {
                            1.0
                        };
                        let mean = if i < self.mean.len() {
                            self.mean[i]
                        } else {
                            0.0
                        };
                        let normalized = (x - mean) / std;
                        // Clip to [-10, 10]
                        normalized.max(-10.0).min(10.0)
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = FeatureComputer::compute_sma(&data, 3);
        assert!((sma[4] - 4.0).abs() < 1e-10); // (3 + 4 + 5) / 3 = 4
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = FeatureComputer::compute_ema(&data, 3);
        assert!(ema[4] > 4.0); // EMA should be slightly above 4 due to recent weighting
    }

    #[test]
    fn test_log_returns() {
        let closes = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let returns = FeatureComputer::compute_log_returns(&closes, 1);
        assert!(returns[0] == 0.0); // First value should be 0
        assert!((returns[1] - (101.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_rsi() {
        // Create a trending series (all up)
        let closes: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi = FeatureComputer::compute_rsi(&closes, 14);
        // All gains, no losses should give RSI close to 1.0
        assert!(rsi[19] > 0.9);
    }

    #[test]
    fn test_feature_count() {
        let n = 10;
        let opens = vec![100.0; n];
        let highs = vec![101.0; n];
        let lows = vec![99.0; n];
        let closes = vec![100.5; n];
        let volumes = vec![1000.0; n];
        let quote_asset_volumes = vec![100000.0; n];
        let taker_buy_base = vec![500.0; n];
        let taker_buy_quote = vec![50000.0; n];
        let num_trades = vec![100.0; n];
        let spread_bps = vec![5.0; n];
        let taker_buy_ratio = vec![0.5; n];

        let features = FeatureComputer::compute_extended_features(
            &opens,
            &highs,
            &lows,
            &closes,
            &volumes,
            &quote_asset_volumes,
            &taker_buy_base,
            &taker_buy_quote,
            &num_trades,
            &spread_bps,
            &taker_buy_ratio,
        );

        assert_eq!(features.len(), n);
        assert_eq!(features[0].len(), NUM_FEATURES);
    }
}
