"""
Feature Engineering Module

CRITICAL: This module defines the exact 27 features used by both
Python (training) and Rust (inference). Any changes here MUST be
reflected in ml-layer-1/inference/src/features.rs.

Features:
- 11 Raw: from candles_1m table
- 16 Derived: computed from raw

Total: 27 features per timestep
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings

from .config import (
    RAW_FEATURES,
    DERIVED_FEATURES,
    ALL_FEATURES,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
)


def compute_log_returns(closes: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Compute log returns: ln(close[t] / close[t-period])
    
    Args:
        closes: Array of close prices (length N)
        period: Lookback period
        
    Returns:
        Array of log returns (length N, first `period` values are 0)
    """
    result = np.zeros_like(closes)
    if len(closes) > period:
        result[period:] = np.log(closes[period:] / closes[:-period])
    return result


def compute_volatility(returns: np.ndarray, period: int) -> np.ndarray:
    """
    Compute rolling volatility (standard deviation of returns).
    
    Args:
        returns: Array of log returns
        period: Rolling window size
        
    Returns:
        Array of volatility values (first `period-1` values are 0)
    """
    result = np.zeros_like(returns)
    if len(returns) >= period:
        for i in range(period - 1, len(returns)):
            result[i] = np.std(returns[i - period + 1:i + 1])
    return result


def compute_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Compute Simple Moving Average.
    
    Args:
        data: Input array
        period: Window size
        
    Returns:
        SMA array (first `period-1` values use available data)
    """
    result = np.zeros_like(data)
    cumsum = np.cumsum(data)
    
    # First period-1 values: use expanding window
    for i in range(min(period - 1, len(data))):
        result[i] = cumsum[i] / (i + 1)
    
    # Rest: proper rolling window
    if len(data) >= period:
        result[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])[1:]) / period
    
    return result


def compute_ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Compute Exponential Moving Average.
    
    Args:
        data: Input array
        period: Window size for smoothing factor
        
    Returns:
        EMA array
    """
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data)
    
    if len(data) == 0:
        return result
    
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    
    return result


def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Compute Relative Strength Index (normalized to 0-1).
    
    Args:
        closes: Array of close prices
        period: RSI period (default 14)
        
    Returns:
        RSI array (0-1 range, first `period` values may be 0.5)
    """
    result = np.full_like(closes, 0.5)  # Default neutral
    
    if len(closes) < period + 1:
        return result
    
    # Calculate price changes
    deltas = np.diff(closes)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gain/loss using EMA
    avg_gain = np.zeros(len(gains))
    avg_loss = np.zeros(len(losses))
    
    # Initial average
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])
    
    # Smoothed averages
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    
    # Calculate RSI
    for i in range(period - 1, len(gains)):
        if avg_loss[i] == 0:
            result[i + 1] = 1.0  # All gains
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i + 1] = 1.0 - (1.0 / (1.0 + rs))  # Normalized to 0-1
    
    return result


def compute_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """
    Compute Volume Weighted Average Price (session-based, using rolling window).
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volumes
        
    Returns:
        VWAP array
    """
    typical_price = (highs + lows + closes) / 3
    
    # Use cumulative for session VWAP (reset daily would need timestamps)
    cum_tp_vol = np.cumsum(typical_price * volumes)
    cum_vol = np.cumsum(volumes)
    
    # Avoid division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, closes)
    
    return vwap


def compute_extended_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    quote_asset_volumes: np.ndarray,
    taker_buy_base_asset_volumes: np.ndarray,
    taker_buy_quote_asset_volumes: np.ndarray,
    number_of_trades: np.ndarray,
    spread_bps: np.ndarray,
    taker_buy_ratio: np.ndarray,
) -> np.ndarray:
    """
    Compute all 27 features from 11 raw OHLCV fields.
    
    CRITICAL: This function defines the exact feature computation that
    must be replicated in Rust (ml-layer-1/inference/src/features.rs).
    
    Args:
        opens: Open prices (N,)
        highs: High prices (N,)
        lows: Low prices (N,)
        closes: Close prices (N,)
        volumes: Volumes (N,)
        quote_asset_volumes: Quote asset volumes (N,)
        taker_buy_base_asset_volumes: Taker buy base volumes (N,)
        taker_buy_quote_asset_volumes: Taker buy quote volumes (N,)
        number_of_trades: Number of trades (N,)
        spread_bps: Spread in basis points (N,)
        taker_buy_ratio: Taker buy ratio (N,)
        
    Returns:
        Feature matrix (N, 27)
    """
    n = len(closes)
    features = np.zeros((n, NUM_FEATURES))
    
    # ==========================================================================
    # RAW FEATURES (0-10): Direct from input
    # ==========================================================================
    features[:, 0] = opens
    features[:, 1] = highs
    features[:, 2] = lows
    features[:, 3] = closes
    features[:, 4] = volumes
    features[:, 5] = quote_asset_volumes
    features[:, 6] = taker_buy_base_asset_volumes
    features[:, 7] = taker_buy_quote_asset_volumes
    features[:, 8] = number_of_trades
    features[:, 9] = spread_bps
    features[:, 10] = taker_buy_ratio
    
    # ==========================================================================
    # DERIVED FEATURES (11-26): Computed from raw
    # ==========================================================================
    
    # Log returns (indices 11-13)
    log_return_1m = compute_log_returns(closes, 1)
    features[:, 11] = log_return_1m
    features[:, 12] = compute_log_returns(closes, 5)
    features[:, 13] = compute_log_returns(closes, 15)
    
    # Volatility (indices 14-16)
    features[:, 14] = compute_volatility(log_return_1m, 5)
    features[:, 15] = compute_volatility(log_return_1m, 15)
    features[:, 16] = compute_volatility(log_return_1m, 30)
    
    # SMA normalized (indices 17-19)
    sma_5 = compute_sma(closes, 5)
    sma_15 = compute_sma(closes, 15)
    sma_30 = compute_sma(closes, 30)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features[:, 17] = np.where(closes > 0, (closes - sma_5) / closes, 0)
        features[:, 18] = np.where(closes > 0, (closes - sma_15) / closes, 0)
        features[:, 19] = np.where(closes > 0, (closes - sma_30) / closes, 0)
    
    # EMA normalized (indices 20-22)
    ema_5 = compute_ema(closes, 5)
    ema_15 = compute_ema(closes, 15)
    ema_30 = compute_ema(closes, 30)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features[:, 20] = np.where(closes > 0, (closes - ema_5) / closes, 0)
        features[:, 21] = np.where(closes > 0, (closes - ema_15) / closes, 0)
        features[:, 22] = np.where(closes > 0, (closes - ema_30) / closes, 0)
    
    # RSI (index 23)
    features[:, 23] = compute_rsi(closes, 14)
    
    # Volume SMA ratio (index 24)
    volume_sma_20 = compute_sma(volumes, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features[:, 24] = np.where(volume_sma_20 > 0, volumes / volume_sma_20, 1.0)
    
    # VWAP deviation (index 25)
    vwap = compute_vwap(highs, lows, closes, volumes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features[:, 25] = np.where(closes > 0, (closes - vwap) / closes, 0)
    
    # Price position (index 26)
    price_range = highs - lows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features[:, 26] = np.where(price_range > 0, (closes - lows) / price_range, 0.5)
    
    return features


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize features.
    
    Args:
        features: Feature matrix (N, 27)
        mean: Pre-computed mean (use for inference)
        std: Pre-computed std (use for inference)
        
    Returns:
        (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        # Prevent division by zero
        std = np.where(std < 1e-10, 1.0, std)
    
    normalized = (features - mean) / std
    
    # Clip extreme values
    normalized = np.clip(normalized, -10, 10)
    
    return normalized, mean, std


def create_sequences(
    features: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Create sliding window sequences for model input.
    
    Args:
        features: Feature matrix (N, 27)
        sequence_length: Window size (default 60)
        
    Returns:
        Sequences array (N - sequence_length + 1, sequence_length, 27)
    """
    n = len(features)
    if n < sequence_length:
        raise ValueError(f"Need at least {sequence_length} samples, got {n}")
    
    num_sequences = n - sequence_length + 1
    sequences = np.zeros((num_sequences, sequence_length, features.shape[1]))
    
    for i in range(num_sequences):
        sequences[i] = features[i:i + sequence_length]
    
    return sequences


def compute_targets(
    closes: np.ndarray,
    horizons: List[int] = [1, 2, 3, 4, 5],
) -> np.ndarray:
    """
    Compute prediction targets (log returns for each horizon).
    
    Args:
        closes: Close prices (N,)
        horizons: List of prediction horizons in minutes
        
    Returns:
        Targets array (N, len(horizons))
    """
    n = len(closes)
    targets = np.zeros((n, len(horizons)))
    
    for i, h in enumerate(horizons):
        if n > h:
            targets[:-h, i] = np.log(closes[h:] / closes[:-h])
    
    return targets


# =============================================================================
# VALIDATION: Ensure feature count matches
# =============================================================================
assert NUM_FEATURES == 27, f"Expected 27 features, got {NUM_FEATURES}"
assert len(RAW_FEATURES) == 11, f"Expected 11 raw features, got {len(RAW_FEATURES)}"
assert len(DERIVED_FEATURES) == 16, f"Expected 16 derived features, got {len(DERIVED_FEATURES)}"
