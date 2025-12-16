"""
Technical indicators for Market Analyzer.

All indicators operate on numpy arrays for efficiency.
"""

import numpy as np
from typing import Tuple, Optional


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Exponential Moving Average.
    
    Args:
        data: Price data array
        period: EMA period
        
    Returns:
        EMA values array (same length as input)
    """
    if len(data) == 0:
        return np.array([])
    
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data, dtype=np.float64)
    result[0] = data[0]
    
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    
    return result


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average.
    
    Args:
        data: Price data array
        period: SMA period
        
    Returns:
        SMA values array (same length as input, NaN for insufficient data)
    """
    if len(data) == 0:
        return np.array([])
    
    result = np.full_like(data, np.nan, dtype=np.float64)
    
    if len(data) >= period:
        cumsum = np.cumsum(data)
        result[period-1:] = (cumsum[period-1:] - np.concatenate([[0], cumsum[:-period]])) / period
    
    return result


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index.
    
    Args:
        closes: Close price array
        period: RSI period (default 14)
        
    Returns:
        RSI values (0-100 scale)
    """
    if len(closes) < period + 1:
        return np.full_like(closes, 50.0)
    
    result = np.full_like(closes, 50.0, dtype=np.float64)
    deltas = np.diff(closes)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate using Wilder's smoothing
    avg_gain = np.zeros(len(gains), dtype=np.float64)
    avg_loss = np.zeros(len(losses), dtype=np.float64)
    
    # Initial average
    avg_gain[period-1] = np.mean(gains[:period])
    avg_loss[period-1] = np.mean(losses[:period])
    
    # Smoothed averages
    for i in range(period, len(gains)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RSI
    for i in range(period-1, len(gains)):
        if avg_loss[i] == 0:
            result[i+1] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            result[i+1] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range.
    
    Args:
        highs: High price array
        lows: Low price array
        closes: Close price array
        period: ATR period
        
    Returns:
        ATR values array
    """
    if len(closes) < 2:
        return np.zeros_like(closes)
    
    # True range calculation
    tr = np.zeros(len(closes), dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ATR as EMA of TR
    return ema(tr, period)


def bollinger_bands(
    closes: np.ndarray, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.
    
    Args:
        closes: Close price array
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = sma(closes, period)
    
    # Calculate rolling standard deviation
    std = np.full_like(closes, np.nan, dtype=np.float64)
    for i in range(period-1, len(closes)):
        std[i] = np.std(closes[i-period+1:i+1])
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower


def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence Divergence).
    
    Args:
        closes: Close price array
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def find_swing_points(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find swing highs and swing lows.
    
    A swing high is a high that is higher than `lookback` bars on each side.
    A swing low is a low that is lower than `lookback` bars on each side.
    
    Args:
        highs: High price array
        lows: Low price array
        lookback: Number of bars on each side to confirm swing
        
    Returns:
        Tuple of (swing_high_indices, swing_low_indices)
    """
    n = len(highs)
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, n - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append(i)
        
        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append(i)
    
    return np.array(swing_highs), np.array(swing_lows)


def find_support_resistance_levels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tolerance_pct: float = 0.15,
    min_touches: int = 2,
    lookback: int = 5
) -> Tuple[list, list]:
    """
    Find support and resistance levels based on swing points.
    
    Args:
        highs: High price array
        lows: Low price array
        closes: Close price array
        tolerance_pct: Percentage tolerance for grouping levels
        min_touches: Minimum touches to confirm a level
        lookback: Lookback for swing point detection
        
    Returns:
        Tuple of (support_levels, resistance_levels) as lists of (price, touches) tuples
    """
    swing_high_idx, swing_low_idx = find_swing_points(highs, lows, lookback)
    
    current_price = closes[-1] if len(closes) > 0 else 0
    
    # Collect swing prices
    swing_high_prices = highs[swing_high_idx] if len(swing_high_idx) > 0 else np.array([])
    swing_low_prices = lows[swing_low_idx] if len(swing_low_idx) > 0 else np.array([])
    
    def cluster_levels(prices: np.ndarray, tolerance: float) -> list:
        """Cluster nearby price levels."""
        if len(prices) == 0:
            return []
        
        sorted_prices = np.sort(prices)
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        for i in range(1, len(sorted_prices)):
            # Check if price is within tolerance of cluster mean
            cluster_mean = np.mean(current_cluster)
            if abs(sorted_prices[i] - cluster_mean) / cluster_mean * 100 <= tolerance:
                current_cluster.append(sorted_prices[i])
            else:
                if len(current_cluster) >= min_touches:
                    clusters.append((np.mean(current_cluster), len(current_cluster)))
                current_cluster = [sorted_prices[i]]
        
        # Don't forget last cluster
        if len(current_cluster) >= min_touches:
            clusters.append((np.mean(current_cluster), len(current_cluster)))
        
        return clusters
    
    # Cluster support levels (swing lows)
    support_clusters = cluster_levels(swing_low_prices, tolerance_pct)
    # Filter to only levels below current price
    support_levels = [(p, t) for p, t in support_clusters if p < current_price]
    support_levels.sort(key=lambda x: x[0], reverse=True)  # Nearest first
    
    # Cluster resistance levels (swing highs)
    resistance_clusters = cluster_levels(swing_high_prices, tolerance_pct)
    # Filter to only levels above current price
    resistance_levels = [(p, t) for p, t in resistance_clusters if p > current_price]
    resistance_levels.sort(key=lambda x: x[0])  # Nearest first
    
    return support_levels, resistance_levels


def calculate_volume_profile(
    closes: np.ndarray,
    volumes: np.ndarray,
    num_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate volume profile (volume at price).
    
    Args:
        closes: Close price array
        volumes: Volume array
        num_bins: Number of price bins
        
    Returns:
        Tuple of (price_levels, volume_at_price)
    """
    if len(closes) == 0:
        return np.array([]), np.array([])
    
    price_min = np.min(closes)
    price_max = np.max(closes)
    
    bins = np.linspace(price_min, price_max, num_bins + 1)
    bin_volumes = np.zeros(num_bins)
    
    # Assign volume to bins
    for i in range(len(closes)):
        bin_idx = np.searchsorted(bins, closes[i]) - 1
        bin_idx = max(0, min(bin_idx, num_bins - 1))
        bin_volumes[bin_idx] += volumes[i]
    
    # Return bin centers and volumes
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return bin_centers, bin_volumes


def detect_divergence(
    prices: np.ndarray,
    indicator: np.ndarray,
    lookback: int = 14
) -> Optional[str]:
    """
    Detect bullish or bearish divergence.
    
    Args:
        prices: Price array (typically closes)
        indicator: Indicator array (RSI, MACD, etc.)
        lookback: Number of bars to look back
        
    Returns:
        "BULLISH", "BEARISH", or None
    """
    if len(prices) < lookback * 2:
        return None
    
    # Find recent swing points in price
    recent_prices = prices[-lookback*2:]
    recent_indicator = indicator[-lookback*2:]
    
    # Simple divergence detection: compare last two lows/highs
    # This is a simplified version - production would be more sophisticated
    
    price_min_idx = np.argmin(recent_prices[-lookback:])
    price_prev_min_idx = np.argmin(recent_prices[:lookback])
    
    price_max_idx = np.argmax(recent_prices[-lookback:])
    price_prev_max_idx = np.argmax(recent_prices[:lookback])
    
    # Bullish divergence: price makes lower low but indicator makes higher low
    if (recent_prices[-lookback:][price_min_idx] < recent_prices[:lookback][price_prev_min_idx] and
        recent_indicator[-lookback:][price_min_idx] > recent_indicator[:lookback][price_prev_min_idx]):
        return "BULLISH"
    
    # Bearish divergence: price makes higher high but indicator makes lower high
    if (recent_prices[-lookback:][price_max_idx] > recent_prices[:lookback][price_prev_max_idx] and
        recent_indicator[-lookback:][price_max_idx] < recent_indicator[:lookback][price_prev_max_idx]):
        return "BEARISH"
    
    return None
