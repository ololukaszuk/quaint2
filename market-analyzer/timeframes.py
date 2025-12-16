"""
Timeframe aggregation - build higher timeframes from 1m candles.
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional
import numpy as np

from models import Candle, CandleArray


# Timeframe definitions in minutes
TIMEFRAME_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


def get_candle_bucket(timestamp: datetime, tf_minutes: int) -> datetime:
    """
    Get the bucket/period start time for a candle.
    
    For example, for 15m timeframe:
    - 10:01 -> 10:00
    - 10:14 -> 10:00
    - 10:15 -> 10:15
    
    Args:
        timestamp: Candle timestamp
        tf_minutes: Timeframe in minutes
        
    Returns:
        Period start timestamp
    """
    # Handle daily candles specially (align to midnight UTC)
    if tf_minutes >= 1440:
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # For sub-daily, align to minute bucket
    total_minutes = timestamp.hour * 60 + timestamp.minute
    bucket_start = (total_minutes // tf_minutes) * tf_minutes
    
    hour = bucket_start // 60
    minute = bucket_start % 60
    
    return timestamp.replace(hour=hour, minute=minute, second=0, microsecond=0)


def aggregate_candles(
    candles: List[Candle],
    timeframe: str
) -> List[Candle]:
    """
    Aggregate 1m candles into higher timeframe.
    
    Args:
        candles: List of 1m candles (must be sorted by time ascending)
        timeframe: Target timeframe ("5m", "15m", "1h", "4h", "1d")
        
    Returns:
        List of aggregated candles
    """
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    if not candles:
        return []
    
    tf_minutes = TIMEFRAME_MINUTES[timeframe]
    
    # If 1m, just return as-is
    if tf_minutes == 1:
        return candles
    
    # Group candles by bucket
    buckets: Dict[datetime, List[Candle]] = {}
    
    for candle in candles:
        bucket_time = get_candle_bucket(candle.time, tf_minutes)
        if bucket_time not in buckets:
            buckets[bucket_time] = []
        buckets[bucket_time].append(candle)
    
    # Aggregate each bucket
    aggregated = []
    for bucket_time in sorted(buckets.keys()):
        bucket_candles = buckets[bucket_time]
        
        # OHLCV aggregation
        agg_candle = Candle(
            time=bucket_time,
            open=bucket_candles[0].open,
            high=max(c.high for c in bucket_candles),
            low=min(c.low for c in bucket_candles),
            close=bucket_candles[-1].close,
            volume=sum(c.volume for c in bucket_candles),
            quote_volume=sum(c.quote_volume for c in bucket_candles),
            taker_buy_volume=sum(c.taker_buy_volume for c in bucket_candles),
            taker_buy_quote_volume=sum(c.taker_buy_quote_volume for c in bucket_candles),
            trades=sum(c.trades for c in bucket_candles),
            spread_bps=np.mean([c.spread_bps for c in bucket_candles if c.spread_bps]),
            taker_buy_ratio=np.mean([c.taker_buy_ratio for c in bucket_candles if c.taker_buy_ratio]),
        )
        aggregated.append(agg_candle)
    
    return aggregated


def build_multi_timeframe_data(
    candles_1m: List[Candle],
    timeframes: List[str]
) -> Dict[str, CandleArray]:
    """
    Build CandleArrays for multiple timeframes from 1m data.
    
    Args:
        candles_1m: List of 1m candles
        timeframes: List of timeframes to build ("5m", "15m", "1h", "4h", "1d")
        
    Returns:
        Dict mapping timeframe to CandleArray
    """
    result = {}
    
    for tf in timeframes:
        if tf == "1m":
            aggregated = candles_1m
        else:
            aggregated = aggregate_candles(candles_1m, tf)
        
        if aggregated:
            result[tf] = CandleArray.from_candles(aggregated)
    
    return result


def get_required_1m_candles(timeframe: str, num_candles: int) -> int:
    """
    Calculate how many 1m candles are needed to produce N candles of a timeframe.
    
    Args:
        timeframe: Target timeframe
        num_candles: Number of target candles needed
        
    Returns:
        Number of 1m candles required
    """
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    tf_minutes = TIMEFRAME_MINUTES[timeframe]
    return num_candles * tf_minutes + tf_minutes  # Extra buffer for alignment
