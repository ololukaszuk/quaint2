"""
Data models for Market Analyzer.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
import logging
import warnings

# Suppress NumPy's timezone warning - we handle UTC explicitly via _ensure_utc()
warnings.filterwarnings('ignore', message='no explicit representation of timezones available for np.datetime64')

logger = logging.getLogger(__name__)

@dataclass
class Candle:
    """Single OHLCV candle."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    taker_buy_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0
    trades: int = 0
    spread_bps: float = 0.0
    taker_buy_ratio: float = 0.5

@dataclass
class CandleArray:
    """
    Efficient storage for candle data as numpy arrays.
    Used for fast indicator calculations.
    
    All timestamps are stored as UTC-aware datetime64[s].
    """
    time: np.ndarray  # datetime64[s] in UTC
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    spread_bps: np.ndarray
    taker_buy_ratio: np.ndarray

    def __len__(self) -> int:
        return len(self.close)

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        """
        Ensure a datetime object is UTC-aware.
        
        - If naive: assumes UTC and adds timezone info
        - If already UTC: returns as-is
        - If different timezone: converts to UTC
        
        Args:
            dt: datetime object (naive or aware)
            
        Returns:
            UTC-aware datetime object
        """
        if dt.tzinfo is None:
            # Naive datetime - assume it's UTC
            logger.debug(f"Converting naive datetime {dt} to UTC-aware")
            return dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo == timezone.utc or dt.tzinfo.tzname(dt) == "UTC":
            # Already UTC
            return dt
        else:
            # Different timezone - convert to UTC
            logger.debug(f"Converting {dt.tzinfo.tzname(dt)} datetime to UTC")
            return dt.astimezone(timezone.utc)

    @classmethod
    def from_candles(cls, candles: List[Candle]) -> "CandleArray":
        """
        Create CandleArray from list of Candle objects.
        
        Ensures all timestamps are UTC-aware before conversion to numpy datetime64.
        This prevents NumPy timezone warnings and ensures data consistency.
        
        Args:
            candles: List of Candle objects
            
        Returns:
            CandleArray with UTC-aware timestamps
        """
        if not candles:
            raise ValueError("Cannot create CandleArray from empty candle list")
        
        # Ensure all times are UTC-aware
        utc_times = [cls._ensure_utc(c.time) for c in candles]
        
        return cls(
            time=np.array(utc_times, dtype='datetime64[s]'),
            open=np.array([c.open for c in candles], dtype=np.float64),
            high=np.array([c.high for c in candles], dtype=np.float64),
            low=np.array([c.low for c in candles], dtype=np.float64),
            close=np.array([c.close for c in candles], dtype=np.float64),
            volume=np.array([c.volume for c in candles], dtype=np.float64),
            spread_bps=np.array([c.spread_bps or 0 for c in candles], dtype=np.float64),
            taker_buy_ratio=np.array([c.taker_buy_ratio or 0.5 for c in candles], dtype=np.float64),
        )


@dataclass
class TrendInfo:
    """Trend information for a single timeframe."""
    direction: str  # "UPTREND", "DOWNTREND", "SIDEWAYS"
    strength: float  # 0.0 to 1.0
    price_vs_ema: str  # "ABOVE", "BELOW", "AT"
    ema_alignment: str  # "BULLISH" (fast>slow>trend), "BEARISH", "MIXED"
    higher_highs: bool = False
    higher_lows: bool = False
    lower_highs: bool = False
    lower_lows: bool = False


@dataclass
class MomentumInfo:
    """Momentum indicators for a single timeframe."""
    rsi: float
    rsi_status: str  # "OVERSOLD", "OVERBOUGHT", "NEUTRAL"
    volume_ratio: float  # Current volume vs average
    spread_bps: float  # Current spread
    taker_buy_ratio: float  # Buy pressure


@dataclass 
class PriceLevel:
    """Support or Resistance level."""
    price: float
    touches: int
    strength: float  # 0.0 to 1.0
    last_touch: datetime
    level_type: str  # "SUPPORT" or "RESISTANCE"
    source_timeframe: str  # Which TF identified this level


@dataclass
class MarketStructure:
    """Market structure analysis."""
    pattern: str  # "HIGHER_HIGHS_LOWS", "LOWER_HIGHS_LOWS", "RANGING", "BREAKOUT"
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    swing_high_time: Optional[datetime] = None
    swing_low_time: Optional[datetime] = None
    trend_breaks: int = 0  # Recent breaks of structure


@dataclass
class MarketContext:
    """Complete market context report."""
    timestamp: datetime
    current_price: float
    
    # Multi-timeframe data
    trends: Dict[str, TrendInfo] = field(default_factory=dict)
    momentum: Dict[str, MomentumInfo] = field(default_factory=dict)
    
    # Key levels
    support_levels: List[PriceLevel] = field(default_factory=list)
    resistance_levels: List[PriceLevel] = field(default_factory=list)
    
    # Structure
    structure: MarketStructure = field(default_factory=lambda: MarketStructure(pattern="UNKNOWN"))
    
    # Raw data reference (optional)
    candle_data: Dict[str, CandleArray] = field(default_factory=dict)
    
    # Advanced analysis (populated separately)
    pivots: Optional[any] = None  # AllPivotPoints
    smc: Optional[any] = None  # SMCAnalysis
    signal: Optional[any] = None  # Signal
