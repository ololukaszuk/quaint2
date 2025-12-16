"""
Market Analyzer - Core analysis logic.

Analyzes multiple timeframes and produces market context reports.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np

from loguru import logger

from config import Config
from database import Database
from models import (
    Candle, CandleArray, TrendInfo, MomentumInfo, 
    PriceLevel, MarketStructure, MarketContext
)
from timeframes import build_multi_timeframe_data, get_required_1m_candles
from pivots import calculate_all_pivots, AllPivotPoints
from smart_money import analyze_smc, SMCAnalysis
from signals import SignalGenerator, Signal
import indicators as ind


class MarketAnalyzer:
    """
    Multi-timeframe market analyzer.
    
    Produces comprehensive market context by analyzing:
    - Trend direction and strength across timeframes
    - Support/resistance levels
    - Momentum indicators (RSI, volume)
    - Market structure (higher highs/lows pattern)
    - Pivot points (Traditional, Fibonacci, Camarilla, Woodie, DeMark)
    - Smart Money Concepts (Order Blocks, FVG, BOS/CHoCH)
    - Multi-confluence trading signals
    """
    
    def __init__(self, db: Database, config: Config):
        self.db = db
        self.config = config
        self.signal_generator = SignalGenerator()
    
    async def analyze(self) -> Optional[MarketContext]:
        """
        Run full market analysis.
        
        Returns:
            MarketContext object with complete analysis, or None if insufficient data
        """
        # Calculate how many 1m candles we need for all timeframes
        # For daily candles: need ~180 days * 1440 min = 259,200 candles
        required_candles = self.config.lookback_candles_1m
        
        logger.debug(f"Fetching {required_candles:,} 1m candles for analysis")
        
        # Fetch 1m candles
        candles_1m = await self.db.get_candles(limit=required_candles)
        
        if len(candles_1m) < 1000:
            logger.warning(f"Insufficient data: only {len(candles_1m)} candles")
            return None
        
        logger.debug(f"Loaded {len(candles_1m):,} 1m candles ({len(candles_1m)/1440:.1f} days)")
        
        # Build multi-timeframe data
        tf_data = build_multi_timeframe_data(candles_1m, self.config.timeframes)
        
        for tf, data in tf_data.items():
            logger.debug(f"  {tf}: {len(data)} candles")
        
        # Current price from latest candle
        current_price = candles_1m[-1].close
        timestamp = candles_1m[-1].time
        
        # Analyze each timeframe
        trends: Dict[str, TrendInfo] = {}
        momentum: Dict[str, MomentumInfo] = {}
        
        for tf in self.config.timeframes:
            if tf not in tf_data or len(tf_data[tf]) < 50:
                continue
            
            data = tf_data[tf]
            
            # Trend analysis
            trends[tf] = self._analyze_trend(data)
            
            # Momentum analysis  
            momentum[tf] = self._analyze_momentum(data)
        
        # Support/Resistance from multiple timeframes
        support_levels, resistance_levels = self._find_key_levels(tf_data, current_price)
        
        # Market structure from 1h or 4h
        structure = self._analyze_structure(tf_data.get("1h") or tf_data.get("4h"))
        
        # ===== ADVANCED ANALYSIS =====
        
        # Pivot Points (from daily data)
        pivots = None
        if "1d" in tf_data and len(tf_data["1d"]) >= 2:
            pivots = calculate_all_pivots(tf_data["1d"], current_price)
            logger.debug(f"Calculated pivot points (Daily P: ${pivots.traditional.pivot:,.0f})")
        
        # Smart Money Concepts (from 1h data)
        smc = None
        if "1h" in tf_data and len(tf_data["1h"]) >= 50:
            smc = analyze_smc(tf_data["1h"], lookback=50)
            logger.debug(f"SMC analysis: bias={smc.current_bias}, OBs={len(smc.bullish_obs)}B/{len(smc.bearish_obs)}S")
        
        # Build context
        context = MarketContext(
            timestamp=timestamp,
            current_price=current_price,
            trends=trends,
            momentum=momentum,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            structure=structure,
            candle_data=tf_data,
            pivots=pivots,
            smc=smc,
        )
        
        # Generate trading signal
        signal = self.signal_generator.generate_signal(context, pivots, smc)
        context.signal = signal
        
        logger.debug(f"Signal: {signal.signal_type.value} ({signal.confidence:.0f}% confidence)")
        
        return context
    
    def _analyze_trend(self, data: CandleArray) -> TrendInfo:
        """Analyze trend for a single timeframe."""
        closes = data.close
        highs = data.high
        lows = data.low
        
        # Calculate EMAs
        ema_fast = ind.ema(closes, self.config.ema_fast)
        ema_slow = ind.ema(closes, self.config.ema_slow)
        ema_trend = ind.ema(closes, self.config.ema_trend)
        
        current_price = closes[-1]
        
        # Price vs EMA
        if current_price > ema_trend[-1] * 1.001:
            price_vs_ema = "ABOVE"
        elif current_price < ema_trend[-1] * 0.999:
            price_vs_ema = "BELOW"
        else:
            price_vs_ema = "AT"
        
        # EMA alignment
        if ema_fast[-1] > ema_slow[-1] > ema_trend[-1]:
            ema_alignment = "BULLISH"
        elif ema_fast[-1] < ema_slow[-1] < ema_trend[-1]:
            ema_alignment = "BEARISH"
        else:
            ema_alignment = "MIXED"
        
        # Swing point analysis for HH/HL/LH/LL
        swing_high_idx, swing_low_idx = ind.find_swing_points(highs, lows, lookback=3)
        
        higher_highs = False
        higher_lows = False
        lower_highs = False
        lower_lows = False
        
        if len(swing_high_idx) >= 2:
            last_two_highs = highs[swing_high_idx[-2:]]
            if last_two_highs[-1] > last_two_highs[-2]:
                higher_highs = True
            else:
                lower_highs = True
        
        if len(swing_low_idx) >= 2:
            last_two_lows = lows[swing_low_idx[-2:]]
            if last_two_lows[-1] > last_two_lows[-2]:
                higher_lows = True
            else:
                lower_lows = True
        
        # Determine trend direction
        if higher_highs and higher_lows and ema_alignment == "BULLISH":
            direction = "UPTREND"
            strength = 1.0
        elif lower_highs and lower_lows and ema_alignment == "BEARISH":
            direction = "DOWNTREND"
            strength = 1.0
        elif ema_alignment == "BULLISH" or (higher_highs or higher_lows):
            direction = "UPTREND"
            strength = 0.6
        elif ema_alignment == "BEARISH" or (lower_highs or lower_lows):
            direction = "DOWNTREND"
            strength = 0.6
        else:
            direction = "SIDEWAYS"
            strength = 0.3
        
        # Adjust strength based on price position
        if price_vs_ema == "ABOVE" and direction == "UPTREND":
            strength = min(1.0, strength + 0.1)
        elif price_vs_ema == "BELOW" and direction == "DOWNTREND":
            strength = min(1.0, strength + 0.1)
        elif price_vs_ema != direction.replace("TREND", ""):
            strength = max(0.1, strength - 0.2)
        
        return TrendInfo(
            direction=direction,
            strength=strength,
            price_vs_ema=price_vs_ema,
            ema_alignment=ema_alignment,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            lower_highs=lower_highs,
            lower_lows=lower_lows,
        )
    
    def _analyze_momentum(self, data: CandleArray) -> MomentumInfo:
        """Analyze momentum for a single timeframe."""
        closes = data.close
        volumes = data.volume
        spread_bps = data.spread_bps
        taker_buy_ratio = data.taker_buy_ratio
        
        # RSI
        rsi_values = ind.rsi(closes, self.config.rsi_period)
        current_rsi = rsi_values[-1]
        
        if current_rsi < self.config.rsi_oversold:
            rsi_status = "OVERSOLD"
        elif current_rsi > self.config.rsi_overbought:
            rsi_status = "OVERBOUGHT"
        else:
            rsi_status = "NEUTRAL"
        
        # Volume ratio (current vs 20-period average)
        vol_sma = ind.sma(volumes, 20)
        if vol_sma[-1] > 0 and not np.isnan(vol_sma[-1]):
            volume_ratio = volumes[-1] / vol_sma[-1]
        else:
            volume_ratio = 1.0
        
        # Current spread and taker buy ratio
        current_spread = spread_bps[-1] if len(spread_bps) > 0 else 0.0
        current_taker_ratio = taker_buy_ratio[-1] if len(taker_buy_ratio) > 0 else 0.5
        
        return MomentumInfo(
            rsi=current_rsi,
            rsi_status=rsi_status,
            volume_ratio=volume_ratio,
            spread_bps=current_spread,
            taker_buy_ratio=current_taker_ratio,
        )
    
    def _find_key_levels(
        self, 
        tf_data: Dict[str, CandleArray],
        current_price: float
    ) -> tuple[List[PriceLevel], List[PriceLevel]]:
        """
        Find support and resistance levels from multiple timeframes.
        
        Higher timeframe levels are weighted more strongly.
        """
        all_supports: List[PriceLevel] = []
        all_resistances: List[PriceLevel] = []
        
        # Timeframe weights (higher TF = more important)
        tf_weights = {
            "5m": 0.3,
            "15m": 0.5,
            "1h": 0.8,
            "4h": 1.0,
            "1d": 1.2,
        }
        
        for tf, data in tf_data.items():
            if len(data) < 50:
                continue
            
            weight = tf_weights.get(tf, 0.5)
            
            supports, resistances = ind.find_support_resistance_levels(
                data.high,
                data.low,
                data.close,
                tolerance_pct=self.config.sr_price_tolerance_pct,
                min_touches=self.config.sr_min_touches,
                lookback=5
            )
            
            # Convert to PriceLevel objects
            for price, touches in supports:
                all_supports.append(PriceLevel(
                    price=price,
                    touches=touches,
                    strength=min(1.0, (touches / 5) * weight),
                    last_touch=datetime.now(timezone.utc),  # Simplified
                    level_type="SUPPORT",
                    source_timeframe=tf,
                ))
            
            for price, touches in resistances:
                all_resistances.append(PriceLevel(
                    price=price,
                    touches=touches,
                    strength=min(1.0, (touches / 5) * weight),
                    last_touch=datetime.now(timezone.utc),
                    level_type="RESISTANCE",
                    source_timeframe=tf,
                ))
        
        # Merge nearby levels from different timeframes
        supports = self._merge_levels(all_supports, current_price, "SUPPORT")
        resistances = self._merge_levels(all_resistances, current_price, "RESISTANCE")
        
        return supports, resistances
    
    def _merge_levels(
        self, 
        levels: List[PriceLevel],
        current_price: float,
        level_type: str
    ) -> List[PriceLevel]:
        """Merge nearby price levels and rank by strength."""
        if not levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x.price)
        
        merged = []
        current_cluster = [sorted_levels[0]]
        
        tolerance = self.config.sr_price_tolerance_pct / 100
        
        for level in sorted_levels[1:]:
            cluster_price = np.mean([l.price for l in current_cluster])
            
            if abs(level.price - cluster_price) / cluster_price <= tolerance:
                current_cluster.append(level)
            else:
                # Finalize current cluster
                merged.append(self._combine_cluster(current_cluster, level_type))
                current_cluster = [level]
        
        # Don't forget last cluster
        if current_cluster:
            merged.append(self._combine_cluster(current_cluster, level_type))
        
        # Sort by distance from current price
        if level_type == "SUPPORT":
            merged = [l for l in merged if l.price < current_price]
            merged.sort(key=lambda x: current_price - x.price)  # Nearest first
        else:
            merged = [l for l in merged if l.price > current_price]
            merged.sort(key=lambda x: x.price - current_price)  # Nearest first
        
        return merged[:5]  # Top 5 levels
    
    def _combine_cluster(self, cluster: List[PriceLevel], level_type: str) -> PriceLevel:
        """Combine a cluster of nearby levels into one."""
        # Weighted average price (weight by strength)
        total_weight = sum(l.strength for l in cluster)
        if total_weight > 0:
            avg_price = sum(l.price * l.strength for l in cluster) / total_weight
        else:
            avg_price = np.mean([l.price for l in cluster])
        
        total_touches = sum(l.touches for l in cluster)
        max_strength = max(l.strength for l in cluster)
        
        # Boost strength if confirmed by multiple timeframes
        tf_count = len(set(l.source_timeframe for l in cluster))
        strength_boost = 1.0 + (tf_count - 1) * 0.15
        
        return PriceLevel(
            price=avg_price,
            touches=total_touches,
            strength=min(1.0, max_strength * strength_boost),
            last_touch=max(l.last_touch for l in cluster),
            level_type=level_type,
            source_timeframe=",".join(sorted(set(l.source_timeframe for l in cluster))),
        )
    
    def _analyze_structure(self, data: Optional[CandleArray]) -> MarketStructure:
        """Analyze market structure (HH/HL/LH/LL pattern)."""
        if data is None or len(data) < 50:
            return MarketStructure(pattern="UNKNOWN")
        
        highs = data.high
        lows = data.low
        
        swing_high_idx, swing_low_idx = ind.find_swing_points(highs, lows, lookback=5)
        
        if len(swing_high_idx) < 2 or len(swing_low_idx) < 2:
            return MarketStructure(pattern="INSUFFICIENT_DATA")
        
        # Get last few swing points
        last_swing_highs = highs[swing_high_idx[-3:]] if len(swing_high_idx) >= 3 else highs[swing_high_idx]
        last_swing_lows = lows[swing_low_idx[-3:]] if len(swing_low_idx) >= 3 else lows[swing_low_idx]
        
        # Determine pattern
        hh = all(last_swing_highs[i] > last_swing_highs[i-1] for i in range(1, len(last_swing_highs)))
        hl = all(last_swing_lows[i] > last_swing_lows[i-1] for i in range(1, len(last_swing_lows)))
        lh = all(last_swing_highs[i] < last_swing_highs[i-1] for i in range(1, len(last_swing_highs)))
        ll = all(last_swing_lows[i] < last_swing_lows[i-1] for i in range(1, len(last_swing_lows)))
        
        if hh and hl:
            pattern = "HIGHER_HIGHS_LOWS"
        elif lh and ll:
            pattern = "LOWER_HIGHS_LOWS"
        elif hh and ll:
            pattern = "EXPANDING"
        elif lh and hl:
            pattern = "CONTRACTING"
        else:
            pattern = "RANGING"
        
        return MarketStructure(
            pattern=pattern,
            last_swing_high=float(highs[swing_high_idx[-1]]) if len(swing_high_idx) > 0 else None,
            last_swing_low=float(lows[swing_low_idx[-1]]) if len(swing_low_idx) > 0 else None,
        )
