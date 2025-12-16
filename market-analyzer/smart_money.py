"""
Smart Money Concepts (SMC) Analysis

Implements institutional trading concepts:
- Order Blocks (OB) - Institutional supply/demand zones
- Fair Value Gaps (FVG) - Imbalance zones likely to be filled
- Break of Structure (BOS) - Trend continuation signal
- Change of Character (CHoCH) - Trend reversal signal
- Liquidity Sweeps - Stop hunts before reversals
- Premium/Discount Zones - Optimal entry zones
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

from models import CandleArray


@dataclass
class OrderBlock:
    """
    Order Block - Zone where institutions placed orders.
    
    Bullish OB: Last bearish candle before a strong up move
    Bearish OB: Last bullish candle before a strong down move
    """
    type: str  # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    index: int  # Candle index in the array
    strength: float  # 0-1 based on move after OB
    mitigated: bool = False  # True if price has returned to OB
    tested_count: int = 0


@dataclass
class FairValueGap:
    """
    Fair Value Gap (Imbalance) - Gap between candles showing imbalance.
    
    Bullish FVG: Gap up (candle2.low > candle0.high)
    Bearish FVG: Gap down (candle2.high < candle0.low)
    """
    type: str  # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    index: int  # Middle candle index
    filled: bool = False
    fill_percentage: float = 0.0


@dataclass
class StructureBreak:
    """
    Break of Structure (BOS) or Change of Character (CHoCH).
    
    BOS: Break in direction of trend (continuation)
    CHoCH: Break against trend (reversal signal)
    """
    type: str  # "BOS" or "CHOCH"
    direction: str  # "BULLISH" or "BEARISH"
    break_level: float
    index: int
    strength: float  # Based on how decisively it broke


@dataclass
class LiquiditySweep:
    """
    Liquidity Sweep - Quick move to grab stops before reversal.
    
    Indicates smart money hunting retail stops.
    """
    type: str  # "HIGH_SWEEP" or "LOW_SWEEP"
    sweep_level: float
    reversal_index: int
    strength: float


@dataclass
class SMCAnalysis:
    """Complete Smart Money Concepts analysis."""
    # Order Blocks
    bullish_obs: List[OrderBlock] = field(default_factory=list)
    bearish_obs: List[OrderBlock] = field(default_factory=list)
    
    # Fair Value Gaps
    bullish_fvgs: List[FairValueGap] = field(default_factory=list)
    bearish_fvgs: List[FairValueGap] = field(default_factory=list)
    
    # Structure
    structure_breaks: List[StructureBreak] = field(default_factory=list)
    current_bias: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    
    # Liquidity
    liquidity_sweeps: List[LiquiditySweep] = field(default_factory=list)
    buy_side_liquidity: List[float] = field(default_factory=list)  # Equal highs
    sell_side_liquidity: List[float] = field(default_factory=list)  # Equal lows
    
    # Premium/Discount
    equilibrium: float = 0.0
    premium_zone: Tuple[float, float] = (0.0, 0.0)  # Above equilibrium
    discount_zone: Tuple[float, float] = (0.0, 0.0)  # Below equilibrium


def find_swing_points_smc(
    highs: np.ndarray,
    lows: np.ndarray,
    lookback: int = 3
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Find swing highs and lows for SMC analysis.
    
    Returns:
        (swing_highs, swing_lows) as lists of (index, price)
    """
    n = len(highs)
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, n - lookback):
        # Swing high: higher than lookback bars on each side
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append((i, float(highs[i])))
        
        # Swing low: lower than lookback bars on each side
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append((i, float(lows[i])))
    
    return swing_highs, swing_lows


def find_order_blocks(
    data: CandleArray,
    min_move_pct: float = 0.3,
    lookback: int = 50
) -> Tuple[List[OrderBlock], List[OrderBlock]]:
    """
    Find Order Blocks - zones where institutions placed orders.
    
    Bullish OB: Last down candle before strong up move
    Bearish OB: Last up candle before strong down move
    
    Args:
        data: CandleArray
        min_move_pct: Minimum % move after OB to qualify
        lookback: How far back to look
        
    Returns:
        (bullish_obs, bearish_obs)
    """
    if len(data) < lookback:
        return [], []
    
    opens = data.open[-lookback:]
    highs = data.high[-lookback:]
    lows = data.low[-lookback:]
    closes = data.close[-lookback:]
    
    bullish_obs = []
    bearish_obs = []
    
    for i in range(2, len(closes) - 3):
        current_price = closes[-1]
        
        # Check for bullish OB: bearish candle followed by strong up move
        if closes[i] < opens[i]:  # Bearish candle
            # Check if followed by strong up move
            move_after = (max(highs[i+1:i+4]) - closes[i]) / closes[i] * 100
            if move_after >= min_move_pct:
                ob = OrderBlock(
                    type="BULLISH",
                    top=float(opens[i]),  # OB zone is the bearish candle body
                    bottom=float(closes[i]),
                    index=i,
                    strength=min(1.0, move_after / (min_move_pct * 3)),
                    mitigated=current_price < opens[i],
                )
                bullish_obs.append(ob)
        
        # Check for bearish OB: bullish candle followed by strong down move
        if closes[i] > opens[i]:  # Bullish candle
            move_after = (closes[i] - min(lows[i+1:i+4])) / closes[i] * 100
            if move_after >= min_move_pct:
                ob = OrderBlock(
                    type="BEARISH",
                    top=float(closes[i]),
                    bottom=float(opens[i]),
                    index=i,
                    strength=min(1.0, move_after / (min_move_pct * 3)),
                    mitigated=current_price > opens[i],
                )
                bearish_obs.append(ob)
    
    # Sort by recency and keep strongest
    bullish_obs.sort(key=lambda x: (-x.strength, -x.index))
    bearish_obs.sort(key=lambda x: (-x.strength, -x.index))
    
    return bullish_obs[:5], bearish_obs[:5]


def find_fair_value_gaps(
    data: CandleArray,
    min_gap_pct: float = 0.05,
    lookback: int = 50
) -> Tuple[List[FairValueGap], List[FairValueGap]]:
    """
    Find Fair Value Gaps (imbalances).
    
    Bullish FVG: candle[i+1].low > candle[i-1].high
    Bearish FVG: candle[i+1].high < candle[i-1].low
    
    Args:
        data: CandleArray
        min_gap_pct: Minimum gap size as percentage
        lookback: How far back to look
        
    Returns:
        (bullish_fvgs, bearish_fvgs)
    """
    if len(data) < lookback:
        return [], []
    
    highs = data.high[-lookback:]
    lows = data.low[-lookback:]
    
    bullish_fvgs = []
    bearish_fvgs = []
    
    current_price = float(data.close[-1])
    
    for i in range(1, len(highs) - 1):
        prev_high = highs[i - 1]
        next_low = lows[i + 1]
        
        # Bullish FVG: gap up
        if next_low > prev_high:
            gap_size = (next_low - prev_high) / prev_high * 100
            if gap_size >= min_gap_pct:
                gap_top = float(next_low)
                gap_bottom = float(prev_high)
                
                # Check if filled
                filled = current_price <= gap_top and current_price >= gap_bottom
                fill_pct = 0.0
                if current_price < gap_top:
                    fill_pct = min(1.0, (gap_top - current_price) / (gap_top - gap_bottom))
                
                bullish_fvgs.append(FairValueGap(
                    type="BULLISH",
                    top=gap_top,
                    bottom=gap_bottom,
                    index=i,
                    filled=current_price < gap_bottom,
                    fill_percentage=fill_pct,
                ))
        
        # Bearish FVG: gap down
        prev_low = lows[i - 1]
        next_high = highs[i + 1]
        
        if next_high < prev_low:
            gap_size = (prev_low - next_high) / prev_low * 100
            if gap_size >= min_gap_pct:
                gap_top = float(prev_low)
                gap_bottom = float(next_high)
                
                filled = current_price >= gap_bottom and current_price <= gap_top
                fill_pct = 0.0
                if current_price > gap_bottom:
                    fill_pct = min(1.0, (current_price - gap_bottom) / (gap_top - gap_bottom))
                
                bearish_fvgs.append(FairValueGap(
                    type="BEARISH",
                    top=gap_top,
                    bottom=gap_bottom,
                    index=i,
                    filled=current_price > gap_top,
                    fill_percentage=fill_pct,
                ))
    
    # Filter unfilled and sort by proximity to current price
    bullish_fvgs = [f for f in bullish_fvgs if not f.filled]
    bearish_fvgs = [f for f in bearish_fvgs if not f.filled]
    
    bullish_fvgs.sort(key=lambda x: abs(current_price - x.top))
    bearish_fvgs.sort(key=lambda x: abs(current_price - x.bottom))
    
    return bullish_fvgs[:5], bearish_fvgs[:5]


def detect_structure_breaks(
    data: CandleArray,
    lookback: int = 50
) -> Tuple[List[StructureBreak], str]:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).
    
    BOS: Trend continuation - break of swing in trend direction
    CHoCH: Trend reversal - first break against the trend
    
    Returns:
        (structure_breaks, current_bias)
    """
    if len(data) < lookback:
        return [], "NEUTRAL"
    
    highs = data.high[-lookback:]
    lows = data.low[-lookback:]
    closes = data.close[-lookback:]
    
    swing_highs, swing_lows = find_swing_points_smc(highs, lows, lookback=3)
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return [], "NEUTRAL"
    
    breaks = []
    
    # Determine initial trend from swing structure
    recent_highs = [p for _, p in swing_highs[-3:]]
    recent_lows = [p for _, p in swing_lows[-3:]]
    
    higher_highs = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
    higher_lows = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
    lower_highs = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
    lower_lows = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
    
    if higher_highs and higher_lows:
        trend = "BULLISH"
    elif lower_highs and lower_lows:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"
    
    current_price = closes[-1]
    
    # Check for breaks of recent swing points
    for i, (idx, level) in enumerate(swing_highs[-3:]):
        if current_price > level:
            # Broke above swing high
            if trend == "BULLISH":
                break_type = "BOS"
            else:
                break_type = "CHOCH"
            
            breaks.append(StructureBreak(
                type=break_type,
                direction="BULLISH",
                break_level=level,
                index=idx,
                strength=min(1.0, (current_price - level) / level * 100),
            ))
    
    for i, (idx, level) in enumerate(swing_lows[-3:]):
        if current_price < level:
            # Broke below swing low
            if trend == "BEARISH":
                break_type = "BOS"
            else:
                break_type = "CHOCH"
            
            breaks.append(StructureBreak(
                type=break_type,
                direction="BEARISH",
                break_level=level,
                index=idx,
                strength=min(1.0, (level - current_price) / level * 100),
            ))
    
    # Update bias based on most recent break
    if breaks:
        recent_break = max(breaks, key=lambda x: x.index)
        if recent_break.type == "CHOCH":
            bias = recent_break.direction
        else:
            bias = trend
    else:
        bias = trend
    
    return breaks, bias


def find_liquidity_pools(
    data: CandleArray,
    tolerance_pct: float = 0.1,
    lookback: int = 50
) -> Tuple[List[float], List[float], List[LiquiditySweep]]:
    """
    Find liquidity pools (equal highs/lows) and recent sweeps.
    
    Equal highs = buy-side liquidity (stops above)
    Equal lows = sell-side liquidity (stops below)
    
    Returns:
        (buy_side_liquidity, sell_side_liquidity, recent_sweeps)
    """
    if len(data) < lookback:
        return [], [], []
    
    highs = data.high[-lookback:]
    lows = data.low[-lookback:]
    closes = data.close[-lookback:]
    
    tolerance = tolerance_pct / 100
    
    # Find equal highs (potential buy-side liquidity)
    buy_side = []
    for i in range(len(highs)):
        for j in range(i + 2, len(highs)):  # At least 2 candles apart
            if abs(highs[i] - highs[j]) / highs[i] <= tolerance:
                level = (highs[i] + highs[j]) / 2
                if level not in buy_side:
                    buy_side.append(float(level))
    
    # Find equal lows (potential sell-side liquidity)
    sell_side = []
    for i in range(len(lows)):
        for j in range(i + 2, len(lows)):
            if abs(lows[i] - lows[j]) / lows[i] <= tolerance:
                level = (lows[i] + lows[j]) / 2
                if level not in sell_side:
                    sell_side.append(float(level))
    
    # Detect recent sweeps (quick move past level and reversal)
    sweeps = []
    current_price = closes[-1]
    
    for level in buy_side:
        # Check if recently swept and reversed
        for i in range(-10, -1):
            if i + len(highs) >= 0:
                if highs[i] > level and closes[i] < level:
                    sweeps.append(LiquiditySweep(
                        type="HIGH_SWEEP",
                        sweep_level=level,
                        reversal_index=i + lookback,
                        strength=min(1.0, (highs[i] - level) / level * 100),
                    ))
                    break
    
    for level in sell_side:
        for i in range(-10, -1):
            if i + len(lows) >= 0:
                if lows[i] < level and closes[i] > level:
                    sweeps.append(LiquiditySweep(
                        type="LOW_SWEEP",
                        sweep_level=level,
                        reversal_index=i + lookback,
                        strength=min(1.0, (level - lows[i]) / level * 100),
                    ))
                    break
    
    # Sort by proximity to current price
    # Buy-side = ABOVE current price (these are stops above)
    # Sell-side = BELOW current price (these are stops below)
    buy_side = [l for l in buy_side if l > current_price]
    sell_side = [l for l in sell_side if l < current_price]
    
    buy_side.sort(key=lambda x: x - current_price)  # Nearest above first
    sell_side.sort(key=lambda x: current_price - x)  # Nearest below first
    
    return buy_side[:5], sell_side[:5], sweeps


def calculate_premium_discount(
    data: CandleArray,
    lookback: int = 50
) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """
    Calculate Premium/Discount zones based on recent range.
    
    Premium: Upper 50% of range (expensive, look to sell)
    Discount: Lower 50% of range (cheap, look to buy)
    
    Returns:
        (equilibrium, premium_zone, discount_zone)
    """
    if len(data) < lookback:
        current = float(data.close[-1]) if len(data) > 0 else 0
        return current, (current, current), (current, current)
    
    highs = data.high[-lookback:]
    lows = data.low[-lookback:]
    
    range_high = float(np.max(highs))
    range_low = float(np.min(lows))
    
    equilibrium = (range_high + range_low) / 2
    
    premium_zone = (equilibrium, range_high)  # Above EQ
    discount_zone = (range_low, equilibrium)  # Below EQ
    
    return equilibrium, premium_zone, discount_zone


def analyze_smc(data: CandleArray, lookback: int = 50) -> SMCAnalysis:
    """
    Complete Smart Money Concepts analysis.
    
    Args:
        data: CandleArray (typically 1h or 4h for best results)
        lookback: Number of candles to analyze
        
    Returns:
        SMCAnalysis with all SMC components
    """
    bullish_obs, bearish_obs = find_order_blocks(data, lookback=lookback)
    bullish_fvgs, bearish_fvgs = find_fair_value_gaps(data, lookback=lookback)
    structure_breaks, current_bias = detect_structure_breaks(data, lookback=lookback)
    buy_liq, sell_liq, sweeps = find_liquidity_pools(data, lookback=lookback)
    equilibrium, premium, discount = calculate_premium_discount(data, lookback=lookback)
    
    return SMCAnalysis(
        bullish_obs=bullish_obs,
        bearish_obs=bearish_obs,
        bullish_fvgs=bullish_fvgs,
        bearish_fvgs=bearish_fvgs,
        structure_breaks=structure_breaks,
        current_bias=current_bias,
        liquidity_sweeps=sweeps,
        buy_side_liquidity=buy_liq,
        sell_side_liquidity=sell_liq,
        equilibrium=equilibrium,
        premium_zone=premium,
        discount_zone=discount,
    )
