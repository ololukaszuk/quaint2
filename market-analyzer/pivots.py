"""
Pivot Points Calculator

Calculates multiple types of pivot points from daily/4h candles:
- Traditional (Floor)
- Fibonacci
- Camarilla
- Woodie
- DeMark
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from models import CandleArray


@dataclass
class PivotLevels:
    """Pivot point levels for a single calculation method."""
    method: str
    pivot: float
    r1: float
    r2: float
    r3: float
    r4: Optional[float] = None
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: Optional[float] = None
    
    def all_levels(self) -> Dict[str, float]:
        """Return all levels as dict."""
        levels = {
            "P": self.pivot,
            "R1": self.r1, "R2": self.r2, "R3": self.r3,
            "S1": self.s1, "S2": self.s2, "S3": self.s3,
        }
        if self.r4 is not None:
            levels["R4"] = self.r4
        if self.s4 is not None:
            levels["S4"] = self.s4
        return levels


def calculate_traditional_pivots(high: float, low: float, close: float) -> PivotLevels:
    """
    Traditional (Floor Trader) Pivot Points.
    
    Most widely used, based on previous session's HLC.
    """
    pivot = (high + low + close) / 3
    
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return PivotLevels(
        method="Traditional",
        pivot=pivot,
        r1=r1, r2=r2, r3=r3,
        s1=s1, s2=s2, s3=s3,
    )


def calculate_fibonacci_pivots(high: float, low: float, close: float) -> PivotLevels:
    """
    Fibonacci Pivot Points.
    
    Uses Fibonacci ratios (0.382, 0.618, 1.0) for level calculation.
    Popular for crypto due to fib retracement usage.
    """
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    r1 = pivot + 0.382 * range_hl
    r2 = pivot + 0.618 * range_hl
    r3 = pivot + 1.000 * range_hl
    
    s1 = pivot - 0.382 * range_hl
    s2 = pivot - 0.618 * range_hl
    s3 = pivot - 1.000 * range_hl
    
    return PivotLevels(
        method="Fibonacci",
        pivot=pivot,
        r1=r1, r2=r2, r3=r3,
        s1=s1, s2=s2, s3=s3,
    )


def calculate_camarilla_pivots(high: float, low: float, close: float) -> PivotLevels:
    """
    Camarilla Pivot Points.
    
    Developed by Nick Scott, focuses on intraday trading.
    Levels are closer together - good for ranging markets.
    R3/S3 are key breakout levels, R4/S4 for extended moves.
    """
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    r1 = close + range_hl * 1.1 / 12
    r2 = close + range_hl * 1.1 / 6
    r3 = close + range_hl * 1.1 / 4
    r4 = close + range_hl * 1.1 / 2
    
    s1 = close - range_hl * 1.1 / 12
    s2 = close - range_hl * 1.1 / 6
    s3 = close - range_hl * 1.1 / 4
    s4 = close - range_hl * 1.1 / 2
    
    return PivotLevels(
        method="Camarilla",
        pivot=pivot,
        r1=r1, r2=r2, r3=r3, r4=r4,
        s1=s1, s2=s2, s3=s3, s4=s4,
    )


def calculate_woodie_pivots(high: float, low: float, close: float, open_price: float) -> PivotLevels:
    """
    Woodie Pivot Points.
    
    Gives more weight to closing price of previous period.
    Good for trend following as it's more responsive.
    """
    pivot = (high + low + 2 * close) / 4
    
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = r1 + (high - low)
    
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = s1 - (high - low)
    
    return PivotLevels(
        method="Woodie",
        pivot=pivot,
        r1=r1, r2=r2, r3=r3,
        s1=s1, s2=s2, s3=s3,
    )


def calculate_demark_pivots(high: float, low: float, close: float, open_price: float) -> PivotLevels:
    """
    DeMark Pivot Points.
    
    Condition-based calculation depending on open/close relationship.
    Only calculates one support and one resistance level.
    """
    if close < open_price:
        x = high + 2 * low + close
    elif close > open_price:
        x = 2 * high + low + close
    else:
        x = high + low + 2 * close
    
    pivot = x / 4
    r1 = x / 2 - low
    s1 = x / 2 - high
    
    return PivotLevels(
        method="DeMark",
        pivot=pivot,
        r1=r1, r2=r1 * 1.01, r3=r1 * 1.02,  # Extended levels
        s1=s1, s2=s1 * 0.99, s3=s1 * 0.98,
    )


@dataclass
class AllPivotPoints:
    """All pivot point calculations for current period."""
    traditional: PivotLevels
    fibonacci: PivotLevels
    camarilla: PivotLevels
    woodie: PivotLevels
    demark: PivotLevels
    
    # Confluence zones where multiple pivots align
    confluence_resistance: List[tuple]  # (price, strength, methods)
    confluence_support: List[tuple]
    
    def get_nearest_resistance(self, price: float) -> Optional[tuple]:
        """Get nearest resistance confluence above price."""
        above = [(p, s, m) for p, s, m in self.confluence_resistance if p > price]
        return min(above, key=lambda x: x[0]) if above else None
    
    def get_nearest_support(self, price: float) -> Optional[tuple]:
        """Get nearest support confluence below price."""
        below = [(p, s, m) for p, s, m in self.confluence_support if p < price]
        return max(below, key=lambda x: x[0]) if below else None


def find_pivot_confluence(
    pivots: List[PivotLevels],
    current_price: float,
    tolerance_pct: float = 0.1
) -> tuple[List[tuple], List[tuple]]:
    """
    Find confluence zones where multiple pivot methods agree.
    
    Args:
        pivots: List of PivotLevels from different methods
        current_price: Current price for above/below classification
        tolerance_pct: Percentage tolerance for grouping (0.1 = 0.1%)
        
    Returns:
        (resistance_confluences, support_confluences)
        Each is list of (price, strength, methods_list)
    """
    all_levels = []
    
    for pivot in pivots:
        levels = pivot.all_levels()
        for name, price in levels.items():
            if price > 0:
                all_levels.append({
                    "price": price,
                    "method": pivot.method,
                    "level": name,
                })
    
    # Sort by price
    all_levels.sort(key=lambda x: x["price"])
    
    # Cluster nearby levels
    clusters = []
    current_cluster = [all_levels[0]] if all_levels else []
    
    for level in all_levels[1:]:
        cluster_avg = np.mean([l["price"] for l in current_cluster])
        if abs(level["price"] - cluster_avg) / cluster_avg * 100 <= tolerance_pct:
            current_cluster.append(level)
        else:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [level]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    # Convert clusters to confluence zones
    resistance = []
    support = []
    
    for cluster in clusters:
        avg_price = np.mean([l["price"] for l in cluster])
        methods = list(set(l["method"] for l in cluster))
        strength = len(methods) / 5  # 5 methods max = 100% strength
        
        zone = (avg_price, strength, methods)
        
        if avg_price > current_price:
            resistance.append(zone)
        else:
            support.append(zone)
    
    # Sort: resistance by price ascending, support by price descending
    resistance.sort(key=lambda x: x[0])
    support.sort(key=lambda x: x[0], reverse=True)
    
    return resistance, support


def calculate_all_pivots(
    data: CandleArray,
    current_price: float
) -> AllPivotPoints:
    """
    Calculate all pivot point types from the previous completed candle.
    
    Args:
        data: CandleArray (typically daily or 4h)
        current_price: Current price for confluence classification
        
    Returns:
        AllPivotPoints with all methods and confluence zones
    """
    if len(data) < 2:
        # Return empty pivots
        empty = PivotLevels("Empty", current_price, current_price, current_price, current_price,
                           s1=current_price, s2=current_price, s3=current_price)
        return AllPivotPoints(empty, empty, empty, empty, empty, [], [])
    
    # Use previous completed candle
    prev_high = float(data.high[-2])
    prev_low = float(data.low[-2])
    prev_close = float(data.close[-2])
    prev_open = float(data.open[-2])
    
    traditional = calculate_traditional_pivots(prev_high, prev_low, prev_close)
    fibonacci = calculate_fibonacci_pivots(prev_high, prev_low, prev_close)
    camarilla = calculate_camarilla_pivots(prev_high, prev_low, prev_close)
    woodie = calculate_woodie_pivots(prev_high, prev_low, prev_close, prev_open)
    demark = calculate_demark_pivots(prev_high, prev_low, prev_close, prev_open)
    
    # Find confluence
    all_pivots = [traditional, fibonacci, camarilla, woodie, demark]
    confluence_r, confluence_s = find_pivot_confluence(all_pivots, current_price)
    
    return AllPivotPoints(
        traditional=traditional,
        fibonacci=fibonacci,
        camarilla=camarilla,
        woodie=woodie,
        demark=demark,
        confluence_resistance=confluence_r,
        confluence_support=confluence_s,
    )
