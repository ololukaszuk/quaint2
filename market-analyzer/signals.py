"""
Signal Generator

Generates BUY/SELL signals based on multi-factor confluence:
- Trend alignment across timeframes
- Support/Resistance proximity
- Momentum (RSI, volume)
- Smart Money Concepts (OB, FVG, structure)
- Pivot point confluence
- Risk/Reward analysis

Each signal includes:
- Direction (BUY/SELL)
- Confidence (0-100%)
- Entry, Stop Loss, Take Profit levels
- Detailed reasoning
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from models import MarketContext, TrendInfo, MomentumInfo, PriceLevel
from pivots import AllPivotPoints
from smart_money import SMCAnalysis


class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class SignalReason:
    """Individual reason contributing to signal."""
    factor: str
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    weight: float  # -1.0 to 1.0
    description: str


@dataclass
class TradeSetup:
    """Complete trade setup with entry, SL, TP."""
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward_ratio: float
    position_invalidation: str  # What would invalidate this setup


@dataclass
class Signal:
    """Trading signal with full analysis."""
    timestamp: datetime
    signal_type: SignalType
    direction: str  # "LONG", "SHORT", "NONE"
    confidence: float  # 0-100
    
    # Price levels
    current_price: float
    setup: Optional[TradeSetup]
    
    # Analysis breakdown
    reasons: List[SignalReason] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Confluence scores
    trend_score: float = 0.0  # -1 to 1
    momentum_score: float = 0.0
    structure_score: float = 0.0
    level_score: float = 0.0
    
    # Summary
    summary: str = ""


class SignalGenerator:
    """
    Multi-factor signal generator.
    
    Combines:
    - Multi-timeframe trend analysis
    - Support/Resistance levels
    - Pivot points
    - Smart Money Concepts
    - Momentum indicators
    - Volume analysis
    
    To produce high-confidence signals only when multiple factors align.
    """
    
    # Minimum confidence to generate a signal
    MIN_CONFIDENCE = 60
    
    # Factor weights (sum should be ~1.0 for each direction)
    WEIGHTS = {
        "htf_trend": 0.20,      # Higher timeframe trend (4h, 1d)
        "mtf_trend": 0.15,      # Medium timeframe trend (1h)
        "ltf_trend": 0.10,      # Lower timeframe trend (15m)
        "momentum": 0.15,       # RSI, divergence
        "volume": 0.10,         # Volume confirmation
        "sr_levels": 0.10,      # Support/Resistance
        "pivot_confluence": 0.05,  # Pivot points
        "smc_structure": 0.10,  # Market structure (BOS/CHoCH)
        "smc_zones": 0.05,      # Order blocks, FVG
    }
    
    def __init__(self):
        pass
    
    def generate_signal(
        self,
        context: MarketContext,
        pivots: Optional[AllPivotPoints] = None,
        smc: Optional[SMCAnalysis] = None,
    ) -> Signal:
        """
        Generate trading signal from market context.
        
        Args:
            context: MarketContext with trend/momentum analysis
            pivots: Pivot point calculations
            smc: Smart Money Concepts analysis
            
        Returns:
            Signal with direction, confidence, and reasoning
        """
        reasons: List[SignalReason] = []
        warnings: List[str] = []
        
        current_price = context.current_price
        
        # ===== 1. TREND ANALYSIS =====
        trend_score, trend_reasons = self._analyze_trends(context.trends)
        reasons.extend(trend_reasons)
        
        # ===== 2. MOMENTUM ANALYSIS =====
        momentum_score, momentum_reasons = self._analyze_momentum(context.momentum)
        reasons.extend(momentum_reasons)
        
        # ===== 3. PRICE ACTION CHECK =====
        price_action_score, price_action_reason = self._analyze_price_action(context)
        if price_action_reason:
            reasons.append(price_action_reason)
        
        # ===== 4. SUPPORT/RESISTANCE =====
        level_score, level_reasons, level_warnings = self._analyze_levels(
            current_price,
            context.support_levels,
            context.resistance_levels,
        )
        reasons.extend(level_reasons)
        warnings.extend(level_warnings)
        
        # ===== 5. PIVOT POINTS =====
        if pivots:
            pivot_score, pivot_reasons = self._analyze_pivots(current_price, pivots)
            reasons.extend(pivot_reasons)
        else:
            pivot_score = 0
        
        # ===== 6. SMART MONEY CONCEPTS =====
        if smc:
            smc_score, smc_reasons, smc_warnings = self._analyze_smc(current_price, smc)
            reasons.extend(smc_reasons)
            warnings.extend(smc_warnings)
            structure_score = smc_score
        else:
            structure_score = 0
        
        # ===== CALCULATE FINAL SCORE =====
        final_score = (
            trend_score * 0.30 +        # Slightly reduced to make room for price action
            momentum_score * 0.20 +     # Slightly reduced
            price_action_score * 0.15 + # NEW: Price action weight
            level_score * 0.15 +
            pivot_score * 0.05 +
            structure_score * 0.15
        )
        
        # ===== DETERMINE SIGNAL TYPE =====
        signal_type, direction, confidence = self._determine_signal(
            final_score, trend_score, momentum_score, price_action_score, warnings
        )
        
        # ===== GENERATE TRADE SETUP =====
        setup = None
        if direction != "NONE" and confidence >= self.MIN_CONFIDENCE:
            setup = self._generate_setup(
                current_price,
                direction,
                context.support_levels,
                context.resistance_levels,
                pivots,
                smc,
            )
        
        # ===== GENERATE SUMMARY =====
        summary = self._generate_summary(
            signal_type, direction, confidence, reasons, warnings
        )
        
        return Signal(
            timestamp=context.timestamp,
            signal_type=signal_type,
            direction=direction,
            confidence=confidence,
            current_price=current_price,
            setup=setup,
            reasons=reasons,
            warnings=warnings,
            trend_score=trend_score,
            momentum_score=momentum_score,
            structure_score=structure_score,
            level_score=level_score,
            summary=summary,
        )
    
    def _analyze_trends(
        self, 
        trends: Dict[str, TrendInfo]
    ) -> Tuple[float, List[SignalReason]]:
        """Analyze multi-timeframe trends."""
        reasons = []
        
        # Timeframe weights (higher TF = more important)
        tf_weights = {
            "1d": 0.30,
            "4h": 0.25,
            "1h": 0.20,
            "15m": 0.15,
            "5m": 0.10,
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for tf, weight in tf_weights.items():
            if tf not in trends:
                continue
            
            trend = trends[tf]
            total_weight += weight
            
            if trend.direction == "UPTREND":
                score = trend.strength * weight
                direction = "BULLISH"
            elif trend.direction == "DOWNTREND":
                score = -trend.strength * weight
                direction = "BEARISH"
            else:
                score = 0
                direction = "NEUTRAL"
            
            total_score += score
            
            reasons.append(SignalReason(
                factor=f"trend_{tf}",
                direction=direction,
                weight=score,
                description=f"{tf} trend: {trend.direction} ({trend.strength:.0%} strength), EMA: {trend.ema_alignment}",
            ))
        
        # Normalize
        if total_weight > 0:
            total_score = total_score / total_weight
        
        # Check for trend alignment bonus
        bullish_count = sum(1 for tf in ["1h", "4h", "1d"] if tf in trends and trends[tf].direction == "UPTREND")
        bearish_count = sum(1 for tf in ["1h", "4h", "1d"] if tf in trends and trends[tf].direction == "DOWNTREND")
        
        if bullish_count >= 2:
            reasons.append(SignalReason(
                factor="trend_alignment",
                direction="BULLISH",
                weight=0.2,
                description=f"HTF trend alignment: {bullish_count}/3 timeframes bullish",
            ))
            total_score = min(1.0, total_score + 0.15)
        elif bearish_count >= 2:
            reasons.append(SignalReason(
                factor="trend_alignment",
                direction="BEARISH",
                weight=-0.2,
                description=f"HTF trend alignment: {bearish_count}/3 timeframes bearish",
            ))
            total_score = max(-1.0, total_score - 0.15)
        
        return total_score, reasons
    
    def _analyze_momentum(
        self,
        momentum: Dict[str, MomentumInfo]
    ) -> Tuple[float, List[SignalReason]]:
        """Analyze momentum indicators."""
        reasons = []
        score = 0.0
        
        # RSI analysis (prefer 1h)
        rsi_tf = "1h" if "1h" in momentum else ("4h" if "4h" in momentum else None)
        
        if rsi_tf and rsi_tf in momentum:
            rsi = momentum[rsi_tf].rsi
            
            if rsi < 30:
                # Oversold - bullish
                rsi_score = (30 - rsi) / 30 * 0.5  # Max 0.5
                reasons.append(SignalReason(
                    factor="rsi_oversold",
                    direction="BULLISH",
                    weight=rsi_score,
                    description=f"RSI {rsi_tf} oversold at {rsi:.1f} - potential bounce",
                ))
                score += rsi_score
            elif rsi > 70:
                # Overbought - bearish
                rsi_score = (rsi - 70) / 30 * 0.5
                reasons.append(SignalReason(
                    factor="rsi_overbought",
                    direction="BEARISH",
                    weight=-rsi_score,
                    description=f"RSI {rsi_tf} overbought at {rsi:.1f} - potential pullback",
                ))
                score -= rsi_score
            else:
                # Neutral zone - slight bias based on direction
                if rsi > 50:
                    reasons.append(SignalReason(
                        factor="rsi_bullish_zone",
                        direction="BULLISH",
                        weight=0.1,
                        description=f"RSI {rsi_tf} in bullish zone at {rsi:.1f}",
                    ))
                    score += 0.1
                elif rsi < 50:
                    reasons.append(SignalReason(
                        factor="rsi_bearish_zone",
                        direction="BEARISH",
                        weight=-0.1,
                        description=f"RSI {rsi_tf} in bearish zone at {rsi:.1f}",
                    ))
                    score -= 0.1
        
        # Volume analysis
        if "1h" in momentum:
            vol_ratio = momentum["1h"].volume_ratio
            
            if vol_ratio > 1.5:
                reasons.append(SignalReason(
                    factor="high_volume",
                    direction="NEUTRAL",
                    weight=0.15,
                    description=f"High volume ({vol_ratio:.1f}x avg) - increased conviction",
                ))
                # Volume amplifies existing direction
                score = score * 1.2
            elif vol_ratio < 0.5:
                reasons.append(SignalReason(
                    factor="low_volume",
                    direction="NEUTRAL",
                    weight=-0.1,
                    description=f"Low volume ({vol_ratio:.1f}x avg) - weak conviction",
                ))
                score = score * 0.8
        
        return np.clip(score, -1.0, 1.0), reasons
    
    def _analyze_price_action(self, context: MarketContext) -> Tuple[float, Optional[SignalReason]]:
        """
        Check if price is actually moving in the expected direction.
        This prevents giving bullish signals when price is clearly falling.
        """
        score = 0.0
        reason = None
        
        # Get 5m candles to check recent price action
        if "5m" not in context.candle_data:
            return 0.0, None
        
        candles_5m = context.candle_data["5m"]
        
        if len(candles_5m) < 12:
            return 0.0, None
        
        # Check last hour of 5m candles (12 candles)
        last_hour_5m = candles_5m[-12:]
        price_change_1h = (last_hour_5m.close[-1] - last_hour_5m.close[0]) / last_hour_5m.close[0]
        
        # Count green vs red candles in last hour
        green_candles_5m = sum(1 for i in range(len(last_hour_5m)) if last_hour_5m.close[i] > last_hour_5m.open[i])
        red_candles_5m = sum(1 for i in range(len(last_hour_5m)) if last_hour_5m.close[i] < last_hour_5m.open[i])
        
        # Determine price action bias
        if price_change_1h > 0.002:  # More than 0.2% up in last hour
            if green_candles_5m > red_candles_5m + 2:  # Clear bullish candles
                score = 0.5
                reason = SignalReason(
                    factor="price_action",
                    direction="BULLISH",
                    weight=0.5,
                    description=f"Price rising (+{price_change_1h*100:.2f}% last hour, {green_candles_5m}/12 green candles)"
                )
        elif price_change_1h < -0.002:  # More than 0.2% down in last hour
            if red_candles_5m > green_candles_5m + 2:  # Clear bearish candles
                score = -0.5
                reason = SignalReason(
                    factor="price_action",
                    direction="BEARISH",
                    weight=-0.5,
                    description=f"Price falling ({price_change_1h*100:.2f}% last hour, {red_candles_5m}/12 red candles)"
                )
        
        return score, reason
    
    def _analyze_levels(
        self,
        price: float,
        supports: List[PriceLevel],
        resistances: List[PriceLevel],
    ) -> Tuple[float, List[SignalReason], List[str]]:
        """Analyze proximity to support/resistance."""
        reasons = []
        warnings = []
        score = 0.0
        
        # Check nearest support
        if supports:
            nearest_support = supports[0]
            distance_pct = (price - nearest_support.price) / price * 100
            
            if distance_pct < 0.3:
                # Very close to support
                reasons.append(SignalReason(
                    factor="at_support",
                    direction="BULLISH",
                    weight=0.3 * nearest_support.strength,
                    description=f"At strong support ${nearest_support.price:,.0f} ({distance_pct:.2f}% away)",
                ))
                score += 0.3 * nearest_support.strength
                warnings.append(f"🚫 At support - avoid SHORT until break below ${nearest_support.price:,.0f}")
            elif distance_pct < 1.0:
                reasons.append(SignalReason(
                    factor="near_support",
                    direction="BULLISH",
                    weight=0.15 * nearest_support.strength,
                    description=f"Near support ${nearest_support.price:,.0f} ({distance_pct:.2f}% away)",
                ))
                score += 0.15 * nearest_support.strength
        
        # Check nearest resistance
        if resistances:
            nearest_resistance = resistances[0]
            distance_pct = (nearest_resistance.price - price) / price * 100
            
            if distance_pct < 0.3:
                # Very close to resistance
                reasons.append(SignalReason(
                    factor="at_resistance",
                    direction="BEARISH",
                    weight=-0.3 * nearest_resistance.strength,
                    description=f"At strong resistance ${nearest_resistance.price:,.0f} ({distance_pct:.2f}% away)",
                ))
                score -= 0.3 * nearest_resistance.strength
                warnings.append(f"🚫 At resistance - avoid LONG until break above ${nearest_resistance.price:,.0f}")
            elif distance_pct < 1.0:
                reasons.append(SignalReason(
                    factor="near_resistance",
                    direction="BEARISH",
                    weight=-0.15 * nearest_resistance.strength,
                    description=f"Near resistance ${nearest_resistance.price:,.0f} ({distance_pct:.2f}% away)",
                ))
                score -= 0.15 * nearest_resistance.strength
        
        return np.clip(score, -1.0, 1.0), reasons, warnings
    
    def _analyze_pivots(
        self,
        price: float,
        pivots: AllPivotPoints
    ) -> Tuple[float, List[SignalReason]]:
        """Analyze pivot point positions and confluence."""
        reasons = []
        score = 0.0
        
        # Check confluence zones
        nearest_resistance = pivots.get_nearest_resistance(price)
        nearest_support = pivots.get_nearest_support(price)
        
        if nearest_support:
            support_price, strength, methods = nearest_support
            distance_pct = (price - support_price) / price * 100
            
            if distance_pct < 0.5 and strength >= 0.4:
                reasons.append(SignalReason(
                    factor="pivot_support_confluence",
                    direction="BULLISH",
                    weight=0.2 * strength,
                    description=f"Pivot confluence support at ${support_price:,.0f} ({len(methods)} methods: {', '.join(methods)})",
                ))
                score += 0.2 * strength
        
        if nearest_resistance:
            resist_price, strength, methods = nearest_resistance
            distance_pct = (resist_price - price) / price * 100
            
            if distance_pct < 0.5 and strength >= 0.4:
                reasons.append(SignalReason(
                    factor="pivot_resistance_confluence",
                    direction="BEARISH",
                    weight=-0.2 * strength,
                    description=f"Pivot confluence resistance at ${resist_price:,.0f} ({len(methods)} methods: {', '.join(methods)})",
                ))
                score -= 0.2 * strength
        
        # Position relative to daily pivot
        daily_pivot = pivots.traditional.pivot
        if price > daily_pivot:
            reasons.append(SignalReason(
                factor="above_daily_pivot",
                direction="BULLISH",
                weight=0.1,
                description=f"Price above daily pivot (${daily_pivot:,.0f})",
            ))
            score += 0.1
        else:
            reasons.append(SignalReason(
                factor="below_daily_pivot",
                direction="BEARISH",
                weight=-0.1,
                description=f"Price below daily pivot (${daily_pivot:,.0f})",
            ))
            score -= 0.1
        
        return np.clip(score, -1.0, 1.0), reasons
    
    def _analyze_smc(
        self,
        price: float,
        smc: SMCAnalysis
    ) -> Tuple[float, List[SignalReason], List[str]]:
        """Analyze Smart Money Concepts."""
        reasons = []
        warnings = []
        score = 0.0
        
        # Market structure bias
        if smc.current_bias == "BULLISH":
            reasons.append(SignalReason(
                factor="smc_bullish_structure",
                direction="BULLISH",
                weight=0.25,
                description="Market structure bullish (HH/HL pattern)",
            ))
            score += 0.25
        elif smc.current_bias == "BEARISH":
            reasons.append(SignalReason(
                factor="smc_bearish_structure",
                direction="BEARISH",
                weight=-0.25,
                description="Market structure bearish (LH/LL pattern)",
            ))
            score -= 0.25
        
        # Check for CHoCH (stronger signal)
        recent_choch = [b for b in smc.structure_breaks if b.type == "CHOCH"]
        if recent_choch:
            latest = recent_choch[-1]
            if latest.direction == "BULLISH":
                reasons.append(SignalReason(
                    factor="bullish_choch",
                    direction="BULLISH",
                    weight=0.3,
                    description=f"Bullish CHoCH - potential trend reversal up",
                ))
                score += 0.3
            else:
                reasons.append(SignalReason(
                    factor="bearish_choch",
                    direction="BEARISH",
                    weight=-0.3,
                    description=f"Bearish CHoCH - potential trend reversal down",
                ))
                score -= 0.3
        
        # Order block proximity
        for ob in smc.bullish_obs[:2]:
            if ob.bottom <= price <= ob.top and not ob.mitigated:
                reasons.append(SignalReason(
                    factor="in_bullish_ob",
                    direction="BULLISH",
                    weight=0.2 * ob.strength,
                    description=f"Price in bullish order block (${ob.bottom:,.0f}-${ob.top:,.0f})",
                ))
                score += 0.2 * ob.strength
                break
        
        for ob in smc.bearish_obs[:2]:
            if ob.bottom <= price <= ob.top and not ob.mitigated:
                reasons.append(SignalReason(
                    factor="in_bearish_ob",
                    direction="BEARISH",
                    weight=-0.2 * ob.strength,
                    description=f"Price in bearish order block (${ob.bottom:,.0f}-${ob.top:,.0f})",
                ))
                score -= 0.2 * ob.strength
                break
        
        # FVG proximity
        for fvg in smc.bullish_fvgs[:2]:
            if fvg.bottom <= price <= fvg.top:
                reasons.append(SignalReason(
                    factor="in_bullish_fvg",
                    direction="BULLISH",
                    weight=0.15,
                    description=f"Price in bullish FVG (${fvg.bottom:,.0f}-${fvg.top:,.0f})",
                ))
                score += 0.15
                break
        
        for fvg in smc.bearish_fvgs[:2]:
            if fvg.bottom <= price <= fvg.top:
                reasons.append(SignalReason(
                    factor="in_bearish_fvg",
                    direction="BEARISH",
                    weight=-0.15,
                    description=f"Price in bearish FVG (${fvg.bottom:,.0f}-${fvg.top:,.0f})",
                ))
                score -= 0.15
                break
        
        # Premium/Discount zone
        if price < smc.discount_zone[1]:
            reasons.append(SignalReason(
                factor="discount_zone",
                direction="BULLISH",
                weight=0.15,
                description=f"Price in discount zone (below ${smc.equilibrium:,.0f} EQ)",
            ))
            score += 0.15
        elif price > smc.premium_zone[0]:
            reasons.append(SignalReason(
                factor="premium_zone",
                direction="BEARISH",
                weight=-0.15,
                description=f"Price in premium zone (above ${smc.equilibrium:,.0f} EQ)",
            ))
            score -= 0.15
        
        # Liquidity pools - show in log but don't spam warnings
        # (warnings are handled separately in main.py)
        
        return np.clip(score, -1.0, 1.0), reasons, warnings
    
    def _determine_signal(
        self,
        final_score: float,
        trend_score: float,
        momentum_score: float,
        price_action_score: float,
        warnings: List[str],
    ) -> Tuple[SignalType, str, float]:
        """Determine signal type, direction, and confidence with STRICTER thresholds."""
        
        # FIXED: Stricter score thresholds to reduce false signals
        if final_score >= 0.5:  # Was 0.6 - Strong buy needs 0.5+
            signal_type = SignalType.STRONG_BUY
            direction = "LONG"
            base_confidence = 75 + (final_score - 0.5) * 50
        elif final_score >= 0.3:  # Was 0.35 - Buy needs 0.3+
            signal_type = SignalType.BUY
            direction = "LONG"
            base_confidence = 65 + (final_score - 0.3) * 50
        elif final_score >= 0.2:  # CRITICAL FIX: Was 0.15 - Weak buy needs 0.2+
            signal_type = SignalType.WEAK_BUY
            direction = "LONG"
            base_confidence = 50 + (final_score - 0.2) * 75
        elif final_score <= -0.5:  # Was -0.6
            signal_type = SignalType.STRONG_SELL
            direction = "SHORT"
            base_confidence = 75 + (abs(final_score) - 0.5) * 50
        elif final_score <= -0.3:  # Was -0.35
            signal_type = SignalType.SELL
            direction = "SHORT"
            base_confidence = 65 + (abs(final_score) - 0.3) * 50
        elif final_score <= -0.2:  # CRITICAL FIX: Was -0.15 - Weak sell needs -0.2 or lower
            signal_type = SignalType.WEAK_SELL
            direction = "SHORT"
            base_confidence = 50 + (abs(final_score) - 0.2) * 75
        else:
            # NEUTRAL zone is now -0.2 to 0.2 (was -0.15 to 0.15) - 33% wider
            signal_type = SignalType.NEUTRAL
            direction = "NONE"
            base_confidence = 0
        
        # NEW: Verify price action matches signal direction
        if direction == "LONG" and price_action_score < -0.2:
            # Trying to go long but price is falling - reduce confidence significantly
            base_confidence -= 20
            if base_confidence < 40:
                # Cancel the signal if confidence too low
                signal_type = SignalType.NEUTRAL
                direction = "NONE"
        elif direction == "SHORT" and price_action_score > 0.2:
            # Trying to go short but price is rising - reduce confidence
            base_confidence -= 20
            if base_confidence < 40:
                signal_type = SignalType.NEUTRAL
                direction = "NONE"
        
        # Adjust confidence based on trend alignment
        if direction == "LONG" and trend_score > 0.3:
            base_confidence += 5
        elif direction == "SHORT" and trend_score < -0.3:
            base_confidence += 5
        
        # Reduce confidence if there are warnings
        warning_penalty = min(15, len(warnings) * 5)
        base_confidence -= warning_penalty
        
        confidence = np.clip(base_confidence, 0, 100)
        
        return signal_type, direction, confidence
    
    def _generate_setup(
        self,
        price: float,
        direction: str,
        supports: List[PriceLevel],
        resistances: List[PriceLevel],
        pivots: Optional[AllPivotPoints],
        smc: Optional[SMCAnalysis],
    ) -> TradeSetup:
        """Generate entry, stop loss, and take profit levels."""
        
        if direction == "LONG":
            # Entry at current price
            entry = price
            
            # Stop loss below nearest support or 1.5% below entry
            if supports:
                sl_level = supports[0].price * 0.998  # Just below support
            else:
                sl_level = price * 0.985
            
            # Take profits at resistances or pivot levels
            tp_levels = []
            
            if resistances:
                tp_levels.extend([r.price for r in resistances[:3]])
            
            if pivots:
                for r in [pivots.traditional.r1, pivots.traditional.r2, pivots.fibonacci.r1]:
                    if r > price:
                        tp_levels.append(r)
            
            tp_levels = sorted(set(tp_levels))[:3]
            
            # Default TPs if none found
            if len(tp_levels) < 3:
                tp_levels = [
                    price * 1.01,
                    price * 1.02,
                    price * 1.035,
                ]
            
            invalidation = f"Break below ${sl_level:,.0f} support"
            
        else:  # SHORT
            entry = price
            
            if resistances:
                sl_level = resistances[0].price * 1.002
            else:
                sl_level = price * 1.015
            
            tp_levels = []
            
            if supports:
                tp_levels.extend([s.price for s in supports[:3]])
            
            if pivots:
                for s in [pivots.traditional.s1, pivots.traditional.s2, pivots.fibonacci.s1]:
                    if s < price:
                        tp_levels.append(s)
            
            tp_levels = sorted(set(tp_levels), reverse=True)[:3]
            
            if len(tp_levels) < 3:
                tp_levels = [
                    price * 0.99,
                    price * 0.98,
                    price * 0.965,
                ]
            
            invalidation = f"Break above ${sl_level:,.0f} resistance"
        
        # Calculate risk/reward
        risk = abs(entry - sl_level)
        reward = abs(tp_levels[1] - entry) if len(tp_levels) > 1 else abs(tp_levels[0] - entry)
        rr = reward / risk if risk > 0 else 0
        
        return TradeSetup(
            entry=entry,
            stop_loss=sl_level,
            take_profit_1=tp_levels[0] if tp_levels else entry,
            take_profit_2=tp_levels[1] if len(tp_levels) > 1 else entry,
            take_profit_3=tp_levels[2] if len(tp_levels) > 2 else entry,
            risk_reward_ratio=rr,
            position_invalidation=invalidation,
        )
    
    def _generate_summary(
        self,
        signal_type: SignalType,
        direction: str,
        confidence: float,
        reasons: List[SignalReason],
        warnings: List[str],
    ) -> str:
        """Generate human-readable summary."""
        
        if direction == "NONE":
            return "No clear signal - market conditions mixed. Wait for better setup."
        
        # Count bullish vs bearish reasons
        bullish = [r for r in reasons if r.direction == "BULLISH"]
        bearish = [r for r in reasons if r.direction == "BEARISH"]
        
        direction_word = "LONG" if direction == "LONG" else "SHORT"
        
        summary = f"{signal_type.value} signal ({confidence:.0f}% confidence). "
        summary += f"Suggesting {direction_word} position. "
        
        # Top reasons
        if direction == "LONG":
            top_reasons = sorted(bullish, key=lambda x: x.weight, reverse=True)[:3]
        else:
            top_reasons = sorted(bearish, key=lambda x: abs(x.weight), reverse=True)[:3]
        
        if top_reasons:
            summary += "Key factors: "
            summary += "; ".join([r.description for r in top_reasons])
        
        if warnings:
            summary += f" WARNINGS: {len(warnings)} concerns noted."
        
        return summary