"""
Signal Generator - v2.0 (Decisive Signals)

Generates BUY/SELL signals based on multi-factor confluence:
- Trend alignment across timeframes
- Support/Resistance proximity
- Momentum (RSI, volume)
- Smart Money Concepts (OB, FVG, structure)
- Pivot point confluence
- Risk/Reward analysis
- RECENT PRICE ACTION (critical for short-term predictions)

IMPORTANT: This analyzer works on 1m candle closes but looks at the bigger picture
to determine what the NEXT price movement will likely be. It must consider:
- Recent candle patterns (last 5-15 candles)
- Momentum shifts
- Volume confirmation
- Whether price is at key levels

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
    price_action_score: float = 0.0
    
    # Summary
    summary: str = ""


class SignalGenerator:
    """
    Multi-factor signal generator with DECISIVE bias.
    
    KEY PRINCIPLE: This runs after each 1m candle closes but looks at
    the BIGGER PICTURE to predict the NEXT price movement. It needs to:
    1. Identify the dominant direction based on higher timeframe trends
    2. Confirm with momentum and volume
    3. Use recent price action to time entries
    4. Only go NEUTRAL when truly indecisive (rare!)
    
    Combines:
    - Multi-timeframe trend analysis
    - Support/Resistance levels
    - Pivot points (all methods)
    - Smart Money Concepts
    - Momentum indicators
    - Volume analysis
    - RECENT PRICE ACTION
    """
    
    # Minimum confidence to generate a signal
    MIN_CONFIDENCE = 55  # Lowered to allow more signals
    
    # Score thresholds - more aggressive to generate clearer signals
    STRONG_THRESHOLD = 0.40  # Was 0.50 - Strong signals
    MEDIUM_THRESHOLD = 0.25  # Was 0.30 - Regular signals  
    WEAK_THRESHOLD = 0.12    # Was 0.20 - Weak signals
    
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
        
        This is the MAIN entry point. It analyzes all available data
        and produces a DECISIVE signal with clear direction.
        """
        reasons: List[SignalReason] = []
        warnings: List[str] = []
        
        current_price = context.current_price
        
        # ===== 1. TREND ANALYSIS (30% weight) =====
        # Higher timeframes should dominate
        trend_score, trend_reasons = self._analyze_trends(context.trends)
        reasons.extend(trend_reasons)
        
        # ===== 2. MOMENTUM ANALYSIS (20% weight) =====
        momentum_score, momentum_reasons = self._analyze_momentum(context.momentum)
        reasons.extend(momentum_reasons)
        
        # ===== 3. RECENT PRICE ACTION (20% weight) =====
        # Critical for short-term predictions!
        price_action_score, price_action_reasons = self._analyze_recent_price_action(context)
        reasons.extend(price_action_reasons)
        
        # ===== 4. SUPPORT/RESISTANCE (15% weight) =====
        level_score, level_reasons, level_warnings = self._analyze_levels(
            current_price,
            context.support_levels,
            context.resistance_levels,
        )
        reasons.extend(level_reasons)
        warnings.extend(level_warnings)
        
        # ===== 5. PIVOT POINTS (5% weight) =====
        if pivots:
            pivot_score, pivot_reasons = self._analyze_pivots(current_price, pivots)
            reasons.extend(pivot_reasons)
        else:
            pivot_score = 0
        
        # ===== 6. SMART MONEY CONCEPTS (10% weight) =====
        if smc:
            smc_score, smc_reasons, smc_warnings = self._analyze_smc(current_price, smc)
            reasons.extend(smc_reasons)
            warnings.extend(smc_warnings)
            structure_score = smc_score
        else:
            structure_score = 0
        
        # ===== CALCULATE FINAL SCORE =====
        # Weighted combination - price action is critical for timing
        final_score = (
            trend_score * 0.30 +        # Higher timeframe direction
            momentum_score * 0.20 +     # Momentum confirmation
            price_action_score * 0.20 + # Recent price action (CRITICAL)
            level_score * 0.15 +        # S/R levels
            pivot_score * 0.05 +        # Pivot confluence
            structure_score * 0.10      # SMC structure
        )
        
        # ===== DETERMINE SIGNAL TYPE =====
        signal_type, direction, confidence = self._determine_signal(
            final_score, trend_score, momentum_score, price_action_score,
            level_score, warnings
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
            price_action_score=price_action_score,
            summary=summary,
        )
    
    def _analyze_trends(
        self, 
        trends: Dict[str, TrendInfo]
    ) -> Tuple[float, List[SignalReason]]:
        """
        Analyze multi-timeframe trends with HIGHER TIMEFRAME DOMINANCE.
        
        Key principle: Trade in the direction of the higher timeframe trend.
        """
        reasons = []
        
        # Weight by timeframe importance (higher = more important)
        tf_weights = {
            "1d": 0.30,   # Daily trend is king
            "4h": 0.25,   # 4H strong influence
            "1h": 0.20,   # 1H good for entries
            "15m": 0.15,  # 15m timing
            "5m": 0.10,   # 5m noise but useful
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        # Analyze each timeframe
        for tf, weight in tf_weights.items():
            if tf not in trends:
                continue
            
            trend = trends[tf]
            total_weight += weight
            
            # Convert direction to score
            if trend.direction == "UPTREND":
                tf_score = trend.strength
            elif trend.direction == "DOWNTREND":
                tf_score = -trend.strength
            else:  # SIDEWAYS
                tf_score = 0
            
            weighted_score += tf_score * weight
            
            # Add reason for significant trends
            if abs(tf_score) > 0.3:
                direction = "BULLISH" if tf_score > 0 else "BEARISH"
                reasons.append(SignalReason(
                    factor=f"trend_{tf}",
                    direction=direction,
                    weight=tf_score * weight,
                    description=f"{tf} {trend.direction} ({trend.strength:.0%} strength)",
                ))
        
        # Normalize score
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0
        
        # Check for trend alignment (bonus for aligned trends)
        if len(trends) >= 3:
            bullish_count = sum(1 for t in trends.values() if t.direction == "UPTREND")
            bearish_count = sum(1 for t in trends.values() if t.direction == "DOWNTREND")
            
            if bullish_count >= 3:
                final_score += 0.15
                reasons.append(SignalReason(
                    factor="trend_alignment",
                    direction="BULLISH",
                    weight=0.15,
                    description=f"Bullish alignment across {bullish_count} timeframes",
                ))
            elif bearish_count >= 3:
                final_score -= 0.15
                reasons.append(SignalReason(
                    factor="trend_alignment",
                    direction="BEARISH",
                    weight=-0.15,
                    description=f"Bearish alignment across {bearish_count} timeframes",
                ))
        
        return np.clip(final_score, -1.0, 1.0), reasons
    
    def _analyze_momentum(
        self,
        momentum: Dict[str, MomentumInfo]
    ) -> Tuple[float, List[SignalReason]]:
        """
        Analyze momentum with RSI and volume.
        
        Key principle: Momentum should confirm the trend direction.
        """
        reasons = []
        score = 0.0
        count = 0
        
        # Priority timeframes for momentum
        priority_tfs = ["1h", "15m", "5m"]
        
        for tf in priority_tfs:
            if tf not in momentum:
                continue
            
            mom = momentum[tf]
            count += 1
            
            # RSI analysis
            rsi = mom.rsi
            if rsi < 30:
                # Oversold - bullish signal
                rsi_score = 0.3 + (30 - rsi) / 30 * 0.3  # 0.3 to 0.6
                reasons.append(SignalReason(
                    factor=f"rsi_{tf}_oversold",
                    direction="BULLISH",
                    weight=rsi_score,
                    description=f"{tf} RSI oversold at {rsi:.1f} - bounce likely",
                ))
                score += rsi_score
            elif rsi > 70:
                # Overbought - bearish signal
                rsi_score = -(0.3 + (rsi - 70) / 30 * 0.3)
                reasons.append(SignalReason(
                    factor=f"rsi_{tf}_overbought",
                    direction="BEARISH",
                    weight=rsi_score,
                    description=f"{tf} RSI overbought at {rsi:.1f} - pullback likely",
                ))
                score += rsi_score
            elif rsi > 55:
                # Bullish momentum
                score += 0.15
            elif rsi < 45:
                # Bearish momentum
                score -= 0.15
            
            # Volume analysis
            vol_ratio = mom.volume_ratio
            if vol_ratio > 1.5:
                # High volume - confirms move
                reasons.append(SignalReason(
                    factor=f"volume_{tf}_high",
                    direction="NEUTRAL",
                    weight=0.1,
                    description=f"{tf} high volume ({vol_ratio:.1f}x avg) - move confirmed",
                ))
                # Don't add to score directly - volume confirms direction
            elif vol_ratio < 0.5:
                # Low volume - weak move
                reasons.append(SignalReason(
                    factor=f"volume_{tf}_low",
                    direction="NEUTRAL",
                    weight=-0.05,
                    description=f"{tf} low volume ({vol_ratio:.1f}x avg) - weak conviction",
                ))
                score *= 0.8  # Reduce confidence
            
            # Taker buy ratio
            tbr = mom.taker_buy_ratio
            if tbr > 0.55:
                score += 0.1
                reasons.append(SignalReason(
                    factor=f"taker_buy_{tf}",
                    direction="BULLISH",
                    weight=0.1,
                    description=f"{tf} buyers dominant ({tbr:.1%} taker buys)",
                ))
            elif tbr < 0.45:
                score -= 0.1
                reasons.append(SignalReason(
                    factor=f"taker_sell_{tf}",
                    direction="BEARISH",
                    weight=-0.1,
                    description=f"{tf} sellers dominant ({tbr:.1%} taker buys)",
                ))
        
        if count > 0:
            score /= count
        
        return np.clip(score, -1.0, 1.0), reasons
    
    def _analyze_recent_price_action(
        self,
        context: MarketContext
    ) -> Tuple[float, List[SignalReason]]:
        """
        Analyze RECENT price action to determine immediate direction.
        
        This is CRITICAL for short-term predictions (1h, 4h).
        Looks at:
        - Recent candle patterns
        - Price momentum
        - Direction of recent moves
        """
        reasons = []
        score = 0.0
        
        # Get recent price data from context
        # We need to look at recent high/low/close movements
        
        # Check 5m trend for immediate direction
        if "5m" in context.trends:
            trend_5m = context.trends["5m"]
            if trend_5m.direction == "UPTREND":
                score += 0.25 * trend_5m.strength
                reasons.append(SignalReason(
                    factor="price_action_5m",
                    direction="BULLISH",
                    weight=0.25 * trend_5m.strength,
                    description=f"5m price action bullish ({trend_5m.strength:.0%})",
                ))
            elif trend_5m.direction == "DOWNTREND":
                score -= 0.25 * trend_5m.strength
                reasons.append(SignalReason(
                    factor="price_action_5m",
                    direction="BEARISH",
                    weight=-0.25 * trend_5m.strength,
                    description=f"5m price action bearish ({trend_5m.strength:.0%})",
                ))
        
        # Check 15m for confirmation
        if "15m" in context.trends:
            trend_15m = context.trends["15m"]
            if trend_15m.direction == "UPTREND":
                score += 0.20 * trend_15m.strength
                reasons.append(SignalReason(
                    factor="price_action_15m",
                    direction="BULLISH",
                    weight=0.20 * trend_15m.strength,
                    description=f"15m price action bullish ({trend_15m.strength:.0%})",
                ))
            elif trend_15m.direction == "DOWNTREND":
                score -= 0.20 * trend_15m.strength
                reasons.append(SignalReason(
                    factor="price_action_15m",
                    direction="BEARISH",
                    weight=-0.20 * trend_15m.strength,
                    description=f"15m price action bearish ({trend_15m.strength:.0%})",
                ))
        
        # Check momentum alignment with trend
        if "5m" in context.momentum:
            mom_5m = context.momentum["5m"]
            rsi = mom_5m.rsi
            
            # RSI momentum
            if 40 <= rsi <= 60:
                # Neutral RSI - look at direction
                pass
            elif rsi > 60:
                score += 0.15
                reasons.append(SignalReason(
                    factor="momentum_5m_bullish",
                    direction="BULLISH",
                    weight=0.15,
                    description=f"5m momentum bullish (RSI {rsi:.1f})",
                ))
            elif rsi < 40:
                score -= 0.15
                reasons.append(SignalReason(
                    factor="momentum_5m_bearish",
                    direction="BEARISH",
                    weight=-0.15,
                    description=f"5m momentum bearish (RSI {rsi:.1f})",
                ))
        
        return np.clip(score, -1.0, 1.0), reasons
    
    def _analyze_levels(
        self,
        price: float,
        supports: List[PriceLevel],
        resistances: List[PriceLevel],
    ) -> Tuple[float, List[SignalReason], List[str]]:
        """Analyze support/resistance levels."""
        reasons = []
        warnings = []
        score = 0.0
        
        # Check nearest support
        if supports:
            nearest_support = supports[0]
            dist_pct = (price - nearest_support.price) / price * 100
            
            if dist_pct < 0.3 and nearest_support.strength > 0.5:
                # Very close to strong support - bullish
                score += 0.25
                reasons.append(SignalReason(
                    factor="near_support",
                    direction="BULLISH",
                    weight=0.25,
                    description=f"Near strong support ${nearest_support.price:,.0f} ({dist_pct:.2f}% away)",
                ))
                warnings.append(f"Close to support ${nearest_support.price:,.0f} - short risky")
            elif dist_pct < 1.0:
                # Close to support
                score += 0.10
                reasons.append(SignalReason(
                    factor="support_nearby",
                    direction="BULLISH",
                    weight=0.10,
                    description=f"Support nearby at ${nearest_support.price:,.0f}",
                ))
        
        # Check nearest resistance
        if resistances:
            nearest_resistance = resistances[0]
            dist_pct = (nearest_resistance.price - price) / price * 100
            
            if dist_pct < 0.3 and nearest_resistance.strength > 0.5:
                # Very close to strong resistance - bearish
                score -= 0.25
                reasons.append(SignalReason(
                    factor="near_resistance",
                    direction="BEARISH",
                    weight=-0.25,
                    description=f"Near strong resistance ${nearest_resistance.price:,.0f} ({dist_pct:.2f}% away)",
                ))
                warnings.append(f"Close to resistance ${nearest_resistance.price:,.0f} - long risky")
            elif dist_pct < 1.0:
                # Close to resistance
                score -= 0.10
                reasons.append(SignalReason(
                    factor="resistance_nearby",
                    direction="BEARISH",
                    weight=-0.10,
                    description=f"Resistance nearby at ${nearest_resistance.price:,.0f}",
                ))
        
        return np.clip(score, -1.0, 1.0), reasons, warnings
    
    def _analyze_pivots(
        self,
        price: float,
        pivots: AllPivotPoints
    ) -> Tuple[float, List[SignalReason]]:
        """Analyze pivot points with confluence from all methods."""
        reasons = []
        score = 0.0
        
        # Check position vs daily pivot
        daily_pivot = pivots.traditional.pivot
        if price > daily_pivot:
            score += 0.10
            reasons.append(SignalReason(
                factor="above_pivot",
                direction="BULLISH",
                weight=0.10,
                description=f"Price above daily pivot ${daily_pivot:,.0f}",
            ))
        else:
            score -= 0.10
            reasons.append(SignalReason(
                factor="below_pivot",
                direction="BEARISH",
                weight=-0.10,
                description=f"Price below daily pivot ${daily_pivot:,.0f}",
            ))
        
        # Check confluence zones
        nearest_r = pivots.get_nearest_resistance(price)
        nearest_s = pivots.get_nearest_support(price)
        
        if nearest_s:
            conf_price, conf_strength, methods = nearest_s
            dist_pct = (price - conf_price) / price * 100
            if dist_pct < 0.5 and conf_strength >= 0.4:
                score += 0.15 * conf_strength
                reasons.append(SignalReason(
                    factor="pivot_confluence_support",
                    direction="BULLISH",
                    weight=0.15 * conf_strength,
                    description=f"Pivot confluence support ${conf_price:,.0f} ({conf_strength:.0%} strength, {', '.join(methods)})",
                ))
        
        if nearest_r:
            conf_price, conf_strength, methods = nearest_r
            dist_pct = (conf_price - price) / price * 100
            if dist_pct < 0.5 and conf_strength >= 0.4:
                score -= 0.15 * conf_strength
                reasons.append(SignalReason(
                    factor="pivot_confluence_resistance",
                    direction="BEARISH",
                    weight=-0.15 * conf_strength,
                    description=f"Pivot confluence resistance ${conf_price:,.0f} ({conf_strength:.0%} strength, {', '.join(methods)})",
                ))
        
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
            score += 0.20
            reasons.append(SignalReason(
                factor="smc_bias",
                direction="BULLISH",
                weight=0.20,
                description="SMC structure bias: BULLISH",
            ))
        elif smc.current_bias == "BEARISH":
            score -= 0.20
            reasons.append(SignalReason(
                factor="smc_bias",
                direction="BEARISH",
                weight=-0.20,
                description="SMC structure bias: BEARISH",
            ))
        
        # Structure breaks (CHoCH, BOS)
        recent_breaks = [b for b in smc.structure_breaks if b.type == "CHOCH"]
        if recent_breaks:
            latest_break = recent_breaks[-1]
            if latest_break.direction == "BULLISH":
                score += 0.25
                reasons.append(SignalReason(
                    factor="bullish_choch",
                    direction="BULLISH",
                    weight=0.25,
                    description=f"Bullish CHoCH at ${latest_break.break_level:,.0f}",
                ))
            else:
                score -= 0.25
                reasons.append(SignalReason(
                    factor="bearish_choch",
                    direction="BEARISH",
                    weight=-0.25,
                    description=f"Bearish CHoCH at ${latest_break.break_level:,.0f}",
                ))
        
        # Order blocks
        for ob in smc.bullish_obs[:2]:
            if ob.bottom <= price <= ob.top:
                score += 0.15 * ob.strength
                reasons.append(SignalReason(
                    factor="in_bullish_ob",
                    direction="BULLISH",
                    weight=0.15 * ob.strength,
                    description=f"Price in bullish OB (${ob.bottom:,.0f}-${ob.top:,.0f})",
                ))
                break
        
        for ob in smc.bearish_obs[:2]:
            if ob.bottom <= price <= ob.top:
                score -= 0.15 * ob.strength
                reasons.append(SignalReason(
                    factor="in_bearish_ob",
                    direction="BEARISH",
                    weight=-0.15 * ob.strength,
                    description=f"Price in bearish OB (${ob.bottom:,.0f}-${ob.top:,.0f})",
                ))
                break
        
        # FVGs
        for fvg in smc.bullish_fvgs[:2]:
            if fvg.bottom <= price <= fvg.top:
                score += 0.10
                reasons.append(SignalReason(
                    factor="in_bullish_fvg",
                    direction="BULLISH",
                    weight=0.10,
                    description=f"Price in bullish FVG (${fvg.bottom:,.0f}-${fvg.top:,.0f})",
                ))
                break
        
        for fvg in smc.bearish_fvgs[:2]:
            if fvg.bottom <= price <= fvg.top:
                score -= 0.10
                reasons.append(SignalReason(
                    factor="in_bearish_fvg",
                    direction="BEARISH",
                    weight=-0.10,
                    description=f"Price in bearish FVG (${fvg.bottom:,.0f}-${fvg.top:,.0f})",
                ))
                break
        
        # Premium/Discount zone
        if price < smc.discount_zone[1]:
            score += 0.15
            reasons.append(SignalReason(
                factor="discount_zone",
                direction="BULLISH",
                weight=0.15,
                description=f"Price in DISCOUNT zone (below ${smc.equilibrium:,.0f})",
            ))
        elif price > smc.premium_zone[0]:
            score -= 0.15
            reasons.append(SignalReason(
                factor="premium_zone",
                direction="BEARISH",
                weight=-0.15,
                description=f"Price in PREMIUM zone (above ${smc.equilibrium:,.0f})",
            ))
        
        return np.clip(score, -1.0, 1.0), reasons, warnings
    
    def _determine_signal(
        self,
        final_score: float,
        trend_score: float,
        momentum_score: float,
        price_action_score: float,
        level_score: float,
        warnings: List[str],
    ) -> Tuple[SignalType, str, float]:
        """
        Determine signal type with MORE DECISIVE thresholds.
        
        Key changes:
        - Lower thresholds to generate more signals
        - Narrower neutral zone
        - Consider alignment between factors
        """
        
        # Check for factor alignment (bonus confidence)
        alignment_bonus = 0
        if trend_score > 0 and momentum_score > 0 and price_action_score > 0:
            alignment_bonus = 10  # All bullish
        elif trend_score < 0 and momentum_score < 0 and price_action_score < 0:
            alignment_bonus = 10  # All bearish
        
        # Determine signal based on final score
        if final_score >= self.STRONG_THRESHOLD:
            signal_type = SignalType.STRONG_BUY
            direction = "LONG"
            base_confidence = 75 + (final_score - self.STRONG_THRESHOLD) * 50
        elif final_score >= self.MEDIUM_THRESHOLD:
            signal_type = SignalType.BUY
            direction = "LONG"
            base_confidence = 65 + (final_score - self.MEDIUM_THRESHOLD) * 50
        elif final_score >= self.WEAK_THRESHOLD:
            signal_type = SignalType.WEAK_BUY
            direction = "LONG"
            base_confidence = 55 + (final_score - self.WEAK_THRESHOLD) * 75
        elif final_score <= -self.STRONG_THRESHOLD:
            signal_type = SignalType.STRONG_SELL
            direction = "SHORT"
            base_confidence = 75 + (abs(final_score) - self.STRONG_THRESHOLD) * 50
        elif final_score <= -self.MEDIUM_THRESHOLD:
            signal_type = SignalType.SELL
            direction = "SHORT"
            base_confidence = 65 + (abs(final_score) - self.MEDIUM_THRESHOLD) * 50
        elif final_score <= -self.WEAK_THRESHOLD:
            signal_type = SignalType.WEAK_SELL
            direction = "SHORT"
            base_confidence = 55 + (abs(final_score) - self.WEAK_THRESHOLD) * 75
        else:
            # Neutral zone is now very narrow: -0.12 to 0.12
            signal_type = SignalType.NEUTRAL
            direction = "NONE"
            base_confidence = 0
        
        # Apply alignment bonus
        base_confidence += alignment_bonus
        
        # Verify price action matches signal direction
        if direction == "LONG" and price_action_score < -0.3:
            base_confidence -= 15  # Price action contradicts
        elif direction == "SHORT" and price_action_score > 0.3:
            base_confidence -= 15  # Price action contradicts
        
        # Reduce confidence for warnings
        warning_penalty = min(10, len(warnings) * 3)
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
            entry = price
            
            # Stop loss below nearest support or 1% below entry
            if supports:
                sl_level = supports[0].price * 0.998
            else:
                sl_level = price * 0.99
            
            # Take profits at resistances or pivot levels
            tp_levels = []
            
            if resistances:
                tp_levels.extend([r.price for r in resistances[:3]])
            
            if pivots:
                for r in [pivots.traditional.r1, pivots.traditional.r2, pivots.fibonacci.r1]:
                    if r > price:
                        tp_levels.append(r)
            
            tp_levels = sorted(set(tp_levels))[:3]
            
            if len(tp_levels) < 3:
                tp_levels = [
                    price * 1.008,
                    price * 1.015,
                    price * 1.025,
                ]
            
            invalidation = f"Break below ${sl_level:,.0f}"
            
        else:  # SHORT
            entry = price
            
            if resistances:
                sl_level = resistances[0].price * 1.002
            else:
                sl_level = price * 1.01
            
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
                    price * 0.992,
                    price * 0.985,
                    price * 0.975,
                ]
            
            invalidation = f"Break above ${sl_level:,.0f}"
        
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
            return "NEUTRAL - Market conditions indecisive. Score too close to zero for directional bias. Wait for clearer setup."
        
        # Count bullish vs bearish reasons
        bullish = [r for r in reasons if r.direction == "BULLISH"]
        bearish = [r for r in reasons if r.direction == "BEARISH"]
        
        direction_word = "LONG" if direction == "LONG" else "SHORT"
        
        summary = f"{signal_type.value} ({confidence:.0f}% confidence) - {direction_word}. "
        
        # Top reasons
        if direction == "LONG":
            top_reasons = sorted(bullish, key=lambda x: x.weight, reverse=True)[:3]
        else:
            top_reasons = sorted(bearish, key=lambda x: abs(x.weight), reverse=True)[:3]
        
        if top_reasons:
            summary += "Key: "
            summary += " | ".join([r.description for r in top_reasons])
        
        if warnings:
            summary += f" ⚠ {len(warnings)} warnings"
        
        return summary