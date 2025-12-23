"""
Prompt Builder for Market Analysis v2.0

Constructs structured prompts for LLM analysis based on enhanced market data.
Now includes:
- All 5 pivot methods (Traditional, Fibonacci, Camarilla, Woodie, DeMark)
- Self-assessment from past predictions
- Clear understanding that analysis runs after X candle closes
- Predictions for +1hr and +4hrs from prediction moment
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import json


def build_system_prompt(analysis_interval_candles: int = 5) -> str:
    """Build system prompt with context about timing and state-based analysis."""
    return f"""You are a senior cryptocurrency market analyst specializing in BTCUSDT short-term price prediction.

⚡ CRITICAL - UNDERSTAND YOUR ROLE:

YOUR PREDICTIONS STAY ACTIVE:
- Once you make a prediction, it remains LIVE until one of these events:
  1. ❌ Price crosses your INVALIDATION level (prediction proven wrong)
  2. ✅ Price reaches your 1h or 4h TARGET (prediction fulfilled)
  3. ⏰ 4+ hours pass with 2%+ price move (prediction becomes stale)
  4. 📥 Market-analyzer detects significant signal change (>60% confidence)

- You will ONLY be called again when one of these conditions occurs
- DO NOT make predictions lightly - they remain active until resolved
- Set your INVALIDATION level thoughtfully - it's the level that proves you WRONG

INVALIDATION RULES (CRITICAL):
- BULLISH prediction → invalidation must be BELOW current price (e.g., at support)
- BEARISH prediction → invalidation must be ABOVE current price (e.g., at resistance)
- Use critical support/resistance levels for invalidation placement
- Consider recent market structure when setting invalidation

TIMING CONTEXT:
- Market-analyzer runs after every {analysis_interval_candles} closed 1-minute candles
- Your predictions are for +1 HOUR and +4 HOURS from NOW
- This is SHORT-TERM prediction - small moves matter (0.2-2% is significant)
- Recent price action in the last few minutes is CRITICAL

Your role:
- Predict the MOST LIKELY price direction in the next 1-4 hours
- Give specific price targets (not vague ranges)
- Set clear invalidation level (where prediction is proven wrong)
- Be DECISIVE - your prediction stays active, so be confident
- Take a clear stance based on the data

You receive comprehensive market data including:
- Multi-timeframe trend analysis (5m, 15m, 1h, 4h)
- Smart Money Concepts (SMC): order blocks, FVGs, structure breaks, liquidity pools
- 5 pivot point methods with confluence zones
- Weighted signal factors from market analyzer
- Momentum indicators (RSI, volume, taker buy ratio)
- Recent price action (last 15-30 candles)

RESPONSE RULES:
- Provide comprehensive reasoning (max 1500 words)
- ALWAYS state direction: BULLISH / BEARISH / NEUTRAL
- ALWAYS give confidence: HIGH / MEDIUM / LOW
- ALWAYS give specific price targets for 1h and 4h
- ALWAYS specify support, resistance, and invalidation levels
- INVALIDATION must make sense with direction (see rules above)
- Explain your reasoning briefly but clearly

DO NOT:
- Hedge excessively ("could go either way")
- Give wide price ranges ("somewhere between X and Y")
- Skip any required sections
- Contradict yourself (e.g., bullish prediction with invalidation above price)
- Set invalidation too close to current price (allow some breathing room)

REMEMBER: Your prediction stays live until invalidated, fulfilled, or stale. Make it count!"""

def format_candles_for_prompt(candles: List[Dict[str, Any]], timeframe: str, limit: int = 30) -> str:
    """Format candles into a compact string for the prompt."""
    if not candles:
        return f"No {timeframe} candle data available."
    
    lines = [f"{timeframe} CANDLES (recent {min(len(candles), limit)}):"]
    lines.append("Time | Open | High | Low | Close | Volume")
    lines.append("-" * 60)
    
    for candle in candles[-limit:]:
        time_str = candle['open_time'].strftime('%m-%d %H:%M') if isinstance(candle['open_time'], datetime) else str(candle['open_time'])[:16]
        lines.append(
            f"{time_str} | {float(candle['open']):,.0f} | {float(candle['high']):,.0f} | "
            f"{float(candle['low']):,.0f} | {float(candle['close']):,.0f} | {float(candle['volume']):,.0f}"
        )
    
    # Add summary stats
    if len(candles) > 5:
        closes = [float(c['close']) for c in candles]
        highs = [float(c['high']) for c in candles]
        lows = [float(c['low']) for c in candles]
        
        lines.append("")
        lines.append(f"Period High: ${max(highs):,.0f} | Period Low: ${min(lows):,.0f}")
        lines.append(f"Range: ${max(highs) - min(lows):,.0f} ({(max(highs) - min(lows)) / min(lows) * 100:.2f}%)")
        
        # Recent momentum
        if len(closes) >= 5:
            recent_change = (closes[-1] - closes[-5]) / closes[-5] * 100
            direction = "UP" if recent_change > 0 else "DOWN"
            lines.append(f"Last 5 candles: {direction} {abs(recent_change):.2f}%")
    
    return "\n".join(lines)


def format_signal_factors(factors: Optional[List[Dict[str, Any]]]) -> str:
    """Format weighted signal factors into readable text."""
    if not factors:
        return "No signal factors available."
    
    lines = ["SIGNAL FACTORS (weighted reasons):"]
    lines.append("-" * 40)
    
    sorted_factors = sorted(factors, key=lambda x: abs(x.get('weight', 0)), reverse=True)
    
    bullish_total = 0
    bearish_total = 0
    
    for factor in sorted_factors[:10]:
        if isinstance(factor, dict):
            weight = float(factor.get('weight', 0))
            desc = factor.get('description', 'Unknown')
        else:
            weight = 0
            desc = str(factor)
        
        if weight > 0:
            bullish_total += weight
            symbol = "🟢"
        elif weight < 0:
            bearish_total += abs(weight)
            symbol = "🔴"
        else:
            symbol = "⚪"
        
        lines.append(f"  {symbol} {weight:+.2f} | {desc}")
    
    lines.append("")
    lines.append(f"Bullish weight: +{bullish_total:.2f} | Bearish weight: -{bearish_total:.2f}")
    net_bias = bullish_total - bearish_total
    bias_str = 'BULLISH' if net_bias > 0.1 else 'BEARISH' if net_bias < -0.1 else 'NEUTRAL'
    lines.append(f"Net bias: {bias_str} ({net_bias:+.2f})")
    
    return "\n".join(lines)


def format_all_pivot_levels(analysis: Dict[str, Any]) -> str:
    """Format all 5 pivot methods with complete levels."""
    lines = ["PIVOT POINTS (5 Methods):"]
    lines.append("-" * 60)
    
    # Daily pivot (main reference)
    pivot = analysis.get('pivot_daily') or analysis.get('daily_pivot')
    if pivot:
        price_vs = analysis.get('price_vs_pivot', 'N/A')
        lines.append(f"Daily Pivot: ${float(pivot):,.0f} (price {price_vs})")
        lines.append("")
    
    # Traditional pivots
    lines.append("TRADITIONAL:")
    trad_r = []
    trad_s = []
    for level in ['r3', 'r2', 'r1']:
        key = f'pivot_{level}_traditional'
        if analysis.get(key):
            trad_r.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    for level in ['s1', 's2', 's3']:
        key = f'pivot_{level}_traditional'
        if analysis.get(key):
            trad_s.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    if trad_r or trad_s:
        lines.append(f"  Resistance: {' | '.join(trad_r)}")
        lines.append(f"  Support:    {' | '.join(trad_s)}")
    
    # Fibonacci pivots
    lines.append("FIBONACCI:")
    fib_r = []
    fib_s = []
    for level in ['r3', 'r2', 'r1']:
        key = f'pivot_{level}_fibonacci'
        if analysis.get(key):
            fib_r.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    for level in ['s1', 's2', 's3']:
        key = f'pivot_{level}_fibonacci'
        if analysis.get(key):
            fib_s.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    if fib_r or fib_s:
        lines.append(f"  Resistance: {' | '.join(fib_r)}")
        lines.append(f"  Support:    {' | '.join(fib_s)}")
    
    # Camarilla pivots (complete R1-R4, S1-S4)
    lines.append("CAMARILLA (intraday breakout levels):")
    cam_r = []
    cam_s = []
    for level in ['r4', 'r3', 'r2', 'r1']:
        key = f'pivot_{level}_camarilla'
        if analysis.get(key):
            cam_r.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    for level in ['s1', 's2', 's3', 's4']:
        key = f'pivot_{level}_camarilla'
        if analysis.get(key):
            cam_s.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    if cam_r or cam_s:
        lines.append(f"  Resistance: {' | '.join(cam_r)}")
        lines.append(f"  Support:    {' | '.join(cam_s)}")
        lines.append("  Note: R3/S3 = key breakout levels, R4/S4 = extended targets")
    
    # Woodie pivots
    lines.append("WOODIE (trend-following):")
    woodie_pivot = analysis.get('pivot_woodie')
    woodie_r = []
    woodie_s = []
    for level in ['r3', 'r2', 'r1']:
        key = f'pivot_{level}_woodie'
        if analysis.get(key):
            woodie_r.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    for level in ['s1', 's2', 's3']:
        key = f'pivot_{level}_woodie'
        if analysis.get(key):
            woodie_s.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    if woodie_pivot:
        lines.append(f"  Pivot: ${float(woodie_pivot):,.0f}")
    if woodie_r or woodie_s:
        lines.append(f"  Resistance: {' | '.join(woodie_r)}")
        lines.append(f"  Support:    {' | '.join(woodie_s)}")
    
    # DeMark pivots
    lines.append("DEMARK (condition-based):")
    demark_pivot = analysis.get('pivot_demark')
    demark_r1 = analysis.get('pivot_r1_demark')
    demark_s1 = analysis.get('pivot_s1_demark')
    if demark_pivot:
        lines.append(f"  Pivot: ${float(demark_pivot):,.0f}")
    if demark_r1:
        lines.append(f"  Resistance (R1): ${float(demark_r1):,.0f}")
    if demark_s1:
        lines.append(f"  Support (S1): ${float(demark_s1):,.0f}")
    
    # Confluence zones
    confluence = analysis.get('pivot_confluence_zones')
    if confluence:
        if isinstance(confluence, str):
            try:
                confluence = json.loads(confluence)
            except:
                confluence = None
        
        if confluence and isinstance(confluence, list):
            lines.append("")
            lines.append("CONFLUENCE ZONES (multiple methods agree):")
            for zone in confluence[:5]:
                if isinstance(zone, dict):
                    zone_type = zone.get('type', 'unknown')
                    price = zone.get('price', 0)
                    strength = zone.get('strength', 0)
                    methods = zone.get('methods', [])
                    if isinstance(methods, list):
                        methods_str = ', '.join(str(m) for m in methods)
                    else:
                        methods_str = str(methods)
                    symbol = "🔴" if zone_type == "resistance" else "🟢"
                    lines.append(f"  {symbol} {zone_type.upper()} ${float(price):,.0f} (strength: {float(strength):.0%}, methods: {methods_str})")
    
    return "\n".join(lines) if len(lines) > 3 else "No pivot data available."


def format_smc_data(analysis: Dict[str, Any]) -> str:
    """Format Smart Money Concepts data."""
    lines = ["SMART MONEY CONCEPTS (SMC):"]
    lines.append("-" * 40)
    
    smc_bias = analysis.get('smc_bias')
    if smc_bias:
        bias_emoji = "🟢" if smc_bias == "BULLISH" else "🔴" if smc_bias == "BEARISH" else "🟡"
        lines.append(f"SMC Bias: {bias_emoji} {smc_bias}")
    
    price_zone = analysis.get('smc_price_zone') or analysis.get('price_zone')
    if price_zone:
        zone_emoji = "🟢" if price_zone == "DISCOUNT" else "🔴" if price_zone == "PREMIUM" else "🟡"
        lines.append(f"Price Zone: {zone_emoji} {price_zone}")
    
    equilibrium = analysis.get('smc_equilibrium') or analysis.get('equilibrium_price')
    if equilibrium:
        lines.append(f"Equilibrium: ${float(equilibrium):,.0f}")
    
    # Order Blocks
    order_blocks = analysis.get('smc_order_blocks')
    if order_blocks:
        if isinstance(order_blocks, str):
            try:
                order_blocks = json.loads(order_blocks)
            except:
                order_blocks = None
        
        if order_blocks and isinstance(order_blocks, list):
            lines.append("")
            lines.append("Order Blocks:")
            for ob in order_blocks[:4]:
                if isinstance(ob, dict):
                    ob_type = ob.get('type', 'unknown')
                    low = ob.get('low', 0)
                    high = ob.get('high', 0)
                    strength = ob.get('strength', 0)
                    symbol = "🟢" if ob_type == "bullish" else "🔴"
                    lines.append(f"  {symbol} {ob_type.upper()}: ${float(low):,.0f} - ${float(high):,.0f} (strength: {float(strength):.0%})")
    
    # FVGs
    fvgs = analysis.get('smc_fvgs')
    if fvgs:
        if isinstance(fvgs, str):
            try:
                fvgs = json.loads(fvgs)
            except:
                fvgs = None
        
        if fvgs and isinstance(fvgs, list):
            lines.append("")
            lines.append("Fair Value Gaps (Imbalances):")
            for fvg in fvgs[:4]:
                if isinstance(fvg, dict):
                    fvg_type = fvg.get('type', 'unknown')
                    low = fvg.get('low', 0)
                    high = fvg.get('high', 0)
                    unfilled = fvg.get('unfilled', True)
                    if unfilled:
                        symbol = "🟢" if fvg_type == "bullish" else "🔴"
                        lines.append(f"  {symbol} {fvg_type.upper()} FVG: ${float(low):,.0f} - ${float(high):,.0f} (UNFILLED)")
    
    # Structure breaks
    breaks = analysis.get('smc_breaks')
    if breaks:
        if isinstance(breaks, str):
            try:
                breaks = json.loads(breaks)
            except:
                breaks = None
        
        if breaks and isinstance(breaks, list):
            lines.append("")
            lines.append("Structure Breaks:")
            for brk in breaks[-3:]:
                if isinstance(brk, dict):
                    brk_type = brk.get('type', 'unknown')
                    direction = brk.get('direction', 'unknown')
                    price = brk.get('price', 0)
                    symbol = "🟢" if direction == "BULLISH" else "🔴"
                    lines.append(f"  {symbol} {brk_type}: {direction} at ${float(price):,.0f}")
    
    # Liquidity
    liquidity = analysis.get('smc_liquidity')
    if liquidity:
        if isinstance(liquidity, str):
            try:
                liquidity = json.loads(liquidity)
            except:
                liquidity = None
        
        if liquidity and isinstance(liquidity, dict):
            lines.append("")
            lines.append("Liquidity Pools (stop hunt targets):")
            buy_side = liquidity.get('buy_side', [])
            sell_side = liquidity.get('sell_side', [])
            if buy_side:
                levels = ", ".join([f"${float(l):,.0f}" for l in buy_side[:3]])
                lines.append(f"  📈 Buy-side (above): {levels}")
            if sell_side:
                levels = ", ".join([f"${float(l):,.0f}" for l in sell_side[:3]])
                lines.append(f"  📉 Sell-side (below): {levels}")
    
    return "\n".join(lines)


def format_support_resistance(analysis: Dict[str, Any]) -> str:
    """Format support and resistance levels."""
    lines = ["SUPPORT/RESISTANCE LEVELS:"]
    lines.append("-" * 40)
    
    # Simple levels
    nearest_sup = analysis.get('nearest_support')
    nearest_res = analysis.get('nearest_resistance')
    
    if nearest_res:
        res_strength = analysis.get('resistance_strength', 0)
        lines.append(f"Nearest Resistance: ${float(nearest_res):,.0f} (strength: {float(res_strength):.0%})")
    
    if nearest_sup:
        sup_strength = analysis.get('support_strength', 0)
        lines.append(f"Nearest Support: ${float(nearest_sup):,.0f} (strength: {float(sup_strength):.0%})")
    
    # Enhanced levels
    support_levels = analysis.get('support_levels')
    resistance_levels = analysis.get('resistance_levels')
    
    if resistance_levels:
        if isinstance(resistance_levels, str):
            try:
                resistance_levels = json.loads(resistance_levels)
            except:
                resistance_levels = None
        
        if resistance_levels and isinstance(resistance_levels, list):
            lines.append("")
            lines.append("All Resistance Levels:")
            for level in resistance_levels[:3]:
                if isinstance(level, dict):
                    price = level.get('price', 0)
                    strength = level.get('strength', 0)
                    touches = level.get('touches', 0)
                    dist = level.get('distance_pct', 0)
                    lines.append(f"  🔴 ${float(price):,.0f} | +{float(dist):.2f}% | strength: {float(strength):.0%} | touches: {touches}")
    
    if support_levels:
        if isinstance(support_levels, str):
            try:
                support_levels = json.loads(support_levels)
            except:
                support_levels = None
        
        if support_levels and isinstance(support_levels, list):
            lines.append("")
            lines.append("All Support Levels:")
            for level in support_levels[:3]:
                if isinstance(level, dict):
                    price = level.get('price', 0)
                    strength = level.get('strength', 0)
                    touches = level.get('touches', 0)
                    dist = level.get('distance_pct', 0)
                    lines.append(f"  🟢 ${float(price):,.0f} | -{float(dist):.2f}% | strength: {float(strength):.0%} | touches: {touches}")
    
    return "\n".join(lines)


def format_momentum(analysis: Dict[str, Any]) -> str:
    """Format momentum indicators."""
    lines = ["MOMENTUM INDICATORS:"]
    lines.append("-" * 40)
    
    # Simple 1H indicators
    rsi_1h = analysis.get('rsi_1h')
    vol_1h = analysis.get('volume_ratio_1h')
    
    if rsi_1h:
        rsi_status = "OVERSOLD 🟢" if rsi_1h < 30 else "OVERBOUGHT 🔴" if rsi_1h > 70 else "NEUTRAL"
        lines.append(f"RSI 1H: {float(rsi_1h):.1f} ({rsi_status})")
    
    if vol_1h:
        vol_status = "HIGH 📈" if vol_1h > 1.5 else "LOW 📉" if vol_1h < 0.5 else "NORMAL"
        lines.append(f"Volume 1H: {float(vol_1h):.2f}x average ({vol_status})")
    
    # Enhanced momentum
    momentum = analysis.get('momentum')
    if momentum:
        if isinstance(momentum, str):
            try:
                momentum = json.loads(momentum)
            except:
                momentum = None
        
        if momentum and isinstance(momentum, dict):
            lines.append("")
            for tf in ['5m', '15m', '1h', '4h']:
                if tf in momentum:
                    data = momentum[tf]
                    if isinstance(data, dict):
                        rsi = data.get('rsi', 0)
                        vol = data.get('volume_ratio', 0)
                        tbr = data.get('taker_buy_ratio', 0)
                        
                        rsi_emoji = "🟢" if rsi < 40 else "🔴" if rsi > 60 else "🟡"
                        tbr_emoji = "🟢" if tbr > 0.55 else "🔴" if tbr < 0.45 else "🟡"
                        
                        lines.append(f"  {tf}: RSI {rsi_emoji}{float(rsi):.1f} | Vol {float(vol):.2f}x | Buyers {tbr_emoji}{float(tbr):.0%}")
    
    return "\n".join(lines)


def format_trends(analysis: Dict[str, Any]) -> str:
    """Format multi-timeframe trends."""
    lines = ["MULTI-TIMEFRAME TRENDS:"]
    lines.append("-" * 40)
    
    trends = analysis.get('trends')
    if trends:
        if isinstance(trends, str):
            try:
                trends = json.loads(trends)
            except:
                trends = None
        
        if trends and isinstance(trends, dict):
            for tf in ['5m', '15m', '1h', '4h', '1d']:
                if tf in trends:
                    data = trends[tf]
                    if isinstance(data, dict):
                        direction = data.get('direction', 'N/A')
                        strength = data.get('strength', 0)
                        ema = data.get('ema', 'N/A')
                        
                        dir_emoji = "🟢" if direction == "UPTREND" else "🔴" if direction == "DOWNTREND" else "🟡"
                        lines.append(f"  {tf:>4}: {dir_emoji} {direction:<12} | Strength: {float(strength):.0%} | EMA: {ema}")
    
    return "\n".join(lines)


def format_warnings(analysis: Dict[str, Any]) -> str:
    """Format warnings."""
    warnings = analysis.get('warnings')
    if not warnings:
        return ""
    
    if isinstance(warnings, str):
        try:
            warnings = json.loads(warnings)
        except:
            return ""
    
    if not warnings or not isinstance(warnings, list):
        return ""
    
    lines = ["⚠️ WARNINGS:"]
    lines.append("-" * 40)
    for warning in warnings[:5]:
        if isinstance(warning, str):
            lines.append(f"  • {warning}")
        elif isinstance(warning, dict):
            msg = warning.get('message', str(warning))
            lines.append(f"  • {msg}")
    
    return "\n".join(lines)


def format_past_predictions(past_analyses: List[Dict[str, Any]]) -> str:
    """Format past prediction performance for self-assessment."""
    if not past_analyses:
        return "No past predictions available for self-assessment."
    
    lines = ["SELF-ASSESSMENT (Your Recent Predictions):"]
    lines.append("-" * 60)
    lines.append("Review your past predictions to improve accuracy:")
    lines.append("")
    
    correct_count = 0
    total_count = 0
    
    for analysis in past_analyses[:10]:
        time_str = analysis['analysis_time'].strftime('%m-%d %H:%M') if isinstance(analysis['analysis_time'], datetime) else str(analysis['analysis_time'])[:16]
        direction = analysis.get('prediction_direction', 'N/A')
        confidence = analysis.get('prediction_confidence', 'N/A')
        price_at_prediction = float(analysis.get('price', 0))
        predicted_1h = analysis.get('predicted_price_1h')
        actual_1h = analysis.get('actual_price_1h')
        correct_1h = analysis.get('direction_correct_1h')
        
        # Build line
        line = f"  {time_str}: {direction} ({confidence}) @ ${price_at_prediction:,.0f}"
        
        if actual_1h and predicted_1h:
            pred_dir = "UP" if predicted_1h > price_at_prediction else "DOWN"
            actual_dir = "UP" if actual_1h > price_at_prediction else "DOWN"
            result = "✅" if correct_1h else "❌"
            line += f" → Predicted: ${float(predicted_1h):,.0f} ({pred_dir})"
            line += f" | Actual: ${float(actual_1h):,.0f} ({actual_dir}) {result}"
            
            total_count += 1
            if correct_1h:
                correct_count += 1
        
        lines.append(line)
    
    if total_count > 0:
        accuracy = correct_count / total_count * 100
        lines.append("")
        lines.append(f"Direction Accuracy (1H): {correct_count}/{total_count} ({accuracy:.0f}%)")
        
        if accuracy < 50:
            lines.append("⚠️ Accuracy below 50% - Consider being more conservative or reversing bias")
        elif accuracy < 60:
            lines.append("📊 Accuracy moderate - Focus on high-confidence setups only")
        else:
            lines.append("✅ Good accuracy - Maintain current approach")
    
    return "\n".join(lines)


def format_market_analysis(analysis: Optional[Dict[str, Any]]) -> str:
    """Format complete market analysis data."""
    if not analysis:
        return "No market analysis data available."
    
    lines = []
    
    # Current signal
    signal_type = analysis.get('signal_type')
    signal_direction = analysis.get('signal_direction')
    signal_confidence = analysis.get('signal_confidence')
    
    if signal_type:
        dir_emoji = "🟢" if signal_direction == "LONG" else "🔴" if signal_direction == "SHORT" else "🟡"
        lines.append(f"MARKET ANALYZER SIGNAL: {dir_emoji} {signal_type} ({signal_direction})")
        lines.append(f"Confidence: {float(signal_confidence):.0f}%")
        lines.append("")
    
    # Signal factors
    signal_factors = analysis.get('signal_factors')
    if signal_factors:
        if isinstance(signal_factors, str):
            try:
                signal_factors = json.loads(signal_factors)
            except:
                signal_factors = None
        lines.append(format_signal_factors(signal_factors))
        lines.append("")
    
    # Trends
    lines.append(format_trends(analysis))
    lines.append("")
    
    # Support/Resistance
    lines.append(format_support_resistance(analysis))
    lines.append("")
    
    # Pivot levels (ALL 5 methods)
    lines.append(format_all_pivot_levels(analysis))
    lines.append("")
    
    # SMC data
    lines.append(format_smc_data(analysis))
    lines.append("")
    
    # Momentum
    lines.append(format_momentum(analysis))
    
    # Warnings
    warnings_str = format_warnings(analysis)
    if warnings_str:
        lines.append("")
        lines.append(warnings_str)
    
    return "\n".join(lines)


def format_signal_history(signals: List[Dict[str, Any]]) -> str:
    """Format signal change history."""
    if not signals:
        return "No signal history available."
    
    lines = ["RECENT SIGNAL CHANGES:"]
    lines.append("-" * 40)
    
    for sig in signals[:10]:
        time_str = sig['signal_time'].strftime('%m-%d %H:%M') if isinstance(sig['signal_time'], datetime) else str(sig['signal_time'])[:16]
        prev = sig.get('previous_signal_type', 'N/A')
        current = sig.get('signal_type', 'N/A')
        direction = sig.get('signal_direction', 'N/A')
        price = float(sig.get('price', 0))
        
        dir_emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "🟡"
        lines.append(f"{time_str}: {prev} → {current} ({dir_emoji}{direction}) @ ${price:,.0f}")
    
    return "\n".join(lines)


def build_analysis_prompt(
    candles_1h: List[Dict[str, Any]],
    candles_15m: List[Dict[str, Any]],
    market_analysis: Optional[Dict[str, Any]],
    signal_history: List[Dict[str, Any]],
    current_price: float,
    past_predictions: Optional[List[Dict[str, Any]]] = None,
    analysis_interval_candles: int = 5,
) -> str:
    """
    Build the complete analysis prompt with enhanced market data.
    
    Args:
        candles_1h: 1-hour candles (last ~120)
        candles_15m: 15-minute candles (last ~20)
        market_analysis: Latest market-analyzer output (enhanced schema)
        signal_history: Recent signal changes
        current_price: Current BTC price
        past_predictions: Past LLM predictions for self-assessment
        analysis_interval_candles: How many 1m candles between analyses
        
    Returns:
        Complete prompt string
    """
    sections = []
    
    # Header with timing context
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    sections.append(f"{'='*70}")
    sections.append(f"BTCUSDT SHORT-TERM PRICE PREDICTION REQUEST")
    sections.append(f"{'='*70}")
    sections.append(f"Timestamp: {timestamp}")
    sections.append(f"Current Price: ${current_price:,.2f}")
    sections.append(f"Analysis triggers every {analysis_interval_candles} closed 1m candles")
    sections.append(f"You are predicting price at +1 HOUR and +4 HOURS from NOW")
    sections.append("")
    
    # Self-assessment from past predictions (if available)
    if past_predictions:
        sections.append(format_past_predictions(past_predictions))
        sections.append("")
    
    # Market Analyzer Output (most important - put first)
    sections.append(format_market_analysis(market_analysis))
    sections.append("")
    
    # Signal History
    sections.append(format_signal_history(signal_history))
    sections.append("")
    
    # 1H Candles
    sections.append(format_candles_for_prompt(candles_1h, "1H", limit=30))
    sections.append("")
    
    # 15M Candles
    sections.append(format_candles_for_prompt(candles_15m, "15M", limit=20))
    sections.append("")
    
    # Request with strict format
    sections.append("=" * 70)
    sections.append("""
ANALYSIS REQUEST:

Based on ALL the data above, provide your SHORT-TERM prediction:

**IMPORTANT: Use this EXACT format:**

### 1. DIRECTION PREDICTION
- Direction: [BULLISH / BEARISH / NEUTRAL]
- Confidence: [HIGH / MEDIUM / LOW]

### 2. PRICE TARGETS
- Expected price in 1 hour: $XX,XXX
- Expected price in 4 hours: $XX,XXX
- Key invalidation level: $XX,XXX (if broken [below/above], prediction is wrong)

### 3. KEY LEVELS TO WATCH
- Critical support: $XX,XXX
- Critical resistance: $XX,XXX

### 4. BRIEF REASONING
[Your 3-4 sentence analysis explaining WHY you expect this direction]
- Reference the weighted signal factors
- Consider SMC bias and price zone
- Note any warnings
- Consider your past prediction accuracy

**CRITICAL RULES:**
✅ If BULLISH: invalidation should be BELOW current price
✅ If BEARISH: invalidation should be ABOVE current price
✅ Be specific with prices (not ranges)
✅ Base 1h/4h targets on S/R levels and pivot points
❌ Don't skip sections
❌ Don't hedge excessively
❌ Don't contradict yourself
""")
    
    return "\n".join(sections)


# Export system prompt builder
SYSTEM_PROMPT = build_system_prompt(5)  # Default 5 candles