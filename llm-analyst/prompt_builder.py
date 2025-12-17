"""
Prompt Builder for Market Analysis

Constructs structured prompts for LLM analysis based on enhanced market data.
Now includes all data from the enhanced market_analysis schema (v2.0).
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
import json


SYSTEM_PROMPT = """You are a senior cryptocurrency market analyst specializing in BTCUSDT technical analysis.

Your role:
- Analyze price action, trends, and market structure
- Identify key support/resistance levels
- Predict short-term price direction with confidence assessment
- Provide clear, actionable insights

You receive comprehensive market data including:
- Multi-timeframe trend analysis
- Smart Money Concepts (SMC) data: order blocks, FVGs, structure breaks, liquidity pools
- Multiple pivot point methods (Traditional, Fibonacci, Camarilla) with confluence zones
- Weighted signal factors from the market analyzer
- Momentum indicators across all timeframes
- Active warnings and risk alerts

Rules:
- Be concise and direct (max 300 words)
- Always state your prediction direction (BULLISH/BEARISH/NEUTRAL)
- Give confidence level (HIGH/MEDIUM/LOW)
- Mention specific price levels
- Consider the SMC bias and smart money positioning
- Pay attention to warnings - they highlight key risks
- Explain your reasoning briefly
- Do NOT hedge excessively - take a stance based on the data"""


def format_candles_for_prompt(candles: List[Dict[str, Any]], timeframe: str) -> str:
    """
    Format candles into a compact string for the prompt.
    
    Format: time|O|H|L|C|V
    """
    if not candles:
        return f"No {timeframe} candle data available."
    
    lines = [f"{timeframe} CANDLES (recent {len(candles)}):"]
    lines.append("Time | Open | High | Low | Close | Volume")
    lines.append("-" * 60)
    
    for candle in candles[-30:]:  # Only show last 30 in detail
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
        lines.append(f"Range: ${max(highs) - min(lows):,.0f} ({(max(highs) - min(lows)) / min(lows) * 100:.1f}%)")
        
        # Simple trend
        if len(closes) >= 20:
            sma_20 = sum(closes[-20:]) / 20
            current = closes[-1]
            trend = "ABOVE" if current > sma_20 else "BELOW"
            lines.append(f"Price vs SMA20: {trend} (SMA20: ${sma_20:,.0f})")
    
    return "\n".join(lines)


def format_signal_factors(factors: Optional[List[Dict[str, Any]]]) -> str:
    """Format weighted signal factors into readable text."""
    if not factors:
        return "No signal factors available."
    
    lines = ["SIGNAL FACTORS (weighted reasons):"]
    lines.append("-" * 40)
    
    # Sort by absolute weight
    sorted_factors = sorted(factors, key=lambda x: abs(x.get('weight', 0)), reverse=True)
    
    bullish_total = 0
    bearish_total = 0
    
    for factor in sorted_factors[:10]:  # Top 10 factors
        weight = int(factor.get('weight', 0))  # Convert to int for formatting
        desc = factor.get('description', 'Unknown')
        
        if weight > 0:
            bullish_total += weight
            symbol = "🟢"
        elif weight < 0:
            bearish_total += abs(weight)
            symbol = "🔴"
        else:
            symbol = "⚪"
        
        lines.append(f"  {symbol} {weight:+3d} | {desc}")
    
    lines.append("")
    lines.append(f"Bullish weight: +{int(bullish_total)} | Bearish weight: -{int(bearish_total)}")
    net_bias = int(bullish_total - bearish_total)
    lines.append(f"Net bias: {'BULLISH' if bullish_total > bearish_total else 'BEARISH' if bearish_total > bullish_total else 'NEUTRAL'} ({net_bias:+d})")
    
    return "\n".join(lines)


def format_pivot_levels(analysis: Dict[str, Any]) -> str:
    """Format all pivot levels from different methods."""
    lines = ["PIVOT POINTS:"]
    lines.append("-" * 40)
    
    # Daily pivot (main reference)
    pivot = analysis.get('pivot_daily') or analysis.get('daily_pivot')
    if pivot:
        price_vs = analysis.get('price_vs_pivot', 'N/A')
        lines.append(f"Daily Pivot: ${float(pivot):,.0f} (price {price_vs})")
    
    # Traditional pivots
    trad_levels = []
    for level in ['r3', 'r2', 'r1', 's1', 's2', 's3']:
        key = f'pivot_{level}_traditional'
        if analysis.get(key):
            trad_levels.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    
    if trad_levels:
        lines.append(f"Traditional: {' | '.join(trad_levels)}")
    
    # Fibonacci pivots
    fib_levels = []
    for level in ['r3', 'r2', 'r1', 's1', 's2', 's3']:
        key = f'pivot_{level}_fibonacci'
        if analysis.get(key):
            fib_levels.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    
    if fib_levels:
        lines.append(f"Fibonacci: {' | '.join(fib_levels)}")
    
    # Camarilla pivots (key breakout levels)
    cam_levels = []
    for level in ['r4', 'r3', 's3', 's4']:
        key = f'pivot_{level}_camarilla'
        if analysis.get(key):
            cam_levels.append(f"{level.upper()}: ${float(analysis[key]):,.0f}")
    
    if cam_levels:
        lines.append(f"Camarilla: {' | '.join(cam_levels)}")
    
    # Confluence zones (where multiple methods agree)
    confluence = analysis.get('pivot_confluence_zones')
    if confluence:
        if isinstance(confluence, str):
            try:
                confluence = json.loads(confluence)
            except:
                confluence = None
        
        if confluence and isinstance(confluence, list):
            lines.append("")
            lines.append("Confluence Zones (multiple methods agree):")
            for zone in confluence[:5]:  # Top 5 confluence zones
                zone_type = zone.get('type', 'unknown')
                price = zone.get('price', 0)
                strength = zone.get('strength', 0)
                methods = zone.get('methods', [])
                lines.append(f"  • {zone_type.upper()} ${float(price):,.0f} (strength: {strength:.0%}, methods: {', '.join(methods)})")
    
    return "\n".join(lines) if len(lines) > 2 else "No pivot data available."


def format_smc_data(analysis: Dict[str, Any]) -> str:
    """Format Smart Money Concepts data."""
    lines = ["SMART MONEY CONCEPTS (SMC):"]
    lines.append("-" * 40)
    
    # Basic SMC info
    smc_bias = analysis.get('smc_bias')
    if smc_bias:
        lines.append(f"SMC Bias: {smc_bias}")
    
    price_zone = analysis.get('smc_price_zone') or analysis.get('price_zone')
    if price_zone:
        lines.append(f"Price Zone: {price_zone}")
    
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
            lines.append("Active Order Blocks:")
            for ob in order_blocks[:5]:  # Top 5
                ob_type = ob.get('type', 'unknown')
                low = ob.get('low', 0)
                high = ob.get('high', 0)
                strength = ob.get('strength', 0)
                dist = ob.get('distance_pct', 0)
                lines.append(f"  • {ob_type.upper()} OB: ${float(low):,.0f}-${float(high):,.0f} (strength: {strength:.0%}, {dist:.1f}% away)")
    
    # Fair Value Gaps
    fvgs = analysis.get('smc_fvgs')
    if fvgs:
        if isinstance(fvgs, str):
            try:
                fvgs = json.loads(fvgs)
            except:
                fvgs = None
        
        if fvgs and isinstance(fvgs, list):
            unfilled = [f for f in fvgs if f.get('unfilled', True)]
            if unfilled:
                lines.append("")
                lines.append(f"Unfilled FVGs ({len(unfilled)}):")
                for fvg in unfilled[:3]:  # Top 3
                    fvg_type = fvg.get('type', 'unknown')
                    low = fvg.get('low', 0)
                    high = fvg.get('high', 0)
                    lines.append(f"  • {fvg_type.upper()} FVG: ${float(low):,.0f}-${float(high):,.0f}")
    
    # Structure Breaks (BOS, CHoCH)
    breaks = analysis.get('smc_breaks')
    if breaks:
        if isinstance(breaks, str):
            try:
                breaks = json.loads(breaks)
            except:
                breaks = None
        
        if breaks and isinstance(breaks, list):
            lines.append("")
            lines.append("Recent Structure Breaks:")
            for brk in breaks[:3]:  # Top 3
                brk_type = brk.get('type', 'unknown')
                direction = brk.get('direction', 'unknown')
                price = brk.get('price', 0)
                lines.append(f"  • {brk_type} {direction} @ ${float(price):,.0f}")
    
    # Liquidity Pools
    liquidity = analysis.get('smc_liquidity')
    if liquidity:
        if isinstance(liquidity, str):
            try:
                liquidity = json.loads(liquidity)
            except:
                liquidity = None
        
        if liquidity and isinstance(liquidity, dict):
            lines.append("")
            buy_side = liquidity.get('buy_side', [])
            sell_side = liquidity.get('sell_side', [])
            
            if buy_side:
                buy_prices = [f"${float(p):,.0f}" for p in buy_side[:3]]
                lines.append(f"Buy-side liquidity (stops above): {', '.join(buy_prices)}")
            
            if sell_side:
                sell_prices = [f"${float(p):,.0f}" for p in sell_side[:3]]
                lines.append(f"Sell-side liquidity (stops below): {', '.join(sell_prices)}")
    
    return "\n".join(lines) if len(lines) > 2 else "No SMC data available."


def format_support_resistance(analysis: Dict[str, Any]) -> str:
    """Format support and resistance levels."""
    lines = ["SUPPORT & RESISTANCE LEVELS:"]
    lines.append("-" * 40)
    
    # Enhanced levels (JSONB arrays)
    support_levels = analysis.get('support_levels')
    if support_levels:
        if isinstance(support_levels, str):
            try:
                support_levels = json.loads(support_levels)
            except:
                support_levels = None
        
        if support_levels and isinstance(support_levels, list):
            lines.append("Support Levels:")
            for level in support_levels[:5]:  # Top 5
                price = level.get('price', 0)
                strength = level.get('strength', 0)
                touches = level.get('touches', 0)
                tf = level.get('timeframe', 'N/A')
                dist = level.get('distance_pct', 0)
                lines.append(f"  • ${float(price):,.0f} (strength: {strength:.0%}, touches: {touches}, TF: {tf}, {dist:.2f}% away)")
    else:
        # Fall back to simple nearest support
        if analysis.get('nearest_support'):
            strength = analysis.get('support_strength', 0)
            lines.append(f"Nearest Support: ${float(analysis['nearest_support']):,.0f} (strength: {float(strength) * 100:.0f}%)")
    
    resistance_levels = analysis.get('resistance_levels')
    if resistance_levels:
        if isinstance(resistance_levels, str):
            try:
                resistance_levels = json.loads(resistance_levels)
            except:
                resistance_levels = None
        
        if resistance_levels and isinstance(resistance_levels, list):
            lines.append("")
            lines.append("Resistance Levels:")
            for level in resistance_levels[:5]:  # Top 5
                price = level.get('price', 0)
                strength = level.get('strength', 0)
                touches = level.get('touches', 0)
                tf = level.get('timeframe', 'N/A')
                dist = level.get('distance_pct', 0)
                lines.append(f"  • ${float(price):,.0f} (strength: {strength:.0%}, touches: {touches}, TF: {tf}, {dist:.2f}% away)")
    else:
        # Fall back to simple nearest resistance
        if analysis.get('nearest_resistance'):
            strength = analysis.get('resistance_strength', 0)
            lines.append(f"Nearest Resistance: ${float(analysis['nearest_resistance']):,.0f} (strength: {float(strength) * 100:.0f}%)")
    
    return "\n".join(lines) if len(lines) > 2 else "No S/R data available."


def format_momentum(analysis: Dict[str, Any]) -> str:
    """Format momentum indicators for all timeframes."""
    lines = ["MOMENTUM INDICATORS:"]
    lines.append("-" * 40)
    
    # Enhanced momentum (all timeframes as JSONB)
    momentum = analysis.get('momentum')
    if momentum:
        if isinstance(momentum, str):
            try:
                momentum = json.loads(momentum)
            except:
                momentum = None
        
        if momentum and isinstance(momentum, dict):
            for tf in ['5m', '15m', '1h', '4h', '1d']:
                if tf in momentum:
                    tf_data = momentum[tf]
                    rsi = tf_data.get('rsi', 0)
                    vol_ratio = tf_data.get('volume_ratio', 0)
                    taker_buy = tf_data.get('taker_buy_ratio', 0.5)
                    
                    rsi_status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
                    
                    lines.append(f"{tf}: RSI={rsi:.1f} ({rsi_status}), Vol={vol_ratio:.2f}x, TakerBuy={taker_buy:.0%}")
    else:
        # Fall back to single timeframe
        if analysis.get('rsi_1h'):
            rsi = float(analysis['rsi_1h'])
            rsi_status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
            lines.append(f"RSI 1H: {rsi:.1f} ({rsi_status})")
        
        if analysis.get('volume_ratio_1h'):
            lines.append(f"Volume Ratio 1H: {float(analysis['volume_ratio_1h']):.2f}x average")
    
    return "\n".join(lines) if len(lines) > 2 else "No momentum data available."


def format_market_structure(analysis: Dict[str, Any]) -> str:
    """Format market structure information."""
    lines = ["MARKET STRUCTURE:"]
    lines.append("-" * 40)
    
    pattern = analysis.get('structure_pattern')
    if pattern:
        lines.append(f"Pattern: {pattern}")
    
    last_high = analysis.get('structure_last_high')
    last_low = analysis.get('structure_last_low')
    
    if last_high:
        lines.append(f"Last Swing High: ${float(last_high):,.0f}")
    if last_low:
        lines.append(f"Last Swing Low: ${float(last_low):,.0f}")
    
    return "\n".join(lines) if len(lines) > 2 else ""


def format_warnings(analysis: Dict[str, Any]) -> str:
    """Format active warnings and alerts."""
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
    
    lines = ["⚠️ ACTIVE WARNINGS:"]
    lines.append("-" * 40)
    
    for warning in warnings:
        msg = warning.get('message', warning.get('type', 'Unknown warning'))
        severity = warning.get('severity', 'MEDIUM')
        
        icon = "🔴" if severity == 'HIGH' else "🟡" if severity == 'MEDIUM' else "🟢"
        lines.append(f"  {icon} [{severity}] {msg}")
    
    return "\n".join(lines)


def format_market_analysis(analysis: Optional[Dict[str, Any]]) -> str:
    """Format complete market-analyzer output for the prompt."""
    if not analysis:
        return "No market analysis data available."
    
    lines = ["=" * 60]
    lines.append("MARKET ANALYZER OUTPUT (Enhanced)")
    lines.append("=" * 60)
    
    # Basic signal info
    lines.append("")
    lines.append("CURRENT SIGNAL:")
    lines.append("-" * 40)
    lines.append(f"Signal: {analysis.get('signal_type', 'N/A')} ({analysis.get('signal_direction', 'N/A')})")
    lines.append(f"Confidence: {analysis.get('signal_confidence', 0):.0f}%")
    lines.append(f"Current Price: ${float(analysis.get('price', 0)):,.2f}")
    
    action = analysis.get('action_recommendation')
    if action:
        lines.append(f"Action Recommendation: {action}")
    
    # Trade setup if available
    if analysis.get('entry_price') and analysis.get('signal_direction') != 'NONE':
        lines.append("")
        lines.append("TRADE SETUP:")
        lines.append(f"  Entry: ${float(analysis['entry_price']):,.0f}")
        if analysis.get('stop_loss'):
            lines.append(f"  Stop Loss: ${float(analysis['stop_loss']):,.0f}")
        if analysis.get('take_profit_1'):
            lines.append(f"  TP1: ${float(analysis['take_profit_1']):,.0f}")
        if analysis.get('take_profit_2'):
            lines.append(f"  TP2: ${float(analysis['take_profit_2']):,.0f}")
        if analysis.get('risk_reward_ratio'):
            lines.append(f"  R:R: {float(analysis['risk_reward_ratio']):.2f}")
    
    # Signal factors
    signal_factors = analysis.get('signal_factors')
    if signal_factors:
        lines.append("")
        lines.append(format_signal_factors(signal_factors))
    
    # Trends
    trends = analysis.get('trends')
    if trends:
        if isinstance(trends, str):
            try:
                trends = json.loads(trends)
            except:
                trends = None
        
        if trends and isinstance(trends, dict):
            lines.append("")
            lines.append("MULTI-TIMEFRAME TRENDS:")
            lines.append("-" * 40)
            
            for tf in ['5m', '15m', '1h', '4h', '1d']:
                if tf in trends:
                    data = trends[tf]
                    if isinstance(data, dict):
                        direction = data.get('direction', 'N/A')
                        strength = data.get('strength', 0)
                        ema_bias = data.get('ema_bias', 'N/A')
                        lines.append(f"  {tf}: {direction} ({strength:.0%}) | EMA: {ema_bias}")
    
    # Support/Resistance
    lines.append("")
    lines.append(format_support_resistance(analysis))
    
    # Pivot levels
    lines.append("")
    lines.append(format_pivot_levels(analysis))
    
    # SMC data
    lines.append("")
    lines.append(format_smc_data(analysis))
    
    # Momentum
    lines.append("")
    lines.append(format_momentum(analysis))
    
    # Market structure
    structure = format_market_structure(analysis)
    if structure:
        lines.append("")
        lines.append(structure)
    
    # Warnings
    warnings = format_warnings(analysis)
    if warnings:
        lines.append("")
        lines.append(warnings)
    
    return "\n".join(lines)


def format_signal_history(signals: List[Dict[str, Any]]) -> str:
    """Format signal change history."""
    if not signals:
        return "No signal history available."
    
    lines = ["RECENT SIGNAL CHANGES:"]
    lines.append("-" * 40)
    
    for sig in signals[:10]:  # Last 10 signals
        time_str = sig['signal_time'].strftime('%m-%d %H:%M') if isinstance(sig['signal_time'], datetime) else str(sig['signal_time'])[:16]
        prev = sig.get('previous_signal_type', 'N/A')
        current = sig.get('signal_type', 'N/A')
        direction = sig.get('signal_direction', 'N/A')
        price = float(sig.get('price', 0))
        
        lines.append(f"{time_str}: {prev} → {current} ({direction}) @ ${price:,.0f}")
        
        # Show key reasons if available (enhanced)
        key_reasons = sig.get('key_reasons')
        if key_reasons:
            if isinstance(key_reasons, str):
                try:
                    key_reasons = json.loads(key_reasons)
                except:
                    key_reasons = None
            
            if key_reasons and isinstance(key_reasons, list):
                for reason in key_reasons[:2]:  # Top 2 reasons per signal
                    if isinstance(reason, dict):
                        desc = reason.get('description', str(reason))
                        weight = reason.get('weight', 0)
                        lines.append(f"    → {desc} ({weight:+d})")
                    else:
                        lines.append(f"    → {reason}")
    
    # Calculate signal stability
    if len(signals) >= 3:
        time_span = signals[0]['signal_time'] - signals[-1]['signal_time']
        hours = time_span.total_seconds() / 3600
        changes_per_hour = len(signals) / hours if hours > 0 else 0
        
        if changes_per_hour > 2:
            stability = "UNSTABLE (frequent changes)"
        elif changes_per_hour > 0.5:
            stability = "MODERATE"
        else:
            stability = "STABLE"
        
        lines.append(f"\nSignal Stability: {stability} ({len(signals)} changes in {hours:.1f}h)")
    
    return "\n".join(lines)


def build_analysis_prompt(
    candles_1h: List[Dict[str, Any]],
    candles_15m: List[Dict[str, Any]],
    market_analysis: Optional[Dict[str, Any]],
    signal_history: List[Dict[str, Any]],
    current_price: float,
) -> str:
    """
    Build the complete analysis prompt with enhanced market data.
    
    Args:
        candles_1h: 1-hour candles (last ~120)
        candles_15m: 15-minute candles (last ~20)
        market_analysis: Latest market-analyzer output (enhanced schema)
        signal_history: Recent signal changes
        current_price: Current BTC price
        
    Returns:
        Complete prompt string
    """
    sections = []
    
    # Header
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    sections.append(f"{'='*70}")
    sections.append(f"BTCUSDT COMPREHENSIVE MARKET ANALYSIS REQUEST")
    sections.append(f"{'='*70}")
    sections.append(f"Timestamp: {timestamp}")
    sections.append(f"Current Price: ${current_price:,.2f}")
    sections.append("")
    
    # Market Analyzer Output (most important - put first)
    sections.append(format_market_analysis(market_analysis))
    sections.append("")
    
    # Signal History
    sections.append(format_signal_history(signal_history))
    sections.append("")
    
    # 1H Candles (main data)
    sections.append(format_candles_for_prompt(candles_1h, "1H"))
    sections.append("")
    
    # 15M Candles (recent detail)
    sections.append(format_candles_for_prompt(candles_15m, "15M"))
    sections.append("")
    
    # Request
    sections.append("=" * 70)
    sections.append("""
ANALYSIS REQUEST:

Based on ALL the data above (signal factors, SMC, pivots, S/R, momentum, structure, warnings), provide:

1. DIRECTION PREDICTION (next 1-4 hours):
   - State clearly: BULLISH / BEARISH / NEUTRAL
   - Confidence: HIGH / MEDIUM / LOW

2. PRICE TARGETS:
   - Expected price in 1 hour: $XX,XXX
   - Expected price in 4 hours: $XX,XXX
   - Key invalidation level: $XX,XXX

3. KEY LEVELS TO WATCH:
   - Critical support: $XX,XXX
   - Critical resistance: $XX,XXX

4. BRIEF REASONING (3-4 sentences):
   - Reference the weighted signal factors
   - Consider SMC bias and liquidity targets
   - Note any relevant warnings
   - Explain your directional bias

Be direct and specific. Use the comprehensive data provided to make an informed prediction.""")
    
    return "\n".join(sections)