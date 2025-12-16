"""
Prompt Builder for Market Analysis

Constructs structured prompts for LLM analysis based on market data.
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

Rules:
- Be concise and direct (max 250 words)
- Always state your prediction direction (BULLISH/BEARISH/NEUTRAL)
- Give confidence level (HIGH/MEDIUM/LOW)
- Mention specific price levels
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


def format_market_analysis(analysis: Optional[Dict[str, Any]]) -> str:
    """Format market-analyzer output for the prompt."""
    if not analysis:
        return "No market analysis data available."
    
    lines = ["MARKET ANALYZER OUTPUT:"]
    lines.append("-" * 40)
    
    # Signal
    lines.append(f"Signal: {analysis.get('signal_type', 'N/A')} ({analysis.get('signal_direction', 'N/A')})")
    lines.append(f"Confidence: {analysis.get('signal_confidence', 0):.0f}%")
    
    # Price and levels
    lines.append(f"Current Price: ${float(analysis.get('price', 0)):,.2f}")
    
    if analysis.get('nearest_support'):
        lines.append(f"Nearest Support: ${float(analysis['nearest_support']):,.0f} (strength: {float(analysis.get('support_strength', 0)) * 100:.0f}%)")
    
    if analysis.get('nearest_resistance'):
        lines.append(f"Nearest Resistance: ${float(analysis['nearest_resistance']):,.0f} (strength: {float(analysis.get('resistance_strength', 0)) * 100:.0f}%)")
    
    # Trends
    if analysis.get('trends'):
        trends = analysis['trends']
        if isinstance(trends, str):
            trends = json.loads(trends)
        
        trend_strs = []
        for tf, data in trends.items():
            if isinstance(data, dict):
                direction = data.get('direction', 'N/A')
                strength = data.get('strength', 0)
                trend_strs.append(f"{tf}: {direction} ({strength:.0%})")
        
        if trend_strs:
            lines.append(f"Trends: {' | '.join(trend_strs)}")
    
    # SMC
    if analysis.get('smc_bias'):
        lines.append(f"SMC Bias: {analysis['smc_bias']} | Zone: {analysis.get('price_zone', 'N/A')}")
    
    if analysis.get('equilibrium_price'):
        lines.append(f"Equilibrium: ${float(analysis['equilibrium_price']):,.0f}")
    
    # Pivot
    if analysis.get('daily_pivot'):
        lines.append(f"Daily Pivot: ${float(analysis['daily_pivot']):,.0f} (price {analysis.get('price_vs_pivot', 'N/A')})")
    
    # Momentum
    if analysis.get('rsi_1h'):
        rsi = float(analysis['rsi_1h'])
        rsi_status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
        lines.append(f"RSI 1H: {rsi:.1f} ({rsi_status})")
    
    if analysis.get('volume_ratio_1h'):
        lines.append(f"Volume Ratio 1H: {float(analysis['volume_ratio_1h']):.2f}x average")
    
    # Trade setup if available
    if analysis.get('entry_price') and analysis.get('signal_direction') != 'NONE':
        lines.append("")
        lines.append("TRADE SETUP:")
        lines.append(f"  Entry: ${float(analysis['entry_price']):,.0f}")
        if analysis.get('stop_loss'):
            lines.append(f"  Stop Loss: ${float(analysis['stop_loss']):,.0f}")
        if analysis.get('take_profit_1'):
            lines.append(f"  TP1: ${float(analysis['take_profit_1']):,.0f}")
        if analysis.get('risk_reward_ratio'):
            lines.append(f"  R:R: {float(analysis['risk_reward_ratio']):.2f}")
    
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
    Build the complete analysis prompt.
    
    Args:
        candles_1h: 1-hour candles (last ~120)
        candles_15m: 15-minute candles (last ~20)
        market_analysis: Latest market-analyzer output
        signal_history: Recent signal changes
        current_price: Current BTC price
        
    Returns:
        Complete prompt string
    """
    sections = []
    
    # Header
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    sections.append(f"=== BTCUSDT MARKET ANALYSIS REQUEST ===")
    sections.append(f"Timestamp: {timestamp}")
    sections.append(f"Current Price: ${current_price:,.2f}")
    sections.append("")
    
    # 1H Candles (main data)
    sections.append(format_candles_for_prompt(candles_1h, "1H"))
    sections.append("")
    
    # 15M Candles (recent detail)
    sections.append(format_candles_for_prompt(candles_15m, "15M"))
    sections.append("")
    
    # Market Analyzer Output
    sections.append(format_market_analysis(market_analysis))
    sections.append("")
    
    # Signal History
    sections.append(format_signal_history(signal_history))
    sections.append("")
    
    # Request
    sections.append("=" * 50)
    sections.append("""
ANALYSIS REQUEST:

Based on the data above, provide:

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

4. BRIEF REASONING (2-3 sentences):
   Explain why you expect this direction.

Be direct and specific. Traders need clear guidance, not hedged opinions.""")
    
    return "\n".join(sections)
