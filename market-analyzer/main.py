#!/usr/bin/env python3
"""
Market Analyzer - Real-time multi-timeframe market context analysis for BTCUSDT.

Listens for new 1m candles and produces human-readable market context reports
with trading signals, pivot points, and Smart Money Concepts analysis.
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from config import Config
from database import Database
from analyzer import MarketAnalyzer
from models import MarketContext


# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)


class MarketAnalyzerService:
    """Main service that orchestrates market analysis."""
    
    def __init__(self):
        self.config = Config()
        self.db: Optional[Database] = None
        self.analyzer: Optional[MarketAnalyzer] = None
        self.running = False
        self.last_candle_time: Optional[datetime] = None
        
    async def start(self):
        """Initialize and start the service."""
        logger.info("=" * 60)
        logger.info("🚀 MARKET ANALYZER SERVICE STARTING")
        logger.info("=" * 60)
        logger.info("Features: Multi-TF Trends, Pivots, SMC, Trading Signals")
        logger.info("")
        
        # Connect to database
        self.db = Database(self.config)
        await self.db.connect()
        
        # Initialize analyzer
        self.analyzer = MarketAnalyzer(self.db, self.config)
        
        # Get initial candle time
        self.last_candle_time = await self.db.get_latest_candle_time()
        if self.last_candle_time:
            logger.info(f"Last candle in DB: {self.last_candle_time}")
        else:
            logger.warning("No candles found in database yet")
        
        # Run initial analysis
        await self.run_analysis()
        
        self.running = True
        logger.info(f"Polling for new candles every {self.config.poll_interval_seconds}s")
        
        # Main loop
        while self.running:
            try:
                await self.check_for_new_candle()
                await asyncio.sleep(self.config.poll_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
    
    async def check_for_new_candle(self):
        """Check if a new candle has arrived and trigger analysis."""
        current_latest = await self.db.get_latest_candle_time()
        
        if current_latest is None:
            return
            
        if self.last_candle_time is None or current_latest > self.last_candle_time:
            logger.info(f"📊 New candle detected: {current_latest}")
            self.last_candle_time = current_latest
            await self.run_analysis()
    
    async def run_analysis(self):
        """Run full market analysis and log results."""
        if not self.analyzer:
            return
            
        try:
            context = await self.analyzer.analyze()
            if context:
                self.log_market_context(context)
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def log_market_context(self, ctx: MarketContext):
        """Log market context in a readable format."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"📈 MARKET CONTEXT REPORT - {ctx.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("=" * 80)
        
        # Current price
        logger.info(f"💰 Current Price: ${ctx.current_price:,.2f}")
        logger.info("")
        
        # ===== TRADING SIGNAL (most important - show first) =====
        if ctx.signal:
            self.log_signal(ctx.signal)
        
        # ===== MULTI-TIMEFRAME TRENDS =====
        logger.info("📊 TREND ANALYSIS (Multi-Timeframe)")
        logger.info("-" * 70)
        for tf, trend in ctx.trends.items():
            trend_emoji = "🟢" if trend.direction == "UPTREND" else "🔴" if trend.direction == "DOWNTREND" else "🟡"
            hh_hl = ""
            if trend.higher_highs and trend.higher_lows:
                hh_hl = "HH/HL ✓"
            elif trend.lower_highs and trend.lower_lows:
                hh_hl = "LH/LL ✓"
            logger.info(f"  {tf:>4}: {trend_emoji} {trend.direction:<12} | Strength: {trend.strength:.0%} | EMA: {trend.ema_alignment:<8} | {hh_hl}")
        logger.info("")
        
        # ===== PIVOT POINTS =====
        if ctx.pivots:
            self.log_pivots(ctx.pivots, ctx.current_price)
        
        # ===== SMART MONEY CONCEPTS =====
        if ctx.smc:
            self.log_smc(ctx.smc, ctx.current_price)
        
        # ===== SUPPORT/RESISTANCE =====
        logger.info("🎯 KEY SUPPORT/RESISTANCE LEVELS")
        logger.info("-" * 70)
        if ctx.support_levels:
            for i, level in enumerate(ctx.support_levels[:3]):
                distance_pct = ((ctx.current_price - level.price) / ctx.current_price) * 100
                strength_bar = "█" * int(level.strength * 10)
                logger.info(f"  Support {i+1}: ${level.price:,.0f} ({distance_pct:+.2f}%) | {strength_bar} {level.strength:.0%} | Touches: {level.touches} | TF: {level.source_timeframe}")
        else:
            logger.info("  No significant supports detected")
            
        if ctx.resistance_levels:
            for i, level in enumerate(ctx.resistance_levels[:3]):
                distance_pct = ((level.price - ctx.current_price) / ctx.current_price) * 100
                strength_bar = "█" * int(level.strength * 10)
                logger.info(f"  Resist {i+1}:  ${level.price:,.0f} (+{distance_pct:.2f}%) | {strength_bar} {level.strength:.0%} | Touches: {level.touches} | TF: {level.source_timeframe}")
        else:
            logger.info("  No significant resistances detected")
        logger.info("")
        
        # ===== MOMENTUM =====
        logger.info("⚡ MOMENTUM INDICATORS")
        logger.info("-" * 70)
        for tf, momentum in ctx.momentum.items():
            rsi_status = "OVERSOLD 🟢" if momentum.rsi < 30 else "OVERBOUGHT 🔴" if momentum.rsi > 70 else "NEUTRAL 🟡"
            vol_emoji = "📈" if momentum.volume_ratio > 1.5 else "📉" if momentum.volume_ratio < 0.5 else "➖"
            taker_emoji = "🟢" if momentum.taker_buy_ratio > 0.55 else "🔴" if momentum.taker_buy_ratio < 0.45 else "🟡"
            logger.info(f"  {tf:>4}: RSI {momentum.rsi:5.1f} {rsi_status:<16} | Vol: {vol_emoji} {momentum.volume_ratio:.2f}x | Taker Buy: {taker_emoji} {momentum.taker_buy_ratio:.0%}")
        logger.info("")
        
        # ===== MARKET STRUCTURE =====
        logger.info("🏗️  MARKET STRUCTURE")
        logger.info("-" * 70)
        pattern_emoji = "🟢" if "HIGHER" in ctx.structure.pattern else "🔴" if "LOWER" in ctx.structure.pattern else "🟡"
        logger.info(f"  Pattern: {pattern_emoji} {ctx.structure.pattern}")
        if ctx.structure.last_swing_high:
            logger.info(f"  Last Swing High: ${ctx.structure.last_swing_high:,.2f}")
        if ctx.structure.last_swing_low:
            logger.info(f"  Last Swing Low:  ${ctx.structure.last_swing_low:,.2f}")
        logger.info("")
        
        # ===== WARNINGS =====
        logger.info("⚠️  WARNINGS & RISK ALERTS")
        logger.info("-" * 70)
        
        warnings = self.generate_warnings(ctx)
        if ctx.signal and ctx.signal.warnings:
            warnings.extend(ctx.signal.warnings)
        
        # Deduplicate
        warnings = list(dict.fromkeys(warnings))
        
        if warnings:
            for warning in warnings:
                logger.warning(f"  {warning}")
        else:
            logger.info("  ✅ No specific warnings - conditions favorable")
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("")
    
    def log_signal(self, signal):
        """Log trading signal prominently."""
        # Big signal banner
        if signal.direction == "LONG":
            logger.info("")
            logger.info("🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢")
            logger.info(f"🚀 SIGNAL: {signal.signal_type.value}")
            logger.info(f"   Direction: LONG 📈 | Confidence: {signal.confidence:.0f}%")
            logger.info("🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢")
        elif signal.direction == "SHORT":
            logger.info("")
            logger.info("🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴")
            logger.info(f"📉 SIGNAL: {signal.signal_type.value}")
            logger.info(f"   Direction: SHORT 📉 | Confidence: {signal.confidence:.0f}%")
            logger.info("🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴")
        else:
            logger.info("")
            logger.info("🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡")
            logger.info(f"⏸️  SIGNAL: NEUTRAL - No clear setup")
            logger.info(f"   Wait for better conditions")
            logger.info("🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡🟡")
        
        logger.info("")
        
        # Trade setup if available
        if signal.setup and signal.direction != "NONE":
            logger.info("📋 TRADE SETUP")
            logger.info("-" * 70)
            
            entry = signal.setup.entry
            sl = signal.setup.stop_loss
            tp1 = signal.setup.take_profit_1
            tp2 = signal.setup.take_profit_2
            tp3 = signal.setup.take_profit_3
            
            sl_pct = (sl - entry) / entry * 100
            tp1_pct = (tp1 - entry) / entry * 100
            tp2_pct = (tp2 - entry) / entry * 100
            tp3_pct = (tp3 - entry) / entry * 100
            
            logger.info(f"  📍 Entry:         ${entry:,.2f}")
            logger.info(f"  🛑 Stop Loss:     ${sl:,.2f} ({sl_pct:+.2f}%)")
            logger.info(f"  🎯 Take Profit 1: ${tp1:,.2f} ({tp1_pct:+.2f}%) - Scale out 33%")
            logger.info(f"  🎯 Take Profit 2: ${tp2:,.2f} ({tp2_pct:+.2f}%) - Scale out 33%")
            logger.info(f"  🎯 Take Profit 3: ${tp3:,.2f} ({tp3_pct:+.2f}%) - Final exit")
            logger.info(f"  📊 Risk/Reward:   {signal.setup.risk_reward_ratio:.2f}R")
            logger.info(f"  ❌ Invalidation:  {signal.setup.position_invalidation}")
            logger.info("")
        
        # Signal reasoning (top factors)
        logger.info("🧠 SIGNAL REASONING (Top Factors)")
        logger.info("-" * 70)
        
        # Sort by absolute weight
        sorted_reasons = sorted(signal.reasons, key=lambda x: abs(x.weight), reverse=True)
        
        for reason in sorted_reasons[:10]:  # Top 10 reasons
            if reason.direction == "BULLISH":
                emoji = "🟢"
                sign = "+"
            elif reason.direction == "BEARISH":
                emoji = "🔴"
                sign = "-"
            else:
                emoji = "🟡"
                sign = " "
            
            weight_pct = abs(reason.weight) * 100
            bar_len = int(weight_pct / 5)
            weight_bar = "█" * bar_len + "░" * (20 - bar_len)
            
            logger.info(f"  {emoji} [{sign}{weight_pct:4.0f}%] {weight_bar} {reason.description[:50]}")
        
        logger.info("")
        logger.info(f"📝 Summary: {signal.summary}")
        logger.info("")
    
    def log_pivots(self, pivots, current_price: float):
        """Log pivot point analysis."""
        logger.info("📐 PIVOT POINTS (Based on Daily)")
        logger.info("-" * 70)
        
        # Position relative to pivots
        t = pivots.traditional
        pos = "ABOVE" if current_price > t.pivot else "BELOW"
        pos_emoji = "🟢" if pos == "ABOVE" else "🔴"
        logger.info(f"  Price is {pos_emoji} {pos} daily pivot (${t.pivot:,.0f})")
        logger.info("")
        
        # Traditional pivots
        logger.info(f"  Traditional:  R3 ${t.r3:>10,.0f} | R2 ${t.r2:>10,.0f} | R1 ${t.r1:>10,.0f} | P ${t.pivot:>10,.0f} | S1 ${t.s1:>10,.0f} | S2 ${t.s2:>10,.0f} | S3 ${t.s3:>10,.0f}")
        
        # Fibonacci pivots
        f = pivots.fibonacci
        logger.info(f"  Fibonacci:    R3 ${f.r3:>10,.0f} | R2 ${f.r2:>10,.0f} | R1 ${f.r1:>10,.0f} | P ${f.pivot:>10,.0f} | S1 ${f.s1:>10,.0f} | S2 ${f.s2:>10,.0f} | S3 ${f.s3:>10,.0f}")
        
        # Camarilla (key levels for intraday)
        c = pivots.camarilla
        logger.info(f"  Camarilla:    R4 ${c.r4:>10,.0f} | R3 ${c.r3:>10,.0f} |                      | S3 ${c.s3:>10,.0f} | S4 ${c.s4:>10,.0f}")
        
        # Confluence zones
        logger.info("")
        nearest_r = pivots.get_nearest_resistance(current_price)
        nearest_s = pivots.get_nearest_support(current_price)
        
        if nearest_s or nearest_r:
            logger.info("  🎯 Pivot Confluence Zones (multiple methods agree):")
            
            if nearest_s:
                price, strength, methods = nearest_s
                dist = (current_price - price) / current_price * 100
                logger.info(f"     🟢 Support:    ${price:,.0f} ({dist:+.2f}%) | Strength: {strength:.0%} | Methods: {', '.join(methods)}")
            
            if nearest_r:
                price, strength, methods = nearest_r
                dist = (price - current_price) / current_price * 100
                logger.info(f"     🔴 Resistance: ${price:,.0f} (+{dist:.2f}%) | Strength: {strength:.0%} | Methods: {', '.join(methods)}")
        
        logger.info("")
    
    def log_smc(self, smc, current_price: float):
        """Log Smart Money Concepts analysis."""
        logger.info("🏦 SMART MONEY CONCEPTS (SMC)")
        logger.info("-" * 70)
        
        # Market bias from structure
        bias_emoji = "🟢" if smc.current_bias == "BULLISH" else "🔴" if smc.current_bias == "BEARISH" else "🟡"
        logger.info(f"  Structure Bias: {bias_emoji} {smc.current_bias}")
        
        # Premium/Discount zone
        if current_price < smc.discount_zone[1]:
            zone = "DISCOUNT ZONE 🟢 (Good for longs)"
        elif current_price > smc.premium_zone[0]:
            zone = "PREMIUM ZONE 🔴 (Good for shorts)"
        else:
            zone = "EQUILIBRIUM ZONE 🟡 (Fair value)"
        logger.info(f"  Price Zone:     {zone}")
        logger.info(f"  Equilibrium:    ${smc.equilibrium:,.0f}")
        
        # Order Blocks
        active_bull_obs = [ob for ob in smc.bullish_obs if not ob.mitigated]
        active_bear_obs = [ob for ob in smc.bearish_obs if not ob.mitigated]
        
        if active_bull_obs or active_bear_obs:
            logger.info("")
            logger.info("  📦 Active Order Blocks:")
            for ob in active_bull_obs[:2]:
                dist = (current_price - ob.top) / current_price * 100
                logger.info(f"     🟢 Bullish OB: ${ob.bottom:,.0f} - ${ob.top:,.0f} | Strength: {ob.strength:.0%} | {dist:+.1f}% from price")
            for ob in active_bear_obs[:2]:
                dist = (ob.bottom - current_price) / current_price * 100
                logger.info(f"     🔴 Bearish OB: ${ob.bottom:,.0f} - ${ob.top:,.0f} | Strength: {ob.strength:.0%} | +{dist:.1f}% from price")
        
        # Fair Value Gaps
        if smc.bullish_fvgs or smc.bearish_fvgs:
            logger.info("")
            logger.info("  📊 Unfilled Fair Value Gaps (Imbalances):")
            for fvg in smc.bullish_fvgs[:2]:
                logger.info(f"     🟢 Bullish FVG: ${fvg.bottom:,.0f} - ${fvg.top:,.0f} (price may retrace here)")
            for fvg in smc.bearish_fvgs[:2]:
                logger.info(f"     🔴 Bearish FVG: ${fvg.bottom:,.0f} - ${fvg.top:,.0f} (price may retrace here)")
        
        # Structure breaks
        recent_choch = [b for b in smc.structure_breaks if b.type == "CHOCH"]
        recent_bos = [b for b in smc.structure_breaks if b.type == "BOS"]
        
        if recent_choch or recent_bos:
            logger.info("")
            logger.info("  🔄 Structure Breaks:")
            for brk in recent_choch[-2:]:
                emoji = "🟢" if brk.direction == "BULLISH" else "🔴"
                logger.info(f"     {emoji} CHoCH (Reversal): {brk.direction} break of ${brk.break_level:,.0f}")
            for brk in recent_bos[-2:]:
                emoji = "🟢" if brk.direction == "BULLISH" else "🔴"
                logger.info(f"     {emoji} BOS (Continuation): {brk.direction} break of ${brk.break_level:,.0f}")
        
        # Liquidity levels
        if smc.buy_side_liquidity or smc.sell_side_liquidity:
            logger.info("")
            logger.info("  💧 Liquidity Pools (stop hunt targets):")
            if smc.buy_side_liquidity[:3]:
                levels = ", ".join([f"${l:,.0f}" for l in smc.buy_side_liquidity[:3]])
                logger.info(f"     📈 Buy-side (above): {levels}")
            if smc.sell_side_liquidity[:3]:
                levels = ", ".join([f"${l:,.0f}" for l in smc.sell_side_liquidity[:3]])
                logger.info(f"     📉 Sell-side (below): {levels}")
        
        # Recent sweeps
        if smc.liquidity_sweeps:
            logger.info("")
            logger.info("  ⚡ Recent Liquidity Sweeps:")
            for sweep in smc.liquidity_sweeps[-2:]:
                if sweep.type == "HIGH_SWEEP":
                    logger.info(f"     🔴 Swept highs at ${sweep.sweep_level:,.0f} - potential reversal DOWN")
                else:
                    logger.info(f"     🟢 Swept lows at ${sweep.sweep_level:,.0f} - potential reversal UP")
        
        logger.info("")
    
    def generate_warnings(self, ctx: MarketContext) -> list[str]:
        """Generate trading warnings based on context."""
        warnings = []
        
        # Near support - warning for shorts
        if ctx.support_levels:
            nearest_support = ctx.support_levels[0]
            distance_pct = ((ctx.current_price - nearest_support.price) / ctx.current_price) * 100
            if distance_pct < 0.5 and nearest_support.strength > 0.5:
                warnings.append(f"🚫 CLOSE TO STRONG SUPPORT (${nearest_support.price:,.0f}) - Short risky before break!")
        
        # Near resistance - warning for longs
        if ctx.resistance_levels:
            nearest_resistance = ctx.resistance_levels[0]
            distance_pct = ((nearest_resistance.price - ctx.current_price) / ctx.current_price) * 100
            if distance_pct < 0.5 and nearest_resistance.strength > 0.5:
                warnings.append(f"🚫 CLOSE TO STRONG RESISTANCE (${nearest_resistance.price:,.0f}) - Long risky before break!")
        
        # RSI extremes
        if "1h" in ctx.momentum:
            rsi_1h = ctx.momentum["1h"].rsi
            if rsi_1h < 25:
                warnings.append(f"📉 RSI 1H EXTREMELY OVERSOLD ({rsi_1h:.1f}) - Bounce likely!")
            elif rsi_1h > 75:
                warnings.append(f"📈 RSI 1H EXTREMELY OVERBOUGHT ({rsi_1h:.1f}) - Pullback likely!")
        
        # Trend conflict between timeframes
        if "15m" in ctx.trends and "4h" in ctx.trends:
            tf_15m = ctx.trends["15m"].direction
            tf_4h = ctx.trends["4h"].direction
            if tf_15m != tf_4h and tf_15m != "SIDEWAYS" and tf_4h != "SIDEWAYS":
                warnings.append(f"⚠️ TREND CONFLICT: 15m={tf_15m}, 4h={tf_4h} - Be cautious!")
        
        # Low volume
        if "1h" in ctx.momentum and ctx.momentum["1h"].volume_ratio < 0.5:
            warnings.append("📉 LOW VOLUME (1H) - Moves may lack conviction, wait for confirmation")
        
        # High spread (volatility incoming)
        if "5m" in ctx.momentum and ctx.momentum["5m"].spread_bps > 20:
            warnings.append(f"⚠️ HIGH SPREAD ({ctx.momentum['5m'].spread_bps:.1f}bps) - Increased volatility!")
        
        # SMC warnings
        if ctx.smc:
            # Price in premium zone trying to long
            if ctx.current_price > ctx.smc.premium_zone[0] and ctx.signal and ctx.signal.direction == "LONG":
                warnings.append("⚠️ Attempting LONG in PREMIUM zone - higher risk entry")
            
            # Price in discount zone trying to short
            if ctx.current_price < ctx.smc.discount_zone[1] and ctx.signal and ctx.signal.direction == "SHORT":
                warnings.append("⚠️ Attempting SHORT in DISCOUNT zone - higher risk entry")
        
        return warnings
    
    async def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down Market Analyzer...")
        self.running = False
        if self.db:
            await self.db.disconnect()
        logger.info("Shutdown complete")


async def main():
    service = MarketAnalyzerService()
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(service.stop()))
    
    try:
        await service.start()
    except KeyboardInterrupt:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
