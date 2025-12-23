#!/usr/bin/env python3
"""
Market Analyzer - Real-time multi-timeframe market context analysis for BTCUSDT.

Listens for new 1m candles and produces human-readable market context reports
with trading signals, pivot points, and Smart Money Concepts analysis.
"""

import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from typing import Optional
import numpy as np

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

def to_python(value):
    """Convert NumPy types to Python native types for PostgreSQL."""
    if value is None:
        return None
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

class MarketAnalyzerService:
    """Main service that orchestrates market analysis."""
    
    def __init__(self):
        self.config = Config()
        self.db: Optional[Database] = None
        self.analyzer: Optional[MarketAnalyzer] = None
        self.llm_controller: Optional[LLMController] = None
        self.running = False
        self.last_candle_time: Optional[datetime] = None
        
        # Track previous signal for change detection
        self.previous_signal_type: Optional[str] = None
        self.previous_signal_direction: Optional[str] = None
      
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
            logger.info(f"Last 1m candle OPEN time in DB: {self.last_candle_time}")
        else:
            logger.warning("No candles found in database yet")
        
        # Load previous signal from DB if available
        await self.load_previous_signal()
        
        # Run initial analysis
        await self.run_analysis()
        
        self.running = True
        logger.info(f"Polling for new candles every {self.config.poll_interval_seconds}s")
        
        # Initialize LLM controller
        self.llm_controller = LLMController(self.config)
        logger.info(f"LLM Analyst: {'enabled' if self.config.llm_analyst_enabled else 'disabled'}")
        
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
    
    async def load_previous_signal(self):
        """Load the most recent signal from database."""
        if not self.db or not self.db.pool:
            return
        
        try:
            async with self.db.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT signal_type, signal_direction 
                    FROM market_signals 
                    ORDER BY signal_time DESC 
                    LIMIT 1
                    """
                )
                if row:
                    self.previous_signal_type = row['signal_type']
                    self.previous_signal_direction = row['signal_direction']
                    logger.info(f"Loaded previous signal: {self.previous_signal_type} ({self.previous_signal_direction})")
        except Exception as e:
            logger.debug(f"Could not load previous signal (table may not exist yet): {e}")
    
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
            if not context:
                return
            
            # Check for signal change
            signal_changed = self.detect_signal_change(context)
            
            # Log results (with toggle)
            if self.config.detailed_logging:
                self.log_market_context(context, signal_changed)
            else:
                self.log_market_context_brief(context, signal_changed)
            
            # Save analysis with actual timestamp
            previous_type_for_save = self.previous_signal_type
            previous_dir_for_save = self.previous_signal_direction
            
            if context.signal:
                current_type = context.signal.signal_type.value
                current_direction = context.signal.direction
                
                if signal_changed and self.config.detailed_logging:
                    logger.debug(f"Signal transition: {previous_type_for_save} ({previous_dir_for_save}) → {current_type} ({current_direction})")
                
                self.previous_signal_type = current_type
                self.previous_signal_direction = current_direction
            
            await self.save_analysis(
                context, 
                signal_changed,
                previous_type_for_save,
                previous_dir_for_save
            )
            
            # NEW: Check if LLM analysis would add value
            trigger_reason = await self.should_request_llm_analysis(context, signal_changed)
            if trigger_reason:
                await self.request_llm_analysis(context, trigger_reason)
                
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            if self.config.detailed_logging:
                import traceback
                traceback.print_exc()
                
    def detect_signal_change(self, ctx: MarketContext) -> bool:
        """Check if signal has changed from previous analysis."""
        if not ctx.signal or ctx.signal.direction == "NONE":
            return False  # Ignore NEUTRAL signals entirely
        
        current_type = ctx.signal.signal_type.value
        current_direction = ctx.signal.direction
        
        if self.previous_signal_type is None:
            return True  # First signal
        
        # Record if EITHER type or direction changed
        # WEAK_BUY→BUY (same LONG) = recorded ✅
        # BUY→STRONG_BUY (same LONG) = recorded ✅
        # STRONG_BUY→LONG (direction stays) = recorded ✅
        # WEAK_BUY→WEAK_BUY (identical) = NOT recorded ✅
        return (current_type != self.previous_signal_type or
                current_direction != self.previous_signal_direction)
    
    async def save_analysis(
        self, 
        ctx: MarketContext, 
        signal_changed: bool,
        previous_type: Optional[str] = None,
        previous_direction: Optional[str] = None
    ):
        """Save analysis to database with ALL pivot columns."""
        if not self.db or not self.db.pool:
            return
        
        try:
            async with self.db.pool.acquire() as conn:
                signal = ctx.signal
                
                # === 1. SIGNAL FACTORS (with weights) ===
                signal_factors = []
                if signal and signal.reasons:
                    for reasons in signal.reasons[:10]:
                        signal_factors.append({
                            "description": reasons.description,
                            "weight": reasons.weight,
                            "type": "bullish" if reasons.weight > 0 else "bearish"
                        })
                
                # === 2. TRENDS ===
                trends_json = {}
                for tf, trend in ctx.trends.items():
                    trends_json[tf] = {
                        "direction": trend.direction,
                        "strength": trend.strength,
                        "ema": getattr(trend, 'ema_alignment', 'UNKNOWN'),
                        "structure": getattr(trend, 'structure', '')
                    }
                
                # === 3. ALL PIVOT POINTS (5 methods) ===
                pivots = ctx.pivots
                t = pivots.traditional if pivots else None
                f = pivots.fibonacci if pivots else None
                c = pivots.camarilla if pivots else None
                w = pivots.woodie if pivots else None
                d = pivots.demark if pivots else None
                
                # === 4. SMC DATA ===
                smc_order_blocks = []
                if ctx.smc:
                    for ob in (ctx.smc.bullish_obs or []):
                        smc_order_blocks.append({
                            "type": "bullish", "low": float(ob.bottom), "high": float(ob.top),
                            "strength": ob.strength,
                            "distance_pct": abs((ob.top - ctx.current_price) / ctx.current_price * 100)
                        })
                    for ob in (ctx.smc.bearish_obs or []):
                        smc_order_blocks.append({
                            "type": "bearish", "low": float(ob.bottom), "high": float(ob.top),
                            "strength": ob.strength,
                            "distance_pct": abs((ctx.current_price - ob.bottom) / ctx.current_price * 100)
                        })

                smc_fvgs = []
                if ctx.smc:
                    for fvg in (ctx.smc.bullish_fvgs or []):
                        smc_fvgs.append({"type": "bullish", "low": float(fvg.bottom), "high": float(fvg.top), "unfilled": not fvg.filled})
                    for fvg in (ctx.smc.bearish_fvgs or []):
                        smc_fvgs.append({"type": "bearish", "low": float(fvg.bottom), "high": float(fvg.top), "unfilled": not fvg.filled})

                smc_breaks = []
                if ctx.smc and ctx.smc.structure_breaks:
                    for brk in ctx.smc.structure_breaks:
                        if brk.type == "CHOCH":
                            smc_breaks.append({"type": "CHoCH", "direction": brk.direction, "price": float(brk.break_level)})

                smc_liquidity = {
                    "buy_side": [float(p) for p in ctx.smc.buy_side_liquidity] if ctx.smc and ctx.smc.buy_side_liquidity else [],
                    "sell_side": [float(p) for p in ctx.smc.sell_side_liquidity] if ctx.smc and ctx.smc.sell_side_liquidity else []
                }
                
                # === 5. SUPPORT/RESISTANCE ===
                support_levels = [{"price": float(l.price), "strength": l.strength, "touches": l.touches,
                                   "timeframe": l.source_timeframe, "distance_pct": abs((l.price - ctx.current_price) / ctx.current_price * 100)}
                                  for l in ctx.support_levels[:3]]
                resistance_levels = [{"price": float(l.price), "strength": l.strength, "touches": l.touches,
                                      "timeframe": l.source_timeframe, "distance_pct": abs((l.price - ctx.current_price) / ctx.current_price * 100)}
                                     for l in ctx.resistance_levels[:3]]
                
                # === 6. MOMENTUM ===
                momentum = {tf: {"rsi": mom.rsi, "volume_ratio": mom.volume_ratio, "taker_buy_ratio": mom.taker_buy_ratio}
                           for tf, mom in ctx.momentum.items()}
                
                # === 7. STRUCTURE ===
                structure_pattern = getattr(ctx.structure, 'pattern', None)
                structure_last_high = getattr(ctx.structure, 'last_high', None)
                structure_last_low = getattr(ctx.structure, 'last_low', None)
                
                # === 8. CONFLUENCE ===
                confluence_zones = []
                if pivots:
                    try:
                        nr = pivots.get_nearest_resistance(ctx.current_price)
                        ns = pivots.get_nearest_support(ctx.current_price)
                        if nr: confluence_zones.append({"type": "resistance", "data": nr})
                        if ns: confluence_zones.append({"type": "support", "data": ns})
                    except: pass
                
                # === 9. WARNINGS ===
                warnings = self.generate_warnings(ctx)
                if signal and signal.warnings: warnings.extend(signal.warnings)
                warnings = list(dict.fromkeys(warnings))[:5]
                
                # === 10. ACTION ===
                action = signal.direction if signal and signal.confidence > 60 else "WAIT"

                # Check if new columns exist (migration 003)
                has_woodie = await conn.fetchval(
                    """
                    SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_schema = 'public'
                        AND table_name   = 'market_analysis'
                        AND column_name  = 'pivot_woodie'
                    )
                    """
                )
                logger.info(f"SCHEMA CHECK: pivot_woodie exists? {has_woodie}")
                logger.info(f"BRANCH: {'FULL INSERT' if has_woodie else 'FALLBACK'}")
                
                if has_woodie:
                    # Full insert with all 5 pivot methods
                    await conn.execute("""
                        INSERT INTO market_analysis (
                            analysis_time, price, signal_type, signal_direction, signal_confidence,
                            entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
                            signal_factors, trends,
                            nearest_support, nearest_resistance, support_strength, resistance_strength,
                            pivot_daily, pivot_r3_traditional, pivot_r2_traditional, pivot_r1_traditional,
                            pivot_s1_traditional, pivot_s2_traditional, pivot_s3_traditional,
                            pivot_r3_fibonacci, pivot_r2_fibonacci, pivot_r1_fibonacci,
                            pivot_s1_fibonacci, pivot_s2_fibonacci, pivot_s3_fibonacci,
                            pivot_camarilla, pivot_r1_camarilla, pivot_r2_camarilla, pivot_r3_camarilla, pivot_r4_camarilla,
                            pivot_s1_camarilla, pivot_s2_camarilla, pivot_s3_camarilla, pivot_s4_camarilla,
                            pivot_woodie, pivot_r1_woodie, pivot_r2_woodie, pivot_r3_woodie,
                            pivot_s1_woodie, pivot_s2_woodie, pivot_s3_woodie,
                            pivot_demark, pivot_r1_demark, pivot_s1_demark,
                            pivot_confluence_zones, price_vs_pivot,
                            smc_bias, price_zone, smc_price_zone, smc_equilibrium,
                            smc_order_blocks, smc_fvgs, smc_breaks, smc_liquidity,
                            support_levels, resistance_levels, momentum,
                            rsi_1h, volume_ratio_1h, daily_pivot, equilibrium_price,
                            structure_pattern, structure_last_high, structure_last_low,
                            warnings, action_recommendation, summary, signal_changed, previous_signal
                        ) VALUES (
                            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
                            $18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,
                            $31,$32,$33,$34,$35,$36,$37,$38,$39,$40,$41,$42,$43,$44,$45,$46,$47,$48,$49,
                            $50,$51,$52,$53,$54,$55,$56,$57,$58,$59,$60,$61,$62,$63,$64,$65,$66,$67,$68,$69,$70,$71,$72,
                            $73,$74
                        )
                    """,
                        ctx.timestamp, to_python(ctx.current_price),
                        signal.signal_type.value if signal else None, signal.direction if signal else None,
                        to_python(signal.confidence) if signal else None,
                        to_python(signal.setup.entry if signal and signal.setup else None),
                        to_python(signal.setup.stop_loss if signal and signal.setup else None),
                        to_python(signal.setup.take_profit_1 if signal and signal.setup else None),
                        to_python(signal.setup.take_profit_2 if signal and signal.setup else None),
                        to_python(signal.setup.take_profit_3 if signal and signal.setup else None),
                        to_python(signal.setup.risk_reward_ratio if signal and signal.setup else None),
                        json.dumps(signal_factors), json.dumps(trends_json),
                        float(support_levels[0]['price']) if support_levels else None,
                        float(resistance_levels[0]['price']) if resistance_levels else None,
                        support_levels[0]['strength'] if support_levels else None,
                        resistance_levels[0]['strength'] if resistance_levels else None,
                        # Traditional
                        to_python(t.pivot if t else None),
                        to_python(t.r3 if t else None), to_python(t.r2 if t else None), to_python(t.r1 if t else None),
                        to_python(t.s1 if t else None), to_python(t.s2 if t else None), to_python(t.s3 if t else None),
                        # Fibonacci
                        to_python(f.r3 if f else None), to_python(f.r2 if f else None), to_python(f.r1 if f else None),
                        to_python(f.s1 if f else None), to_python(f.s2 if f else None), to_python(f.s3 if f else None),
                        # Camarilla (complete)
                        to_python(c.pivot if c else None),
                        to_python(c.r1 if c else None), to_python(c.r2 if c else None), to_python(c.r3 if c else None), to_python(c.r4 if c else None),
                        to_python(c.s1 if c else None), to_python(c.s2 if c else None), to_python(c.s3 if c else None), to_python(c.s4 if c else None),
                        # Woodie
                        to_python(w.pivot if w else None),
                        to_python(w.r1 if w else None), to_python(w.r2 if w else None), to_python(w.r3 if w else None),
                        to_python(w.s1 if w else None), to_python(w.s2 if w else None), to_python(w.s3 if w else None),
                        # DeMark
                        to_python(d.pivot if d else None), to_python(d.r1 if d else None), to_python(d.s1 if d else None),
                        # Rest
                        json.dumps(confluence_zones), "ABOVE" if t and ctx.current_price > t.pivot else "BELOW",
                        ctx.smc.current_bias if ctx.smc else None, self.get_price_zone(ctx), self.get_price_zone(ctx),
                        to_python(ctx.smc.equilibrium if ctx.smc else None),
                        json.dumps(smc_order_blocks), json.dumps(smc_fvgs), json.dumps(smc_breaks), json.dumps(smc_liquidity),
                        json.dumps(support_levels), json.dumps(resistance_levels), json.dumps(momentum),
                        ctx.momentum.get('1h').rsi if ctx.momentum.get('1h') else None,
                        ctx.momentum.get('1h').volume_ratio if ctx.momentum.get('1h') else None,
                        to_python(t.pivot if t else None), to_python(ctx.smc.equilibrium if ctx.smc else None),
                        structure_pattern, to_python(structure_last_high), to_python(structure_last_low),
                        json.dumps(warnings), action, self.generate_summary(ctx), signal_changed, previous_type
                    )
                else:
                # === INSERT QUERY (FALLBACK)===
                    await conn.execute(
                        """
                        INSERT INTO market_analysis (
                            analysis_time, price, signal_type, signal_direction, signal_confidence,
                            entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
                            signal_factors, trends,
                            nearest_support, nearest_resistance, support_strength, resistance_strength,
                            pivot_daily, pivot_r3_traditional, pivot_r2_traditional, pivot_r1_traditional,
                            pivot_s1_traditional, pivot_s2_traditional, pivot_s3_traditional,
                            pivot_r3_fibonacci, pivot_r2_fibonacci, pivot_r1_fibonacci,
                            pivot_s1_fibonacci, pivot_s2_fibonacci, pivot_s3_fibonacci,
                            pivot_r4_camarilla, pivot_r3_camarilla, pivot_s3_camarilla, pivot_s4_camarilla,
                            pivot_confluence_zones, price_vs_pivot,
                            smc_bias, price_zone, smc_price_zone, smc_equilibrium,
                            smc_order_blocks, smc_fvgs, smc_breaks, smc_liquidity,
                            support_levels, resistance_levels, momentum,
                            rsi_1h, volume_ratio_1h,
                            daily_pivot, equilibrium_price,
                            structure_pattern, structure_last_high, structure_last_low,
                            warnings, action_recommendation,
                            summary, signal_changed, previous_signal
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                            $14, $15, $16, $17,
                            $18, $19, $20, $21, $22, $23, $24,
                            $25, $26, $27, $28, $29, $30,
                            $31, $32, $33, $34,
                            $35, $36,
                            $37, $38, $39, $40,
                            $41, $42, $43, $44,
                            $45, $46, $47,
                            $48, $49,
                            $50, $51,
                            $52, $53, $54,
                            $55, $56,
                            $57, $58, $59
                        )
                        """,
                        # $1-$5: Basic price & signal
                        datetime.now(timezone.utc), to_python(ctx.current_price),
                        signal.signal_type.value if signal else None,
                        signal.direction if signal else None,
                        to_python(signal.confidence) if signal else None,
                        
                        # $6-$11: Trade setup
                        to_python(signal.setup.entry if signal and signal.setup else None),
                        to_python(signal.setup.stop_loss if signal and signal.setup else None),
                        to_python(signal.setup.take_profit_1 if signal and signal.setup else None),
                        to_python(signal.setup.take_profit_2 if signal and signal.setup else None),
                        to_python(signal.setup.take_profit_3 if signal and signal.setup else None),
                        to_python(signal.setup.risk_reward_ratio if signal and signal.setup else None),
                        
                        # $12-$13: Signal & trends
                        json.dumps(signal_factors),
                        json.dumps(trends_json),
                        
                        # $14-$17: Nearest support/resistance (from support_levels[0] & resistance_levels[0])
                        float(support_levels[0]['price']) if support_levels else None,  # $14 nearest_support
                        float(resistance_levels[0]['price']) if resistance_levels else None,  # $15 nearest_resistance
                        support_levels[0]['strength'] if support_levels else None,  # $16 support_strength
                        resistance_levels[0]['strength'] if resistance_levels else None,  # $17 resistance_strength
                        
                        # $18-$30: Pivot points (traditional)
                        to_python(t.pivot if t else None),
                        to_python(t.r3 if t else None), to_python(t.r2 if t else None), to_python(t.r1 if t else None),
                        to_python(t.s1 if t else None), to_python(t.s2 if t else None), to_python(t.s3 if t else None),
                        
                        # $25-$30: Pivot points (fibonacci)
                        to_python(f.r3 if f else None), to_python(f.r2 if f else None), to_python(f.r1 if f else None),
                        to_python(f.s1 if f else None), to_python(f.s2 if f else None), to_python(f.s3 if f else None),
                        
                        # $31-$34: Pivot points (camarilla)
                        to_python(c.r4 if c else None), to_python(c.r3 if c else None), to_python(c.s3 if c else None), to_python(c.s4 if c else None),
                        
                        # $35-$36: Confluence zones & price vs pivot
                        json.dumps(confluence_zones),
                        "ABOVE" if t and ctx.current_price > t.pivot else "BELOW",
                        
                        # $37-$40: SMC bias & zones
                        ctx.smc.current_bias if ctx.smc else None,
                        self.get_price_zone(ctx),  # $38 price_zone (old schema)
                        self.get_price_zone(ctx),  # $39 smc_price_zone (new schema, same value)
                        to_python(ctx.smc.equilibrium if ctx.smc else None),
                        
                        # $41-$44: SMC data
                        json.dumps(smc_order_blocks), json.dumps(smc_fvgs),
                        json.dumps(smc_breaks), json.dumps(smc_liquidity),
                        
                        # $45-$47: Levels & momentum
                        json.dumps(support_levels), json.dumps(resistance_levels),
                        json.dumps(momentum),
                        
                        # $48-$49: Momentum indicators (1H)
                        ctx.momentum.get('1h').rsi if ctx.momentum.get('1h') else None,
                        ctx.momentum.get('1h').volume_ratio if ctx.momentum.get('1h') else None,
                        
                        # $50-$51: Pivot duplicates (old schema)
                        to_python(t.pivot if t else None),  # $50 daily_pivot
                        to_python(ctx.smc.equilibrium if ctx.smc else None),  # $51 equilibrium_price
                        
                        # $52-$54: Market structure
                        structure_pattern, to_python(structure_last_high), to_python(structure_last_low),
                        
                        # $55-$56: Warnings & action
                        json.dumps(warnings), action,
                        
                        # $57-$59: Summary & metadata
                        self.generate_summary(ctx), signal_changed, previous_type
                    )

                # Insert signal if changed
                if signal and signal.direction != "NONE":
                    current_type = signal.signal_type.value
                    current_direction = signal.direction
                    if previous_type != current_type or previous_direction != current_direction:
                        logger.info(f"💾 Signal change: {previous_type}({previous_direction}) → {current_type}({current_direction})")
                        await conn.execute("""
                            INSERT INTO market_signals (
                                signal_time, signal_type, signal_direction, signal_confidence, price,
                                entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3,
                                risk_reward_ratio, previous_signal_type, previous_direction, summary,
                                key_reasons, signal_factors, smc_bias, pivot_daily, nearest_support, nearest_resistance
                            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
                        """,
                            datetime.now(timezone.utc), signal.signal_type.value, signal.direction, to_python(signal.confidence),
                            to_python(ctx.current_price),
                            to_python(signal.setup.entry if signal.setup else None),
                            to_python(signal.setup.stop_loss if signal.setup else None),
                            to_python(signal.setup.take_profit_1 if signal.setup else None),
                            to_python(signal.setup.take_profit_2 if signal.setup else None),
                            to_python(signal.setup.take_profit_3 if signal.setup else None),
                            to_python(signal.setup.risk_reward_ratio if signal.setup else None),
                            previous_type, previous_direction, self.generate_summary(ctx),
                            json.dumps(signal_factors), json.dumps(signal_factors),
                            ctx.smc.current_bias if ctx.smc else None, to_python(t.pivot if t else None),
                            float(support_levels[0]['price']) if support_levels else None,
                            float(resistance_levels[0]['price']) if resistance_levels else None
                        )
                        logger.info("✅ Signal inserted")
                                
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            import traceback
            traceback.print_exc()

    async def should_request_llm_analysis(
        self, 
        ctx: MarketContext,
        signal_changed: bool
    ) -> Optional[str]:
        """
        Decide if LLM commentary would add value.
        
        Returns trigger reason if yes, None if no.
        """
        if not self.config.llm_requests_enabled:
            return None
        
        # Reason 1: Significant signal change
        if signal_changed and ctx.signal and ctx.signal.direction != "NONE":
            return "signal_change"
        
        # Reason 2: At major pivot level (within 0.3%)
        if ctx.pivots:
            pivot = ctx.pivots.traditional.pivot
            distance_pct = abs((ctx.current_price - pivot) / pivot * 100)
            if distance_pct < 0.3:
                return "key_level_pivot"
        
        # Reason 3: Conflicting timeframe signals
        if ctx.trends:
            trends = [t.direction for t in ctx.trends.values()]
            if "UPTREND" in trends and "DOWNTREND" in trends:
                return "timeframe_conflict"
        
        # Reason 4: Testing major support/resistance (within 0.3%)
        if ctx.support_levels and ctx.resistance_levels:
            nearest_support = ctx.support_levels[0].price
            nearest_resistance = ctx.resistance_levels[0].price
            dist_support = abs((ctx.current_price - nearest_support) / ctx.current_price * 100)
            dist_resistance = abs((ctx.current_price - nearest_resistance) / ctx.current_price * 100)
            if dist_support < 0.3:
                return "testing_support"
            if dist_resistance < 0.3:
                return "testing_resistance"
        
        return None

    async def request_llm_analysis(
        self,
        ctx: MarketContext,
        trigger_reason: str
    ):
        """Insert LLM request into database for llm-analyst to process."""
        if not self.db or not self.db.pool:
            return
        
        try:
            async with self.db.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO llm_requests (
                        request_time, price, signal_type, signal_direction, trigger_reason, status
                    ) VALUES ($1, $2, $3, $4, $5, 'pending')
                    """,
                    datetime.now(timezone.utc),
                    ctx.current_price,
                    ctx.signal.signal_type.value if ctx.signal else None,
                    ctx.signal.direction if ctx.signal else None,
                    trigger_reason
                )
                logger.info(f"🎯 LLM request queued: {trigger_reason}")
        except Exception as e:
            logger.warning(f"Failed to queue LLM request: {e}")

    def get_price_zone(self, ctx: MarketContext) -> str:
        """Get current price zone from SMC analysis."""
        if not ctx.smc:
            return "UNKNOWN"
        if ctx.current_price < ctx.smc.discount_zone[1]:
            return "DISCOUNT"
        elif ctx.current_price > ctx.smc.premium_zone[0]:
            return "PREMIUM"
        return "EQUILIBRIUM"
    
    def generate_summary(self, ctx: MarketContext) -> str:
        """Generate 2-3 line human readable summary."""
        parts = []
        
        # Price and signal
        signal = ctx.signal
        if signal:
            if signal.direction == "LONG":
                parts.append(f"BTC ${ctx.current_price:,.0f} - {signal.signal_type.value} ({signal.confidence:.0f}% confidence)")
            elif signal.direction == "SHORT":
                parts.append(f"BTC ${ctx.current_price:,.0f} - {signal.signal_type.value} ({signal.confidence:.0f}% confidence)")
            else:
                parts.append(f"BTC ${ctx.current_price:,.0f} - No clear signal, market undecided")
        
        # Trend summary
        trend_4h = ctx.trends.get("4h")
        trend_1h = ctx.trends.get("1h")
        if trend_4h and trend_1h:
            if trend_4h.direction == trend_1h.direction:
                parts.append(f"Trend aligned {trend_4h.direction} on 1H/4H")
            else:
                parts.append(f"Trend conflict: 1H {trend_1h.direction}, 4H {trend_4h.direction}")
        
        # Key level proximity
        if ctx.support_levels and ctx.resistance_levels:
            sup = ctx.support_levels[0]
            res = ctx.resistance_levels[0]
            sup_dist = (ctx.current_price - sup.price) / ctx.current_price * 100
            res_dist = (res.price - ctx.current_price) / ctx.current_price * 100
            
            if sup_dist < 0.5:
                parts.append(f"⚠️ At support ${sup.price:,.0f}")
            elif res_dist < 0.5:
                parts.append(f"⚠️ At resistance ${res.price:,.0f}")
            else:
                parts.append(f"S: ${sup.price:,.0f} ({sup_dist:.1f}%) | R: ${res.price:,.0f} ({res_dist:.1f}%)")
        
        return " | ".join(parts)
    
    def log_market_context(self, ctx: MarketContext, signal_changed: bool):
        """Log market context in a readable format."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"📈 MARKET CONTEXT REPORT - {ctx.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("=" * 80)
        
        # Current price
        logger.info(f"💰 Current Price: ${ctx.current_price:,.2f}")
        logger.info("")
        
        # ===== SIGNAL CHANGE ALERT =====
        if signal_changed and ctx.signal and ctx.signal.direction != "NONE":
            logger.info("🔔" * 20)
            logger.info(f"🔔 SIGNAL CHANGED! {self.previous_signal_type} → {ctx.signal.signal_type.value}")
            logger.info("🔔" * 20)
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
            for warning in warnings[:5]:  # Max 5 warnings
                logger.warning(f"  {warning}")
        else:
            logger.info("  ✅ No specific warnings - conditions favorable")
        logger.info("")
        
        # ===== SUMMARY (2-3 lines at the end) =====
        summary = self.generate_summary(ctx)
        logger.info("=" * 80)
        logger.info("📋 SUMMARY")
        logger.info("=" * 80)
        logger.info(f"  {summary}")
        
        # Quick action line
        if ctx.signal:
            if ctx.signal.direction == "LONG" and ctx.signal.confidence >= 60:
                logger.info(f"  ➡️  ACTION: Consider LONG entry at ${ctx.current_price:,.0f}, SL ${ctx.signal.setup.stop_loss:,.0f if ctx.signal.setup else 'N/A'}")
            elif ctx.signal.direction == "SHORT" and ctx.signal.confidence >= 60:
                logger.info(f"  ➡️  ACTION: Consider SHORT entry at ${ctx.current_price:,.0f}, SL ${ctx.signal.setup.stop_loss:,.0f if ctx.signal.setup else 'N/A'}")
            else:
                logger.info(f"  ➡️  ACTION: WAIT - No high-confidence setup")
        
        logger.info("=" * 80)
        logger.info("")

    def log_market_context_brief(self, ctx: MarketContext, signal_changed: bool):
        """Brief logging when detailed_logging=false."""
        timestamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
        
        # Signal emoji
        if ctx.signal:
            if ctx.signal.direction == "LONG":
                signal_emoji = "🟢"
            elif ctx.signal.direction == "SHORT":
                signal_emoji = "🔴"
            else:
                signal_emoji = "⚪"
        else:
            signal_emoji = "⚪"
        
        # Change indicator
        change_flag = "🔔" if signal_changed else ""
        
        # Brief one-liner
        logger.info(
            f"{timestamp} | {signal_emoji} ${ctx.current_price:,.0f} | "
            f"{ctx.signal.signal_type.value if ctx.signal else 'NEUTRAL'} "
            f"({ctx.signal.confidence:.0f}%) {change_flag}"
        )

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
            
            # Liquidity sweeps - only show most significant (max 2)
            high_sweeps = [s for s in ctx.smc.liquidity_sweeps if s.type == "HIGH_SWEEP"]
            low_sweeps = [s for s in ctx.smc.liquidity_sweeps if s.type == "LOW_SWEEP"]
            
            if high_sweeps:
                # Get the most significant (highest strength)
                best_high = max(high_sweeps, key=lambda x: x.strength)
                warnings.append(f"⚠️ Liquidity swept above ${best_high.sweep_level:,.0f} - watch for reversal DOWN")
            
            if low_sweeps:
                best_low = max(low_sweeps, key=lambda x: x.strength)
                warnings.append(f"⚠️ Liquidity swept below ${best_low.sweep_level:,.0f} - watch for reversal UP")
        
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
