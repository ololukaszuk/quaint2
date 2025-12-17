"""
Database access for LLM Analyst.

Fetches candles, market analysis (enhanced), and signal history.
Saves LLM analysis with full logging details.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any

import asyncpg
from loguru import logger

from config import Config


def convert_decimals(obj: Any) -> Any:
    """Recursively convert Decimal objects to floats for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    return obj


class Database:
    """Async PostgreSQL database connection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Establish database connection pool."""
        logger.info(f"Connecting to database at {self.config.db_host}:{self.config.db_port}")
        
        self.pool = await asyncpg.create_pool(
            host=self.config.db_host,
            port=self.config.db_port,
            database=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            min_size=1,
            max_size=3,
        )
        logger.info("Database connected")
    
    async def disconnect(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database disconnected")
    
    async def get_latest_candle_time(self) -> Optional[datetime]:
        """Get the timestamp of the most recent 1m candle."""
        if not self.pool:
            return None
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT time FROM candles_1m ORDER BY time DESC LIMIT 1"
            )
            return row['time'] if row else None
    
    async def get_candles_1h(self, limit: int = 120) -> List[Dict[str, Any]]:
        """
        Get aggregated 1H candles from 1m data.
        
        Returns list of dicts with: time, open, high, low, close, volume
        """
        if not self.pool:
            return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    time_bucket('1 hour', time) AS open_time,
                    FIRST(open, time) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, time) AS close,
                    SUM(volume) AS volume
                FROM candles_1m
                WHERE time > NOW() - INTERVAL '7 days'
                GROUP BY time_bucket('1 hour', time)
                ORDER BY open_time DESC
                LIMIT $1
                """,
                limit
            )
            
            # Return in chronological order
            return [dict(row) for row in reversed(rows)]
    
    async def get_candles_15m(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get aggregated 15m candles from 1m data.
        
        Returns list of dicts with: time, open, high, low, close, volume
        """
        if not self.pool:
            return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    time_bucket('15 minutes', time) AS open_time,
                    FIRST(open, time) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, time) AS close,
                    SUM(volume) AS volume
                FROM candles_1m
                WHERE time > NOW() - INTERVAL '24 hours'
                GROUP BY time_bucket('15 minutes', time)
                ORDER BY open_time DESC
                LIMIT $1
                """,
                limit
            )
            
            return [dict(row) for row in reversed(rows)]
    
    async def get_latest_market_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent market analysis with ALL enriched data from enhanced schema.
        
        This includes:
        - Basic signal info (type, direction, confidence)
        - Trade setup (entry, SL, TPs)
        - Signal factors (weighted reasons)
        - All pivot levels (Traditional, Fibonacci, Camarilla)
        - Pivot confluence zones
        - Complete SMC data (order blocks, FVGs, breaks, liquidity)
        - Support/Resistance levels with metadata
        - Momentum for all timeframes
        - Market structure
        - Warnings
        - Action recommendation
        """
        if not self.pool:
            return None

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        -- Basic info
                        analysis_time,
                        price,
                        signal_type,
                        signal_direction,
                        signal_confidence,
                        
                        -- Trade setup
                        entry_price,
                        stop_loss,
                        take_profit_1,
                        take_profit_2,
                        take_profit_3,
                        risk_reward_ratio,
                        
                        -- Trends (multi-timeframe)
                        trends,
                        
                        -- Original S/R (may be NULL in older records)
                        nearest_support,
                        nearest_resistance,
                        support_strength,
                        resistance_strength,
                        
                        -- Enhanced S/R (JSONB arrays with all levels)
                        support_levels,
                        resistance_levels,
                        
                        -- SMC basic
                        smc_bias,
                        price_zone,
                        equilibrium_price,
                        
                        -- SMC enhanced (JSONB)
                        smc_price_zone,
                        smc_equilibrium,
                        smc_order_blocks,
                        smc_fvgs,
                        smc_breaks,
                        smc_liquidity,
                        
                        -- Pivots (traditional)
                        daily_pivot,
                        pivot_daily,
                        price_vs_pivot,
                        pivot_r1_traditional,
                        pivot_r2_traditional,
                        pivot_r3_traditional,
                        pivot_s1_traditional,
                        pivot_s2_traditional,
                        pivot_s3_traditional,
                        
                        -- Pivots (Fibonacci)
                        pivot_r1_fibonacci,
                        pivot_r2_fibonacci,
                        pivot_r3_fibonacci,
                        pivot_s1_fibonacci,
                        pivot_s2_fibonacci,
                        pivot_s3_fibonacci,
                        
                        -- Pivots (Camarilla)
                        pivot_r3_camarilla,
                        pivot_r4_camarilla,
                        pivot_s3_camarilla,
                        pivot_s4_camarilla,
                        
                        -- Pivot confluence
                        pivot_confluence_zones,
                        
                        -- Momentum (old single timeframe)
                        rsi_1h,
                        volume_ratio_1h,
                        
                        -- Momentum (enhanced - all timeframes as JSONB)
                        momentum,
                        
                        -- Signal factors (weighted reasons)
                        signal_factors,
                        
                        -- Market structure
                        structure_pattern,
                        structure_last_high,
                        structure_last_low,
                        
                        -- Warnings and action
                        warnings,
                        action_recommendation,
                        
                        -- Summary
                        summary,
                        signal_changed,
                        previous_signal
                        
                    FROM market_analysis
                    ORDER BY analysis_time DESC
                    LIMIT 1
                    """
                )
                
                if not row:
                    return None
                
                # Convert to dict and handle JSONB fields
                result = dict(row)
                
                # Parse JSONB fields that might be returned as strings
                jsonb_fields = [
                    'trends', 'support_levels', 'resistance_levels',
                    'smc_order_blocks', 'smc_fvgs', 'smc_breaks', 'smc_liquidity',
                    'pivot_confluence_zones', 'momentum', 'signal_factors', 'warnings'
                ]
                
                for field in jsonb_fields:
                    if field in result and result[field]:
                        if isinstance(result[field], str):
                            try:
                                result[field] = json.loads(result[field])
                            except (json.JSONDecodeError, TypeError):
                                pass
                
                return result
                
        except Exception as e:
            logger.warning(f"Could not fetch market analysis: {e}")
            return None
    
    async def get_signal_history(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Get recent signal changes with enhanced key_reasons."""
        if not self.pool:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT 
                        signal_time,
                        signal_type,
                        signal_direction,
                        signal_confidence,
                        price,
                        previous_signal_type,
                        summary,
                        key_reasons,
                        -- Enhanced fields from migration 002
                        signal_factors,
                        smc_bias,
                        pivot_daily,
                        nearest_support,
                        nearest_resistance
                    FROM market_signals
                    ORDER BY signal_time DESC
                    LIMIT $1
                    """,
                    limit
                )
                
                signals = []
                for row in rows:
                    signal = dict(row)
                    
                    # Parse JSONB fields
                    for field in ['key_reasons', 'signal_factors']:
                        if field in signal and signal[field]:
                            if isinstance(signal[field], str):
                                try:
                                    signal[field] = json.loads(signal[field])
                                except (json.JSONDecodeError, TypeError):
                                    pass
                    
                    signals.append(signal)
                
                return signals
                
        except Exception as e:
            logger.warning(f"Could not fetch signal history (table may not exist): {e}")
            return []
    
    async def get_candle_count_since(self, since: datetime) -> int:
        """Count 1m candles since given timestamp."""
        if not self.pool:
            return 0
        
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM candles_1m WHERE time > $1",
                since
            )
            return count or 0
    
    async def save_llm_analysis(
        self,
        analysis_time: datetime,
        price: float,
        prediction_direction: str,
        prediction_confidence: str,
        predicted_price_1h: Optional[float],
        predicted_price_4h: Optional[float],
        key_levels: str,
        reasoning: str,
        full_response: str,
        model_name: str,
        response_time_seconds: float,
        # New enhanced fields
        invalidation_level: Optional[float] = None,
        critical_support: Optional[float] = None,
        critical_resistance: Optional[float] = None,
        market_context: Optional[Dict[str, Any]] = None,
        signal_factors_used: Optional[List[Dict[str, Any]]] = None,
        smc_bias_at_analysis: Optional[str] = None,
        trends_at_analysis: Optional[Dict[str, Any]] = None,
        warnings_at_analysis: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Save LLM analysis to database with enhanced logging.
        
        Stores not just the prediction, but also the market context that was
        used to make the prediction (for later analysis of what inputs led to
        good/bad predictions).
        """
        if not self.pool:
            return
        
        try:
            async with self.pool.acquire() as conn:
                # Check if table exists
                table_exists = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'llm_analysis')"
                )
                
                if not table_exists:
                    logger.debug("llm_analysis table doesn't exist yet - skipping save")
                    return
                
                # Check if enhanced columns exist
                has_enhanced = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'llm_analysis' 
                        AND column_name = 'market_context'
                    )
                    """
                )
                
                if has_enhanced:
                    # Use enhanced insert with all new fields
                    await conn.execute(
                        """
                        INSERT INTO llm_analysis (
                            analysis_time, price, prediction_direction, prediction_confidence,
                            predicted_price_1h, predicted_price_4h, key_levels, reasoning,
                            full_response, model_name, response_time_seconds,
                            invalidation_level, critical_support, critical_resistance,
                            market_context, signal_factors_used, smc_bias_at_analysis,
                            trends_at_analysis, warnings_at_analysis
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                            $12, $13, $14, $15, $16, $17, $18, $19
                        )
                        """,
                        analysis_time, price, prediction_direction, prediction_confidence,
                        predicted_price_1h, predicted_price_4h, key_levels, reasoning,
                        full_response, model_name, response_time_seconds,
                        invalidation_level, critical_support, critical_resistance,
                        json.dumps(convert_decimals(market_context)) if market_context else None,
                        json.dumps(convert_decimals(signal_factors_used)) if signal_factors_used else None,
                        smc_bias_at_analysis,
                        json.dumps(convert_decimals(trends_at_analysis)) if trends_at_analysis else None,
                        json.dumps(convert_decimals(warnings_at_analysis)) if warnings_at_analysis else None
                    )
                else:
                    # Fall back to original schema
                    await conn.execute(
                        """
                        INSERT INTO llm_analysis (
                            analysis_time, price, prediction_direction, prediction_confidence,
                            predicted_price_1h, predicted_price_4h, key_levels, reasoning,
                            full_response, model_name, response_time_seconds
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        """,
                        analysis_time, price, prediction_direction, prediction_confidence,
                        predicted_price_1h, predicted_price_4h, key_levels, reasoning,
                        full_response, model_name, response_time_seconds
                    )
                
                logger.debug("LLM analysis saved to database")
                
        except Exception as e:
            logger.warning(f"Could not save LLM analysis: {e}")
    
    async def get_recent_llm_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent LLM analyses for context/comparison."""
        if not self.pool:
            return []
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT 
                        analysis_time,
                        price,
                        prediction_direction,
                        prediction_confidence,
                        predicted_price_1h,
                        actual_price_1h,
                        direction_correct_1h,
                        reasoning
                    FROM llm_analysis
                    WHERE actual_price_1h IS NOT NULL
                    ORDER BY analysis_time DESC
                    LIMIT $1
                    """,
                    limit
                )
                return [dict(row) for row in rows]
        except Exception as e:
            logger.warning(f"Could not fetch recent LLM analyses: {e}")
            return []