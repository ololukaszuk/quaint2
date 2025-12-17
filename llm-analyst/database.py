"""
Database access for LLM Analyst.

Fetches candles, market analysis, and signal history.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import asyncpg
from loguru import logger

from config import Config


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
        """Get the most recent market analysis with ALL enriched data."""
        if not self.pool:
            return None

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        analysis_time,
                        price,
                        signal_type,
                        signal_direction,
                        signal_confidence,
                        entry_price,
                        stop_loss,
                        take_profit_1,
                        take_profit_2,
                        take_profit_3,
                        risk_reward_ratio,
                        trends,
                        nearest_support,
                        nearest_resistance,
                        support_strength,
                        resistance_strength,
                        smc_bias,
                        price_zone,
                        equilibrium_price,
                        daily_pivot,
                        price_vs_pivot,
                        rsi_1h,
                        volume_ratio_1h,
                        summary,
                        signal_factors,
                        support_levels,
                        resistance_levels,
                        momentum,
                        pivot_daily,
                        pivot_r1_traditional,
                        pivot_r2_traditional,
                        pivot_r3_traditional,
                        pivot_s1_traditional,
                        pivot_s2_traditional,
                        pivot_s3_traditional,
                        pivot_r1_fibonacci,
                        pivot_r2_fibonacci,
                        pivot_r3_fibonacci,
                        pivot_s1_fibonacci,
                        pivot_s2_fibonacci,
                        pivot_s3_fibonacci,
                        pivot_r4_camarilla,
                        pivot_r3_camarilla,
                        pivot_s3_camarilla,
                        pivot_s4_camarilla,
                        pivot_confluence_zones,
                        smc_order_blocks,
                        smc_fvgs,
                        smc_breaks,
                        smc_liquidity
                    FROM market_analysis
                    ORDER BY analysis_time DESC
                    LIMIT 1
                    """
                )
                return dict(row) if row else None
        except Exception as e:
            logger.warning(f"Could not fetch market analysis: {e}")
            return None
    
    async def get_signal_history(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Get recent signal changes."""
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
                        key_reasons
                    FROM market_signals
                    ORDER BY signal_time DESC
                    LIMIT $1
                    """,
                    limit
                )
                return [dict(row) for row in rows]
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
        response_time_seconds: float
    ):
        """Save LLM analysis to database."""
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
