"""
Database connection and queries for Market Analyzer.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional

import asyncpg
from loguru import logger

from config import Config
from models import Candle


class Database:
    """Async PostgreSQL database connection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Establish connection pool."""
        logger.info(f"Connecting to database at {self.config.db_host}:{self.config.db_port}")
        self.pool = await asyncpg.create_pool(
            host=self.config.db_host,
            port=self.config.db_port,
            database=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("Database connection pool established")
    
    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def get_latest_candle_time(self) -> Optional[datetime]:
        """Get the timestamp of the most recent candle."""
        if not self.pool:
            return None
            
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT time FROM candles_1m ORDER BY time DESC LIMIT 1"
            )
            if row:
                return row['time'].replace(tzinfo=timezone.utc)
            return None
    
    async def get_candles(
        self, 
        limit: int = 500,
        before: Optional[datetime] = None
    ) -> List[Candle]:
        """
        Fetch 1-minute candles from database.
        
        Args:
            limit: Maximum number of candles to fetch
            before: Fetch candles before this time (default: now)
            
        Returns:
            List of Candle objects, oldest first
        """
        if not self.pool:
            return []
        
        async with self.pool.acquire() as conn:
            if before:
                rows = await conn.fetch(
                    """
                    SELECT time, open, high, low, close, volume,
                           quote_asset_volume, taker_buy_base_asset_volume,
                           taker_buy_quote_asset_volume, number_of_trades,
                           spread_bps, taker_buy_ratio
                    FROM candles_1m
                    WHERE time <= $1
                    ORDER BY time DESC
                    LIMIT $2
                    """,
                    before,
                    limit
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT time, open, high, low, close, volume,
                           quote_asset_volume, taker_buy_base_asset_volume,
                           taker_buy_quote_asset_volume, number_of_trades,
                           spread_bps, taker_buy_ratio
                    FROM candles_1m
                    ORDER BY time DESC
                    LIMIT $1
                    """,
                    limit
                )
        
        # Convert to Candle objects (reverse to get oldest first)
        candles = []
        for row in reversed(rows):
            candles.append(Candle(
                time=row['time'].replace(tzinfo=timezone.utc),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                quote_volume=float(row['quote_asset_volume'] or 0),
                taker_buy_volume=float(row['taker_buy_base_asset_volume'] or 0),
                taker_buy_quote_volume=float(row['taker_buy_quote_asset_volume'] or 0),
                trades=int(row['number_of_trades'] or 0),
                spread_bps=float(row['spread_bps'] or 0),
                taker_buy_ratio=float(row['taker_buy_ratio'] or 0.5),
            ))
        
        return candles
    
    async def get_candles_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[Candle]:
        """
        Fetch candles within a time range.
        
        Args:
            start: Start time (inclusive)
            end: End time (inclusive)
            
        Returns:
            List of Candle objects, oldest first
        """
        if not self.pool:
            return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, open, high, low, close, volume,
                       quote_asset_volume, taker_buy_base_asset_volume,
                       taker_buy_quote_asset_volume, number_of_trades,
                       spread_bps, taker_buy_ratio
                FROM candles_1m
                WHERE time >= $1 AND time <= $2
                ORDER BY time ASC
                """,
                start,
                end
            )
        
        candles = []
        for row in rows:
            candles.append(Candle(
                time=row['time'].replace(tzinfo=timezone.utc),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                quote_volume=float(row['quote_asset_volume'] or 0),
                taker_buy_volume=float(row['taker_buy_base_asset_volume'] or 0),
                taker_buy_quote_volume=float(row['taker_buy_quote_asset_volume'] or 0),
                trades=int(row['number_of_trades'] or 0),
                spread_bps=float(row['spread_bps'] or 0),
                taker_buy_ratio=float(row['taker_buy_ratio'] or 0.5),
            ))
        
        return candles
