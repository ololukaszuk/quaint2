"""
Data Loader Module

Loads candle data from TimescaleDB and prepares it for training.
"""

import asyncio
import asyncpg
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
import logging

from .config import DatabaseConfig, SEQUENCE_LENGTH, PREDICTION_HORIZONS
from .feature_engineering import (
    compute_extended_features,
    normalize_features,
    create_sequences,
    compute_targets,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Async data loader for TimescaleDB candle data."""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """Establish database connection pool."""
        self.pool = await asyncpg.create_pool(
            host=self.db_config.host,
            port=self.db_config.port,
            database=self.db_config.name,
            user=self.db_config.user,
            password=self.db_config.password,
            min_size=2,
            max_size=10,
        )
        logger.info(f"Connected to database {self.db_config.name}")
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection closed")
    
    async def fetch_candles(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        months: int = 13,
    ) -> Tuple[np.ndarray, ...]:
        """
        Fetch candle data from database.
        
        Args:
            start_time: Start of data range (default: now - months)
            end_time: End of data range (default: now)
            months: Number of months to fetch if start_time not specified
            
        Returns:
            Tuple of numpy arrays for each field
        """
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=months * 30)
        
        query = """
            SELECT 
                time,
                open,
                high,
                low,
                close,
                volume,
                COALESCE(quote_asset_volume, volume * close) as quote_asset_volume,
                COALESCE(taker_buy_base_asset_volume, volume * 0.5) as taker_buy_base_asset_volume,
                COALESCE(taker_buy_quote_asset_volume, volume * close * 0.5) as taker_buy_quote_asset_volume,
                COALESCE(number_of_trades, 0) as number_of_trades,
                COALESCE(spread_bps, 0) as spread_bps,
                COALESCE(taker_buy_ratio, 0.5) as taker_buy_ratio
            FROM candles_1m
            WHERE time >= $1 AND time < $2
            ORDER BY time ASC
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, start_time, end_time)
        
        if not rows:
            raise ValueError(f"No data found between {start_time} and {end_time}")
        
        logger.info(f"Fetched {len(rows)} candles from {start_time} to {end_time}")
        
        # Convert to numpy arrays
        times = np.array([row['time'] for row in rows])
        opens = np.array([float(row['open']) for row in rows])
        highs = np.array([float(row['high']) for row in rows])
        lows = np.array([float(row['low']) for row in rows])
        closes = np.array([float(row['close']) for row in rows])
        volumes = np.array([float(row['volume']) for row in rows])
        quote_asset_volumes = np.array([float(row['quote_asset_volume']) for row in rows])
        taker_buy_base_asset_volumes = np.array([float(row['taker_buy_base_asset_volume']) for row in rows])
        taker_buy_quote_asset_volumes = np.array([float(row['taker_buy_quote_asset_volume']) for row in rows])
        number_of_trades = np.array([float(row['number_of_trades']) for row in rows])
        spread_bps = np.array([float(row['spread_bps']) for row in rows])
        taker_buy_ratio = np.array([float(row['taker_buy_ratio']) for row in rows])
        
        return (
            times,
            opens,
            highs,
            lows,
            closes,
            volumes,
            quote_asset_volumes,
            taker_buy_base_asset_volumes,
            taker_buy_quote_asset_volumes,
            number_of_trades,
            spread_bps,
            taker_buy_ratio,
        )
    
    async def prepare_training_data(
        self,
        months: int = 13,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> Tuple[dict, dict, dict, dict]:
        """
        Prepare complete training, validation, and test datasets.
        
        Args:
            months: Number of months of data to use
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            (train_data, val_data, test_data, norm_params)
        """
        # Fetch raw data
        data = await self.fetch_candles(months=months)
        (
            times, opens, highs, lows, closes, volumes,
            quote_asset_volumes, taker_buy_base_asset_volumes,
            taker_buy_quote_asset_volumes, number_of_trades,
            spread_bps, taker_buy_ratio
        ) = data
        
        # Compute extended features
        features = compute_extended_features(
            opens, highs, lows, closes, volumes,
            quote_asset_volumes, taker_buy_base_asset_volumes,
            taker_buy_quote_asset_volumes, number_of_trades,
            spread_bps, taker_buy_ratio
        )
        
        logger.info(f"Computed features shape: {features.shape}")
        
        # Compute targets
        targets = compute_targets(closes, PREDICTION_HORIZONS)
        
        # Split data chronologically
        n = len(features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_features = features[:train_end]
        val_features = features[train_end:val_end]
        test_features = features[val_end:]
        
        train_targets = targets[:train_end]
        val_targets = targets[train_end:val_end]
        test_targets = targets[val_end:]
        
        train_times = times[:train_end]
        val_times = times[train_end:val_end]
        test_times = times[val_end:]
        
        # Normalize using training data statistics
        train_features_norm, mean, std = normalize_features(train_features)
        val_features_norm, _, _ = normalize_features(val_features, mean, std)
        test_features_norm, _, _ = normalize_features(test_features, mean, std)
        
        norm_params = {
            'mean': mean.tolist(),
            'std': std.tolist(),
        }
        
        # Create sequences
        train_sequences = create_sequences(train_features_norm, SEQUENCE_LENGTH)
        val_sequences = create_sequences(val_features_norm, SEQUENCE_LENGTH)
        test_sequences = create_sequences(test_features_norm, SEQUENCE_LENGTH)
        
        # Align targets with sequences (drop first SEQUENCE_LENGTH - 1)
        train_seq_targets = train_targets[SEQUENCE_LENGTH - 1:]
        val_seq_targets = val_targets[SEQUENCE_LENGTH - 1:]
        test_seq_targets = test_targets[SEQUENCE_LENGTH - 1:]
        
        train_seq_times = train_times[SEQUENCE_LENGTH - 1:]
        val_seq_times = val_times[SEQUENCE_LENGTH - 1:]
        test_seq_times = test_times[SEQUENCE_LENGTH - 1:]
        
        # Trim to match sequence count
        min_train = min(len(train_sequences), len(train_seq_targets))
        min_val = min(len(val_sequences), len(val_seq_targets))
        min_test = min(len(test_sequences), len(test_seq_targets))
        
        train_data = {
            'sequences': train_sequences[:min_train],
            'targets': train_seq_targets[:min_train],
            'times': train_seq_times[:min_train],
        }
        
        val_data = {
            'sequences': val_sequences[:min_val],
            'targets': val_seq_targets[:min_val],
            'times': val_seq_times[:min_val],
        }
        
        test_data = {
            'sequences': test_sequences[:min_test],
            'targets': test_seq_targets[:min_test],
            'times': test_seq_times[:min_test],
        }
        
        logger.info(f"Training samples: {len(train_data['sequences'])}")
        logger.info(f"Validation samples: {len(val_data['sequences'])}")
        logger.info(f"Test samples: {len(test_data['sequences'])}")
        
        return train_data, val_data, test_data, norm_params


async def load_training_data(config: DatabaseConfig, months: int = 13) -> Tuple[dict, dict, dict, dict]:
    """
    Convenience function to load training data.
    
    Args:
        config: Database configuration
        months: Number of months of data
        
    Returns:
        (train_data, val_data, test_data, norm_params)
    """
    loader = DataLoader(config)
    await loader.connect()
    try:
        return await loader.prepare_training_data(months=months)
    finally:
        await loader.close()
