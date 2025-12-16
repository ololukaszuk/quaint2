"""
Configuration for Market Analyzer service.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Service configuration loaded from environment variables."""
    
    # Database
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "btc_ml_production"))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "mltrader"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    
    # Polling
    poll_interval_seconds: int = field(default_factory=lambda: int(os.getenv("POLL_INTERVAL", "5")))
    
    # Analysis settings
    timeframes: List[str] = field(default_factory=lambda: ["5m", "15m", "1h", "4h", "1d"])
    
    # How much historical data to load for each timeframe analysis
    lookback_candles: int = field(default_factory=lambda: int(os.getenv("LOOKBACK_CANDLES", "500")))
    
    # Support/Resistance detection
    sr_min_touches: int = 2
    sr_price_tolerance_pct: float = 0.15  # 0.15% tolerance for grouping levels
    
    # Trend detection
    ema_fast: int = 8
    ema_slow: int = 21
    ema_trend: int = 50
    
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # Health check
    health_port: int = field(default_factory=lambda: int(os.getenv("HEALTH_PORT", "8082")))
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.db_password:
            raise ValueError("DB_PASSWORD environment variable is required")
