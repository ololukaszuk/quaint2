"""
LLM Analyst Configuration
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration from environment variables."""
    
    # Database
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "btc_ml_production"))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "mltrader"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    
    # Ollama - full URL, no assumptions about network topology
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "deepseek-r1:32b"))
    ollama_timeout: int = field(default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "300")))  # 5 min for reasoning models
    
    # Analysis settings
    analysis_interval_candles: int = field(default_factory=lambda: int(os.getenv("ANALYSIS_INTERVAL_CANDLES", "5")))  # Every 5 1m candles
    candles_1h_lookback: int = field(default_factory=lambda: int(os.getenv("CANDLES_1H_LOOKBACK", "120")))  # ~5 days
    candles_15m_lookback: int = field(default_factory=lambda: int(os.getenv("CANDLES_15M_LOOKBACK", "20")))  # Recent detail
    signal_history_count: int = field(default_factory=lambda: int(os.getenv("SIGNAL_HISTORY_COUNT", "15")))
    
    # Polling
    poll_interval_seconds: int = field(default_factory=lambda: int(os.getenv("POLL_INTERVAL", "10")))
    
    # Health check
    health_port: int = field(default_factory=lambda: int(os.getenv("HEALTH_PORT", "8083")))
    
    @property
    def db_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def ollama_generate_url(self) -> str:
        """Full URL for Ollama generate endpoint."""
        base = self.ollama_base_url.rstrip("/")
        return f"{base}/api/generate"
