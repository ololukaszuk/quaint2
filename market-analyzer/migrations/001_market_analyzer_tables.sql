-- ============================================================================
-- MARKET ANALYZER TABLES
-- Add to init.sql or run separately after database is created
-- ============================================================================

-- ============================================================================
-- TABLE: market_analysis
-- Stores each analysis snapshot for historical tracking and change detection
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_analysis (
    id                      BIGSERIAL NOT NULL,
    analysis_time           TIMESTAMPTZ NOT NULL,
    
    -- Current price at analysis time
    price                   NUMERIC(20,8) NOT NULL,
    
    -- Signal
    signal_type             TEXT NOT NULL,  -- STRONG_BUY, BUY, WEAK_BUY, NEUTRAL, WEAK_SELL, SELL, STRONG_SELL
    signal_direction        TEXT NOT NULL,  -- LONG, SHORT, NONE
    signal_confidence       NUMERIC(5,2),   -- 0-100
    
    -- Trade setup (nullable if NEUTRAL)
    entry_price             NUMERIC(20,8),
    stop_loss               NUMERIC(20,8),
    take_profit_1           NUMERIC(20,8),
    take_profit_2           NUMERIC(20,8),
    take_profit_3           NUMERIC(20,8),
    risk_reward_ratio       NUMERIC(5,2),
    
    -- Multi-timeframe trends (as JSON for flexibility)
    trends                  JSONB,          -- {"5m": {"direction": "UPTREND", "strength": 0.8}, ...}
    
    -- Key levels
    nearest_support         NUMERIC(20,8),
    nearest_resistance      NUMERIC(20,8),
    support_strength        NUMERIC(5,4),
    resistance_strength     NUMERIC(5,4),
    
    -- SMC
    smc_bias                TEXT,           -- BULLISH, BEARISH, NEUTRAL
    price_zone              TEXT,           -- PREMIUM, DISCOUNT, EQUILIBRIUM
    equilibrium_price       NUMERIC(20,8),
    
    -- Pivot (daily)
    daily_pivot             NUMERIC(20,8),
    price_vs_pivot          TEXT,           -- ABOVE, BELOW
    
    -- Momentum
    rsi_1h                  NUMERIC(5,2),
    volume_ratio_1h         NUMERIC(5,2),
    
    -- Summary text
    summary                 TEXT,
    
    -- Change tracking
    signal_changed          BOOLEAN DEFAULT FALSE,  -- True if signal changed from previous
    previous_signal         TEXT,                   -- Previous signal_type for comparison
    
    -- Metadata
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (id, analysis_time)
);

-- Convert to hypertable
SELECT create_hypertable(
    'market_analysis',
    'analysis_time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Index for recent analyses
CREATE INDEX IF NOT EXISTS idx_market_analysis_time_desc 
    ON market_analysis (analysis_time DESC);

-- Index for signal changes
CREATE INDEX IF NOT EXISTS idx_market_analysis_signal_changed 
    ON market_analysis (analysis_time DESC) 
    WHERE signal_changed = TRUE;

-- Index for signal type queries
CREATE INDEX IF NOT EXISTS idx_market_analysis_signal_type 
    ON market_analysis (signal_type, analysis_time DESC);

-- Retention policy - keep 3 months of analysis history
SELECT add_retention_policy('market_analysis', INTERVAL '3 months', if_not_exists => TRUE);

-- ============================================================================
-- TABLE: market_signals
-- Stores only signal CHANGES for quick alerting
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_signals (
    id                      BIGSERIAL NOT NULL,
    signal_time             TIMESTAMPTZ NOT NULL,
    
    -- Signal info
    signal_type             TEXT NOT NULL,
    signal_direction        TEXT NOT NULL,
    signal_confidence       NUMERIC(5,2),
    
    -- Price at signal
    price                   NUMERIC(20,8) NOT NULL,
    
    -- Trade setup
    entry_price             NUMERIC(20,8),
    stop_loss               NUMERIC(20,8),
    take_profit_1           NUMERIC(20,8),
    take_profit_2           NUMERIC(20,8),
    take_profit_3           NUMERIC(20,8),
    risk_reward_ratio       NUMERIC(5,2),
    
    -- What changed
    previous_signal_type    TEXT,
    previous_direction      TEXT,
    
    -- Context
    summary                 TEXT,
    key_reasons             TEXT[],         -- Top 3 reasons for signal
    
    -- Tracking
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (id, signal_time)
);

-- Convert to hypertable
SELECT create_hypertable(
    'market_signals',
    'signal_time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Index for recent signals
CREATE INDEX IF NOT EXISTS idx_market_signals_time_desc 
    ON market_signals (signal_time DESC);

-- Retention policy - keep 6 months of signals
SELECT add_retention_policy('market_signals', INTERVAL '6 months', if_not_exists => TRUE);

-- ============================================================================
-- VIEW: Latest signal
-- ============================================================================

CREATE OR REPLACE VIEW v_latest_signal AS
SELECT 
    signal_time,
    signal_type,
    signal_direction,
    signal_confidence,
    price,
    entry_price,
    stop_loss,
    take_profit_1,
    take_profit_2,
    take_profit_3,
    risk_reward_ratio,
    summary
FROM market_signals
ORDER BY signal_time DESC
LIMIT 1;

-- ============================================================================
-- VIEW: Recent signal changes
-- ============================================================================

CREATE OR REPLACE VIEW v_recent_signal_changes AS
SELECT 
    signal_time,
    signal_type,
    signal_direction,
    signal_confidence,
    price,
    previous_signal_type,
    previous_direction,
    summary
FROM market_signals
ORDER BY signal_time DESC
LIMIT 20;

-- ============================================================================
-- FUNCTION: Get signal history for backtesting
-- ============================================================================

CREATE OR REPLACE FUNCTION get_signal_history(
    p_start TIMESTAMPTZ,
    p_end TIMESTAMPTZ,
    p_direction TEXT DEFAULT NULL  -- Filter by 'LONG', 'SHORT', or NULL for all
)
RETURNS TABLE (
    signal_time TIMESTAMPTZ,
    signal_type TEXT,
    signal_direction TEXT,
    signal_confidence NUMERIC,
    price NUMERIC,
    entry_price NUMERIC,
    stop_loss NUMERIC,
    take_profit_1 NUMERIC,
    risk_reward_ratio NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ms.signal_time,
        ms.signal_type,
        ms.signal_direction,
        ms.signal_confidence,
        ms.price,
        ms.entry_price,
        ms.stop_loss,
        ms.take_profit_1,
        ms.risk_reward_ratio
    FROM market_signals ms
    WHERE ms.signal_time >= p_start 
      AND ms.signal_time <= p_end
      AND (p_direction IS NULL OR ms.signal_direction = p_direction)
    ORDER BY ms.signal_time;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Done
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Market Analyzer Tables Created Successfully';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Tables: market_analysis, market_signals';
    RAISE NOTICE 'Views: v_latest_signal, v_recent_signal_changes';
    RAISE NOTICE 'Retention: 3 months (analysis), 6 months (signals)';
    RAISE NOTICE '============================================';
END $$;
