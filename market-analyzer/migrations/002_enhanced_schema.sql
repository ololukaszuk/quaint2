-- ============================================================================
-- MARKET ANALYZER ENHANCED SCHEMA
-- Stores ALL data shown in logs for complete API access
-- Version: 2.0 - Complete log data capture
-- ============================================================================

-- ============================================================================
-- TABLE: market_analysis (ENHANCED)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_analysis_v2 (
    id                      BIGSERIAL NOT NULL,
    analysis_time           TIMESTAMPTZ NOT NULL,
    
    -- ========== PRICE & SIGNAL ==========
    price                   NUMERIC(20,8) NOT NULL,
    signal_type             TEXT NOT NULL,
    signal_direction        TEXT NOT NULL,
    signal_confidence       NUMERIC(5,2),
    
    -- Trade setup
    entry_price             NUMERIC(20,8),
    stop_loss               NUMERIC(20,8),
    take_profit_1           NUMERIC(20,8),
    take_profit_2           NUMERIC(20,8),
    take_profit_3           NUMERIC(20,8),
    risk_reward_ratio       NUMERIC(5,2),
    
    -- ========== SIGNAL REASONING (NEW) ==========
    -- Top factors with weights from log output
    signal_factors          JSONB,  -- [{"description": "At strong resistance", "weight": -30, "type": "bearish"}, ...]
    
    -- ========== MULTI-TIMEFRAME TRENDS (ENHANCED) ==========
    trends                  JSONB,  -- {"5m": {"direction": "UPTREND", "strength": 1.0, "ema": "BULLISH", "structure": "HH/HL"}, ...}
    
    -- ========== PIVOT POINTS (COMPLETE) ==========
    pivot_daily             NUMERIC(20,8),
    price_vs_pivot          TEXT,
    
    -- Traditional pivots
    pivot_r3_traditional    NUMERIC(20,8),
    pivot_r2_traditional    NUMERIC(20,8),
    pivot_r1_traditional    NUMERIC(20,8),
    pivot_s1_traditional    NUMERIC(20,8),
    pivot_s2_traditional    NUMERIC(20,8),
    pivot_s3_traditional    NUMERIC(20,8),
    
    -- Fibonacci pivots
    pivot_r3_fibonacci      NUMERIC(20,8),
    pivot_r2_fibonacci      NUMERIC(20,8),
    pivot_r1_fibonacci      NUMERIC(20,8),
    pivot_s1_fibonacci      NUMERIC(20,8),
    pivot_s2_fibonacci      NUMERIC(20,8),
    pivot_s3_fibonacci      NUMERIC(20,8),
    
    -- Camarilla pivots
    pivot_r4_camarilla      NUMERIC(20,8),
    pivot_r3_camarilla      NUMERIC(20,8),
    pivot_s3_camarilla      NUMERIC(20,8),
    pivot_s4_camarilla      NUMERIC(20,8),
    
    -- Pivot confluence zones
    pivot_confluence_zones  JSONB,  -- [{"price": 87332, "type": "support", "strength": 0.2, "methods": ["Camarilla"]}, ...]
    
    -- ========== SMART MONEY CONCEPTS (COMPLETE) ==========
    smc_bias                TEXT,
    smc_price_zone          TEXT,
    smc_equilibrium         NUMERIC(20,8),
    
    -- Order blocks
    smc_order_blocks        JSONB,  -- [{"type": "bullish", "low": 87132, "high": 87588, "strength": 1.0, "distance_pct": 0.2}, ...]
    
    -- Fair value gaps
    smc_fvgs                JSONB,  -- [{"type": "bullish", "low": 86426, "high": 86820, "unfilled": true}, ...]
    
    -- Structure breaks
    smc_breaks              JSONB,  -- [{"type": "CHoCH", "direction": "BULLISH", "price": 86535}, ...]
    
    -- Liquidity pools
    smc_liquidity           JSONB,  -- {"buy_side": [87782, 87793, 87800], "sell_side": [87607, 87537, 87338]}
    
    -- ========== SUPPORT/RESISTANCE LEVELS (ALL LEVELS) ==========
    support_levels          JSONB,  -- [{"price": 87556, "strength": 0.62, "touches": 11, "timeframes": ["15m", "5m"], "distance_pct": 0.21}, ...]
    resistance_levels       JSONB,  -- [{"price": 87769, "strength": 1.0, "touches": 28, "timeframes": ["15m", "5m"], "distance_pct": 0.03}, ...]
    
    -- ========== MOMENTUM (ALL TIMEFRAMES) ==========
    momentum                JSONB,  -- {"5m": {"rsi": 52.1, "volume_ratio": 0.33, "taker_buy_ratio": 0.67}, ...}
    
    -- ========== MARKET STRUCTURE (NEW) ==========
    structure_pattern       TEXT,       -- CONTRACTING, EXPANDING, etc.
    structure_last_high     NUMERIC(20,8),
    structure_last_low      NUMERIC(20,8),
    
    -- ========== WARNINGS/ALERTS (NEW) ==========
    warnings                JSONB,  -- [{"type": "CLOSE_TO_SUPPORT", "message": "CLOSE TO STRONG SUPPORT ($87,556)", ...}, ...]
    
    -- ========== SUMMARY & METADATA ==========
    summary                 TEXT,
    action_recommendation   TEXT,   -- WAIT, LONG, SHORT, etc.
    signal_changed          BOOLEAN DEFAULT FALSE,
    previous_signal         TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (id, analysis_time)
);

-- Convert to hypertable
SELECT create_hypertable(
    'market_analysis_v2',
    'analysis_time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_market_analysis_v2_time_desc ON market_analysis_v2 (analysis_time DESC);
CREATE INDEX IF NOT EXISTS idx_market_analysis_v2_signal_changed ON market_analysis_v2 (analysis_time DESC) WHERE signal_changed = TRUE;
CREATE INDEX IF NOT EXISTS idx_market_analysis_v2_signal_type ON market_analysis_v2 (signal_type, analysis_time DESC);

-- Retention policy
SELECT add_retention_policy('market_analysis_v2', INTERVAL '3 months', if_not_exists => TRUE);

-- ============================================================================
-- TABLE: market_signals (ENHANCED)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_signals_v2 (
    id                      BIGSERIAL NOT NULL,
    signal_time             TIMESTAMPTZ NOT NULL,
    
    -- Signal info
    signal_type             TEXT NOT NULL,
    signal_direction        TEXT NOT NULL,
    signal_confidence       NUMERIC(5,2),
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
    key_reasons             JSONB,  -- CHANGED to JSONB: [{"description": "...", "weight": -30}, ...]
    
    -- Metadata
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (id, signal_time)
);

-- Convert to hypertable
SELECT create_hypertable(
    'market_signals_v2',
    'signal_time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_market_signals_v2_time_desc ON market_signals_v2 (signal_time DESC);

-- Retention policy
SELECT add_retention_policy('market_signals_v2', INTERVAL '6 months', if_not_exists => TRUE);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Drop existing views first
DROP VIEW IF EXISTS v_latest_signal CASCADE;
DROP VIEW IF EXISTS v_latest_analysis CASCADE;

CREATE VIEW v_latest_analysis AS
SELECT 
    analysis_time,
    price,
    signal_type,
    signal_direction,
    signal_confidence,
    signal_factors,
    trends,
    support_levels,
    resistance_levels,
    momentum,
    smc_bias,
    smc_price_zone,
    smc_order_blocks,
    smc_fvgs,
    pivot_daily,
    warnings,
    summary,
    action_recommendation
FROM market_analysis_v2
ORDER BY analysis_time DESC
LIMIT 1;

CREATE VIEW v_latest_signal AS
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
    key_reasons,
    summary
FROM market_signals_v2
ORDER BY signal_time DESC
LIMIT 1;

-- ============================================================================
-- Done
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Market Analyzer Enhanced Schema Created';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Tables: market_analysis_v2, market_signals_v2';
    RAISE NOTICE 'Complete log data capture enabled';
    RAISE NOTICE '============================================';
END $$;
