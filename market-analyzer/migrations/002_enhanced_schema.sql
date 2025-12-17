-- ============================================================================
-- MARKET ANALYZER ENHANCED SCHEMA - MIGRATION 002
-- Upgrade market_analysis table with complete data capture
-- Version: 2.0 - Add all analysis details (signals, pivots, SMC, levels)
-- ============================================================================

-- ============================================================================
-- STEP 1: Add missing columns to market_analysis table
-- ============================================================================

-- Signal factors (reasoning for signal)
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS signal_factors JSONB;
-- [{\"description\": \"At strong resistance\", \"weight\": -30, \"type\": \"bearish\"}, ...]

-- Support/Resistance levels (complete list)
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS support_levels JSONB;
-- [{\"price\": 87556, \"strength\": 0.62, \"touches\": 11, \"timeframe\": \"15m\", \"distance_pct\": 0.21}, ...]

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS resistance_levels JSONB;
-- [{\"price\": 87769, \"strength\": 1.0, \"touches\": 28, \"timeframe\": \"15m\", \"distance_pct\": 0.03}, ...]

-- Momentum for all timeframes
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS momentum JSONB;
-- {\"5m\": {\"rsi\": 52.1, \"volume_ratio\": 0.33, \"taker_buy_ratio\": 0.67}, ...}

-- ========== PIVOT POINTS (COMPLETE) ==========

-- Traditional pivots
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_daily NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS price_vs_pivot TEXT; -- ABOVE or BELOW

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r3_traditional NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r2_traditional NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r1_traditional NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s1_traditional NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s2_traditional NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s3_traditional NUMERIC(20,8);

-- Fibonacci pivots
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r3_fibonacci NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r2_fibonacci NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r1_fibonacci NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s1_fibonacci NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s2_fibonacci NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s3_fibonacci NUMERIC(20,8);

-- Camarilla pivots
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r4_camarilla NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r3_camarilla NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s3_camarilla NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s4_camarilla NUMERIC(20,8);

-- Pivot confluence zones
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_confluence_zones JSONB;
-- [{\"price\": 87332, \"type\": \"support\", \"strength\": 0.2, \"methods\": [\"Camarilla\"]}, ...]

-- ========== SMART MONEY CONCEPTS (COMPLETE) ==========

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS smc_price_zone TEXT; -- PREMIUM, DISCOUNT, EQUILIBRIUM (redundant with price_zone but explicit)

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS smc_equilibrium NUMERIC(20,8);

-- Order blocks
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS smc_order_blocks JSONB;
-- [{\"type\": \"bullish\", \"low\": 87132, \"high\": 87588, \"strength\": 1.0, \"distance_pct\": 0.2}, ...]

-- Fair value gaps
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS smc_fvgs JSONB;
-- [{\"type\": \"bullish\", \"low\": 86426, \"high\": 86820, \"unfilled\": true}, ...]

-- Structure breaks (CHoCH, BOS)
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS smc_breaks JSONB;
-- [{\"type\": \"CHoCH\", \"direction\": \"BULLISH\", \"price\": 86535}, ...]

-- Liquidity pools
ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS smc_liquidity JSONB;
-- {\"buy_side\": [87782, 87793, 87800], \"sell_side\": [87607, 87537, 87338]}

-- ========== MARKET STRUCTURE ==========

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS structure_pattern TEXT; -- CONTRACTING, EXPANDING, HIGHER_HIGHS_LOWS, LOWER_HIGHS_LOWS, etc.

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS structure_last_high NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS structure_last_low NUMERIC(20,8);

-- ========== WARNINGS/ALERTS ==========

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS warnings JSONB;
-- [{\"type\": \"CLOSE_TO_SUPPORT\", \"message\": \"CLOSE TO STRONG SUPPORT ($87,556)\", \"severity\": \"MEDIUM\"}, ...]

-- ========== ACTION RECOMMENDATION ==========

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS action_recommendation TEXT; -- WAIT, LONG, SHORT

-- ============================================================================
-- STEP 2: Create indexes on new columns for query performance
-- ============================================================================

-- Index on pivot points (common queries)
CREATE INDEX IF NOT EXISTS idx_market_analysis_pivot_daily
ON market_analysis (pivot_daily);

-- Index on SMC bias (common filter)
CREATE INDEX IF NOT EXISTS idx_market_analysis_smc_bias
ON market_analysis (smc_bias, analysis_time DESC);

-- Index on action recommendation (for trading views)
CREATE INDEX IF NOT EXISTS idx_market_analysis_action
ON market_analysis (action_recommendation, analysis_time DESC);

-- JSONB index on signal factors (for querying specific reasons)
CREATE INDEX IF NOT EXISTS idx_market_analysis_signal_factors_jsonb
ON market_analysis USING gin(signal_factors);

-- JSONB index on momentum (for querying RSI, volume, etc.)
CREATE INDEX IF NOT EXISTS idx_market_analysis_momentum_jsonb
ON market_analysis USING gin(momentum);

-- JSONB index on warnings (for alert queries)
CREATE INDEX IF NOT EXISTS idx_market_analysis_warnings_jsonb
ON market_analysis USING gin(warnings);

-- ============================================================================
-- STEP 3: Update market_signals table if needed
-- ============================================================================

-- Ensure signal_factors is JSONB (in case it was TEXT before)
ALTER TABLE market_signals
ALTER COLUMN key_reasons TYPE JSONB USING 
  CASE 
    WHEN key_reasons IS NOT NULL THEN to_jsonb(key_reasons)
    ELSE NULL
  END;

-- Add new columns to market_signals for richer signal data
ALTER TABLE market_signals
ADD COLUMN IF NOT EXISTS signal_factors JSONB;

ALTER TABLE market_signals
ADD COLUMN IF NOT EXISTS smc_bias TEXT;

ALTER TABLE market_signals
ADD COLUMN IF NOT EXISTS pivot_daily NUMERIC(20,8);

ALTER TABLE market_signals
ADD COLUMN IF NOT EXISTS nearest_support NUMERIC(20,8);

ALTER TABLE market_signals
ADD COLUMN IF NOT EXISTS nearest_resistance NUMERIC(20,8);

-- ============================================================================
-- STEP 4: Update existing views
-- ============================================================================

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
    smc_equilibrium,
    smc_order_blocks,
    smc_fvgs,
    smc_breaks,
    smc_liquidity,
    pivot_daily,
    price_vs_pivot,
    warnings,
    summary,
    action_recommendation,
    signal_changed,
    previous_signal
FROM market_analysis
ORDER BY analysis_time DESC
LIMIT 1;

-- ============================================================================
-- STEP 5: Create new utility views
-- ============================================================================

DROP VIEW IF EXISTS v_latest_warnings CASCADE;

CREATE VIEW v_latest_warnings AS
SELECT
    analysis_time,
    price,
    warnings,
    summary
FROM market_analysis
WHERE warnings IS NOT NULL
  AND jsonb_array_length(warnings) > 0
ORDER BY analysis_time DESC
LIMIT 10;

DROP VIEW IF EXISTS v_pivot_analysis CASCADE;

CREATE VIEW v_pivot_analysis AS
SELECT
    analysis_time,
    price,
    price_vs_pivot,
    pivot_daily,
    pivot_r3_traditional,
    pivot_r2_traditional,
    pivot_r1_traditional,
    pivot_s1_traditional,
    pivot_s2_traditional,
    pivot_s3_traditional,
    CASE 
        WHEN price > pivot_daily THEN 'ABOVE'
        WHEN price < pivot_daily THEN 'BELOW'
        ELSE 'AT'
    END as position_vs_pivot
FROM market_analysis
ORDER BY analysis_time DESC
LIMIT 100;

DROP VIEW IF EXISTS v_smc_analysis CASCADE;

CREATE VIEW v_smc_analysis AS
SELECT
    analysis_time,
    price,
    smc_bias,
    smc_price_zone,
    smc_equilibrium,
    smc_order_blocks,
    smc_fvgs,
    smc_breaks,
    smc_liquidity
FROM market_analysis
ORDER BY analysis_time DESC
LIMIT 100;

-- ============================================================================
-- STEP 6: Create helper functions
-- ============================================================================

DROP FUNCTION IF EXISTS get_analysis_between(TIMESTAMPTZ, TIMESTAMPTZ);

CREATE FUNCTION get_analysis_between(
    p_start TIMESTAMPTZ,
    p_end TIMESTAMPTZ
)
RETURNS TABLE (
    analysis_time TIMESTAMPTZ,
    price NUMERIC,
    signal_type TEXT,
    signal_direction TEXT,
    signal_confidence NUMERIC,
    smc_bias TEXT,
    smc_price_zone TEXT,
    pivot_daily NUMERIC,
    warnings JSONB,
    summary TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ma.analysis_time,
        ma.price,
        ma.signal_type,
        ma.signal_direction,
        ma.signal_confidence,
        ma.smc_bias,
        ma.smc_price_zone,
        ma.pivot_daily,
        ma.warnings,
        ma.summary
    FROM market_analysis ma
    WHERE ma.analysis_time >= p_start
      AND ma.analysis_time <= p_end
    ORDER BY ma.analysis_time DESC;
END;
$$ LANGUAGE plpgsql;

DROP FUNCTION IF EXISTS get_warning_history(TIMESTAMPTZ, TIMESTAMPTZ);

CREATE FUNCTION get_warning_history(
    p_start TIMESTAMPTZ,
    p_end TIMESTAMPTZ
)
RETURNS TABLE (
    analysis_time TIMESTAMPTZ,
    price NUMERIC,
    warning_count INT,
    warnings JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ma.analysis_time,
        ma.price,
        CASE 
            WHEN ma.warnings IS NULL THEN 0
            ELSE jsonb_array_length(ma.warnings)
        END as warning_count,
        ma.warnings
    FROM market_analysis ma
    WHERE ma.analysis_time >= p_start
      AND ma.analysis_time <= p_end
      AND ma.warnings IS NOT NULL
    ORDER BY ma.analysis_time DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- STEP 7: Print migration results
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Migration 002: Enhanced Schema Applied';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Added columns:';
    RAISE NOTICE '  • Signal factors (JSONB)';
    RAISE NOTICE '  • Support/Resistance levels (JSONB)';
    RAISE NOTICE '  • Momentum (JSONB)';
    RAISE NOTICE '  • Pivot points (all methods)';
    RAISE NOTICE '  • SMC data (OBs, FVGs, breaks, liquidity)';
    RAISE NOTICE '  • Market structure';
    RAISE NOTICE '  • Warnings';
    RAISE NOTICE '  • Action recommendation';
    RAISE NOTICE '';
    RAISE NOTICE 'Created:';
    RAISE NOTICE '  • 7 new indexes for query performance';
    RAISE NOTICE '  • 4 new utility views (analysis, warnings, pivots, SMC)';
    RAISE NOTICE '  • 2 helper functions for time-range queries';
    RAISE NOTICE '';
    RAISE NOTICE 'Status: ✅ Migration 002 Complete';
    RAISE NOTICE '============================================';
END $$;
