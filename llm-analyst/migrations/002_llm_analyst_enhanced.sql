-- ============================================================================
-- LLM ANALYST ENHANCED SCHEMA - MIGRATION 002
-- Upgrade llm_analysis table to store full market context at analysis time
-- Version: 2.0 - Add market context, signal factors, SMC bias, trends, warnings
-- ============================================================================

-- ============================================================================
-- STEP 1: Add new columns for enhanced logging
-- ============================================================================

-- Invalidation level (where the prediction is wrong)
ALTER TABLE llm_analysis
ADD COLUMN IF NOT EXISTS invalidation_level NUMERIC(20,8);

-- Critical levels extracted from LLM response
ALTER TABLE llm_analysis
ADD COLUMN IF NOT EXISTS critical_support NUMERIC(20,8);

ALTER TABLE llm_analysis
ADD COLUMN IF NOT EXISTS critical_resistance NUMERIC(20,8);

-- Market context at the time of analysis (what the LLM saw)
-- This is crucial for understanding why predictions were made
ALTER TABLE llm_analysis
ADD COLUMN IF NOT EXISTS market_context JSONB;
-- {
--   "signal_type": "WEAK_BUY",
--   "signal_direction": "LONG",
--   "signal_confidence": 44.5,
--   "smc_bias": "BULLISH",
--   "price_zone": "DISCOUNT",
--   "action_recommendation": "WAIT",
--   "nearest_support": 87556,
--   "nearest_resistance": 87769
-- }

-- Signal factors that were used for the analysis
ALTER TABLE llm_analysis
ADD COLUMN IF NOT EXISTS signal_factors_used JSONB;
-- [{"description": "At strong resistance", "weight": -30}, ...]

-- SMC bias at time of analysis
ALTER TABLE llm_analysis
ADD COLUMN IF NOT EXISTS smc_bias_at_analysis TEXT;

-- Trends at time of analysis
ALTER TABLE llm_analysis
ADD COLUMN IF NOT EXISTS trends_at_analysis JSONB;
-- {"5m": {"direction": "UPTREND", "strength": 0.6}, ...}

-- Warnings at time of analysis
ALTER TABLE llm_analysis
ADD COLUMN IF NOT EXISTS warnings_at_analysis JSONB;
-- [{"type": "CLOSE_TO_RESISTANCE", "message": "...", "severity": "MEDIUM"}, ...]

-- ============================================================================
-- STEP 2: Add indexes for new columns
-- ============================================================================

-- Index on SMC bias for correlation analysis
CREATE INDEX IF NOT EXISTS idx_llm_analysis_smc_bias
ON llm_analysis (smc_bias_at_analysis, analysis_time DESC);

-- JSONB index on market context for querying specific conditions
CREATE INDEX IF NOT EXISTS idx_llm_analysis_market_context_jsonb
ON llm_analysis USING gin(market_context);

-- JSONB index on signal factors for querying specific factors
CREATE INDEX IF NOT EXISTS idx_llm_analysis_signal_factors_jsonb
ON llm_analysis USING gin(signal_factors_used);

-- Index for accuracy analysis by direction
CREATE INDEX IF NOT EXISTS idx_llm_analysis_direction_accuracy
ON llm_analysis (prediction_direction, direction_correct_1h, analysis_time DESC)
WHERE direction_correct_1h IS NOT NULL;

-- ============================================================================
-- STEP 3: Create enhanced views
-- ============================================================================

-- Drop and recreate enhanced prediction view
DROP VIEW IF EXISTS v_llm_predictions_enhanced CASCADE;

CREATE VIEW v_llm_predictions_enhanced AS
SELECT 
    analysis_time,
    price,
    prediction_direction,
    prediction_confidence,
    predicted_price_1h,
    predicted_price_4h,
    invalidation_level,
    critical_support,
    critical_resistance,
    reasoning,
    model_name,
    response_time_seconds,
    -- Market context
    market_context->>'signal_type' as market_signal_type,
    market_context->>'signal_direction' as market_signal_direction,
    (market_context->>'signal_confidence')::NUMERIC as market_signal_confidence,
    smc_bias_at_analysis,
    market_context->>'action_recommendation' as market_action,
    -- Accuracy
    actual_price_1h,
    actual_price_4h,
    direction_correct_1h,
    direction_correct_4h,
    -- Calculate price movement
    CASE 
        WHEN actual_price_1h IS NOT NULL THEN 
            ROUND(((actual_price_1h - price) / price * 100)::NUMERIC, 2)
        ELSE NULL
    END as actual_move_1h_pct,
    CASE 
        WHEN predicted_price_1h IS NOT NULL THEN 
            ROUND(((predicted_price_1h - price) / price * 100)::NUMERIC, 2)
        ELSE NULL
    END as predicted_move_1h_pct
FROM llm_analysis
ORDER BY analysis_time DESC;

-- View for analyzing prediction accuracy by market conditions
DROP VIEW IF EXISTS v_llm_accuracy_by_conditions CASCADE;

CREATE VIEW v_llm_accuracy_by_conditions AS
SELECT 
    smc_bias_at_analysis,
    market_context->>'signal_type' as market_signal,
    prediction_direction,
    COUNT(*) as total_predictions,
    COUNT(direction_correct_1h) as evaluated,
    SUM(CASE WHEN direction_correct_1h THEN 1 ELSE 0 END) as correct_1h,
    ROUND(AVG(CASE WHEN direction_correct_1h THEN 1.0 ELSE 0.0 END) * 100, 1) as accuracy_1h_pct,
    SUM(CASE WHEN direction_correct_4h THEN 1 ELSE 0 END) as correct_4h,
    ROUND(AVG(CASE WHEN direction_correct_4h THEN 1.0 ELSE 0.0 END) * 100, 1) as accuracy_4h_pct,
    ROUND(AVG(response_time_seconds), 1) as avg_response_time
FROM llm_analysis
WHERE analysis_time > NOW() - INTERVAL '7 days'
GROUP BY 
    smc_bias_at_analysis,
    market_context->>'signal_type',
    prediction_direction
HAVING COUNT(direction_correct_1h) >= 3
ORDER BY accuracy_1h_pct DESC;

-- View for tracking LLM vs market-analyzer agreement
DROP VIEW IF EXISTS v_llm_market_agreement CASCADE;

CREATE VIEW v_llm_market_agreement AS
SELECT 
    analysis_time,
    price,
    -- LLM prediction
    prediction_direction as llm_direction,
    prediction_confidence as llm_confidence,
    -- Market analyzer signal
    market_context->>'signal_direction' as market_direction,
    (market_context->>'signal_confidence')::NUMERIC as market_confidence,
    smc_bias_at_analysis,
    -- Agreement check
    CASE 
        WHEN prediction_direction = 'BULLISH' AND market_context->>'signal_direction' IN ('LONG') THEN 'AGREE_BULLISH'
        WHEN prediction_direction = 'BEARISH' AND market_context->>'signal_direction' IN ('SHORT') THEN 'AGREE_BEARISH'
        WHEN prediction_direction = 'NEUTRAL' AND market_context->>'signal_direction' = 'NONE' THEN 'AGREE_NEUTRAL'
        ELSE 'DISAGREE'
    END as agreement,
    -- Outcome
    direction_correct_1h,
    direction_correct_4h
FROM llm_analysis
WHERE market_context IS NOT NULL
ORDER BY analysis_time DESC;

-- ============================================================================
-- STEP 4: Create helper functions
-- ============================================================================

-- Function to get LLM accuracy for specific market conditions
DROP FUNCTION IF EXISTS get_llm_accuracy_for_conditions(TEXT, TEXT);

CREATE FUNCTION get_llm_accuracy_for_conditions(
    p_smc_bias TEXT DEFAULT NULL,
    p_market_signal TEXT DEFAULT NULL
)
RETURNS TABLE (
    condition_smc_bias TEXT,
    condition_market_signal TEXT,
    total_predictions BIGINT,
    evaluated BIGINT,
    accuracy_1h NUMERIC,
    accuracy_4h NUMERIC,
    avg_response_time NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        la.smc_bias_at_analysis,
        la.market_context->>'signal_type',
        COUNT(*)::BIGINT,
        COUNT(la.direction_correct_1h)::BIGINT,
        ROUND(AVG(CASE WHEN la.direction_correct_1h THEN 1.0 ELSE 0.0 END) * 100, 1),
        ROUND(AVG(CASE WHEN la.direction_correct_4h THEN 1.0 ELSE 0.0 END) * 100, 1),
        ROUND(AVG(la.response_time_seconds), 1)
    FROM llm_analysis la
    WHERE (p_smc_bias IS NULL OR la.smc_bias_at_analysis = p_smc_bias)
      AND (p_market_signal IS NULL OR la.market_context->>'signal_type' = p_market_signal)
      AND la.analysis_time > NOW() - INTERVAL '30 days'
    GROUP BY la.smc_bias_at_analysis, la.market_context->>'signal_type'
    HAVING COUNT(la.direction_correct_1h) >= 3
    ORDER BY 5 DESC;  -- Order by accuracy_1h
END;
$$ LANGUAGE plpgsql;

-- Function to analyze what factors correlate with correct predictions
DROP FUNCTION IF EXISTS analyze_successful_predictions();

CREATE FUNCTION analyze_successful_predictions()
RETURNS TABLE (
    signal_type TEXT,
    prediction_direction TEXT,
    smc_bias TEXT,
    total_correct BIGINT,
    total_predictions BIGINT,
    success_rate NUMERIC,
    common_factors TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        la.market_context->>'signal_type',
        la.prediction_direction,
        la.smc_bias_at_analysis,
        SUM(CASE WHEN la.direction_correct_1h THEN 1 ELSE 0 END)::BIGINT,
        COUNT(*)::BIGINT,
        ROUND(AVG(CASE WHEN la.direction_correct_1h THEN 1.0 ELSE 0.0 END) * 100, 1),
        '' -- Placeholder for common factors (would need more complex aggregation)
    FROM llm_analysis la
    WHERE la.analysis_time > NOW() - INTERVAL '7 days'
      AND la.direction_correct_1h IS NOT NULL
    GROUP BY 
        la.market_context->>'signal_type',
        la.prediction_direction,
        la.smc_bias_at_analysis
    HAVING COUNT(*) >= 5
    ORDER BY 6 DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- STEP 5: Update the accuracy update function to use enhanced data
-- ============================================================================

DROP FUNCTION IF EXISTS update_llm_accuracy();

CREATE OR REPLACE FUNCTION update_llm_accuracy()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER := 0;
    rec RECORD;
    price_1h NUMERIC(20,8);
    price_4h NUMERIC(20,8);
BEGIN
    -- Find predictions that need accuracy update (1h old, no actual_price_1h)
    FOR rec IN 
        SELECT id, analysis_time, price, prediction_direction, predicted_price_1h, predicted_price_4h
        FROM llm_analysis
        WHERE actual_price_1h IS NULL
          AND analysis_time < NOW() - INTERVAL '1 hour'
        ORDER BY analysis_time
        LIMIT 100
    LOOP
        -- Get actual price 1h after prediction
        SELECT close INTO price_1h
        FROM candles_1m
        WHERE time >= rec.analysis_time + INTERVAL '59 minutes'
          AND time < rec.analysis_time + INTERVAL '61 minutes'
        ORDER BY time
        LIMIT 1;
        
        -- Get actual price 4h after prediction
        SELECT close INTO price_4h
        FROM candles_1m
        WHERE time >= rec.analysis_time + INTERVAL '239 minutes'
          AND time < rec.analysis_time + INTERVAL '241 minutes'
        ORDER BY time
        LIMIT 1;
        
        -- Update if we have data
        IF price_1h IS NOT NULL THEN
            UPDATE llm_analysis
            SET 
                actual_price_1h = price_1h,
                actual_price_4h = price_4h,
                direction_correct_1h = CASE 
                    WHEN rec.prediction_direction = 'BULLISH' AND price_1h > rec.price THEN TRUE
                    WHEN rec.prediction_direction = 'BEARISH' AND price_1h < rec.price THEN TRUE
                    WHEN rec.prediction_direction = 'NEUTRAL' AND ABS(price_1h - rec.price) / rec.price < 0.002 THEN TRUE
                    ELSE FALSE
                END,
                direction_correct_4h = CASE
                    WHEN price_4h IS NULL THEN NULL
                    WHEN rec.prediction_direction = 'BULLISH' AND price_4h > rec.price THEN TRUE
                    WHEN rec.prediction_direction = 'BEARISH' AND price_4h < rec.price THEN TRUE
                    WHEN rec.prediction_direction = 'NEUTRAL' AND ABS(price_4h - rec.price) / rec.price < 0.005 THEN TRUE
                    ELSE FALSE
                END
            WHERE id = rec.id AND analysis_time = rec.analysis_time;
            
            updated_count := updated_count + 1;
        END IF;
    END LOOP;
    
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- STEP 6: Print migration results
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Migration 002: LLM Analyst Enhanced Schema';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Added columns:';
    RAISE NOTICE '  • invalidation_level (price level)';
    RAISE NOTICE '  • critical_support (price level)';
    RAISE NOTICE '  • critical_resistance (price level)';
    RAISE NOTICE '  • market_context (JSONB - full context)';
    RAISE NOTICE '  • signal_factors_used (JSONB - weighted factors)';
    RAISE NOTICE '  • smc_bias_at_analysis (TEXT)';
    RAISE NOTICE '  • trends_at_analysis (JSONB)';
    RAISE NOTICE '  • warnings_at_analysis (JSONB)';
    RAISE NOTICE '';
    RAISE NOTICE 'Created:';
    RAISE NOTICE '  • 4 new indexes for query performance';
    RAISE NOTICE '  • 3 new views (enhanced predictions, accuracy by conditions, agreement)';
    RAISE NOTICE '  • 2 helper functions for analysis';
    RAISE NOTICE '';
    RAISE NOTICE 'Purpose:';
    RAISE NOTICE '  Store full market context with each LLM prediction';
    RAISE NOTICE '  to enable analysis of what conditions lead to';
    RAISE NOTICE '  accurate vs inaccurate predictions.';
    RAISE NOTICE '';
    RAISE NOTICE 'Status: ✅ Migration 002 Complete';
    RAISE NOTICE '============================================';
END $$;
