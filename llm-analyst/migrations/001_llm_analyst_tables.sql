-- ============================================================================
-- LLM ANALYST TABLES
-- Stores AI-generated market analysis and predictions
-- ============================================================================

-- ============================================================================
-- TABLE: llm_analysis
-- Stores each LLM analysis with predictions and reasoning
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_analysis (
    id                      BIGSERIAL NOT NULL,
    analysis_time           TIMESTAMPTZ NOT NULL,
    
    -- Price at analysis time
    price                   NUMERIC(20,8) NOT NULL,
    
    -- Prediction
    prediction_direction    TEXT NOT NULL,  -- BULLISH, BEARISH, NEUTRAL
    prediction_confidence   TEXT NOT NULL,  -- HIGH, MEDIUM, LOW
    
    -- Price targets
    predicted_price_1h      NUMERIC(20,8),
    predicted_price_4h      NUMERIC(20,8),
    
    -- Key levels identified
    key_levels              TEXT,
    
    -- Reasoning summary
    reasoning               TEXT,
    
    -- Full LLM response
    full_response           TEXT,
    
    -- Model info
    model_name              TEXT,
    response_time_seconds   NUMERIC(8,2),
    
    -- Accuracy tracking (filled later when we know actual price)
    actual_price_1h         NUMERIC(20,8),
    actual_price_4h         NUMERIC(20,8),
    direction_correct_1h    BOOLEAN,
    direction_correct_4h    BOOLEAN,
    
    -- Metadata
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    PRIMARY KEY (id, analysis_time)
);

-- Convert to hypertable
SELECT create_hypertable(
    'llm_analysis',
    'analysis_time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Index for recent analyses
CREATE INDEX IF NOT EXISTS idx_llm_analysis_time_desc 
    ON llm_analysis (analysis_time DESC);

-- Index for direction queries
CREATE INDEX IF NOT EXISTS idx_llm_analysis_direction 
    ON llm_analysis (prediction_direction, analysis_time DESC);

-- Index for accuracy tracking (null actual prices)
CREATE INDEX IF NOT EXISTS idx_llm_analysis_pending_accuracy 
    ON llm_analysis (analysis_time DESC) 
    WHERE actual_price_1h IS NULL;

-- Retention policy - keep 3 months
SELECT add_retention_policy('llm_analysis', INTERVAL '3 months', if_not_exists => TRUE);

-- ============================================================================
-- VIEW: Recent LLM predictions
-- ============================================================================

CREATE OR REPLACE VIEW v_recent_llm_predictions AS
SELECT 
    analysis_time,
    price,
    prediction_direction,
    prediction_confidence,
    predicted_price_1h,
    predicted_price_4h,
    reasoning,
    model_name,
    response_time_seconds,
    direction_correct_1h,
    direction_correct_4h
FROM llm_analysis
ORDER BY analysis_time DESC
LIMIT 50;

-- ============================================================================
-- VIEW: LLM accuracy stats
-- ============================================================================

CREATE OR REPLACE VIEW v_llm_accuracy_stats AS
SELECT 
    model_name,
    COUNT(*) as total_predictions,
    COUNT(direction_correct_1h) as evaluated_1h,
    COUNT(direction_correct_4h) as evaluated_4h,
    AVG(CASE WHEN direction_correct_1h THEN 1.0 ELSE 0.0 END) as accuracy_1h,
    AVG(CASE WHEN direction_correct_4h THEN 1.0 ELSE 0.0 END) as accuracy_4h,
    AVG(response_time_seconds) as avg_response_time
FROM llm_analysis
WHERE analysis_time > NOW() - INTERVAL '7 days'
GROUP BY model_name;

-- ============================================================================
-- FUNCTION: Update LLM accuracy after price realized
-- Call this periodically to fill in actual prices and accuracy
-- ============================================================================

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
        WHERE open_time >= rec.analysis_time + INTERVAL '59 minutes'
          AND open_time < rec.analysis_time + INTERVAL '61 minutes'
        ORDER BY open_time
        LIMIT 1;
        
        -- Get actual price 4h after prediction
        SELECT close INTO price_4h
        FROM candles_1m
        WHERE open_time >= rec.analysis_time + INTERVAL '239 minutes'
          AND open_time < rec.analysis_time + INTERVAL '241 minutes'
        ORDER BY open_time
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
-- Schedule accuracy updates (runs hourly)
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.jobs 
        WHERE proc_name = 'update_llm_accuracy'
    ) THEN
        PERFORM add_job(
            'update_llm_accuracy'::REGPROC,
            INTERVAL '1 hour'
        );
    END IF;
END $$;

-- ============================================================================
-- Done
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'LLM Analyst Tables Created Successfully';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Tables: llm_analysis';
    RAISE NOTICE 'Views: v_recent_llm_predictions, v_llm_accuracy_stats';
    RAISE NOTICE 'Jobs: update_llm_accuracy (hourly)';
    RAISE NOTICE 'Retention: 3 months';
    RAISE NOTICE '============================================';
END $$;
