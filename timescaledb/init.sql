-- ============================================================================
-- BTC ML PRODUCTION DATABASE SCHEMA
-- TimescaleDB Schema for Cryptocurrency Price Prediction System
-- ============================================================================
-- Database: btc_ml_production
-- User: mltrader
-- TimescaleDB Extension Required
-- ============================================================================

-- ============================================================================
-- SECTION 1: DATABASE SETUP & EXTENSIONS
-- ============================================================================

-- Connect to postgres first to create the database
-- \c postgres

-- Create database (run as superuser)
-- CREATE DATABASE btc_ml_production OWNER mltrader;

-- Connect to the new database
-- \c btc_ml_production

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- pg_cron is optional - requires shared_preload_libraries configuration
-- Will be created if available, otherwise skipped silently
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_cron;
    RAISE NOTICE 'pg_cron extension enabled';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_cron extension not available: % - continuing without scheduled jobs', SQLERRM;
END $$;

-- ============================================================================
-- SECTION 2: CUSTOM TYPES & DOMAINS
-- ============================================================================

-- Domain for price values with appropriate precision
CREATE DOMAIN price_type AS NUMERIC(20,8)
    CHECK (VALUE >= 0);

-- Domain for ratio values (0.0 to 1.0)
CREATE DOMAIN ratio_type AS NUMERIC(5,4)
    CHECK (VALUE >= 0 AND VALUE <= 1);

-- Domain for basis points (spread)
CREATE DOMAIN bps_type AS NUMERIC(10,4)
    CHECK (VALUE >= 0);

-- ============================================================================
-- SECTION 3: TABLE - candles_1m (Primary Hypertable)
-- ============================================================================
-- Stores 1-minute OHLCV candlestick data from exchange
-- This is the core time-series table for all price data
-- ============================================================================

CREATE TABLE candles_1m (
    -- Primary timestamp (indexed, partitioned by TimescaleDB)
    time                            TIMESTAMPTZ NOT NULL,
    
    -- OHLCV core data
    open                            NUMERIC(20,8) NOT NULL,
    high                            NUMERIC(20,8) NOT NULL,
    low                             NUMERIC(20,8) NOT NULL,
    close                           NUMERIC(20,8) NOT NULL,
    volume                          NUMERIC(20,8) NOT NULL,
    
    -- Additional exchange-provided metrics
    quote_asset_volume              NUMERIC(20,8),
    taker_buy_base_asset_volume     NUMERIC(20,8),
    taker_buy_quote_asset_volume    NUMERIC(20,8),
    number_of_trades                BIGINT,
    
    -- Computed/derived columns (populated by trigger)
    spread_bps                      NUMERIC(10,4),      -- (high-low)/close * 10000
    taker_buy_ratio                 NUMERIC(5,4),       -- taker_buy_base/volume
    mid_price                       NUMERIC(20,8),      -- (high+low)/2
    
    -- Constraints ensuring OHLC data integrity
    CONSTRAINT candles_1m_pkey PRIMARY KEY (time),
    CONSTRAINT candles_1m_high_check CHECK (high >= low AND high >= open AND high >= close),
    CONSTRAINT candles_1m_low_check CHECK (low <= open AND low <= close),
    CONSTRAINT candles_1m_volume_positive CHECK (volume >= 0),
    CONSTRAINT candles_1m_prices_positive CHECK (
        open > 0 AND high > 0 AND low > 0 AND close > 0
    )
);

-- Convert to TimescaleDB hypertable with monthly partitions
-- chunk_time_interval = 1 month for optimal query performance
SELECT create_hypertable(
    'candles_1m',
    'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Index for recent data queries (most common access pattern)
CREATE INDEX idx_candles_1m_time_desc ON candles_1m (time DESC);

-- Index for volatility filtering (spread-based queries)
CREATE INDEX idx_candles_1m_spread_bps ON candles_1m (spread_bps DESC)
    WHERE spread_bps IS NOT NULL;

-- Composite index for range queries with spread filtering
CREATE INDEX idx_candles_1m_time_spread ON candles_1m (time DESC, spread_bps DESC);

-- ============================================================================
-- SECTION 4: TABLE - predictions (Model Outputs & Accuracy)
-- ============================================================================
-- Stores all model predictions and their accuracy metrics
-- Used for model performance tracking and backtesting
-- ============================================================================

CREATE TABLE predictions (
    id                  BIGSERIAL NOT NULL,
    
    -- Model identification
    model_name          TEXT NOT NULL,  -- mamba, lgb, tft, gru, emd_lstm
    
    -- Timing information
    prediction_time     TIMESTAMPTZ NOT NULL,   -- When the prediction was made
    target_time         TIMESTAMPTZ NOT NULL,   -- What time the prediction is for
    horizon             INTEGER NOT NULL,        -- Minutes ahead: 1, 5, 15, 30, 60, 240, 1440
    
    -- Prediction values
    predicted_close     NUMERIC(20,8) NOT NULL,
    predicted_direction SMALLINT,               -- -1 (down), 0 (neutral), 1 (up)
    confidence          NUMERIC(5,4),           -- 0.0 to 1.0
    
    -- Actual values (populated later when target_time arrives)
    actual_close        NUMERIC(20,8),          -- Nullable until verified
    
    -- Accuracy metrics (computed after actual_close is known)
    accuracy            NUMERIC(5,4),           -- Direction accuracy (0 or 1)
    rmse                NUMERIC(20,8),          -- Root Mean Square Error
    mape                NUMERIC(5,4),           -- Mean Absolute Percentage Error
    
    -- Metadata
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Primary key must include partition column for hypertables
    PRIMARY KEY (id, prediction_time),
    
    -- Constraints
    CONSTRAINT predictions_horizon_check CHECK (
        horizon IN (1, 5, 15, 30, 60, 240, 1440)
    ),
    CONSTRAINT predictions_direction_check CHECK (
        predicted_direction IN (-1, 0, 1)
    ),
    CONSTRAINT predictions_confidence_check CHECK (
        confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
    ),
    CONSTRAINT predictions_accuracy_check CHECK (
        accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)
    ),
    CONSTRAINT predictions_target_after_prediction CHECK (
        target_time > prediction_time
    )
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable(
    'predictions',
    'prediction_time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Primary query pattern: filter by model, horizon, recent predictions
CREATE INDEX idx_predictions_model_horizon_time 
    ON predictions (model_name, horizon, prediction_time DESC);

-- For accuracy computation: find predictions by target_time
CREATE INDEX idx_predictions_target_time 
    ON predictions (target_time);

-- For finding predictions that need accuracy updates
CREATE INDEX idx_predictions_pending_accuracy 
    ON predictions (target_time)
    WHERE actual_close IS NULL;

-- ============================================================================
-- SECTION 5: TABLE - ensemble_models (A/B Testing & Model Tracking)
-- ============================================================================
-- Tracks ensemble model configurations and their performance
-- Supports A/B testing between different model combinations
-- ============================================================================

CREATE TABLE ensemble_models (
    id              BIGSERIAL PRIMARY KEY,
    
    -- Ensemble identification
    ensemble_name   TEXT NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    
    -- Model composition
    models          TEXT[] NOT NULL,        -- Array of model names
    weights         NUMERIC[] NOT NULL,     -- Weighting for each model
    
    -- Performance metrics by horizon
    accuracy_1m     NUMERIC(5,4),           -- 1-minute accuracy
    accuracy_5m     NUMERIC(5,4),           -- 5-minute accuracy
    accuracy_15m    NUMERIC(5,4),           -- 15-minute accuracy
    accuracy_1h     NUMERIC(5,4),           -- 1-hour accuracy
    
    -- Metadata
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active       BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Constraints
    CONSTRAINT ensemble_models_unique_name_version UNIQUE (ensemble_name, version),
    CONSTRAINT ensemble_models_weights_match CHECK (
        array_length(models, 1) = array_length(weights, 1)
    ),
    CONSTRAINT ensemble_models_accuracy_range CHECK (
        (accuracy_1m IS NULL OR (accuracy_1m >= 0 AND accuracy_1m <= 1)) AND
        (accuracy_5m IS NULL OR (accuracy_5m >= 0 AND accuracy_5m <= 1)) AND
        (accuracy_15m IS NULL OR (accuracy_15m >= 0 AND accuracy_15m <= 1)) AND
        (accuracy_1h IS NULL OR (accuracy_1h >= 0 AND accuracy_1h <= 1))
    )
);

-- Index for finding active ensemble(s)
CREATE INDEX idx_ensemble_models_active ON ensemble_models (is_active)
    WHERE is_active = TRUE;

-- Index for version lookups
CREATE INDEX idx_ensemble_models_name_version 
    ON ensemble_models (ensemble_name, version DESC);

-- ============================================================================
-- SECTION 6: TABLE - feature_cache (Pre-computed Features)
-- ============================================================================
-- Stores pre-computed normalized features for ML models
-- Reduces computation time during inference
-- ============================================================================

CREATE TABLE feature_cache (
    time            TIMESTAMPTZ NOT NULL PRIMARY KEY,
    
    -- Feature storage as JSON for flexibility
    features_json   JSONB NOT NULL,
    
    -- Validity flag (set to FALSE if underlying data changes)
    is_valid        BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Metadata
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable for efficient time-range queries
SELECT create_hypertable(
    'feature_cache',
    'time',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- Index for finding invalid features that need recomputation
CREATE INDEX idx_feature_cache_invalid 
    ON feature_cache (time DESC)
    WHERE is_valid = FALSE;

-- GIN index for JSONB queries if needed
CREATE INDEX idx_feature_cache_features_gin 
    ON feature_cache USING GIN (features_json);

-- ============================================================================
-- SECTION 7: TABLE - data_quality_logs (Gap Detection & Integrity)
-- ============================================================================
-- Logs data quality events including gaps, duplicates, and errors
-- Essential for monitoring data pipeline health
-- ============================================================================

CREATE TABLE data_quality_logs (
    id                  BIGSERIAL PRIMARY KEY,
    
    -- Event classification
    event_type          TEXT NOT NULL,          -- gap_detected, gap_backfilled, duplicate_found, error
    
    -- Gap details
    gap_start           TIMESTAMPTZ,            -- Start of detected gap
    gap_end             TIMESTAMPTZ,            -- End of detected gap
    candles_missing     INTEGER,                -- Number of missing candles
    candles_recovered   INTEGER,                -- Number recovered during backfill
    
    -- Source information
    source              TEXT,                   -- websocket, rest_api
    error_message       TEXT,                   -- Detailed error message if applicable
    
    -- Resolution tracking
    resolved            BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at         TIMESTAMPTZ,
    
    -- Metadata
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT data_quality_logs_event_type_check CHECK (
        event_type IN ('gap_detected', 'gap_backfilled', 'duplicate_found', 'error')
    ),
    CONSTRAINT data_quality_logs_source_check CHECK (
        source IS NULL OR source IN ('data_feeder', 'gap_handler', 'websocket', 'rest_api', 'manual', 'system')
    ),
    CONSTRAINT data_quality_logs_gap_consistency CHECK (
        (event_type != 'gap_detected' AND event_type != 'gap_backfilled') OR
        (gap_start IS NOT NULL AND gap_end IS NOT NULL AND gap_start < gap_end)
    )
);

-- Index for finding unresolved issues
CREATE INDEX idx_data_quality_logs_unresolved 
    ON data_quality_logs (resolved, created_at DESC)
    WHERE resolved = FALSE;

-- Index for recent events
CREATE INDEX idx_data_quality_logs_recent 
    ON data_quality_logs (created_at DESC);

-- Index by event type for analysis
CREATE INDEX idx_data_quality_logs_event_type 
    ON data_quality_logs (event_type, created_at DESC);

-- ============================================================================
-- SECTION 8: FUNCTIONS - Computed Columns for candles_1m
-- ============================================================================

-- Function to compute derived columns for candles
CREATE OR REPLACE FUNCTION compute_candle_derived_columns()
RETURNS TRIGGER AS $$
BEGIN
    -- Compute spread in basis points: (high - low) / close * 10000
    IF NEW.close > 0 THEN
        NEW.spread_bps := ((NEW.high - NEW.low) / NEW.close) * 10000;
    ELSE
        NEW.spread_bps := NULL;
    END IF;
    
    -- Compute taker buy ratio: taker_buy_base / volume
    IF NEW.volume > 0 AND NEW.taker_buy_base_asset_volume IS NOT NULL THEN
        NEW.taker_buy_ratio := NEW.taker_buy_base_asset_volume / NEW.volume;
    ELSE
        NEW.taker_buy_ratio := NULL;
    END IF;
    
    -- Compute mid price: (high + low) / 2
    NEW.mid_price := (NEW.high + NEW.low) / 2;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-compute derived columns on INSERT
CREATE TRIGGER trg_compute_candle_derived
    BEFORE INSERT ON candles_1m
    FOR EACH ROW
    EXECUTE FUNCTION compute_candle_derived_columns();

-- Also trigger on UPDATE (in case of corrections)
CREATE TRIGGER trg_compute_candle_derived_update
    BEFORE UPDATE ON candles_1m
    FOR EACH ROW
    WHEN (
        OLD.high IS DISTINCT FROM NEW.high OR
        OLD.low IS DISTINCT FROM NEW.low OR
        OLD.close IS DISTINCT FROM NEW.close OR
        OLD.volume IS DISTINCT FROM NEW.volume OR
        OLD.taker_buy_base_asset_volume IS DISTINCT FROM NEW.taker_buy_base_asset_volume
    )
    EXECUTE FUNCTION compute_candle_derived_columns();

-- ============================================================================
-- SECTION 9: FUNCTIONS - Gap Detection
-- ============================================================================

-- Function to detect gaps in candle data
CREATE OR REPLACE FUNCTION detect_candle_gap()
RETURNS TRIGGER AS $$
DECLARE
    prev_candle_time TIMESTAMPTZ;
    expected_prev_time TIMESTAMPTZ;
    gap_minutes INTEGER;
BEGIN
    -- Calculate the expected previous candle time (1 minute before)
    expected_prev_time := NEW.time - INTERVAL '1 minute';
    
    -- Look for the previous candle
    SELECT time INTO prev_candle_time
    FROM candles_1m
    WHERE time = expected_prev_time;
    
    -- If previous candle doesn't exist, check if there's a gap
    IF prev_candle_time IS NULL THEN
        -- Find the most recent candle before this one
        SELECT time INTO prev_candle_time
        FROM candles_1m
        WHERE time < NEW.time
        ORDER BY time DESC
        LIMIT 1;
        
        -- If we found a previous candle and there's a gap > 1 minute
        IF prev_candle_time IS NOT NULL THEN
            gap_minutes := EXTRACT(EPOCH FROM (NEW.time - prev_candle_time)) / 60;
            
            -- Only log if gap is significant (> 1 minute)
            -- Allow for small tolerance to avoid noise from timing issues
            IF gap_minutes > 1 THEN
                INSERT INTO data_quality_logs (
                    event_type,
                    gap_start,
                    gap_end,
                    candles_missing,
                    source,
                    resolved
                ) VALUES (
                    'gap_detected',
                    prev_candle_time + INTERVAL '1 minute',
                    NEW.time,
                    gap_minutes - 1,
                    'system',
                    FALSE
                );
            END IF;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for gap detection (runs after successful insert)
CREATE TRIGGER trg_detect_candle_gap
    AFTER INSERT ON candles_1m
    FOR EACH ROW
    EXECUTE FUNCTION detect_candle_gap();

-- ============================================================================
-- SECTION 10: FUNCTIONS - Upsert for Concurrent Safety
-- ============================================================================

-- Function to safely insert/update candles (handles duplicates)
CREATE OR REPLACE FUNCTION upsert_candle(
    p_time TIMESTAMPTZ,
    p_open NUMERIC(20,8),
    p_high NUMERIC(20,8),
    p_low NUMERIC(20,8),
    p_close NUMERIC(20,8),
    p_volume NUMERIC(20,8),
    p_quote_asset_volume NUMERIC(20,8) DEFAULT NULL,
    p_taker_buy_base_asset_volume NUMERIC(20,8) DEFAULT NULL,
    p_taker_buy_quote_asset_volume NUMERIC(20,8) DEFAULT NULL,
    p_number_of_trades BIGINT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO candles_1m (
        time, open, high, low, close, volume,
        quote_asset_volume, taker_buy_base_asset_volume,
        taker_buy_quote_asset_volume, number_of_trades
    ) VALUES (
        p_time, p_open, p_high, p_low, p_close, p_volume,
        p_quote_asset_volume, p_taker_buy_base_asset_volume,
        p_taker_buy_quote_asset_volume, p_number_of_trades
    )
    ON CONFLICT (time) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        quote_asset_volume = EXCLUDED.quote_asset_volume,
        taker_buy_base_asset_volume = EXCLUDED.taker_buy_base_asset_volume,
        taker_buy_quote_asset_volume = EXCLUDED.taker_buy_quote_asset_volume,
        number_of_trades = EXCLUDED.number_of_trades
    WHERE
        -- Only update if values actually changed (avoids unnecessary writes)
        candles_1m.open IS DISTINCT FROM EXCLUDED.open OR
        candles_1m.high IS DISTINCT FROM EXCLUDED.high OR
        candles_1m.low IS DISTINCT FROM EXCLUDED.low OR
        candles_1m.close IS DISTINCT FROM EXCLUDED.close OR
        candles_1m.volume IS DISTINCT FROM EXCLUDED.volume;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECTION 11: FUNCTIONS - Accuracy Computation
-- ============================================================================

-- Function to compute accuracy metrics for predictions
-- Runs daily to update predictions with actual values
CREATE OR REPLACE FUNCTION compute_accuracy()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER := 0;
    rec RECORD;
BEGIN
    -- Update predictions where target_time has passed and actual_close is not set
    FOR rec IN
        SELECT p.id, p.target_time, p.predicted_close, p.predicted_direction,
               c.close AS actual_close
        FROM predictions p
        JOIN candles_1m c ON c.time = p.target_time
        WHERE p.actual_close IS NULL
          AND p.target_time <= NOW() - INTERVAL '5 minutes'  -- Allow 5 min buffer
        ORDER BY p.target_time
        LIMIT 10000  -- Process in batches
    LOOP
        -- Calculate accuracy metrics
        UPDATE predictions
        SET
            actual_close = rec.actual_close,
            -- Direction accuracy: 1 if correct, 0 if wrong
            accuracy = CASE
                WHEN rec.predicted_direction = 1 AND rec.actual_close > rec.predicted_close THEN 1
                WHEN rec.predicted_direction = -1 AND rec.actual_close < rec.predicted_close THEN 1
                WHEN rec.predicted_direction = 0 AND ABS(rec.actual_close - rec.predicted_close) / rec.predicted_close < 0.001 THEN 1
                ELSE 0
            END,
            -- RMSE: sqrt((predicted - actual)^2)
            rmse = SQRT(POWER(rec.predicted_close - rec.actual_close, 2)),
            -- MAPE: |predicted - actual| / actual * 100
            mape = CASE
                WHEN rec.actual_close > 0 THEN
                    ABS(rec.predicted_close - rec.actual_close) / rec.actual_close
                ELSE NULL
            END
        WHERE id = rec.id;
        
        updated_count := updated_count + 1;
    END LOOP;
    
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECTION 12: FUNCTIONS - Data Cleanup
-- ============================================================================

-- Function to clean up old data safely
-- Runs daily to maintain data retention policies
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS TABLE (
    candles_deleted BIGINT,
    predictions_deleted BIGINT,
    feature_cache_deleted BIGINT,
    logs_deleted BIGINT
) AS $$
DECLARE
    v_candles_deleted BIGINT := 0;
    v_predictions_deleted BIGINT := 0;
    v_feature_cache_deleted BIGINT := 0;
    v_logs_deleted BIGINT := 0;
    candle_cutoff TIMESTAMPTZ;
    prediction_cutoff TIMESTAMPTZ;
    feature_cutoff TIMESTAMPTZ;
    log_cutoff TIMESTAMPTZ;
BEGIN
    -- Calculate cutoff dates
    candle_cutoff := NOW() - INTERVAL '12 months';
    prediction_cutoff := NOW() - INTERVAL '3 months';
    feature_cutoff := NOW() - INTERVAL '1 month';
    log_cutoff := NOW() - INTERVAL '6 months';  -- Keep logs longer for audit
    
    -- Delete old candles (use drop_chunks for efficiency with hypertables)
    -- This is much faster than DELETE for TimescaleDB
    SELECT drop_chunks('candles_1m', older_than => candle_cutoff)
    INTO v_candles_deleted;
    
    -- Delete old predictions
    WITH deleted AS (
        DELETE FROM predictions
        WHERE prediction_time < prediction_cutoff
        RETURNING 1
    )
    SELECT COUNT(*) INTO v_predictions_deleted FROM deleted;
    
    -- Delete old feature cache entries
    WITH deleted AS (
        DELETE FROM feature_cache
        WHERE time < feature_cutoff
        RETURNING 1
    )
    SELECT COUNT(*) INTO v_feature_cache_deleted FROM deleted;
    
    -- Delete old resolved logs (keep unresolved for investigation)
    WITH deleted AS (
        DELETE FROM data_quality_logs
        WHERE created_at < log_cutoff
          AND resolved = TRUE
        RETURNING 1
    )
    SELECT COUNT(*) INTO v_logs_deleted FROM deleted;
    
    -- Run vacuum on the tables (analyze statistics)
    -- Note: VACUUM cannot run inside a transaction block in a function
    -- This will be handled separately or by autovacuum
    
    RETURN QUERY SELECT v_candles_deleted, v_predictions_deleted, 
                        v_feature_cache_deleted, v_logs_deleted;
END;
$$ LANGUAGE plpgsql;

-- Wrapper to run cleanup and log results
CREATE OR REPLACE FUNCTION run_scheduled_cleanup()
RETURNS VOID AS $$
DECLARE
    result RECORD;
    cleanup_message TEXT;
BEGIN
    SELECT * INTO result FROM cleanup_old_data();
    
    -- Build proper message
    cleanup_message := format(
        'Scheduled cleanup completed: candles=%s, predictions=%s, features=%s, logs=%s',
        COALESCE(result.candles_deleted::TEXT, '0'),
        COALESCE(result.predictions_deleted::TEXT, '0'),
        COALESCE(result.feature_cache_deleted::TEXT, '0'),
        COALESCE(result.logs_deleted::TEXT, '0')
    );
    
    -- Log with proper event_type
    INSERT INTO data_quality_logs (
        event_type,
        source,
        error_message,
        resolved,
        resolved_at
    ) VALUES (
        'cleanup',  -- Changed from 'error'
        'system',
        cleanup_message,
        TRUE,
        NOW()
    );
    
    RAISE NOTICE '%', cleanup_message;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECTION 13: FUNCTIONS - Ensemble Model Management
-- ============================================================================

-- Function to update timestamp on ensemble model changes
CREATE OR REPLACE FUNCTION update_ensemble_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ensemble_updated
    BEFORE UPDATE ON ensemble_models
    FOR EACH ROW
    EXECUTE FUNCTION update_ensemble_timestamp();

-- Function to ensure only one active ensemble at a time
CREATE OR REPLACE FUNCTION ensure_single_active_ensemble()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_active = TRUE THEN
        -- Deactivate all other ensembles
        UPDATE ensemble_models
        SET is_active = FALSE, updated_at = NOW()
        WHERE id != NEW.id AND is_active = TRUE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_single_active_ensemble
    BEFORE INSERT OR UPDATE OF is_active ON ensemble_models
    FOR EACH ROW
    WHEN (NEW.is_active = TRUE)
    EXECUTE FUNCTION ensure_single_active_ensemble();

-- ============================================================================
-- SECTION 14: COMPRESSION POLICY (TimescaleDB)
-- ============================================================================

-- Enable compression on candles_1m hypertable
ALTER TABLE candles_1m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = '',
    timescaledb.compress_orderby = 'time DESC'
);

-- Add compression policy: compress chunks older than 1 week
SELECT add_compression_policy('candles_1m', INTERVAL '1 week');

-- Enable compression on predictions hypertable
ALTER TABLE predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'model_name',
    timescaledb.compress_orderby = 'prediction_time DESC'
);

-- Compress prediction chunks older than 1 month
SELECT add_compression_policy('predictions', INTERVAL '1 month');

-- Enable compression on feature_cache
ALTER TABLE feature_cache SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('feature_cache', INTERVAL '1 week');

-- ============================================================================
-- SECTION 15: RETENTION POLICIES (TimescaleDB)
-- ============================================================================

-- Add retention policy for candles_1m: drop chunks older than 13 months
SELECT add_retention_policy('candles_1m', INTERVAL '13 months');

-- Add retention policy for predictions: drop chunks older than 3 months
SELECT add_retention_policy('predictions', INTERVAL '3 months');

-- Add retention policy for feature_cache: drop chunks older than 1 month
SELECT add_retention_policy('feature_cache', INTERVAL '1 month');

-- ============================================================================
-- SECTION 16: SCHEDULED JOBS (pg_cron)
-- ============================================================================

-- Note: pg_cron jobs are optional - script continues if pg_cron is not available
-- pg_cron must be in shared_preload_libraries and extension created

DO $$
BEGIN
    -- Check if pg_cron extension exists
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Schedule cleanup job: daily at 02:00 UTC
        PERFORM cron.schedule(
            'cleanup_old_data',
            '0 2 * * *',
            'SELECT run_scheduled_cleanup()'
        );

        -- Schedule accuracy computation: daily at 03:00 UTC
        PERFORM cron.schedule(
            'compute_accuracy',
            '0 3 * * *',
            'SELECT compute_accuracy()'
        );

        -- Schedule vacuum analyze: weekly on Sunday at 04:00 UTC
        PERFORM cron.schedule(
            'vacuum_analyze_weekly',
            '0 4 * * 0',
            'VACUUM ANALYZE candles_1m; VACUUM ANALYZE predictions; VACUUM ANALYZE feature_cache'
        );

        RAISE NOTICE 'pg_cron scheduled jobs created successfully';
    ELSE
        RAISE NOTICE 'pg_cron extension not available - scheduled jobs skipped';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_cron scheduling failed: % - continuing without scheduled jobs', SQLERRM;
END $$;

-- ============================================================================
-- SECTION 17: USER PERMISSIONS
-- ============================================================================

-- Grant full access to mltrader user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mltrader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mltrader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mltrader;

-- Create read-only user (optional)
-- Uncomment to enable
/*
CREATE USER readonly_user WITH PASSWORD 'your_secure_password';
GRANT CONNECT ON DATABASE btc_ml_production TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user;
*/

-- ============================================================================
-- SECTION 18: HELPER VIEWS
-- ============================================================================

-- View: Recent model performance summary
CREATE OR REPLACE VIEW v_model_performance AS
SELECT
    model_name,
    horizon,
    COUNT(*) AS total_predictions,
    COUNT(accuracy) AS evaluated_predictions,
    AVG(accuracy) AS avg_accuracy,
    AVG(rmse) AS avg_rmse,
    AVG(mape) AS avg_mape,
    MAX(prediction_time) AS last_prediction
FROM predictions
WHERE prediction_time > NOW() - INTERVAL '7 days'
GROUP BY model_name, horizon
ORDER BY model_name, horizon;

-- View: Data quality summary
CREATE OR REPLACE VIEW v_data_quality_summary AS
SELECT
    event_type,
    COUNT(*) AS total_events,
    SUM(CASE WHEN resolved THEN 0 ELSE 1 END) AS unresolved_count,
    SUM(COALESCE(candles_missing, 0)) AS total_candles_missing,
    SUM(COALESCE(candles_recovered, 0)) AS total_candles_recovered,
    MAX(created_at) AS latest_event
FROM data_quality_logs
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY event_type
ORDER BY total_events DESC;

-- View: Active ensemble with performance
CREATE OR REPLACE VIEW v_active_ensemble AS
SELECT
    ensemble_name,
    version,
    models,
    weights,
    accuracy_1m,
    accuracy_5m,
    accuracy_15m,
    accuracy_1h,
    updated_at
FROM ensemble_models
WHERE is_active = TRUE;

-- View: Latest candle data with derived metrics
CREATE OR REPLACE VIEW v_latest_candles AS
SELECT
    time,
    open,
    high,
    low,
    close,
    volume,
    spread_bps,
    taker_buy_ratio,
    mid_price,
    number_of_trades
FROM candles_1m
ORDER BY time DESC
LIMIT 100;

-- ============================================================================
-- SECTION 19: UTILITY FUNCTIONS
-- ============================================================================

-- Function to get candles with gap filling (returns NULL for missing minutes)
CREATE OR REPLACE FUNCTION get_candles_with_gaps(
    p_start TIMESTAMPTZ,
    p_end TIMESTAMPTZ
)
RETURNS TABLE (
    "time" TIMESTAMPTZ,
    "open" NUMERIC(20,8),
    "high" NUMERIC(20,8),
    "low" NUMERIC(20,8),
    "close" NUMERIC(20,8),
    "volume" NUMERIC(20,8),
    is_gap BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH time_series AS (
        SELECT generate_series(p_start, p_end, INTERVAL '1 minute') AS ts
    )
    SELECT
        ts.ts AS "time",
        c.open,
        c.high,
        c.low,
        c.close,
        c.volume,
        (c.time IS NULL) AS is_gap
    FROM time_series ts
    LEFT JOIN candles_1m c ON c.time = ts.ts
    ORDER BY ts.ts;
END;
$$ LANGUAGE plpgsql;

-- Function to mark gap as backfilled
CREATE OR REPLACE FUNCTION mark_gap_resolved(
    p_gap_start TIMESTAMPTZ,
    p_gap_end TIMESTAMPTZ,
    p_candles_recovered INTEGER
)
RETURNS VOID AS $$
BEGIN
    UPDATE data_quality_logs
    SET
        resolved = TRUE,
        resolved_at = NOW(),
        candles_recovered = p_candles_recovered
    WHERE event_type = 'gap_detected'
      AND gap_start = p_gap_start
      AND gap_end = p_gap_end
      AND resolved = FALSE;
      
    -- Log the backfill event
    INSERT INTO data_quality_logs (
        event_type,
        gap_start,
        gap_end,
        candles_recovered,
        source,
        resolved,
        resolved_at
    ) VALUES (
        'gap_backfilled',
        p_gap_start,
        p_gap_end,
        p_candles_recovered,
        'system',
        TRUE,
        NOW()
    );
END;
$$ LANGUAGE plpgsql;

-- Function to bulk insert candles efficiently
CREATE OR REPLACE FUNCTION bulk_insert_candles(
    p_data JSONB  -- Array of candle objects
)
RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    candle JSONB;
BEGIN
    FOR candle IN SELECT * FROM jsonb_array_elements(p_data)
    LOOP
        PERFORM upsert_candle(
            (candle->>'time')::TIMESTAMPTZ,
            (candle->>'open')::NUMERIC(20,8),
            (candle->>'high')::NUMERIC(20,8),
            (candle->>'low')::NUMERIC(20,8),
            (candle->>'close')::NUMERIC(20,8),
            (candle->>'volume')::NUMERIC(20,8),
            (candle->>'quote_asset_volume')::NUMERIC(20,8),
            (candle->>'taker_buy_base_asset_volume')::NUMERIC(20,8),
            (candle->>'taker_buy_quote_asset_volume')::NUMERIC(20,8),
            (candle->>'number_of_trades')::BIGINT
        );
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECTION 20: INITIAL DATA VALIDATION
-- ============================================================================

-- Function to validate database setup
CREATE OR REPLACE FUNCTION validate_schema()
RETURNS TABLE (
    check_name TEXT,
    status TEXT,
    details TEXT
) AS $$
BEGIN
    -- Check TimescaleDB extension
    RETURN QUERY
    SELECT
        'TimescaleDB Extension'::TEXT,
        CASE WHEN EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')
             THEN 'OK'::TEXT ELSE 'MISSING'::TEXT END,
        ''::TEXT;
    
    -- Check hypertables
    RETURN QUERY
    SELECT
        'Hypertable: candles_1m'::TEXT,
        CASE WHEN EXISTS (
            SELECT 1 FROM timescaledb_information.hypertables 
            WHERE hypertable_name = 'candles_1m'
        ) THEN 'OK'::TEXT ELSE 'MISSING'::TEXT END,
        ''::TEXT;
    
    RETURN QUERY
    SELECT
        'Hypertable: predictions'::TEXT,
        CASE WHEN EXISTS (
            SELECT 1 FROM timescaledb_information.hypertables 
            WHERE hypertable_name = 'predictions'
        ) THEN 'OK'::TEXT ELSE 'MISSING'::TEXT END,
        ''::TEXT;
    
    -- Check compression policies
    RETURN QUERY
    SELECT
        'Compression Policy: candles_1m'::TEXT,
        CASE WHEN EXISTS (
            SELECT 1 FROM timescaledb_information.jobs 
            WHERE proc_name = 'policy_compression' 
              AND hypertable_name = 'candles_1m'
        ) THEN 'OK'::TEXT ELSE 'MISSING'::TEXT END,
        ''::TEXT;
    
    -- Check retention policies
    RETURN QUERY
    SELECT
        'Retention Policy: candles_1m'::TEXT,
        CASE WHEN EXISTS (
            SELECT 1 FROM timescaledb_information.jobs 
            WHERE proc_name = 'policy_retention' 
              AND hypertable_name = 'candles_1m'
        ) THEN 'OK'::TEXT ELSE 'MISSING'::TEXT END,
        ''::TEXT;
    
    -- Check pg_cron extension
    RETURN QUERY
    SELECT
        'pg_cron Extension'::TEXT,
        CASE WHEN EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron')
             THEN 'OK'::TEXT ELSE 'MISSING'::TEXT END,
        ''::TEXT;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- Run validation after setup
-- SELECT * FROM validate_schema();

-- Display summary
DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'BTC ML Production Schema Created Successfully';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Tables created: candles_1m, predictions, ensemble_models, feature_cache, data_quality_logs';
    RAISE NOTICE 'Hypertables: candles_1m (1 month chunks), predictions (1 week chunks), feature_cache (1 week chunks)';
    RAISE NOTICE 'Compression: Enabled for chunks > 1 week (candles) / 1 month (predictions)';
    RAISE NOTICE 'Retention: 13 months (candles), 3 months (predictions), 1 month (features)';
    RAISE NOTICE 'Scheduled jobs: cleanup (02:00 UTC), accuracy (03:00 UTC), vacuum (Sunday 04:00 UTC)';
    RAISE NOTICE '============================================';
END $$;
