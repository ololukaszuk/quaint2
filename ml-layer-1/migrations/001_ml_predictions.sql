-- ML Layer 1 - Predictions Tables
-- Stores real-time ML predictions from inference service

-- Predictions table (hypertable)
CREATE TABLE IF NOT EXISTS ml_predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
    interval TEXT NOT NULL DEFAULT '1m',
    
    -- Current market state
    current_price DECIMAL(20, 8) NOT NULL,
    
    -- Multi-horizon predictions (1, 5, 15 min)
    predicted_1min DECIMAL(20, 8),
    predicted_5min DECIMAL(20, 8),
    predicted_15min DECIMAL(20, 8),
    
    -- Confidence scores
    confidence_1min REAL,
    confidence_5min REAL,
    confidence_15min REAL,
    
    -- Model metadata
    model_version TEXT,
    
    -- Evaluation (filled later when actual prices are known)
    actual_1min DECIMAL(20, 8),
    actual_5min DECIMAL(20, 8),
    actual_15min DECIMAL(20, 8),
    
    -- Errors
    error_1min REAL,
    error_5min REAL,
    error_15min REAL,
    
    -- Direction correct (boolean)
    direction_correct_1min BOOLEAN,
    direction_correct_5min BOOLEAN,
    direction_correct_15min BOOLEAN,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('ml_predictions', 'time', if_not_exists => TRUE);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_time 
    ON ml_predictions (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_created_at 
    ON ml_predictions (created_at DESC);

-- Retention policy (keep 30 days)
SELECT add_retention_policy('ml_predictions', INTERVAL '30 days', if_not_exists => TRUE);

-- Continuous aggregate for hourly stats
CREATE MATERIALIZED VIEW IF NOT EXISTS ml_predictions_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol,
    COUNT(*) as prediction_count,
    AVG(current_price) as avg_price,
    AVG(predicted_1min) as avg_predicted_1min,
    AVG(predicted_5min) as avg_predicted_5min,
    AVG(predicted_15min) as avg_predicted_15min,
    -- Accuracy metrics (when available)
    AVG(CASE WHEN direction_correct_1min IS NOT NULL THEN 
        CASE WHEN direction_correct_1min THEN 1.0 ELSE 0.0 END 
    END) as accuracy_1min,
    AVG(CASE WHEN direction_correct_5min IS NOT NULL THEN 
        CASE WHEN direction_correct_5min THEN 1.0 ELSE 0.0 END 
    END) as accuracy_5min,
    AVG(CASE WHEN direction_correct_15min IS NOT NULL THEN 
        CASE WHEN direction_correct_15min THEN 1.0 ELSE 0.0 END 
    END) as accuracy_15min,
    AVG(error_1min) as avg_error_1min,
    AVG(error_5min) as avg_error_5min,
    AVG(error_15min) as avg_error_15min
FROM ml_predictions
GROUP BY bucket, symbol;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('ml_predictions_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Model versions table
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    model_path TEXT NOT NULL,
    normalization_params JSONB,
    training_metrics JSONB,
    deployed_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE,
    
    -- Training info
    trained_on_samples INT,
    training_duration_seconds REAL,
    test_rmse REAL,
    test_mape REAL
);

-- Ensure only one active model
CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_models_active 
    ON ml_models (is_active) 
    WHERE is_active = TRUE;

-- Model performance tracking
CREATE TABLE IF NOT EXISTS ml_model_performance (
    id SERIAL PRIMARY KEY,
    model_version TEXT NOT NULL,
    evaluation_time TIMESTAMPTZ DEFAULT NOW(),
    
    -- Metrics per horizon
    accuracy_1min REAL,
    accuracy_5min REAL,
    accuracy_15min REAL,
    
    mape_1min REAL,
    mape_5min REAL,
    mape_15min REAL,
    
    rmse_1min REAL,
    rmse_5min REAL,
    rmse_15min REAL,
    
    -- Sample size
    evaluated_samples INT,
    time_period_start TIMESTAMPTZ,
    time_period_end TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_model_performance_version 
    ON ml_model_performance (model_version, evaluation_time DESC);

COMMENT ON TABLE ml_predictions IS 'Real-time ML predictions from inference service';
COMMENT ON TABLE ml_models IS 'ML model versions and metadata';
COMMENT ON TABLE ml_model_performance IS 'Model accuracy tracking over time';
