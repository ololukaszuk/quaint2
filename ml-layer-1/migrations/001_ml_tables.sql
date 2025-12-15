-- ============================================================================
-- ML Layer 1: Additional Tables for ML Inference
-- Run this migration after init.sql if not already present
-- ============================================================================

-- ============================================================================
-- TABLE: model_versions
-- Tracks all trained model versions and their metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    
    -- Model identification
    model_type VARCHAR(20) NOT NULL,  -- 'mamba' or 'lgbm'
    version INTEGER NOT NULL,
    
    -- Storage location
    model_path TEXT NOT NULL,
    
    -- Performance metrics by horizon
    accuracy_1m NUMERIC(5,4),
    accuracy_2m NUMERIC(5,4),
    accuracy_3m NUMERIC(5,4),
    accuracy_4m NUMERIC(5,4),
    accuracy_5m NUMERIC(5,4),
    
    -- Aggregate metrics
    rmse NUMERIC(20,8),
    mae NUMERIC(20,8),
    
    -- Metadata
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    training_samples BIGINT,
    training_duration_seconds INTEGER,
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Constraints
    CONSTRAINT model_versions_unique_type_version UNIQUE (model_type, version),
    CONSTRAINT model_versions_type_check CHECK (model_type IN ('mamba', 'lgbm'))
);

-- Index for finding active models
CREATE INDEX IF NOT EXISTS idx_model_versions_active 
    ON model_versions (model_type, is_active) 
    WHERE is_active = TRUE;

-- Index for version lookups
CREATE INDEX IF NOT EXISTS idx_model_versions_type_version 
    ON model_versions (model_type, version DESC);

-- ============================================================================
-- TABLE: active_ensembles
-- Tracks ensemble configurations for A/B testing
-- ============================================================================
CREATE TABLE IF NOT EXISTS active_ensembles (
    id SERIAL PRIMARY KEY,
    
    -- Component model versions
    mamba_version_id INTEGER REFERENCES model_versions(id),
    lgbm_version_id INTEGER REFERENCES model_versions(id),
    
    -- Ensemble weights
    mamba_weight NUMERIC(5,4) NOT NULL DEFAULT 0.5,
    
    -- Performance metrics
    accuracy_1m NUMERIC(5,4),
    accuracy_5m NUMERIC(5,4),
    rmse NUMERIC(20,8),
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    is_test BOOLEAN NOT NULL DEFAULT FALSE,  -- For A/B testing
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT active_ensembles_weight_check CHECK (
        mamba_weight >= 0 AND mamba_weight <= 1
    )
);

-- Index for finding active ensemble
CREATE INDEX IF NOT EXISTS idx_active_ensembles_active 
    ON active_ensembles (is_active) 
    WHERE is_active = TRUE;

-- ============================================================================
-- FUNCTION: Ensure only one active ensemble
-- ============================================================================
CREATE OR REPLACE FUNCTION ensure_single_active_ensemble_ml()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_active = TRUE THEN
        -- Deactivate all other ensembles
        UPDATE active_ensembles
        SET is_active = FALSE
        WHERE id != NEW.id AND is_active = TRUE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger (drop first if exists)
DROP TRIGGER IF EXISTS trg_single_active_ensemble_ml ON active_ensembles;
CREATE TRIGGER trg_single_active_ensemble_ml
    BEFORE INSERT OR UPDATE OF is_active ON active_ensembles
    FOR EACH ROW
    WHEN (NEW.is_active = TRUE)
    EXECUTE FUNCTION ensure_single_active_ensemble_ml();

-- ============================================================================
-- Add unique constraint to predictions for upsert
-- ============================================================================
DO $$
BEGIN
    -- Add unique constraint if not exists
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'predictions_unique_key'
    ) THEN
        -- Drop existing partial index if any
        DROP INDEX IF EXISTS idx_predictions_unique;
        
        -- Add columns if not exist
        ALTER TABLE predictions 
            ADD COLUMN IF NOT EXISTS model_name TEXT,
            ADD COLUMN IF NOT EXISTS horizon INTEGER;
            
        -- Create unique index for upsert
        CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unique 
            ON predictions (prediction_time, model_name, horizon);
    END IF;
END $$;

-- ============================================================================
-- VIEW: v_ml_model_performance
-- ============================================================================
CREATE OR REPLACE VIEW v_ml_model_performance AS
SELECT 
    mv.model_type,
    mv.version,
    mv.accuracy_1m,
    mv.accuracy_5m,
    mv.rmse,
    mv.trained_at,
    mv.is_active,
    ae.id as ensemble_id,
    ae.mamba_weight,
    ae.is_test as ensemble_is_test
FROM model_versions mv
LEFT JOIN active_ensembles ae ON 
    (mv.model_type = 'mamba' AND ae.mamba_version_id = mv.id) OR
    (mv.model_type = 'lgbm' AND ae.lgbm_version_id = mv.id)
ORDER BY mv.trained_at DESC;

-- ============================================================================
-- VIEW: v_prediction_accuracy_24h
-- ============================================================================
CREATE OR REPLACE VIEW v_prediction_accuracy_24h AS
SELECT 
    model_name,
    horizon,
    COUNT(*) as total_predictions,
    COUNT(accuracy) as evaluated,
    AVG(CASE WHEN accuracy IS NOT NULL THEN accuracy ELSE NULL END) as avg_accuracy,
    AVG(rmse) as avg_rmse,
    AVG(mape) as avg_mape
FROM predictions
WHERE prediction_time > NOW() - INTERVAL '24 hours'
GROUP BY model_name, horizon
ORDER BY model_name, horizon;

-- ============================================================================
-- Grant permissions
-- ============================================================================
GRANT ALL PRIVILEGES ON model_versions TO mltrader;
GRANT ALL PRIVILEGES ON active_ensembles TO mltrader;
GRANT ALL PRIVILEGES ON SEQUENCE model_versions_id_seq TO mltrader;
GRANT ALL PRIVILEGES ON SEQUENCE active_ensembles_id_seq TO mltrader;

-- ============================================================================
-- Summary
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'ML Layer 1 Schema Migration Complete';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Tables created: model_versions, active_ensembles';
    RAISE NOTICE 'Views created: v_ml_model_performance, v_prediction_accuracy_24h';
    RAISE NOTICE 'Triggers: Single active ensemble enforcement';
    RAISE NOTICE '============================================';
END $$;
