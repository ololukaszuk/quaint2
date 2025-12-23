-- ============================================================================
-- Migration 003: LLM Request Queue
-- 
-- Creates coordination table between market-analyzer and llm-analyst.
-- Market-analyzer writes requests, llm-analyst processes them.
-- ============================================================================

-- ============================================================================
-- TABLE: llm_requests
-- Queue for LLM analysis requests from market-analyzer
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_requests (
    id                  BIGSERIAL PRIMARY KEY,
    request_time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Market context at request time
    analysis_id         BIGINT,  -- FK to market_analysis.id (optional)
    price               NUMERIC(20,8) NOT NULL,
    signal_type         TEXT,
    signal_direction    TEXT,
    
    -- Why LLM was requested
    trigger_reason      TEXT NOT NULL,  
    -- Possible values:
    --   'signal_change' - Signal type or direction changed
    --   'key_level_pivot' - Price near major pivot point
    --   'timeframe_conflict' - Conflicting trends across timeframes
    --   'testing_support' - Price testing major support
    --   'testing_resistance' - Price testing major resistance
    --   'high_volatility' - Unusual volatility detected
    
    -- Processing status
    status              TEXT NOT NULL DEFAULT 'pending',  
    -- Values: 'pending', 'processing', 'completed', 'failed'
    
    processed_at        TIMESTAMPTZ,
    llm_analysis_id     BIGINT,  -- FK to llm_analysis.id when completed
    error_message       TEXT,    -- Error details if failed
    
    -- Metadata
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for finding pending requests (main query pattern)
CREATE INDEX idx_llm_requests_pending 
    ON llm_requests (request_time ASC) 
    WHERE status = 'pending';

-- Index for tracking processing status
CREATE INDEX idx_llm_requests_status 
    ON llm_requests (status, request_time DESC);

-- Index for analyzing trigger patterns
CREATE INDEX idx_llm_requests_trigger 
    ON llm_requests (trigger_reason, request_time DESC);

-- Comments
COMMENT ON TABLE llm_requests IS 'Queue for coordinating LLM analysis requests from market-analyzer';
COMMENT ON COLUMN llm_requests.trigger_reason IS 'Why market-analyzer requested LLM commentary';
COMMENT ON COLUMN llm_requests.status IS 'pending: waiting, processing: in progress, completed: done, failed: error';
COMMENT ON COLUMN llm_requests.analysis_id IS 'Optional reference to market_analysis row that triggered this request';

-- ============================================================================
-- VIEW: Pending LLM Requests Summary
-- ============================================================================

CREATE OR REPLACE VIEW v_llm_requests_pending AS
SELECT 
    id,
    request_time,
    price,
    trigger_reason,
    EXTRACT(EPOCH FROM (NOW() - request_time)) as wait_seconds
FROM llm_requests
WHERE status = 'pending'
ORDER BY request_time ASC;

COMMENT ON VIEW v_llm_requests_pending IS 'Quick view of pending LLM analysis requests with wait time';

-- ============================================================================
-- VIEW: LLM Request Statistics
-- ============================================================================

CREATE OR REPLACE VIEW v_llm_request_stats AS
SELECT 
    trigger_reason,
    COUNT(*) as total_requests,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    AVG(EXTRACT(EPOCH FROM (processed_at - request_time))) FILTER (WHERE status = 'completed') as avg_processing_seconds,
    MAX(request_time) as last_request_time
FROM llm_requests
WHERE request_time > NOW() - INTERVAL '7 days'
GROUP BY trigger_reason
ORDER BY total_requests DESC;

COMMENT ON VIEW v_llm_request_stats IS 'Statistics on LLM request triggers and processing over last 7 days';

-- ============================================================================
-- FUNCTION: Clean old completed/failed requests
-- ============================================================================

CREATE OR REPLACE FUNCTION cleanup_old_llm_requests()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Delete completed/failed requests older than 7 days
    DELETE FROM llm_requests
    WHERE status IN ('completed', 'failed')
      AND request_time < NOW() - INTERVAL '7 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_llm_requests IS 'Removes old completed/failed LLM requests (keep 7 days)';

-- ============================================================================
-- Done
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'LLM Request Queue Created Successfully';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Table: llm_requests';
    RAISE NOTICE 'Views: v_llm_requests_pending, v_llm_request_stats';
    RAISE NOTICE 'Function: cleanup_old_llm_requests()';
    RAISE NOTICE '';
    RAISE NOTICE 'Market-analyzer will queue requests here';
    RAISE NOTICE 'LLM-analyst will process pending requests';
    RAISE NOTICE '============================================';
END $$;
