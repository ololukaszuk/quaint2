-- ============================================================================
-- MARKET ANALYZER MIGRATION 003 - COMPLETE PIVOT POINTS
-- Add missing pivot columns: Camarilla R1/R2/S1/S2, Woodie, DeMark
-- Version: 3.0 - Complete pivot data for all methods
-- ============================================================================

-- ============================================================================
-- STEP 1: Add missing Camarilla pivots (R1, R2, S1, S2)
-- Camarilla has R1-R4 and S1-S4, but only R3/R4/S3/S4 were added previously
-- ============================================================================

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r1_camarilla NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r2_camarilla NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s1_camarilla NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s2_camarilla NUMERIC(20,8);

-- ============================================================================
-- STEP 2: Add Woodie pivots (complete set: P, R1-R3, S1-S3)
-- ============================================================================

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_woodie NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r1_woodie NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r2_woodie NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r3_woodie NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s1_woodie NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s2_woodie NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s3_woodie NUMERIC(20,8);

-- ============================================================================
-- STEP 3: Add DeMark pivots (P, R1, S1 - DeMark only uses one R/S level)
-- ============================================================================

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_demark NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_r1_demark NUMERIC(20,8);

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_s1_demark NUMERIC(20,8);

-- ============================================================================
-- STEP 4: Add Camarilla pivot point (the center pivot was missing)
-- ============================================================================

ALTER TABLE market_analysis
ADD COLUMN IF NOT EXISTS pivot_camarilla NUMERIC(20,8);

-- ============================================================================
-- STEP 5: Create view for complete pivot analysis
-- ============================================================================

DROP VIEW IF EXISTS v_complete_pivot_analysis CASCADE;

CREATE VIEW v_complete_pivot_analysis AS
SELECT
    analysis_time,
    price,
    price_vs_pivot,
    
    -- Traditional Pivots
    pivot_daily as traditional_p,
    pivot_r1_traditional, pivot_r2_traditional, pivot_r3_traditional,
    pivot_s1_traditional, pivot_s2_traditional, pivot_s3_traditional,
    
    -- Fibonacci Pivots
    pivot_r1_fibonacci, pivot_r2_fibonacci, pivot_r3_fibonacci,
    pivot_s1_fibonacci, pivot_s2_fibonacci, pivot_s3_fibonacci,
    
    -- Camarilla Pivots (complete)
    pivot_camarilla as camarilla_p,
    pivot_r1_camarilla, pivot_r2_camarilla, pivot_r3_camarilla, pivot_r4_camarilla,
    pivot_s1_camarilla, pivot_s2_camarilla, pivot_s3_camarilla, pivot_s4_camarilla,
    
    -- Woodie Pivots (complete)
    pivot_woodie as woodie_p,
    pivot_r1_woodie, pivot_r2_woodie, pivot_r3_woodie,
    pivot_s1_woodie, pivot_s2_woodie, pivot_s3_woodie,
    
    -- DeMark Pivots (complete)
    pivot_demark as demark_p,
    pivot_r1_demark,
    pivot_s1_demark,
    
    -- Confluence
    pivot_confluence_zones
    
FROM market_analysis
ORDER BY analysis_time DESC
LIMIT 100;

-- ============================================================================
-- STEP 6: Create function to get nearest pivots from all methods
-- ============================================================================

DROP FUNCTION IF EXISTS get_nearest_pivots(NUMERIC);

CREATE FUNCTION get_nearest_pivots(current_price NUMERIC)
RETURNS TABLE (
    method TEXT,
    pivot_type TEXT,
    price NUMERIC,
    distance_pct NUMERIC
) AS $$
DECLARE
    latest_analysis RECORD;
BEGIN
    -- Get the latest analysis
    SELECT * INTO latest_analysis
    FROM market_analysis
    ORDER BY analysis_time DESC
    LIMIT 1;
    
    IF latest_analysis IS NULL THEN
        RETURN;
    END IF;
    
    -- Return all pivots with distance from current price
    RETURN QUERY
    SELECT 
        m.method,
        m.pivot_type,
        m.price,
        ABS((m.price - current_price) / current_price * 100) as distance_pct
    FROM (
        VALUES 
            ('Traditional', 'P', latest_analysis.pivot_daily),
            ('Traditional', 'R1', latest_analysis.pivot_r1_traditional),
            ('Traditional', 'R2', latest_analysis.pivot_r2_traditional),
            ('Traditional', 'R3', latest_analysis.pivot_r3_traditional),
            ('Traditional', 'S1', latest_analysis.pivot_s1_traditional),
            ('Traditional', 'S2', latest_analysis.pivot_s2_traditional),
            ('Traditional', 'S3', latest_analysis.pivot_s3_traditional),
            ('Fibonacci', 'R1', latest_analysis.pivot_r1_fibonacci),
            ('Fibonacci', 'R2', latest_analysis.pivot_r2_fibonacci),
            ('Fibonacci', 'R3', latest_analysis.pivot_r3_fibonacci),
            ('Fibonacci', 'S1', latest_analysis.pivot_s1_fibonacci),
            ('Fibonacci', 'S2', latest_analysis.pivot_s2_fibonacci),
            ('Fibonacci', 'S3', latest_analysis.pivot_s3_fibonacci),
            ('Camarilla', 'P', latest_analysis.pivot_camarilla),
            ('Camarilla', 'R1', latest_analysis.pivot_r1_camarilla),
            ('Camarilla', 'R2', latest_analysis.pivot_r2_camarilla),
            ('Camarilla', 'R3', latest_analysis.pivot_r3_camarilla),
            ('Camarilla', 'R4', latest_analysis.pivot_r4_camarilla),
            ('Camarilla', 'S1', latest_analysis.pivot_s1_camarilla),
            ('Camarilla', 'S2', latest_analysis.pivot_s2_camarilla),
            ('Camarilla', 'S3', latest_analysis.pivot_s3_camarilla),
            ('Camarilla', 'S4', latest_analysis.pivot_s4_camarilla),
            ('Woodie', 'P', latest_analysis.pivot_woodie),
            ('Woodie', 'R1', latest_analysis.pivot_r1_woodie),
            ('Woodie', 'R2', latest_analysis.pivot_r2_woodie),
            ('Woodie', 'R3', latest_analysis.pivot_r3_woodie),
            ('Woodie', 'S1', latest_analysis.pivot_s1_woodie),
            ('Woodie', 'S2', latest_analysis.pivot_s2_woodie),
            ('Woodie', 'S3', latest_analysis.pivot_s3_woodie),
            ('DeMark', 'P', latest_analysis.pivot_demark),
            ('DeMark', 'R1', latest_analysis.pivot_r1_demark),
            ('DeMark', 'S1', latest_analysis.pivot_s1_demark)
    ) AS m(method, pivot_type, price)
    WHERE m.price IS NOT NULL
    ORDER BY distance_pct ASC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- STEP 7: Print migration results
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Migration 003: Complete Pivots Applied';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Added Camarilla columns:';
    RAISE NOTICE '  • pivot_r1_camarilla, pivot_r2_camarilla';
    RAISE NOTICE '  • pivot_s1_camarilla, pivot_s2_camarilla';
    RAISE NOTICE '  • pivot_camarilla (center)';
    RAISE NOTICE '';
    RAISE NOTICE 'Added Woodie columns:';
    RAISE NOTICE '  • pivot_woodie (center)';
    RAISE NOTICE '  • pivot_r1_woodie, pivot_r2_woodie, pivot_r3_woodie';
    RAISE NOTICE '  • pivot_s1_woodie, pivot_s2_woodie, pivot_s3_woodie';
    RAISE NOTICE '';
    RAISE NOTICE 'Added DeMark columns:';
    RAISE NOTICE '  • pivot_demark (center)';
    RAISE NOTICE '  • pivot_r1_demark, pivot_s1_demark';
    RAISE NOTICE '';
    RAISE NOTICE 'Created:';
    RAISE NOTICE '  • v_complete_pivot_analysis view';
    RAISE NOTICE '  • get_nearest_pivots(price) function';
    RAISE NOTICE '';
    RAISE NOTICE 'Status: ✅ Migration 003 Complete';
    RAISE NOTICE '============================================';
END $$;
