// Package main provides database operations for the gap handler service.
package main

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/lib/pq"
	"go.uber.org/zap"
)

// DatabaseBackfiller handles all database operations for backfilling.
type DatabaseBackfiller struct {
	db     *sql.DB
	logger *zap.Logger
}

// Candle represents a candle record for the database.
type Candle struct {
	Time                     time.Time
	Open                     float64
	High                     float64
	Low                      float64
	Close                    float64
	Volume                   float64
	QuoteAssetVolume         float64
	TakerBuyBaseAssetVolume  float64
	TakerBuyQuoteAssetVolume float64
	NumberOfTrades           int64
}

// NewDatabaseBackfiller creates a new database backfiller.
func NewDatabaseBackfiller(cfg *Config, logger *zap.Logger) (*DatabaseBackfiller, error) {
	db, err := sql.Open("postgres", cfg.DatabaseDSN())
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(cfg.DBMaxConns)
	db.SetMaxIdleConns(cfg.DBMaxConns)
	db.SetConnMaxLifetime(30 * time.Minute)
	db.SetConnMaxIdleTime(5 * time.Minute)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		metricsDatabaseStatus.Set(0)
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	metricsDatabaseStatus.Set(1)
	logger.Info("Database connection established")

	return &DatabaseBackfiller{
		db:     db,
		logger: logger,
	}, nil
}

// Close closes the database connection.
func (d *DatabaseBackfiller) Close() error {
	return d.db.Close()
}

// Ping tests the database connection.
func (d *DatabaseBackfiller) Ping(ctx context.Context) error {
	if err := d.db.PingContext(ctx); err != nil {
		metricsDatabaseStatus.Set(0)
		return err
	}
	metricsDatabaseStatus.Set(1)
	return nil
}

// GetExistingTimestamps returns timestamps that already exist in the database
// for the given time range.
func (d *DatabaseBackfiller) GetExistingTimestamps(ctx context.Context, startTime, endTime time.Time) (map[time.Time]bool, error) {
	start := time.Now()

	query := `
		SELECT time 
		FROM candles_1m 
		WHERE time >= $1 AND time < $2
	`

	rows, err := d.db.QueryContext(ctx, query, startTime, endTime)
	if err != nil {
		metricsDatabaseErrors.Inc()
		RecordDatabaseOperation(time.Since(start).Seconds())
		return nil, fmt.Errorf("query existing timestamps: %w", err)
	}
	defer rows.Close()

	existing := make(map[time.Time]bool)
	for rows.Next() {
		var t time.Time
		if err := rows.Scan(&t); err != nil {
			continue
		}
		existing[t.UTC()] = true
	}

	if err := rows.Err(); err != nil {
		metricsDatabaseErrors.Inc()
		return nil, fmt.Errorf("iterate rows: %w", err)
	}

	RecordDatabaseOperation(time.Since(start).Seconds())

	d.logger.Debug("Found existing timestamps",
		zap.Int("count", len(existing)),
		zap.Time("start", startTime),
		zap.Time("end", endTime),
	)

	return existing, nil
}

// InsertCandles inserts a batch of candles into the database.
// Returns the number of candles actually inserted (after deduplication).
func (d *DatabaseBackfiller) InsertCandles(ctx context.Context, candles []Candle) (int, error) {
	if len(candles) == 0 {
		return 0, nil
	}

	start := time.Now()

	// Start transaction
	tx, err := d.db.BeginTx(ctx, nil)
	if err != nil {
		metricsDatabaseErrors.Inc()
		return 0, fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Prepare statement for batch insert
	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO candles_1m (
			time, open, high, low, close, volume,
			quote_asset_volume, taker_buy_base_asset_volume,
			taker_buy_quote_asset_volume, number_of_trades
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
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
			candles_1m.open IS DISTINCT FROM EXCLUDED.open OR
			candles_1m.high IS DISTINCT FROM EXCLUDED.high OR
			candles_1m.low IS DISTINCT FROM EXCLUDED.low OR
			candles_1m.close IS DISTINCT FROM EXCLUDED.close OR
			candles_1m.volume IS DISTINCT FROM EXCLUDED.volume
	`)
	if err != nil {
		metricsDatabaseErrors.Inc()
		return 0, fmt.Errorf("prepare statement: %w", err)
	}
	defer stmt.Close()

	inserted := 0
	for _, candle := range candles {
		_, err := stmt.ExecContext(ctx,
			candle.Time,
			candle.Open,
			candle.High,
			candle.Low,
			candle.Close,
			candle.Volume,
			candle.QuoteAssetVolume,
			candle.TakerBuyBaseAssetVolume,
			candle.TakerBuyQuoteAssetVolume,
			candle.NumberOfTrades,
		)
		if err != nil {
			d.logger.Warn("Failed to insert candle",
				zap.Time("time", candle.Time),
				zap.Error(err),
			)
			continue
		}
		inserted++
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		metricsDatabaseErrors.Inc()
		return 0, fmt.Errorf("commit transaction: %w", err)
	}

	duration := time.Since(start)
	RecordDatabaseOperation(duration.Seconds())
	metricsCandlesInserted.Add(float64(inserted))

	d.logger.Debug("Inserted candles",
		zap.Int("inserted", inserted),
		zap.Int("total", len(candles)),
		zap.Duration("duration", duration),
	)

	return inserted, nil
}

// InsertCandlesBatch inserts candles in batches with the specified batch size.
func (d *DatabaseBackfiller) InsertCandlesBatch(ctx context.Context, candles []Candle, batchSize int) (int, error) {
	if len(candles) == 0 {
		return 0, nil
	}

	totalInserted := 0

	for i := 0; i < len(candles); i += batchSize {
		end := i + batchSize
		if end > len(candles) {
			end = len(candles)
		}

		batch := candles[i:end]
		inserted, err := d.InsertCandles(ctx, batch)
		if err != nil {
			return totalInserted, fmt.Errorf("insert batch %d: %w", i/batchSize, err)
		}

		totalInserted += inserted

		d.logger.Debug("Batch progress",
			zap.Int("batch", i/batchSize+1),
			zap.Int("batch_inserted", inserted),
			zap.Int("total_inserted", totalInserted),
		)
	}

	return totalInserted, nil
}

// LogGapDetected logs a gap detection event to data_quality_logs.
func (d *DatabaseBackfiller) LogGapDetected(ctx context.Context, gapStart, gapEnd time.Time, candlesMissing int) error {
	_, err := d.db.ExecContext(ctx, `
		INSERT INTO data_quality_logs (
			event_type, gap_start, gap_end, candles_missing, source, resolved
		) VALUES ('gap_detected', $1, $2, $3, 'gap_handler', false)
	`, gapStart, gapEnd, candlesMissing)

	if err != nil {
		metricsDatabaseErrors.Inc()
		return fmt.Errorf("log gap detected: %w", err)
	}

	return nil
}

// LogGapBackfilled logs a successful backfill and marks the gap as resolved.
func (d *DatabaseBackfiller) LogGapBackfilled(ctx context.Context, gapStart, gapEnd time.Time, candlesRecovered int) error {
	start := time.Now()

	// Start transaction
	tx, err := d.db.BeginTx(ctx, nil)
	if err != nil {
		metricsDatabaseErrors.Inc()
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Mark existing gap_detected entries as resolved
	_, err = tx.ExecContext(ctx, `
		UPDATE data_quality_logs
		SET resolved = true, resolved_at = NOW(), candles_recovered = $3
		WHERE event_type = 'gap_detected'
		  AND gap_start = $1
		  AND gap_end = $2
		  AND resolved = false
	`, gapStart, gapEnd, candlesRecovered)
	if err != nil {
		d.logger.Warn("Failed to update gap_detected entry", zap.Error(err))
	}

	// Insert gap_backfilled log entry
	_, err = tx.ExecContext(ctx, `
		INSERT INTO data_quality_logs (
			event_type, gap_start, gap_end, candles_recovered, source, resolved, resolved_at
		) VALUES ('gap_backfilled', $1, $2, $3, 'gap_handler', true, NOW())
	`, gapStart, gapEnd, candlesRecovered)
	if err != nil {
		metricsDatabaseErrors.Inc()
		return fmt.Errorf("log gap backfilled: %w", err)
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		metricsDatabaseErrors.Inc()
		return fmt.Errorf("commit transaction: %w", err)
	}

	RecordDatabaseOperation(time.Since(start).Seconds())

	d.logger.Info("Logged gap backfill",
		zap.Time("gap_start", gapStart),
		zap.Time("gap_end", gapEnd),
		zap.Int("candles_recovered", candlesRecovered),
	)

	return nil
}

// LogError logs an error event to data_quality_logs.
func (d *DatabaseBackfiller) LogError(ctx context.Context, source, errorMessage string) error {
	_, err := d.db.ExecContext(ctx, `
		INSERT INTO data_quality_logs (
			event_type, source, error_message, resolved
		) VALUES ('error', $1, $2, false)
	`, source, errorMessage)

	if err != nil {
		metricsDatabaseErrors.Inc()
		return fmt.Errorf("log error: %w", err)
	}

	return nil
}

// GetUnresolvedGaps returns all unresolved gap_detected entries.
func (d *DatabaseBackfiller) GetUnresolvedGaps(ctx context.Context) ([]GapInfo, error) {
	rows, err := d.db.QueryContext(ctx, `
		SELECT gap_start, gap_end, candles_missing, created_at
		FROM data_quality_logs
		WHERE event_type = 'gap_detected'
		  AND resolved = false
		ORDER BY gap_start ASC
	`)
	if err != nil {
		metricsDatabaseErrors.Inc()
		return nil, fmt.Errorf("query unresolved gaps: %w", err)
	}
	defer rows.Close()

	var gaps []GapInfo
	for rows.Next() {
		var gap GapInfo
		if err := rows.Scan(&gap.GapStart, &gap.GapEnd, &gap.CandlesMissing, &gap.DetectedAt); err != nil {
			continue
		}
		gaps = append(gaps, gap)
	}

	return gaps, rows.Err()
}

// GetDataQualitySummary returns a summary of data quality events.
func (d *DatabaseBackfiller) GetDataQualitySummary(ctx context.Context) (*DataQualitySummary, error) {
	summary := &DataQualitySummary{}

	// Count unresolved gaps
	err := d.db.QueryRowContext(ctx, `
		SELECT COUNT(*), COALESCE(SUM(candles_missing), 0)
		FROM data_quality_logs
		WHERE event_type = 'gap_detected'
		  AND resolved = false
	`).Scan(&summary.UnresolvedGaps, &summary.TotalCandlesMissing)
	if err != nil {
		return nil, fmt.Errorf("count unresolved gaps: %w", err)
	}

	// Count backfills in last 24 hours
	err = d.db.QueryRowContext(ctx, `
		SELECT COUNT(*), COALESCE(SUM(candles_recovered), 0)
		FROM data_quality_logs
		WHERE event_type = 'gap_backfilled'
		  AND created_at > NOW() - INTERVAL '24 hours'
	`).Scan(&summary.BackfillsLast24h, &summary.CandlesRecoveredLast24h)
	if err != nil {
		return nil, fmt.Errorf("count recent backfills: %w", err)
	}

	// Count errors in last 24 hours
	err = d.db.QueryRowContext(ctx, `
		SELECT COUNT(*)
		FROM data_quality_logs
		WHERE event_type = 'error'
		  AND created_at > NOW() - INTERVAL '24 hours'
	`).Scan(&summary.ErrorsLast24h)
	if err != nil {
		return nil, fmt.Errorf("count recent errors: %w", err)
	}

	return summary, nil
}

// CountCandles returns the total number of candles in the database.
func (d *DatabaseBackfiller) CountCandles(ctx context.Context) (int64, error) {
	var count int64
	err := d.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM candles_1m
	`).Scan(&count)
	if err != nil {
		metricsDatabaseErrors.Inc()
		return 0, fmt.Errorf("count candles: %w", err)
	}
	return count, nil
}

// GetOldestCandleTime returns the timestamp of the oldest candle in the database.
// Returns nil if no candles exist.
func (d *DatabaseBackfiller) GetOldestCandleTime(ctx context.Context) (*time.Time, error) {
	var oldest sql.NullTime
	err := d.db.QueryRowContext(ctx, `
		SELECT MIN(time) FROM candles_1m
	`).Scan(&oldest)
	if err != nil {
		metricsDatabaseErrors.Inc()
		return nil, fmt.Errorf("get oldest candle: %w", err)
	}
	if !oldest.Valid {
		return nil, nil
	}
	return &oldest.Time, nil
}

// GetNewestCandleTime returns the timestamp of the newest candle in the database.
// Returns nil if no candles exist.
func (d *DatabaseBackfiller) GetNewestCandleTime(ctx context.Context) (*time.Time, error) {
	var newest sql.NullTime
	err := d.db.QueryRowContext(ctx, `
		SELECT MAX(time) FROM candles_1m
	`).Scan(&newest)
	if err != nil {
		metricsDatabaseErrors.Inc()
		return nil, fmt.Errorf("get newest candle: %w", err)
	}
	if !newest.Valid {
		return nil, nil
	}
	return &newest.Time, nil
}

// GetDataCoverage returns the oldest and newest candle times, plus total count.
func (d *DatabaseBackfiller) GetDataCoverage(ctx context.Context) (oldest *time.Time, newest *time.Time, count int64, err error) {
	oldest, err = d.GetOldestCandleTime(ctx)
	if err != nil {
		return nil, nil, 0, err
	}
	newest, err = d.GetNewestCandleTime(ctx)
	if err != nil {
		return nil, nil, 0, err
	}
	count, err = d.CountCandles(ctx)
	if err != nil {
		return nil, nil, 0, err
	}
	return oldest, newest, count, nil
}

// GapInfo represents information about a detected gap.
type GapInfo struct {
	GapStart       time.Time
	GapEnd         time.Time
	CandlesMissing int
	DetectedAt     time.Time
}

// DataQualitySummary contains summary statistics about data quality.
type DataQualitySummary struct {
	UnresolvedGaps          int
	TotalCandlesMissing     int
	BackfillsLast24h        int
	CandlesRecoveredLast24h int
	ErrorsLast24h           int
}
