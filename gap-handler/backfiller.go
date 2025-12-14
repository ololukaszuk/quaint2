// Package main provides the core gap backfill logic.
package main

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// GapBackfiller handles gap detection and backfill operations.
type GapBackfiller struct {
	apiClient        *BinanceAPIClient
	database         *DatabaseBackfiller
	logger           *zap.Logger
	config           *Config
	mu               sync.Mutex
	activeCount      int32
	totalBackfills   int64
	totalRecovered   int64
	historicalStatus HistoricalBackfillStatus
}

// BackfillRequest represents a request to backfill a gap.
type BackfillRequest struct {
	GapStart       time.Time `json:"gap_start"`
	GapEnd         time.Time `json:"gap_end"`
	CandlesMissing int64     `json:"candles_missing,omitempty"`
	DetectedAt     time.Time `json:"detected_at,omitempty"`
}

// BackfillResponse represents the response from a backfill operation.
type BackfillResponse struct {
	Status           string    `json:"status"`
	GapStart         time.Time `json:"gap_start"`
	GapEnd           time.Time `json:"gap_end"`
	CandlesExpected  int       `json:"candles_expected"`
	CandlesFetched   int       `json:"candles_fetched"`
	CandlesInserted  int       `json:"candles_inserted"`
	CandlesSkipped   int       `json:"candles_skipped"`
	CandlesRecovered int       `json:"candles_recovered"`
	DurationSeconds  float64   `json:"duration_seconds"`
	Error            string    `json:"error,omitempty"`
}

// NewGapBackfiller creates a new gap backfiller.
func NewGapBackfiller(
	apiClient *BinanceAPIClient,
	database *DatabaseBackfiller,
	config *Config,
	logger *zap.Logger,
) *GapBackfiller {
	return &GapBackfiller{
		apiClient: apiClient,
		database:  database,
		config:    config,
		logger:    logger,
	}
}

// Backfill performs a gap backfill operation.
func (b *GapBackfiller) Backfill(ctx context.Context, req BackfillRequest) (*BackfillResponse, error) {
	start := time.Now()
	response := &BackfillResponse{
		GapStart: req.GapStart,
		GapEnd:   req.GapEnd,
	}

	// Validate request
	if req.GapEnd.Before(req.GapStart) || req.GapEnd.Equal(req.GapStart) {
		response.Status = "error"
		response.Error = "gap_end must be after gap_start"
		return response, fmt.Errorf("invalid gap range: end before start")
	}

	// Check concurrent backfill limit
	current := atomic.AddInt32(&b.activeCount, 1)
	defer atomic.AddInt32(&b.activeCount, -1)
	metricsActiveBackfills.Set(float64(current))

	if int(current) > b.config.MaxConcurrentBackfills {
		atomic.AddInt32(&b.activeCount, -1)
		response.Status = "rejected"
		response.Error = "too many concurrent backfills"
		return response, fmt.Errorf("concurrent backfill limit exceeded")
	}

	b.logger.Info("Starting backfill",
		zap.Time("gap_start", req.GapStart),
		zap.Time("gap_end", req.GapEnd),
		zap.Int32("active_backfills", current),
	)

	// Calculate expected candles
	duration := req.GapEnd.Sub(req.GapStart)
	response.CandlesExpected = int(duration.Minutes())

	// Create backfill context with timeout
	backfillCtx, cancel := context.WithTimeout(ctx, b.config.BackfillTimeout)
	defer cancel()

	// Step 1: Get existing timestamps to avoid duplicates
	existing, err := b.database.GetExistingTimestamps(backfillCtx, req.GapStart, req.GapEnd)
	if err != nil {
		response.Status = "error"
		response.Error = fmt.Sprintf("failed to get existing timestamps: %v", err)
		response.DurationSeconds = time.Since(start).Seconds()
		RecordBackfillDuration(response.DurationSeconds, "error")
		return response, err
	}

	response.CandlesSkipped = len(existing)

	// Step 2: Fetch klines from Binance
	klines, err := b.apiClient.FetchAllKlines(backfillCtx, req.GapStart, req.GapEnd)
	if err != nil {
		response.Status = "error"
		response.Error = fmt.Sprintf("failed to fetch klines: %v", err)
		response.DurationSeconds = time.Since(start).Seconds()
		RecordBackfillDuration(response.DurationSeconds, "error")

		// Log error to database
		_ = b.database.LogError(backfillCtx, "gap_handler", fmt.Sprintf("Backfill failed: %v", err))

		return response, err
	}

	response.CandlesFetched = len(klines)

	// Step 3: Filter out duplicates
	var newCandles []Candle
	for _, kline := range klines {
		candleTime := kline.ToTime()
		if _, exists := existing[candleTime]; !exists {
			newCandles = append(newCandles, Candle{
				Time:                     candleTime,
				Open:                     kline.Open,
				High:                     kline.High,
				Low:                      kline.Low,
				Close:                    kline.Close,
				Volume:                   kline.Volume,
				QuoteAssetVolume:         kline.QuoteAssetVolume,
				TakerBuyBaseAssetVolume:  kline.TakerBuyBaseAssetVolume,
				TakerBuyQuoteAssetVolume: kline.TakerBuyQuoteAssetVolume,
				NumberOfTrades:           kline.NumberOfTrades,
			})
		}
	}

	b.logger.Debug("Filtered duplicates",
		zap.Int("fetched", len(klines)),
		zap.Int("existing", len(existing)),
		zap.Int("new", len(newCandles)),
	)

	// Step 4: Insert new candles in batches
	if len(newCandles) > 0 {
		inserted, err := b.database.InsertCandlesBatch(backfillCtx, newCandles, 1000)
		if err != nil {
			response.Status = "partial"
			response.Error = fmt.Sprintf("partial insert: %v", err)
			response.CandlesInserted = inserted
			response.CandlesRecovered = inserted
		} else {
			response.CandlesInserted = inserted
			response.CandlesRecovered = inserted
		}
	}

	// Step 5: Log successful backfill
	if response.CandlesRecovered > 0 || len(newCandles) == 0 {
		if err := b.database.LogGapBackfilled(backfillCtx, req.GapStart, req.GapEnd, response.CandlesRecovered); err != nil {
			b.logger.Warn("Failed to log backfill completion", zap.Error(err))
		}
	}

	// Update response
	response.DurationSeconds = time.Since(start).Seconds()

	if response.Error == "" {
		response.Status = "backfilled"
	}

	// Update metrics
	metricsCandlesRecovered.Add(float64(response.CandlesRecovered))
	metricsCandlesDuplicate.Add(float64(len(existing)))
	RecordBackfillDuration(response.DurationSeconds, response.Status)

	// Update internal counters
	atomic.AddInt64(&b.totalBackfills, 1)
	atomic.AddInt64(&b.totalRecovered, int64(response.CandlesRecovered))

	b.logger.Info("Backfill completed",
		zap.String("status", response.Status),
		zap.Int("candles_fetched", response.CandlesFetched),
		zap.Int("candles_inserted", response.CandlesInserted),
		zap.Int("candles_skipped", response.CandlesSkipped),
		zap.Float64("duration_seconds", response.DurationSeconds),
	)

	return response, nil
}

// BackfillUnresolved attempts to backfill all unresolved gaps.
func (b *GapBackfiller) BackfillUnresolved(ctx context.Context) ([]*BackfillResponse, error) {
	gaps, err := b.database.GetUnresolvedGaps(ctx)
	if err != nil {
		return nil, fmt.Errorf("get unresolved gaps: %w", err)
	}

	if len(gaps) == 0 {
		b.logger.Info("No unresolved gaps to backfill")
		return nil, nil
	}

	b.logger.Info("Starting backfill of unresolved gaps",
		zap.Int("count", len(gaps)),
	)

	var responses []*BackfillResponse
	for _, gap := range gaps {
		req := BackfillRequest{
			GapStart:       gap.GapStart,
			GapEnd:         gap.GapEnd,
			CandlesMissing: int64(gap.CandlesMissing),
			DetectedAt:     gap.DetectedAt,
		}

		resp, err := b.Backfill(ctx, req)
		if err != nil {
			b.logger.Error("Failed to backfill gap",
				zap.Time("gap_start", gap.GapStart),
				zap.Time("gap_end", gap.GapEnd),
				zap.Error(err),
			)
		}
		responses = append(responses, resp)
	}

	return responses, nil
}

// GetStats returns backfill statistics.
func (b *GapBackfiller) GetStats() BackfillStats {
	return BackfillStats{
		ActiveBackfills:  int(atomic.LoadInt32(&b.activeCount)),
		TotalBackfills:   atomic.LoadInt64(&b.totalBackfills),
		TotalRecovered:   atomic.LoadInt64(&b.totalRecovered),
	}
}

// BackfillStats contains backfill statistics.
type BackfillStats struct {
	ActiveBackfills int
	TotalBackfills  int64
	TotalRecovered  int64
}

// ValidateBackfillRequest validates a backfill request.
func ValidateBackfillRequest(req *BackfillRequest) error {
	if req.GapStart.IsZero() {
		return fmt.Errorf("gap_start is required")
	}
	if req.GapEnd.IsZero() {
		return fmt.Errorf("gap_end is required")
	}
	if req.GapEnd.Before(req.GapStart) {
		return fmt.Errorf("gap_end must be after gap_start")
	}
	if req.GapEnd.Equal(req.GapStart) {
		return fmt.Errorf("gap_end must be after gap_start")
	}

	// Limit max gap size to prevent abuse
	maxGap := 7 * 24 * time.Hour // 1 week
	if req.GapEnd.Sub(req.GapStart) > maxGap {
		return fmt.Errorf("gap too large: maximum is 7 days")
	}

	return nil
}

// HistoricalBackfillStatus tracks the progress of historical backfill.
type HistoricalBackfillStatus struct {
	IsRunning       bool      `json:"is_running"`
	StartedAt       time.Time `json:"started_at,omitempty"`
	TargetStartTime time.Time `json:"target_start_time"`
	CurrentProgress time.Time `json:"current_progress,omitempty"`
	ChunksTotal     int       `json:"chunks_total"`
	ChunksCompleted int       `json:"chunks_completed"`
	CandlesRecovered int64    `json:"candles_recovered"`
	LastError       string    `json:"last_error,omitempty"`
}

// StartHistoricalBackfill initiates background backfill of historical data.
// It calculates the gap between target retention and existing data, then
// backfills in chunks (default 7 days each) to respect API limits.
func (b *GapBackfiller) StartHistoricalBackfill(ctx context.Context, retentionMonths int, chunkDays int) {
	b.mu.Lock()
	if b.historicalStatus.IsRunning {
		b.mu.Unlock()
		b.logger.Info("Historical backfill already running, skipping")
		return
	}
	b.historicalStatus.IsRunning = true
	b.historicalStatus.StartedAt = time.Now()
	b.historicalStatus.CandlesRecovered = 0
	b.historicalStatus.ChunksCompleted = 0
	b.historicalStatus.LastError = ""
	b.mu.Unlock()

	go b.runHistoricalBackfill(ctx, retentionMonths, chunkDays)
}

// runHistoricalBackfill performs the actual historical backfill work.
func (b *GapBackfiller) runHistoricalBackfill(ctx context.Context, retentionMonths int, chunkDays int) {
	defer func() {
		b.mu.Lock()
		b.historicalStatus.IsRunning = false
		b.mu.Unlock()
	}()

	b.logger.Info("Starting historical backfill",
		zap.Int("retention_months", retentionMonths),
		zap.Int("chunk_days", chunkDays),
	)

	// Calculate target start time (now - retention period)
	targetStart := time.Now().UTC().AddDate(0, -retentionMonths, 0).Truncate(time.Minute)
	
	b.mu.Lock()
	b.historicalStatus.TargetStartTime = targetStart
	b.mu.Unlock()

	// Get current data coverage
	oldest, newest, count, err := b.database.GetDataCoverage(ctx)
	if err != nil {
		b.logger.Error("Failed to get data coverage", zap.Error(err))
		b.mu.Lock()
		b.historicalStatus.LastError = err.Error()
		b.mu.Unlock()
		return
	}

	b.logger.Info("Current data coverage",
		zap.Int64("candle_count", count),
		zap.Timep("oldest", oldest),
		zap.Timep("newest", newest),
		zap.Time("target_start", targetStart),
	)

	// Determine what needs to be backfilled
	var gaps []BackfillRequest

	now := time.Now().UTC().Truncate(time.Minute)

	if oldest == nil {
		// No data at all - backfill entire retention period
		b.logger.Info("No existing data, will backfill entire retention period")
		gaps = b.calculateChunks(targetStart, now, chunkDays)
	} else {
		// Have some data - check for gaps before oldest and after newest
		if oldest.After(targetStart) {
			// Need to backfill before oldest data
			b.logger.Info("Backfilling historical gap before oldest data",
				zap.Time("from", targetStart),
				zap.Time("to", *oldest),
			)
			gaps = append(gaps, b.calculateChunks(targetStart, *oldest, chunkDays)...)
		}
		
		// Also check if there's a gap after newest (shouldn't happen normally, but just in case)
		if newest != nil && newest.Before(now.Add(-2*time.Minute)) {
			b.logger.Info("Backfilling gap after newest data",
				zap.Time("from", *newest),
				zap.Time("to", now),
			)
			gaps = append(gaps, b.calculateChunks(newest.Add(time.Minute), now, chunkDays)...)
		}
	}

	if len(gaps) == 0 {
		b.logger.Info("No historical gaps to backfill - data is complete")
		return
	}

	b.mu.Lock()
	b.historicalStatus.ChunksTotal = len(gaps)
	b.mu.Unlock()

	b.logger.Info("Historical backfill plan created",
		zap.Int("total_chunks", len(gaps)),
		zap.Time("from", gaps[0].GapStart),
		zap.Time("to", gaps[len(gaps)-1].GapEnd),
	)

	// Process chunks sequentially (to respect rate limits)
	for i, gap := range gaps {
		select {
		case <-ctx.Done():
			b.logger.Info("Historical backfill cancelled")
			return
		default:
		}

		b.logger.Info("Processing historical chunk",
			zap.Int("chunk", i+1),
			zap.Int("total", len(gaps)),
			zap.Time("start", gap.GapStart),
			zap.Time("end", gap.GapEnd),
		)

		b.mu.Lock()
		b.historicalStatus.CurrentProgress = gap.GapStart
		b.mu.Unlock()

		resp, err := b.Backfill(ctx, gap)
		if err != nil {
			b.logger.Error("Historical chunk backfill failed",
				zap.Int("chunk", i+1),
				zap.Error(err),
			)
			b.mu.Lock()
			b.historicalStatus.LastError = err.Error()
			b.mu.Unlock()
			// Continue with next chunk despite error
		}

		b.mu.Lock()
		b.historicalStatus.ChunksCompleted++
		if resp != nil {
			b.historicalStatus.CandlesRecovered += int64(resp.CandlesRecovered)
		}
		b.mu.Unlock()

		// Small delay between chunks to be nice to the API
		select {
		case <-ctx.Done():
			return
		case <-time.After(1 * time.Second):
		}
	}

	b.logger.Info("Historical backfill completed",
		zap.Int("chunks_completed", len(gaps)),
		zap.Int64("total_candles_recovered", b.historicalStatus.CandlesRecovered),
	)
}

// calculateChunks divides a time range into chunks of the specified size.
func (b *GapBackfiller) calculateChunks(start, end time.Time, chunkDays int) []BackfillRequest {
	var chunks []BackfillRequest
	chunkDuration := time.Duration(chunkDays) * 24 * time.Hour

	current := start
	for current.Before(end) {
		chunkEnd := current.Add(chunkDuration)
		if chunkEnd.After(end) {
			chunkEnd = end
		}

		chunks = append(chunks, BackfillRequest{
			GapStart: current,
			GapEnd:   chunkEnd,
		})

		current = chunkEnd
	}

	return chunks
}

// GetHistoricalBackfillStatus returns the current status of historical backfill.
func (b *GapBackfiller) GetHistoricalBackfillStatus() HistoricalBackfillStatus {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.historicalStatus
}
