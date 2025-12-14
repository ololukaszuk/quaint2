// Package main provides health check HTTP handlers.
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// HealthHandler handles health check requests.
type HealthHandler struct {
	database    *DatabaseBackfiller
	apiClient   *BinanceAPIClient
	backfiller  *GapBackfiller
	logger      *zap.Logger
	startTime   time.Time
	errors24h   int64
}

// HealthResponse represents the health check response.
type HealthResponse struct {
	Status            string    `json:"status"`
	UptimeSeconds     int64     `json:"uptime_seconds"`
	BackfillsCompleted int64    `json:"backfills_completed"`
	CandlesRecovered   int64    `json:"candles_recovered"`
	Errors24h          int64    `json:"errors_24h"`
	BinanceAPIStatus   string   `json:"binance_api_status"`
	DatabaseStatus     string   `json:"database_status"`
	PendingBackfills   int      `json:"pending_backfills"`
	ActiveBackfills    int      `json:"active_backfills"`
	Timestamp          string   `json:"timestamp"`
	Details            *HealthDetails `json:"details,omitempty"`
}

// HealthDetails contains detailed health information.
type HealthDetails struct {
	UnresolvedGaps          int   `json:"unresolved_gaps"`
	CandlesMissing          int   `json:"candles_missing"`
	BackfillsLast24h        int   `json:"backfills_last_24h"`
	CandlesRecoveredLast24h int   `json:"candles_recovered_last_24h"`
	RateLimiterStats        *RateLimiterHealthStats `json:"rate_limiter,omitempty"`
}

// RateLimiterHealthStats contains rate limiter statistics for health.
type RateLimiterHealthStats struct {
	ConsecutiveFailures int     `json:"consecutive_failures"`
	TokensAvailable     float64 `json:"tokens_available"`
	InBackoff           bool    `json:"in_backoff"`
}

// NewHealthHandler creates a new health handler.
func NewHealthHandler(
	database *DatabaseBackfiller,
	apiClient *BinanceAPIClient,
	backfiller *GapBackfiller,
	logger *zap.Logger,
) *HealthHandler {
	return &HealthHandler{
		database:   database,
		apiClient:  apiClient,
		backfiller: backfiller,
		logger:     logger,
		startTime:  time.Now(),
	}
}

// IncrementErrors increments the 24h error counter.
func (h *HealthHandler) IncrementErrors() {
	atomic.AddInt64(&h.errors24h, 1)
}

// ResetErrors resets the 24h error counter.
func (h *HealthHandler) ResetErrors() {
	atomic.StoreInt64(&h.errors24h, 0)
}

// ServeHTTP handles GET /health requests.
func (h *HealthHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	response := h.buildHealthResponse(ctx)

	w.Header().Set("Content-Type", "application/json")

	switch response.Status {
	case "healthy":
		w.WriteHeader(http.StatusOK)
	case "degraded":
		w.WriteHeader(http.StatusOK)
	default:
		w.WriteHeader(http.StatusServiceUnavailable)
	}

	json.NewEncoder(w).Encode(response)
}

// buildHealthResponse builds the health check response.
func (h *HealthHandler) buildHealthResponse(ctx context.Context) HealthResponse {
	response := HealthResponse{
		Timestamp:     time.Now().UTC().Format(time.RFC3339),
		UptimeSeconds: int64(time.Since(h.startTime).Seconds()),
		Errors24h:     atomic.LoadInt64(&h.errors24h),
	}

	// Get backfiller stats
	stats := h.backfiller.GetStats()
	response.ActiveBackfills = stats.ActiveBackfills
	response.BackfillsCompleted = stats.TotalBackfills
	response.CandlesRecovered = stats.TotalRecovered

	// Check database connection
	if err := h.database.Ping(ctx); err != nil {
		response.DatabaseStatus = "disconnected"
		h.logger.Warn("Database health check failed", zap.Error(err))
	} else {
		response.DatabaseStatus = "connected"
	}

	// Check Binance API
	if err := h.apiClient.Ping(ctx); err != nil {
		response.BinanceAPIStatus = "offline"
		h.logger.Warn("Binance API health check failed", zap.Error(err))
	} else {
		response.BinanceAPIStatus = "online"
	}

	// Get data quality summary
	if summary, err := h.database.GetDataQualitySummary(ctx); err == nil {
		response.PendingBackfills = summary.UnresolvedGaps
		response.Details = &HealthDetails{
			UnresolvedGaps:          summary.UnresolvedGaps,
			CandlesMissing:          summary.TotalCandlesMissing,
			BackfillsLast24h:        summary.BackfillsLast24h,
			CandlesRecoveredLast24h: summary.CandlesRecoveredLast24h,
		}
	}

	// Determine overall status
	response.Status = h.determineStatus(response)

	return response
}

// determineStatus determines the overall health status.
func (h *HealthHandler) determineStatus(resp HealthResponse) string {
	// Unhealthy if database is disconnected
	if resp.DatabaseStatus != "connected" {
		return "unhealthy"
	}

	// Degraded conditions
	if resp.BinanceAPIStatus != "online" {
		return "degraded"
	}

	if resp.Errors24h > 10 {
		return "degraded"
	}

	if resp.PendingBackfills > 10 {
		return "degraded"
	}

	return "healthy"
}

// ReadyHandler handles GET /ready requests.
type ReadyHandler struct {
	database  *DatabaseBackfiller
	apiClient *BinanceAPIClient
}

// NewReadyHandler creates a new ready handler.
func NewReadyHandler(database *DatabaseBackfiller, apiClient *BinanceAPIClient) *ReadyHandler {
	return &ReadyHandler{
		database:  database,
		apiClient: apiClient,
	}
}

// ServeHTTP handles GET /ready requests.
func (h *ReadyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	// Check database
	if err := h.database.Ping(ctx); err != nil {
		http.Error(w, "Database not ready", http.StatusServiceUnavailable)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// LiveHandler handles GET /live requests.
type LiveHandler struct{}

// NewLiveHandler creates a new live handler.
func NewLiveHandler() *LiveHandler {
	return &LiveHandler{}
}

// ServeHTTP handles GET /live requests.
func (h *LiveHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

// StartUptimeCounter starts a goroutine that increments the uptime metric.
func StartUptimeCounter(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metricsUptime.Inc()
		}
	}
}

// StartErrorResetJob starts a daily job to reset error counters.
func StartErrorResetJob(ctx context.Context, healthHandler *HealthHandler, logger *zap.Logger) {
	// Calculate time until next midnight UTC
	now := time.Now().UTC()
	nextMidnight := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC)
	initialDelay := nextMidnight.Sub(now)

	timer := time.NewTimer(initialDelay)
	defer timer.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
			logger.Info("Resetting 24h error counter")
			healthHandler.ResetErrors()
			timer.Reset(24 * time.Hour)
		}
	}
}
