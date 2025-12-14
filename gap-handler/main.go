// Package main is the entry point for the gap handler service.
//
// The gap handler service backfills missing Binance candlestick data
// by fetching from the Binance REST API and inserting into TimescaleDB.
//
// Features:
// - HTTP endpoint for backfill requests
// - Rate limiting with exponential backoff
// - Batch database writes with deduplication
// - Prometheus metrics
// - Health check endpoints
// - Graceful shutdown
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func main() {
	// Initialize logger
	logger := initLogger()
	defer logger.Sync()

	logger.Info("Starting Gap Handler Service")

	// Load configuration
	cfg, err := LoadConfig()
	if err != nil {
		logger.Fatal("Failed to load configuration", zap.Error(err))
	}

	logger.Info("Configuration loaded",
		zap.String("db_host", cfg.DBHost),
		zap.Int("db_port", cfg.DBPort),
		zap.String("db_name", cfg.DBName),
		zap.Int("port", cfg.Port),
		zap.Int("max_concurrent_backfills", cfg.MaxConcurrentBackfills),
		zap.Duration("backfill_timeout", cfg.BackfillTimeout),
		zap.Bool("historical_backfill_enabled", cfg.HistoricalBackfillEnabled),
		zap.Int("historical_retention_months", cfg.HistoricalRetentionMonths),
		zap.Int("historical_chunk_days", cfg.HistoricalChunkDays),
	)

	// Create context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize rate limiter
	rateLimiter := NewRateLimiter(cfg.RequestsPerSecond, cfg.BurstSize, logger)

	// Initialize Binance API client
	apiClient := NewBinanceAPIClient(cfg, rateLimiter, logger)

	// Initialize database
	database, err := NewDatabaseBackfiller(cfg, logger)
	if err != nil {
		logger.Fatal("Failed to connect to database", zap.Error(err))
	}
	defer database.Close()

	// Initialize backfiller
	backfiller := NewGapBackfiller(apiClient, database, cfg, logger)

	// Initialize health handler
	healthHandler := NewHealthHandler(database, apiClient, backfiller, logger)

	// Start uptime counter
	go StartUptimeCounter(ctx)

	// Start error reset job
	go StartErrorResetJob(ctx, healthHandler, logger)

	// Start historical backfill if enabled
	if cfg.HistoricalBackfillEnabled {
		logger.Info("Historical backfill enabled, starting background process",
			zap.Int("retention_months", cfg.HistoricalRetentionMonths),
			zap.Int("chunk_days", cfg.HistoricalChunkDays),
		)
		// Small delay to let the service fully initialize
		go func() {
			time.Sleep(5 * time.Second)
			backfiller.StartHistoricalBackfill(ctx, cfg.HistoricalRetentionMonths, cfg.HistoricalChunkDays)
		}()
	} else {
		logger.Info("Historical backfill disabled")
	}

	// Create router
	router := mux.NewRouter()

	// Backfill endpoint
	router.HandleFunc("/backfill", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		handleBackfill(w, r, backfiller, healthHandler, logger)
	}).Methods(http.MethodPost)

	// Health endpoints
	router.Handle("/health", healthHandler).Methods(http.MethodGet)
	router.Handle("/ready", NewReadyHandler(database, apiClient)).Methods(http.MethodGet)
	router.Handle("/live", NewLiveHandler()).Methods(http.MethodGet)

	// Metrics endpoint
	router.Handle("/metrics", promhttp.Handler()).Methods(http.MethodGet)

	// Backfill unresolved gaps endpoint
	router.HandleFunc("/backfill/unresolved", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		handleBackfillUnresolved(w, r, backfiller, logger)
	}).Methods(http.MethodPost)

	// Historical backfill status endpoint
	router.HandleFunc("/backfill/historical/status", func(w http.ResponseWriter, r *http.Request) {
		status := backfiller.GetHistoricalBackfillStatus()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(status)
	}).Methods(http.MethodGet)

	// Trigger historical backfill manually
	router.HandleFunc("/backfill/historical/start", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		status := backfiller.GetHistoricalBackfillStatus()
		if status.IsRunning {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			json.NewEncoder(w).Encode(map[string]string{
				"error": "historical backfill already running",
			})
			return
		}

		backfiller.StartHistoricalBackfill(r.Context(), cfg.HistoricalRetentionMonths, cfg.HistoricalChunkDays)
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{
			"status": "started",
		})
	}).Methods(http.MethodPost)

	// Graceful shutdown endpoint
	router.HandleFunc("/shutdown", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "shutting_down"}`))

		// Trigger shutdown
		go func() {
			time.Sleep(100 * time.Millisecond)
			cancel()
		}()
	}).Methods(http.MethodPost)

	// Create HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Port),
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: cfg.BackfillTimeout + 30*time.Second, // Allow time for backfill + response
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine
	go func() {
		logger.Info("HTTP server starting",
			zap.String("addr", server.Addr),
		)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("HTTP server error", zap.Error(err))
		}
	}()

	// Wait for shutdown signal
	waitForShutdown(ctx, cancel, server, logger)

	logger.Info("Gap Handler Service stopped")
}

// initLogger initializes the zap logger.
func initLogger() *zap.Logger {
	logLevel := os.Getenv("LOG_LEVEL")
	if logLevel == "" {
		logLevel = "info"
	}

	var level zapcore.Level
	switch logLevel {
	case "debug":
		level = zapcore.DebugLevel
	case "info":
		level = zapcore.InfoLevel
	case "warn":
		level = zapcore.WarnLevel
	case "error":
		level = zapcore.ErrorLevel
	default:
		level = zapcore.InfoLevel
	}

	config := zap.Config{
		Level:            zap.NewAtomicLevelAt(level),
		Development:      false,
		Encoding:         "json",
		EncoderConfig:    zap.NewProductionEncoderConfig(),
		OutputPaths:      []string{"stdout"},
		ErrorOutputPaths: []string{"stderr"},
	}

	config.EncoderConfig.TimeKey = "timestamp"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	logger, err := config.Build()
	if err != nil {
		panic(fmt.Sprintf("Failed to initialize logger: %v", err))
	}

	return logger
}

// handleBackfill handles POST /backfill requests.
func handleBackfill(
	w http.ResponseWriter,
	r *http.Request,
	backfiller *GapBackfiller,
	healthHandler *HealthHandler,
	logger *zap.Logger,
) {
	var req BackfillRequest

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		logger.Warn("Invalid request body", zap.Error(err))
		http.Error(w, `{"error": "invalid request body"}`, http.StatusBadRequest)
		return
	}

	// Validate request
	if err := ValidateBackfillRequest(&req); err != nil {
		logger.Warn("Invalid backfill request",
			zap.Error(err),
			zap.Time("gap_start", req.GapStart),
			zap.Time("gap_end", req.GapEnd),
		)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{
			"error": err.Error(),
		})
		return
	}

	logger.Info("Received backfill request",
		zap.Time("gap_start", req.GapStart),
		zap.Time("gap_end", req.GapEnd),
	)

	// Perform backfill
	response, err := backfiller.Backfill(r.Context(), req)
	if err != nil {
		healthHandler.IncrementErrors()
		logger.Error("Backfill failed",
			zap.Error(err),
			zap.Time("gap_start", req.GapStart),
			zap.Time("gap_end", req.GapEnd),
		)
	}

	// Send response
	w.Header().Set("Content-Type", "application/json")

	switch response.Status {
	case "backfilled":
		w.WriteHeader(http.StatusOK)
	case "partial":
		w.WriteHeader(http.StatusOK)
	case "rejected":
		w.WriteHeader(http.StatusTooManyRequests)
	default:
		w.WriteHeader(http.StatusInternalServerError)
	}

	json.NewEncoder(w).Encode(response)
}

// handleBackfillUnresolved handles POST /backfill/unresolved requests.
func handleBackfillUnresolved(
	w http.ResponseWriter,
	r *http.Request,
	backfiller *GapBackfiller,
	logger *zap.Logger,
) {
	logger.Info("Starting backfill of unresolved gaps")

	responses, err := backfiller.BackfillUnresolved(r.Context())
	if err != nil {
		logger.Error("Failed to backfill unresolved gaps", zap.Error(err))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{
			"error": err.Error(),
		})
		return
	}

	// Calculate summary
	totalRecovered := 0
	successful := 0
	failed := 0

	for _, resp := range responses {
		if resp.Status == "backfilled" {
			successful++
			totalRecovered += resp.CandlesRecovered
		} else {
			failed++
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":           "completed",
		"gaps_processed":   len(responses),
		"gaps_successful":  successful,
		"gaps_failed":      failed,
		"candles_recovered": totalRecovered,
		"details":          responses,
	})
}

// waitForShutdown waits for shutdown signals and gracefully stops the server.
func waitForShutdown(ctx context.Context, cancel context.CancelFunc, server *http.Server, logger *zap.Logger) {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	select {
	case sig := <-sigChan:
		logger.Info("Received signal, initiating shutdown",
			zap.String("signal", sig.String()),
		)
	case <-ctx.Done():
		logger.Info("Context cancelled, initiating shutdown")
	}

	// Cancel context to stop background tasks
	cancel()

	// Give server time to finish current requests
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	logger.Info("Shutting down HTTP server")
	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("HTTP server shutdown error", zap.Error(err))
	}

	logger.Info("Graceful shutdown completed")
}
