package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/rs/cors"
)

var (
	db     *pgxpool.Pool
	logger *log.Logger
	apiKey string
)

// ============================================================================
// Response structures
// ============================================================================

type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
	Count   int         `json:"count,omitempty"`
}

type Candle struct {
	Time                     time.Time `json:"time"`
	Open                     float64   `json:"open,string"`
	High                     float64   `json:"high,string"`
	Low                      float64   `json:"low,string"`
	Close                    float64   `json:"close,string"`
	Volume                   float64   `json:"volume,string"`
	QuoteAssetVolume         *float64  `json:"quote_asset_volume,string,omitempty"`
	TakerBuyBaseAssetVolume  *float64  `json:"taker_buy_base_asset_volume,string,omitempty"`
	TakerBuyQuoteAssetVolume *float64  `json:"taker_buy_quote_asset_volume,string,omitempty"`
	NumberOfTrades           *int64    `json:"number_of_trades,omitempty"`
	SpreadBps                *float64  `json:"spread_bps,string,omitempty"`
	TakerBuyRatio            *float64  `json:"taker_buy_ratio,string,omitempty"`
	MidPrice                 *float64  `json:"mid_price,string,omitempty"`
}

type DataQualityLog struct {
	ID               int64      `json:"id"`
	EventType        string     `json:"event_type"`
	GapStart         *time.Time `json:"gap_start,omitempty"`
	GapEnd           *time.Time `json:"gap_end,omitempty"`
	CandlesMissing   *int64     `json:"candles_missing,omitempty"`
	CandlesRecovered *int64     `json:"candles_recovered,omitempty"`
	Source           string     `json:"source"`
	ErrorMessage     *string    `json:"error_message,omitempty"`
	Resolved         bool       `json:"resolved"`
	ResolvedAt       *time.Time `json:"resolved_at,omitempty"`
	CreatedAt        time.Time  `json:"created_at"`
}

// MLPrediction15m - ML model predictions for 15-minute intervals
type MLPrediction15m struct {
	ID                    int64      `json:"id"`
	Time                  time.Time  `json:"time"`
	TargetTimeStart       time.Time  `json:"target_time_start"`
	TargetTimeEnd         time.Time  `json:"target_time_end"`
	IntervalOpenPrice     *float64   `json:"interval_open_price,string,omitempty"`
	IntervalClosePrice    *float64   `json:"interval_close_price,string,omitempty"`
	ActualDirection       *int       `json:"actual_direction,omitempty"`
	ActualDirectionLabel  *string    `json:"actual_direction_label,omitempty"`
	PredictedDirection    int        `json:"predicted_direction"`
	PredictedDirectionLabel string   `json:"predicted_direction_label"`
	PredictedCertainty    float64    `json:"predicted_certainty"`
	PredictionCount       int        `json:"prediction_count"`
	AvgCertainty          *float64   `json:"avg_certainty,omitempty"`
	WasPredictionCorrect  *bool      `json:"was_prediction_correct,omitempty"`
}

// MLPredictionStats - Aggregated statistics for ML predictions
type MLPredictionStats struct {
	TotalPredictions     int     `json:"total_predictions"`
	CorrectPredictions   int     `json:"correct_predictions"`
	Accuracy             float64 `json:"accuracy"`
	IntervalsTotal       int     `json:"intervals_total"`
	IntervalsWon         int     `json:"intervals_won"`
	IntervalWinRate      float64 `json:"interval_win_rate"`
	AvgConfidence        float64 `json:"avg_confidence"`
	HighConfidenceAcc    float64 `json:"high_confidence_accuracy"`
	EarlyAccuracy        float64 `json:"early_accuracy"`
	MidAccuracy          float64 `json:"mid_accuracy"`
	LateAccuracy         float64 `json:"late_accuracy"`
}

// LLMAnalysis - Enhanced with market context fields (schema v2.0)
type LLMAnalysis struct {
	ID                   int64           `json:"id"`
	AnalysisTime         time.Time       `json:"analysis_time"`
	Price                float64         `json:"price,string"`
	PredictionDirection  string          `json:"prediction_direction"`
	PredictionConfidence string          `json:"prediction_confidence"`
	PredictedPrice1h     *float64        `json:"predicted_price_1h,string,omitempty"`
	PredictedPrice4h     *float64        `json:"predicted_price_4h,string,omitempty"`
	KeyLevels            *string         `json:"key_levels,omitempty"`
	Reasoning            *string         `json:"reasoning,omitempty"`
	FullResponse         *string         `json:"full_response,omitempty"`
	ModelName            string          `json:"model_name"`
	ResponseTimeSeconds  *float64        `json:"response_time_seconds,omitempty"`
	ActualPrice1h        *float64        `json:"actual_price_1h,string,omitempty"`
	ActualPrice4h        *float64        `json:"actual_price_4h,string,omitempty"`
	DirectionCorrect1h   *bool           `json:"direction_correct_1h,omitempty"`
	DirectionCorrect4h   *bool           `json:"direction_correct_4h,omitempty"`
	// Enhanced fields (migration 002)
	InvalidationLevel  *float64        `json:"invalidation_level,string,omitempty"`
	CriticalSupport    *float64        `json:"critical_support,string,omitempty"`
	CriticalResistance *float64        `json:"critical_resistance,string,omitempty"`
	MarketContext      json.RawMessage `json:"market_context,omitempty"`
	SignalFactorsUsed  json.RawMessage `json:"signal_factors_used,omitempty"`
	SMCBiasAtAnalysis  *string         `json:"smc_bias_at_analysis,omitempty"`
	TrendsAtAnalysis   json.RawMessage `json:"trends_at_analysis,omitempty"`
	WarningsAtAnalysis json.RawMessage `json:"warnings_at_analysis,omitempty"`
	CreatedAt          time.Time       `json:"created_at"`
}

// MarketAnalysis - Complete enhanced schema v2.0
type MarketAnalysis struct {
	ID               int64           `json:"id"`
	AnalysisTime     time.Time       `json:"analysis_time"`
	Price            float64         `json:"price,string"`
	SignalType       string          `json:"signal_type"`
	SignalDirection  string          `json:"signal_direction"`
	SignalConfidence float64         `json:"signal_confidence"`

	// Trade setup
	EntryPrice      *float64 `json:"entry_price,string,omitempty"`
	StopLoss        *float64 `json:"stop_loss,string,omitempty"`
	TakeProfit1     *float64 `json:"take_profit_1,string,omitempty"`
	TakeProfit2     *float64 `json:"take_profit_2,string,omitempty"`
	TakeProfit3     *float64 `json:"take_profit_3,string,omitempty"`
	RiskRewardRatio *float64 `json:"risk_reward_ratio,omitempty"`

	// Signal reasoning (JSONB)
	SignalFactors json.RawMessage `json:"signal_factors,omitempty"`

	// Trends (JSONB - multi-timeframe)
	Trends json.RawMessage `json:"trends,omitempty"`

	// Support/Resistance - Original simple fields
	NearestSupport     *float64 `json:"nearest_support,string,omitempty"`
	NearestResistance  *float64 `json:"nearest_resistance,string,omitempty"`
	SupportStrength    *float64 `json:"support_strength,omitempty"`
	ResistanceStrength *float64 `json:"resistance_strength,omitempty"`

	// Support/Resistance - Enhanced (JSONB arrays with all levels)
	SupportLevels    json.RawMessage `json:"support_levels,omitempty"`
	ResistanceLevels json.RawMessage `json:"resistance_levels,omitempty"`

	// SMC - Original fields
	SMCBias          *string  `json:"smc_bias,omitempty"`
	PriceZone        *string  `json:"price_zone,omitempty"`
	EquilibriumPrice *float64 `json:"equilibrium_price,string,omitempty"`

	// SMC - Enhanced (full data)
	SMCPriceZone   *string         `json:"smc_price_zone,omitempty"`
	SMCEquilibrium *float64        `json:"smc_equilibrium,string,omitempty"`
	SMCOrderBlocks json.RawMessage `json:"smc_order_blocks,omitempty"`
	SMCFVGs        json.RawMessage `json:"smc_fvgs,omitempty"`
	SMCBreaks      json.RawMessage `json:"smc_breaks,omitempty"`
	SMCLiquidity   json.RawMessage `json:"smc_liquidity,omitempty"`

	// Pivot - Original simple fields
	DailyPivot   *float64 `json:"daily_pivot,string,omitempty"`
	PriceVsPivot *string  `json:"price_vs_pivot,omitempty"`

	// Pivot - Enhanced (all 5 methods: Traditional, Fibonacci, Camarilla, Woodie, DeMark)
	PivotDaily           *float64        `json:"pivot_daily,string,omitempty"`
	PivotR1Traditional   *float64        `json:"pivot_r1_traditional,string,omitempty"`
	PivotR2Traditional   *float64        `json:"pivot_r2_traditional,string,omitempty"`
	PivotR3Traditional   *float64        `json:"pivot_r3_traditional,string,omitempty"`
	PivotS1Traditional   *float64        `json:"pivot_s1_traditional,string,omitempty"`
	PivotS2Traditional   *float64        `json:"pivot_s2_traditional,string,omitempty"`
	PivotS3Traditional   *float64        `json:"pivot_s3_traditional,string,omitempty"`
	PivotR1Fibonacci     *float64        `json:"pivot_r1_fibonacci,string,omitempty"`
	PivotR2Fibonacci     *float64        `json:"pivot_r2_fibonacci,string,omitempty"`
	PivotR3Fibonacci     *float64        `json:"pivot_r3_fibonacci,string,omitempty"`
	PivotS1Fibonacci     *float64        `json:"pivot_s1_fibonacci,string,omitempty"`
	PivotS2Fibonacci     *float64        `json:"pivot_s2_fibonacci,string,omitempty"`
	PivotS3Fibonacci     *float64        `json:"pivot_s3_fibonacci,string,omitempty"`
	PivotCamarilla       *float64        `json:"pivot_camarilla,string,omitempty"`
	PivotR1Camarilla     *float64        `json:"pivot_r1_camarilla,string,omitempty"`
	PivotR2Camarilla     *float64        `json:"pivot_r2_camarilla,string,omitempty"`
	PivotR3Camarilla     *float64        `json:"pivot_r3_camarilla,string,omitempty"`
	PivotR4Camarilla     *float64        `json:"pivot_r4_camarilla,string,omitempty"`
	PivotS1Camarilla     *float64        `json:"pivot_s1_camarilla,string,omitempty"`
	PivotS2Camarilla     *float64        `json:"pivot_s2_camarilla,string,omitempty"`
	PivotS3Camarilla     *float64        `json:"pivot_s3_camarilla,string,omitempty"`
	PivotS4Camarilla     *float64        `json:"pivot_s4_camarilla,string,omitempty"`
	PivotWoodie          *float64        `json:"pivot_woodie,string,omitempty"`
	PivotR1Woodie        *float64        `json:"pivot_r1_woodie,string,omitempty"`
	PivotR2Woodie        *float64        `json:"pivot_r2_woodie,string,omitempty"`
	PivotR3Woodie        *float64        `json:"pivot_r3_woodie,string,omitempty"`
	PivotS1Woodie        *float64        `json:"pivot_s1_woodie,string,omitempty"`
	PivotS2Woodie        *float64        `json:"pivot_s2_woodie,string,omitempty"`
	PivotS3Woodie        *float64        `json:"pivot_s3_woodie,string,omitempty"`
	PivotDeMark          *float64        `json:"pivot_demark,string,omitempty"`
	PivotR1DeMark        *float64        `json:"pivot_r1_demark,string,omitempty"`
	PivotS1DeMark        *float64        `json:"pivot_s1_demark,string,omitempty"`
	PivotConfluenceZones json.RawMessage `json:"pivot_confluence_zones,omitempty"`

	// Momentum - Original simple fields
	RSI1h         *float64 `json:"rsi_1h,omitempty"`
	VolumeRatio1h *float64 `json:"volume_ratio_1h,omitempty"`

	// Momentum - Enhanced (all timeframes as JSONB)
	Momentum json.RawMessage `json:"momentum,omitempty"`

	// Market structure
	StructurePattern  *string  `json:"structure_pattern,omitempty"`
	StructureLastHigh *float64 `json:"structure_last_high,string,omitempty"`
	StructureLastLow  *float64 `json:"structure_last_low,string,omitempty"`

	// Warnings
	Warnings             json.RawMessage `json:"warnings,omitempty"`
	ActionRecommendation *string         `json:"action_recommendation,omitempty"`

	// Summary
	Summary        *string   `json:"summary,omitempty"`
	SignalChanged  bool      `json:"signal_changed"`
	PreviousSignal *string   `json:"previous_signal,omitempty"`
	CreatedAt      time.Time `json:"created_at"`
}

// MarketSignal - Enhanced with signal factors
type MarketSignal struct {
	ID                int64           `json:"id"`
	SignalTime        time.Time       `json:"signal_time"`
	SignalType        string          `json:"signal_type"`
	SignalDirection   string          `json:"signal_direction"`
	SignalConfidence  float64         `json:"signal_confidence"`
	Price             float64         `json:"price,string"`
	EntryPrice        *float64        `json:"entry_price,string,omitempty"`
	StopLoss          *float64        `json:"stop_loss,string,omitempty"`
	TakeProfit1       *float64        `json:"take_profit_1,string,omitempty"`
	TakeProfit2       *float64        `json:"take_profit_2,string,omitempty"`
	TakeProfit3       *float64        `json:"take_profit_3,string,omitempty"`
	RiskRewardRatio   *float64        `json:"risk_reward_ratio,omitempty"`
	PreviousSignalType *string        `json:"previous_signal_type,omitempty"`
	PreviousDirection *string         `json:"previous_direction,omitempty"`
	Summary           *string         `json:"summary,omitempty"`
	KeyReasons        json.RawMessage `json:"key_reasons,omitempty"`
	// Enhanced fields
	SignalFactors     json.RawMessage `json:"signal_factors,omitempty"`
	SMCBias           *string         `json:"smc_bias,omitempty"`
	PivotDaily        *float64        `json:"pivot_daily,string,omitempty"`
	NearestSupport    *float64        `json:"nearest_support,string,omitempty"`
	NearestResistance *float64        `json:"nearest_resistance,string,omitempty"`
	CreatedAt         time.Time       `json:"created_at"`
}

// ============================================================================
// Main and Setup
// ============================================================================

func main() {
	logger = log.New(os.Stdout, "[DATA-API] ", log.LstdFlags)

	// Load API key from environment
	apiKey = os.Getenv("API_KEY")
	if apiKey == "" {
		logger.Fatal("API_KEY environment variable is required for authentication")
	}
	logger.Printf("✅ API key authentication enabled (key length: %d)", len(apiKey))

	// Connect to database
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		logger.Fatal("DATABASE_URL environment variable is required")
	}

	var err error
	db, err = pgxpool.New(context.Background(), dbURL)
	if err != nil {
		logger.Fatalf("Unable to connect to database: %v", err)
	}
	defer db.Close()

	// Test connection
	if err := db.Ping(context.Background()); err != nil {
		logger.Fatalf("Unable to ping database: %v", err)
	}
	logger.Println("✅ Connected to database")

	// Setup router
	r := mux.NewRouter()

	// API routes (all protected except health)
	api := r.PathPrefix("/api/v1").Subrouter()

	// Public endpoint (no auth required)
	api.HandleFunc("/health", healthHandler).Methods("GET")

	// Protected endpoints (require API key)
	protected := api.PathPrefix("").Subrouter()
	protected.Use(apiKeyMiddleware)
	protected.HandleFunc("/candles", getCandlesHandler).Methods("GET")
	protected.HandleFunc("/data-quality-logs", getDataQualityLogsHandler).Methods("GET")
	protected.HandleFunc("/llm-analysis", getLLMAnalysisHandler).Methods("GET")
	protected.HandleFunc("/market-analysis", getMarketAnalysisHandler).Methods("GET")
	protected.HandleFunc("/market-signals", getMarketSignalsHandler).Methods("GET")

	// ML Predictions endpoints
	protected.HandleFunc("/ml/predictions", getMLPredictionsHandler).Methods("GET")
	protected.HandleFunc("/ml/predictions/stats", getMLPredictionStatsHandler).Methods("GET")
	protected.HandleFunc("/ml/predictions/latest", getMLLatestPredictionHandler).Methods("GET")
	protected.HandleFunc("/ml/predictions/intervals", getMLIntervalSummaryHandler).Methods("GET")

	// CORS
	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "OPTIONS"},
		AllowedHeaders:   []string{"*"},
		AllowCredentials: true,
	})

	handler := c.Handler(r)

	// TLS configuration
	useTLS := os.Getenv("USE_TLS") != "false" // Default to true
	port := os.Getenv("PORT")
	if port == "" {
		if useTLS {
			port = "8443"
		} else {
			port = "8080"
		}
	}

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server
	go func() {
		if useTLS {
			certFile := os.Getenv("TLS_CERT_FILE")
			keyFile := os.Getenv("TLS_KEY_FILE")

			if certFile == "" || keyFile == "" {
				// Use self-signed certificate
				certFile = "/certs/server.crt"
				keyFile = "/certs/server.key"
				logger.Println("⚠️  Using self-signed certificate")
			}

			// Configure TLS
			srv.TLSConfig = &tls.Config{
				MinVersion:               tls.VersionTLS12,
				PreferServerCipherSuites: true,
			}

			logger.Printf("🚀 HTTPS server starting on port %s", port)
			if err := srv.ListenAndServeTLS(certFile, keyFile); err != nil && err != http.ErrServerClosed {
				logger.Fatalf("HTTPS server error: %v", err)
			}
		} else {
			logger.Printf("🚀 HTTP server starting on port %s", port)
			if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				logger.Fatalf("HTTP server error: %v", err)
			}
		}
	}()

	logger.Println("✅ Server started successfully")

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Println("🛑 Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Printf("Server forced to shutdown: %v", err)
	}

	logger.Println("✅ Server stopped")
}

// ============================================================================
// Middleware
// ============================================================================

func apiKeyMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Get Authorization header
		authHeader := r.Header.Get("Authorization")

		// Check format: "Bearer <api-key>"
		const bearerPrefix = "Bearer "
		if !strings.HasPrefix(authHeader, bearerPrefix) {
			logger.Printf("❌ Unauthorized request from %s: missing or invalid Authorization header", r.RemoteAddr)
			respondJSON(w, http.StatusUnauthorized, APIResponse{
				Success: false,
				Error:   "Unauthorized: missing or invalid Authorization header. Use: Authorization: Bearer <api-key>",
			})
			return
		}

		// Extract and validate token
		token := strings.TrimPrefix(authHeader, bearerPrefix)
		if token != apiKey {
			logger.Printf("❌ Unauthorized request from %s: invalid API key", r.RemoteAddr)
			respondJSON(w, http.StatusUnauthorized, APIResponse{
				Success: false,
				Error:   "Unauthorized: invalid API key",
			})
			return
		}

		// Valid API key - proceed
		next.ServeHTTP(w, r)
	})
}

// ============================================================================
// Handlers
// ============================================================================

func healthHandler(w http.ResponseWriter, r *http.Request) {
	if err := db.Ping(r.Context()); err != nil {
		respondJSON(w, http.StatusServiceUnavailable, APIResponse{
			Success: false,
			Error:   "Database unavailable",
		})
		return
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data: map[string]string{
			"status":  "healthy",
			"service": "data-api",
			"version": "2.0",
			"time":    time.Now().UTC().Format(time.RFC3339),
		},
	})
}

func getCandlesHandler(w http.ResponseWriter, r *http.Request) {
	// Parse query parameters
	limit := parseInt(r.URL.Query().Get("limit"), 100)
	if limit > 10000 {
		limit = 10000 // Max limit
	}

	startTime := r.URL.Query().Get("start_time")
	endTime := r.URL.Query().Get("end_time")

	// Build query
	query := `SELECT time, open, high, low, close, volume, quote_asset_volume,
		taker_buy_base_asset_volume, taker_buy_quote_asset_volume, number_of_trades,
		spread_bps, taker_buy_ratio, mid_price
		FROM candles_1m WHERE 1=1`

	args := []interface{}{}
	argCount := 1

	if startTime != "" {
		query += fmt.Sprintf(" AND time >= $%d", argCount)
		args = append(args, startTime)
		argCount++
	}

	if endTime != "" {
		query += fmt.Sprintf(" AND time <= $%d", argCount)
		args = append(args, endTime)
		argCount++
	}

	query += fmt.Sprintf(" ORDER BY time DESC LIMIT $%d", argCount)
	args = append(args, limit)

	// Execute query
	rows, err := db.Query(r.Context(), query, args...)
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}
	defer rows.Close()

	candles := []Candle{}
	for rows.Next() {
		var c Candle
		err := rows.Scan(&c.Time, &c.Open, &c.High, &c.Low, &c.Close, &c.Volume,
			&c.QuoteAssetVolume, &c.TakerBuyBaseAssetVolume, &c.TakerBuyQuoteAssetVolume,
			&c.NumberOfTrades, &c.SpreadBps, &c.TakerBuyRatio, &c.MidPrice)
		if err != nil {
			respondJSON(w, http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Scan error: %v", err),
			})
			return
		}
		candles = append(candles, c)
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    candles,
		Count:   len(candles),
	})
}

func getDataQualityLogsHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 100)
	if limit > 1000 {
		limit = 1000
	}

	eventType := r.URL.Query().Get("event_type")
	resolved := r.URL.Query().Get("resolved")

	query := `SELECT id, event_type, gap_start, gap_end, candles_missing, candles_recovered,
		source, error_message, resolved, resolved_at, created_at
		FROM data_quality_logs WHERE 1=1`

	args := []interface{}{}
	argCount := 1

	if eventType != "" {
		query += fmt.Sprintf(" AND event_type = $%d", argCount)
		args = append(args, eventType)
		argCount++
	}

	if resolved != "" {
		query += fmt.Sprintf(" AND resolved = $%d", argCount)
		resolvedBool := resolved == "true"
		args = append(args, resolvedBool)
		argCount++
	}

	query += fmt.Sprintf(" ORDER BY id DESC LIMIT $%d", argCount)
	args = append(args, limit)

	rows, err := db.Query(r.Context(), query, args...)
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}
	defer rows.Close()

	logs := []DataQualityLog{}
	for rows.Next() {
		var l DataQualityLog
		err := rows.Scan(&l.ID, &l.EventType, &l.GapStart, &l.GapEnd, &l.CandlesMissing,
			&l.CandlesRecovered, &l.Source, &l.ErrorMessage, &l.Resolved, &l.ResolvedAt, &l.CreatedAt)
		if err != nil {
			respondJSON(w, http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Scan error: %v", err),
			})
			return
		}
		logs = append(logs, l)
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    logs,
		Count:   len(logs),
	})
}

func getLLMAnalysisHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 50)
	if limit > 1000 {
		limit = 1000
	}

	// Check if enhanced columns exist
	hasEnhanced := checkColumnExists(r.Context(), "llm_analysis", "market_context")

	var query string
	if hasEnhanced {
		query = `SELECT id, analysis_time, price, prediction_direction, prediction_confidence,
			predicted_price_1h, predicted_price_4h, key_levels, reasoning, full_response,
			model_name, response_time_seconds, actual_price_1h, actual_price_4h,
			direction_correct_1h, direction_correct_4h,
			invalidation_level, critical_support, critical_resistance,
			market_context, signal_factors_used, smc_bias_at_analysis,
			trends_at_analysis, warnings_at_analysis, created_at
			FROM llm_analysis
			ORDER BY id DESC, analysis_time DESC
			LIMIT $1`
	} else {
		query = `SELECT id, analysis_time, price, prediction_direction, prediction_confidence,
			predicted_price_1h, predicted_price_4h, key_levels, reasoning, full_response,
			model_name, response_time_seconds, actual_price_1h, actual_price_4h,
			direction_correct_1h, direction_correct_4h, created_at
			FROM llm_analysis
			ORDER BY id DESC, analysis_time DESC
			LIMIT $1`
	}

	rows, err := db.Query(r.Context(), query, limit)
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}
	defer rows.Close()

	analyses := []LLMAnalysis{}
	for rows.Next() {
		var a LLMAnalysis
		var scanErr error

		if hasEnhanced {
			scanErr = rows.Scan(&a.ID, &a.AnalysisTime, &a.Price, &a.PredictionDirection,
				&a.PredictionConfidence, &a.PredictedPrice1h, &a.PredictedPrice4h,
				&a.KeyLevels, &a.Reasoning, &a.FullResponse, &a.ModelName,
				&a.ResponseTimeSeconds, &a.ActualPrice1h, &a.ActualPrice4h,
				&a.DirectionCorrect1h, &a.DirectionCorrect4h,
				&a.InvalidationLevel, &a.CriticalSupport, &a.CriticalResistance,
				&a.MarketContext, &a.SignalFactorsUsed, &a.SMCBiasAtAnalysis,
				&a.TrendsAtAnalysis, &a.WarningsAtAnalysis, &a.CreatedAt)
		} else {
			scanErr = rows.Scan(&a.ID, &a.AnalysisTime, &a.Price, &a.PredictionDirection,
				&a.PredictionConfidence, &a.PredictedPrice1h, &a.PredictedPrice4h,
				&a.KeyLevels, &a.Reasoning, &a.FullResponse, &a.ModelName,
				&a.ResponseTimeSeconds, &a.ActualPrice1h, &a.ActualPrice4h,
				&a.DirectionCorrect1h, &a.DirectionCorrect4h, &a.CreatedAt)
		}

		if scanErr != nil {
			respondJSON(w, http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Scan error: %v", scanErr),
			})
			return
		}
		analyses = append(analyses, a)
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    analyses,
		Count:   len(analyses),
	})
}

func getMarketAnalysisHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 50)
	if limit > 1000 {
		limit = 1000
	}

	signalType := r.URL.Query().Get("signal_type")

	// Build query with all enhanced columns
	query := `SELECT 
		id, analysis_time, price, signal_type, signal_direction, signal_confidence,
		entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
		signal_factors, trends,
		nearest_support, nearest_resistance, support_strength, resistance_strength,
		support_levels, resistance_levels,
		smc_bias, price_zone, equilibrium_price,
		smc_price_zone, smc_equilibrium, smc_order_blocks, smc_fvgs, smc_breaks, smc_liquidity,
		daily_pivot, price_vs_pivot,
		pivot_daily, pivot_r1_traditional, pivot_r2_traditional, pivot_r3_traditional,
		pivot_s1_traditional, pivot_s2_traditional, pivot_s3_traditional,
		pivot_r1_fibonacci, pivot_r2_fibonacci, pivot_r3_fibonacci,
		pivot_s1_fibonacci, pivot_s2_fibonacci, pivot_s3_fibonacci,
		pivot_camarilla, pivot_r1_camarilla, pivot_r2_camarilla, pivot_r3_camarilla, pivot_r4_camarilla,
		pivot_s1_camarilla, pivot_s2_camarilla, pivot_s3_camarilla, pivot_s4_camarilla,
		pivot_woodie, pivot_r1_woodie, pivot_r2_woodie, pivot_r3_woodie,
		pivot_s1_woodie, pivot_s2_woodie, pivot_s3_woodie,
		pivot_demark, pivot_r1_demark, pivot_s1_demark,
		pivot_confluence_zones,
		rsi_1h, volume_ratio_1h, momentum,
		structure_pattern, structure_last_high, structure_last_low,
		warnings, action_recommendation,
		summary, signal_changed, previous_signal, created_at
		FROM market_analysis WHERE 1=1`

	args := []interface{}{}
	argCount := 1

	if signalType != "" {
		query += fmt.Sprintf(" AND signal_type = $%d", argCount)
		args = append(args, signalType)
		argCount++
	}

	query += fmt.Sprintf(" ORDER BY id DESC, analysis_time DESC LIMIT $%d", argCount)
	args = append(args, limit)

	rows, err := db.Query(r.Context(), query, args...)
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}
	defer rows.Close()

	analyses := []MarketAnalysis{}
	for rows.Next() {
		var a MarketAnalysis
		err := rows.Scan(
			&a.ID, &a.AnalysisTime, &a.Price, &a.SignalType, &a.SignalDirection, &a.SignalConfidence,
			&a.EntryPrice, &a.StopLoss, &a.TakeProfit1, &a.TakeProfit2, &a.TakeProfit3, &a.RiskRewardRatio,
			&a.SignalFactors, &a.Trends,
			&a.NearestSupport, &a.NearestResistance, &a.SupportStrength, &a.ResistanceStrength,
			&a.SupportLevels, &a.ResistanceLevels,
			&a.SMCBias, &a.PriceZone, &a.EquilibriumPrice,
			&a.SMCPriceZone, &a.SMCEquilibrium, &a.SMCOrderBlocks, &a.SMCFVGs, &a.SMCBreaks, &a.SMCLiquidity,
			&a.DailyPivot, &a.PriceVsPivot,
			&a.PivotDaily, &a.PivotR1Traditional, &a.PivotR2Traditional, &a.PivotR3Traditional,
			&a.PivotS1Traditional, &a.PivotS2Traditional, &a.PivotS3Traditional,
			&a.PivotR1Fibonacci, &a.PivotR2Fibonacci, &a.PivotR3Fibonacci,
			&a.PivotS1Fibonacci, &a.PivotS2Fibonacci, &a.PivotS3Fibonacci,
			&a.PivotCamarilla, &a.PivotR1Camarilla, &a.PivotR2Camarilla, &a.PivotR3Camarilla, &a.PivotR4Camarilla,
			&a.PivotS1Camarilla, &a.PivotS2Camarilla, &a.PivotS3Camarilla, &a.PivotS4Camarilla,
			&a.PivotWoodie, &a.PivotR1Woodie, &a.PivotR2Woodie, &a.PivotR3Woodie,
			&a.PivotS1Woodie, &a.PivotS2Woodie, &a.PivotS3Woodie,
			&a.PivotDeMark, &a.PivotR1DeMark, &a.PivotS1DeMark,
			&a.PivotConfluenceZones,
			&a.RSI1h, &a.VolumeRatio1h, &a.Momentum,
			&a.StructurePattern, &a.StructureLastHigh, &a.StructureLastLow,
			&a.Warnings, &a.ActionRecommendation,
			&a.Summary, &a.SignalChanged, &a.PreviousSignal, &a.CreatedAt,
		)
		if err != nil {
			respondJSON(w, http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Scan error: %v", err),
			})
			return
		}
		analyses = append(analyses, a)
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    analyses,
		Count:   len(analyses),
	})
}

func getMarketSignalsHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 100)
	if limit > 1000 {
		limit = 1000
	}

	// Check if enhanced columns exist
	hasEnhanced := checkColumnExists(r.Context(), "market_signals", "signal_factors")

	var query string
	if hasEnhanced {
		query = `SELECT id, signal_time, signal_type, signal_direction, signal_confidence, price,
			entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
			previous_signal_type, previous_direction, summary, key_reasons,
			signal_factors, smc_bias, pivot_daily, nearest_support, nearest_resistance,
			created_at
			FROM market_signals
			ORDER BY id DESC, signal_time DESC
			LIMIT $1`
	} else {
		query = `SELECT id, signal_time, signal_type, signal_direction, signal_confidence, price,
			entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
			previous_signal_type, previous_direction, summary, key_reasons, created_at
			FROM market_signals
			ORDER BY id DESC, signal_time DESC
			LIMIT $1`
	}

	rows, err := db.Query(r.Context(), query, limit)
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}
	defer rows.Close()

	signals := []MarketSignal{}
	for rows.Next() {
		var s MarketSignal
		var scanErr error

		if hasEnhanced {
			scanErr = rows.Scan(&s.ID, &s.SignalTime, &s.SignalType, &s.SignalDirection,
				&s.SignalConfidence, &s.Price, &s.EntryPrice, &s.StopLoss, &s.TakeProfit1,
				&s.TakeProfit2, &s.TakeProfit3, &s.RiskRewardRatio, &s.PreviousSignalType,
				&s.PreviousDirection, &s.Summary, &s.KeyReasons,
				&s.SignalFactors, &s.SMCBias, &s.PivotDaily, &s.NearestSupport, &s.NearestResistance,
				&s.CreatedAt)
		} else {
			scanErr = rows.Scan(&s.ID, &s.SignalTime, &s.SignalType, &s.SignalDirection,
				&s.SignalConfidence, &s.Price, &s.EntryPrice, &s.StopLoss, &s.TakeProfit1,
				&s.TakeProfit2, &s.TakeProfit3, &s.RiskRewardRatio, &s.PreviousSignalType,
				&s.PreviousDirection, &s.Summary, &s.KeyReasons, &s.CreatedAt)
		}

		if scanErr != nil {
			respondJSON(w, http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Scan error: %v", scanErr),
			})
			return
		}
		signals = append(signals, s)
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    signals,
		Count:   len(signals),
	})
}

// ============================================================================
// Utilities
// ============================================================================

func respondJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(payload)
}

func parseInt(s string, defaultVal int) int {
	if s == "" {
		return defaultVal
	}
	val, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return defaultVal
	}
	return val
}

// checkColumnExists checks if a column exists in a table
func checkColumnExists(ctx context.Context, table, column string) bool {
	var exists bool
	err := db.QueryRow(ctx, `
		SELECT EXISTS (
			SELECT FROM information_schema.columns 
			WHERE table_name = $1 AND column_name = $2
		)
	`, table, column).Scan(&exists)
	if err != nil {
		return false
	}
	return exists
}

// ============================================================================
// ML Predictions Handlers
// ============================================================================

func getMLPredictionsHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 100)
	if limit > 1000 {
		limit = 1000
	}

	// Optional filters
	intervalStart := r.URL.Query().Get("interval_start")
	onlyCompleted := r.URL.Query().Get("completed") == "true"
	minCertainty := r.URL.Query().Get("min_certainty")

	query := `SELECT id, time, target_time_start, target_time_end,
		interval_open_price, interval_close_price,
		actual_direction, actual_direction_label,
		predicted_direction, predicted_direction_label,
		predicted_certainty, prediction_count, avg_certainty,
		was_prediction_correct
		FROM predictions_15m WHERE 1=1`

	args := []interface{}{}
	argCount := 1

	if intervalStart != "" {
		query += fmt.Sprintf(" AND target_time_start = $%d", argCount)
		args = append(args, intervalStart)
		argCount++
	}

	if onlyCompleted {
		query += " AND was_prediction_correct IS NOT NULL"
	}

	if minCertainty != "" {
		if cert, err := strconv.ParseFloat(minCertainty, 64); err == nil {
			query += fmt.Sprintf(" AND predicted_certainty >= $%d", argCount)
			args = append(args, cert)
			argCount++
		}
	}

	query += fmt.Sprintf(" ORDER BY time DESC LIMIT $%d", argCount)
	args = append(args, limit)

	rows, err := db.Query(r.Context(), query, args...)
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}
	defer rows.Close()

	predictions := []MLPrediction15m{}
	for rows.Next() {
		var p MLPrediction15m
		err := rows.Scan(
			&p.ID, &p.Time, &p.TargetTimeStart, &p.TargetTimeEnd,
			&p.IntervalOpenPrice, &p.IntervalClosePrice,
			&p.ActualDirection, &p.ActualDirectionLabel,
			&p.PredictedDirection, &p.PredictedDirectionLabel,
			&p.PredictedCertainty, &p.PredictionCount, &p.AvgCertainty,
			&p.WasPredictionCorrect,
		)
		if err != nil {
			respondJSON(w, http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Scan error: %v", err),
			})
			return
		}
		predictions = append(predictions, p)
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    predictions,
		Count:   len(predictions),
	})
}

func getMLPredictionStatsHandler(w http.ResponseWriter, r *http.Request) {
	// Time range filter (optional)
	hoursBack := parseInt(r.URL.Query().Get("hours"), 24)
	
	query := `
	WITH completed AS (
		SELECT 
			was_prediction_correct,
			predicted_certainty,
			target_time_start,
			EXTRACT(MINUTE FROM time)::int % 15 as minute_in_interval
		FROM predictions_15m
		WHERE was_prediction_correct IS NOT NULL
		AND time > NOW() - INTERVAL '%d hours'
	),
	interval_results AS (
		SELECT 
			target_time_start,
			COUNT(*) as total,
			SUM(CASE WHEN was_prediction_correct THEN 1 ELSE 0 END) as correct
		FROM completed
		GROUP BY target_time_start
	)
	SELECT 
		(SELECT COUNT(*) FROM completed) as total_predictions,
		(SELECT SUM(CASE WHEN was_prediction_correct THEN 1 ELSE 0 END) FROM completed) as correct_predictions,
		(SELECT COUNT(*) FROM interval_results) as intervals_total,
		(SELECT SUM(CASE WHEN correct > total/2 THEN 1 ELSE 0 END) FROM interval_results) as intervals_won,
		(SELECT AVG(predicted_certainty) FROM completed) as avg_confidence,
		(SELECT AVG(CASE WHEN was_prediction_correct THEN 1.0 ELSE 0.0 END) 
			FROM completed WHERE predicted_certainty >= 0.8) as high_conf_accuracy,
		(SELECT AVG(CASE WHEN was_prediction_correct THEN 1.0 ELSE 0.0 END) 
			FROM completed WHERE minute_in_interval < 5) as early_accuracy,
		(SELECT AVG(CASE WHEN was_prediction_correct THEN 1.0 ELSE 0.0 END) 
			FROM completed WHERE minute_in_interval >= 5 AND minute_in_interval < 10) as mid_accuracy,
		(SELECT AVG(CASE WHEN was_prediction_correct THEN 1.0 ELSE 0.0 END) 
			FROM completed WHERE minute_in_interval >= 10) as late_accuracy
	`
	
	query = fmt.Sprintf(query, hoursBack)

	var stats MLPredictionStats
	var avgConf, highConfAcc, earlyAcc, midAcc, lateAcc *float64
	
	err := db.QueryRow(r.Context(), query).Scan(
		&stats.TotalPredictions,
		&stats.CorrectPredictions,
		&stats.IntervalsTotal,
		&stats.IntervalsWon,
		&avgConf,
		&highConfAcc,
		&earlyAcc,
		&midAcc,
		&lateAcc,
	)
	
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}

	// Calculate derived fields
	if stats.TotalPredictions > 0 {
		stats.Accuracy = float64(stats.CorrectPredictions) / float64(stats.TotalPredictions)
	}
	if stats.IntervalsTotal > 0 {
		stats.IntervalWinRate = float64(stats.IntervalsWon) / float64(stats.IntervalsTotal)
	}
	if avgConf != nil {
		stats.AvgConfidence = *avgConf
	}
	if highConfAcc != nil {
		stats.HighConfidenceAcc = *highConfAcc
	}
	if earlyAcc != nil {
		stats.EarlyAccuracy = *earlyAcc
	}
	if midAcc != nil {
		stats.MidAccuracy = *midAcc
	}
	if lateAcc != nil {
		stats.LateAccuracy = *lateAcc
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    stats,
	})
}

func getMLLatestPredictionHandler(w http.ResponseWriter, r *http.Request) {
	query := `SELECT id, time, target_time_start, target_time_end,
		interval_open_price, interval_close_price,
		actual_direction, actual_direction_label,
		predicted_direction, predicted_direction_label,
		predicted_certainty, prediction_count, avg_certainty,
		was_prediction_correct
		FROM predictions_15m
		ORDER BY time DESC
		LIMIT 1`

	var p MLPrediction15m
	err := db.QueryRow(r.Context(), query).Scan(
		&p.ID, &p.Time, &p.TargetTimeStart, &p.TargetTimeEnd,
		&p.IntervalOpenPrice, &p.IntervalClosePrice,
		&p.ActualDirection, &p.ActualDirectionLabel,
		&p.PredictedDirection, &p.PredictedDirectionLabel,
		&p.PredictedCertainty, &p.PredictionCount, &p.AvgCertainty,
		&p.WasPredictionCorrect,
	)

	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    p,
	})
}

func getMLIntervalSummaryHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 20)
	if limit > 100 {
		limit = 100
	}

	query := `SELECT 
		target_time_start,
		target_time_end,
		MIN(interval_open_price) as open_price,
		MAX(interval_close_price) as close_price,
		MAX(actual_direction_label) as actual_direction,
		COUNT(*) as prediction_count,
		SUM(CASE WHEN was_prediction_correct THEN 1 ELSE 0 END) as correct_count,
		AVG(predicted_certainty) as avg_certainty,
		SUM(CASE WHEN was_prediction_correct THEN 1 ELSE 0 END)::float / COUNT(*) as accuracy
		FROM predictions_15m
		WHERE was_prediction_correct IS NOT NULL
		GROUP BY target_time_start, target_time_end
		ORDER BY target_time_start DESC
		LIMIT $1`

	rows, err := db.Query(r.Context(), query, limit)
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}
	defer rows.Close()

	type IntervalSummary struct {
		TargetTimeStart   time.Time `json:"target_time_start"`
		TargetTimeEnd     time.Time `json:"target_time_end"`
		OpenPrice         *float64  `json:"open_price,string,omitempty"`
		ClosePrice        *float64  `json:"close_price,string,omitempty"`
		ActualDirection   *string   `json:"actual_direction,omitempty"`
		PredictionCount   int       `json:"prediction_count"`
		CorrectCount      int       `json:"correct_count"`
		AvgCertainty      float64   `json:"avg_certainty"`
		Accuracy          float64   `json:"accuracy"`
	}

	intervals := []IntervalSummary{}
	for rows.Next() {
		var s IntervalSummary
		err := rows.Scan(
			&s.TargetTimeStart, &s.TargetTimeEnd,
			&s.OpenPrice, &s.ClosePrice,
			&s.ActualDirection,
			&s.PredictionCount, &s.CorrectCount,
			&s.AvgCertainty, &s.Accuracy,
		)
		if err != nil {
			respondJSON(w, http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Scan error: %v", err),
			})
			return
		}
		intervals = append(intervals, s)
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    intervals,
		Count:   len(intervals),
	})
}