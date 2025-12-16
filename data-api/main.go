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

// Response structures
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

type LLMAnalysis struct {
	ID                   int64      `json:"id"`
	AnalysisTime         time.Time  `json:"analysis_time"`
	Price                float64    `json:"price,string"`
	PredictionDirection  string     `json:"prediction_direction"`
	PredictionConfidence string     `json:"prediction_confidence"`
	PredictedPrice1h     *float64   `json:"predicted_price_1h,string,omitempty"`
	PredictedPrice4h     *float64   `json:"predicted_price_4h,string,omitempty"`
	KeyLevels            *string    `json:"key_levels,omitempty"`
	Reasoning            *string    `json:"reasoning,omitempty"`
	FullResponse         *string    `json:"full_response,omitempty"`
	ModelName            string     `json:"model_name"`
	ResponseTimeSeconds  *float64   `json:"response_time_seconds,omitempty"`
	ActualPrice1h        *float64   `json:"actual_price_1h,string,omitempty"`
	ActualPrice4h        *float64   `json:"actual_price_4h,string,omitempty"`
	DirectionCorrect1h   *bool      `json:"direction_correct_1h,omitempty"`
	DirectionCorrect4h   *bool      `json:"direction_correct_4h,omitempty"`
	CreatedAt            time.Time  `json:"created_at"`
}

type MarketAnalysis struct {
    ID                   int64           `json:"id"`
    AnalysisTime         time.Time       `json:"analysis_time"`
    Price                float64         `json:"price,string"`
    SignalType           string          `json:"signal_type"`
    SignalDirection      string          `json:"signal_direction"`
    SignalConfidence     float64         `json:"signal_confidence"`
    
    // Trade setup
    EntryPrice           *float64        `json:"entry_price,string,omitempty"`
    StopLoss             *float64        `json:"stop_loss,string,omitempty"`
    TakeProfit1          *float64        `json:"take_profit_1,string,omitempty"`
    TakeProfit2          *float64        `json:"take_profit_2,string,omitempty"`
    TakeProfit3          *float64        `json:"take_profit_3,string,omitempty"`
    RiskRewardRatio      *float64        `json:"risk_reward_ratio,omitempty"`
    
    // Signal reasoning
    SignalFactors        json.RawMessage `json:"signal_factors,omitempty"`
    
    // Trends (enhanced)
    Trends               json.RawMessage `json:"trends"`
    
    // Complete pivot data
    PivotDaily           *float64        `json:"pivot_daily,string,omitempty"`
    PivotR3Traditional   *float64        `json:"pivot_r3_traditional,string,omitempty"`
    PivotR2Traditional   *float64        `json:"pivot_r2_traditional,string,omitempty"`
    PivotR1Traditional   *float64        `json:"pivot_r1_traditional,string,omitempty"`
    PivotS1Traditional   *float64        `json:"pivot_s1_traditional,string,omitempty"`
    PivotS2Traditional   *float64        `json:"pivot_s2_traditional,string,omitempty"`
    PivotS3Traditional   *float64        `json:"pivot_s3_traditional,string,omitempty"`
    PivotR3Fibonacci     *float64        `json:"pivot_r3_fibonacci,string,omitempty"`
    PivotR2Fibonacci     *float64        `json:"pivot_r2_fibonacci,string,omitempty"`
    PivotR1Fibonacci     *float64        `json:"pivot_r1_fibonacci,string,omitempty"`
    PivotS1Fibonacci     *float64        `json:"pivot_s1_fibonacci,string,omitempty"`
    PivotS2Fibonacci     *float64        `json:"pivot_s2_fibonacci,string,omitempty"`
    PivotS3Fibonacci     *float64        `json:"pivot_s3_fibonacci,string,omitempty"`
    PivotR4Camarilla     *float64        `json:"pivot_r4_camarilla,string,omitempty"`
    PivotR3Camarilla     *float64        `json:"pivot_r3_camarilla,string,omitempty"`
    PivotS3Camarilla     *float64        `json:"pivot_s3_camarilla,string,omitempty"`
    PivotS4Camarilla     *float64        `json:"pivot_s4_camarilla,string,omitempty"`
    PivotConfluenceZones json.RawMessage `json:"pivot_confluence_zones,omitempty"`
    PriceVsPivot         *string         `json:"price_vs_pivot,omitempty"`
    
    // SMC (complete)
    SMCBias              *string         `json:"smc_bias,omitempty"`
    SMCPriceZone         *string         `json:"smc_price_zone,omitempty"`
    SMCEquilibrium       *float64        `json:"smc_equilibrium,string,omitempty"`
    SMCOrderBlocks       json.RawMessage `json:"smc_order_blocks,omitempty"`
    SMCFVGs              json.RawMessage `json:"smc_fvgs,omitempty"`
    SMCBreaks            json.RawMessage `json:"smc_breaks,omitempty"`
    SMCLiquidity         json.RawMessage `json:"smc_liquidity,omitempty"`
    
    // Support/Resistance (all levels)
    SupportLevels        json.RawMessage `json:"support_levels,omitempty"`
    ResistanceLevels     json.RawMessage `json:"resistance_levels,omitempty"`
    
    // Momentum (all timeframes)
    Momentum             json.RawMessage `json:"momentum,omitempty"`
    
    // Market structure
    StructurePattern     *string         `json:"structure_pattern,omitempty"`
    StructureLastHigh    *float64        `json:"structure_last_high,string,omitempty"`
    StructureLastLow     *float64        `json:"structure_last_low,string,omitempty"`
    
    // Warnings
    Warnings             json.RawMessage `json:"warnings,omitempty"`
    ActionRecommendation *string         `json:"action_recommendation,omitempty"`
    
    // Summary
    Summary              *string         `json:"summary,omitempty"`
    SignalChanged        bool            `json:"signal_changed"`
    PreviousSignal       *string         `json:"previous_signal,omitempty"`
    CreatedAt            time.Time       `json:"created_at"`
}

type MarketSignal struct {
    ID                   int64           `json:"id"`
    SignalTime           time.Time       `json:"signal_time"`
    SignalType           string          `json:"signal_type"`
    SignalDirection      string          `json:"signal_direction"`
    SignalConfidence     float64         `json:"signal_confidence"`
    Price                float64         `json:"price,string"`
    EntryPrice           *float64        `json:"entry_price,string,omitempty"`
    StopLoss             *float64        `json:"stop_loss,string,omitempty"`
    TakeProfit1          *float64        `json:"take_profit_1,string,omitempty"`
    TakeProfit2          *float64        `json:"take_profit_2,string,omitempty"`
    TakeProfit3          *float64        `json:"take_profit_3,string,omitempty"`
    RiskRewardRatio      *float64        `json:"risk_reward_ratio,omitempty"`
    PreviousSignalType   *string         `json:"previous_signal_type,omitempty"`
    PreviousDirection    *string         `json:"previous_direction,omitempty"`
    Summary              *string         `json:"summary,omitempty"`
    KeyReasons           json.RawMessage `json:"key_reasons,omitempty"` // Changed from TEXT[] to JSONB
    CreatedAt            time.Time       `json:"created_at"`
}

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

// apiKeyMiddleware validates API key in Authorization header
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

	query := `SELECT id, analysis_time, price, prediction_direction, prediction_confidence,
		predicted_price_1h, predicted_price_4h, key_levels, reasoning, full_response,
		model_name, response_time_seconds, actual_price_1h, actual_price_4h,
		direction_correct_1h, direction_correct_4h, created_at
		FROM llm_analysis
		ORDER BY id DESC, analysis_time DESC
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

	analyses := []LLMAnalysis{}
	for rows.Next() {
		var a LLMAnalysis
		err := rows.Scan(&a.ID, &a.AnalysisTime, &a.Price, &a.PredictionDirection,
			&a.PredictionConfidence, &a.PredictedPrice1h, &a.PredictedPrice4h,
			&a.KeyLevels, &a.Reasoning, &a.FullResponse, &a.ModelName,
			&a.ResponseTimeSeconds, &a.ActualPrice1h, &a.ActualPrice4h,
			&a.DirectionCorrect1h, &a.DirectionCorrect4h, &a.CreatedAt)
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

func getMarketAnalysisHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 50)
	if limit > 1000 {
		limit = 1000
	}

	signalType := r.URL.Query().Get("signal_type")

	query := `SELECT 
		id, analysis_time, price, signal_type, signal_direction, signal_confidence,
		entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
		signal_factors, trends,
		pivot_daily, pivot_r3_traditional, pivot_r2_traditional, pivot_r1_traditional,
		pivot_s1_traditional, pivot_s2_traditional, pivot_s3_traditional,
		pivot_r3_fibonacci, pivot_r2_fibonacci, pivot_r1_fibonacci,
		pivot_s1_fibonacci, pivot_s2_fibonacci, pivot_s3_fibonacci,
		pivot_r4_camarilla, pivot_r3_camarilla, pivot_s3_camarilla, pivot_s4_camarilla,
		pivot_confluence_zones, price_vs_pivot,
		smc_bias, smc_price_zone, smc_equilibrium,
		smc_order_blocks, smc_fvgs, smc_breaks, smc_liquidity,
		support_levels, resistance_levels, momentum,
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
		err := rows.Scan(&a.ID, &a.AnalysisTime, &a.Price, &a.SignalType, &a.SignalDirection,
			&a.SignalConfidence, &a.EntryPrice, &a.StopLoss, &a.TakeProfit1, &a.TakeProfit2,
			&a.TakeProfit3, &a.RiskRewardRatio, &a.Trends, &a.NearestSupport, &a.NearestResistance,
			&a.SupportStrength, &a.ResistanceStrength, &a.SMCBias, &a.PriceZone,
			&a.EquilibriumPrice, &a.DailyPivot, &a.PriceVsPivot, &a.RSI1h, &a.VolumeRatio1h,
			&a.Summary, &a.SignalChanged, &a.PreviousSignal, &a.CreatedAt)
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

	query := `SELECT id, signal_time, signal_type, signal_direction, signal_confidence, price,
		entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
		previous_signal_type, previous_direction, summary, key_reasons, created_at
		FROM market_signals
		ORDER BY id DESC, signal_time DESC
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

	signals := []MarketSignal{}
	for rows.Next() {
		var s MarketSignal
		err := rows.Scan(&s.ID, &s.SignalTime, &s.SignalType, &s.SignalDirection,
			&s.SignalConfidence, &s.Price, &s.EntryPrice, &s.StopLoss, &s.TakeProfit1,
			&s.TakeProfit2, &s.TakeProfit3, &s.RiskRewardRatio, &s.PreviousSignalType,
			&s.PreviousDirection, &s.Summary, &s.KeyReasons, &s.CreatedAt)
		if err != nil {
			respondJSON(w, http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Scan error: %v", err),
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