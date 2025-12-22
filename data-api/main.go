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

// MarketAnalysis - Complete enhanced schema v3.0 with all pivot methods
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

	// Pivot - Traditional (complete)
	PivotDaily         *float64 `json:"pivot_daily,string,omitempty"`
	PivotR1Traditional *float64 `json:"pivot_r1_traditional,string,omitempty"`
	PivotR2Traditional *float64 `json:"pivot_r2_traditional,string,omitempty"`
	PivotR3Traditional *float64 `json:"pivot_r3_traditional,string,omitempty"`
	PivotS1Traditional *float64 `json:"pivot_s1_traditional,string,omitempty"`
	PivotS2Traditional *float64 `json:"pivot_s2_traditional,string,omitempty"`
	PivotS3Traditional *float64 `json:"pivot_s3_traditional,string,omitempty"`

	// Pivot - Fibonacci (complete)
	PivotR1Fibonacci *float64 `json:"pivot_r1_fibonacci,string,omitempty"`
	PivotR2Fibonacci *float64 `json:"pivot_r2_fibonacci,string,omitempty"`
	PivotR3Fibonacci *float64 `json:"pivot_r3_fibonacci,string,omitempty"`
	PivotS1Fibonacci *float64 `json:"pivot_s1_fibonacci,string,omitempty"`
	PivotS2Fibonacci *float64 `json:"pivot_s2_fibonacci,string,omitempty"`
	PivotS3Fibonacci *float64 `json:"pivot_s3_fibonacci,string,omitempty"`

	// Pivot - Camarilla (complete: R1-R4, S1-S4)
	PivotCamarilla   *float64 `json:"pivot_camarilla,string,omitempty"`
	PivotR1Camarilla *float64 `json:"pivot_r1_camarilla,string,omitempty"`
	PivotR2Camarilla *float64 `json:"pivot_r2_camarilla,string,omitempty"`
	PivotR3Camarilla *float64 `json:"pivot_r3_camarilla,string,omitempty"`
	PivotR4Camarilla *float64 `json:"pivot_r4_camarilla,string,omitempty"`
	PivotS1Camarilla *float64 `json:"pivot_s1_camarilla,string,omitempty"`
	PivotS2Camarilla *float64 `json:"pivot_s2_camarilla,string,omitempty"`
	PivotS3Camarilla *float64 `json:"pivot_s3_camarilla,string,omitempty"`
	PivotS4Camarilla *float64 `json:"pivot_s4_camarilla,string,omitempty"`

	// Pivot - Woodie (complete)
	PivotWoodie   *float64 `json:"pivot_woodie,string,omitempty"`
	PivotR1Woodie *float64 `json:"pivot_r1_woodie,string,omitempty"`
	PivotR2Woodie *float64 `json:"pivot_r2_woodie,string,omitempty"`
	PivotR3Woodie *float64 `json:"pivot_r3_woodie,string,omitempty"`
	PivotS1Woodie *float64 `json:"pivot_s1_woodie,string,omitempty"`
	PivotS2Woodie *float64 `json:"pivot_s2_woodie,string,omitempty"`
	PivotS3Woodie *float64 `json:"pivot_s3_woodie,string,omitempty"`

	// Pivot - DeMark (complete)
	PivotDeMark   *float64 `json:"pivot_demark,string,omitempty"`
	PivotR1DeMark *float64 `json:"pivot_r1_demark,string,omitempty"`
	PivotS1DeMark *float64 `json:"pivot_s1_demark,string,omitempty"`

	// Pivot confluence
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
	ID               int64           `json:"id"`
	SignalTime       time.Time       `json:"signal_time"`
	SignalType       string          `json:"signal_type"`
	SignalDirection  string          `json:"signal_direction"`
	SignalConfidence float64         `json:"signal_confidence"`
	Price            float64         `json:"price,string"`
	EntryPrice       *float64        `json:"entry_price,string,omitempty"`
	StopLoss         *float64        `json:"stop_loss,string,omitempty"`
	TakeProfit1      *float64        `json:"take_profit_1,string,omitempty"`
	TakeProfit2      *float64        `json:"take_profit_2,string,omitempty"`
	TakeProfit3      *float64        `json:"take_profit_3,string,omitempty"`
	RiskRewardRatio  *float64        `json:"risk_reward_ratio,omitempty"`
	PreviousSignalType *string       `json:"previous_signal_type,omitempty"`
	PreviousDirection  *string       `json:"previous_direction,omitempty"`
	Summary          *string         `json:"summary,omitempty"`
	KeyReasons       json.RawMessage `json:"key_reasons,omitempty"`
	SignalFactors    json.RawMessage `json:"signal_factors,omitempty"`
	SMCBias          *string         `json:"smc_bias,omitempty"`
	PivotDaily       *float64        `json:"pivot_daily,string,omitempty"`
	NearestSupport   *float64        `json:"nearest_support,string,omitempty"`
	NearestResistance *float64       `json:"nearest_resistance,string,omitempty"`
	CreatedAt        time.Time       `json:"created_at"`
}

// ============================================================================
// Main
// ============================================================================

func main() {
	logger = log.New(os.Stdout, "[DATA-API] ", log.LstdFlags)
	logger.Println("Starting Data API v3.0 (Complete Pivots)...")

	// Get config from environment
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://postgres:postgres@localhost:5432/market_data"
	}
	apiKey = os.Getenv("API_KEY")

	// Connect to database
	var err error
	config, err := pgxpool.ParseConfig(dbURL)
	if err != nil {
		logger.Fatalf("Failed to parse database URL: %v", err)
	}

	// Disable TLS for local development
	config.ConnConfig.TLSConfig = &tls.Config{InsecureSkipVerify: true}

	db, err = pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		logger.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// Test connection
	if err := db.Ping(context.Background()); err != nil {
		logger.Fatalf("Failed to ping database: %v", err)
	}
	logger.Println("Connected to database")

	// Setup router
	r := mux.NewRouter()

	// API routes
	api := r.PathPrefix("/api/v1").Subrouter()
	api.HandleFunc("/health", healthHandler).Methods("GET")
	api.HandleFunc("/candles", getCandlesHandler).Methods("GET")
	api.HandleFunc("/candles/latest", getLatestCandleHandler).Methods("GET")
	api.HandleFunc("/candles/range", getCandlesRangeHandler).Methods("GET")
	api.HandleFunc("/data-quality", getDataQualityHandler).Methods("GET")
	api.HandleFunc("/llm-analysis", getLLMAnalysisHandler).Methods("GET")
	api.HandleFunc("/llm-analysis/latest", getLatestLLMAnalysisHandler).Methods("GET")
	api.HandleFunc("/market-analysis", getMarketAnalysisHandler).Methods("GET")
	api.HandleFunc("/market-analysis/latest", getLatestMarketAnalysisHandler).Methods("GET")
	api.HandleFunc("/market-signals", getMarketSignalsHandler).Methods("GET")
	api.HandleFunc("/pivots/complete", getCompletePivotsHandler).Methods("GET")

	// CORS
	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"*"},
		AllowCredentials: true,
	})

	// Server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	server := &http.Server{
		Addr:         ":" + port,
		Handler:      c.Handler(r),
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
	}

	// Graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		logger.Println("Shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		server.Shutdown(ctx)
	}()

	logger.Printf("Listening on port %s", port)
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		logger.Fatalf("Server error: %v", err)
	}
}

// ============================================================================
// Handlers
// ============================================================================

func healthHandler(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    map[string]string{"status": "healthy", "version": "3.0"},
	})
}

func getCandlesHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 100)
	if limit > 10000 {
		limit = 10000
	}

	rows, err := db.Query(r.Context(),
		`SELECT time, open, high, low, close, volume,
		        quote_asset_volume, taker_buy_base_asset_volume,
		        taker_buy_quote_asset_volume, number_of_trades,
		        spread_bps, taker_buy_ratio, mid_price
		 FROM candles_1m
		 ORDER BY time DESC
		 LIMIT $1`, limit)
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
			&c.QuoteAssetVolume, &c.TakerBuyBaseAssetVolume,
			&c.TakerBuyQuoteAssetVolume, &c.NumberOfTrades,
			&c.SpreadBps, &c.TakerBuyRatio, &c.MidPrice)
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

func getLatestCandleHandler(w http.ResponseWriter, r *http.Request) {
	var c Candle
	err := db.QueryRow(r.Context(),
		`SELECT time, open, high, low, close, volume,
		        quote_asset_volume, taker_buy_base_asset_volume,
		        taker_buy_quote_asset_volume, number_of_trades,
		        spread_bps, taker_buy_ratio, mid_price
		 FROM candles_1m
		 ORDER BY time DESC
		 LIMIT 1`).Scan(&c.Time, &c.Open, &c.High, &c.Low, &c.Close, &c.Volume,
		&c.QuoteAssetVolume, &c.TakerBuyBaseAssetVolume,
		&c.TakerBuyQuoteAssetVolume, &c.NumberOfTrades,
		&c.SpreadBps, &c.TakerBuyRatio, &c.MidPrice)
	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    c,
	})
}

func getCandlesRangeHandler(w http.ResponseWriter, r *http.Request) {
	startStr := r.URL.Query().Get("start")
	endStr := r.URL.Query().Get("end")

	if startStr == "" || endStr == "" {
		respondJSON(w, http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "start and end parameters required (ISO format)",
		})
		return
	}

	start, err := time.Parse(time.RFC3339, startStr)
	if err != nil {
		respondJSON(w, http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Invalid start time format",
		})
		return
	}

	end, err := time.Parse(time.RFC3339, endStr)
	if err != nil {
		respondJSON(w, http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Invalid end time format",
		})
		return
	}

	rows, err := db.Query(r.Context(),
		`SELECT time, open, high, low, close, volume
		 FROM candles_1m
		 WHERE time >= $1 AND time <= $2
		 ORDER BY time ASC`, start, end)
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
		err := rows.Scan(&c.Time, &c.Open, &c.High, &c.Low, &c.Close, &c.Volume)
		if err != nil {
			continue
		}
		candles = append(candles, c)
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    candles,
		Count:   len(candles),
	})
}

func getDataQualityHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 50)
	if limit > 500 {
		limit = 500
	}

	rows, err := db.Query(r.Context(),
		`SELECT id, event_type, gap_start, gap_end, candles_missing,
		        candles_recovered, source, error_message, resolved,
		        resolved_at, created_at
		 FROM data_quality_log
		 ORDER BY created_at DESC
		 LIMIT $1`, limit)
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
		err := rows.Scan(&l.ID, &l.EventType, &l.GapStart, &l.GapEnd,
			&l.CandlesMissing, &l.CandlesRecovered, &l.Source,
			&l.ErrorMessage, &l.Resolved, &l.ResolvedAt, &l.CreatedAt)
		if err != nil {
			continue
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
	if limit > 500 {
		limit = 500
	}

	// Check for enhanced schema
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

func getLatestLLMAnalysisHandler(w http.ResponseWriter, r *http.Request) {
	var a LLMAnalysis
	hasEnhanced := checkColumnExists(r.Context(), "llm_analysis", "market_context")

	var err error
	if hasEnhanced {
		err = db.QueryRow(r.Context(),
			`SELECT id, analysis_time, price, prediction_direction, prediction_confidence,
				predicted_price_1h, predicted_price_4h, key_levels, reasoning, full_response,
				model_name, response_time_seconds, actual_price_1h, actual_price_4h,
				direction_correct_1h, direction_correct_4h,
				invalidation_level, critical_support, critical_resistance,
				market_context, signal_factors_used, smc_bias_at_analysis,
				trends_at_analysis, warnings_at_analysis, created_at
				FROM llm_analysis
				ORDER BY analysis_time DESC
				LIMIT 1`).Scan(&a.ID, &a.AnalysisTime, &a.Price, &a.PredictionDirection,
			&a.PredictionConfidence, &a.PredictedPrice1h, &a.PredictedPrice4h,
			&a.KeyLevels, &a.Reasoning, &a.FullResponse, &a.ModelName,
			&a.ResponseTimeSeconds, &a.ActualPrice1h, &a.ActualPrice4h,
			&a.DirectionCorrect1h, &a.DirectionCorrect4h,
			&a.InvalidationLevel, &a.CriticalSupport, &a.CriticalResistance,
			&a.MarketContext, &a.SignalFactorsUsed, &a.SMCBiasAtAnalysis,
			&a.TrendsAtAnalysis, &a.WarningsAtAnalysis, &a.CreatedAt)
	} else {
		err = db.QueryRow(r.Context(),
			`SELECT id, analysis_time, price, prediction_direction, prediction_confidence,
				predicted_price_1h, predicted_price_4h, key_levels, reasoning, full_response,
				model_name, response_time_seconds, actual_price_1h, actual_price_4h,
				direction_correct_1h, direction_correct_4h, created_at
				FROM llm_analysis
				ORDER BY analysis_time DESC
				LIMIT 1`).Scan(&a.ID, &a.AnalysisTime, &a.Price, &a.PredictionDirection,
			&a.PredictionConfidence, &a.PredictedPrice1h, &a.PredictedPrice4h,
			&a.KeyLevels, &a.Reasoning, &a.FullResponse, &a.ModelName,
			&a.ResponseTimeSeconds, &a.ActualPrice1h, &a.ActualPrice4h,
			&a.DirectionCorrect1h, &a.DirectionCorrect4h, &a.CreatedAt)
	}

	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    a,
	})
}

func getLatestMarketAnalysisHandler(w http.ResponseWriter, r *http.Request) {
	var a MarketAnalysis

	// Query with all pivot columns
	err := db.QueryRow(r.Context(),
		`SELECT id, analysis_time, price, signal_type, signal_direction, signal_confidence,
			entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
			signal_factors, trends,
			nearest_support, nearest_resistance, support_strength, resistance_strength,
			support_levels, resistance_levels,
			smc_bias, price_zone, equilibrium_price,
			smc_price_zone, smc_equilibrium, smc_order_blocks, smc_fvgs, smc_breaks, smc_liquidity,
			daily_pivot, price_vs_pivot,
			pivot_daily,
			COALESCE(pivot_r1_traditional, 0), COALESCE(pivot_r2_traditional, 0), COALESCE(pivot_r3_traditional, 0),
			COALESCE(pivot_s1_traditional, 0), COALESCE(pivot_s2_traditional, 0), COALESCE(pivot_s3_traditional, 0),
			COALESCE(pivot_r1_fibonacci, 0), COALESCE(pivot_r2_fibonacci, 0), COALESCE(pivot_r3_fibonacci, 0),
			COALESCE(pivot_s1_fibonacci, 0), COALESCE(pivot_s2_fibonacci, 0), COALESCE(pivot_s3_fibonacci, 0),
			COALESCE(pivot_camarilla, 0),
			COALESCE(pivot_r1_camarilla, 0), COALESCE(pivot_r2_camarilla, 0), COALESCE(pivot_r3_camarilla, 0), COALESCE(pivot_r4_camarilla, 0),
			COALESCE(pivot_s1_camarilla, 0), COALESCE(pivot_s2_camarilla, 0), COALESCE(pivot_s3_camarilla, 0), COALESCE(pivot_s4_camarilla, 0),
			COALESCE(pivot_woodie, 0),
			COALESCE(pivot_r1_woodie, 0), COALESCE(pivot_r2_woodie, 0), COALESCE(pivot_r3_woodie, 0),
			COALESCE(pivot_s1_woodie, 0), COALESCE(pivot_s2_woodie, 0), COALESCE(pivot_s3_woodie, 0),
			COALESCE(pivot_demark, 0),
			COALESCE(pivot_r1_demark, 0), COALESCE(pivot_s1_demark, 0),
			pivot_confluence_zones,
			rsi_1h, volume_ratio_1h, momentum,
			structure_pattern, structure_last_high, structure_last_low,
			warnings, action_recommendation,
			summary, signal_changed, previous_signal, created_at
		FROM market_analysis
		ORDER BY analysis_time DESC
		LIMIT 1`).Scan(
		&a.ID, &a.AnalysisTime, &a.Price, &a.SignalType, &a.SignalDirection, &a.SignalConfidence,
		&a.EntryPrice, &a.StopLoss, &a.TakeProfit1, &a.TakeProfit2, &a.TakeProfit3, &a.RiskRewardRatio,
		&a.SignalFactors, &a.Trends,
		&a.NearestSupport, &a.NearestResistance, &a.SupportStrength, &a.ResistanceStrength,
		&a.SupportLevels, &a.ResistanceLevels,
		&a.SMCBias, &a.PriceZone, &a.EquilibriumPrice,
		&a.SMCPriceZone, &a.SMCEquilibrium, &a.SMCOrderBlocks, &a.SMCFVGs, &a.SMCBreaks, &a.SMCLiquidity,
		&a.DailyPivot, &a.PriceVsPivot,
		&a.PivotDaily,
		&a.PivotR1Traditional, &a.PivotR2Traditional, &a.PivotR3Traditional,
		&a.PivotS1Traditional, &a.PivotS2Traditional, &a.PivotS3Traditional,
		&a.PivotR1Fibonacci, &a.PivotR2Fibonacci, &a.PivotR3Fibonacci,
		&a.PivotS1Fibonacci, &a.PivotS2Fibonacci, &a.PivotS3Fibonacci,
		&a.PivotCamarilla,
		&a.PivotR1Camarilla, &a.PivotR2Camarilla, &a.PivotR3Camarilla, &a.PivotR4Camarilla,
		&a.PivotS1Camarilla, &a.PivotS2Camarilla, &a.PivotS3Camarilla, &a.PivotS4Camarilla,
		&a.PivotWoodie,
		&a.PivotR1Woodie, &a.PivotR2Woodie, &a.PivotR3Woodie,
		&a.PivotS1Woodie, &a.PivotS2Woodie, &a.PivotS3Woodie,
		&a.PivotDeMark,
		&a.PivotR1DeMark, &a.PivotS1DeMark,
		&a.PivotConfluenceZones,
		&a.RSI1h, &a.VolumeRatio1h, &a.Momentum,
		&a.StructurePattern, &a.StructureLastHigh, &a.StructureLastLow,
		&a.Warnings, &a.ActionRecommendation,
		&a.Summary, &a.SignalChanged, &a.PreviousSignal, &a.CreatedAt,
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
		Data:    a,
	})
}

func getMarketAnalysisHandler(w http.ResponseWriter, r *http.Request) {
	limit := parseInt(r.URL.Query().Get("limit"), 100)
	if limit > 1000 {
		limit = 1000
	}
	signalType := r.URL.Query().Get("signal_type")

	query := `SELECT 
		id, analysis_time, price, signal_type, signal_direction, signal_confidence,
		entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
		signal_factors, trends,
		nearest_support, nearest_resistance, support_strength, resistance_strength,
		support_levels, resistance_levels,
		smc_bias, price_zone, equilibrium_price,
		smc_price_zone, smc_equilibrium, smc_order_blocks, smc_fvgs, smc_breaks, smc_liquidity,
		daily_pivot, price_vs_pivot,
		pivot_daily,
		pivot_r1_traditional, pivot_r2_traditional, pivot_r3_traditional,
		pivot_s1_traditional, pivot_s2_traditional, pivot_s3_traditional,
		pivot_r1_fibonacci, pivot_r2_fibonacci, pivot_r3_fibonacci,
		pivot_s1_fibonacci, pivot_s2_fibonacci, pivot_s3_fibonacci,
		pivot_camarilla,
		pivot_r1_camarilla, pivot_r2_camarilla, pivot_r3_camarilla, pivot_r4_camarilla,
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
			&a.PivotDaily,
			&a.PivotR1Traditional, &a.PivotR2Traditional, &a.PivotR3Traditional,
			&a.PivotS1Traditional, &a.PivotS2Traditional, &a.PivotS3Traditional,
			&a.PivotR1Fibonacci, &a.PivotR2Fibonacci, &a.PivotR3Fibonacci,
			&a.PivotS1Fibonacci, &a.PivotS2Fibonacci, &a.PivotS3Fibonacci,
			&a.PivotCamarilla,
			&a.PivotR1Camarilla, &a.PivotR2Camarilla, &a.PivotR3Camarilla, &a.PivotR4Camarilla,
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

	query := `SELECT id, signal_time, signal_type, signal_direction, signal_confidence, price,
		entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, risk_reward_ratio,
		previous_signal_type, previous_direction, summary, key_reasons,
		signal_factors, smc_bias, pivot_daily, nearest_support, nearest_resistance,
		created_at
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
			&s.PreviousDirection, &s.Summary, &s.KeyReasons,
			&s.SignalFactors, &s.SMCBias, &s.PivotDaily, &s.NearestSupport, &s.NearestResistance,
			&s.CreatedAt)

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

// getCompletePivotsHandler returns complete pivot data for all 5 methods
func getCompletePivotsHandler(w http.ResponseWriter, r *http.Request) {
	type PivotSet struct {
		Method string             `json:"method"`
		Pivot  *float64           `json:"pivot,string,omitempty"`
		R1     *float64           `json:"r1,string,omitempty"`
		R2     *float64           `json:"r2,string,omitempty"`
		R3     *float64           `json:"r3,string,omitempty"`
		R4     *float64           `json:"r4,string,omitempty"`
		S1     *float64           `json:"s1,string,omitempty"`
		S2     *float64           `json:"s2,string,omitempty"`
		S3     *float64           `json:"s3,string,omitempty"`
		S4     *float64           `json:"s4,string,omitempty"`
	}

	type CompletePivots struct {
		AnalysisTime  time.Time       `json:"analysis_time"`
		Price         float64         `json:"price,string"`
		PriceVsPivot  *string         `json:"price_vs_pivot,omitempty"`
		Traditional   PivotSet        `json:"traditional"`
		Fibonacci     PivotSet        `json:"fibonacci"`
		Camarilla     PivotSet        `json:"camarilla"`
		Woodie        PivotSet        `json:"woodie"`
		DeMark        PivotSet        `json:"demark"`
		Confluence    json.RawMessage `json:"confluence_zones,omitempty"`
	}

	var cp CompletePivots
	err := db.QueryRow(r.Context(),
		`SELECT 
			analysis_time, price, price_vs_pivot,
			pivot_daily, pivot_r1_traditional, pivot_r2_traditional, pivot_r3_traditional,
			pivot_s1_traditional, pivot_s2_traditional, pivot_s3_traditional,
			pivot_r1_fibonacci, pivot_r2_fibonacci, pivot_r3_fibonacci,
			pivot_s1_fibonacci, pivot_s2_fibonacci, pivot_s3_fibonacci,
			pivot_camarilla, pivot_r1_camarilla, pivot_r2_camarilla, pivot_r3_camarilla, pivot_r4_camarilla,
			pivot_s1_camarilla, pivot_s2_camarilla, pivot_s3_camarilla, pivot_s4_camarilla,
			pivot_woodie, pivot_r1_woodie, pivot_r2_woodie, pivot_r3_woodie,
			pivot_s1_woodie, pivot_s2_woodie, pivot_s3_woodie,
			pivot_demark, pivot_r1_demark, pivot_s1_demark,
			pivot_confluence_zones
		FROM market_analysis
		ORDER BY analysis_time DESC
		LIMIT 1`).Scan(
		&cp.AnalysisTime, &cp.Price, &cp.PriceVsPivot,
		&cp.Traditional.Pivot, &cp.Traditional.R1, &cp.Traditional.R2, &cp.Traditional.R3,
		&cp.Traditional.S1, &cp.Traditional.S2, &cp.Traditional.S3,
		&cp.Fibonacci.R1, &cp.Fibonacci.R2, &cp.Fibonacci.R3,
		&cp.Fibonacci.S1, &cp.Fibonacci.S2, &cp.Fibonacci.S3,
		&cp.Camarilla.Pivot, &cp.Camarilla.R1, &cp.Camarilla.R2, &cp.Camarilla.R3, &cp.Camarilla.R4,
		&cp.Camarilla.S1, &cp.Camarilla.S2, &cp.Camarilla.S3, &cp.Camarilla.S4,
		&cp.Woodie.Pivot, &cp.Woodie.R1, &cp.Woodie.R2, &cp.Woodie.R3,
		&cp.Woodie.S1, &cp.Woodie.S2, &cp.Woodie.S3,
		&cp.DeMark.Pivot, &cp.DeMark.R1, &cp.DeMark.S1,
		&cp.Confluence,
	)

	if err != nil {
		respondJSON(w, http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   fmt.Sprintf("Query error: %v", err),
		})
		return
	}

	// Set method names
	cp.Traditional.Method = "Traditional"
	cp.Fibonacci.Method = "Fibonacci"
	cp.Fibonacci.Pivot = cp.Traditional.Pivot // Fib uses same pivot
	cp.Camarilla.Method = "Camarilla"
	cp.Woodie.Method = "Woodie"
	cp.DeMark.Method = "DeMark"

	respondJSON(w, http.StatusOK, APIResponse{
		Success: true,
		Data:    cp,
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