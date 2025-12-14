// Package main provides configuration management for the gap handler service.
package main

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/joho/godotenv"
)

// Config holds all configuration for the gap handler service.
type Config struct {
	// Database configuration
	DBHost     string
	DBPort     int
	DBName     string
	DBUser     string
	DBPassword string
	DBMaxConns int

	// Binance API configuration
	BinanceAPIURL string
	Symbol        string
	Interval      string

	// Server configuration
	Port                   int
	MaxConcurrentBackfills int
	BackfillTimeout        time.Duration

	// Rate limiting
	RequestsPerSecond int
	BurstSize         int

	// Historical backfill configuration
	HistoricalRetentionMonths int           // How many months of data to maintain (default: 12)
	HistoricalBackfillEnabled bool          // Enable automatic historical backfill on startup
	HistoricalChunkDays       int           // Days per backfill chunk (default: 7, max allowed by API)

	// Logging
	LogLevel string
}

// LoadConfig loads configuration from environment variables.
func LoadConfig() (*Config, error) {
	// Load .env file if present (ignore errors if not found)
	_ = godotenv.Load()

	cfg := &Config{
		DBHost:                    getEnvOrDefault("DB_HOST", "timescaledb"),
		DBPort:                    getEnvIntOrDefault("DB_PORT", 5432),
		DBName:                    getEnvOrDefault("DB_NAME", "btc_ml_production"),
		DBUser:                    getEnvOrDefault("DB_USER", "mltrader"),
		DBPassword:                os.Getenv("DB_PASSWORD"),
		DBMaxConns:                getEnvIntOrDefault("DB_MAX_CONNS", 2),
		BinanceAPIURL:             getEnvOrDefault("BINANCE_API_URL", "https://api.binance.com"),
		Symbol:                    getEnvOrDefault("TRADING_SYMBOL", "BTCUSDT"),
		Interval:                  getEnvOrDefault("CANDLE_INTERVAL", "1m"),
		Port:                      getEnvIntOrDefault("GAP_HANDLER_PORT", 9000),
		MaxConcurrentBackfills:    getEnvIntOrDefault("MAX_CONCURRENT_BACKFILLS", 5),
		BackfillTimeout:           time.Duration(getEnvIntOrDefault("BACKFILL_TIMEOUT_SECONDS", 300)) * time.Second,
		RequestsPerSecond:         getEnvIntOrDefault("REQUESTS_PER_SECOND", 10),
		BurstSize:                 getEnvIntOrDefault("BURST_SIZE", 10),
		HistoricalRetentionMonths: getEnvIntOrDefault("HISTORICAL_RETENTION_MONTHS", 12),
		HistoricalBackfillEnabled: getEnvOrDefault("HISTORICAL_BACKFILL_ENABLED", "true") == "true",
		HistoricalChunkDays:       getEnvIntOrDefault("HISTORICAL_CHUNK_DAYS", 7),
		LogLevel:                  getEnvOrDefault("LOG_LEVEL", "info"),
	}

	// Validate required fields
	if cfg.DBPassword == "" {
		return nil, fmt.Errorf("DB_PASSWORD is required")
	}

	return cfg, nil
}

// DatabaseDSN returns the PostgreSQL connection string.
func (c *Config) DatabaseDSN() string {
	return fmt.Sprintf(
		"host=%s port=%d dbname=%s user=%s password=%s sslmode=disable",
		c.DBHost, c.DBPort, c.DBName, c.DBUser, c.DBPassword,
	)
}

// BinanceKlinesURL returns the full URL for the klines endpoint.
func (c *Config) BinanceKlinesURL() string {
	return fmt.Sprintf("%s/api/v3/klines", c.BinanceAPIURL)
}

// getEnvOrDefault returns the environment variable value or a default.
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvIntOrDefault returns the environment variable as int or a default.
func getEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}
