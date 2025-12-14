// Package main provides a Binance API client for fetching kline data.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/go-resty/resty/v2"
	"go.uber.org/zap"
)

// BinanceKline represents a single candlestick from Binance API.
type BinanceKline struct {
	OpenTime                 int64
	Open                     float64
	High                     float64
	Low                      float64
	Close                    float64
	Volume                   float64
	CloseTime                int64
	QuoteAssetVolume         float64
	NumberOfTrades           int64
	TakerBuyBaseAssetVolume  float64
	TakerBuyQuoteAssetVolume float64
}

// BinanceAPIClient handles communication with the Binance API.
type BinanceAPIClient struct {
	client      *resty.Client
	baseURL     string
	symbol      string
	interval    string
	rateLimiter *RateLimiter
	logger      *zap.Logger
	backoff     BackoffStrategy
}

// NewBinanceAPIClient creates a new Binance API client.
func NewBinanceAPIClient(cfg *Config, rateLimiter *RateLimiter, logger *zap.Logger) *BinanceAPIClient {
	client := resty.New().
		SetTimeout(30 * time.Second).
		SetRetryCount(0). // We handle retries ourselves
		SetHeader("Accept", "application/json")

	return &BinanceAPIClient{
		client:      client,
		baseURL:     cfg.BinanceKlinesURL(),
		symbol:      cfg.Symbol,
		interval:    cfg.Interval,
		rateLimiter: rateLimiter,
		logger:      logger,
		backoff:     DefaultBackoffStrategy(),
	}
}

// FetchKlines fetches klines from Binance for the specified time range.
// startTime and endTime are Unix timestamps in milliseconds.
// limit is the maximum number of candles to fetch (max 1000).
func (c *BinanceAPIClient) FetchKlines(ctx context.Context, startTime, endTime int64, limit int) ([]BinanceKline, error) {
	if limit > 1000 {
		limit = 1000
	}

	// Wait for rate limiter
	if err := c.rateLimiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limiter wait: %w", err)
	}

	startFetch := time.Now()

	resp, err := c.client.R().
		SetContext(ctx).
		SetQueryParams(map[string]string{
			"symbol":    c.symbol,
			"interval":  c.interval,
			"startTime": strconv.FormatInt(startTime, 10),
			"endTime":   strconv.FormatInt(endTime, 10),
			"limit":     strconv.Itoa(limit),
		}).
		Get(c.baseURL)

	latency := time.Since(startFetch).Seconds()

	if err != nil {
		metricsAPIErrors.Inc()
		RecordAPIRequest("error", latency)
		c.rateLimiter.RecordFailure(0, c.backoff)
		return nil, fmt.Errorf("API request failed: %w", err)
	}

	statusCode := resp.StatusCode()

	// Handle rate limiting and errors
	if statusCode == 429 || statusCode == 418 {
		c.rateLimiter.RecordFailure(statusCode, c.backoff)
		RecordAPIRequest("rate_limited", latency)
		return nil, fmt.Errorf("rate limited: status %d", statusCode)
	}

	if statusCode >= 500 {
		c.rateLimiter.RecordFailure(statusCode, c.backoff)
		RecordAPIRequest("server_error", latency)
		metricsAPIErrors.Inc()
		return nil, fmt.Errorf("server error: status %d", statusCode)
	}

	if statusCode != 200 {
		c.rateLimiter.RecordFailure(statusCode, c.backoff)
		RecordAPIRequest("error", latency)
		metricsAPIErrors.Inc()
		return nil, fmt.Errorf("unexpected status: %d, body: %s", statusCode, string(resp.Body()))
	}

	// Record success
	c.rateLimiter.RecordSuccess()
	RecordAPIRequest("success", latency)
	metricsBinanceAPIStatus.Set(1)

	// Parse response
	klines, err := parseKlinesResponse(resp.Body())
	if err != nil {
		metricsAPIErrors.Inc()
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	c.logger.Debug("Fetched klines from Binance",
		zap.Int("count", len(klines)),
		zap.Int64("start_time", startTime),
		zap.Int64("end_time", endTime),
		zap.Float64("latency_seconds", latency),
	)

	return klines, nil
}

// FetchKlinesWithRetry fetches klines with retry logic.
func (c *BinanceAPIClient) FetchKlinesWithRetry(ctx context.Context, startTime, endTime int64, limit int) ([]BinanceKline, error) {
	maxRetries := 3
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 1s, 2s, 4s
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			c.logger.Info("Retrying API request",
				zap.Int("attempt", attempt+1),
				zap.Duration("backoff", backoff),
			)

			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		klines, err := c.FetchKlines(ctx, startTime, endTime, limit)
		if err == nil {
			return klines, nil
		}

		lastErr = err
		c.logger.Warn("API request failed",
			zap.Int("attempt", attempt+1),
			zap.Error(err),
		)
	}

	return nil, fmt.Errorf("all retries failed: %w", lastErr)
}

// FetchAllKlines fetches all klines for a time range, handling pagination.
func (c *BinanceAPIClient) FetchAllKlines(ctx context.Context, startTime, endTime time.Time) ([]BinanceKline, error) {
	var allKlines []BinanceKline

	startMs := startTime.UnixMilli()
	endMs := endTime.UnixMilli()
	currentStart := startMs

	for currentStart < endMs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		klines, err := c.FetchKlinesWithRetry(ctx, currentStart, endMs, 1000)
		if err != nil {
			return allKlines, fmt.Errorf("fetch klines: %w", err)
		}

		if len(klines) == 0 {
			break
		}

		allKlines = append(allKlines, klines...)

		// Move start to after the last candle
		lastKline := klines[len(klines)-1]
		currentStart = lastKline.CloseTime + 1

		c.logger.Debug("Pagination progress",
			zap.Int("batch_size", len(klines)),
			zap.Int("total_fetched", len(allKlines)),
			zap.Int64("next_start", currentStart),
		)

		// Stop if we've fetched all candles
		if len(klines) < 1000 {
			break
		}
	}

	return allKlines, nil
}

// Ping tests connectivity to the Binance API.
func (c *BinanceAPIClient) Ping(ctx context.Context) error {
	resp, err := c.client.R().
		SetContext(ctx).
		Get(c.baseURL[:len(c.baseURL)-len("/api/v3/klines")] + "/api/v3/ping")

	if err != nil {
		metricsBinanceAPIStatus.Set(0)
		return err
	}

	if resp.StatusCode() != 200 {
		metricsBinanceAPIStatus.Set(0)
		return fmt.Errorf("ping failed: status %d", resp.StatusCode())
	}

	metricsBinanceAPIStatus.Set(1)
	return nil
}

// parseKlinesResponse parses the raw JSON response from Binance.
// Binance returns klines as arrays: [[openTime, open, high, low, close, volume, closeTime, ...], ...]
func parseKlinesResponse(body []byte) ([]BinanceKline, error) {
	var rawKlines [][]interface{}
	if err := json.Unmarshal(body, &rawKlines); err != nil {
		return nil, fmt.Errorf("json unmarshal: %w", err)
	}

	klines := make([]BinanceKline, 0, len(rawKlines))

	for i, raw := range rawKlines {
		if len(raw) < 12 {
			continue // Skip malformed entries
		}

		kline, err := parseKlineArray(raw)
		if err != nil {
			// Log and skip malformed entries
			continue
		}

		// Validate kline data
		if kline.Open <= 0 || kline.High <= 0 || kline.Low <= 0 || kline.Close <= 0 {
			continue // Skip invalid price data
		}

		if kline.High < kline.Low {
			continue // Invalid OHLC relationship
		}

		klines = append(klines, kline)

		_ = i // Suppress unused warning
	}

	return klines, nil
}

// parseKlineArray parses a single kline array from Binance.
func parseKlineArray(raw []interface{}) (BinanceKline, error) {
	kline := BinanceKline{}

	// OpenTime (index 0) - int64
	if v, ok := raw[0].(float64); ok {
		kline.OpenTime = int64(v)
	} else {
		return kline, fmt.Errorf("invalid open_time")
	}

	// Open (index 1) - string
	if v, ok := raw[1].(string); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			kline.Open = f
		}
	}

	// High (index 2) - string
	if v, ok := raw[2].(string); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			kline.High = f
		}
	}

	// Low (index 3) - string
	if v, ok := raw[3].(string); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			kline.Low = f
		}
	}

	// Close (index 4) - string
	if v, ok := raw[4].(string); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			kline.Close = f
		}
	}

	// Volume (index 5) - string
	if v, ok := raw[5].(string); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			kline.Volume = f
		}
	}

	// CloseTime (index 6) - int64
	if v, ok := raw[6].(float64); ok {
		kline.CloseTime = int64(v)
	}

	// QuoteAssetVolume (index 7) - string
	if v, ok := raw[7].(string); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			kline.QuoteAssetVolume = f
		}
	}

	// NumberOfTrades (index 8) - int64
	if v, ok := raw[8].(float64); ok {
		kline.NumberOfTrades = int64(v)
	}

	// TakerBuyBaseAssetVolume (index 9) - string
	if v, ok := raw[9].(string); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			kline.TakerBuyBaseAssetVolume = f
		}
	}

	// TakerBuyQuoteAssetVolume (index 10) - string
	if v, ok := raw[10].(string); ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			kline.TakerBuyQuoteAssetVolume = f
		}
	}

	return kline, nil
}

// ToTime converts OpenTime to time.Time.
func (k *BinanceKline) ToTime() time.Time {
	return time.UnixMilli(k.OpenTime).UTC()
}

// SpreadBPS calculates the spread in basis points.
func (k *BinanceKline) SpreadBPS() float64 {
	if k.Close > 0 {
		return ((k.High - k.Low) / k.Close) * 10000
	}
	return 0
}

// TakerBuyRatio calculates the taker buy ratio.
func (k *BinanceKline) TakerBuyRatio() float64 {
	if k.Volume > 0 {
		return k.TakerBuyBaseAssetVolume / k.Volume
	}
	return 0.5
}

// MidPrice calculates the mid price.
func (k *BinanceKline) MidPrice() float64 {
	return (k.High + k.Low) / 2
}
