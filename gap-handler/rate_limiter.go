// Package main provides rate limiting for Binance API requests.
package main

import (
	"context"
	"sync"
	"time"

	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

// RateLimiter implements token bucket rate limiting with exponential backoff.
type RateLimiter struct {
	limiter        *rate.Limiter
	logger         *zap.Logger
	mu             sync.Mutex
	consecutiveFails int
	lastFailure    time.Time
	backoffUntil   time.Time
}

// BackoffStrategy defines the backoff behavior.
type BackoffStrategy struct {
	// Base delay for first retry
	BaseDelay time.Duration
	// Maximum delay
	MaxDelay time.Duration
	// Rate limit delay (429 response)
	RateLimitDelay time.Duration
	// IP ban delay (418 response)
	IPBanDelay time.Duration
}

// DefaultBackoffStrategy returns the default backoff configuration.
func DefaultBackoffStrategy() BackoffStrategy {
	return BackoffStrategy{
		BaseDelay:      1 * time.Second,
		MaxDelay:       60 * time.Second,
		RateLimitDelay: 60 * time.Second,
		IPBanDelay:     300 * time.Second,
	}
}

// NewRateLimiter creates a new rate limiter.
// requestsPerSecond: token refill rate
// burstSize: maximum tokens (burst capacity)
func NewRateLimiter(requestsPerSecond int, burstSize int, logger *zap.Logger) *RateLimiter {
	return &RateLimiter{
		limiter: rate.NewLimiter(rate.Limit(requestsPerSecond), burstSize),
		logger:  logger,
	}
}

// Wait blocks until a token is available or context is cancelled.
func (rl *RateLimiter) Wait(ctx context.Context) error {
	rl.mu.Lock()
	backoffUntil := rl.backoffUntil
	rl.mu.Unlock()

	// Check if we're in backoff period
	if time.Now().Before(backoffUntil) {
		waitDuration := time.Until(backoffUntil)
		rl.logger.Info("In backoff period, waiting",
			zap.Duration("wait_duration", waitDuration),
		)

		select {
		case <-time.After(waitDuration):
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return rl.limiter.Wait(ctx)
}

// RecordSuccess records a successful request.
func (rl *RateLimiter) RecordSuccess() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	rl.consecutiveFails = 0
}

// RecordFailure records a failed request and returns the backoff duration.
func (rl *RateLimiter) RecordFailure(statusCode int, strategy BackoffStrategy) time.Duration {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	rl.consecutiveFails++
	rl.lastFailure = time.Now()

	var backoffDuration time.Duration

	switch statusCode {
	case 429:
		// Rate limited
		backoffDuration = strategy.RateLimitDelay
		rl.logger.Warn("Rate limited by Binance (429)",
			zap.Duration("backoff", backoffDuration),
		)
		metricsRateLimitHits.Inc()

	case 418:
		// IP banned
		backoffDuration = strategy.IPBanDelay
		rl.logger.Error("IP banned by Binance (418)",
			zap.Duration("backoff", backoffDuration),
		)
		metricsRateLimitHits.Inc()

	default:
		// Exponential backoff based on consecutive failures
		backoffDuration = rl.calculateExponentialBackoff(strategy)
		rl.logger.Warn("Request failed, applying exponential backoff",
			zap.Int("status_code", statusCode),
			zap.Int("consecutive_fails", rl.consecutiveFails),
			zap.Duration("backoff", backoffDuration),
		)
	}

	rl.backoffUntil = time.Now().Add(backoffDuration)
	return backoffDuration
}

// calculateExponentialBackoff calculates backoff based on consecutive failures.
func (rl *RateLimiter) calculateExponentialBackoff(strategy BackoffStrategy) time.Duration {
	// 1st failure: 1s, 2nd: 2s, 3rd: 4s, 4th+: 60s
	if rl.consecutiveFails >= 4 {
		return strategy.MaxDelay
	}

	backoff := strategy.BaseDelay
	for i := 1; i < rl.consecutiveFails; i++ {
		backoff *= 2
	}

	if backoff > strategy.MaxDelay {
		backoff = strategy.MaxDelay
	}

	return backoff
}

// ResetBackoff resets the backoff state.
func (rl *RateLimiter) ResetBackoff() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	rl.consecutiveFails = 0
	rl.backoffUntil = time.Time{}
}

// Stats returns current rate limiter statistics.
func (rl *RateLimiter) Stats() RateLimiterStats {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	return RateLimiterStats{
		ConsecutiveFailures: rl.consecutiveFails,
		LastFailure:         rl.lastFailure,
		BackoffUntil:        rl.backoffUntil,
		TokensAvailable:     rl.limiter.Tokens(),
	}
}

// RateLimiterStats contains rate limiter statistics.
type RateLimiterStats struct {
	ConsecutiveFailures int
	LastFailure         time.Time
	BackoffUntil        time.Time
	TokensAvailable     float64
}
