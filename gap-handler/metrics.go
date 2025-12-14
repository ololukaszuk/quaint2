// Package main provides Prometheus metrics for the gap handler service.
package main

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Backfill metrics
	metricsBackfillDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gap_handler_backfill_duration_seconds",
			Help:    "Duration of backfill operations in seconds",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 10), // 0.1s to ~102s
		},
		[]string{"status"},
	)

	metricsBackfillsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gap_handler_backfills_total",
			Help: "Total number of backfill operations",
		},
		[]string{"status"},
	)

	metricsCandlesRecovered = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "gap_handler_candles_recovered_total",
			Help: "Total number of candles recovered through backfill",
		},
	)

	// API metrics
	metricsAPIRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gap_handler_api_requests_total",
			Help: "Total number of Binance API requests",
		},
		[]string{"status"},
	)

	metricsAPIErrors = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "gap_handler_api_errors_total",
			Help: "Total number of API errors",
		},
	)

	metricsAPILatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "gap_handler_api_latency_seconds",
			Help:    "Binance API request latency in seconds",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 8), // 0.1s to ~25s
		},
	)

	// Rate limiting metrics
	metricsRateLimitHits = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "gap_handler_rate_limit_hits_total",
			Help: "Total number of rate limit hits (429/418 responses)",
		},
	)

	// Database metrics
	metricsDatabaseErrors = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "gap_handler_database_errors_total",
			Help: "Total number of database errors",
		},
	)

	metricsDatabaseLatency = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "gap_handler_database_latency_seconds",
			Help:    "Database operation latency in seconds",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 10), // 1ms to ~1s
		},
	)

	metricsCandlesInserted = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "gap_handler_candles_inserted_total",
			Help: "Total number of candles inserted into database",
		},
	)

	metricsCandlesDuplicate = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "gap_handler_candles_duplicate_total",
			Help: "Total number of duplicate candles skipped",
		},
	)

	// Queue metrics
	metricsPendingBackfills = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "gap_handler_pending_backfills",
			Help: "Number of pending backfill operations",
		},
	)

	metricsActiveBackfills = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "gap_handler_active_backfills",
			Help: "Number of currently running backfill operations",
		},
	)

	// Health metrics
	metricsUptime = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "gap_handler_uptime_seconds_total",
			Help: "Total uptime in seconds",
		},
	)

	metricsBinanceAPIStatus = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "gap_handler_binance_api_status",
			Help: "Binance API status (1=online, 0=offline)",
		},
	)

	metricsDatabaseStatus = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "gap_handler_database_status",
			Help: "Database connection status (1=connected, 0=disconnected)",
		},
	)
)

// MetricsSummary holds aggregated metrics for health endpoint.
type MetricsSummary struct {
	BackfillsCompleted int64
	CandlesRecovered   int64
	Errors24h          int64
	APIRequestsTotal   int64
	RateLimitHits      int64
}

// RecordBackfillDuration records the duration of a backfill operation.
func RecordBackfillDuration(duration float64, status string) {
	metricsBackfillDuration.WithLabelValues(status).Observe(duration)
	metricsBackfillsTotal.WithLabelValues(status).Inc()
}

// RecordAPIRequest records an API request.
func RecordAPIRequest(status string, latency float64) {
	metricsAPIRequests.WithLabelValues(status).Inc()
	metricsAPILatency.Observe(latency)
}

// RecordDatabaseOperation records a database operation.
func RecordDatabaseOperation(latency float64) {
	metricsDatabaseLatency.Observe(latency)
}
