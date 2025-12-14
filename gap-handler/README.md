# Gap Handler Service

A production-ready Go service that backfills missing Binance candlestick data by fetching from the Binance REST API and inserting into TimescaleDB.

## Features

- **HTTP API for Backfill Requests**: Accepts POST requests to backfill specific time ranges
- **Rate Limiting**: Token bucket with exponential backoff for Binance API compliance
- **Batch Database Writes**: Efficient batch inserts with deduplication
- **Prometheus Metrics**: Comprehensive metrics for monitoring
- **Health Check Endpoints**: /health, /ready, /live for orchestration
- **Graceful Shutdown**: Handles SIGTERM/SIGINT cleanly

## Architecture

```
main()
├── load configuration from .env
├── initialize database connection
├── initialize Binance API client
├── initialize rate limiter
├── HTTP routes:
│   ├── POST /backfill → GapBackfiller
│   ├── POST /backfill/unresolved → Backfill all pending gaps
│   ├── GET /health → HealthCheck
│   ├── GET /ready → ReadinessCheck
│   ├── GET /live → LivenessCheck
│   ├── GET /metrics → Prometheus metrics
│   └── POST /shutdown → graceful shutdown
├── start HTTP server on port 9000
└── graceful shutdown on SIGTERM
```

## Prerequisites

- Go 1.21+
- TimescaleDB instance with schema from `timescaledb_init.sql`
- Network access to Binance REST API

## Configuration

Create a `.env` file in the project root:

```env
# Database
DB_HOST=timescaledb
DB_PORT=5432
DB_NAME=btc_ml_production
DB_USER=mltrader
DB_PASSWORD=your_secure_password
DB_MAX_CONNS=2

# Binance API
BINANCE_API_URL=https://api.binance.com
TRADING_SYMBOL=BTCUSDT
CANDLE_INTERVAL=1m

# Server
GAP_HANDLER_PORT=9000
MAX_CONCURRENT_BACKFILLS=5
BACKFILL_TIMEOUT_SECONDS=300

# Rate Limiting
REQUESTS_PER_SECOND=10
BURST_SIZE=10

# Logging
LOG_LEVEL=info
```

## Building

### Development Build

```bash
cd gap-handler
go build -o gap-handler .
```

### Production Build

```bash
cd gap-handler
CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o gap-handler .
```

### With Docker

```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o gap-handler .

FROM alpine:3.19
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/gap-handler /usr/local/bin/
CMD ["gap-handler"]
```

Build and run:

```bash
docker build -t gap-handler .
docker run -d --env-file .env --name gap-handler -p 9000:9000 gap-handler
```

## Running

### Direct Execution

```bash
# Development
go run .

# Production
./gap-handler

# With custom log level
LOG_LEVEL=debug ./gap-handler
```

## API Endpoints

### POST /backfill

Backfill a specific gap in candlestick data.

**Request:**
```json
{
  "gap_start": "2025-12-13T19:00:00Z",
  "gap_end": "2025-12-13T20:00:00Z"
}
```

**Response:**
```json
{
  "status": "backfilled",
  "gap_start": "2025-12-13T19:00:00Z",
  "gap_end": "2025-12-13T20:00:00Z",
  "candles_expected": 60,
  "candles_fetched": 60,
  "candles_inserted": 58,
  "candles_skipped": 2,
  "candles_recovered": 58,
  "duration_seconds": 2.5
}
```

**Status Codes:**
- `200 OK`: Backfill successful or partial
- `400 Bad Request`: Invalid request parameters
- `429 Too Many Requests`: Concurrent backfill limit reached
- `500 Internal Server Error`: Backfill failed

### POST /backfill/unresolved

Backfill all unresolved gaps in the database.

**Response:**
```json
{
  "status": "completed",
  "gaps_processed": 3,
  "gaps_successful": 2,
  "gaps_failed": 1,
  "candles_recovered": 120,
  "details": [...]
}
```

### GET /health

Returns comprehensive health status.

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "backfills_completed": 5,
  "candles_recovered": 2500,
  "errors_24h": 0,
  "binance_api_status": "online",
  "database_status": "connected",
  "pending_backfills": 0,
  "active_backfills": 0,
  "timestamp": "2025-12-13T20:10:00Z",
  "details": {
    "unresolved_gaps": 0,
    "candles_missing": 0,
    "backfills_last_24h": 5,
    "candles_recovered_last_24h": 2500
  }
}
```

**Status Values:**
- `healthy`: All systems operational
- `degraded`: Working but with issues
- `unhealthy`: Critical failure

### GET /ready

Returns `200 OK` if the service is ready (database connected).

### GET /live

Returns `200 OK` if the service is running.

### GET /metrics

Prometheus-compatible metrics endpoint.

**Metrics exposed:**
- `gap_handler_backfill_duration_seconds` - Backfill operation duration
- `gap_handler_backfills_total` - Total backfill operations
- `gap_handler_candles_recovered_total` - Total candles recovered
- `gap_handler_api_requests_total` - Binance API requests
- `gap_handler_api_errors_total` - API errors
- `gap_handler_api_latency_seconds` - API request latency
- `gap_handler_rate_limit_hits_total` - Rate limit hits
- `gap_handler_database_errors_total` - Database errors
- `gap_handler_database_latency_seconds` - Database operation latency
- `gap_handler_candles_inserted_total` - Candles inserted
- `gap_handler_candles_duplicate_total` - Duplicate candles skipped
- `gap_handler_pending_backfills` - Pending backfills
- `gap_handler_active_backfills` - Active backfills
- `gap_handler_uptime_seconds_total` - Service uptime
- `gap_handler_binance_api_status` - Binance API status (1=online)
- `gap_handler_database_status` - Database status (1=connected)

### POST /shutdown

Triggers graceful shutdown.

## Integration with Data Feeder

The data feeder service detects gaps and triggers backfill:

1. Data feeder detects gap (60+ seconds without candle)
2. Feeder POSTs to gap-handler:
   ```
   POST http://gap-handler:9000/backfill
   {"gap_start": "2025-12-13T19:00:00Z", "gap_end": "2025-12-13T20:00:00Z"}
   ```
3. Gap handler processes backfill
4. Gap handler responds with summary
5. Data feeder resumes normal operation

## Rate Limiting

The service implements Binance rate limit compliance:

- **Token Bucket**: 10 requests/second capacity
- **Backoff Strategy**:
  - 1st failure: 1s wait
  - 2nd failure: 2s wait
  - 3rd failure: 4s wait
  - 4th+ failure: 60s wait
- **429 Response**: 60s backoff
- **418 Response** (IP ban): 300s backoff

## Performance Targets

| Metric | Target |
|--------|--------|
| API request latency | 500-2000ms (Binance latency) |
| Database write | 2-5ms per batch |
| Memory usage | 20-30MB idle |
| Throughput | 100+ concurrent backfills |
| Rate compliance | <1200 requests/minute |

## Database Schema

The service uses the following tables from `timescaledb_init.sql`:

- `candles_1m`: Primary hypertable for candlestick data
- `data_quality_logs`: Gap detection and backfill logging

Key operations:
- INSERT with ON CONFLICT for safe upsert
- Batch inserts in transactions
- Query existing timestamps for deduplication

## Module Structure

```
gap-handler/
├── main.go         # Application entry point and HTTP routing
├── config.go       # Configuration from environment
├── backfiller.go   # Core backfill logic
├── binance.go      # Binance REST API client
├── database.go     # TimescaleDB operations
├── rate_limiter.go # Token bucket rate limiting
├── health.go       # Health check handlers
├── metrics.go      # Prometheus metrics
├── go.mod          # Go module definition
└── README.md       # This file
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gap-handler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gap-handler
  template:
    metadata:
      labels:
        app: gap-handler
    spec:
      containers:
      - name: gap-handler
        image: gap-handler:latest
        ports:
        - containerPort: 9000
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        livenessProbe:
          httpGet:
            path: /live
            port: 9000
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "32Mi"
            cpu: "50m"
          limits:
            memory: "128Mi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: gap-handler
spec:
  selector:
    app: gap-handler
  ports:
  - port: 9000
    targetPort: 9000
```

## Monitoring

### Prometheus Scrape Config

```yaml
- job_name: 'gap-handler'
  static_configs:
    - targets: ['gap-handler:9000']
  metrics_path: '/metrics'
```

### Grafana Dashboard Queries

```promql
# Backfill success rate
sum(rate(gap_handler_backfills_total{status="backfilled"}[5m])) /
sum(rate(gap_handler_backfills_total[5m]))

# Candles recovered per minute
rate(gap_handler_candles_recovered_total[1m])

# API latency p99
histogram_quantile(0.99, rate(gap_handler_api_latency_seconds_bucket[5m]))

# Database latency p95
histogram_quantile(0.95, rate(gap_handler_database_latency_seconds_bucket[5m]))
```

## Error Handling

The service handles errors gracefully:

- **API Errors**: Logged and retried with exponential backoff
- **Database Errors**: Logged, metrics incremented, operation retried
- **Rate Limits**: Automatic backoff per Binance requirements
- **Malformed Data**: Skipped with warning log

All errors are tracked in:
- Prometheus metrics for alerting
- `data_quality_logs` table for audit
- Structured JSON logs for debugging

## License

MIT
