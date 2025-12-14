# Binance Data Feeder Service

A production-ready Rust service that streams cryptocurrency price data from Binance and stores it in TimescaleDB.

## Features

- **Dual WebSocket Streams**: Subscribes to both `@kline_1m` (candlestick) and `@bookTicker` (bid/ask) streams
- **Automatic Reconnection**: Exponential backoff with 1s, 2s, 4s, 8s (max) delays
- **Batch Database Writes**: Accumulates candles and writes every 10 candles or 1 second
- **Retry Logic**: Up to 3 retries with exponential backoff on database errors
- **Gap Detection**: Monitors for missing candles and triggers backfill
- **Feature Computation**: Real-time min-max and z-score normalization using ndarray
- **Health Check Endpoint**: HTTP server with `/health`, `/ready`, and `/live` endpoints
- **Graceful Shutdown**: Handles SIGTERM/SIGINT for clean termination

## Architecture

```
main()
├── spawn BinanceWebSocketClient
│   └── Channel<CandleData>
├── spawn CandleProcessor
│   ├── Gap Detection
│   ├── Feature Computation
│   └── Forward to DatabaseWriter
├── spawn DatabaseWriter
│   ├── Batch writes
│   ├── Retry logic
│   └── Channel<WriteResult>
├── spawn HealthCheck server
│   └── Port: 8080
└── graceful_shutdown() on SIGTERM
```

## Prerequisites

- Rust 1.70+ (with Cargo)
- TimescaleDB instance running
- Network access to Binance WebSocket API

## Configuration

Create a `.env` file in the project root:

```env
# Binance WebSocket
BINANCE_STREAM_URL=wss://stream.binance.com:9443/ws
TRADING_SYMBOL=btcusdt

# Database
DB_HOST=timescaledb
DB_PORT=5432
DB_NAME=btc_ml_production
DB_USER=mltrader
DB_PASSWORD=your_secure_password
DB_POOL_MIN=2
DB_POOL_MAX=5
DB_IDLE_TIMEOUT=60

# Feature computation
FEATURE_UPDATE_INTERVAL=60

# Gap detection
GAP_DETECTION_THRESHOLD=70
GAP_HANDLER_URL=http://gap-handler:9000/backfill

# Health check
HEALTH_CHECK_PORT=8080

# Logging
LOG_LEVEL=info
```

## Building

### Development Build

```bash
cd data-feeder
cargo build
```

### Release Build (Optimized)

```bash
cd data-feeder
cargo build --release
```

The release build enables:
- LTO (Link Time Optimization)
- Single codegen unit
- Panic abort
- Binary stripping

## Running

### Direct Execution

```bash
# Development
cargo run

# Production
cargo run --release

# With custom log level
LOG_LEVEL=debug cargo run --release
```

### Docker

```dockerfile
FROM rust:1.74-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/data-feeder /usr/local/bin/
CMD ["data-feeder"]
```

Build and run:

```bash
docker build -t data-feeder .
docker run -d --env-file .env --name data-feeder data-feeder
```

## Health Check

The service exposes health endpoints on the configured port (default: 8080).

### GET /health

Returns comprehensive health status:

```json
{
  "status": "healthy",
  "uptime_seconds": 12345,
  "candles_processed": 50000,
  "candles_in_db": 50000,
  "last_candle_time": "2025-12-13T20:00:00Z",
  "database_connected": true,
  "websocket_connected": true,
  "pending_backfills": 0,
  "errors_24h": 0,
  "memory_mb": 15.2,
  "timestamp": "2025-12-13T20:05:12Z",
  "details": {
    "buffer_size": 0,
    "db_pool_size": null,
    "features_warmed_up": true,
    "gap_check_last": "2025-12-13T20:00:00Z"
  }
}
```

Status values:
- `healthy`: All systems operational
- `degraded`: Working but with issues (high errors, memory, or stale data)
- `unhealthy`: Critical failure (DB or WS disconnected)

### GET /ready

Returns `200 OK` if the service is ready to receive traffic (DB and WS connected).

### GET /live

Returns `200 OK` if the service is running (always succeeds if process is alive).

## Performance Targets

| Metric | Target |
|--------|--------|
| WebSocket latency | 1-3ms |
| Database write latency | 2-4ms per candle |
| Feature computation | <1ms per candle |
| Memory usage (idle) | 10-20MB |
| Throughput (burst) | 500+ inserts/second |

## Database Schema

The service expects the TimescaleDB schema from `timescaledb_init.sql`:

- `candles_1m`: Primary hypertable for 1-minute OHLCV data
- `data_quality_logs`: Gap detection and error logging
- `feature_cache`: Pre-computed normalized features

Key functions used:
- `upsert_candle()`: Safe insert/update with conflict handling
- Triggers for computing derived columns (spread_bps, taker_buy_ratio, mid_price)
- Triggers for automatic gap detection

## Module Structure

```
src/
├── main.rs         # Application entry point and orchestration
├── binance.rs      # WebSocket client for Binance streams
├── database.rs     # TimescaleDB writer with pooling
├── gap_detector.rs # Gap detection and backfill triggering
├── features.rs     # Feature computation using ndarray
├── health.rs       # HTTP health check server
├── errors.rs       # Error types and result aliases
└── config.rs       # Environment configuration
```

## Error Handling

The service uses a custom error type (`DataFeederError`) with variants for:
- WebSocket errors
- Database errors
- Configuration errors
- Channel communication errors
- Gap detection errors
- Feature computation errors
- HTTP errors

All errors are logged and tracked in the health metrics.

## Graceful Shutdown

On receiving SIGTERM or SIGINT:
1. Sets shutdown flag
2. Stops WebSocket client
3. Flushes remaining candles to database
4. Waits up to 10 seconds for tasks to complete
5. Force exits if timeout reached

## Monitoring

Use the `/health` endpoint with your monitoring system:

```bash
# Prometheus scrape config example
- job_name: 'data-feeder'
  static_configs:
    - targets: ['data-feeder:8080']
  metrics_path: '/health'
```

For Kubernetes:

```yaml
livenessProbe:
  httpGet:
    path: /live
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

## License

MIT
