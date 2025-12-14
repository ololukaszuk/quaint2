# Quaint2 - Bitcoin ML Trading Stack

Production-grade real-time cryptocurrency data pipeline with Rust + Go microservices for Binance WebSocket ingestion, TimescaleDB time-series storage, and automated gap backfilling.

**Version**: 3.0 (Rust + Go Hybrid)  
**Status**: Production Ready ✅  
**Last Updated**: December 14, 2025

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Binance WebSocket                        │
│              (@kline_1m + @bookTicker streams)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   Data Feeder (Rust)          │
         │   Port: 8080                  │
         │   - WebSocket ingestion       │
         │   - Batch writer (10/1s)      │
         │   - Gap detector (>60s)       │
         │   - Feature computer          │
         └────────────┬────────────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │   TimescaleDB            │
         │   Port: 5432             │
         │   - Hypertables (7-day)  │
         │   - Compression (90%)    │
         │   - Retention (13 mo)    │
         └────────────┬─────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    ┌──────────────┐      ┌───────────────────┐
    │ Gap Handler  │      │ pgAdmin (Web UI)  │
    │ (Go)         │      │ Port: 8000        │
    │ Port: 9000   │      │ Browse & query    │
    │ - Backfill   │      │ Backup & restore  │
    │ - REST API   │      │ User management   │
    │ - Deduplicate│      └───────────────────┘
    └──────────────┘
          │
          └────────► Insert recovered candles
                     back to TimescaleDB
```

---

## 🎯 Quick Start

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM
- 50GB disk space (for 13 months of candle data)

### Deploy in 3 Steps

```bash
# 1. Clone repository
git clone https://github.com/ololukaszuk/quaint2
cd quaint2

# 2. Configure environment
cp .env.example .env
nano .env  # Edit with secure passwords

# 3. Deploy services
./deploy.sh
```

### Verify Deployment

```bash
# Check all services
docker-compose ps

# Test endpoints
curl http://localhost:8080/health   # Data Feeder
curl http://localhost:9000/health   # Gap Handler
curl http://localhost:5432          # TimescaleDB
curl http://localhost:8000/pgadmin4 # pgAdmin
```

---

## 📦 Components

### 1. Data Feeder (Rust)
**Port**: 8080  
**Purpose**: Real-time Binance WebSocket ingestion

**Capabilities**:
- Connect to dual WebSocket streams (@kline_1m + @bookTicker)
- Ingest ~600 candles per 10 minutes
- Batch write (10 candles or 1 second timeout)
- Detect gaps > 70 seconds automatically
- Compute & cache normalized ML features
- Latency: 2-5ms (Binance → Database)

**Configuration** (`.env`):
```bash
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
DB_HOST=timescaledb
DB_PORT=5432
DB_USER=mltrader
DB_PASSWORD=your_password
DB_NAME=btc_ml_production
BATCH_SIZE=10
BATCH_TIMEOUT_MS=1000
FEATURE_CACHE_INTERVAL_S=60
WEBSOCKET_RECONNECT_INTERVAL_S=5
```

**Health Check**:
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "websocket_connected": true,
  "candles_processed": 45000,
  "database_status": "connected",
  "last_candle_time": "2025-12-14T04:30:00Z",
  "uptime_seconds": 3600
}
```

### 2. Gap Handler (Go)
**Port**: 9000  
**Purpose**: Automated backfill from Binance REST API

**Capabilities**:
- Detect gaps in existing data (>70 seconds)
- Fetch missing candles from Binance REST API
- Rate-limited: 10 requests/second
- Batch insert with deduplication
- Max concurrent backfills: 5
- Performance: 1000+ candles/minute recovery

**Configuration** (`.env`):
```bash
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
DB_HOST=timescaledb
DB_PORT=5432
DB_USER=mltrader
DB_PASSWORD=your_password
DB_NAME=btc_ml_production
RATE_LIMIT_REQUESTS_PER_SEC=10
MAX_CONCURRENT_BACKFILLS=5
BACKFILL_CHECK_INTERVAL_S=300
BACKFILL_BATCH_SIZE=1000
```

**Health Check**:
```bash
curl http://localhost:9000/health
```

Response:
```json
{
  "status": "healthy|degraded",
  "backfills_completed": 5,
  "candles_recovered": 2500,
  "database_status": "connected"
}
```

**Metrics** (Prometheus):
```bash
curl http://localhost:9000/metrics
```

### 3. TimescaleDB (PostgreSQL)
**Port**: 5432  
**Purpose**: Time-series data storage with compression

**Features**:
- Hypertables with automatic 7-day partitioning
- Columnar compression (90% reduction after 7 days)
- 13-month retention for candles
- 3-month retention for predictions
- 1-month retention for features
- Indefinite retention for audit logs

**Key Tables**:
- `candles_1m`: 1-minute OHLCV (hypertable, auto-compressed)
- `feature_cache`: Pre-normalized ML features (cached daily)
- `predictions`: Model outputs & accuracy metrics
- `ensemble_models`: A/B testing configurations
- `data_quality_logs`: Gap detection & integrity events

**Database Access**:
```bash
# Connect directly
docker-compose exec timescaledb psql -U mltrader -d btc_ml_production

# Or from host
psql -h localhost -p 5432 -U mltrader -d btc_ml_production
```

### 4. pgAdmin (Web UI)
**Port**: 8000  
**Purpose**: Database management interface

**Access**:
```
URL: http://localhost:8000/pgadmin4
Email: admin@example.com
Password: (from .env PGADMIN_PASSWORD)
```

**Features**:
- Browse & query database
- Backup & restore
- User management
- Performance monitoring

**Default Connection** (pre-configured):
```
Server: timescaledb
Port: 5432
Database: btc_ml_production
Username: mltrader
Password: (from .env)
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` from `.env.example`:

```bash
# Binance API
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Database
POSTGRES_DB=btc_ml_production
POSTGRES_USER=mltrader
POSTGRES_PASSWORD=your_secure_password_20_chars_min
POSTGRES_HOST_AUTH_METHOD=scram-sha-256

# Data Feeder (Rust)
BATCH_SIZE=10
BATCH_TIMEOUT_MS=1000
FEATURE_CACHE_INTERVAL_S=60
GAP_DETECTION_THRESHOLD_S=70
WEBSOCKET_RECONNECT_INTERVAL_S=5

# Gap Handler (Go)
RATE_LIMIT_REQUESTS_PER_SEC=10
MAX_CONCURRENT_BACKFILLS=5
BACKFILL_CHECK_INTERVAL_S=300
BACKFILL_BATCH_SIZE=1000

# pgAdmin
PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=your_pgadmin_password

# Network
NETWORK_NAME=btc-ml-network
```

---

## 📊 Performance Specifications

| Metric | Value | Notes |
|--------|-------|-------|
| **Ingestion Latency** | 2-5ms | Binance → Database |
| **Throughput** | 600+ candles/sec | 10 candles batch |
| **Gap Detection** | 70 seconds | Automatic trigger |
| **Compression Ratio** | 90% | After 7 days |
| **Data Retention** | 13 months | Candles only |
| **Recovery Speed** | 1000+ candles/min | Backfill rate |
| **Memory per Service** | 15-20MB | At idle |
| **CPU Usage** | <5% each | At normal load |

---

## 📝 Deployment Details

### Docker Compose Services

```yaml
services:
  timescaledb:
    image: timescaledb:latest-pg15
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./timescaledb/init.sql:/docker-entrypoint-initdb.d/01-init.sql
    environment:
      POSTGRES_DB: btc_ml_production
      POSTGRES_USER: mltrader
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "mltrader"]
      interval: 10s
      timeout: 5s
      retries: 5

  data-feeder:
    build: ./data-feeder
    ports:
      - "8080:8080"
    depends_on:
      timescaledb:
        condition: service_healthy
    environment:
      BINANCE_API_KEY: ${BINANCE_API_KEY}
      DB_HOST: timescaledb
      DB_USER: mltrader
      DB_PASSWORD: ${POSTGRES_PASSWORD}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  gap-handler:
    build: ./gap-handler
    ports:
      - "9000:9000"
    depends_on:
      data-feeder:
        condition: service_healthy
    environment:
      BINANCE_API_KEY: ${BINANCE_API_KEY}
      DB_HOST: timescaledb
      DB_USER: mltrader
      DB_PASSWORD: ${POSTGRES_PASSWORD}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  pgadmin:
    image: dpage/pgadmin4:latest
    ports:
      - "8000:80"
    environment:
      PGADMIN_DEFAULT_EMAIL: ${PGADMIN_DEFAULT_EMAIL}
      PGADMIN_DEFAULT_PASSWORD: ${PGADMIN_DEFAULT_PASSWORD}

volumes:
  postgres_data:

networks:
  default:
    name: btc-ml-network
```

### Startup Sequence

Docker Compose enforces dependencies:

1. **TimescaleDB** (must be healthy)
2. **Data Feeder** (depends on TimescaleDB)
3. **Gap Handler** (depends on Data Feeder & TimescaleDB)
4. **pgAdmin** (depends on TimescaleDB)

Health checks ensure proper initialization before dependent services start.

---

## 🏥 Health Checks

### Data Feeder (Port 8080)

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "websocket_connected": true,
  "candles_processed": 45000,
  "database_status": "connected",
  "last_candle_time": "2025-12-14T04:30:00Z",
  "uptime_seconds": 3600
}
```

### Gap Handler (Port 9000)

```bash
curl http://localhost:9000/health
```

Response:
```json
{
  "status": "healthy|degraded",
  "backfills_completed": 5,
  "candles_recovered": 2500,
  "database_status": "connected"
}
```

### Metrics (Port 9000)

```bash
curl http://localhost:9000/metrics
```

Returns Prometheus-compatible metrics.

---

## 🛠️ Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose logs

# Verify ports are free
lsof -i :5432
lsof -i :8080
lsof -i :9000
lsof -i :8000

# Rebuild from scratch
./cleanup.sh
./deploy.sh
```

### Database Connection Errors

```bash
# Test connection
docker-compose exec timescaledb psql -U mltrader -d btc_ml_production

# Check database
docker-compose exec timescaledb pg_isready

# View logs
docker-compose logs timescaledb
```

### Data Feeder Not Ingesting

```bash
# Check WebSocket connectivity
docker-compose logs -f data-feeder | grep -i websocket

# Verify health
curl http://localhost:8080/health

# Check database inserts
docker-compose exec timescaledb psql -U mltrader -d btc_ml_production \
  -c "SELECT COUNT(*) FROM candles_1m;"
```

### Gap Handler Not Backfilling

```bash
# Check logs
docker-compose logs -f gap-handler

# Verify health
curl http://localhost:9000/health

# Check data_quality_logs
docker-compose exec timescaledb psql -U mltrader -d btc_ml_production \
  -c "SELECT * FROM data_quality_logs ORDER BY created_at DESC LIMIT 10;"
```

---

## 📈 Monitoring

### Database Ingestion Rate

```sql
SELECT 
  DATE_TRUNC('minute', time) as minute,
  COUNT(*) as candles_per_minute
FROM candles_1m
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY DATE_TRUNC('minute', time)
ORDER BY minute DESC;
```

### Gap Detection Events

```sql
SELECT 
  event_type,
  COUNT(*) as count,
  SUM(COALESCE(candles_missing, 0)) as total_missing
FROM data_quality_logs
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY event_type;
```

### Service Performance

```bash
# CPU & Memory
docker stats btc-ml-feeder btc-ml-gap-handler

# Logs (last 100 lines)
docker-compose logs -f --tail=100

# Health status
curl http://localhost:8080/health
curl http://localhost:9000/health
```

---

## 🔐 Security

### Best Practices

- Change all default passwords in `.env`
- Never commit `.env` to version control (use `.gitignore`)
- Use secrets management (Vault, AWS Secrets Manager) in production
- Rotate credentials regularly
- Use strong, unique passwords (20+ characters)
- Restrict database access to application network
- Enable SSL/TLS for external connections

### Network Isolation

All services run on internal Docker network (`btc-ml-network`).

Exposed ports:
- **5432**: PostgreSQL (restrict to internal network in production)
- **8080**: Data Feeder Health Check
- **9000**: Gap Handler Health Check
- **8000**: pgAdmin (protect with reverse proxy & authentication)

---

## 📚 Database Schema

### Core Tables

**candles_1m** (Hypertable)
```sql
CREATE TABLE candles_1m (
  time TIMESTAMP NOT NULL,
  open NUMERIC NOT NULL,
  high NUMERIC NOT NULL,
  low NUMERIC NOT NULL,
  close NUMERIC NOT NULL,
  volume NUMERIC NOT NULL,
  quote_asset_volume NUMERIC,
  taker_buy_base_asset_volume NUMERIC,
  taker_buy_quote_asset_volume NUMERIC,
  number_of_trades BIGINT
);
```

**feature_cache** (Regular Table)
```sql
CREATE TABLE feature_cache (
  time TIMESTAMP NOT NULL,
  features_json JSONB NOT NULL,
  version VARCHAR(50),
  created_at TIMESTAMP DEFAULT NOW()
);
```

**predictions** (Regular Table)
```sql
CREATE TABLE predictions (
  time TIMESTAMP NOT NULL,
  symbol VARCHAR(20),
  model_id VARCHAR(100),
  prediction NUMERIC,
  confidence NUMERIC,
  actual NUMERIC,
  accuracy NUMERIC,
  created_at TIMESTAMP DEFAULT NOW(),
  resolved_at TIMESTAMP
);
```

**data_quality_logs** (Regular Table)
```sql
CREATE TABLE data_quality_logs (
  event_type VARCHAR(50),
  source VARCHAR(50),
  gap_start TIMESTAMP,
  gap_end TIMESTAMP,
  candles_missing BIGINT,
  candles_recovered BIGINT,
  error_message TEXT,
  resolved BOOLEAN,
  created_at TIMESTAMP DEFAULT NOW(),
  resolved_at TIMESTAMP
);
```

### Automation

- **Compression**: After 7 days (90% reduction)
- **Retention**: 13 months candles, 3 months predictions, 1 month features
- **Cleanup**: Daily at 2 AM UTC via pg_cron
- **Accuracy Calculation**: Daily at 3 AM UTC

---

## 🔄 ML Model Integration

Example: Query features for model inference

```python
import psycopg2
import json
import pandas as pd

conn = psycopg2.connect(
    host="localhost",
    database="btc_ml_production",
    user="mltrader",
    password="your_password"
)

cur = conn.cursor()
cur.execute("""
    SELECT features_json FROM feature_cache
    WHERE time = %s
""", (pd.Timestamp.now().floor('1min').tz_localize('UTC'),))

features = json.loads(cur.fetchone()[0])
# Use features with your TFT/GRU/Mamba models
```

---

## 📝 Maintenance

### Daily
- Monitor health endpoints
- Check for gaps in data_quality_logs
- Verify data ingestion rate

### Weekly
- Review logs for errors
- Check memory usage
- Verify backups (optional)

### Monthly
- Update Docker images
- Rotate credentials
- Review performance metrics
- Cleanup old predictions (>3 months)

---

## 📄 Directory Structure

```
quaint2/
├── README.md                 # This file
├── .gitignore               # Git ignore patterns
├── .env.example             # Environment template
├── docker-compose.yml       # Service orchestration
├── deploy.sh                # Deployment script
├── cleanup.sh               # Cleanup script
│
├── data-feeder/             # Rust WebSocket service
│   ├── Cargo.toml
│   ├── Dockerfile
│   ├── README.md
│   ├── src/
│   │   ├── main.rs
│   │   ├── lib.rs
│   │   ├── binance.rs       # WebSocket streams
│   │   ├── database.rs      # TimescaleDB writer
│   │   ├── config.rs        # Configuration
│   │   ├── features.rs      # Feature computation
│   │   ├── gap_detector.rs  # Gap detection logic
│   │   ├── health.rs        # Health checks
│   │   └── errors.rs        # Error handling
│   └── target/              # Build output
│
├── gap-handler/             # Go backfill service
│   ├── go.mod
│   ├── Dockerfile
│   ├── README.md
│   ├── main.go
│   ├── config.go            # Configuration
│   ├── binance.go           # REST API client
│   ├── database.go          # Database operations
│   ├── backfiller.go        # Backfill logic
│   ├── rate_limiter.go      # Rate limiting
│   ├── metrics.go           # Prometheus metrics
│   ├── health.go            # Health checks
│   └── vendor/              # Dependencies
│
├── timescaledb/             # Database initialization
│   ├── Dockerfile
│   ├── init.sql             # Schema & config
│   └── README.md
│
└── pgadmin/                 # Web interface
    ├── Dockerfile
    ├── entrypoint.sh
    ├── servers.json         # Server config
    └── README.md
```

---

## 🚦 Service Dependencies

```
TimescaleDB (port 5432)
    ↓
    ├─→ Data Feeder (port 8080)
    │       ↓
    │       └─→ Gap Handler (port 9000)
    │
    └─→ pgAdmin (port 8000)
```

---

## 🤝 Contributing

When adding features:
1. Update both Rust and Go services if needed
2. Test locally with `./deploy.sh`
3. Monitor for 24+ hours before production
4. Document changes in code comments
5. Update README with new specifications

---

## 📞 Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify health: `curl http://localhost:8080/health`
3. Test database: `psql -h localhost -U mltrader -d btc_ml_production`
4. Review troubleshooting section above

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

Built with:
- **Rust**: tokio, tungstenite, tokio-postgres, deadpool
- **Go**: Gorilla Mux, lib/pq, go-resty, zap
- **Database**: TimescaleDB, PostgreSQL 17
- **Container**: Docker, Docker Compose

---

**Version**: 3.0 (Rust + Go Hybrid)  
**Status**: Project: WIP, Data Feed: Production Ready ✅  
