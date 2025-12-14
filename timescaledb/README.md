# TimescaleDB Configuration

Production-grade PostgreSQL extension for time-series data with built-in compression, retention policies, and automated maintenance.

## Overview

TimescaleDB provides:
- **Hypertables**: Automatically partitioned time-series tables
- **Compression**: Column-oriented compression (reduces storage 90%+)
- **Retention**: Automatic deletion of old data
- **Continuous Aggregates**: Pre-computed analytics
- **pg_cron**: Scheduled jobs for automated tasks
- **Performance**: 500+ inserts/second on commodity hardware

**Docker Image**: Custom build (timescaledb:latest-pg17)
**Port**: 5432
**Database**: btc_ml_production
**User**: mltrader (read/write), postgres (superuser)

## Architecture

```
PostgreSQL 17 (Base)
        ↓
TimescaleDB Extension
├── Hypertable Support (automatic partitioning)
├── Compression (background jobs)
├── Retention Policies (auto-cleanup)
├── Continuous Aggregates (pre-computed stats)
└── pg_cron (scheduled maintenance)
        ↓
User Schemas
└── public
    ├── candles_1m (hypertable, compressed)
    ├── feature_cache (regular table)
    ├── predictions (regular table)
    ├── ensemble_models (regular table)
    ├── data_quality_logs (regular table)
    └── [indexes, functions, views]
```

## Configuration

### Environment Variables

```bash
# Docker Compose
POSTGRES_DB=btc_ml_production
POSTGRES_USER=mltrader
POSTGRES_PASSWORD=<secure_password>
POSTGRES_HOST_AUTH_METHOD=scram-sha-256  # Password encryption method
```

### Dockerfile

**File**: `timescaledb/Dockerfile`

```dockerfile
FROM timescaledb:latest-pg17

# Copy initialization script
COPY timescaledb/init.sql /docker-entrypoint-initdb.d/01-init.sql

# TimescaleDB + pg_cron extensions loaded automatically
# Custom parameters via docker-compose command
```

### PostgreSQL Configuration

**Command-line Parameters** (from docker-compose.yml):

```bash
postgres
  -c shared_preload_libraries=timescaledb,pg_cron
  -c cron.database_name=btc_ml_production
  -c max_connections=200
  -c shared_buffers=256MB
```

**Breakdown**:
- `shared_preload_libraries=timescaledb,pg_cron`: Load extensions at startup
- `cron.database_name`: Database for pg_cron jobs
- `max_connections=200`: Allow up to 200 simultaneous connections
- `shared_buffers=256MB`: Shared memory for caching

## Database Schema

### Init Script

**File**: `timescaledb/init.sql` (37KB, comprehensive)

**Executed on first startup** by Docker entrypoint.

**Steps**:

1. **Create extensions**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS timescaledb;
   CREATE EXTENSION IF NOT EXISTS pg_cron;
   ```

2. **Create hypertable** (time-series optimized):
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
     number_of_trades BIGINT,
     PRIMARY KEY (time)
   );
   
   SELECT create_hypertable('candles_1m', 'time', if_not_exists => TRUE);
   ```

3. **Create regular tables**:
   - `feature_cache`: Pre-computed ML features
   - `predictions`: Model output results
   - `ensemble_models`: A/B testing configurations
   - `data_quality_logs`: Gap detection and backfill events

4. **Create indexes**:
   - Time-based indexes for fast range queries
   - Symbol indexes (when multi-symbol supported)
   - Foreign keys for referential integrity

5. **Set compression policy**:
   ```sql
   SELECT add_compression_policy('candles_1m', INTERVAL '7 days');
   ```

6. **Set retention policy**:
   ```sql
   SELECT add_retention_policy('candles_1m', INTERVAL '13 months');
   ```

7. **Create cron jobs** (pg_cron):
   - Daily accuracy calculation (3 AM UTC)
   - Weekly data quality reports (Sunday 2 AM UTC)
   - Monthly cleanup of old predictions (1 AM UTC on 1st)

### Table Details

#### candles_1m (Hypertable)

**Purpose**: Store 1-minute OHLCV (Open, High, Low, Close, Volume) data

**Columns**:
- `time`: Timestamp (primary key, NOT NULL)
- `open`: Opening price (NUMERIC)
- `high`: Highest price (NUMERIC)
- `low`: Lowest price (NUMERIC)
- `close`: Closing price (NUMERIC)
- `volume`: Quote asset volume (NUMERIC)
- `quote_asset_volume`: Quote asset volume (optional)
- `taker_buy_base_asset_volume`: Taker buy base volume (optional)
- `taker_buy_quote_asset_volume`: Taker buy quote volume (optional)
- `number_of_trades`: Trade count (BIGINT)

**Partitioning**: Automatic, 7-day chunks (optimized by TimescaleDB)

**Compression**: After 7 days, automatic columnar compression
- Original size: ~8KB per day (600 candles)
- Compressed size: ~800 bytes per day (90% reduction)
- Total 13 months: ~6-8 GB compressed vs 60 GB uncompressed

**Retention**: Delete data older than 13 months automatically

**Indexes**:
- Primary key on `time`
- Maybe additional indexes on frequently filtered columns

**Query Performance**:
```sql
-- Fast (uses partition pruning)
SELECT * FROM candles_1m 
WHERE time > NOW() - INTERVAL '1 day';

-- Fast (index scan)
SELECT close FROM candles_1m 
WHERE time = '2025-12-13 21:00:00';
```

#### feature_cache (Regular Table)

**Purpose**: Cache pre-computed ML features for fast inference

**Columns**:
- `time`: Timestamp (primary key)
- `features_json`: JSON object with normalized features
- `version`: Feature version (for model tracking)
- `created_at`: Timestamp when computed

**Retention**: Automatic cleanup (1 month via cron job)

**Indexing**: On `time` and `version`

#### predictions (Regular Table)

**Purpose**: Store model predictions and actual outcomes

**Columns**:
- `time`: Timestamp of prediction
- `symbol`: Trading pair (e.g., BTCUSDT)
- `model_id`: Which model generated prediction
- `prediction`: Predicted direction/price
- `confidence`: Confidence score (0-1)
- `actual`: Actual outcome
- `accuracy`: Prediction accuracy
- `created_at`: Timestamp
- `resolved_at`: When outcome known

**Retention**: Automatic cleanup (3 months via cron job)

#### data_quality_logs (Regular Table)

**Purpose**: Track gaps, backfills, and data quality events

**Columns**:
- `event_type`: 'gap_detected', 'gap_backfilled', 'error'
- `source`: 'data_feeder', 'gap_handler', 'system'
- `gap_start`: Start of gap (if applicable)
- `gap_end`: End of gap (if applicable)
- `candles_missing`: Count of missing candles
- `candles_recovered`: Count of recovered candles
- `error_message`: Error details (if error type)
- `resolved`: Whether resolved
- `created_at`: Event timestamp
- `resolved_at`: When resolved

**Retention**: Keep indefinitely (audit trail)

## Operations

### Docker Commands

```bash
# Start TimescaleDB
docker-compose up timescaledb

# Check status
docker-compose ps timescaledb

# View logs
docker-compose logs -f timescaledb

# Execute SQL
docker-compose exec timescaledb psql -U mltrader -d btc_ml_production

# Backup
docker-compose exec timescaledb pg_dump -U mltrader btc_ml_production > backup.sql

# Restore
docker-compose exec -T timescaledb psql -U mltrader -d btc_ml_production < backup.sql
```

### Health Check

```bash
# Docker health check (automatically runs every 10s)
pg_isready -U mltrader -d btc_ml_production

# Manual check
psql -h localhost -U mltrader -d btc_ml_production -c "SELECT 1;"
```

**Health status**: Docker Compose shows:
- `(healthy)`: Database ready
- `(unhealthy)`: Database not responding

### Monitoring

#### System Metadata

```sql
-- Hypertable chunks
SELECT * FROM timescaledb_information.chunks 
WHERE hypertable_name = 'candles_1m'
ORDER BY range_start DESC
LIMIT 5;

-- Compression status
SELECT 
  chunk_name,
  chunk_size,
  compressed_chunk_size,
  ROUND(100.0 * compressed_chunk_size / chunk_size, 1) as compression_ratio
FROM timescaledb_information.detailed_index_size
WHERE table_name = 'candles_1m'
ORDER BY chunk_name DESC;

-- Disk usage
SELECT
  table_name,
  pg_size_pretty(total_bytes) as total,
  pg_size_pretty(index_bytes) as indexes,
  pg_size_pretty(toast_bytes) as toast
FROM pg_table_size_pretty()
WHERE table_name = 'candles_1m';
```

#### Data Quality

```sql
-- Latest candles
SELECT time, close, volume FROM candles_1m 
ORDER BY time DESC LIMIT 10;

-- Candles per hour (ingestion rate)
SELECT 
  DATE_TRUNC('hour', time) as hour,
  COUNT(*) as candles_per_hour
FROM candles_1m
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', time)
ORDER BY hour DESC;

-- Gaps detected
SELECT 
  event_type,
  COUNT(*) as count,
  SUM(COALESCE(candles_missing, 0)) as total_missing,
  SUM(COALESCE(candles_recovered, 0)) as total_recovered
FROM data_quality_logs
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY event_type;

-- Unresolved gaps
SELECT *
FROM data_quality_logs
WHERE event_type = 'gap_detected'
  AND resolved = false
ORDER BY created_at ASC;
```

## Performance Tuning

### Connection Pool Optimization

**Default Settings**:
- `max_connections = 200`: 5 (feeder) + 10 (handler) + 50 (pgadmin) + buffer

**Adjustment Formula**:
```
max_connections = (data_feeder_pool) + (gap_handler_pool) + (pgadmin) + (application) + buffer
= 5 + 10 + 50 + 50 + 20 = 135 (safe)
```

If seeing "too many connections" errors, increase `max_connections` in docker-compose command.

### Shared Buffers

**Current**: 256 MB

**Formula**: 25% of available RAM

```bash
# For 4GB RAM
256 MB ✓ (25% of 4GB)

# For 8GB RAM  
2 GB (better performance)

# For 16GB RAM
4 GB (enterprise grade)
```

### Work Memory

**Default**: 4 MB (PG default)

**Increase for complex queries**:
```sql
SET work_mem = '256MB';  -- For sorts, hash joins
```

## Maintenance

### Automatic (via pg_cron)

**Scheduled Jobs** (defined in init.sql):

1. **Data Accuracy Calculation** (3 AM UTC daily)
   ```sql
   SELECT calculate_prediction_accuracy();
   ```

2. **Data Quality Report** (2 AM UTC Sundays)
   ```sql
   SELECT generate_quality_report();
   ```

3. **Old Predictions Cleanup** (1 AM UTC, monthly 1st)
   ```sql
   DELETE FROM predictions 
   WHERE created_at < NOW() - INTERVAL '3 months';
   ```

### Manual Maintenance

**Vacuum (cleanup dead rows)**:
```sql
VACUUM ANALYZE candles_1m;
```

**Reindex (rebuild indexes)**:
```sql
REINDEX TABLE candles_1m;
```

**Force compression** (if not waiting 7 days):
```sql
SELECT compress_chunk(i) FROM show_chunks('candles_1m', '7 days'::interval) i;
```

## Backup & Restore

### Automated Backup

```bash
# Full backup
docker-compose exec timescaledb pg_dump \
  -U mltrader \
  --format=custom \
  --compress=9 \
  btc_ml_production > backup_$(date +%Y%m%d).dump

# Size: 10-50 MB (depends on compression)
```

### Restore from Backup

```bash
# Create empty database
docker-compose exec timescaledb createdb \
  -U mltrader \
  btc_ml_production

# Restore (takes minutes)
docker-compose exec -T timescaledb pg_restore \
  -U mltrader \
  -d btc_ml_production \
  backup_20251213.dump
```

### Point-in-Time Recovery

**Not configured by default** (requires WAL archiving)

To enable:
```sql
-- Add to PostgreSQL config
wal_level = replica
archive_mode = on
archive_command = 'cp %p /archive/%f'
```

## Troubleshooting

### Database won't start

**Error**: "database "btc_ml_production" does not exist"

**Solution**:
1. Check init.sql exists: `ls timescaledb/init.sql`
2. Remove persistent volume: `docker-compose down -v`
3. Restart: `docker-compose up timescaledb`

### High disk usage

**Error**: Database volume >50 GB

**Solution**:
1. Check compression status:
   ```sql
   SELECT * FROM timescaledb_information.chunks 
   WHERE is_compressed = false;
   ```
2. Force compression of old chunks:
   ```sql
   SELECT compress_chunk(i) 
   FROM show_chunks('candles_1m', '7 days'::interval) i
   WHERE NOT is_compressed(i);
   ```
3. Check for large individual queries:
   ```sql
   SELECT * FROM pg_stat_statements 
   ORDER BY calls DESC LIMIT 10;
   ```

### Slow queries

**Error**: Queries taking >1 second

**Solution**:
1. Enable query logging:
   ```sql
   SET log_min_duration_statement = 1000;  -- Log >1s queries
   ```
2. Analyze query plan:
   ```sql
   EXPLAIN ANALYZE SELECT ...;
   ```
3. Add indexes if needed:
   ```sql
   CREATE INDEX idx_feature_cache_time ON feature_cache(time DESC);
   ```

### Connection pool exhausted

**Error**: "sorry, too many clients already"

**Solution**:
1. Check active connections:
   ```sql
   SELECT count(*), state FROM pg_stat_activity GROUP BY state;
   ```
2. Kill idle connections:
   ```sql
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE state = 'idle' 
     AND query_start < NOW() - INTERVAL '30 min';
   ```
3. Increase max_connections in docker-compose.yml

## Extensions Used

### TimescaleDB

**Features**:
- Hypertable with automatic partitioning
- Compression (columnar storage)
- Retention policies
- Continuous aggregates
- Custom aggregate functions

**Version**: Latest (usually 2.x)

### pg_cron

**Features**:
- Schedule SQL jobs
- Run at specific times (cron syntax)
- Essential for automated maintenance

**Version**: Latest (usually 1.x)

## Security

### Authentication

**Method**: SCRAM-SHA-256 (secure password hashing)

```bash
POSTGRES_HOST_AUTH_METHOD=scram-sha-256
```

### User Roles

**Superuser**: `postgres` (for maintenance, not used by applications)

**Application user**: `mltrader` (limited privileges)
- CREATE TABLE
- INSERT, UPDATE, SELECT
- No DROP TABLE (prevents accidental data loss)

### Network Isolation

**Port 5432**: Only accessible from Docker network

**Access from host**:
```bash
psql -h localhost -p 5432 -U mltrader -d btc_ml_production
```

**Access from other containers**:
```bash
psql -h timescaledb -U mltrader -d btc_ml_production  # Uses Docker DNS
```

## Integration with Data Pipeline

### Data Feeder Integration

- Writes 10 candles every 1 second
- ~600 candles per 10 minutes
- ~86,400 candles per day
- Approximately **1 MB per day** (uncompressed)

### Gap Handler Integration

- Reads existing timestamps (for deduplication)
- Inserts recovered candles
- Logs to data_quality_logs
- Queries unresolved gaps

### ML Model Integration

- Reads from feature_cache for inference
- Writes predictions to predictions table
- Queries historical data for training

## Known Limitations

1. **Single database**: Not sharded (suitable for <100GB)
2. **Single symbol**: BTCUSDT only (can add more symbols)
3. **No replication**: No built-in failover
4. **Local storage**: Data lost if container removed without volume

## Future Enhancements

- [ ] Implement WAL archiving for point-in-time recovery
- [ ] Add database replication for high availability
- [ ] Implement read replicas for analytics workloads
- [ ] Add multi-symbol support
- [ ] Set up automated backups to cloud storage

## Support

For TimescaleDB documentation: https://docs.timescale.com/

For issues:
1. Check logs: `docker-compose logs timescaledb`
2. Verify health: `pg_isready -h localhost -U mltrader`
3. Run diagnostics: `SELECT * FROM timescaledb_information.about();`
4. Check PostgreSQL docs: https://www.postgresql.org/docs/17/
