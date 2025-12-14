# pgAdmin Configuration

Web-based PostgreSQL administration interface for database management and monitoring.

## Overview

pgAdmin 4 provides:
- Visual database structure browser
- Query editor with syntax highlighting
- Real-time query execution
- Server activity monitoring
- User and role management
- Backup/restore utilities
- Dashboard with system stats

**Docker Image**: `dpage/pgadmin4:latest`
**Port**: 8000 (default)
**Access**: http://localhost:8000/pgadmin4

## Quick Start

### Access

1. **URL**: http://localhost:8000/pgadmin4
2. **Email**: admin@example.com
3. **Password**: From `.env` (PGADMIN_PASSWORD)

### Configure Server Connection

pgAdmin auto-connects to TimescaleDB via pre-configured `servers.json`:

**File**: `pgadmin/servers.json`
```json
{
  "Servers": {
    "1": {
      "Name": "BTC ML Production",
      "Group": "Production",
      "Host": "timescaledb",
      "Port": 5432,
      "MaintenanceDB": "postgres",
      "Username": "mltrader",
      "Password": "from_env",
      "SSLMode": "prefer",
      "Shared": false,
      "Comment": "Production TimescaleDB instance"
    }
  }
}
```

**Credentials are auto-injected** via `entrypoint.sh` script.

## Configuration

### Environment Variables

```bash
# From .env file
PGADMIN_PASSWORD=<secure_password>
PGADMIN_PORT=8000
```

### entrypoint.sh Script

**File**: `pgadmin/entrypoint.sh`

Responsibilities:
1. Reads `pgpass` file for authentication
2. Injects DB_PASSWORD into servers.json
3. Sets SCRIPT_NAME=/pgadmin4
4. Starts pgAdmin service

**Features**:
- Automatic credential injection (no hardcoding passwords)
- Path configuration for reverse proxy support
- Graceful handling of missing configuration

### pgpass Credentials File

**File**: `pgadmin/pgpass` (600 permissions required)

Format: `hostname:port:database:username:password`

```
timescaledb:5432:btc_ml_production:mltrader:password
```

**Security Notes**:
- File must have 600 permissions (read/write owner only)
- Never commit to version control
- Regenerate for each deployment
- Rotate passwords regularly

## Docker Integration

### Volume Mounts

```yaml
pgadmin:
  volumes:
    - pgadmin_data:/var/lib/pgadmin          # Persistent storage
    - ./pgadmin/servers.json:/pgadmin4/servers.json:ro   # Read-only config
    - ./pgadmin/entrypoint.sh:/custom-entrypoint.sh:ro   # Read-only script
    - ./pgadmin/pgpass:/pgadmin/pgpass:ro    # Read-only credentials
```

**Directories**:
- `/var/lib/pgadmin`: User data, saved connections, query history
- `/pgadmin4`: Configuration files
- `/pgadmin/`: Credentials

### Network Configuration

```yaml
networks:
  - btc-ml-network  # Docker bridge network
```

**Access within Docker**:
- From other containers: http://pgadmin:80
- From host: http://localhost:8000/pgadmin4

**Environment**:
- SCRIPT_NAME=/pgadmin4 (for reverse proxy behind /pgadmin4 path)
- PGADMIN_DEFAULT_EMAIL=admin@example.com (login email)
- PGADMIN_DEFAULT_PASSWORD=from_.env (login password)

## Usage

### Connecting to TimescaleDB

**Via Pre-configured Server**:
1. Login to pgAdmin
2. Left sidebar → Servers
3. Click "BTC ML Production" (auto-configured)
4. Browser shows database objects

**Manual Connection** (if auto-configuration fails):
1. Right-click "Servers" → Create → Server
2. General tab: Name = "BTC ML Production"
3. Connection tab:
   - Host: timescaledb
   - Port: 5432
   - Maintenance database: postgres
   - Username: mltrader
   - Password: (from .env)
4. Click Save

### Common Tasks

#### Browse Schema

Left sidebar path:
```
BTC ML Production
├── Databases
│   └── btc_ml_production
│       ├── Schemas
│       │   └── public
│       │       ├── Tables
│       │       │   ├── candles_1m (hypertable)
│       │       │   ├── feature_cache
│       │       │   ├── predictions
│       │       │   ├── ensemble_models
│       │       │   ├── data_quality_logs
│       │       │   └── ...
│       │       ├── Indexes
│       │       ├── Functions
│       │       └── Extensions
│       ├── Users
│       └── Crons (pg_cron jobs)
```

#### Execute SQL Query

1. Right-click database → Query Tool
2. Paste SQL in editor
3. Click Execute (or F6)
4. Results displayed below

**Example Queries**:

```sql
-- Check recent candles
SELECT time, close, volume FROM candles_1m 
ORDER BY time DESC LIMIT 10;

-- Data quality status
SELECT event_type, COUNT(*), MAX(created_at)
FROM data_quality_logs
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY event_type;

-- Feature cache size
SELECT COUNT(*) as cached_features, MAX(time) as latest
FROM feature_cache;

-- Hypertable chunks
SELECT * FROM timescaledb_information.chunks 
WHERE hypertable_name = 'candles_1m';
```

#### Monitor Server Activity

**Dashboard**:
1. Tools → Server activity
2. Shows:
   - Active connections
   - Database size
   - Queries being executed
   - Locks and blocking
   - Cache hit ratio

#### Backup Database

1. Right-click database → Backup
2. Format: Custom (recommended)
3. Filename: btc_ml_production_YYYY-MM-DD.dump
4. Sections:
   - ☑ Pre-data (schema)
   - ☑ Data
   - ☑ Post-data (constraints)
5. Click Backup

**Result**: `.dump` file in local directory

#### Restore Backup

1. Right-click database → Restore
2. Select backup file
3. Format: Custom
4. Click Restore

**Warning**: Overwrites existing data

#### Create User

1. Right-click Login/Group Roles → Create → Login/Group Role
2. Properties tab:
   - Name: username
   - Password: secure_password
   - Can login: ☑
   - Superuser: ☐ (unless needed)
3. Click Save

#### View Logs

**Server Logs**:
1. Tools → Server Log
2. Shows PostgreSQL server logs
3. Filter by level (error, warning, etc.)
4. Tail follows new logs

## Troubleshooting

### Cannot connect to TimescaleDB

**Error**: "Unable to connect to server"

**Solution**:
1. Verify timescaledb container is running: `docker ps`
2. Check if TimescaleDB is healthy: `docker-compose ps`
3. Test database directly: 
   ```bash
   psql -h timescaledb -U mltrader -d btc_ml_production
   ```
4. Verify network: `docker network inspect btc-ml-network`

### Login fails

**Error**: "Invalid username or password"

**Solution**:
1. Check .env PGADMIN_PASSWORD is set
2. Verify entrypoint.sh executed correctly
3. Check docker logs: `docker logs btc-ml-pgadmin`
4. Reset to defaults in docker-compose.yml

### Query timeout

**Error**: "Query interrupted"

**Solution**:
1. Set statement timeout in PostgreSQL:
   ```sql
   SET statement_timeout TO '5 min';
   ```
2. Kill long-running query:
   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE query ILIKE '%problematic_query%';
   ```

### pgAdmin data persisted but not accessible

**Solution**:
1. Check volume: `docker volume ls | grep pgadmin`
2. Verify volume permissions: `docker inspect btc-ml-pgadmin`
3. Clean and restart: 
   ```bash
   docker-compose down -v  # WARNING: deletes pgadmin data
   docker-compose up pgadmin
   ```

## Security Best Practices

1. **Change default password**:
   - Set PGADMIN_PASSWORD in .env to strong password
   - Never use "admin" or "password"

2. **Restrict network access**:
   - pgAdmin only accessible from trusted networks
   - Do NOT expose on public internet
   - Use reverse proxy (nginx) with authentication if remote access needed

3. **Disable unnecessary features**:
   - Tools → Preferences → System
   - Disable features not needed
   - Limit user role capabilities

4. **Monitor user activity**:
   - Tools → User Management
   - Review who has access
   - Audit logs for sensitive operations

5. **Backup credentials**:
   - Store pgpass file securely
   - Use secrets management (Vault, AWS Secrets Manager)
   - Rotate passwords monthly

## Advanced Configuration

### Reverse Proxy Setup (nginx)

```nginx
location /pgadmin4 {
    proxy_pass http://pgadmin:80;
    proxy_set_header X-Script-Name /pgadmin4;
    proxy_set_header Host $host;
}
```

### Environment Variable Reference

```bash
# Available in dpage/pgadmin4:latest
PGADMIN_DEFAULT_EMAIL       # Login email (default: admin@example.com)
PGADMIN_DEFAULT_PASSWORD    # Login password
SCRIPT_NAME                 # Web path (default: /, set to /pgadmin4)
PGADMIN_CONFIG_ENHANCED_COOKIE_PROTECTION    # true/false
PGADMIN_CONFIG_COOKIE_DEFAULT__SAMESITE      # 'Lax' or 'Strict'
PGADMIN_CONFIG_CONSOLE_LOG_LEVEL            # 10 (debug) to 40 (error)
```

## Performance Tips

1. **For large tables** (>1M rows):
   - Use WHERE clause to limit results
   - Use LIMIT when browsing data
   - Avoid SELECT * queries

2. **Query optimization**:
   - Use EXPLAIN ANALYZE before complex queries
   - Create indexes on frequently filtered columns
   - Check query plans in Tools → Query Tool

3. **Server resource limits**:
   - Monitor memory: `docker stats btc-ml-pgadmin`
   - Limit connections in pg_hba.conf if needed

## Integration with Data Pipeline

pgAdmin helps monitor the data pipeline:

1. **Watch candle ingestion**:
   ```sql
   SELECT COUNT(*) as candles_per_minute,
     MAX(time) as latest
   FROM candles_1m
   WHERE time > NOW() - INTERVAL '5 minutes'
   GROUP BY DATE_TRUNC('minute', time);
   ```

2. **Track gap detection**:
   ```sql
   SELECT * FROM data_quality_logs 
   WHERE created_at > NOW() - INTERVAL '1 hour'
   ORDER BY created_at DESC;
   ```

3. **Monitor feature updates**:
   ```sql
   SELECT COUNT(*) as cached, MAX(time) as latest
   FROM feature_cache;
   ```

4. **Check backfill progress**:
   ```sql
   SELECT 
     event_type,
     COUNT(*) as count,
     SUM(COALESCE(candles_recovered, 0)) as recovered
   FROM data_quality_logs
   WHERE created_at > NOW() - INTERVAL '24 hours'
   GROUP BY event_type;
   ```

## Support

For pgAdmin documentation: https://www.pgadmin.org/docs/

For issues:
1. Check logs: `docker-compose logs pgadmin`
2. Verify database: `psql -h localhost -U mltrader -d btc_ml_production`
3. Review pgAdmin status: Tools → Server Activity
