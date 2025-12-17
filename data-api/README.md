# Data API Service

Secure REST API for accessing BTC trading data from TimescaleDB with **API Key authentication** and HTTPS support.

> **ðŸ†• Enhanced Schema v2.0:** This API now supports the enhanced database schema that captures 100% of market-analyzer log data. See [Database Schema](#database-schema) section for migration details.

## Features

- ðŸ” **API Key authentication** - Bearer token protection
- âœ… **HTTPS by default** with self-signed certificates
- âœ… **Custom certificate support** via mounted volumes
- âœ… **5 REST endpoints** for different data types
- âœ… **Query parameters** for filtering and pagination
- âœ… **CORS enabled** for web applications
- âœ… **Health checks** for monitoring
- âœ… **JSON responses** with proper error handling

## Quick Start

### 1. Generate API Key

**Generate a secure API key:**
```bash
# On Linux/Mac
openssl rand -base64 32

# Or use Python
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Example output:
# k7x9mP2nQ4vL8wR5tY6uI3oP1aS0dF9gH2jK4lM7nB5c
```

### 2. Environment Variables

Required in `.env`:
```bash
# API Key (REQUIRED - use output from above command)
DATA_API_KEY=k7x9mP2nQ4vL8wR5tY6uI3oP1aS0dF9gH2jK4lM7nB5c

# Database connection
DATABASE_URL=postgresql://mltrader:your_password@timescaledb:5432/btc_ml_production

# Optional - TLS configuration
USE_TLS=true              # Set to "false" to disable HTTPS
PORT=8443                 # Default: 8443 (HTTPS) or 8080 (HTTP)
TLS_CERT_FILE=/certs/custom.crt  # Optional: custom certificate
TLS_KEY_FILE=/certs/custom.key   # Optional: custom key
```

### 3. Start Service

**With Docker Compose:**
```bash
docker-compose up data-api
```

**Standalone:**
```bash
docker build -t data-api ./data-api
docker run -p 8443:8443 --env-file .env data-api
```

### 4. Test Connection

**Include API key in Authorization header:**
```bash
curl -k -H "Authorization: Bearer k7x9mP2nQ4vL8wR5tY6uI3oP1aS0dF9gH2jK4lM7nB5c" \
  https://localhost:8443/api/v1/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "service": "data-api",
    "time": "2025-12-16T20:15:00Z"
  }
}
```

## Authentication

### API Key Format

All protected endpoints require an API key in the `Authorization` header:

```
Authorization: Bearer <your-api-key>
```

### Public Endpoints (No Auth Required)

- `GET /api/v1/health` - Health check

### Protected Endpoints (Auth Required)

All data endpoints require authentication:
- `GET /api/v1/candles`
- `GET /api/v1/data-quality-logs`
- `GET /api/v1/llm-analysis`
- `GET /api/v1/market-analysis`
- `GET /api/v1/market-signals`

### Authentication Errors

**Missing header:**
```json
{
  "success": false,
  "error": "Unauthorized: missing or invalid Authorization header. Use: Authorization: Bearer <api-key>"
}
```
HTTP Status: `401 Unauthorized`

**Invalid API key:**
```json
{
  "success": false,
  "error": "Unauthorized: invalid API key"
}
```
HTTP Status: `401 Unauthorized`

## API Endpoints

All endpoints return JSON in this format:
```json
{
  "success": true,
  "data": [...],
  "count": 100,
  "error": "error message if failed"
}
```

### 1. Health Check

**GET** `/api/v1/health`

Check service and database connectivity.

**Example:**
```bash
# Health check (no auth required)
curl -k https://localhost:8443/api/v1/health
```

---

### 2. Candles (OHLCV Data)

**GET** `/api/v1/candles`

Get 1-minute candle data. **Requires authentication.**

**Query Parameters:**
- `limit` (int, default: 100, max: 10000) - Number of records
- `start_time` (ISO 8601) - Filter by start time
- `end_time` (ISO 8601) - Filter by end time

**Example:**
```bash
# Set your API key
API_KEY="k7x9mP2nQ4vL8wR5tY6uI3oP1aS0dF9gH2jK4lM7nB5c"

# Get latest 100 candles
curl -k -H "Authorization: Bearer $API_KEY" \
  "https://localhost:8443/api/v1/candles?limit=100"

# Get candles in time range
curl -k -H "Authorization: Bearer $API_KEY" \
  "https://localhost:8443/api/v1/candles?start_time=2025-12-16T19:00:00Z&end_time=2025-12-16T20:00:00Z"
```

**Response Fields:**
```json
{
  "time": "2025-12-16T20:09:00Z",
  "open": "87593.00000000",
  "high": "87687.18000000",
  "low": "87593.00000000",
  "close": "87639.81000000",
  "volume": "6.79507000",
  "quote_asset_volume": "595495.76825340",
  "taker_buy_base_asset_volume": "5.70270000",
  "taker_buy_quote_asset_volume": "499746.56648440",
  "number_of_trades": 3718,
  "spread_bps": "10.7463",
  "taker_buy_ratio": "0.8392",
  "mid_price": "87640.09000000"
}
```

---

### 3. Data Quality Logs

**GET** `/api/v1/data-quality-logs`

Get data quality events (gaps, errors, cleanups). **Requires authentication.**

**Query Parameters:**
- `limit` (int, default: 100, max: 1000)
- `event_type` (string) - Filter by type: `gap_detected`, `error`, `cleanup`
- `resolved` (boolean) - Filter by resolution status: `true`, `false`

**Example:**
```bash
API_KEY="your_api_key_here"

# Get all unresolved gaps
curl -k -H "Authorization: Bearer $API_KEY" \
  "https://localhost:8443/api/v1/data-quality-logs?event_type=gap_detected&resolved=false"

# Get recent cleanup logs
curl -k -H "Authorization: Bearer $API_KEY" \
  "https://localhost:8443/api/v1/data-quality-logs?event_type=cleanup&limit=10"
```

**Response Fields:**
```json
{
  "id": 58,
  "event_type": "gap_detected",
  "gap_start": "2025-12-16T14:33:00Z",
  "gap_end": "2025-12-16T14:34:00Z",
  "candles_missing": 1,
  "candles_recovered": null,
  "source": "system",
  "error_message": null,
  "resolved": false,
  "resolved_at": null,
  "created_at": "2025-12-16T14:35:00.283671Z"
}
```

---

### 4. LLM Analysis

**GET** `/api/v1/llm-analysis`

Get AI predictions and analysis from DeepSeek/Ollama. **Requires authentication.**

**Query Parameters:**
- `limit` (int, default: 50, max: 1000)

**Example:**
```bash
curl -k -H "Authorization: Bearer $API_KEY" \
  "https://localhost:8443/api/v1/llm-analysis?limit=20"
```

**Response Fields (Original Schema v1.0):**
```json
{
  "id": 24,
  "analysis_time": "2025-12-16T20:08:21.938424Z",
  "price": "87584.01000000",
  "prediction_direction": "BULLISH",
  "prediction_confidence": "LOW",
  "predicted_price_1h": null,
  "predicted_price_4h": null,
  "key_levels": "S: $87,417 | R: $87,553",
  "reasoning": "The analysis indicates weak bullish sentiment...",
  "full_response": "**BTCUSDT Price Prediction Analysis**...",
  "model_name": "deepseek-r1:32b",
  "response_time_seconds": 14.25,
  "actual_price_1h": null,
  "actual_price_4h": null,
  "direction_correct_1h": null,
  "direction_correct_4h": null,
  "created_at": "2025-12-16T20:08:21.939582Z"
}
```

**Response Fields (Enhanced Schema v2.0):**

After applying the LLM analyst migration 002, the API returns complete market context:

```json
{
  "id": 24,
  "analysis_time": "2025-12-16T20:08:21.938424Z",
  "price": "87584.01000000",
  "prediction_direction": "BULLISH",
  "prediction_confidence": "MEDIUM",
  "predicted_price_1h": "87800.00",
  "predicted_price_4h": "88200.00",
  "key_levels": "S: $87,417 | R: $87,753",
  "reasoning": "The bullish CHoCH suggests potential reversal...",
  "full_response": "**BTCUSDT Price Prediction Analysis**...",
  "model_name": "deepseek-r1:8b",
  "response_time_seconds": 4.25,
  
  // 🆕 Enhanced fields (NEW in v2.0)
  "invalidation_level": "87100.00",
  "critical_support": "87100.00",
  "critical_resistance": "87800.00",
  
  // 🆕 Market context at time of prediction
  "market_context": {
    "signal_type": "WEAK_BUY",
    "signal_direction": "LONG",
    "signal_confidence": 44.5,
    "smc_bias": "BULLISH",
    "price_zone": "DISCOUNT",
    "action_recommendation": "WAIT",
    "nearest_support": 87100,
    "nearest_resistance": 87750
  },
  
  // 🆕 Signal factors the LLM analyzed
  "signal_factors_used": [
    {"description": "Bullish CHoCH - potential trend reversal up", "weight": 30},
    {"description": "At strong resistance $87,753 (0.22% away)", "weight": -30},
    {"description": "4h trend: DOWNTREND (100% strength), EMA: BEARISH", "weight": -25}
  ],
  
  // 🆕 SMC bias at analysis time
  "smc_bias_at_analysis": "BULLISH",
  
  // 🆕 Trends at analysis time
  "trends_at_analysis": {
    "5m": {"direction": "UPTREND", "strength": 0.8},
    "15m": {"direction": "UPTREND", "strength": 0.6},
    "1h": {"direction": "SIDEWAYS", "strength": 0.3},
    "4h": {"direction": "DOWNTREND", "strength": 1.0}
  },
  
  // 🆕 Warnings at analysis time
  "warnings_at_analysis": [
    {"type": "CLOSE_TO_RESISTANCE", "message": "Near $87,750", "severity": "MEDIUM"}
  ],
  
  // Accuracy tracking (filled after 1h/4h)
  "actual_price_1h": "87820.00",
  "actual_price_4h": null,
  "direction_correct_1h": true,
  "direction_correct_4h": null,
  "created_at": "2025-12-16T20:08:21.939582Z"
}
```

**Purpose of Enhanced Fields:**

The market context fields store exactly what data the LLM saw when making its prediction. This enables:
- Analysis of which market conditions lead to accurate predictions
- Comparison of LLM prediction vs market-analyzer signal
- Debugging why certain predictions failed
- Training data for future model improvements

**Migration Required:**

To enable enhanced fields, apply the LLM analyst migration:
```bash
docker cp llm-analyst/migrations/002_llm_analyst_enhanced.sql btc-ml-timescaledb:/tmp/
docker exec -it btc-ml-timescaledb psql -U mltrader -d btc_ml_production \
  -f /tmp/002_llm_analyst_enhanced.sql
```

---

### 5. Market Analysis

**GET** `/api/v1/market-analysis`

Get technical analysis from market-analyzer service.

**Query Parameters:**
- `limit` (int, default: 50, max: 1000)
- `signal_type` (string) - Filter by signal: `STRONG_BUY`, `BUY`, `WEAK_BUY`, `NEUTRAL`, `WEAK_SELL`, `SELL`, `STRONG_SELL`

**Example:**
```bash
# Get latest analysis
curl -k "https://localhost:8443/api/v1/market-analysis?limit=10"

# Get only strong buy signals
curl -k "https://localhost:8443/api/v1/market-analysis?signal_type=STRONG_BUY"
```

**Response Fields (Enhanced Schema v2.0):**

After upgrading to the enhanced schema, the API returns complete log data:

```json
{
  "id": 136,
  "analysis_time": "2025-12-16T20:12:00Z",
  "price": "87445.95000000",
  
  // Signal
  "signal_type": "WEAK_BUY",
  "signal_direction": "LONG",
  "signal_confidence": 44.23,
  
  // Trade setup
  "entry_price": null,
  "stop_loss": null,
  "take_profit_1": null,
  "take_profit_2": null,
  "take_profit_3": null,
  "risk_reward_ratio": null,
  
  // ðŸ†• Signal reasoning with weights (NEW in v2.0)
  "signal_factors": [
    {
      "description": "At strong resistance $87,769 (0.03% away)",
      "weight": -30,
      "type": "bearish"
    },
    {
      "description": "Bullish CHoCH - potential trend reversal up",
      "weight": 30,
      "type": "bullish"
    }
  ],
  
  // Enhanced trends with EMA and structure (v2.0)
  "trends": {
    "5m": {
      "direction": "UPTREND",
      "strength": 1.0,
      "ema": "BULLISH",
      "structure": "HH/HL"
    },
    "1h": {
      "direction": "UPTREND",
      "strength": 0.7,
      "ema": "MIXED",
      "structure": "HH/HL"
    },
    "4h": {
      "direction": "DOWNTREND",
      "strength": 1.0,
      "ema": "BEARISH",
      "structure": "LH/LL"
    }
  },
  
  // ðŸ†• Complete pivot data (NEW in v2.0)
  "pivot_daily": "87210.45333333",
  "pivot_r3_traditional": "94180.00",
  "pivot_r2_traditional": "92116.00",
  "pivot_r1_traditional": "89274.00",
  "pivot_s1_traditional": "84368.00",
  "pivot_s2_traditional": "82304.00",
  "pivot_s3_traditional": "79462.00",
  "pivot_r3_fibonacci": "92116.00",
  "pivot_r2_fibonacci": "90242.00",
  "pivot_r1_fibonacci": "89085.00",
  "pivot_s1_fibonacci": "85336.00",
  "pivot_s2_fibonacci": "84179.00",
  "pivot_s3_fibonacci": "82304.00",
  "pivot_r4_camarilla": "89130.00",
  "pivot_r3_camarilla": "87781.00",
  "pivot_s3_camarilla": "85083.00",
  "pivot_s4_camarilla": "83734.00",
  "pivot_confluence_zones": [
    {
      "price": 87332,
      "type": "support",
      "strength": 0.2,
      "distance_pct": 0.46,
      "methods": ["Camarilla"]
    }
  ],
  "price_vs_pivot": "ABOVE",
  
  // ðŸ†• Complete SMC data (NEW in v2.0)
  "smc_bias": "BULLISH",
  "smc_price_zone": "DISCOUNT",
  "smc_equilibrium": "87599.64000000",
  "smc_order_blocks": [
    {
      "type": "bullish",
      "low": 87132,
      "high": 87588,
      "strength": 1.0,
      "distance_pct": 0.2
    }
  ],
  "smc_fvgs": [
    {
      "type": "bullish",
      "low": 86426,
      "high": 86820,
      "unfilled": true
    }
  ],
  "smc_breaks": [
    {
      "type": "CHoCH",
      "direction": "BULLISH",
      "price": 86535
    }
  ],
  "smc_liquidity": {
    "buy_side": [87782, 87793, 87800],
    "sell_side": [87607, 87537, 87338]
  },
  
  // ðŸ†• All support/resistance levels (NEW in v2.0)
  "support_levels": [
    {
      "price": 87556,
      "strength": 0.62,
      "touches": 11,
      "timeframes": ["15m", "5m"],
      "distance_pct": 0.21
    },
    {
      "price": 87339,
      "strength": 0.78,
      "touches": 13,
      "timeframes": ["5m"],
      "distance_pct": 0.45
    }
  ],
  "resistance_levels": [
    {
      "price": 87769,
      "strength": 1.0,
      "touches": 28,
      "timeframes": ["15m", "5m"],
      "distance_pct": 0.03
    }
  ],
  
  // ðŸ†• Momentum for all timeframes (NEW in v2.0)
  "momentum": {
    "5m": {
      "rsi": 52.1,
      "volume_ratio": 0.33,
      "taker_buy_ratio": 0.67
    },
    "15m": {
      "rsi": 54.9,
      "volume_ratio": 0.19,
      "taker_buy_ratio": 0.63
    },
    "1h": {
      "rsi": 55.6,
      "volume_ratio": 0.10,
      "taker_buy_ratio": 0.55
    },
    "4h": {
      "rsi": 44.2,
      "volume_ratio": 0.30,
      "taker_buy_ratio": 0.53
    },
    "1d": {
      "rsi": 40.9,
      "volume_ratio": 1.04,
      "taker_buy_ratio": 0.49
    }
  },
  
  // ðŸ†• Market structure (NEW in v2.0)
  "structure_pattern": "CONTRACTING",
  "structure_last_high": "88175.98",
  "structure_last_low": "86107.43",
  
  // ðŸ†• Warnings and alerts (NEW in v2.0)
  "warnings": [
    {
      "type": "CLOSE_TO_SUPPORT",
      "message": "CLOSE TO STRONG SUPPORT ($87,556) - Short risky before break!",
      "severity": "high"
    },
    {
      "type": "TREND_CONFLICT",
      "message": "TREND CONFLICT: 15m=UPTREND, 4h=DOWNTREND - Be cautious!",
      "severity": "medium"
    }
  ],
  "action_recommendation": "WAIT",
  
  // Summary and metadata
  "summary": "BTC $87,446 - WEAK_BUY (44% confidence)...",
  "signal_changed": false,
  "previous_signal": "WEAK_BUY",
  "created_at": "2025-12-16T20:13:06.838209Z"
}
```

**Note:** Fields marked with ðŸ†• are available after applying the enhanced schema migration. See the [Database Schema](#database-schema) section below for migration instructions.

---

### 6. Market Signals

**GET** `/api/v1/market-signals`

Get signal change events only (when signals shift).

**Query Parameters:**
- `limit` (int, default: 100, max: 1000)

**Example:**
```bash
curl -k "https://localhost:8443/api/v1/market-signals?limit=50"
```

**Response Fields (Enhanced Schema v2.0):**
```json
{
  "id": 5,
  "signal_time": "2025-12-16T20:11:00Z",
  "signal_type": "WEAK_BUY",
  "signal_direction": "LONG",
  "signal_confidence": 40.92,
  "price": "87560.02000000",
  "entry_price": null,
  "stop_loss": null,
  "take_profit_1": null,
  "take_profit_2": null,
  "take_profit_3": null,
  "risk_reward_ratio": null,
  "previous_signal_type": "NEUTRAL",
  "previous_direction": "NONE",
  "summary": "BTC $87,560 - WEAK_BUY (41% confidence)...",
  
  // ðŸ†• Enhanced key_reasons with weights (JSONB in v2.0)
  "key_reasons": [
    {
      "description": "At strong resistance $87,753 (0.22% away)",
      "weight": -30
    },
    {
      "description": "Bullish CHoCH - potential trend reversal up",
      "weight": 30
    },
    {
      "description": "4h trend: DOWNTREND (100% strength), EMA: BEARISH",
      "weight": -25
    }
  ],
  
  "created_at": "2025-12-16T20:12:06.649739Z"
}
```

**Note:** In schema v1.0, `key_reasons` was a simple text array. In v2.0, it's JSONB with weight information matching the log output.

---

## TLS/HTTPS Configuration

### Self-Signed Certificate (Default)

The service generates a self-signed certificate automatically on startup. Use `-k` flag with curl to ignore SSL warnings:

```bash
curl -k https://localhost:8443/api/v1/health
```

**Browser Warning:**
Browsers will show a security warning. Click "Advanced" â†’ "Proceed to localhost".

---

### Custom Certificate (Production)

**1. Place your certificate files:**
```
project_root/
â”œâ”€â”€ certs/
â”‚   â”œâ”€â”€ server.crt  # Your SSL certificate
â”‚   â””â”€â”€ server.key  # Your private key
```

**2. Update `.env`:**
```bash
TLS_CERT_FILE=/certs/server.crt
TLS_KEY_FILE=/certs/server.key
```

**3. Update `docker-compose.yml`:**
```yaml
data-api:
  volumes:
    - ./certs:/certs:ro  # Mount certificate directory
```

**4. Restart service:**
```bash
docker-compose restart data-api
```

---

### Disable HTTPS (Development Only)

**Update `.env`:**
```bash
USE_TLS=false
PORT=8080
```

**Access via HTTP:**
```bash
curl http://localhost:8080/api/v1/health
```

---

## Integration Examples

### Python

```python
import requests
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for self-signed cert
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Your API key
API_KEY = "k7x9mP2nQ4vL8wR5tY6uI3oP1aS0dF9gH2jK4lM7nB5c"

# Get latest candles
response = requests.get(
    'https://localhost:8443/api/v1/candles',
    params={'limit': 100},
    headers={'Authorization': f'Bearer {API_KEY}'},
    verify=False  # Skip SSL verification for self-signed
)

if response.status_code == 200:
    data = response.json()
    if data['success']:
        candles = data['data']
        print(f"Got {data['count']} candles")
elif response.status_code == 401:
    print("Authentication failed - check your API key")
else:
    print(f"Error: {response.status_code}")
```

### JavaScript/Node.js

```javascript
const https = require('https');
const axios = require('axios');

// Your API key
const API_KEY = 'k7x9mP2nQ4vL8wR5tY6uI3oP1aS0dF9gH2jK4lM7nB5c';

// Create agent that accepts self-signed certificates
const agent = new https.Agent({
  rejectUnauthorized: false
});

// Get market analysis
axios.get('https://localhost:8443/api/v1/market-analysis', {
  params: { limit: 10 },
  headers: { 'Authorization': `Bearer ${API_KEY}` },
  httpsAgent: agent
})
.then(response => {
  console.log(`Got ${response.data.count} analyses`);
  console.log(response.data.data[0]);
})
.catch(error => {
  if (error.response?.status === 401) {
    console.error('Authentication failed - check your API key');
  } else {
    console.error('Error:', error.message);
  }
});
```

### Go

```go
package main

import (
    "crypto/tls"
    "encoding/json"
    "fmt"
    "net/http"
)

func main() {
    apiKey := "k7x9mP2nQ4vL8wR5tY6uI3oP1aS0dF9gH2jK4lM7nB5c"
    
    // Create client that accepts self-signed certificates
    tr := &http.Transport{
        TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
    }
    client := &http.Client{Transport: tr}

    req, _ := http.NewRequest("GET", "https://localhost:8443/api/v1/candles?limit=10", nil)
    req.Header.Set("Authorization", "Bearer "+apiKey)
    
    resp, err := client.Do(req)
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()

    if resp.StatusCode == 401 {
        fmt.Println("Authentication failed - check your API key")
        return
    }

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)
    
    fmt.Printf("Success: %v, Count: %.0f\n", 
        result["success"], result["count"])
}
```

---

## Monitoring

### Health Check Endpoint

```bash
# Check if service is healthy
curl -k https://localhost:8443/api/v1/health
```

**Healthy Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "service": "data-api",
    "time": "2025-12-16T20:15:00Z"
  }
}
```

**Unhealthy Response (503):**
```json
{
  "success": false,
  "error": "Database unavailable"
}
```

### Docker Health Check

Docker Compose automatically monitors health:
```bash
docker-compose ps data-api
```

**Output:**
```
NAME          STATUS
data-api      Up 5 minutes (healthy)
```

### Logs

```bash
# View real-time logs
docker-compose logs -f data-api

# Last 100 lines
docker-compose logs --tail=100 data-api
```

---

## Troubleshooting

### Issue: "Unauthorized: missing or invalid Authorization header"

**Cause:** Missing or malformed API key in request

**Solution:**
```bash
# âŒ Wrong - no auth header
curl -k https://localhost:8443/api/v1/candles

# âŒ Wrong - missing "Bearer " prefix
curl -k -H "Authorization: your_key" https://localhost:8443/api/v1/candles

# âœ… Correct
curl -k -H "Authorization: Bearer your_key" https://localhost:8443/api/v1/candles
```

---

### Issue: "Unauthorized: invalid API key"

**Cause:** API key doesn't match the one in `.env`

**Solution:**
1. Check your API key in `.env`:
   ```bash
   grep DATA_API_KEY .env
   ```
2. Make sure you're using the exact same key in requests
3. Restart service after changing `.env`:
   ```bash
   docker-compose restart data-api
   ```

---

### Issue: "connection refused"

**Cause:** Service not started or wrong port

**Solution:**
```bash
# Check service status
docker-compose ps data-api

# Check logs
docker-compose logs data-api

# Verify port in .env
grep PORT .env
```

---

### Issue: "SSL certificate problem"

**Cause:** Self-signed certificate not trusted

**Solution:**

**For curl:**
```bash
curl -k https://localhost:8443/api/v1/health
```

**For Python:**
```python
requests.get(url, verify=False)
```

**For production:** Use a proper certificate from Let's Encrypt or your CA.

---

### Issue: "Database unavailable"

**Cause:** TimescaleDB not running or wrong credentials

**Solution:**
```bash
# Check database
docker-compose ps timescaledb

# Test connection
docker-compose exec timescaledb psql -U mltrader -d btc_ml_production -c "SELECT 1;"

# Verify DATABASE_URL in .env
grep DATABASE_URL .env
```

---

### Issue: "Query error: permission denied"

**Cause:** Database user lacks permissions

**Solution:**
```sql
-- Grant read permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mltrader;
```

---

## Performance

### Response Times

- Health check: ~5ms
- Candles (100 records): ~20ms
- Market analysis (50 records): ~30ms
- LLM analysis (50 records): ~40ms

### Limits

- Max candles per request: 10,000
- Max logs/analysis per request: 1,000
- Concurrent connections: 100
- Request timeout: 15 seconds

### Optimization Tips

1. **Use pagination** - Request smaller chunks with `limit` parameter
2. **Filter by time** - Use `start_time`/`end_time` for candles
3. **Filter by type** - Use `signal_type` or `event_type` filters
4. **Cache responses** - Results don't change after insertion

---

## Security Considerations

### Production Deployment

1. **Use proper SSL certificate**
   - Get certificate from Let's Encrypt or your CA
   - Do NOT use self-signed certificates in production

2. **Secure API Key Management**
   - Generate strong keys (32+ characters): `openssl rand -base64 32`
   - Store in environment variables, never hardcode
   - Use secrets management (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Rotate keys regularly (quarterly recommended)
   - Use different keys for dev/staging/production

3. **Additional Security Layers**
   - **Rate limiting** (add with nginx/traefik)
   - **IP whitelisting** (restrict to known IPs)
   - **Request logging** (monitor for suspicious activity)
   - **HTTPS only** (never disable TLS in production)

4. **Network isolation**
   - Keep database on private network
   - Expose only data-api to public
   - Use firewall rules
   - Consider VPN or VPC for internal access

5. **Read-only database user**
   - Grant only SELECT permissions
   - Prevent accidental modifications
   ```sql
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO mltrader;
   REVOKE INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public FROM mltrader;
   ```

6. **Monitoring & Alerts**
   - Log all failed authentication attempts
   - Alert on suspicious patterns (rapid requests, etc.)
   - Monitor for API key leaks in public repos

### API Key Security Best Practices

```bash
# âœ… DO: Use environment variables
export DATA_API_KEY="$(openssl rand -base64 32)"
docker-compose up data-api

# âœ… DO: Use secrets management
aws secretsmanager get-secret-value --secret-id data-api-key

# âŒ DON'T: Hardcode in scripts
curl -H "Authorization: Bearer hardcoded_key_123"  # BAD!

# âŒ DON'T: Commit to git
git add .env  # BAD! Use .env.example instead

# âŒ DON'T: Share in chat/email
# Use secure channels for key distribution
```

---

## Known Issues

### 1. Market signals missing trade setup

**Symptom:** `entry_price`, `stop_loss`, `take_profit_*` are NULL

**Cause:** Signal confidence < 60% doesn't generate setup (see issue in market-analyzer)

**Workaround:** Apply the fix in `market-analyzer_signals_fix.py`

---

### 2. Data quality logs show "error" for cleanups

**Symptom:** Cleanup logs have `event_type='error'`

**Cause:** Logging function uses wrong event_type

**Workaround:** Apply the fix in `timescaledb_cleanup_fix.sql`

---

## Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `DATABASE_URL` | *(none)* | **Yes** | PostgreSQL connection string |
| `API_KEY` | *(none)* | **Yes** | API key for authentication (32+ chars) |
| `USE_TLS` | `true` | No | Enable HTTPS (`true`/`false`) |
| `PORT` | `8443` (TLS) or `8080` | No | Server port |
| `TLS_CERT_FILE` | `/certs/server.crt` | No | Path to SSL certificate |
| `TLS_KEY_FILE` | `/certs/server.key` | No | Path to SSL private key |

---

## API Response Format

### Success Response
```json
{
  "success": true,
  "data": [...],
  "count": 100
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error description"
}
```

### HTTP Status Codes

- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `500 Internal Server Error` - Server/database error
- `503 Service Unavailable` - Database connection failed

---

## Database Schema

### Schema Versions

**v1.0 (Original)**
- Basic signal and trend data
- Single nearest support/resistance levels
- Limited momentum (1h only)
- Simple text array for key_reasons

**v2.0 (Enhanced - Current)**
- ✅ Complete log data capture
- ✅ All pivot methods with confluence zones
- ✅ Complete SMC analysis (order blocks, FVGs, breaks, liquidity)
- ✅ All support/resistance levels with metadata
- ✅ Momentum indicators for all timeframes (5m, 15m, 1h, 4h, 1d)
- ✅ Market structure patterns and swing points
- ✅ Warnings and risk alerts
- ✅ Weighted signal factors (matching log bar charts)
- ✅ Enhanced key_reasons with weights (JSONB)
- ✅ **LLM market context storage** (what the LLM saw when predicting)
- ✅ **LLM accuracy analysis by market conditions**

### Migration to v2.0

To upgrade your database to the enhanced schema:

```bash
# 1. Backup your database first!
docker exec btc-ml-timescaledb pg_dump -U mltrader -d btc_ml_production > backup.sql

# 2. Apply market-analyzer enhanced schema
docker cp market-analyzer/migrations/002_enhanced_schema.sql btc-ml-timescaledb:/tmp/
docker exec -it btc-ml-timescaledb psql -U mltrader -d btc_ml_production \
  -f /tmp/002_enhanced_schema.sql

# 3. Apply LLM analyst enhanced schema (NEW)
docker cp llm-analyst/migrations/002_llm_analyst_enhanced.sql btc-ml-timescaledb:/tmp/
docker exec -it btc-ml-timescaledb psql -U mltrader -d btc_ml_production \
  -f /tmp/002_llm_analyst_enhanced.sql

# 4. Verify migration
docker exec -it btc-ml-timescaledb psql -U mltrader -d btc_ml_production \
  -c "\d market_analysis" | grep signal_factors
docker exec -it btc-ml-timescaledb psql -U mltrader -d btc_ml_production \
  -c "\d llm_analysis" | grep market_context

# 5. Rebuild and restart
docker-compose build data-api llm-analyst
docker-compose restart data-api llm-analyst
```

**Files Needed:**
- `market-analyzer/migrations/002_enhanced_schema.sql` - Market analysis enhancement
- `llm-analyst/migrations/002_llm_analyst_enhanced.sql` - LLM analysis enhancement (NEW)
- `migrations/MIGRATION_GUIDE.md` - Detailed instructions with code examples
- `migrations/LOG_TO_DATABASE_MAPPING.md` - Complete field mapping reference
- `migrations/MIGRATION_GUIDE.md` - Detailed instructions with code examples
- `migrations/LOG_TO_DATABASE_MAPPING.md` - Complete field mapping reference

### Go Struct Updates for v2.0

After applying the database migration, update `main.go` structs:

```go
type MarketAnalysis struct {
    // ... existing fields ...
    
    // NEW in v2.0
    SignalFactors        json.RawMessage `json:"signal_factors,omitempty"`
    PivotR3Traditional   *float64        `json:"pivot_r3_traditional,string,omitempty"`
    PivotR2Traditional   *float64        `json:"pivot_r2_traditional,string,omitempty"`
    // ... (add all pivot fields)
    PivotConfluenceZones json.RawMessage `json:"pivot_confluence_zones,omitempty"`
    SMCOrderBlocks       json.RawMessage `json:"smc_order_blocks,omitempty"`
    SMCFVGs              json.RawMessage `json:"smc_fvgs,omitempty"`
    SMCBreaks            json.RawMessage `json:"smc_breaks,omitempty"`
    SMCLiquidity         json.RawMessage `json:"smc_liquidity,omitempty"`
    SupportLevels        json.RawMessage `json:"support_levels,omitempty"`
    ResistanceLevels     json.RawMessage `json:"resistance_levels,omitempty"`
    Momentum             json.RawMessage `json:"momentum,omitempty"`
    StructurePattern     *string         `json:"structure_pattern,omitempty"`
    StructureLastHigh    *float64        `json:"structure_last_high,string,omitempty"`
    StructureLastLow     *float64        `json:"structure_last_low,string,omitempty"`
    Warnings             json.RawMessage `json:"warnings,omitempty"`
    ActionRecommendation *string         `json:"action_recommendation,omitempty"`
}

type MarketSignal struct {
    // ... existing fields ...
    
    // CHANGED in v2.0: TEXT[] -> JSONB
    KeyReasons           json.RawMessage `json:"key_reasons,omitempty"`
}
```

Update the SELECT query to include all new columns. See `migrations/MIGRATION_GUIDE.md` for complete code.

### What You Get After Migration

The API will return **exactly the same data as the market-analyzer logs**:

**Before (v1.0):** ~30% of log data
```json
{
  "signal_type": "WEAK_BUY",
  "trends": {"1h": {"direction": "UPTREND"}},
  "nearest_support": "87556.00"
}
```

**After (v2.0):** 100% of log data
```json
{
  "signal_type": "WEAK_BUY",
  "signal_factors": [{"description": "...", "weight": -30}],
  "trends": {"5m": {...}, "15m": {...}, "1h": {...}, "4h": {...}, "1d": {...}},
  "pivot_r3_traditional": "94180.00",
  "pivot_confluence_zones": [...],
  "smc_order_blocks": [...],
  "smc_fvgs": [...],
  "support_levels": [{"price": 87556, "strength": 0.62, "touches": 11}],
  "resistance_levels": [...],
  "momentum": {"5m": {...}, "1h": {...}, "4h": {...}},
  "structure_pattern": "CONTRACTING",
  "warnings": [...]
}
```

---

## Development

### Build Locally

```bash
cd data-api
go mod download
go build -o data-api main.go
./data-api
```

### Run Tests

```bash
# Set your API key
export API_KEY="your_api_key_here"

# Test health endpoint (no auth needed)
curl -k https://localhost:8443/api/v1/health

# Test protected endpoints (auth required)
curl -k -H "Authorization: Bearer $API_KEY" "https://localhost:8443/api/v1/candles?limit=1"
curl -k -H "Authorization: Bearer $API_KEY" "https://localhost:8443/api/v1/data-quality-logs?limit=1"
curl -k -H "Authorization: Bearer $API_KEY" "https://localhost:8443/api/v1/llm-analysis?limit=1"
curl -k -H "Authorization: Bearer $API_KEY" "https://localhost:8443/api/v1/market-analysis?limit=1"
curl -k -H "Authorization: Bearer $API_KEY" "https://localhost:8443/api/v1/market-signals?limit=1"
```

### Add New Endpoints

1. Define response struct in `main.go`
2. Add handler function
3. Register route in `main()`
4. Update this README

---

## License

Part of the quaint2 BTC ML trading system.