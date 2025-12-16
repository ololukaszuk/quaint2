# LLM Market Analyst

AI-powered market commentary and prediction service using DeepSeek (via Ollama).

## Overview

This service:
1. Polls database for new 1m candles
2. Every 5 candles (configurable), aggregates data and queries LLM
3. LLM analyzes price action, trends, and market-analyzer signals
4. Outputs prediction with direction, confidence, price targets, and reasoning
5. Saves analysis to database for accuracy tracking

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   TimescaleDB   │────▶│   LLM Analyst   │────▶│     Ollama      │
│  - candles_1m   │     │    (Python)     │     │  deepseek-r1    │
│  - market_*     │     │                 │◀────│                 │
│  - llm_analysis │◀────│                 │     └─────────────────┘
└─────────────────┘     └─────────────────┘
```

## Data Flow

1. **Input Data:**
   - 120 x 1H candles (~5 days of hourly data)
   - 20 x 15M candles (recent detail)
   - Latest market-analyzer output (signal, S/R, trends, SMC)
   - Last 15 signal changes (stability indicator)

2. **LLM Processing:**
   - Structured prompt with all market context
   - System prompt defining analyst role
   - DeepSeek-R1 reasoning for market analysis

3. **Output:**
   - Direction: BULLISH / BEARISH / NEUTRAL
   - Confidence: HIGH / MEDIUM / LOW
   - Price targets (1H, 4H)
   - Key levels (support, resistance)
   - Reasoning summary

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `DB_NAME` | btc_ml_production | Database name |
| `DB_USER` | mltrader | Database user |
| `DB_PASSWORD` | (required) | Database password |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Full Ollama API URL |
| `OLLAMA_MODEL` | deepseek-r1:32b | Model to use |
| `OLLAMA_TIMEOUT` | 300 | Request timeout (seconds) |
| `ANALYSIS_INTERVAL_CANDLES` | 5 | Run analysis every N 1m candles |
| `CANDLES_1H_LOOKBACK` | 120 | Number of 1H candles to include |
| `CANDLES_15M_LOOKBACK` | 20 | Number of 15M candles to include |
| `SIGNAL_HISTORY_COUNT` | 15 | Number of signal changes to include |
| `POLL_INTERVAL` | 10 | Seconds between candle checks |
| `HEALTH_PORT` | 8083 | Health check HTTP port |

## Running

### With Docker Compose

Add to your `docker-compose.yml`:

```yaml
llm-analyst:
  build:
    context: ./llm-analyst
    dockerfile: Dockerfile
  container_name: btc-ml-llm-analyst
  environment:
    - DB_HOST=timescaledb
    - DB_PORT=5432
    - DB_NAME=${DB_NAME}
    - DB_USER=${DB_USER}
    - DB_PASSWORD=${DB_PASSWORD}
    - OLLAMA_BASE_URL=http://host.docker.internal:11434  # Or your Ollama URL
    - OLLAMA_MODEL=deepseek-r1:32b
    - ANALYSIS_INTERVAL_CANDLES=5
  depends_on:
    timescaledb:
      condition: service_healthy
    market-analyzer:
      condition: service_started
  restart: unless-stopped
```

```bash
docker compose up -d llm-analyst
docker compose logs -f llm-analyst
```

### Local Development

```bash
cd llm-analyst
pip install -r requirements.txt
export DB_PASSWORD=your_password
export OLLAMA_BASE_URL=http://localhost:11434
python main.py
```

## Sample Output

```
================================================================================
🤖 LLM MARKET ANALYSIS - 2025-01-15 14:35:00 UTC
================================================================================
Model: deepseek-r1:32b | Response time: 45.3s | Tokens: 512

🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
🚀 PREDICTION: BULLISH
   Confidence: MEDIUM
🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢

📊 PRICE TARGETS
------------------------------------------------------------
  Current Price:  $87,500.00
  Expected (1H):  $88,200 (+0.80%)
  Expected (4H):  $89,500 (+2.29%)
  Invalidation:   $86,800

🎯 KEY LEVELS
------------------------------------------------------------
  Critical Support:    $86,100 (+1.60% from price)
  Critical Resistance: $89,000 (+1.71% from price)

💭 REASONING
------------------------------------------------------------
  Price is consolidating above the daily pivot with bullish 
  structure on lower timeframes. The recent CHoCH at $86,535 
  suggests smart money accumulation. Volume is elevated which 
  supports the move. However, 4H trend remains bearish so this 
  could be a relief rally within a larger downtrend.

================================================================================
```

## Database Tables

### llm_analysis
Stores every LLM analysis with predictions and accuracy tracking.

| Column | Type | Description |
|--------|------|-------------|
| analysis_time | TIMESTAMPTZ | When analysis was run |
| price | NUMERIC | Price at analysis time |
| prediction_direction | TEXT | BULLISH/BEARISH/NEUTRAL |
| prediction_confidence | TEXT | HIGH/MEDIUM/LOW |
| predicted_price_1h | NUMERIC | Expected price in 1 hour |
| predicted_price_4h | NUMERIC | Expected price in 4 hours |
| reasoning | TEXT | Summary of LLM reasoning |
| actual_price_1h | NUMERIC | Actual price 1h later (filled by job) |
| direction_correct_1h | BOOLEAN | Was direction prediction correct? |

### Accuracy Tracking

A scheduled job (`update_llm_accuracy`) runs hourly to:
1. Find predictions older than 1 hour
2. Look up actual prices at 1h and 4h marks
3. Calculate if direction prediction was correct
4. Update the record

Query accuracy stats:
```sql
SELECT * FROM v_llm_accuracy_stats;
```

## Prompt Engineering

The prompt is structured to give the LLM maximum context:

1. **Header** - Timestamp and current price
2. **1H Candles** - Last 30 shown in detail, with period stats
3. **15M Candles** - Recent price action detail
4. **Market Analyzer** - Signal, trends, S/R, SMC analysis
5. **Signal History** - Recent changes and stability metric
6. **Request** - Specific questions for the LLM to answer

The system prompt establishes the LLM as a senior analyst and instructs it to be direct and specific.

## Future Enhancements

- [ ] Telegram/Discord notifications on strong signals
- [ ] Confidence calibration based on historical accuracy
- [ ] Multiple model comparison (A/B testing)
- [ ] Web dashboard with prediction history
- [ ] Correlation with actual market movements
