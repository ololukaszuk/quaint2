# LLM Market Analyst Service

AI-powered market commentary using local LLM models (DeepSeek/Ollama) with **full utilization of enhanced market-analyzer data**.

> **🆕 Enhanced Schema v2.0:** This service now fully utilizes the enhanced market_analysis schema, including signal factors, SMC data, pivot levels, and warnings.

## Features

- 🤖 **Local LLM inference** via Ollama (DeepSeek, Llama, Mistral, etc.)
- 📊 **Full market context** - Uses all enhanced schema fields (SMC, pivots, signal factors)
- 📈 **Price predictions** with confidence levels
- 🎯 **Key level identification** (support/resistance)
- 💾 **Enhanced database logging** - Stores market context with each prediction
- 📉 **Accuracy tracking** - Automatic evaluation after 1h/4h
- ⚡ **Configurable triggers** - Run every N candles

## Quick Start

### Prerequisites

1. **Ollama** running with a model:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull a model
   ollama pull deepseek-r1:8b
   # or
   ollama pull llama3:8b
   ```

2. **TimescaleDB** with market data and enhanced schema (migration 002)

3. **Python 3.10+** with dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=btc_ml_production
DB_USER=mltrader
DB_PASSWORD=your_password

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:8b

# Analysis settings
ANALYSIS_INTERVAL_CANDLES=5    # Run every 5 closed 1m candles
POLL_INTERVAL_SECONDS=30       # Check for new candles every 30s
CANDLES_1H_LOOKBACK=120        # 1H candles to analyze
CANDLES_15M_LOOKBACK=20        # 15M candles to analyze
SIGNAL_HISTORY_COUNT=15        # Recent signals to include
```

### Run with Docker

```bash
docker-compose up llm-analyst
```

### Run Standalone

```bash
cd llm-analyst
python main.py
```

## Enhanced Data Utilization

The service now fetches and uses ALL enhanced market_analysis fields:

### Signal Factors
```
Top Signal Factors:
  🟢 +30 | Bullish CHoCH - potential trend reversal up
  🔴 -30 | At strong resistance $87,753 (0.22% away)
  🔴 -25 | 4h trend: DOWNTREND (100% strength), EMA: BEARISH
  🟢 +25 | Above SMC equilibrium ($87,356)
  🟡  +0 | RSI neutral 42-58 range
```

### Pivot Levels (All Methods)
- Traditional pivots (R1-R3, S1-S3)
- Fibonacci pivots (R1-R3, S1-S3)
- Camarilla pivots (R3-R4, S3-S4)
- Confluence zones (where methods agree)

### Smart Money Concepts
- Order blocks (bullish/bearish)
- Fair Value Gaps (unfilled)
- Structure breaks (BOS, CHoCH)
- Liquidity pools (buy/sell side)
- SMC bias and price zone

### Momentum (All Timeframes)
- RSI for 5m, 15m, 1h, 4h, 1d
- Volume ratios
- Taker buy ratios

### Warnings
- Active risk alerts
- Proximity to key levels
- Divergence warnings

## Database Schema

### Tables

#### `llm_analysis` (Enhanced)

The LLM analysis table now stores full market context with each prediction:

```sql
-- Original fields
id, analysis_time, price, prediction_direction, prediction_confidence,
predicted_price_1h, predicted_price_4h, key_levels, reasoning,
full_response, model_name, response_time_seconds,
actual_price_1h, actual_price_4h, direction_correct_1h, direction_correct_4h

-- Enhanced fields (migration 002)
invalidation_level      -- Price level where prediction is invalid
critical_support        -- Key support from LLM
critical_resistance     -- Key resistance from LLM
market_context          -- JSONB: Full market-analyzer state
signal_factors_used     -- JSONB: Weighted factors shown to LLM
smc_bias_at_analysis    -- SMC bias at time of analysis
trends_at_analysis      -- JSONB: Multi-TF trends
warnings_at_analysis    -- JSONB: Active warnings
```

### Views

- `v_llm_predictions_enhanced` - Predictions with market context and accuracy
- `v_llm_accuracy_by_conditions` - Accuracy breakdown by market conditions
- `v_llm_market_agreement` - LLM vs market-analyzer agreement tracking

### Helper Functions

- `get_llm_accuracy_for_conditions(smc_bias, market_signal)` - Query accuracy by conditions
- `analyze_successful_predictions()` - Find what factors correlate with correct predictions

## Migrations

### Migration 001 (Original)
Creates basic `llm_analysis` table with:
- Prediction fields
- Accuracy tracking
- Hourly accuracy update job

### Migration 002 (Enhanced)
Adds:
- Market context fields (JSONB)
- Signal factors, trends, warnings storage
- Enhanced views for analysis
- Accuracy-by-conditions views

Apply migrations:
```bash
# Apply migration 002
docker cp migrations/002_llm_analyst_enhanced.sql btc-ml-timescaledb:/tmp/
docker exec -it btc-ml-timescaledb psql -U mltrader -d btc_ml_production \
  -f /tmp/002_llm_analyst_enhanced.sql
```

## Output Example

```
================================================================================
🤖 LLM MARKET ANALYSIS - 2025-01-15 14:30:00 UTC
================================================================================
Model: deepseek-r1:8b | Response time: 4.2s | Tokens: 387

📈 MARKET CONTEXT (from market-analyzer):
------------------------------------------------------------
  Signal: WEAK_BUY (LONG)
  Confidence: 44%
  SMC Bias: BULLISH
  Action: WAIT

  Top Signal Factors:
    🟢 +30 | Bullish CHoCH - potential trend reversal up
    🔴 -30 | At strong resistance $87,753
    🔴 -25 | 4h trend: DOWNTREND

  ⚠️ Active Warnings:
    • CLOSE TO STRONG RESISTANCE ($87,753)

🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
🚀 LLM PREDICTION: BULLISH
   Confidence: MEDIUM
🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢

📊 PRICE TARGETS
------------------------------------------------------------
  Current Price:  $87,500.00
  Expected (1H):  $87,800 (+0.34%)
  Expected (4H):  $88,200 (+0.80%)
  Invalidation:   $87,100

🎯 KEY LEVELS
------------------------------------------------------------
  Critical Support:    $87,100 (-0.46% from price)
  Critical Resistance: $87,800 (+0.34% from price)

💭 REASONING
------------------------------------------------------------
  The bullish CHoCH signal suggests a potential trend reversal. While
  we're near strong resistance at $87,753, the SMC bias is bullish
  and price is above equilibrium. Volume is low but taker buy ratio
  is favorable. Watch for a break above resistance for confirmation.
```

## API Access

LLM analysis is available via the data-api:

```bash
# Get recent LLM analyses with market context
curl -k -H "Authorization: Bearer YOUR_API_KEY" \
  "https://localhost:8443/api/v1/llm-analysis?limit=10"
```

Response includes all enhanced fields:
```json
{
  "id": 123,
  "analysis_time": "2025-01-15T14:30:00Z",
  "price": "87500.00",
  "prediction_direction": "BULLISH",
  "prediction_confidence": "MEDIUM",
  "predicted_price_1h": "87800.00",
  "predicted_price_4h": "88200.00",
  "invalidation_level": "87100.00",
  "critical_support": "87100.00",
  "critical_resistance": "87800.00",
  "market_context": {
    "signal_type": "WEAK_BUY",
    "signal_direction": "LONG",
    "signal_confidence": 44.5,
    "smc_bias": "BULLISH",
    "action_recommendation": "WAIT"
  },
  "signal_factors_used": [...],
  "smc_bias_at_analysis": "BULLISH",
  "trends_at_analysis": {...},
  "warnings_at_analysis": [...],
  "direction_correct_1h": true,
  "direction_correct_4h": null
}
```

## Accuracy Analysis

Query accuracy by market conditions:

```sql
-- Which conditions lead to best predictions?
SELECT * FROM v_llm_accuracy_by_conditions;

-- Accuracy when SMC is bullish
SELECT * FROM get_llm_accuracy_for_conditions('BULLISH', NULL);

-- Accuracy for specific signal types
SELECT * FROM get_llm_accuracy_for_conditions(NULL, 'STRONG_BUY');
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `deepseek-r1:8b` | Model to use |
| `ANALYSIS_INTERVAL_CANDLES` | `5` | Candles between analyses |
| `POLL_INTERVAL_SECONDS` | `30` | Check interval |
| `CANDLES_1H_LOOKBACK` | `120` | 1H candles to include |
| `CANDLES_15M_LOOKBACK` | `20` | 15M candles to include |
| `SIGNAL_HISTORY_COUNT` | `15` | Recent signals to include |

## Troubleshooting

### Ollama Connection Failed
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
systemctl restart ollama
```

### Missing Enhanced Columns
```bash
# Apply migration 002
docker exec -it btc-ml-timescaledb psql -U mltrader -d btc_ml_production \
  -f /tmp/002_llm_analyst_enhanced.sql
```

### Slow Responses
- Use a smaller model (e.g., `deepseek-r1:7b` instead of `14b`)
- Ensure GPU acceleration is enabled
- Reduce `CANDLES_1H_LOOKBACK` to include less history

## Future Enhancements

- [ ] Multi-model ensemble (run multiple models, combine predictions)
- [ ] Fine-tuning on historical predictions
- [ ] Real-time accuracy dashboard
- [ ] Slack/Discord notifications for predictions
- [ ] Backtesting framework for LLM predictions