# Market Analyzer

Real-time multi-timeframe market context analyzer for BTCUSDT with trading signals.

## Features

### 🎯 Trading Signals
- **Multi-confluence scoring** - signals only when multiple factors align
- **BUY/SELL signals** with confidence percentage (0-100%)
- **Trade setups** - Entry, Stop Loss, Take Profit 1/2/3 levels
- **Risk/Reward calculation**
- **Detailed reasoning** - see exactly why signal was generated

### 📐 Pivot Points (5 Methods)
- **Traditional (Floor)** - classic pivot calculation
- **Fibonacci** - uses fib ratios (0.382, 0.618, 1.0)
- **Camarilla** - intraday focused, tighter levels
- **Woodie** - gives more weight to close price
- **DeMark** - condition-based calculation
- **Confluence zones** - where multiple methods agree

### 🏦 Smart Money Concepts (SMC)
- **Order Blocks** - institutional supply/demand zones
- **Fair Value Gaps (FVG)** - imbalance zones likely to be filled
- **Break of Structure (BOS)** - trend continuation signals
- **Change of Character (CHoCH)** - trend reversal signals
- **Liquidity Sweeps** - stop hunts before reversals
- **Premium/Discount Zones** - optimal entry zones relative to range

### 📊 Multi-Timeframe Analysis
- Builds 5m, 15m, 1h, 4h, 1d from 1m candles
- Trend direction and strength per timeframe
- EMA alignment (8/21/50)
- Higher Highs/Higher Lows pattern detection

### ⚡ Momentum Analysis
- RSI with oversold/overbought alerts
- Volume ratio vs 20-period average
- Taker buy ratio (buying pressure)
- Spread analysis

## Architecture

```
market-analyzer/
├── main.py           # Service orchestration, logging
├── analyzer.py       # Core analysis coordination
├── signals.py        # Multi-confluence signal generator
├── pivots.py         # 5 pivot point calculation methods
├── smart_money.py    # SMC: Order Blocks, FVG, BOS/CHoCH
├── indicators.py     # Technical indicators (EMA, RSI, ATR)
├── timeframes.py     # Aggregation from 1m to higher TFs
├── database.py       # PostgreSQL/TimescaleDB queries
├── models.py         # Data structures
├── config.py         # Configuration from environment
├── Dockerfile
├── requirements.txt
└── README.md
```

## Sample Output

```
================================================================================
📈 MARKET CONTEXT REPORT - 2024-01-15 14:32:00 UTC
================================================================================
💰 Current Price: $102,350.00

🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢
🚀 SIGNAL: BUY
   Direction: LONG 📈 | Confidence: 72%
🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢

📋 TRADE SETUP
----------------------------------------------------------------------
  📍 Entry:         $102,350.00
  🛑 Stop Loss:     $101,200.00 (-1.12%)
  🎯 Take Profit 1: $103,100.00 (+0.73%) - Scale out 33%
  🎯 Take Profit 2: $104,200.00 (+1.81%) - Scale out 33%
  🎯 Take Profit 3: $105,800.00 (+3.37%) - Final exit
  📊 Risk/Reward:   1.85R
  ❌ Invalidation:  Break below $101,200 support

🧠 SIGNAL REASONING (Top Factors)
----------------------------------------------------------------------
  🟢 [+ 25%] ████████████░░░░░░░░ 4h trend: UPTREND (80% strength)
  🟢 [+ 20%] ████████░░░░░░░░░░░░ HTF trend alignment: 2/3 bullish
  🟢 [+ 15%] ██████░░░░░░░░░░░░░░ Price in discount zone (below EQ)
  🟢 [+ 12%] █████░░░░░░░░░░░░░░░ In bullish order block
  🔴 [- 10%] ████░░░░░░░░░░░░░░░░ Near resistance $102,800

📐 PIVOT POINTS (Based on Daily)
----------------------------------------------------------------------
  Price is 🟢 ABOVE daily pivot ($101,500)

  Traditional:  R3 $105,200 | R2 $103,800 | R1 $102,600 | P $101,500
  Fibonacci:    R3 $104,800 | R2 $103,500 | R1 $102,400 | P $101,500

  🎯 Pivot Confluence Zones:
     🟢 Support:    $101,200 | Strength: 80% | Methods: Traditional, Fibonacci
     🔴 Resistance: $102,800 | Strength: 60% | Methods: Traditional, Camarilla

🏦 SMART MONEY CONCEPTS (SMC)
----------------------------------------------------------------------
  Structure Bias: 🟢 BULLISH
  Price Zone:     DISCOUNT ZONE 🟢 (Good for longs)
  Equilibrium:    $102,000

  📦 Active Order Blocks:
     🟢 Bullish OB: $101,800 - $102,100 | Strength: 75%

  📊 Unfilled Fair Value Gaps:
     🟢 Bullish FVG: $101,500 - $101,700 (price may retrace here)

  💧 Liquidity Pools:
     📈 Buy-side (above): $103,500, $104,200
     📉 Sell-side (below): $101,000, $100,500

📊 TREND ANALYSIS (Multi-Timeframe)
----------------------------------------------------------------------
   5m: 🟢 UPTREND      | Strength: 60% | EMA: BULLISH  | HH/HL ✓
  15m: 🟢 UPTREND      | Strength: 70% | EMA: BULLISH  | HH/HL ✓
   1h: 🟡 SIDEWAYS     | Strength: 30% | EMA: MIXED    |
   4h: 🟢 UPTREND      | Strength: 80% | EMA: BULLISH  | HH/HL ✓
   1d: 🔴 DOWNTREND    | Strength: 50% | EMA: BEARISH  |

⚠️  WARNINGS & RISK ALERTS
----------------------------------------------------------------------
  ⚠️ TREND CONFLICT: 15m=UPTREND, 1d=DOWNTREND - Be cautious!

================================================================================
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `DB_NAME` | btc_ml_production | Database name |
| `DB_USER` | mltrader | Database user |
| `DB_PASSWORD` | (required) | Database password |
| `POLL_INTERVAL` | 5 | Seconds between candle checks |
| `LOOKBACK_CANDLES` | 300000 | Candles per timeframe for analysis |
| `HEALTH_PORT` | 8082 | Health check HTTP port |

## Running

### Docker Compose (Recommended)

```bash
# Start the service
docker compose up -d market-analyzer

# Watch logs
docker compose logs -f market-analyzer
```

### Local Development

```bash
cd market-analyzer
pip install -r requirements.txt
export DB_PASSWORD=your_password
python main.py
```

## Signal Logic

Signals are generated based on multi-factor confluence:

| Factor | Weight | Description |
|--------|--------|-------------|
| HTF Trend (4h/1d) | 35% | Higher timeframe trend direction |
| Momentum (RSI/Volume) | 25% | RSI levels, volume confirmation |
| S/R Levels | 15% | Proximity to support/resistance |
| SMC Structure | 15% | Order blocks, FVG, BOS/CHoCH |
| Pivot Confluence | 10% | Multiple pivot methods agreeing |

### Signal Types

| Signal | Score Range | Description |
|--------|-------------|-------------|
| STRONG_BUY | > 0.6 | High confidence long setup |
| BUY | 0.35 - 0.6 | Moderate confidence long |
| WEAK_BUY | 0.15 - 0.35 | Low confidence long |
| NEUTRAL | -0.15 - 0.15 | No clear direction |
| WEAK_SELL | -0.35 - -0.15 | Low confidence short |
| SELL | -0.6 - -0.35 | Moderate confidence short |
| STRONG_SELL | < -0.6 | High confidence short setup |

## Trading Warnings

The analyzer generates warnings for risky conditions:

- **Near Support** - Warns against shorting within 0.5% of strong support
- **Near Resistance** - Warns against longing within 0.5% of strong resistance
- **RSI Extremes** - Alerts for oversold (<25) or overbought (>75)
- **Trend Conflicts** - Warns when timeframes disagree
- **Low Volume** - Warns when volume is below 50% of average
- **Premium/Discount Mismatch** - Warns when signal contradicts zone

## Smart Money Concepts Explained

### Order Blocks (OB)
Zones where institutional orders were placed. Bullish OB = last bearish candle before strong up move. Price often returns to these zones.

### Fair Value Gaps (FVG)
Imbalances in price where there's a gap between candles. These gaps tend to get "filled" as price retraces.

### Break of Structure (BOS)
When price breaks a swing high/low in the direction of the trend. Confirms trend continuation.

### Change of Character (CHoCH)
First break of structure against the trend. Signals potential reversal.

### Liquidity Pools
Equal highs = buy-side liquidity (stops above). Equal lows = sell-side liquidity (stops below). Smart money hunts these before reversing.

## Database Schema

### Tables

#### `market_analysis`
Stores every analysis snapshot with complete market context.

**Key fields:**
- `signal_type`, `signal_direction`, `signal_confidence` - Trading signal
- `signal_factors` (JSONB) - Top 10 weighted factors with bar chart data
- `trends` (JSONB) - Multi-timeframe trend data with EMA and structure
- `pivot_*` columns - All pivot levels (Traditional, Fibonacci, Camarilla)
- `pivot_confluence_zones` (JSONB) - Where multiple methods agree
- `smc_*` columns - Complete SMC data (order blocks, FVGs, breaks, liquidity)
- `support_levels`, `resistance_levels` (JSONB) - All S/R levels with metadata
- `momentum` (JSONB) - RSI, volume, taker buy for all timeframes
- `structure_pattern`, `structure_last_high/low` - Market structure
- `warnings` (JSONB) - Risk alerts and warnings
- `action_recommendation` - WAIT, LONG, SHORT

**Retention:** 3 months

#### `market_signals`
Stores only signal CHANGES for alerting and backtesting.

**Key fields:**
- Signal info and trade setup
- `key_reasons` (JSONB) - Top reasons with weights
- `previous_signal_type`, `previous_direction` - What changed

**Retention:** 6 months

### Schema Version History

**v1.0** (Original)
- Basic signal and trend data
- Single support/resistance levels
- Limited momentum (1h only)

**v2.0** (Enhanced - Current)
- Complete log data capture
- All pivot methods and confluence
- Complete SMC analysis
- All S/R levels with metadata
- Momentum for all timeframes
- Market structure and warnings
- Weighted signal factors

### Migration

To upgrade from v1.0 to v2.0 schema:

```bash
# Backup first!
docker exec btc-ml-timescaledb pg_dump -U mltrader -d btc_ml_production > backup.sql

# Apply migration
docker cp migrations/migrate_to_enhanced_schema.sql btc-ml-timescaledb:/tmp/
docker exec -it btc-ml-timescaledb psql -U mltrader -d btc_ml_production -f /tmp/migrate_to_enhanced_schema.sql

# Update code and restart
docker-compose restart market-analyzer
```

See `migrations/MIGRATION_GUIDE.md` for detailed instructions.

### Data Access

All analysis data is available via the `data-api` service:

```bash
# Get latest analysis with complete data
curl -k -H "Authorization: Bearer YOUR_API_KEY" \
  "https://localhost:8443/api/v1/market-analysis?limit=1"

# Get recent signal changes
curl -k -H "Authorization: Bearer YOUR_API_KEY" \
  "https://localhost:8443/api/v1/market-signals?limit=10"
```

The API returns exactly the same data shown in logs - all pivot levels, SMC data, weighted factors, warnings, etc.

## Future Enhancements

- [ ] Telegram/Discord notifications
- [x] Store signals in database for backtesting (✅ Implemented)
- [ ] Web dashboard
- [ ] Pattern recognition (double top, H&S)
- [ ] Order flow analysis (CVD, delta)
- [ ] Correlation with other assets
- [ ] News sentiment integration
