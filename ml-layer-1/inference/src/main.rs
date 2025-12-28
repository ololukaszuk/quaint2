/*!
ML Inference Service
Runs ONNX inference with proper 9 features matching train.py
*/

use anyhow::Result;
use chrono::{DateTime, Utc};
use ndarray::Array1;
use ort::{Environment, ExecutionProvider, Session, SessionBuilder, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;
use tokio_postgres::{Client, NoTls};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Serialize, Deserialize)]
struct NormalizationParams {
    mean: Vec<f64>,
    std: Vec<f64>,
}

#[derive(Debug)]
struct Candle {
    time: DateTime<Utc>,
    close: f64,
    high: f64,
    low: f64,
    volume: f64,
    taker_buy_volume: f64,  // NEW
    num_trades: f64,        // NEW
    spread_bps: f64,        // NEW
}

struct ModelSession {
    session: Session,
    horizon: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("=" .repeat(80));
    info!("ML Inference Service - Starting");
    info!("=" .repeat(80));

    dotenvy::dotenv().ok();

    let db_host = std::env::var("DB_HOST").unwrap_or_else(|_| "localhost".to_string());
    let db_port = std::env::var("DB_PORT").unwrap_or_else(|_| "5432".to_string());
    let db_name = std::env::var("DB_NAME").unwrap_or_else(|_| "btc_ml_production".to_string());
    let db_user = std::env::var("DB_USER").unwrap_or_else(|_| "mltrader".to_string());
    let db_password = std::env::var("DB_PASSWORD").unwrap_or_else(|_| "password".to_string());
    let models_dir = std::env::var("ML_MODELS_DIR").unwrap_or_else(|_| "/app/models".to_string());
    let norm_path = format!("{}/normalization_params.json", models_dir);

    let conn_str = format!(
        "host={} port={} dbname={} user={} password={}",
        db_host, db_port, db_name, db_user, db_password
    );

    info!("Connecting to database: {}:{}/{}", db_host, db_port, db_name);
    let (client, connection) = tokio_postgres::connect(&conn_str, NoTls).await?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            error!("Database connection error: {}", e);
        }
    });

    let client = Arc::new(client);
    info!("✓ Database connected");

    info!("Loading ONNX models...");
    let environment = Arc::new(Environment::builder().with_name("ml-inference").build()?);

    let mut models = HashMap::new();
    for horizon in [1, 5, 15] {
        let model_path = format!("{}/lstm_btc_{}min.onnx", models_dir, horizon);
        
        if Path::new(&model_path).exists() {
            let session = SessionBuilder::new(&environment)?
                .with_execution_providers([
                    ExecutionProvider::TensorRT(Default::default()),
                    ExecutionProvider::CUDA(Default::default()),
                ])?
                .with_model_from_file(&model_path)?;
            
            models.insert(horizon, ModelSession { session, horizon });
            info!("✓ Loaded {}-min model", horizon);
        } else {
            warn!("Model not found: {}", model_path);
        }
    }

    if models.is_empty() {
        anyhow::bail!("No models found! Train models first.");
    }

    info!("Loading normalization params: {}", norm_path);
    let norm_json = std::fs::read_to_string(&norm_path)?;
    let norm_params: NormalizationParams = serde_json::from_str(&norm_json)?;
    info!("✓ Normalization params loaded");

    info!("=" .repeat(80));
    info!("Starting inference loop (checks for new data every 30s)");
    info!("=" .repeat(80));

    let mut tick = interval(Duration::from_secs(30));
    let mut last_processed_time: Option<DateTime<Utc>> = None;

    loop {
        tick.tick().await;

        match get_latest_candle_time(&client).await {
            Ok(latest_time) => {
                if last_processed_time.map_or(true, |t| latest_time > t) {
                    match run_inference(&client, &models, &norm_params, latest_time).await {
                        Ok(latency) => {
                            info!("✓ Inference complete: {:.2}ms at {}", latency, latest_time);
                            last_processed_time = Some(latest_time);
                        }
                        Err(e) => {
                            error!("Inference failed: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to check latest candle: {}", e);
            }
        }
    }
}

async fn get_latest_candle_time(client: &Arc<Client>) -> Result<DateTime<Utc>> {
    let row = client
        .query_one(
            "SELECT time FROM candles_1m 
             ORDER BY time DESC LIMIT 1",
            &[],
        )
        .await?;

    Ok(row.get(0))
}

async fn run_inference(
    client: &Arc<Client>,
    models: &HashMap<usize, ModelSession>,
    norm_params: &NormalizationParams,
    candle_time: DateTime<Utc>,
) -> Result<f64> {
    let start = Instant::now();

    let candles = fetch_candles(client).await?;

    if candles.len() < 60 {
        anyhow::bail!("Insufficient candles: {} < 60", candles.len());
    }

    let latest_60 = &candles[candles.len() - 60..];

    let features = compute_features(latest_60)?;

    let features_norm = normalize_features(&features, norm_params);

    let mut predictions = HashMap::new();
    
    for (horizon, model_session) in models {
        let pred = run_onnx_inference(&model_session.session, &features_norm)?;
        predictions.insert(*horizon, pred);
    }

    let current_price = latest_60.last().unwrap().close;
    write_predictions(
        client,
        candle_time,
        current_price,
        predictions.get(&1).copied(),
        predictions.get(&5).copied(),
        predictions.get(&15).copied(),
    )
    .await?;

    let latency = start.elapsed().as_secs_f64() * 1000.0;
    Ok(latency)
}

async fn fetch_candles(client: &Arc<Client>) -> Result<Vec<Candle>> {
    let rows = client
        .query(
            "SELECT time, close, high, low, volume, 
                    taker_buy_base_asset_volume, 
                    number_of_trades,
                    COALESCE(spread_bps, 0.0) as spread_bps
             FROM candles_1m 
             ORDER BY time DESC 
             LIMIT 100",
            &[],
        )
        .await?;

    let mut candles = Vec::new();
    for row in rows {
        candles.push(Candle {
            time: row.get(0),
            close: row.get::<_, rust_decimal::Decimal>(1).to_string().parse()?,
            high: row.get::<_, rust_decimal::Decimal>(2).to_string().parse()?,
            low: row.get::<_, rust_decimal::Decimal>(3).to_string().parse()?,
            volume: row.get::<_, rust_decimal::Decimal>(4).to_string().parse()?,
            taker_buy_volume: row.get::<_, rust_decimal::Decimal>(5).to_string().parse()?,
            num_trades: row.get::<_, i64>(6) as f64,
            spread_bps: row.get::<_, rust_decimal::Decimal>(7).to_string().parse()?,
        });
    }

    candles.reverse();
    Ok(candles)
}

fn compute_features(candles: &[Candle]) -> Result<Vec<f64>> {
    let n = candles.len();
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let taker_buys: Vec<f64> = candles.iter().map(|c| c.taker_buy_volume).collect();
    let num_trades: Vec<f64> = candles.iter().map(|c| c.num_trades).collect();
    let spreads: Vec<f64> = candles.iter().map(|c| c.spread_bps).collect();

    let last_idx = n - 1;

    // Original 9 features (same as before)
    let sma60 = calculate_sma(&closes, 60);
    let std60 = calculate_std(&closes, 60);
    let price_norm = (closes[last_idx] - sma60[last_idx]) / (std60 + 1e-8);

    let rsi = calculate_rsi(&closes, 14);
    let rsi_val = rsi[last_idx];

    let ema12 = calculate_ema(&closes, 12);
    let ema26 = calculate_ema(&closes, 26);
    let macd: Vec<f64> = ema12.iter().zip(&ema26).map(|(a, b)| a - b).collect();
    let signal = calculate_ema(&macd, 9);
    let histogram: Vec<f64> = macd.iter().zip(&signal).map(|(m, s)| m - s).collect();
    
    let macd_val = macd[last_idx];
    let signal_val = signal[last_idx];
    let hist_val = histogram[last_idx];

    let sma20 = calculate_sma(&closes, 20);
    let std20 = calculate_std(&closes, 20);
    let upper = sma20[last_idx] + 2.0 * std20;
    let lower = sma20[last_idx] - 2.0 * std20;
    let bb_position = ((closes[last_idx] - lower) / (upper - lower + 1e-8)).clamp(0.0, 1.0);

    let sma_vol = calculate_sma(&volumes, 20);
    let vol_ratio = volumes[last_idx] / (sma_vol[last_idx] + 1e-8);

    let returns = if last_idx > 0 {
        (closes[last_idx] - closes[last_idx - 1]) / (closes[last_idx - 1] + 1e-8)
    } else {
        0.0
    };

    let atr = calculate_atr(&highs, &lows, &closes, 14);
    let atr_val = atr[last_idx];

    // NEW FEATURES (10-13)
    let taker_buy_ratio = taker_buys[last_idx] / (volumes[last_idx] + 1e-8);
    
    let spread_val = spreads[last_idx];
    
    let vol_momentum = if last_idx > 0 {
        (volumes[last_idx] - volumes[last_idx - 1]) / (volumes[last_idx - 1] + 1e-8)
    } else {
        0.0
    };
    
    let trades_per_volume = num_trades[last_idx] / (volumes[last_idx] + 1e-8);

    Ok(vec![
        price_norm,
        rsi_val,
        macd_val,
        signal_val,
        hist_val,
        bb_position,
        vol_ratio,
        returns,
        atr_val,
        taker_buy_ratio,  // 10
        spread_val,       // 11
        vol_momentum,     // 12
        trades_per_volume, // 13
    ])
}

fn calculate_sma(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut sma = vec![0.0; n];
    
    for i in 0..n {
        if i < period - 1 {
            sma[i] = prices[..=i].iter().sum::<f64>() / (i + 1) as f64;
        } else {
            sma[i] = prices[i - period + 1..=i].iter().sum::<f64>() / period as f64;
        }
    }
    
    sma
}

fn calculate_std(prices: &[f64], period: usize) -> f64 {
    let n = prices.len();
    if n < period {
        return 0.0;
    }
    
    let window = &prices[n - period..];
    let mean = window.iter().sum::<f64>() / period as f64;
    let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
    variance.sqrt()
}

fn calculate_ema(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut ema = vec![0.0; n];
    ema[0] = prices[0];
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    
    for i in 1..n {
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }
    
    ema
}

fn calculate_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut rsi = vec![0.5; n];
    
    if n < period + 1 {
        return rsi;
    }
    
    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];
    
    for i in 1..n {
        let delta = prices[i] - prices[i - 1];
        if delta > 0.0 {
            gains[i] = delta;
        } else {
            losses[i] = -delta;
        }
    }
    
    let mut avg_gain = gains[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_loss = losses[1..=period].iter().sum::<f64>() / period as f64;
    
    for i in period..n {
        avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
        
        let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 0.0 };
        rsi[i] = 1.0 - (1.0 / (1.0 + rs));
    }
    
    rsi
}

fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    let mut tr = vec![0.0; n];
    
    tr[0] = highs[0] - lows[0];
    
    for i in 1..n {
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
    }
    
    calculate_ema(&tr, period)
}

fn normalize_features(features: &[f64], params: &NormalizationParams) -> Vec<f64> {
    features
        .iter()
        .zip(&params.mean)
        .zip(&params.std)
        .map(|((f, m), s)| ((f - m) / s).clamp(-10.0, 10.0))
        .collect()
}

fn run_onnx_inference(session: &Session, features: &[f64]) -> Result<f64> {
    let mut input_data = vec![0.0f32; 60 * 13];  // 9 → 13
    
    for (i, &f) in features.iter().enumerate() {
        input_data[59 * 13 + i] = f as f32;
    }
    
    let input_tensor = Array1::from_vec(input_data)
        .into_shape((1, 60, 13))?  // 9 → 13
        .into_dyn();
    
    let outputs = session.run(vec![Value::from_array(session.allocator(), &input_tensor)?])?;
    
    let output: Vec<f32> = outputs[0].try_extract()?.view().to_slice().unwrap().to_vec();
    
    Ok(output[0] as f64)
}

async fn write_predictions(
    client: &Arc<Client>,
    candle_time: DateTime<Utc>,
    current_price: f64,
    pred_1min: Option<f64>,
    pred_5min: Option<f64>,
    pred_15min: Option<f64>,
) -> Result<()> {
    client
        .execute(
            "INSERT INTO ml_predictions 
             (time, current_price, predicted_1min, predicted_5min, predicted_15min,
              confidence_1min, confidence_5min, confidence_15min,
              model_version)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            &[
                &candle_time,
                &rust_decimal::Decimal::from_f64_retain(current_price).unwrap(),
                &pred_1min.map(|p| rust_decimal::Decimal::from_f64_retain(p).unwrap()),
                &pred_5min.map(|p| rust_decimal::Decimal::from_f64_retain(p).unwrap()),
                &pred_15min.map(|p| rust_decimal::Decimal::from_f64_retain(p).unwrap()),
                &pred_1min.map(|_| 0.75f32),
                &pred_5min.map(|_| 0.70f32),
                &pred_15min.map(|_| 0.65f32),
                &"multi-horizon-v1",
            ],
        )
        .await?;

    Ok(())
}