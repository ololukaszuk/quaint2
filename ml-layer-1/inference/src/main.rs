/*!
ML Inference Service
Runs ONNX inference and writes predictions to TimescaleDB

Flow: Candles (DB) → Inference → Predictions (DB) → data-ws streams
*/

use anyhow::Result;
use chrono::{DateTime, Utc};
use ndarray::Array1;
use ort::{Environment, ExecutionProvider, Session, SessionBuilder, Value};
use serde::{Deserialize, Serialize};
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
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("=" .repeat(80));
    info!("ML Inference Service - Starting");
    info!("=" .repeat(80));

    // Load config
    dotenvy::dotenv().ok();

    let db_host = std::env::var("DB_HOST").unwrap_or_else(|_| "localhost".to_string());
    let db_port = std::env::var("DB_PORT").unwrap_or_else(|_| "5432".to_string());
    let db_name = std::env::var("DB_NAME").unwrap_or_else(|_| "btc_ml_production".to_string());
    let db_user = std::env::var("DB_USER").unwrap_or_else(|_| "mltrader".to_string());
    let db_password = std::env::var("DB_PASSWORD").unwrap_or_else(|_| "password".to_string());
    let model_path = std::env::var("ML_MODEL_PATH")
        .unwrap_or_else(|_| "/app/models/lstm_btc_15min.onnx".to_string());
    let norm_path = std::env::var("ML_NORM_PATH")
        .unwrap_or_else(|_| "/app/models/normalization_params.json".to_string());

    // Connect to database
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

    // Load ONNX model
    info!("Loading ONNX model: {}", model_path);
    let environment = Arc::new(Environment::builder().with_name("ml-inference").build()?);

    let session = SessionBuilder::new(&environment)?
        .with_execution_providers([
            ExecutionProvider::TensorRT(Default::default()),
            ExecutionProvider::CUDA(Default::default()),
        ])?
        .with_model_from_file(&model_path)?;

    info!("✓ ONNX model loaded");

    // Load normalization params
    info!("Loading normalization params: {}", norm_path);
    let norm_json = std::fs::read_to_string(&norm_path)?;
    let norm_params: NormalizationParams = serde_json::from_str(&norm_json)?;
    info!("✓ Normalization params loaded");

    info!("=" .repeat(80));
    info!("Starting inference loop (checks for new data every 30s)");
    info!("=" .repeat(80));

    // Inference loop - check for new data frequently
    let mut tick = interval(Duration::from_secs(30));
    let mut last_processed_time: Option<DateTime<Utc>> = None;

    loop {
        tick.tick().await;

        // Check if there's new data
        match get_latest_candle_time(&client).await {
            Ok(latest_time) => {
                // Only run inference if there's new data
                if last_processed_time.map_or(true, |t| latest_time > t) {
                    match run_inference(&client, &session, &norm_params, latest_time).await {
                        Ok(latency) => {
                            info!("✓ Inference complete: {:.2}ms at {}", latency, latest_time);
                            last_processed_time = Some(latest_time);
                        }
                        Err(e) => {
                            error!("Inference failed: {}", e);
                        }
                    }
                } else {
                    // No new data yet
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
            "SELECT time FROM klines 
             WHERE symbol = 'BTCUSDT' AND interval = '1m'
             ORDER BY time DESC LIMIT 1",
            &[],
        )
        .await?;

    Ok(row.get(0))
}

async fn run_inference(
    client: &Arc<Client>,
    session: &Session,
    norm_params: &NormalizationParams,
    candle_time: DateTime<Utc>,
) -> Result<f64> {
    let start = Instant::now();

    // 1. Fetch last 60 candles
    let candles = fetch_candles(client).await?;

    if candles.len() < 60 {
        anyhow::bail!("Insufficient candles: {} < 60", candles.len());
    }

    let latest_60 = &candles[candles.len() - 60..];

    // 2. Compute features
    let features = compute_features(latest_60)?;

    // 3. Normalize
    let features_norm = normalize_features(&features, norm_params);

    // 4. Run ONNX inference (multi-horizon)
    let (pred_1min, pred_5min, pred_15min) = run_onnx_inference(session, &features_norm)?;

    // 5. Write to database
    let current_price = latest_60.last().unwrap().close;
    write_predictions(client, candle_time, current_price, pred_1min, pred_5min, pred_15min).await?;

    let latency = start.elapsed().as_secs_f64() * 1000.0;
    Ok(latency)
}

async fn fetch_candles(client: &Arc<Client>) -> Result<Vec<Candle>> {
    let rows = client
        .query(
            "SELECT time, close, high, low, volume 
             FROM klines 
             WHERE symbol = 'BTCUSDT' AND interval = '1m'
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
        });
    }

    candles.reverse(); // Oldest first
    Ok(candles)
}

fn compute_features(candles: &[Candle]) -> Result<Vec<f64>> {
    // Simplified: just use price, volume for now
    // In production, implement full 9 features matching training
    let mut features = Vec::new();

    let close = candles.last().unwrap().close;
    let volume = candles.last().unwrap().volume;

    // Price norm (simplified)
    let prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let mean = prices.iter().sum::<f64>() / prices.len() as f64;
    let std = (prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64)
        .sqrt();
    features.push((close - mean) / (std + 1e-8));

    // RSI (simplified - use last close)
    features.push(0.5);

    // MACD, Signal, Hist
    features.push(0.0);
    features.push(0.0);
    features.push(0.0);

    // BB position
    features.push(0.5);

    // Volume ratio
    features.push(1.0);

    // Returns
    features.push(0.0);

    // ATR
    features.push(1.0);

    Ok(features)
}

fn normalize_features(features: &[f64], params: &NormalizationParams) -> Vec<f64> {
    features
        .iter()
        .zip(&params.mean)
        .zip(&params.std)
        .map(|((f, m), s)| ((f - m) / s).clamp(-10.0, 10.0))
        .collect()
}

fn run_onnx_inference(session: &Session, features: &[f64]) -> Result<(f64, f64, f64)> {
    // Create input tensor: (1, 60, 9)
    let mut input_data = vec![0.0f32; 60 * 9];

    // Fill last timestep with features
    for (i, &f) in features.iter().enumerate() {
        input_data[59 * 9 + i] = f as f32;
    }

    let input_tensor = Array1::from_vec(input_data)
        .into_shape((1, 60, 9))?
        .into_dyn();

    let outputs = session.run(vec![Value::from_array(session.allocator(), &input_tensor)?])?;

    let output: Vec<f32> = outputs[0].try_extract()?.view().to_slice().unwrap().to_vec();

    // Base prediction
    let pred_base = output[0] as f64;
    
    // Multi-horizon: simple extrapolation for now
    // TODO: Train separate models for each horizon
    let pred_1min = pred_base;
    let pred_5min = pred_base * 1.002; // Slightly higher
    let pred_15min = pred_base * 1.005; // Even higher

    Ok((pred_1min, pred_5min, pred_15min))
}

async fn write_predictions(
    client: &Arc<Client>,
    candle_time: DateTime<Utc>,
    current_price: f64,
    pred_1min: f64,
    pred_5min: f64,
    pred_15min: f64,
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
                &rust_decimal::Decimal::from_f64_retain(pred_1min).unwrap(),
                &rust_decimal::Decimal::from_f64_retain(pred_5min).unwrap(),
                &rust_decimal::Decimal::from_f64_retain(pred_15min).unwrap(),
                &0.75f32, // confidence (TODO: calculate from model)
                &0.70f32,
                &0.65f32,
                &"v1.0",
            ],
        )
        .await?;

    Ok(())
}
