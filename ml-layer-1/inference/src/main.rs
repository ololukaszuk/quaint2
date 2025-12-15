//! ML Inference Standalone Binary
//!
//! Can be run independently for testing or as a separate service.
//! In production, use the library functions from data-feeder.

use ml_layer_1_inference::{
    check_hot_reload, initialize, run_inference_pipeline, DatabaseConfig, InferenceConfig,
    ModelManager,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tokio_postgres::NoTls;
use tracing::{error, info, Level};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("Starting ML Inference Service");

    // Load database config
    let _ = dotenvy::dotenv();
    let db_config = DatabaseConfig::from_env();

    // Connect to database
    let (client, connection) = tokio_postgres::connect(&db_config.connection_string(), NoTls)
        .await
        .map_err(|e| format!("Database connection failed: {}", e))?;

    // Spawn connection handler
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            error!("Database connection error: {}", e);
        }
    });

    let client = Arc::new(client);
    info!("Connected to database");

    // Initialize inference
    let (config, model_manager) = initialize(Some(&client), None).await?;

    if !config.enabled {
        info!("ML inference is disabled. Set ML_ENABLED=true to enable.");
        info!("Exiting...");
        return Ok(());
    }

    info!(
        "ML inference initialized: mamba_weight={}",
        config.mamba_weight
    );

    // Main inference loop
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    let mut hot_reload_interval = tokio::time::interval(Duration::from_secs(300)); // 5 min

    info!("Starting inference loop (60s interval)");

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Run inference
                match run_inference_pipeline(client.clone(), &model_manager).await {
                    Ok(prediction) => {
                        info!(
                            "Prediction: {:?} (conf: {:?}, latency: {:.2}ms)",
                            prediction.predictions,
                            prediction.confidences,
                            prediction.latency_ms
                        );
                    }
                    Err(e) => {
                        error!("Inference error: {}", e);
                    }
                }
            }

            _ = hot_reload_interval.tick() => {
                // Check for model updates
                if let Err(e) = check_hot_reload(&client, &model_manager).await {
                    error!("Hot-reload check error: {}", e);
                }
            }

            _ = signal::ctrl_c() => {
                info!("Received shutdown signal");
                break;
            }
        }
    }

    info!("ML Inference Service stopped");
    Ok(())
}
