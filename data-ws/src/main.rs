/*!
Data WebSocket Service
Streams ML predictions from TimescaleDB via WebSocket

Security: HTTPS (self-signed) + API KEY authentication
Flow: ml-predictions (DB) → data-ws → WebSocket clients
*/

use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade},
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use axum_server::tls_rustls::RustlsConfig;
use deadpool_postgres::{Config, Pool, Runtime};
use futures_util::{SinkExt, StreamExt};
use rcgen::{Certificate, CertificateParams, DistinguishedName};
use rustls::ServerConfig;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Clone)]
struct AppState {
    pool: Pool,
    api_key: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Prediction {
    time: String,
    current_price: f64,
    predicted_1min: Option<f64>,
    predicted_5min: Option<f64>,
    predicted_15min: Option<f64>,
    confidence_1min: Option<f32>,
    confidence_5min: Option<f32>,
    confidence_15min: Option<f32>,
    model_version: Option<String>,
}

#[derive(Serialize)]
struct APIResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("=" .repeat(80));
    info!("Data WebSocket Service - Starting");
    info!("=" .repeat(80));

    // Load config
    dotenvy::dotenv().ok();

    let bind_addr = std::env::var("DATA_WS_BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8443".to_string());
    let api_key = std::env::var("API_KEY")
        .unwrap_or_else(|_| "your_secure_api_key_here".to_string());

    // Database pool
    let mut cfg = Config::new();
    cfg.host = Some(std::env::var("DB_HOST").unwrap_or_else(|_| "localhost".to_string()));
    cfg.port = Some(
        std::env::var("DB_PORT")
            .unwrap_or_else(|_| "5432".to_string())
            .parse()
            .unwrap_or(5432),
    );
    cfg.dbname = Some(std::env::var("DB_NAME").unwrap_or_else(|_| "btc_ml_production".to_string()));
    cfg.user = Some(std::env::var("DB_USER").unwrap_or_else(|_| "mltrader".to_string()));
    cfg.password = Some(std::env::var("DB_PASSWORD").unwrap_or_else(|_| "password".to_string()));

    let pool = cfg.create_pool(Some(Runtime::Tokio1), tokio_postgres::NoTls)?;

    info!("✓ Database pool created");
    info!("✓ API key authentication enabled");

    // Generate self-signed cert
    info!("Generating self-signed TLS certificate...");
    let cert = generate_self_signed_cert()?;
    let tls_config = create_tls_config(&cert)?;

    info!("✓ Self-signed certificate generated");

    // App state
    let state = AppState { pool, api_key };

    // Build router
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/api/v1/predictions/latest", get(latest_prediction_handler))
        .route("/api/v1/predictions/stream", get(websocket_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            api_key_middleware,
        ))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr: SocketAddr = bind_addr.parse()?;
    info!("Starting HTTPS server on {}", addr);
    info!("Endpoints:");
    info!("  GET  /health                       - Health check");
    info!("  GET  /api/v1/predictions/latest    - Latest prediction");
    info!("  WS   /api/v1/predictions/stream    - WebSocket stream");
    info!("Authentication: Bearer <api-key>");
    info!("=" .repeat(80));

    axum_server::bind_rustls(addr, tls_config)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

// API Key middleware
async fn api_key_middleware<B>(
    State(state): State<AppState>,
    req: Request<B>,
    next: Next<B>,
) -> Response {
    // Skip auth for health endpoint
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok());

    if let Some(auth) = auth_header {
        if let Some(token) = auth.strip_prefix("Bearer ") {
            if token == state.api_key {
                return next.run(req).await;
            }
        }
    }

    warn!("Unauthorized request: invalid or missing API key");
    (
        StatusCode::UNAUTHORIZED,
        Json(APIResponse::<()> {
            success: false,
            data: None,
            error: Some("Unauthorized: invalid API key".to_string()),
        }),
    )
        .into_response()
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "data-ws",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

async fn latest_prediction_handler(
    State(state): State<AppState>,
) -> Result<Json<APIResponse<Prediction>>, StatusCode> {
    let client = state
        .pool
        .get()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let row = client
        .query_opt(
            "SELECT time, current_price, predicted_1min, predicted_5min, 
                    predicted_15min, confidence_1min, 
                    confidence_5min, confidence_15min,
                    model_version
             FROM ml_predictions
             ORDER BY time DESC
             LIMIT 1",
            &[],
        )
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    match row {
        Some(row) => {
            let pred = Prediction {
                time: row.get::<_, chrono::DateTime<chrono::Utc>>(0).to_rfc3339(),
                current_price: row
                    .get::<_, rust_decimal::Decimal>(1)
                    .to_string()
                    .parse()
                    .unwrap(),
                predicted_1min: row
                    .get::<_, Option<rust_decimal::Decimal>>(2)
                    .map(|d| d.to_string().parse().unwrap()),
                predicted_5min: row
                    .get::<_, Option<rust_decimal::Decimal>>(3)
                    .map(|d| d.to_string().parse().unwrap()),
                predicted_15min: row
                    .get::<_, Option<rust_decimal::Decimal>>(4)
                    .map(|d| d.to_string().parse().unwrap()),
                confidence_1min: row.get(5),
                confidence_5min: row.get(6),
                confidence_15min: row.get(7),
                model_version: row.get(8),
            };

            Ok(Json(APIResponse {
                success: true,
                data: Some(pred),
                error: None,
            }))
        }
        None => Ok(Json(APIResponse {
            success: false,
            data: None,
            error: Some("No predictions available".to_string()),
        })),
    }
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| websocket_connection(socket, state))
}

async fn websocket_connection(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    info!("WebSocket client connected");

    // Spawn send task
    let send_state = state.clone();
    let send_task = tokio::spawn(async move {
        let mut tick = interval(Duration::from_secs(5));

        loop {
            tick.tick().await;

            // Fetch latest prediction
            match fetch_latest_prediction(&send_state.pool).await {
                Ok(Some(pred)) => {
                    let json = serde_json::to_string(&pred).unwrap();
                    if sender.send(Message::Text(json)).await.is_err() {
                        break;
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    error!("Error fetching prediction: {}", e);
                }
            }
        }
    });

    // Receive task
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Close(_) = msg {
                break;
            }
        }
    });

    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }

    info!("WebSocket client disconnected");
}

async fn fetch_latest_prediction(pool: &Pool) -> anyhow::Result<Option<Prediction>> {
    let client = pool.get().await?;

    let row = client
        .query_opt(
            "SELECT time, current_price, predicted_1min, predicted_5min, 
                    predicted_15min, confidence_1min, 
                    confidence_5min, confidence_15min,
                    model_version
             FROM ml_predictions
             ORDER BY time DESC
             LIMIT 1",
            &[],
        )
        .await?;

    Ok(row.map(|row| Prediction {
        time: row.get::<_, chrono::DateTime<chrono::Utc>>(0).to_rfc3339(),
        current_price: row
            .get::<_, rust_decimal::Decimal>(1)
            .to_string()
            .parse()
            .unwrap(),
        predicted_1min: row
            .get::<_, Option<rust_decimal::Decimal>>(2)
            .map(|d| d.to_string().parse().unwrap()),
        predicted_5min: row
            .get::<_, Option<rust_decimal::Decimal>>(3)
            .map(|d| d.to_string().parse().unwrap()),
        predicted_15min: row
            .get::<_, Option<rust_decimal::Decimal>>(4)
            .map(|d| d.to_string().parse().unwrap()),
        confidence_1min: row.get(5),
        confidence_5min: row.get(6),
        confidence_15min: row.get(7),
        model_version: row.get(8),
    }))
}

fn generate_self_signed_cert() -> anyhow::Result<Certificate> {
    let mut params = CertificateParams::new(vec!["localhost".to_string()]);
    params.distinguished_name = DistinguishedName::new();
    params.distinguished_name.push(
        rcgen::DnType::CommonName,
        "BTC ML Data WebSocket Service",
    );

    Ok(Certificate::from_params(params)?)
}

fn create_tls_config(cert: &Certificate) -> anyhow::Result<RustlsConfig> {
    let cert_pem = cert.serialize_pem()?;
    let key_pem = cert.serialize_private_key_pem();

    Ok(RustlsConfig::from_pem(
        cert_pem.as_bytes().to_vec(),
        key_pem.as_bytes().to_vec(),
    )?)
}
