/*!
Data WebSocket Service
Streams ML predictions from TimescaleDB via WebSocket

Security: HTTPS (self-signed) + API KEY authentication
Flow: ml-predictions (DB) → data-ws → WebSocket clients
*/

use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use deadpool_postgres::{Config, Pool, Runtime};
use futures_util::{SinkExt, StreamExt};
use rcgen::{CertificateParams, DistinguishedName, KeyPair};
use rustls::ServerConfig;
use rustls_pemfile::{certs, pkcs8_private_keys};
use serde::{Deserialize, Serialize};
use std::io::BufReader;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tokio_rustls::TlsAcceptor;
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

#[derive(Debug, Serialize)]
struct APIResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

#[tokio::main]
async fn main() {
    // Catch panics BEFORE logger init
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("PANIC: {:?}", panic_info);
    }));

    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("=============================================================================");
    info!("Data WebSocket Service - Starting");
    info!("=============================================================================");

    // Run
    if let Err(e) = run().await {
        error!("Fatal error: {:?}", e);
        std::process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    eprintln!("1. Starting run()");
    dotenvy::dotenv().ok();

    eprintln!("2. Reading env vars");
    let db_host = std::env::var("DB_HOST").unwrap_or_else(|_| "localhost".to_string());
    let db_port = std::env::var("DB_PORT").unwrap_or_else(|_| "5432".to_string());
    let db_name = std::env::var("DB_NAME").unwrap_or_else(|_| "btc_ml_production".to_string());
    let db_user = std::env::var("DB_USER").unwrap_or_else(|_| "mltrader".to_string());
    let db_password = std::env::var("DB_PASSWORD").unwrap_or_else(|_| "password".to_string());
    let api_key = std::env::var("API_KEY").unwrap_or_else(|_| "your_api_key_here".to_string());
    let bind_addr = std::env::var("DATA_WS_BIND_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8443".to_string())
        .parse::<SocketAddr>()?;

    // Database pool
    eprintln!("3. Creating DB pool");
    let mut cfg = Config::new();
    cfg.host = Some(db_host.clone());
    cfg.port = Some(db_port.parse()?);
    cfg.dbname = Some(db_name.clone());
    cfg.user = Some(db_user.clone());
    cfg.password = Some(db_password);

    let pool = cfg.create_pool(Some(Runtime::Tokio1), tokio_postgres::NoTls)?;

    info!("✓ Database pool created: {}:{}/{}", db_host, db_port, db_name);

    let state = AppState {
        pool: pool.clone(),
        api_key: api_key.clone(),
    };

    // Test DB connection
    eprintln!("4. Testing DB connection");
    match pool.get().await {
        Ok(_) => info!("✓ Database connection test successful"),
        Err(e) => {
            error!("✗ Database connection failed: {}", e);
            return Err(e.into());
        }
    }

    // Build app
    eprintln!("5. Building app");
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/api/v1/predictions/latest", get(latest_prediction_handler))
        .route("/api/v1/predictions/stream", get(websocket_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Generate self-signed cert
    eprintln!("6. Generating cert");
    info!("Generating self-signed TLS certificate...");
    let cert = generate_self_signed_cert()?;
    let cert_pem = cert.cert.pem();
    let key_pem = cert.key_pair.serialize_pem();
    eprintln!("7. Cert generated");

    // Create TLS config
    let certs = certs(&mut BufReader::new(cert_pem.as_bytes()))
        .collect::<Result<Vec<_>, _>>()?;
    
    let keys = pkcs8_private_keys(&mut BufReader::new(key_pem.as_bytes()))
        .collect::<Result<Vec<_>, _>>()?;

    let mut server_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, rustls::pki_types::PrivateKeyDer::Pkcs8(keys[0].clone()))?;

    server_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

    let tls_acceptor = TlsAcceptor::from(Arc::new(server_config));

    info!("✓ TLS configured (self-signed certificate)");
    info!("=============================================================================");
    info!("🚀 HTTPS server listening on: https://{}", bind_addr);
    info!("=============================================================================");

    // Generate cert files
    std::fs::write("/tmp/cert.pem", cert_pem)?;
    std::fs::write("/tmp/key.pem", key_pem)?;

    // Use axum-server with rustls
    let rustls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(
        "/tmp/cert.pem",
        "/tmp/key.pem"
    ).await?;

    axum_server::bind_rustls(bind_addr, rustls_config)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn auth_middleware<B>(
    State(state): State<AppState>,
    req: Request<B>,
    next: Next<B>,
) -> Response {
    // Health endpoint doesn't need auth
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    // Check Authorization header
    if let Some(auth_header) = req.headers().get("Authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            let token = auth_str.strip_prefix("Bearer ").unwrap_or(auth_str);
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
    ws.on_upgrade(|socket| websocket_stream(socket, state))
}

async fn websocket_stream(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    let mut tick = interval(Duration::from_secs(5));

    loop {
        tokio::select! {
            _ = tick.tick() => {
                match fetch_latest_prediction(&state.pool).await {
                    Ok(Some(pred)) => {
                        let json = serde_json::to_string(&pred).unwrap();
                        if sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Ok(None) => {
                        // No data yet
                    }
                    Err(e) => {
                        error!("Error fetching prediction: {}", e);
                    }
                }
            }
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => {
                        info!("WebSocket client disconnected");
                        break;
                    }
                    Some(Ok(Message::Ping(_))) => {
                        // Auto-handled by axum
                    }
                    _ => {}
                }
            }
        }
    }
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

fn generate_self_signed_cert() -> anyhow::Result<rcgen::CertifiedKey> {
    let mut params = CertificateParams::new(vec!["localhost".to_string()])?;
    params.distinguished_name = DistinguishedName::new();
    params.distinguished_name.push(
        rcgen::DnType::CommonName,
        "BTC ML Data WebSocket Service",
    );

    let key_pair = KeyPair::generate()?;
    let cert = params.self_signed(&key_pair)?;

    Ok(rcgen::CertifiedKey { cert, key_pair })
}