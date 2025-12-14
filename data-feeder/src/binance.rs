//! Binance WebSocket client for streaming cryptocurrency data.
//!
//! This module handles dual stream connections for:
//! - @kline_1m: 1-minute candlestick data
//! - @bookTicker: best bid/ask price updates
//!
//! Features:
//! - Auto-reconnection with exponential backoff
//! - Event alignment between streams
//! - Ring buffer during reconnection

use crate::errors::{DataFeederError, Result};
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

/// Maximum exponential backoff delay in seconds.
const MAX_BACKOFF_SECS: u64 = 8;

/// Size of the ring buffer for holding candles during reconnection.
const RING_BUFFER_SIZE: usize = 100;

/// Represents a complete 1-minute candle with all computed fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleData {
    /// Candle open time (timestamp)
    pub time: DateTime<Utc>,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Base asset volume
    pub volume: f64,
    /// Quote asset volume
    pub quote_asset_volume: f64,
    /// Taker buy base asset volume
    pub taker_buy_base_asset_volume: f64,
    /// Taker buy quote asset volume
    pub taker_buy_quote_asset_volume: f64,
    /// Number of trades
    pub number_of_trades: i64,
    /// Spread in basis points (computed from book ticker)
    pub spread_bps: Option<f64>,
    /// Best bid price at candle close
    pub best_bid: Option<f64>,
    /// Best ask price at candle close
    pub best_ask: Option<f64>,
}

/// Raw kline data from Binance WebSocket.
#[derive(Debug, Deserialize)]
struct BinanceKlineEvent {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time: u64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "k")]
    kline: BinanceKline,
}

#[derive(Debug, Deserialize)]
struct BinanceKline {
    #[serde(rename = "t")]
    open_time: i64,
    #[serde(rename = "T")]
    close_time: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "i")]
    interval: String,
    #[serde(rename = "o")]
    open: String,
    #[serde(rename = "c")]
    close: String,
    #[serde(rename = "h")]
    high: String,
    #[serde(rename = "l")]
    low: String,
    #[serde(rename = "v")]
    volume: String,
    #[serde(rename = "n")]
    number_of_trades: i64,
    #[serde(rename = "x")]
    is_closed: bool,
    #[serde(rename = "q")]
    quote_volume: String,
    #[serde(rename = "V")]
    taker_buy_base_volume: String,
    #[serde(rename = "Q")]
    taker_buy_quote_volume: String,
}

/// Raw book ticker data from Binance WebSocket.
#[derive(Debug, Deserialize)]
struct BinanceBookTicker {
    #[serde(rename = "e")]
    event_type: Option<String>,
    #[serde(rename = "u")]
    update_id: u64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "b")]
    best_bid: String,
    #[serde(rename = "B")]
    best_bid_qty: String,
    #[serde(rename = "a")]
    best_ask: String,
    #[serde(rename = "A")]
    best_ask_qty: String,
}

/// Combined stream message wrapper.
#[derive(Debug, Deserialize)]
struct CombinedStreamMessage {
    stream: String,
    data: serde_json::Value,
}

/// Current best bid/ask state.
#[derive(Debug, Default, Clone)]
struct BookTickerState {
    best_bid: f64,
    best_ask: f64,
    last_update: u64,
}

/// Binance WebSocket client for streaming price data.
pub struct BinanceWebSocketClient {
    /// WebSocket URL
    url: String,
    /// Channel sender for candle data
    candle_tx: mpsc::Sender<CandleData>,
    /// Current book ticker state (shared with reader task)
    book_ticker_state: Arc<RwLock<BookTickerState>>,
    /// Ring buffer for candles during reconnection
    ring_buffer: Arc<RwLock<VecDeque<CandleData>>>,
    /// Connection status
    is_connected: Arc<AtomicBool>,
    /// Total candles processed
    candles_processed: Arc<AtomicU64>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

impl BinanceWebSocketClient {
    /// Create a new WebSocket client.
    pub fn new(url: String, candle_tx: mpsc::Sender<CandleData>) -> Self {
        Self {
            url,
            candle_tx,
            book_ticker_state: Arc::new(RwLock::new(BookTickerState::default())),
            ring_buffer: Arc::new(RwLock::new(VecDeque::with_capacity(RING_BUFFER_SIZE))),
            is_connected: Arc::new(AtomicBool::new(false)),
            candles_processed: Arc::new(AtomicU64::new(0)),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if the client is currently connected.
    pub fn is_connected(&self) -> bool {
        self.is_connected.load(Ordering::SeqCst)
    }

    /// Get the total number of candles processed.
    pub fn candles_processed(&self) -> u64 {
        self.candles_processed.load(Ordering::SeqCst)
    }

    /// Signal the client to shutdown.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Run the WebSocket client with auto-reconnection.
    pub async fn run(&self) -> Result<()> {
        let mut backoff_secs = 1u64;

        loop {
            if self.shutdown.load(Ordering::SeqCst) {
                info!("WebSocket client shutting down");
                break;
            }

            match self.connect_and_stream().await {
                Ok(_) => {
                    // Clean exit (shouldn't happen normally)
                    info!("WebSocket stream ended cleanly");
                    backoff_secs = 1;
                }
                Err(e) => {
                    self.is_connected.store(false, Ordering::SeqCst);
                    error!("WebSocket error: {}", e);

                    if self.shutdown.load(Ordering::SeqCst) {
                        break;
                    }

                    // Exponential backoff
                    warn!(
                        "Reconnecting in {} seconds (backoff)",
                        backoff_secs
                    );
                    tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                    backoff_secs = (backoff_secs * 2).min(MAX_BACKOFF_SECS);
                }
            }
        }

        Ok(())
    }

    /// Connect to WebSocket and process messages.
    async fn connect_and_stream(&self) -> Result<()> {
        info!("Connecting to Binance WebSocket: {}", self.url);

        let (ws_stream, response) = connect_async(&self.url).await?;
        info!(
            "WebSocket connected, status: {}",
            response.status()
        );

        self.is_connected.store(true, Ordering::SeqCst);

        // Flush ring buffer after reconnection
        self.flush_ring_buffer().await;

        let (mut write, mut read) = ws_stream.split();

        // Ping/pong handler
        let shutdown_clone = self.shutdown.clone();
        let ping_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                if shutdown_clone.load(Ordering::SeqCst) {
                    break;
                }
                if write.send(Message::Ping(vec![])).await.is_err() {
                    break;
                }
            }
        });

        // Process incoming messages
        while let Some(msg) = read.next().await {
            if self.shutdown.load(Ordering::SeqCst) {
                break;
            }

            match msg {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.handle_message(&text).await {
                        warn!("Error handling message: {}", e);
                    }
                }
                Ok(Message::Ping(data)) => {
                    debug!("Received ping, sending pong");
                    // Pong is handled automatically by tungstenite
                    let _ = data;
                }
                Ok(Message::Pong(_)) => {
                    debug!("Received pong");
                }
                Ok(Message::Close(frame)) => {
                    info!("WebSocket closed: {:?}", frame);
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    error!("WebSocket read error: {}", e);
                    break;
                }
            }
        }

        ping_handle.abort();
        self.is_connected.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Handle an incoming WebSocket message.
    async fn handle_message(&self, text: &str) -> Result<()> {
        // Parse as combined stream message
        let msg: CombinedStreamMessage = serde_json::from_str(text)?;

        if msg.stream.ends_with("@kline_1m") {
            self.handle_kline(&msg.data).await?;
        } else if msg.stream.ends_with("@bookTicker") {
            self.handle_book_ticker(&msg.data)?;
        }

        Ok(())
    }

    /// Handle a kline (candlestick) message.
    async fn handle_kline(&self, data: &serde_json::Value) -> Result<()> {
        let event: BinanceKlineEvent = serde_json::from_value(data.clone())?;

        // Only process closed candles
        if !event.kline.is_closed {
            return Ok(());
        }

        let kline = &event.kline;

        // Parse numeric values
        let open: f64 = kline.open.parse().unwrap_or(0.0);
        let high: f64 = kline.high.parse().unwrap_or(0.0);
        let low: f64 = kline.low.parse().unwrap_or(0.0);
        let close: f64 = kline.close.parse().unwrap_or(0.0);
        let volume: f64 = kline.volume.parse().unwrap_or(0.0);
        let quote_volume: f64 = kline.quote_volume.parse().unwrap_or(0.0);
        let taker_buy_base: f64 = kline.taker_buy_base_volume.parse().unwrap_or(0.0);
        let taker_buy_quote: f64 = kline.taker_buy_quote_volume.parse().unwrap_or(0.0);

        // Get current book ticker state for spread calculation
        let book_state = self.book_ticker_state.read().clone();
        let (spread_bps, best_bid, best_ask) = if book_state.best_bid > 0.0 && book_state.best_ask > 0.0 {
            let mid = (book_state.best_bid + book_state.best_ask) / 2.0;
            let spread = ((book_state.best_ask - book_state.best_bid) / mid) * 10000.0;
            (Some(spread), Some(book_state.best_bid), Some(book_state.best_ask))
        } else {
            (None, None, None)
        };

        // Convert timestamp to DateTime
        let time = DateTime::from_timestamp_millis(kline.open_time)
            .unwrap_or_else(Utc::now);

        let candle = CandleData {
            time,
            open,
            high,
            low,
            close,
            volume,
            quote_asset_volume: quote_volume,
            taker_buy_base_asset_volume: taker_buy_base,
            taker_buy_quote_asset_volume: taker_buy_quote,
            number_of_trades: kline.number_of_trades,
            spread_bps,
            best_bid,
            best_ask,
        };

        // Send to channel or buffer
        if self.candle_tx.try_send(candle.clone()).is_err() {
            // Channel full, buffer locally
            let mut buffer = self.ring_buffer.write();
            if buffer.len() >= RING_BUFFER_SIZE {
                buffer.pop_front();
            }
            buffer.push_back(candle);
            warn!("Channel full, candle buffered locally");
        } else {
            self.candles_processed.fetch_add(1, Ordering::SeqCst);
            debug!(
                "Candle sent: time={}, close={}, volume={}",
                time, close, volume
            );
        }

        Ok(())
    }

    /// Handle a book ticker (best bid/ask) message.
    fn handle_book_ticker(&self, data: &serde_json::Value) -> Result<()> {
        let ticker: BinanceBookTicker = serde_json::from_value(data.clone())?;

        let best_bid: f64 = ticker.best_bid.parse().unwrap_or(0.0);
        let best_ask: f64 = ticker.best_ask.parse().unwrap_or(0.0);

        let mut state = self.book_ticker_state.write();
        state.best_bid = best_bid;
        state.best_ask = best_ask;
        state.last_update = ticker.update_id;

        Ok(())
    }

    /// Flush buffered candles after reconnection.
    async fn flush_ring_buffer(&self) {
        let candles: Vec<CandleData> = {
            let mut buffer = self.ring_buffer.write();
            buffer.drain(..).collect()
        };

        if candles.is_empty() {
            return;
        }

        info!("Flushing {} buffered candles after reconnection", candles.len());

        for candle in candles {
            if self.candle_tx.try_send(candle).is_ok() {
                self.candles_processed.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_data_serialization() {
        let candle = CandleData {
            time: Utc::now(),
            open: 100000.0,
            high: 100500.0,
            low: 99500.0,
            close: 100200.0,
            volume: 1000.0,
            quote_asset_volume: 100000000.0,
            taker_buy_base_asset_volume: 500.0,
            taker_buy_quote_asset_volume: 50000000.0,
            number_of_trades: 5000,
            spread_bps: Some(5.0),
            best_bid: Some(100199.0),
            best_ask: Some(100201.0),
        };

        let json = serde_json::to_string(&candle).unwrap();
        let parsed: CandleData = serde_json::from_str(&json).unwrap();

        assert_eq!(candle.close, parsed.close);
        assert_eq!(candle.volume, parsed.volume);
    }
}
