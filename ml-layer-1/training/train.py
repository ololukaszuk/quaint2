"""
ML Layer 1 - Training Pipeline
Exports 3 separate ONNX models (1min, 5min, 15min)
Uses LOG RETURNS for better training stability
"""

import os
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import json
from pathlib import Path

import warnings
import logging

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import tf2onnx

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# Enable mixed precision for RTX 5090
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
tf.config.experimental.enable_tensor_float_32_execution(True)

class FeatureComputer:
    """Compute 9 technical features matching Rust implementation"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 0.5
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            upval = delta if delta > 0 else 0
            downval = -delta if delta < 0 else 0
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 1.0 - (1.0 / (1 + rs)) if (1 + rs) != 0 else 0.5
        
        return rsi
    
    @staticmethod
    def calculate_ema(prices, period):
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2.0 / (period + 1)
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    @staticmethod
    def calculate_sma(prices, period):
        sma = np.zeros_like(prices)
        for i in range(len(prices)):
            if i < period - 1:
                sma[i] = np.mean(prices[:i+1])
            else:
                sma[i] = np.mean(prices[i-period+1:i+1])
        return sma
    
    @staticmethod
    def compute_features(closes, highs, lows, volumes, taker_buy_volumes, num_trades, spreads):
        """13 features including orderflow"""
        n = len(closes)
        features = np.zeros((n, 13))  # 9 → 13
        
        # Original 9 features (unchanged)
        # 1. price_norm
        sma60 = FeatureComputer.calculate_sma(closes, 60)
        std60 = np.array([np.std(closes[:i+1] if i < 59 else closes[i-59:i+1]) if i > 0 else 1.0 for i in range(n)])
        features[:, 0] = (closes - sma60) / (std60 + 1e-8)
        
        # 2. RSI
        features[:, 1] = FeatureComputer.calculate_rsi(closes, 14)
        
        # 3-5. MACD
        ema12 = FeatureComputer.calculate_ema(closes, 12)
        ema26 = FeatureComputer.calculate_ema(closes, 26)
        macd = ema12 - ema26
        signal = FeatureComputer.calculate_ema(macd, 9)
        features[:, 2] = macd
        features[:, 3] = signal
        features[:, 4] = macd - signal
        
        # 6. BB position
        sma20 = FeatureComputer.calculate_sma(closes, 20)
        std20 = np.array([np.std(closes[:i+1] if i < 19 else closes[i-19:i+1]) if i > 0 else 1.0 for i in range(n)])
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        bb_position = np.clip((closes - lower) / (upper - lower + 1e-8), 0, 1)
        features[:, 5] = bb_position
        
        # 7. Volume ratio
        sma_vol = FeatureComputer.calculate_sma(volumes, 20)
        features[:, 6] = volumes / (sma_vol + 1e-8)
        
        # 8. Returns
        returns = np.zeros(n)
        returns[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-8)
        features[:, 7] = returns
        
        # 9. ATR
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        features[:, 8] = FeatureComputer.calculate_ema(tr, 14)
        
        # ====== NEW FEATURES ======
        
        # 10. Taker buy ratio (buy pressure)
        features[:, 9] = taker_buy_volumes / (volumes + 1e-8)
        
        # 11. Spread (volatility/liquidity)
        features[:, 10] = spreads
        
        # 12. Volume momentum (change in volume)
        vol_momentum = np.zeros(n)
        vol_momentum[1:] = (volumes[1:] - volumes[:-1]) / (volumes[:-1] + 1e-8)
        features[:, 11] = vol_momentum
        
        # 13. Trades per volume (activity intensity)
        features[:, 12] = num_trades / (volumes + 1e-8)
        
        return features
def fetch_data(db_config, months=12):
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    end = datetime.utcnow()
    start = end - timedelta(days=months * 30)
    
    # ZMIEŃ - dodaj nowe kolumny
    cur.execute("""
        SELECT EXTRACT(EPOCH FROM time)::bigint, 
               close, high, low, volume,
               taker_buy_base_asset_volume,
               number_of_trades,
               spread_bps
        FROM candles_1m
        WHERE time >= %s AND time <= %s
        ORDER BY time ASC
    """, (start, end))
    
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    if len(rows) < 1000:
        raise ValueError(f"Insufficient data: {len(rows)}")
    
    timestamps = np.array([r[0] for r in rows])
    closes = np.array([r[1] for r in rows], dtype=np.float32)
    highs = np.array([r[2] for r in rows], dtype=np.float32)
    lows = np.array([r[3] for r in rows], dtype=np.float32)
    volumes = np.array([r[4] for r in rows], dtype=np.float32)
    taker_buy_volumes = np.array([r[5] for r in rows], dtype=np.float32)  # NEW
    num_trades = np.array([r[6] for r in rows], dtype=np.float32)  # NEW
    spreads = np.array([r[7] if r[7] is not None else 0.0 for r in rows], dtype=np.float32)  # NEW
    
    return timestamps, closes, highs, lows, volumes, taker_buy_volumes, num_trades, spreads

def create_sequences(features, targets, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

def build_model():
    model = keras.Sequential([
        layers.Masking(mask_value=0., input_shape=(60, 13)),  # 9 → 13
        layers.LSTM(128, return_sequences=True, time_major=False,
                    recurrent_activation='sigmoid'),
        layers.Dropout(0.3),
        layers.LSTM(64, time_major=False,
                    recurrent_activation='sigmoid'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear', dtype='float32'),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def register_model_in_db(db_config, version, horizon, model_path, norm_params, metrics, samples, duration):
    """Register trained model in database"""
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    # Now insert new model
    cur.execute("""
        INSERT INTO ml_models 
        (version, model_path, normalization_params, training_metrics, 
         is_active, trained_on_samples, training_duration_seconds, test_rmse, test_mape, horizon)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (version) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            normalization_params = EXCLUDED.normalization_params,
            training_metrics = EXCLUDED.training_metrics,
            is_active = EXCLUDED.is_active,
            test_rmse = EXCLUDED.test_rmse,
            test_mape = EXCLUDED.test_mape,
            horizon = EXCLUDED.horizon
    """, (
        f"{version}_{horizon}min",  # 1
        model_path,                  # 2
        json.dumps(norm_params),     # 3
        json.dumps(metrics),         # 4
        True,                        # 5
        samples,                     # 6
        duration,                    # 7
        metrics.get('test_rmse_log'), # 8
        metrics.get('test_mape'),    # 9
        horizon                      # 10
    ))
    
    conn.commit()
    cur.close()
    conn.close()

def main():
    print("=" * 80)
    print("ML Layer 1 - Multi-Horizon Training (3 Models)")
    print("Using LOG RETURNS for stable training")
    print("=" * 80)
    
    DB_CONFIG = {
        'host': os.getenv("DB_HOST", "localhost"),
        'port': os.getenv("DB_PORT", "5432"),
        'dbname': os.getenv("DB_NAME", "btc_ml_production"),
        'user': os.getenv("DB_USER", "mltrader"),
        'password': os.getenv("DB_PASSWORD", "your_secure_password_here")
    }
    
    MODELS_DIR = Path(os.getenv("ML_MODELS_DIR", "/app/models"))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    # Fetch data
    print("\nFetching data from DB...")
    timestamps, closes, highs, lows, volumes, taker_buy_volumes, num_trades, spreads = fetch_data(DB_CONFIG, 12)
    print(f"✓ Fetched {len(closes):,} candles")

    # Compute features
    print("\nComputing features (13 total)...")
    features = FeatureComputer.compute_features(closes, highs, lows, volumes, taker_buy_volumes, num_trades, spreads)

    # Train 3 horizons
    horizons = [1, 5, 15]
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Single scaler for all models
    scaler = StandardScaler()
    
    for idx, horizon in enumerate(horizons):
        print(f"\n{'='*80}")
        print(f"Training {horizon}-minute model")
        print('='*80)
        
        # Create LOG RETURNS targets
        # log(price_future / price_current)
        targets = np.log(closes[horizon:] / closes[:-horizon])
        
        # Trim features to match targets
        features_h = features[:-horizon]
        
        # Normalize features (fit only on first horizon)
        if idx == 0:
            features_norm = np.clip(scaler.fit_transform(features_h), -10, 10)
        else:
            features_norm = np.clip(scaler.transform(features_h), -10, 10)
        
        # Split: 60% train, 20% val, 20% test
        n = len(features_norm)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train, y_train = create_sequences(features_norm[:train_end], targets[:train_end])
        X_val, y_val = create_sequences(features_norm[train_end:val_end], targets[train_end:val_end])
        X_test, y_test = create_sequences(features_norm[val_end:], targets[val_end:])
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Build and train model
        model = build_model()
        print(f"\nTraining {horizon}min model...")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            ],
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test, batch_size=256, verbose=0).flatten()
        
        # MSE on log returns
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Direction accuracy (up/down)
        direction_correct = ((y_pred > 0) == (y_test > 0)).mean() * 100
        
        # MAPE on actual prices (convert log returns back)
        # For display only - inference will use log returns
        test_base_idx = val_end + 60  # Account for sequence length
        actual_prices = closes[test_base_idx + horizon : test_base_idx + horizon + len(y_test)]
        pred_prices = closes[test_base_idx : test_base_idx + len(y_test)] * np.exp(y_pred)
        mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100
        
        print(f"\n{horizon}min Test Metrics:")
        print(f"  RMSE (log returns): {rmse:.6f}")
        print(f"  MAPE (actual $): {mape:.2f}%")
        print(f"  Direction Accuracy: {direction_correct:.2f}%")
        
        # Export to ONNX
        onnx_path = MODELS_DIR / f'lstm_btc_{horizon}min.onnx'
        input_sig = [tf.TensorSpec(shape=(None, 60, 9), dtype=tf.float32, name='input')]
        tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=14, output_path=str(onnx_path))
        
        print(f"✓ Exported: {onnx_path}")
        
        # Register in DB
        metrics = {
            'horizon': horizon,
            'test_rmse_log': float(rmse),
            'test_mape': float(mape),
            'direction_acc': float(direction_correct),
            'uses_log_returns': True  # IMPORTANT!
        }
        
        register_model_in_db(
            DB_CONFIG,
            version,
            horizon,
            str(onnx_path),
            {'mean': scaler.mean_.tolist(), 'std': scaler.scale_.tolist()},
            metrics,
            len(X_train),
            (datetime.now() - start_time).total_seconds()
        )
    
    # Save normalization params
    norm_params = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'uses_log_returns': True  # Tell inference to use exp()
    }
    with open(MODELS_DIR / 'normalization_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ All 3 models trained and exported!")
    print(f"✓ Version: {version}")
    print(f"✓ Duration: {(datetime.now() - start_time).total_seconds():.1f}s")
    print(f"✓ Output format: LOG RETURNS (use exp() for price)")
    print('='*80)

if __name__ == "__main__":
    main()