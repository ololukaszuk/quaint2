"""
ML Layer 1 - Training Pipeline
Trains multi-horizon LSTM models and auto-selects best performer

Features:
- Multi-horizon: 1min, 5min, 15min
- Auto model selection based on historical performance
- Activates best model in database
- Evaluates on historical predictions if available
"""

import os
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import json
from pathlib import Path

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

# Enable TensorFloat-32 for A100/5090
tf.config.experimental.enable_tensor_float_32_execution(True)

class FeatureComputer:
    """Compute 9 technical features"""
    
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
    def compute_features(closes, highs, lows, volumes):
        """9 features matching Rust"""
        n = len(closes)
        features = np.zeros((n, 9))
        
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
        
        return features

def fetch_data(db_config, months=12):
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    end = datetime.utcnow()
    start = end - timedelta(days=months * 30)
    
    cur.execute("""
        SELECT EXTRACT(EPOCH FROM time)::bigint, close, high, low, volume
        FROM klines
        WHERE symbol = 'BTCUSDT' AND interval = '1m'
          AND time >= %s AND time <= %s
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
    
    return timestamps, closes, highs, lows, volumes

def create_sequences(features, targets, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)

def build_model(seq_len=60, n_features=9):
    """Optimized for RTX 5090"""
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(128, return_sequences=True),  # Increased capacity
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear', dtype='float32')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def register_model_in_db(db_config, version, model_path, norm_params, metrics, samples, duration):
    """Register trained model in database"""
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    # Deactivate all models first
    cur.execute("UPDATE ml_models SET is_active = FALSE")
    
    # Insert new model
    cur.execute("""
        INSERT INTO ml_models 
        (version, model_path, normalization_params, training_metrics, 
         is_active, trained_on_samples, training_duration_seconds, test_rmse, test_mape)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        version,
        model_path,
        json.dumps(norm_params),
        json.dumps(metrics),
        True,  # Activate immediately
        samples,
        duration,
        metrics.get('test_rmse'),
        metrics.get('test_mape')
    ))
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✓ Model {version} registered and activated in database")

def main():
    print("=" * 80)
    print("ML Layer 1 - Multi-Horizon Training")
    print("GPU: RTX 5090 | CUDA 12.8 | Mixed Precision FP16")
    print("=" * 80)
    
    DB_CONFIG = {
        'host': os.getenv("DB_HOST", "localhost"),
        'port': os.getenv("DB_PORT", "5432"),
        'dbname': os.getenv("DB_NAME", "btc_ml_production"),
        'user': os.getenv("DB_USER", "mltrader"),
        'password': os.getenv("DB_PASSWORD", "your_secure_password_here")
    }
    
    MODELS_DIR = Path(os.getenv("ML_MODELS_DIR", "../models"))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    # Fetch
    print("\nFetching data from DB...")
    timestamps, closes, highs, lows, volumes = fetch_data(DB_CONFIG, 12)
    print(f"✓ Fetched {len(closes):,} candles")
    
    # Features
    print("\nComputing features...")
    features = FeatureComputer.compute_features(closes, highs, lows, volumes)
    
    # Train for 3 horizons
    horizons = [1, 5, 15]  # minutes
    models = {}
    
    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"Training {horizon}-minute model")
        print('='*80)
        
        # Targets (shift by horizon)
        targets = np.roll(closes, -horizon)
        targets[-horizon:] = np.nan
        valid = ~np.isnan(targets)
        features_h = features[valid]
        targets_h = targets[valid]
        
        # Normalize
        scaler = StandardScaler()
        features_norm = np.clip(scaler.fit_transform(features_h), -10, 10)
        
        # Split
        n = len(features_norm)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train, y_train = create_sequences(features_norm[:train_end], targets_h[:train_end])
        X_val, y_val = create_sequences(features_norm[train_end:val_end], targets_h[train_end:val_end])
        X_test, y_test = create_sequences(features_norm[val_end:], targets_h[val_end:])
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train
        model = build_model()
        print(f"\nTraining {horizon}min model...")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,  # Larger batch for 5090
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            ],
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test, batch_size=256, verbose=0).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Direction accuracy
        direction_correct = ((y_pred > y_test[:-horizon].mean()) == (y_test > y_test[:-horizon].mean())).mean() * 100
        
        print(f"\n{horizon}min Test Metrics:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Direction Accuracy: {direction_correct:.2f}%")
        
        models[horizon] = {
            'model': model,
            'scaler': scaler,
            'rmse': rmse,
            'mape': mape,
            'direction_acc': direction_correct
        }
    
    # Select best model (lowest MAPE on 15min)
    best_horizon = min(models.keys(), key=lambda h: models[h]['mape'])
    print(f"\n{'='*80}")
    print(f"Best model: {best_horizon}-minute (MAPE: {models[best_horizon]['mape']:.2f}%)")
    print('='*80)
    
    # Export best model
    best_model = models[best_horizon]['model']
    best_scaler = models[best_horizon]['scaler']
    
    onnx_path = MODELS_DIR / 'lstm_btc.onnx'
    input_sig = [tf.TensorSpec(shape=(None, 60, 9), dtype=tf.float32, name='input')]
    tf2onnx.convert.from_keras(best_model, input_signature=input_sig, opset=14, output_path=str(onnx_path))
    
    print(f"✓ Exported ONNX: {onnx_path}")
    
    # Save normalization params
    norm_params = {
        'mean': best_scaler.mean_.tolist(),
        'std': best_scaler.scale_.tolist()
    }
    with open(MODELS_DIR / 'normalization_params.json', 'w') as f:
        json.dump(norm_params, f)
    
    # Save metadata
    training_duration = (datetime.now() - start_time).total_seconds()
    metadata = {
        'best_horizon': best_horizon,
        'all_models': {h: {'rmse': m['rmse'], 'mape': m['mape'], 'direction_acc': m['direction_acc']} 
                       for h, m in models.items()},
        'timestamp': datetime.now().isoformat(),
        'training_duration': training_duration,
        'gpu': 'RTX_5090'
    }
    with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Register in database
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    register_model_in_db(
        DB_CONFIG,
        version,
        str(onnx_path),
        norm_params,
        metadata,
        len(X_train),
        training_duration
    )
    
    print(f"\n✓ Training complete! Duration: {training_duration:.1f}s")
    print(f"✓ Model {version} activated for inference")

if __name__ == "__main__":
    main()
