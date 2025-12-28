# Data WebSocket Service

WebSocket server that streams ML predictions from TimescaleDB.

## 🔐 Security

- **HTTPS**: Self-signed certificate (auto-generated)
- **API Key**: Bearer token authentication
- **CORS**: Configurable

## 🔌 Endpoints

### REST API

#### GET /health
Health check (no auth required)

```bash
curl -k https://localhost:8443/health
```

#### GET /api/v1/predictions/latest
Latest prediction (requires API key)

```bash
curl -k -H "Authorization: Bearer YOUR_API_KEY" \
  https://localhost:8443/api/v1/predictions/latest
```

Response:
```json
{
  "success": true,
  "data": {
    "time": "2024-12-28T12:00:00Z",
    "current_price": 50000.0,
    "predicted_15min": 50125.42,
    "confidence_15min": 0.78,
    "model_version": "v1.0"
  }
}
```

### WebSocket

#### WS /api/v1/predictions/stream
Real-time prediction stream (requires API key)

```javascript
const ws = new WebSocket('wss://localhost:8443/api/v1/predictions/stream');
ws.onopen = () => {
  // Send API key (or use query param)
};
ws.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  console.log(prediction);
};
```

## 🚀 Deployment

### Docker

```bash
docker build -t data-ws .
docker run -p 8443:8443 \
  -e API_KEY=your_secure_key \
  -e DB_HOST=timescaledb \
  data-ws
```

### Environment Variables

```bash
DATA_WS_BIND_ADDR=0.0.0.0:8443
API_KEY=your_secure_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=btc_ml_production
DB_USER=mltrader
DB_PASSWORD=password
```

## 🔄 Data Flow

```
ml_predictions (TimescaleDB)
    ↓ (queries every 5s)
data-ws
    ↓ (streams via WebSocket)
Clients
```

## 📊 Example Client (Python)

```python
import websockets
import ssl
import json

async def stream():
    ssl_context = ssl._create_unverified_context()
    
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    
    async with websockets.connect(
        "wss://localhost:8443/api/v1/predictions/stream",
        extra_headers=headers,
        ssl=ssl_context
    ) as ws:
        async for message in ws:
            pred = json.loads(message)
            print(f"Price: {pred['current_price']} → {pred['predicted_15min']}")

asyncio.run(stream())
```

## 🎯 Integration

Similar to data-api:
- HTTPS with self-signed cert
- Bearer token authentication
- JSON responses
- CORS enabled

Perfect for frontend dashboards and trading bots!
