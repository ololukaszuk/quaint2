# ============================================================================
# ML Layer 1 - Python Training Dockerfile
# For daily retraining and model export
# ============================================================================

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY . .

# Create models output directory
RUN mkdir -p /app/models

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=info

# Health check - verify Python and imports work
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import lightgbm; print('OK')" || exit 1

# Default command: run daily retrain script
CMD ["python", "scripts/daily_retrain.py"]
