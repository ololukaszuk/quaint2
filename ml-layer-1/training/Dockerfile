# ============================================================================
# ML Layer 1 - Python Training Dockerfile
# CUDA 12.8 + RTX 5090 Support (ubuntu22.04 base)
# For daily retraining and model export
# ============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip FIRST (without --break-system-packages)
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.8 support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install ONNX Runtime GPU 1.19.2
RUN pip install --no-cache-dir onnxruntime-gpu==1.19.2

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create the training package directory structure
RUN mkdir -p /app/training/scripts

# Copy training code into the training/ package directory
COPY __init__.py /app/training/
COPY config.py /app/training/
COPY data_loader.py /app/training/
COPY trainer.py /app/training/
COPY evaluator.py /app/training/
COPY export_models.py /app/training/
COPY feature_engineering.py /app/training/
COPY lightgbm_models.py /app/training/
COPY mamba_model.py /app/training/
COPY metrics.py /app/training/
COPY predictor.py /app/training/
COPY scripts/__init__.py /app/training/scripts/
COPY scripts/daily_retrain.py /app/training/scripts/

# Create models output directory
RUN mkdir -p /app/models

# Set CUDA paths
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create non-root user (training:1000)
RUN groupadd -g 1000 training && \
    useradd -m -u 1000 -g training training && \
    chown -R training:training /app

# Switch to non-root user
USER training

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV ML_DEVICE=cuda

# Health check - verify Python, CUDA, and imports work
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('OK')" || exit 1

# Default command: run daily retrain script
ENTRYPOINT ["python", "-m", "training.scripts.daily_retrain"]
