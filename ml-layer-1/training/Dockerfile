# Najnowszy NVIDIA NGC TensorFlow z RTX 5090 support
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install PostgreSQL client libs
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/training

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Install only non-TensorFlow deps
RUN grep -v "^tensorflow" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt || \
    pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN mkdir -p /app/models

ENV CUDA_VISIBLE_DEVICES=0
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_LAUNCH_BLOCKING=0
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"


CMD ["python", "train.py"]