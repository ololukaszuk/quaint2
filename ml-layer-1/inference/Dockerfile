# ============================================================================
# ML Layer 1 - Rust Inference Dockerfile (Option A - PyTorch Fixed)
# CUDA 12.8 + RTX 5090 Support (ubuntu22.04 base)
# Multi-stage build for production deployment
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Compile Rust binary with CUDA support
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    libssl-dev \
    libpq-dev \
    unzip \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust 1.84
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.84.0
ENV PATH="/root/.cargo/bin:${PATH}"

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Pre-download and structure libtorch correctly for tch-rs
WORKDIR /opt

RUN mkdir -p /root/.cache/torch-install && \
    curl -L --max-redirs 5 --retry 3 --retry-delay 2 \
    -o /tmp/libtorch.zip \
    "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip" && \
    unzip -q /tmp/libtorch.zip -d /root/.cache/torch-install && \
    rm /tmp/libtorch.zip

# Optional: sanity check location of torch.h
RUN echo "=== Discovering libtorch structure ===" && \
    find /root/.cache/torch-install/libtorch -maxdepth 5 -name "torch.h" -print || true && \
    echo "=== Directory listing ===" && \
    ls -la /root/.cache/torch-install/libtorch/ && \
    ls -la /root/.cache/torch-install/libtorch/include/ 2>/dev/null || echo "include dir not found at expected location"

# Set environment variables for tch-rs
# IMPORTANT: LIBTORCH_INCLUDE points to libtorch ROOT, torch-sys appends `/include`
ENV LIBTORCH=/root/.cache/torch-install/libtorch
ENV LIBTORCH_INCLUDE=/root/.cache/torch-install/libtorch
ENV LIBTORCH_LIB=/root/.cache/torch-install/libtorch
ENV LD_LIBRARY_PATH=/root/.cache/torch-install/libtorch/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_HOME=/root/.cache/torch-install
ENV DYLD_LIBRARY_PATH=/root/.cache/torch-install/libtorch/lib:$DYLD_LIBRARY_PATH
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV LIBTORCH_CXX11_ABI=1

# Verify libtorch structure
RUN echo "=== Verifying libtorch structure ===" && \
    ls -la /root/.cache/torch-install/libtorch/ && \
    ls -la /root/.cache/torch-install/libtorch/include/ && \
    ls -la /root/.cache/torch-install/libtorch/lib/ && \
    echo "=== PyTorch version check bypassed: LIBTORCH_BYPASS_VERSION_CHECK=1 ===" && \
    echo "=== C++11 ABI enabled: LIBTORCH_CXX11_ABI=1 ==="

# Download and install ONNX Runtime GPU 1.19.2
RUN curl -L --max-redirs 5 --retry 3 --retry-delay 2 \
    -o onnxruntime.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz" && \
    tar xzf onnxruntime.tgz && \
    mv onnxruntime-linux-x64-gpu-1.19.2 onnxruntime && \
    rm onnxruntime.tgz

ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

# Create build directory
WORKDIR /build

# Copy Cargo files first for dependency caching
COPY Cargo.toml ./

# Create dummy main for dependency build
RUN mkdir -p src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs && \
    echo 'pub fn dummy() {}' > src/lib.rs

# Build dependencies only (caching layer)
# Use global LIBTORCH* env; do NOT override LIBTORCH_INCLUDE here
RUN LIBTORCH_BYPASS_VERSION_CHECK=1 \
    LIBTORCH_CXX11_ABI=1 \
    cargo build --release --features ml-inference 2>&1 | tee build.log || true

# Remove dummy source
RUN rm -rf src

# Copy actual source files
COPY src ./src/

# Touch to ensure rebuild
RUN touch src/main.rs src/lib.rs

# Build the release binary with explicit env (but no custom LIBTORCH_INCLUDE)
RUN LIBTORCH_BYPASS_VERSION_CHECK=1 \
    LIBTORCH_CXX11_ABI=1 \
    cargo build --release --features ml-inference

# Strip binary for smaller size
RUN strip target/release/ml-inference 2>/dev/null || true

# Verify binary
RUN ls -lh target/release/ml-inference && \
    ldd target/release/ml-inference | head -20

# Copy libraries for runtime (error-tolerant)
RUN mkdir -p /opt/libs && \
    cp /root/.cache/torch-install/libtorch/lib/*.so* /opt/libs/ 2>/dev/null || echo "Warning: some libtorch libs not found" && \
    cp /opt/onnxruntime/lib/*.so* /opt/libs/ 2>/dev/null || echo "Warning: some ONNX Runtime libs not found" && \
    ls -la /opt/libs/

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal CUDA runtime image
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy libraries from builder
COPY --from=builder /opt/libs/ /usr/local/lib/

# Update library cache
RUN ldconfig

# Create non-root user (inference:1000)
RUN groupadd -g 1000 inference && \
    useradd -m -u 1000 -g inference inference

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/ml-inference /app/inference

# Create models directory
RUN mkdir -p /app/models && chown -R inference:inference /app

# Switch to non-root user
USER inference

# Expose health port
EXPOSE 8081

# Environment defaults
ENV RUST_LOG=info
ENV ML_ENABLED=true
ENV ML_DEVICE=cuda
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8081/health || exit 1

# Run the binary
ENTRYPOINT ["/app/inference"]
