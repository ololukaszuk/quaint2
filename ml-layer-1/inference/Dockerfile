# ============================================================================
# ML Layer 1 - Rust Inference Dockerfile
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
    && rm -rf /var/lib/apt/lists/*

# Install Rust 1.83
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.83.0
ENV PATH="/root/.cargo/bin:${PATH}"

# Download and install libtorch with CUDA 12.8 support
WORKDIR /opt
RUN curl -L -o libtorch.zip "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu128.zip" \
    && unzip -q libtorch.zip \
    && rm libtorch.zip

# Download and install ONNX Runtime GPU 1.19.2
RUN curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz" \
    && tar xzf onnxruntime.tgz \
    && mv onnxruntime-linux-x64-gpu-1.19.2 onnxruntime \
    && rm onnxruntime.tgz

# Set environment for libtorch and onnxruntime
ENV LIBTORCH=/opt/libtorch
ENV LIBTORCH_INCLUDE=/opt/libtorch
ENV LIBTORCH_LIB=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/onnxruntime/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV CUDA_HOME=/usr/local/cuda

# Create build directory
WORKDIR /build

# Copy Cargo files first for dependency caching
COPY Cargo.toml ./

# Create dummy main for dependency build
RUN mkdir -p src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs && \
    echo 'pub fn dummy() {}' > src/lib.rs

# Build dependencies only (caching layer)
RUN cargo build --release --features ml-inference 2>/dev/null || true

# Remove dummy source
RUN rm -rf src

# Copy actual source files
COPY src ./src/

# Touch to ensure rebuild
RUN touch src/main.rs src/lib.rs

# Build the release binary
RUN cargo build --release --features ml-inference

# Strip binary for smaller size
RUN strip target/release/ml-inference 2>/dev/null || true

# Verify binary
RUN ls -lh target/release/ml-inference

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

# Copy libtorch and onnxruntime libraries
COPY --from=builder /opt/libtorch/lib/*.so* /usr/local/lib/
COPY --from=builder /opt/onnxruntime/lib/*.so* /usr/local/lib/

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
