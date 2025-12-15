# ============================================================================
# ML Layer 1 - Rust Inference Dockerfile
# Multi-stage build for standalone inference service
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Compile Rust binary with ML dependencies
# -----------------------------------------------------------------------------
FROM rust:1.83-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    libpq-dev \
    curl \
    unzip \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download and install libtorch (CPU version)
WORKDIR /opt
RUN curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip \
    && unzip libtorch.zip \
    && rm libtorch.zip

# Download and install ONNX Runtime
RUN curl -L -o onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz \
    && tar xzf onnxruntime.tgz \
    && mv onnxruntime-linux-x64-1.16.3 onnxruntime \
    && rm onnxruntime.tgz

# Set environment for libtorch and onnxruntime
ENV LIBTORCH=/opt/libtorch
ENV LIBTORCH_INCLUDE=/opt/libtorch
ENV LIBTORCH_LIB=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/onnxruntime/lib:$LD_LIBRARY_PATH
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so

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
RUN strip target/release/ml-layer-1-inference 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal image with ML libraries
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim

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

# Create non-root user
RUN useradd -m -u 1000 inference

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/ml-layer-1-inference /app/inference

# Create models directory
RUN mkdir -p /app/models && chown -R inference:inference /app

# Switch to non-root user
USER inference

# Expose health port (if implementing HTTP health endpoint)
EXPOSE 8081

# Environment defaults
ENV RUST_LOG=info
ENV ML_ENABLED=true
ENV ML_DEVICE=cpu

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8081/health || exit 1

# Run the binary
ENTRYPOINT ["/app/inference"]
