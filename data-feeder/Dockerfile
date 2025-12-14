# ============================================================================
# BTC ML Production - Rust Data Feeder Dockerfile
# Multi-stage build for minimal production image
# Target size: <50MB
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Compile Rust binary with all optimizations
# -----------------------------------------------------------------------------
FROM rust:1.83-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    musl-dev \
    openssl-dev \
    openssl-libs-static \
    pkgconfig \
    perl \
    make

# Create build directory
WORKDIR /build

# Copy Cargo files first for dependency caching
COPY data-feeder/Cargo.toml data-feeder/Cargo.lock* ./

# Create dummy source to build dependencies (caching layer)
RUN mkdir -p src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs && \
    echo 'pub fn dummy() {}' > src/lib.rs

# Build dependencies only (this layer gets cached)
RUN cargo build --release 2>/dev/null || true

# Remove dummy source
RUN rm -rf src

# Copy actual source files
COPY data-feeder/src ./src/

# Touch main.rs to ensure it's newer than cached dependencies
RUN touch src/main.rs

# Build the release binary with full optimizations
# RUSTFLAGS for size optimization (removed lto=thin due to Alpine compatibility)
ENV RUSTFLAGS="-C target-feature=-crt-static -C opt-level=3"

RUN cargo build --release

# Strip the binary for smaller size
RUN strip target/release/data-feeder 2>/dev/null || true

# Verify binary exists and show size
RUN ls -lh target/release/data-feeder

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal Alpine image
# -----------------------------------------------------------------------------
FROM alpine:3.19

# Add labels for image metadata
LABEL maintainer="ML Trading Team"
LABEL description="Binance WebSocket data feeder for TimescaleDB"
LABEL version="1.0"

# Install runtime dependencies only
RUN apk add --no-cache \
    ca-certificates \
    openssl \
    libgcc \
    wget \
    && rm -rf /var/cache/apk/*

# Create non-root user for security
RUN addgroup -g 1000 feeder && \
    adduser -D -u 1000 -G feeder -h /app feeder

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/data-feeder /app/data-feeder

# Create logs directory
RUN mkdir -p /app/logs && \
    chown -R feeder:feeder /app

# Switch to non-root user
USER feeder

# Expose health check port
EXPOSE 8080

# Environment defaults (can be overridden)
ENV RUST_LOG=info
ENV RUST_BACKTRACE=0

# Health check configuration
# Checks /health endpoint every 10 seconds
# Allows 30 seconds for startup before checking
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8080/health || exit 1

# Graceful shutdown: SIGTERM is handled by the application
STOPSIGNAL SIGTERM

# Run the binary
ENTRYPOINT ["/app/data-feeder"]
