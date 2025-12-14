# ============================================================================
# BTC ML Production - Go Gap Handler Dockerfile
# Multi-stage build for minimal production image
# Target size: <30MB
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Compile Go binary with all optimizations
# -----------------------------------------------------------------------------
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    git \
    ca-certificates \
    tzdata

# Create build directory
WORKDIR /build

# Copy go.mod AND source files first
COPY gap-handler/go.mod ./
COPY gap-handler/*.go ./

# Generate go.sum and download dependencies
# go mod tidy analyzes source code to find all imports, then computes checksums
RUN go mod tidy && go mod download

# Build the binary with optimizations
# CGO_ENABLED=0 for static binary
# -ldflags for smaller binary size
# -trimpath removes file system paths from binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w -X main.version=1.0.0 -X main.buildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -trimpath \
    -o gap-handler \
    .

# Verify binary exists and show size
RUN ls -lh gap-handler

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal Alpine image
# -----------------------------------------------------------------------------
FROM alpine:3.19

# Add labels for image metadata
LABEL maintainer="ML Trading Team"
LABEL description="Gap backfill handler for Binance REST API"
LABEL version="1.0"

# Install runtime dependencies only
# curl for health checks, ca-certificates for HTTPS
RUN apk add --no-cache \
    ca-certificates \
    curl \
    tzdata \
    && rm -rf /var/cache/apk/*

# Create non-root user for security
RUN addgroup -g 1000 handler && \
    adduser -D -u 1000 -G handler -h /app handler

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/gap-handler /app/gap-handler

# Copy timezone data for time handling
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# Create logs directory
RUN mkdir -p /app/logs && \
    chown -R handler:handler /app

# Switch to non-root user
USER handler

# Expose service port
EXPOSE 9000

# Environment defaults (can be overridden)
ENV LOG_LEVEL=info
ENV TZ=UTC

# Health check configuration
# Checks /health endpoint every 10 seconds
# Allows 30 seconds for startup before checking
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fsS http://localhost:9000/health || exit 1

# Graceful shutdown: SIGTERM is handled by the application
STOPSIGNAL SIGTERM

# Run the binary
ENTRYPOINT ["/app/gap-handler"]
