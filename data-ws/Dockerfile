# Data WebSocket Service - Dockerfile
FROM rust:1.75-slim AS builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    cargo build --release && rm -rf src

COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates libssl3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/target/release/data-ws /app/

ENV RUST_LOG=info
EXPOSE 8443

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -fk https://localhost:8443/health || exit 1

CMD ["/app/data-ws"]
