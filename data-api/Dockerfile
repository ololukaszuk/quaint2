FROM golang:1.21-alpine AS builder

WORKDIR /build

# Install dependencies
RUN apk add --no-cache git

# Copy source
COPY main.go ./

# Work on go mod file
COPY go.mod ./
RUN go mod tidy && go mod download

# Build
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o data-api .

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates openssl

WORKDIR /app

# Copy binary
COPY --from=builder /build/data-api .

# Create directory for certificates
RUN mkdir -p /certs

# Generate self-signed certificate (will be replaced with proper cert in production)
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /certs/server.key \
    -out /certs/server.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT:-8080}/api/v1/health || exit 1

EXPOSE 8443
EXPOSE 8080

CMD ["./data-api"]
