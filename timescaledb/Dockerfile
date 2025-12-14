# ============================================================================
# TimescaleDB with pg_cron - Custom Build for BTC ML Production
# Extends timescale/timescaledb:latest-pg17 with pg_cron extension
# ============================================================================

FROM timescale/timescaledb:latest-pg17

# Install pg_cron for PostgreSQL 17
USER root

# Install build dependencies for compiling pg_cron
RUN apk add --no-cache \
    postgresql17-dev \
    gcc \
    musl-dev \
    make \
    perl

# Download and compile pg_cron for PostgreSQL 17
# Use with_llvm=no to skip LLVM bitcode generation (requires clang)
WORKDIR /tmp
RUN wget https://github.com/citusdata/pg_cron/archive/refs/tags/v1.6.4.tar.gz && \
    tar -xzf v1.6.4.tar.gz && \
    cd pg_cron-1.6.4 && \
    make with_llvm=no && \
    cp pg_cron.so /usr/local/lib/postgresql/ && \
    cp pg_cron.control /usr/local/share/postgresql/extension/ && \
    cp pg_cron--*.sql /usr/local/share/postgresql/extension/

# Verify pg_cron was installed
RUN ls -la /usr/local/lib/postgresql/pg_cron.so && \
    ls -la /usr/local/share/postgresql/extension/pg_cron.control

# Clean up build dependencies to reduce image size
RUN apk del --no-cache \
    postgresql17-dev \
    gcc \
    musl-dev \
    make \
    perl && \
    rm -rf /tmp/pg_cron* && \
    rm -rf /var/cache/apk/*

# Return to postgres user
USER postgres

# Note: pg_cron requires shared_preload_libraries configuration
# This is handled via command args in docker-compose.yml
