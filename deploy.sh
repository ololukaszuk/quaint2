#!/bin/bash

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ BTC ML Production - Deployment Script                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

# Check if .env exists, if not generate it with secure passwords
if [ ! -f .env ]; then
  echo -e "${YELLOW}⚠ .env file not found. Generating with secure passwords...${NC}"
  
  if [ ! -f .env.example ]; then
    echo -e "${RED}✗ .env.example not found!${NC}"
    exit 1
  fi
  
  # Generate secure random passwords
  DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
  PGADMIN_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
  
  # Copy template and substitute passwords
  cp .env.example .env
  
  # Use sed to safely replace password placeholders
  if command -v sed &> /dev/null; then
    # For macOS and Linux compatibility
    if [[ "$OSTYPE" == "darwin"* ]]; then
      sed -i '' "s|DB_PASSWORD=.*|DB_PASSWORD=${DB_PASSWORD}|g" .env
      sed -i '' "s|PGADMIN_PASSWORD=.*|PGADMIN_PASSWORD=${PGADMIN_PASSWORD}|g" .env
    else
      sed -i "s|DB_PASSWORD=.*|DB_PASSWORD=${DB_PASSWORD}|g" .env
      sed -i "s|PGADMIN_PASSWORD=.*|PGADMIN_PASSWORD=${PGADMIN_PASSWORD}|g" .env
    fi
  fi
  
  echo -e "${GREEN}✓ Created .env with secure passwords${NC}"
  echo -e "${BLUE}Generated Credentials:${NC}"
  echo -e " DB_PASSWORD:       ${YELLOW}${DB_PASSWORD}${NC}"
  echo -e " PGADMIN_PASSWORD:  ${YELLOW}${PGADMIN_PASSWORD}${NC}"
  echo ""
  echo -e "${YELLOW}⚠ IMPORTANT: Save these credentials in a secure location!${NC}"
  echo ""
else
  echo -e "${GREEN}✓ Using existing .env file${NC}"
fi

# Load environment
source .env

# Create required directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p timescaledb
mkdir -p data-feeder/logs
mkdir -p gap-handler/logs
mkdir -p pgadmin
echo -e "${GREEN}✓ Directories created${NC}"

# Check for required files
echo -e "${BLUE}Checking required files...${NC}"

required_files=(
  "timescaledb/Dockerfile"
  "timescaledb/init.sql"
  "data-feeder/Dockerfile"
  "data-feeder/Cargo.toml"
  "gap-handler/Dockerfile"
  "gap-handler/go.mod"
  "pgadmin/servers.json"
  "docker-compose.yml"
)

missing_files=()

for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    missing_files+=("$file")
  fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
  echo -e "${RED}✗ Missing required files:${NC}"
  for file in "${missing_files[@]}"; do
    echo -e "${RED} - $file${NC}"
  done
  echo -e "${YELLOW}Please generate these files from the prompts before continuing.${NC}"
  exit 1
fi

echo -e "${GREEN}✓ All required files present${NC}"

# Validate environment variables
echo -e "${BLUE}Validating environment variables...${NC}"

required_vars=(
  "DB_PASSWORD"
  "PGADMIN_PASSWORD"
  "DB_HOST"
  "DB_PORT"
  "DB_NAME"
  "DB_USER"
)

missing_vars=()

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    missing_vars+=("$var")
  fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
  echo -e "${RED}✗ Missing environment variables:${NC}"
  for var in "${missing_vars[@]}"; do
    echo -e "${RED} - $var${NC}"
  done
  echo -e "${YELLOW}Please update .env file with required variables.${NC}"
  exit 1
fi

echo -e "${GREEN}✓ All environment variables set${NC}"

# Build Docker images
echo -e "${BLUE}Building Docker images...${NC}"
echo -e "${BLUE}This may take 10-15 minutes on first run (pg_cron compilation)...${NC}"

docker-compose build --no-cache

if [ $? -ne 0 ]; then
  echo -e "${RED}✗ Docker build failed!${NC}"
  echo -e "${YELLOW}Check logs for errors. This may take a few minutes first time.${NC}"
  exit 1
fi

echo -e "${GREEN}✓ Docker images built successfully${NC}"

# Start services
echo -e "${BLUE}Starting services...${NC}"

docker-compose up -d

if [ $? -ne 0 ]; then
  echo -e "${RED}✗ Failed to start services!${NC}"
  exit 1
fi

echo -e "${GREEN}✓ Services started${NC}"

# Wait for TimescaleDB to be ready
echo -e "${BLUE}Waiting for TimescaleDB to be ready...${NC}"

max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
  if docker-compose exec -T timescaledb pg_isready -U mltrader > /dev/null 2>&1; then
    echo -e "${GREEN}✓ TimescaleDB is ready${NC}"
    break
  fi
  attempt=$((attempt+1))
  echo -n "."
  sleep 1
done

if [ $attempt -eq $max_attempts ]; then
  echo -e "${RED}✗ TimescaleDB failed to start!${NC}"
  echo -e "${YELLOW}Check logs: docker-compose logs timescaledb${NC}"
  exit 1
fi

echo -e "${BLUE}Waiting for pg_cron and data-feeder initialization...${NC}"
sleep 10

# Check health endpoints
echo -e "${BLUE}Checking service health...${NC}"

echo -n " Data Feeder: "
if curl -s http://localhost:${FEEDER_PORT:-8080}/health > /dev/null 2>&1; then
  echo -e "${GREEN}✓ Healthy${NC}"
else
  echo -e "${YELLOW}⚠ Not ready yet (may take a moment)${NC}"
fi

echo -n " Gap Handler: "
if curl -s http://localhost:${GAP_HANDLER_PORT:-9000}/health > /dev/null 2>&1; then
  echo -e "${GREEN}✓ Healthy${NC}"
else
  echo -e "${YELLOW}⚠ Not ready yet (may take a moment)${NC}"
fi

# Display service information
echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║ Deployment Complete!                                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"

echo ""

echo -e "${BLUE}Service Endpoints:${NC}"
echo -e " Data Feeder:   http://localhost:${FEEDER_PORT:-8080}/health"
echo -e " Gap Handler:   http://localhost:${GAP_HANDLER_PORT:-9000}/health"
echo -e " pgAdmin:       http://localhost:${PGADMIN_PORT:-8000}"
echo -e " Database:      localhost:${DB_PORT:-5432}"

echo ""

echo -e "${BLUE}Database Access:${NC}"
echo -e " Host:    ${DB_HOST:-localhost}"
echo -e " Port:    ${DB_PORT:-5432}"
echo -e " User:    ${DB_USER:-mltrader}"
echo -e " Database:${DB_NAME:-btc_ml_production}"
echo -e " Password:${DB_PASSWORD}"

echo ""

echo -e "${BLUE}pgAdmin Access:${NC}"
echo -e " Email:   admin@example.com"
echo -e " Password:${PGADMIN_PASSWORD}"

echo ""

echo -e "${BLUE}pg_cron Scheduled Jobs:${NC}"
echo -e " To view: docker-compose exec timescaledb psql -U mltrader -d btc_ml_production -c \"SELECT * FROM cron.job;\""

echo ""

echo -e "${BLUE}Useful Commands:${NC}"
echo -e " View all logs:        ${YELLOW}docker-compose logs -f${NC}"
echo -e " View feeder logs:     ${YELLOW}docker-compose logs -f data-feeder${NC}"
echo -e " View gap handler:     ${YELLOW}docker-compose logs -f gap-handler${NC}"
echo -e " View database logs:   ${YELLOW}docker-compose logs -f timescaledb${NC}"
echo -e " Stop services:        ${YELLOW}docker-compose down${NC}"
echo -e " Clean up:             ${YELLOW}./cleanup.sh${NC}"

echo ""

echo -e "${BLUE}Next Steps:${NC}"
echo -e " 1. Monitor logs for 5-10 minutes to ensure stability"
echo -e " 2. Check database: psql -h localhost -U mltrader -d btc_ml_production"
echo -e " 3. Query candles: SELECT COUNT(*) FROM candles_1m;"
echo -e " 4. Access pgAdmin at http://localhost:${PGADMIN_PORT:-8000}"
echo -e " 5. Verify pg_cron jobs are running"

echo ""

echo -e "${GREEN}✓ Deployment script completed successfully${NC}"

echo ""
