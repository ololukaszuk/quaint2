#!/bin/bash

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ BTC ML Production - Cleanup Script                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

# Confirm cleanup
echo -e "${YELLOW}⚠ This will stop and remove all Docker containers and clean up logs.${NC}"
echo -e "${YELLOW}Database volumes and pgAdmin data will be preserved by default.${NC}"

read -p "Continue? (yes/no): " -n 3 -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
  echo -e "${YELLOW}Cleanup cancelled.${NC}"
  exit 0
fi

# Stop services
echo -e "${BLUE}Stopping services...${NC}"

docker-compose down

if [ $? -ne 0 ]; then
  echo -e "${RED}✗ Failed to stop services!${NC}"
  exit 1
fi

echo -e "${GREEN}✓ Services stopped${NC}"

# Clean logs
echo -e "${BLUE}Cleaning logs...${NC}"

if [ -d "data-feeder/logs" ]; then
  rm -rf data-feeder/logs/*
  echo -e "${GREEN}✓ Data feeder logs cleaned${NC}"
fi

if [ -d "gap-handler/logs" ]; then
  rm -rf gap-handler/logs/*
  echo -e "${GREEN}✓ Gap handler logs cleaned${NC}"
fi

# Optional: Remove volumes
echo ""

echo -e "${YELLOW}Do you want to remove Docker volumes?${NC}"
echo -e "${YELLOW}WARNING: This will delete all database data and pgAdmin configuration!${NC}"

read -p "Remove volumes? (yes/no): " -n 3 -r
echo

if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
  docker-compose down -v
  echo -e "${GREEN}✓ Volumes removed${NC}"
fi

# Optional: Remove images
echo ""

echo -e "${YELLOW}Do you want to remove Docker images?${NC}"
echo -e "${YELLOW}This will free up disk space but requires rebuilding on next deploy.${NC}"

read -p "Remove images? (yes/no): " -n 3 -r
echo

if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
  echo -e "${BLUE}Removing images...${NC}"
  docker-compose down --rmi all
  echo -e "${GREEN}✓ Images removed${NC}"
  
  # Clean build cache for fresh pg_cron compile
  echo -e "${BLUE}Cleaning Docker build cache...${NC}"
  docker builder prune -f
  echo -e "${GREEN}✓ Build cache cleared${NC}"
fi

echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║ Cleanup Complete!                                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"

echo ""

echo -e "${BLUE}To redeploy, run:${NC}"
echo -e " ${YELLOW}./deploy.sh${NC}"

echo ""

# Show disk space saved
if command -v docker &> /dev/null; then
  echo -e "${BLUE}Docker Disk Space:${NC}"
  docker system df
  echo ""
fi
