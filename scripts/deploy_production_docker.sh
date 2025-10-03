#!/bin/bash
#
# PyCog-Zero Production Docker Deployment Script
# ==============================================
#
# Production deployment script for containerized PyCog-Zero cognitive agent systems.
# Based on scripts/build_cpp2py_pipeline.sh with production-ready enhancements.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Production configuration
PROD_IMAGE_NAME="${PROD_IMAGE_NAME:-pycog-zero-production}"
PROD_VERSION="${PROD_VERSION:-latest}"
PROD_PORT="${PROD_PORT:-8080}"
PROD_ENV="${PROD_ENV:-production}"
MEMORY_LIMIT="${MEMORY_LIMIT:-4g}"
CPU_LIMIT="${CPU_LIMIT:-2}"

echo -e "${BLUE}PyCog-Zero Production Docker Deployment${NC}"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo "Image Name: $PROD_IMAGE_NAME:$PROD_VERSION"
echo "Production Port: $PROD_PORT"
echo "Environment: $PROD_ENV"
echo "Memory Limit: $MEMORY_LIMIT"
echo "CPU Limit: $CPU_LIMIT"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_header() {
    echo -e "${PURPLE}=== $1 ===${NC}"
}

# Check prerequisites
print_header "Checking Prerequisites"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
print_status "Docker is available"

# Check Docker Compose (optional but recommended)
if command -v docker-compose &> /dev/null; then
    print_status "Docker Compose is available"
    COMPOSE_AVAILABLE=true
else
    print_warning "Docker Compose not available (optional)"
    COMPOSE_AVAILABLE=false
fi

# Validate Docker daemon
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    echo "Please start Docker daemon"
    exit 1
fi
print_status "Docker daemon is running"

# Build production image
print_header "Building Production Image"

print_info "Building PyCog-Zero production image..."
CACHE_DATE=$(date +%Y-%m-%d:%H:%M:%S)

docker build \
    -f "$PROJECT_ROOT/DockerfileLocal" \
    -t "$PROD_IMAGE_NAME:$PROD_VERSION" \
    --build-arg CACHE_DATE="$CACHE_DATE" \
    --build-arg BRANCH=local \
    "$PROJECT_ROOT" || {
    print_error "Failed to build production image"
    exit 1
}

print_status "Production image built successfully"

# Create production configuration
print_header "Creating Production Configuration"

# Create production directory structure
PROD_DIR="$PROJECT_ROOT/production"
mkdir -p "$PROD_DIR"/{config,data,logs,backups}

print_info "Creating production configuration files..."

# Production environment file
cat > "$PROD_DIR/config/.env.production" << EOF
# PyCog-Zero Production Environment Configuration
PROD_ENV=production
PROD_VERSION=$PROD_VERSION

# Server Configuration
HOST=0.0.0.0
PORT=80
INTERNAL_PORT=$PROD_PORT

# Security Configuration
AUTH_LOGIN=admin
# Set AUTH_PASSWORD in your deployment environment
# AUTH_PASSWORD=your_secure_password_here

# API Configuration
API_KEY=\${API_KEY:-generate_secure_api_key}

# Memory and Performance
MEMORY_LIMIT=$MEMORY_LIMIT
CPU_LIMIT=$CPU_LIMIT

# Cognitive Features
OPENCOG_ENABLED=true
COGNITIVE_REASONING=true
PLN_INTEGRATION=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=/app/logs

# Persistence Configuration
DATA_DIR=/app/data
BACKUP_DIR=/app/backups
PERSISTENT_RUNTIME_ID=\${A0_PERSISTENT_RUNTIME_ID:-}

# Docker Configuration
DOCKERIZED=true
CODE_EXEC_DOCKER_ENABLED=false
CODE_EXEC_SSH_ENABLED=true

# Optional: External Services
# DATABASE_URL=
# REDIS_URL=
# ELASTICSEARCH_URL=
EOF

print_status "Production environment configuration created"

# Docker Compose for production
if [ "$COMPOSE_AVAILABLE" = true ]; then
    print_info "Creating Docker Compose configuration..."
    
    cat > "$PROD_DIR/docker-compose.production.yml" << EOF
version: '3.8'

services:
  pycog-zero:
    image: $PROD_IMAGE_NAME:$PROD_VERSION
    container_name: pycog-zero-production
    restart: unless-stopped
    ports:
      - "$PROD_PORT:80"
    environment:
      - PROD_ENV=production
      - DOCKERIZED=true
    env_file:
      - ./config/.env.production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./backups:/app/backups
    deploy:
      resources:
        limits:
          memory: $MEMORY_LIMIT
          cpus: '$CPU_LIMIT'
        reservations:
          memory: 1g
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - pycog-network

  # Optional: Redis for caching and session storage
  redis:
    image: redis:7-alpine
    container_name: pycog-zero-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - ./data/redis:/data
    networks:
      - pycog-network
    profiles:
      - full

  # Optional: PostgreSQL for persistent data
  postgres:
    image: postgres:15-alpine
    container_name: pycog-zero-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: pycog_zero
      POSTGRES_USER: pycog
      POSTGRES_PASSWORD: \${POSTGRES_PASSWORD:-secure_password}
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    networks:
      - pycog-network
    profiles:
      - full

networks:
  pycog-network:
    driver: bridge

volumes:
  data:
  logs:
  backups:
EOF

    print_status "Docker Compose configuration created"
fi

# Create production startup script
print_info "Creating production startup script..."

cat > "$PROD_DIR/start-production.sh" << 'EOF'
#!/bin/bash
"""
PyCog-Zero Production Startup Script
===================================
"""

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

PROD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$PROD_DIR/docker-compose.production.yml"

echo -e "${BLUE}PyCog-Zero Production Startup${NC}"
echo "=============================="

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    print_error "Docker Compose file not found: $COMPOSE_FILE"
    exit 1
fi

# Load environment variables
if [ -f "$PROD_DIR/config/.env.production" ]; then
    set -a
    source "$PROD_DIR/config/.env.production"
    set +a
    print_status "Production environment loaded"
else
    print_warning "Production environment file not found"
fi

# Ensure required environment variables
if [ -z "$AUTH_PASSWORD" ]; then
    print_warning "AUTH_PASSWORD not set. Please set it in your environment or .env.production file"
fi

if [ -z "$API_KEY" ]; then
    print_warning "API_KEY not set. Generating random API key..."
    export API_KEY=$(openssl rand -hex 32)
    echo "Generated API_KEY: $API_KEY"
    echo "Please save this API key securely!"
fi

# Create necessary directories
mkdir -p "$PROD_DIR"/{data,logs,backups}
print_status "Directory structure verified"

# Start services
print_info "Starting PyCog-Zero production services..."

if command -v docker-compose &> /dev/null; then
    docker-compose -f "$COMPOSE_FILE" up -d
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    docker compose -f "$COMPOSE_FILE" up -d
else
    print_error "Docker Compose not available"
    exit 1
fi

print_status "Production services started"

# Wait for health check
print_info "Waiting for services to be healthy..."
sleep 30

# Check service status
if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
    print_status "Services are running"
    
    # Get the actual port
    ACTUAL_PORT=$(docker-compose -f "$COMPOSE_FILE" port pycog-zero 80 2>/dev/null | cut -d: -f2 || echo "$PROD_PORT")
    
    echo ""
    echo -e "${GREEN}🚀 PyCog-Zero Production is running!${NC}"
    echo ""
    echo "Web Interface: http://localhost:$ACTUAL_PORT"
    echo "API Endpoint: http://localhost:$ACTUAL_PORT/api"
    echo ""
    echo "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "To stop services: docker-compose -f $COMPOSE_FILE down"
    echo ""
else
    print_error "Services failed to start properly"
    docker-compose -f "$COMPOSE_FILE" logs
    exit 1
fi
EOF

chmod +x "$PROD_DIR/start-production.sh"
print_status "Production startup script created"

# Create production stop script
print_info "Creating production stop script..."

cat > "$PROD_DIR/stop-production.sh" << 'EOF'
#!/bin/bash
"""
PyCog-Zero Production Stop Script
================================
"""

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

PROD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$PROD_DIR/docker-compose.production.yml"

echo -e "${BLUE}PyCog-Zero Production Stop${NC}"
echo "=========================="

print_info "Stopping PyCog-Zero production services..."

if command -v docker-compose &> /dev/null; then
    docker-compose -f "$COMPOSE_FILE" down
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    docker compose -f "$COMPOSE_FILE" down
fi

print_status "Production services stopped"
EOF

chmod +x "$PROD_DIR/stop-production.sh"
print_status "Production stop script created"

# Create health check endpoint enhancement
print_header "Creating Health Check Configuration"

print_info "Creating health check script..."

cat > "$PROD_DIR/health-check.sh" << 'EOF'
#!/bin/bash
"""
PyCog-Zero Production Health Check
=================================
"""

PROD_PORT="${PROD_PORT:-8080}"
HEALTH_URL="http://localhost:$PROD_PORT/health"

# Check if service is responding
if curl -f -s "$HEALTH_URL" > /dev/null 2>&1; then
    echo "✓ PyCog-Zero is healthy"
    exit 0
else
    echo "✗ PyCog-Zero health check failed"
    exit 1
fi
EOF

chmod +x "$PROD_DIR/health-check.sh"
print_status "Health check script created"

# Create backup script
print_info "Creating backup script..."

cat > "$PROD_DIR/backup-production.sh" << 'EOF'
#!/bin/bash
"""
PyCog-Zero Production Backup Script
==================================
"""

set -e

PROD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="$PROD_DIR/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="pycog-zero-backup-$TIMESTAMP"

echo "Creating backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup configuration
cp -r "$PROD_DIR/config" "$BACKUP_DIR/$BACKUP_NAME/"

# Backup data
if [ -d "$PROD_DIR/data" ]; then
    cp -r "$PROD_DIR/data" "$BACKUP_DIR/$BACKUP_NAME/"
fi

# Backup logs (last 7 days)
if [ -d "$PROD_DIR/logs" ]; then
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME/logs"
    find "$PROD_DIR/logs" -mtime -7 -type f -exec cp {} "$BACKUP_DIR/$BACKUP_NAME/logs/" \;
fi

# Create compressed archive
cd "$BACKUP_DIR"
tar -czf "$BACKUP_NAME.tar.gz" "$BACKUP_NAME"
rm -rf "$BACKUP_NAME"

echo "✓ Backup created: $BACKUP_DIR/$BACKUP_NAME.tar.gz"

# Clean old backups (keep last 10)
ls -t "$BACKUP_DIR"/*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm -f

echo "✓ Old backups cleaned"
EOF

chmod +x "$PROD_DIR/backup-production.sh"
print_status "Backup script created"

# Summary
print_header "Production Deployment Summary"

print_status "Production Docker image built: $PROD_IMAGE_NAME:$PROD_VERSION"
print_status "Production configuration created in: $PROD_DIR"
print_status "Docker Compose configuration ready"
print_status "Startup/stop scripts created"
print_status "Health check and backup scripts ready"

echo ""
print_info "Next steps for production deployment:"
echo "  1. Review and customize: $PROD_DIR/config/.env.production"
echo "  2. Set secure passwords for AUTH_PASSWORD and API_KEY"
echo "  3. Start production: cd $PROD_DIR && ./start-production.sh"
echo "  4. Access via: http://localhost:$PROD_PORT"
echo "  5. Monitor with: cd $PROD_DIR && ./health-check.sh"
echo "  6. Backup data: cd $PROD_DIR && ./backup-production.sh"

echo ""
echo -e "${GREEN}PyCog-Zero Production Docker Deployment completed successfully!${NC}"