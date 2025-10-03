#!/bin/bash

set -e

# PyCog-Zero Production Deployment Script
# This script deploys PyCog-Zero to production environment with full cognitive capabilities

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="${PROJECT_ROOT}/deploy/production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if running as root for production deployment
    if [[ $EUID -eq 0 ]]; then
        warning "Running as root. This is acceptable for production deployment."
    fi
    
    success "Prerequisites check passed"
}

# Validate configuration
validate_configuration() {
    log "Validating production configuration..."
    
    if [[ ! -f "${DEPLOY_DIR}/production-config/config_production.json" ]]; then
        error "Production configuration file not found"
        exit 1
    fi
    
    if [[ ! -f "${DEPLOY_DIR}/docker-compose-production.yml" ]]; then
        error "Production Docker Compose file not found"
        exit 1
    fi
    
    # Validate JSON configuration
    if ! python3 -m json.tool "${DEPLOY_DIR}/production-config/config_production.json" > /dev/null; then
        error "Invalid JSON in production configuration"
        exit 1
    fi
    
    success "Configuration validation passed"
}

# Pre-deployment backup
backup_existing() {
    log "Creating backup of existing deployment..."
    
    BACKUP_DIR="/var/backups/pycog-zero/$(date +%Y%m%d_%H%M%S)"
    
    if docker ps -q --filter "name=pycog-zero-production" | grep -q .; then
        mkdir -p "${BACKUP_DIR}"
        
        # Backup volumes
        docker run --rm -v pycog-zero-data:/data -v "${BACKUP_DIR}:/backup" alpine \
            sh -c "cd /data && tar czf /backup/pycog-zero-data.tar.gz ."
        
        docker run --rm -v pycog-zero-config:/data -v "${BACKUP_DIR}:/backup" alpine \
            sh -c "cd /data && tar czf /backup/pycog-zero-config.tar.gz ."
        
        success "Backup created at ${BACKUP_DIR}"
    else
        log "No existing deployment found, skipping backup"
    fi
}

# Build production image
build_production_image() {
    log "Building production Docker image..."
    
    cd "${PROJECT_ROOT}"
    
    # Build with production optimizations
    docker build \
        -f DockerfileLocal \
        --build-arg BRANCH=production \
        --build-arg CACHE_DATE="$(date +%Y-%m-%d:%H:%M:%S)" \
        -t pycog-zero:production \
        .
    
    success "Production image built successfully"
}

# Deploy services
deploy_services() {
    log "Deploying production services..."
    
    cd "${DEPLOY_DIR}"
    
    # Stop existing services
    if docker-compose -f docker-compose-production.yml ps -q | grep -q .; then
        log "Stopping existing services..."
        docker-compose -f docker-compose-production.yml down
    fi
    
    # Start services
    docker-compose -f docker-compose-production.yml up -d
    
    success "Services deployed successfully"
}

# Health check
health_check() {
    log "Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log "Health check attempt ${attempt}/${max_attempts}..."
        
        if curl -f -s http://localhost:80/health > /dev/null; then
            success "PyCog-Zero is healthy and responding"
            return 0
        fi
        
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed after ${max_attempts} attempts"
    return 1
}

# Validate cognitive features
validate_cognitive_features() {
    log "Validating cognitive features..."
    
    # Test cognitive endpoints
    local cognitive_tests=(
        "http://localhost:80/cognitive/metrics"
        "http://localhost:80/api/cognitive/status"
    )
    
    for endpoint in "${cognitive_tests[@]}"; do
        if curl -f -s "${endpoint}" > /dev/null; then
            success "Cognitive endpoint ${endpoint} is responding"
        else
            warning "Cognitive endpoint ${endpoint} is not responding (this may be expected)"
        fi
    done
}

# Main deployment function
main() {
    log "Starting PyCog-Zero production deployment..."
    
    check_prerequisites
    validate_configuration
    backup_existing
    build_production_image
    deploy_services
    
    log "Waiting for services to start..."
    sleep 30
    
    if health_check; then
        validate_cognitive_features
        
        success "PyCog-Zero production deployment completed successfully!"
        log "Services available at:"
        log "  - Web Interface: http://localhost:80"
        log "  - Alternative Port: http://localhost:50001"
        log "  - Monitoring: http://localhost:9090"
        log "  - Health Check: http://localhost:80/health"
        
        log "To view logs: docker-compose -f ${DEPLOY_DIR}/docker-compose-production.yml logs"
        log "To stop services: docker-compose -f ${DEPLOY_DIR}/docker-compose-production.yml down"
    else
        error "Deployment completed but health checks failed"
        log "Check logs: docker-compose -f ${DEPLOY_DIR}/docker-compose-production.yml logs"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    "--help"|"-h")
        echo "PyCog-Zero Production Deployment Script"
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --validate     Only validate configuration"
        echo "  --backup       Only create backup"
        echo "  --health       Only perform health check"
        exit 0
        ;;
    "--validate")
        check_prerequisites
        validate_configuration
        success "Validation completed successfully"
        exit 0
        ;;
    "--backup")
        backup_existing
        exit 0
        ;;
    "--health")
        health_check
        exit 0
        ;;
    "")
        main
        ;;
    *)
        error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac