#!/bin/bash
#
# PyCog-Zero Master Production Deployment Script
# ==============================================
#
# Master deployment script that orchestrates the complete production deployment
# process using the individual deployment scripts based on build_cpp2py_pipeline.sh
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

# Default configuration
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-docker}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"
AUTO_START="${AUTO_START:-true}"
CONFIGURE_MONITORING="${CONFIGURE_MONITORING:-true}"

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

usage() {
    echo "PyCog-Zero Master Production Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --type TYPE            Deployment type: docker or standalone (default: docker)"
    echo "  --skip-validation      Skip configuration validation"
    echo "  --no-auto-start        Don't automatically start services after deployment"
    echo "  --no-monitoring        Don't configure monitoring"
    echo "  --help                 Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DEPLOYMENT_TYPE        Same as --type"
    echo "  SKIP_VALIDATION        Same as --skip-validation (true/false)"
    echo "  AUTO_START             Same as --no-auto-start inverse (true/false)"
    echo "  CONFIGURE_MONITORING   Same as --no-monitoring inverse (true/false)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Docker deployment with defaults"
    echo "  $0 --type standalone                 # Standalone deployment"
    echo "  $0 --type docker --no-auto-start     # Docker deployment, manual start"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --no-auto-start)
            AUTO_START=false
            shift
            ;;
        --no-monitoring)
            CONFIGURE_MONITORING=false
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate deployment type
if [[ "$DEPLOYMENT_TYPE" != "docker" && "$DEPLOYMENT_TYPE" != "standalone" ]]; then
    print_error "Invalid deployment type: $DEPLOYMENT_TYPE"
    echo "Valid types: docker, standalone"
    exit 1
fi

echo -e "${BLUE}PyCog-Zero Master Production Deployment${NC}"
echo "========================================"
echo "Deployment Type: $DEPLOYMENT_TYPE"
echo "Skip Validation: $SKIP_VALIDATION"
echo "Auto Start: $AUTO_START"
echo "Configure Monitoring: $CONFIGURE_MONITORING"
echo ""

# Step 1: Pre-deployment validation
if [ "$SKIP_VALIDATION" = false ]; then
    print_header "Pre-deployment Validation"
    
    # Check if we're in the right directory
    if [ ! -f "$PROJECT_ROOT/scripts/build_cpp2py_pipeline.sh" ]; then
        print_error "Base build script not found. Are you in the correct directory?"
        exit 1
    fi
    
    # Check required commands
    required_commands=("python3" "git" "curl")
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        required_commands+=("docker")
    fi
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    print_status "Pre-deployment validation passed"
    echo ""
fi

# Step 2: Run the appropriate deployment script
print_header "Running Deployment Script"

if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
    print_info "Executing Docker production deployment..."
    "$SCRIPT_DIR/deploy_production_docker.sh"
    DEPLOYMENT_DIR="$PROJECT_ROOT/production"
elif [ "$DEPLOYMENT_TYPE" = "standalone" ]; then
    print_info "Executing standalone production deployment..."
    if [[ $EUID -eq 0 ]]; then
        "$SCRIPT_DIR/deploy_production_standalone.sh"
        DEPLOYMENT_DIR="/opt/pycog-zero"
    else
        print_info "Running as non-root user - will install to user directory"
        "$SCRIPT_DIR/deploy_production_standalone.sh"
        DEPLOYMENT_DIR="$HOME/pycog-zero-production"
    fi
fi

print_status "Deployment script completed"
echo ""

# Step 3: Initialize and configure production settings
print_header "Configuring Production Environment"

print_info "Initializing production configuration..."
if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
    CONFIG_DIR="$DEPLOYMENT_DIR/config"
else
    CONFIG_DIR="$DEPLOYMENT_DIR/config"
fi

"$SCRIPT_DIR/production_config_manager.sh" init --config-dir "$CONFIG_DIR"

print_info "Generating secure secrets..."
"$SCRIPT_DIR/production_config_manager.sh" generate-secrets --config-dir "$CONFIG_DIR"

print_info "Validating configuration..."
"$SCRIPT_DIR/production_config_manager.sh" validate --config-dir "$CONFIG_DIR"

print_status "Production configuration completed"
echo ""

# Step 4: Configure monitoring (if enabled)
if [ "$CONFIGURE_MONITORING" = true ]; then
    print_header "Setting Up Monitoring"
    
    print_info "Testing monitoring capabilities..."
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        "$SCRIPT_DIR/production_monitor.sh" --prod-dir "$DEPLOYMENT_DIR" status || true
    else
        "$SCRIPT_DIR/production_monitor.sh" --prod-dir "$DEPLOYMENT_DIR" status || true
    fi
    
    print_status "Monitoring setup completed"
    echo ""
fi

# Step 5: Start services (if auto-start enabled)
if [ "$AUTO_START" = true ]; then
    print_header "Starting Production Services"
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        print_info "Starting Docker production services..."
        cd "$DEPLOYMENT_DIR"
        ./start-production.sh
        
        # Wait a moment for services to start
        sleep 10
        
        # Basic health check
        if ./health-check.sh; then
            print_status "Services started successfully"
        else
            print_warning "Services started but health check failed"
        fi
        
    elif [ "$DEPLOYMENT_TYPE" = "standalone" ]; then
        print_info "Starting standalone production services..."
        
        if [[ $EUID -eq 0 ]] || id "pycog" &>/dev/null; then
            "$DEPLOYMENT_DIR/scripts/start.sh"
            
            # Wait a moment for service to start
            sleep 10
            
            # Basic health check
            if "$DEPLOYMENT_DIR/scripts/health-check.sh"; then
                print_status "Service started successfully"
            else
                print_warning "Service started but health check failed"
            fi
        else
            print_info "Manual start required. Run: $DEPLOYMENT_DIR/scripts/start.sh"
        fi
    fi
    
    echo ""
fi

# Step 6: Display deployment summary
print_header "Deployment Summary"

print_status "PyCog-Zero production deployment completed successfully!"
echo ""

print_info "Deployment Details:"
echo "  Type: $DEPLOYMENT_TYPE"
echo "  Location: $DEPLOYMENT_DIR"
echo "  Configuration: $CONFIG_DIR"
echo ""

if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
    print_info "Docker Deployment Commands:"
    echo "  Start services: cd $DEPLOYMENT_DIR && ./start-production.sh"
    echo "  Stop services: cd $DEPLOYMENT_DIR && ./stop-production.sh"
    echo "  Health check: cd $DEPLOYMENT_DIR && ./health-check.sh"
    echo "  Backup: cd $DEPLOYMENT_DIR && ./backup-production.sh"
    echo ""
    
    # Try to get the actual port
    if [ -f "$DEPLOYMENT_DIR/config/.env.production" ]; then
        PORT=$(grep "^INTERNAL_PORT=" "$DEPLOYMENT_DIR/config/.env.production" | cut -d'=' -f2 || echo "8080")
    else
        PORT="8080"
    fi
    
    print_info "Access Points:"
    echo "  Web Interface: http://localhost:$PORT"
    echo "  API Endpoint: http://localhost:$PORT/api"

elif [ "$DEPLOYMENT_TYPE" = "standalone" ]; then
    print_info "Standalone Deployment Commands:"
    echo "  Start service: $DEPLOYMENT_DIR/scripts/start.sh"
    echo "  Stop service: $DEPLOYMENT_DIR/scripts/stop.sh"
    echo "  Health check: $DEPLOYMENT_DIR/scripts/health-check.sh"
    echo "  Backup: $DEPLOYMENT_DIR/scripts/backup.sh"
    echo ""
    
    if systemctl list-unit-files | grep -q pycog-zero; then
        echo "  System service: sudo systemctl start/stop/status pycog-zero"
        echo ""
    fi
    
    # Try to get the actual port
    if [ -f "$CONFIG_DIR/.env.production" ]; then
        PORT=$(grep "^PORT=" "$CONFIG_DIR/.env.production" | cut -d'=' -f2 || echo "8080")
        HOST=$(grep "^HOST=" "$CONFIG_DIR/.env.production" | cut -d'=' -f2 || echo "0.0.0.0")
    else
        PORT="8080"
        HOST="0.0.0.0"
    fi
    
    print_info "Access Points:"
    echo "  Web Interface: http://localhost:$PORT"
    echo "  API Endpoint: http://localhost:$PORT/api"
fi

echo ""
print_info "Monitoring Commands:"
echo "  Status: $SCRIPT_DIR/production_monitor.sh status"
echo "  Health: $SCRIPT_DIR/production_monitor.sh health"
echo "  Metrics: $SCRIPT_DIR/production_monitor.sh metrics"
echo "  Monitor: $SCRIPT_DIR/production_monitor.sh monitor"

echo ""
print_info "Configuration Management:"
echo "  Validate: $SCRIPT_DIR/production_config_manager.sh validate"
echo "  Backup: $SCRIPT_DIR/production_config_manager.sh backup"
echo "  Generate new secrets: $SCRIPT_DIR/production_config_manager.sh generate-secrets"

echo ""
print_warning "IMPORTANT: Review and customize the production configuration!"
echo "  Configuration file: $CONFIG_DIR/.env.production"
echo "  Secrets file: $CONFIG_DIR/.secrets.production"
echo ""

print_status "Production deployment process completed successfully!"