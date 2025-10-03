#!/bin/bash
#
# PyCog-Zero Production Configuration Manager
# ===========================================
#
# Configuration management script for PyCog-Zero production deployments.
# Handles environment configuration, secrets management, and system tuning.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

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

# Default configuration
CONFIG_DIR="${CONFIG_DIR:-/opt/pycog-zero/config}"
PROD_ENV="${PROD_ENV:-production}"

usage() {
    echo "PyCog-Zero Production Configuration Manager"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  init            Initialize production configuration"
    echo "  generate-secrets Generate secure API keys and passwords"
    echo "  validate        Validate configuration files"
    echo "  backup          Backup configuration files"
    echo "  restore         Restore configuration from backup"
    echo "  tune            Apply system performance tuning"
    echo "  monitor-setup   Setup monitoring configuration"
    echo "  ssl-setup       Setup SSL certificates"
    echo ""
    echo "Options:"
    echo "  --config-dir DIR    Configuration directory (default: $CONFIG_DIR)"
    echo "  --env ENV           Environment name (default: $PROD_ENV)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 init --config-dir /opt/pycog-zero/config"
    echo "  $0 generate-secrets"
    echo "  $0 validate"
    echo "  $0 tune --env production"
    echo ""
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        init|generate-secrets|validate|backup|restore|tune|monitor-setup|ssl-setup)
            COMMAND="$1"
            shift
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --env)
            PROD_ENV="$2"
            shift 2
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

if [ -z "$COMMAND" ]; then
    usage
    exit 1
fi

echo -e "${BLUE}PyCog-Zero Production Configuration Manager${NC}"
echo "============================================="
echo "Command: $COMMAND"
echo "Config Directory: $CONFIG_DIR"
echo "Environment: $PROD_ENV"
echo ""

# Ensure config directory exists
mkdir -p "$CONFIG_DIR"

init_config() {
    print_header "Initializing Production Configuration"
    
    # Create base configuration structure
    mkdir -p "$CONFIG_DIR"/{env,ssl,monitoring,backup}
    
    # Create main environment file if it doesn't exist
    if [ ! -f "$CONFIG_DIR/.env.$PROD_ENV" ]; then
        print_info "Creating base environment configuration..."
        
        cat > "$CONFIG_DIR/.env.$PROD_ENV" << EOF
# PyCog-Zero Production Environment Configuration
# Generated on $(date)

# Environment
PROD_ENV=$PROD_ENV
DEPLOYMENT_TYPE=production

# Server Configuration
HOST=0.0.0.0
PORT=8080

# Security Configuration (CHANGE THESE!)
AUTH_LOGIN=admin
AUTH_PASSWORD=CHANGE_ME_SECURE_PASSWORD
API_KEY=CHANGE_ME_SECURE_API_KEY

# Session Configuration
SESSION_SECRET_KEY=CHANGE_ME_SESSION_SECRET
JWT_SECRET_KEY=CHANGE_ME_JWT_SECRET

# Cognitive Features
OPENCOG_ENABLED=true
COGNITIVE_REASONING=true
PLN_INTEGRATION=true
ATTENTION_ALLOCATION=true

# Performance Configuration
MAX_WORKERS=4
WORKER_TIMEOUT=30
MAX_REQUESTS_PER_WORKER=1000

# Logging Configuration
LOG_LEVEL=INFO
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=10
STRUCTURED_LOGGING=true

# Database Configuration (Optional)
# DATABASE_URL=postgresql://user:password@localhost/pycog_zero
# REDIS_URL=redis://localhost:6379/0

# External Services (Optional)
# ELASTICSEARCH_URL=http://localhost:9200
# PROMETHEUS_URL=http://localhost:9090

# Monitoring and Health
HEALTH_CHECK_INTERVAL=30
METRICS_ENABLED=true
PERFORMANCE_MONITORING=true

# Security Headers
SECURE_HEADERS=true
CSRF_PROTECTION=true
RATE_LIMITING=true

# File Upload Limits
MAX_CONTENT_LENGTH=16MB
UPLOAD_FOLDER=/opt/pycog-zero/data/uploads

# Cache Configuration
CACHE_TYPE=simple
CACHE_DEFAULT_TIMEOUT=300

# Development flags (should be false in production)
DEBUG=false
TESTING=false
DEVELOPMENT=false
EOF
        print_status "Base configuration created"
    else
        print_warning "Configuration file already exists: $CONFIG_DIR/.env.$PROD_ENV"
    fi
    
    # Create logging configuration
    cat > "$CONFIG_DIR/logging.yaml" << EOF
version: 1
disable_existing_loggers: false

formatters:
  default:
    format: '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
  detailed:
    format: '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: /opt/pycog-zero/logs/app.log
    maxBytes: 104857600  # 100MB
    backupCount: 10
    
  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: /opt/pycog-zero/logs/error.log
    maxBytes: 104857600  # 100MB
    backupCount: 10

  json_file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /opt/pycog-zero/logs/app.json
    maxBytes: 104857600  # 100MB
    backupCount: 10

loggers:
  werkzeug:
    level: WARNING
  
  urllib3:
    level: WARNING
    
  requests:
    level: WARNING

root:
  level: INFO
  handlers: [console, file, error_file, json_file]
EOF
    
    print_status "Logging configuration created"
    
    # Create security configuration
    cat > "$CONFIG_DIR/security.yaml" << EOF
# PyCog-Zero Production Security Configuration

# Rate Limiting
rate_limiting:
  enabled: true
  default_limits:
    - "100 per day"
    - "20 per hour"
  api_limits:
    - "1000 per day"
    - "100 per hour"
  per_method: true

# CORS Configuration
cors:
  enabled: true
  origins:
    - "https://yourdomain.com"
    - "https://api.yourdomain.com"
  methods: ["GET", "POST", "PUT", "DELETE"]
  allow_headers: ["Content-Type", "Authorization", "X-API-KEY"]

# Security Headers
security_headers:
  strict_transport_security:
    enabled: true
    max_age: 31536000
    include_subdomains: true
  content_security_policy:
    enabled: true
    policy: "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
  x_frame_options: "DENY"
  x_content_type_options: "nosniff"
  x_xss_protection: "1; mode=block"

# Authentication
authentication:
  session_timeout: 3600  # 1 hour
  max_login_attempts: 5
  lockout_duration: 900  # 15 minutes
  password_policy:
    min_length: 12
    require_uppercase: true
    require_lowercase: true
    require_numbers: true
    require_special: true

# API Security
api_security:
  require_api_key: true
  api_key_header: "X-API-KEY"
  request_signing: false
  token_expiry: 86400  # 24 hours
EOF
    
    print_status "Security configuration created"
    
    # Create monitoring configuration template
    cat > "$CONFIG_DIR/monitoring/prometheus.yml" << EOF
# Prometheus configuration for PyCog-Zero monitoring

global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'pycog-zero'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'system'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - localhost:9093
EOF
    
    print_status "Monitoring configuration created"
    
    print_status "Production configuration initialization complete"
}

generate_secrets() {
    print_header "Generating Secure Secrets"
    
    ENV_FILE="$CONFIG_DIR/.env.$PROD_ENV"
    
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file not found: $ENV_FILE"
        print_info "Run '$0 init' first to create base configuration"
        exit 1
    fi
    
    # Generate secure random values
    AUTH_PASSWORD=$(openssl rand -base64 32 | tr -d /=+ | cut -c -16)
    API_KEY=$(openssl rand -hex 32)
    SESSION_SECRET=$(openssl rand -hex 32)
    JWT_SECRET=$(openssl rand -hex 32)
    
    print_info "Generated new secure secrets"
    
    # Update environment file
    sed -i "s/AUTH_PASSWORD=.*/AUTH_PASSWORD=$AUTH_PASSWORD/" "$ENV_FILE"
    sed -i "s/API_KEY=.*/API_KEY=$API_KEY/" "$ENV_FILE"
    sed -i "s/SESSION_SECRET_KEY=.*/SESSION_SECRET_KEY=$SESSION_SECRET/" "$ENV_FILE"
    sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" "$ENV_FILE"
    
    print_status "Secrets updated in environment file"
    
    # Create secure secrets file
    SECRETS_FILE="$CONFIG_DIR/.secrets.$PROD_ENV"
    cat > "$SECRETS_FILE" << EOF
# PyCog-Zero Production Secrets
# Generated on $(date)
# Keep this file secure and do not commit to version control

AUTH_PASSWORD=$AUTH_PASSWORD
API_KEY=$API_KEY
SESSION_SECRET_KEY=$SESSION_SECRET
JWT_SECRET_KEY=$JWT_SECRET
EOF
    
    chmod 600 "$SECRETS_FILE"
    print_status "Secrets file created: $SECRETS_FILE"
    
    echo ""
    print_info "IMPORTANT: Save these credentials securely!"
    echo -e "${YELLOW}Admin Password: $AUTH_PASSWORD${NC}"
    echo -e "${YELLOW}API Key: $API_KEY${NC}"
    echo ""
}

validate_config() {
    print_header "Validating Configuration"
    
    ENV_FILE="$CONFIG_DIR/.env.$PROD_ENV"
    
    if [ ! -f "$ENV_FILE" ]; then
        print_error "Environment file not found: $ENV_FILE"
        exit 1
    fi
    
    # Load environment file
    set -a
    source "$ENV_FILE"
    set +a
    
    # Validation checks
    validation_errors=0
    
    # Check for default/insecure values
    if [[ "$AUTH_PASSWORD" == "CHANGE_ME_SECURE_PASSWORD" ]]; then
        print_error "AUTH_PASSWORD still has default value"
        ((validation_errors++))
    fi
    
    if [[ "$API_KEY" == "CHANGE_ME_SECURE_API_KEY" ]]; then
        print_error "API_KEY still has default value"
        ((validation_errors++))
    fi
    
    if [[ "$SESSION_SECRET_KEY" == "CHANGE_ME_SESSION_SECRET" ]]; then
        print_error "SESSION_SECRET_KEY still has default value"
        ((validation_errors++))
    fi
    
    # Check password strength
    if [[ ${#AUTH_PASSWORD} -lt 12 ]]; then
        print_error "AUTH_PASSWORD is too short (minimum 12 characters)"
        ((validation_errors++))
    fi
    
    # Check API key strength
    if [[ ${#API_KEY} -lt 32 ]]; then
        print_error "API_KEY is too short (minimum 32 characters)"
        ((validation_errors++))
    fi
    
    # Check production flags
    if [[ "$DEBUG" == "true" ]]; then
        print_warning "DEBUG is enabled in production"
    fi
    
    if [[ "$DEVELOPMENT" == "true" ]]; then
        print_warning "DEVELOPMENT mode is enabled in production"
    fi
    
    # Check file permissions
    if [[ -f "$CONFIG_DIR/.secrets.$PROD_ENV" ]]; then
        perms=$(stat -c %a "$CONFIG_DIR/.secrets.$PROD_ENV")
        if [[ "$perms" != "600" ]]; then
            print_warning "Secrets file has incorrect permissions: $perms (should be 600)"
        else
            print_status "Secrets file permissions are correct"
        fi
    fi
    
    if [[ $validation_errors -eq 0 ]]; then
        print_status "Configuration validation passed"
    else
        print_error "Configuration validation failed with $validation_errors errors"
        exit 1
    fi
}

backup_config() {
    print_header "Backing Up Configuration"
    
    BACKUP_DIR="$CONFIG_DIR/backup"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_NAME="config-backup-$TIMESTAMP"
    
    mkdir -p "$BACKUP_DIR"
    
    # Create backup archive
    tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" -C "$(dirname "$CONFIG_DIR")" "$(basename "$CONFIG_DIR")"
    
    print_status "Configuration backed up to: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
    
    # Keep only last 10 backups
    ls -t "$BACKUP_DIR"/config-backup-*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm -f
    print_status "Old backups cleaned up"
}

tune_system() {
    print_header "Applying System Performance Tuning"
    
    print_info "Applying kernel parameters for production..."
    
    # Create sysctl configuration for PyCog-Zero
    cat > "/etc/sysctl.d/99-pycog-zero.conf" << EOF
# PyCog-Zero Production System Tuning

# Network tuning
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1200
net.ipv4.tcp_max_tw_buckets = 1440000

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File descriptor limits
fs.file-max = 2097152

# Security
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2
EOF
    
    # Apply sysctl changes
    sysctl -p /etc/sysctl.d/99-pycog-zero.conf
    print_status "Kernel parameters applied"
    
    # Set up limits for production user
    if id "pycog" &>/dev/null; then
        cat > "/etc/security/limits.d/pycog.conf" << EOF
# PyCog-Zero Production Limits
pycog soft nofile 65535
pycog hard nofile 65535
pycog soft nproc 32768
pycog hard nproc 32768
EOF
        print_status "User limits configured"
    fi
    
    print_status "System tuning complete"
}

# Execute command
case $COMMAND in
    init)
        init_config
        ;;
    generate-secrets)
        generate_secrets
        ;;
    validate)
        validate_config
        ;;
    backup)
        backup_config
        ;;
    tune)
        tune_system
        ;;
    *)
        print_error "Command not implemented: $COMMAND"
        exit 1
        ;;
esac

echo ""
print_status "Configuration management task completed successfully!"