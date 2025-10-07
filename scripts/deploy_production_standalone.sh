#!/bin/bash
#
# PyCog-Zero Standalone Production Deployment Script
# ==================================================
#
# Production deployment script for non-containerized PyCog-Zero cognitive agent systems.
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
PROD_ENV="${PROD_ENV:-production}"
PROD_PORT="${PROD_PORT:-8080}"
PROD_HOST="${PROD_HOST:-0.0.0.0}"
PROD_USER="${PROD_USER:-pycog}"
PROD_DIR="${PROD_DIR:-/opt/pycog-zero}"
SERVICE_NAME="${SERVICE_NAME:-pycog-zero}"

echo -e "${BLUE}PyCog-Zero Standalone Production Deployment${NC}"
echo "==========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Production Directory: $PROD_DIR" 
echo "Production Port: $PROD_PORT"
echo "Production Host: $PROD_HOST"
echo "Environment: $PROD_ENV"
echo "Service Name: $SERVICE_NAME"
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

# Check if running as root for system-wide installation
if [[ $EUID -eq 0 ]]; then
    SYSTEM_INSTALL=true
    print_info "Running as root - system-wide installation"
else
    SYSTEM_INSTALL=false
    print_info "Running as user - user installation"
    PROD_DIR="$HOME/pycog-zero-production"
fi

# Check prerequisites
print_header "Checking Prerequisites"

# Check Python version
echo "Checking Python environment..."
python_version=$(python3 --version 2>&1)
print_info "Python version: $python_version"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_status "Python 3.8+ detected"
else
    print_error "Python 3.8+ required"
    exit 1
fi

# Check system dependencies
print_info "Checking system dependencies..."

required_packages=("curl" "wget" "git" "supervisor")
missing_packages=()

for package in "${required_packages[@]}"; do
    if ! command -v "$package" &> /dev/null; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    print_warning "Missing packages: ${missing_packages[*]}"
    print_info "Installing missing packages..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y "${missing_packages[@]}"
    elif command -v yum &> /dev/null; then
        sudo yum install -y "${missing_packages[@]}"
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y "${missing_packages[@]}"
    else
        print_error "Package manager not found. Please install: ${missing_packages[*]}"
        exit 1
    fi
    
    print_status "System dependencies installed"
else
    print_status "All system dependencies available"
fi

# Create production user (if system install and user doesn't exist)
if [ "$SYSTEM_INSTALL" = true ] && [ "$PROD_USER" != "root" ]; then
    if ! id "$PROD_USER" &>/dev/null; then
        print_info "Creating production user: $PROD_USER"
        useradd -r -s /bin/bash -d "$PROD_DIR" -m "$PROD_USER"
        print_status "Production user created"
    else
        print_status "Production user already exists"
    fi
fi

# Setup production directory
print_header "Setting Up Production Environment"

print_info "Creating production directory structure..."

# Create main directories
sudo mkdir -p "$PROD_DIR"/{app,config,data,logs,backups,scripts}

# Set ownership
if [ "$SYSTEM_INSTALL" = true ] && [ "$PROD_USER" != "root" ]; then
    sudo chown -R "$PROD_USER:$PROD_USER" "$PROD_DIR"
fi

print_status "Production directory structure created"

# Copy application files
print_info "Copying application files..."

# Copy core application
sudo cp -r "$PROJECT_ROOT"/* "$PROD_DIR/app/" 2>/dev/null || {
    print_info "Copying with elevated permissions..."
    sudo cp -r "$PROJECT_ROOT"/* "$PROD_DIR/app/"
}

# Remove development files
sudo rm -rf "$PROD_DIR/app"/{.git,.gitignore,.vscode,.dockerignore,docker,tests,tmp}
sudo rm -rf "$PROD_DIR/app"/test_*.py
sudo rm -rf "$PROD_DIR/app"/demo_*.py

print_status "Application files copied"

# Create Python virtual environment
print_info "Creating Python virtual environment..."

VENV_PATH="$PROD_DIR/venv"

if [ "$SYSTEM_INSTALL" = true ] && [ "$PROD_USER" != "root" ]; then
    sudo -u "$PROD_USER" python3 -m venv "$VENV_PATH"
else
    python3 -m venv "$VENV_PATH"
fi

print_status "Virtual environment created"

# Install dependencies
print_info "Installing dependencies..."

VENV_PYTHON="$VENV_PATH/bin/python3"
VENV_PIP="$VENV_PATH/bin/pip"

# Upgrade pip
if [ "$SYSTEM_INSTALL" = true ] && [ "$PROD_USER" != "root" ]; then
    sudo -u "$PROD_USER" "$VENV_PIP" install --upgrade pip
else
    "$VENV_PIP" install --upgrade pip
fi

# Install base requirements
if [ -f "$PROD_DIR/app/requirements.txt" ]; then
    print_info "Installing base requirements..."
    if [ "$SYSTEM_INSTALL" = true ] && [ "$PROD_USER" != "root" ]; then
        sudo -u "$PROD_USER" "$VENV_PIP" install -r "$PROD_DIR/app/requirements.txt" || {
            print_warning "Some base requirements may have failed to install"
        }
    else
        "$VENV_PIP" install -r "$PROD_DIR/app/requirements.txt" || {
            print_warning "Some base requirements may have failed to install"
        }
    fi
    print_status "Base requirements installed"
fi

# Install cognitive requirements
if [ -f "$PROD_DIR/app/requirements-cognitive.txt" ]; then
    print_info "Installing cognitive requirements..."
    if [ "$SYSTEM_INSTALL" = true ] && [ "$PROD_USER" != "root" ]; then
        sudo -u "$PROD_USER" "$VENV_PIP" install -r "$PROD_DIR/app/requirements-cognitive.txt" || {
            print_warning "Some cognitive requirements may have failed to install"
        }
    else
        "$VENV_PIP" install -r "$PROD_DIR/app/requirements-cognitive.txt" || {
            print_warning "Some cognitive requirements may have failed to install"
        }
    fi
    print_status "Cognitive requirements processed"
fi

# Install production dependencies
print_info "Installing production dependencies..."
if [ "$SYSTEM_INSTALL" = true ] && [ "$PROD_USER" != "root" ]; then
    sudo -u "$PROD_USER" "$VENV_PIP" install gunicorn supervisor
else
    "$VENV_PIP" install gunicorn supervisor
fi
print_status "Production dependencies installed"

# Create production configuration
print_header "Creating Production Configuration"

print_info "Creating production configuration files..."

# Production environment file
cat > "$PROD_DIR/config/.env.production" << EOF
# PyCog-Zero Production Environment Configuration
PROD_ENV=production
PROD_DIR=$PROD_DIR

# Server Configuration
HOST=$PROD_HOST
PORT=$PROD_PORT

# Security Configuration
AUTH_LOGIN=admin
# Set AUTH_PASSWORD in your deployment environment
# AUTH_PASSWORD=your_secure_password_here

# API Configuration
API_KEY=\${API_KEY:-generate_secure_api_key}

# Cognitive Features
OPENCOG_ENABLED=true
COGNITIVE_REASONING=true
PLN_INTEGRATION=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=$PROD_DIR/logs

# Persistence Configuration
DATA_DIR=$PROD_DIR/data
BACKUP_DIR=$PROD_DIR/backups
PERSISTENT_RUNTIME_ID=\${A0_PERSISTENT_RUNTIME_ID:-}

# Virtual Environment
VENV_PATH=$VENV_PATH

# Optional: External Services
# DATABASE_URL=
# REDIS_URL=
# ELASTICSEARCH_URL=
EOF

print_status "Production environment configuration created"

# Create Gunicorn configuration
cat > "$PROD_DIR/config/gunicorn.conf.py" << EOF
# PyCog-Zero Gunicorn Configuration
import os

# Server socket
bind = "$PROD_HOST:$PROD_PORT"
backlog = 2048

# Worker processes
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "$PROD_DIR/logs/access.log"
errorlog = "$PROD_DIR/logs/error.log"
loglevel = "info"

# Process naming
proc_name = "pycog-zero"

# Server mechanics
daemon = False
pidfile = "$PROD_DIR/pycog-zero.pid"
user = "$PROD_USER" if "$PROD_USER" != "root" else None
group = "$PROD_USER" if "$PROD_USER" != "root" else None
tmp_upload_dir = None

# SSL (uncomment and configure for HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"
EOF

print_status "Gunicorn configuration created"

# Create systemd service (if system install)
if [ "$SYSTEM_INSTALL" = true ]; then
    print_info "Creating systemd service..."
    
    cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=PyCog-Zero Cognitive Agent System
After=network.target

[Service]
Type=forking
User=$PROD_USER
Group=$PROD_USER
WorkingDirectory=$PROD_DIR/app
Environment=PATH=$VENV_PATH/bin
EnvironmentFile=$PROD_DIR/config/.env.production
ExecStart=$VENV_PATH/bin/gunicorn --config $PROD_DIR/config/gunicorn.conf.py run_ui:webapp
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
    print_status "Systemd service created"
fi

# Create startup script
print_info "Creating startup script..."

cat > "$PROD_DIR/scripts/start.sh" << EOF
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

print_status() { echo -e "\${GREEN}✓\${NC} \$1"; }
print_warning() { echo -e "\${YELLOW}⚠\${NC} \$1"; }
print_error() { echo -e "\${RED}✗\${NC} \$1"; }
print_info() { echo -e "\${BLUE}ℹ\${NC} \$1"; }

PROD_DIR="$PROD_DIR"
VENV_PATH="$VENV_PATH"

echo -e "\${BLUE}PyCog-Zero Production Startup\${NC}"
echo "=============================="

# Load environment variables
if [ -f "\$PROD_DIR/config/.env.production" ]; then
    set -a
    source "\$PROD_DIR/config/.env.production"
    set +a
    print_status "Production environment loaded"
else
    print_warning "Production environment file not found"
fi

# Activate virtual environment
source "\$VENV_PATH/bin/activate"
print_status "Virtual environment activated"

# Change to application directory
cd "\$PROD_DIR/app"

# Start the application
print_info "Starting PyCog-Zero production server..."

if command -v systemctl &> /dev/null && [ -f "/etc/systemd/system/$SERVICE_NAME.service" ]; then
    # Start via systemd
    sudo systemctl start $SERVICE_NAME
    sudo systemctl enable $SERVICE_NAME
    print_status "Service started via systemd"
    
    echo ""
    echo -e "\${GREEN}🚀 PyCog-Zero Production is running!\${NC}"
    echo ""
    echo "Web Interface: http://$PROD_HOST:$PROD_PORT"
    echo "API Endpoint: http://$PROD_HOST:$PROD_PORT/api"
    echo ""
    echo "To view logs: sudo journalctl -u $SERVICE_NAME -f"
    echo "To stop service: sudo systemctl stop $SERVICE_NAME"
    echo ""
else
    # Start directly with Gunicorn
    exec gunicorn --config "\$PROD_DIR/config/gunicorn.conf.py" run_ui:webapp
fi
EOF

chmod +x "$PROD_DIR/scripts/start.sh"
print_status "Startup script created"

# Create stop script
cat > "$PROD_DIR/scripts/stop.sh" << EOF
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

print_status() { echo -e "\${GREEN}✓\${NC} \$1"; }
print_info() { echo -e "\${BLUE}ℹ\${NC} \$1"; }

echo -e "\${BLUE}PyCog-Zero Production Stop\${NC}"
echo "=========================="

if command -v systemctl &> /dev/null && [ -f "/etc/systemd/system/$SERVICE_NAME.service" ]; then
    # Stop via systemd
    sudo systemctl stop $SERVICE_NAME
    print_status "Service stopped via systemd"
else
    # Stop via PID file
    if [ -f "$PROD_DIR/pycog-zero.pid" ]; then
        kill \$(cat "$PROD_DIR/pycog-zero.pid")
        rm -f "$PROD_DIR/pycog-zero.pid"
        print_status "Service stopped via PID file"
    else
        print_info "No PID file found, service may not be running"
    fi
fi
EOF

chmod +x "$PROD_DIR/scripts/stop.sh"
print_status "Stop script created"

# Create health check script
cat > "$PROD_DIR/scripts/health-check.sh" << EOF
#!/bin/bash
"""
PyCog-Zero Production Health Check
=================================
"""

HEALTH_URL="http://$PROD_HOST:$PROD_PORT/health"

# Check if service is responding
if curl -f -s "\$HEALTH_URL" > /dev/null 2>&1; then
    echo "✓ PyCog-Zero is healthy"
    exit 0
else
    echo "✗ PyCog-Zero health check failed"
    exit 1
fi
EOF

chmod +x "$PROD_DIR/scripts/health-check.sh"
print_status "Health check script created"

# Create backup script
cat > "$PROD_DIR/scripts/backup.sh" << EOF
#!/bin/bash
"""
PyCog-Zero Production Backup Script
==================================
"""

set -e

PROD_DIR="$PROD_DIR"
BACKUP_DIR="\$PROD_DIR/backups"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="pycog-zero-backup-\$TIMESTAMP"

echo "Creating backup: \$BACKUP_NAME"

# Create backup directory
mkdir -p "\$BACKUP_DIR/\$BACKUP_NAME"

# Backup configuration
cp -r "\$PROD_DIR/config" "\$BACKUP_DIR/\$BACKUP_NAME/"

# Backup data
if [ -d "\$PROD_DIR/data" ]; then
    cp -r "\$PROD_DIR/data" "\$BACKUP_DIR/\$BACKUP_NAME/"
fi

# Backup logs (last 7 days)
if [ -d "\$PROD_DIR/logs" ]; then
    mkdir -p "\$BACKUP_DIR/\$BACKUP_NAME/logs"
    find "\$PROD_DIR/logs" -mtime -7 -type f -exec cp {} "\$BACKUP_DIR/\$BACKUP_NAME/logs/" \\;
fi

# Create compressed archive
cd "\$BACKUP_DIR"
tar -czf "\$BACKUP_NAME.tar.gz" "\$BACKUP_NAME"
rm -rf "\$BACKUP_NAME"

echo "✓ Backup created: \$BACKUP_DIR/\$BACKUP_NAME.tar.gz"

# Clean old backups (keep last 10)
ls -t "\$BACKUP_DIR"/*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm -f

echo "✓ Old backups cleaned"
EOF

chmod +x "$PROD_DIR/scripts/backup.sh"
print_status "Backup script created"

# Set final permissions
if [ "$SYSTEM_INSTALL" = true ] && [ "$PROD_USER" != "root" ]; then
    chown -R "$PROD_USER:$PROD_USER" "$PROD_DIR"
fi

# Summary
print_header "Production Deployment Summary"

print_status "Production environment created in: $PROD_DIR"
print_status "Virtual environment configured: $VENV_PATH"
print_status "Dependencies installed"
print_status "Configuration files created"
print_status "Startup/stop scripts created"
print_status "Health check and backup scripts ready"

if [ "$SYSTEM_INSTALL" = true ]; then
    print_status "Systemd service configured: $SERVICE_NAME"
fi

echo ""
print_info "Next steps for production deployment:"
echo "  1. Review and customize: $PROD_DIR/config/.env.production"
echo "  2. Set secure passwords for AUTH_PASSWORD and API_KEY"
echo "  3. Start production: $PROD_DIR/scripts/start.sh"
echo "  4. Access via: http://$PROD_HOST:$PROD_PORT"
echo "  5. Monitor with: $PROD_DIR/scripts/health-check.sh"
echo "  6. Backup data: $PROD_DIR/scripts/backup.sh"

if [ "$SYSTEM_INSTALL" = true ]; then
    echo "  7. Enable auto-start: sudo systemctl enable $SERVICE_NAME"
fi

echo ""
echo -e "${GREEN}PyCog-Zero Standalone Production Deployment completed successfully!${NC}"