# PyCog-Zero Production Deployment Guide

This guide covers production deployment of PyCog-Zero cognitive agent systems using the production deployment scripts based on `scripts/build_cpp2py_pipeline.sh`.

## Overview

The production deployment system provides:
- **Containerized deployments** with Docker and Docker Compose
- **Standalone deployments** for non-containerized environments
- **Configuration management** with security and performance tuning
- **Monitoring and health checking** with alerting capabilities
- **Backup and persistence** management

## Quick Start

### Docker Production Deployment (Recommended)

```bash
# 1. Run the production Docker deployment script
./scripts/deploy_production_docker.sh

# 2. Configure production settings
cd production
nano config/.env.production  # Set secure passwords and API keys

# 3. Start production services
./start-production.sh

# 4. Access the system
open http://localhost:8080
```

### Standalone Production Deployment

```bash
# 1. Run the standalone deployment script (requires sudo for system install)
sudo ./scripts/deploy_production_standalone.sh

# 2. Configure production settings
sudo nano /opt/pycog-zero/config/.env.production

# 3. Start production services
sudo /opt/pycog-zero/scripts/start.sh
```

## Deployment Scripts

### 1. Docker Production Deployment (`deploy_production_docker.sh`)

**Purpose**: Creates a production-ready Docker deployment with all necessary services.

**Features**:
- Builds production Docker image based on `DockerfileLocal`
- Creates Docker Compose configuration with Redis and PostgreSQL support
- Sets up production directory structure with proper configurations
- Includes health checks, monitoring, and backup scripts
- Configures resource limits and security settings

**Usage**:
```bash
# Basic deployment
./scripts/deploy_production_docker.sh

# Custom configuration
PROD_PORT=9000 MEMORY_LIMIT=8g ./scripts/deploy_production_docker.sh
```

**Environment Variables**:
- `PROD_IMAGE_NAME`: Production image name (default: `pycog-zero-production`)
- `PROD_VERSION`: Image version tag (default: `latest`)
- `PROD_PORT`: External port mapping (default: `8080`)
- `PROD_ENV`: Environment name (default: `production`)
- `MEMORY_LIMIT`: Container memory limit (default: `4g`)
- `CPU_LIMIT`: Container CPU limit (default: `2`)

### 2. Standalone Production Deployment (`deploy_production_standalone.sh`)

**Purpose**: Creates a non-containerized production deployment with systemd service integration.

**Features**:
- Creates dedicated production user and directory structure
- Sets up Python virtual environment with all dependencies
- Configures Gunicorn for production WSGI serving
- Creates systemd service for automatic startup
- Includes monitoring, backup, and health check scripts

**Usage**:
```bash
# System-wide installation (recommended)
sudo ./scripts/deploy_production_standalone.sh

# User installation
PROD_DIR=$HOME/pycog-zero-production ./scripts/deploy_production_standalone.sh
```

**Configuration Options**:
- `PROD_DIR`: Production installation directory (default: `/opt/pycog-zero`)
- `PROD_USER`: Production user account (default: `pycog`)
- `PROD_PORT`: Application port (default: `8080`)
- `PROD_HOST`: Bind address (default: `0.0.0.0`)
- `SERVICE_NAME`: Systemd service name (default: `pycog-zero`)

### 3. Configuration Manager (`production_config_manager.sh`)

**Purpose**: Manages production configuration, secrets, and system tuning.

**Commands**:
```bash
# Initialize production configuration
./scripts/production_config_manager.sh init

# Generate secure secrets
./scripts/production_config_manager.sh generate-secrets

# Validate configuration
./scripts/production_config_manager.sh validate

# Backup configuration
./scripts/production_config_manager.sh backup

# Apply system performance tuning
sudo ./scripts/production_config_manager.sh tune
```

### 4. Production Monitor (`production_monitor.sh`)

**Purpose**: Comprehensive monitoring, health checking, and alerting for production deployments.

**Commands**:
```bash
# Show current status
./scripts/production_monitor.sh status

# Perform health check
./scripts/production_monitor.sh health

# Show detailed metrics
./scripts/production_monitor.sh metrics

# Start continuous monitoring
./scripts/production_monitor.sh monitor

# Check alert conditions
./scripts/production_monitor.sh alerts

# Analyze logs
./scripts/production_monitor.sh logs

# Run performance diagnostics
./scripts/production_monitor.sh performance

# Check cognitive system status
./scripts/production_monitor.sh cognitive
```

## Production Directory Structure

### Docker Deployment
```
production/
├── config/
│   ├── .env.production          # Environment configuration
│   ├── logging.yaml             # Logging configuration
│   ├── security.yaml            # Security settings
│   └── monitoring/              # Monitoring configurations
├── data/                        # Persistent data
├── logs/                        # Application logs
├── backups/                     # Configuration backups
├── docker-compose.production.yml  # Docker Compose config
├── start-production.sh          # Startup script
├── stop-production.sh           # Stop script
├── health-check.sh              # Health check script
└── backup-production.sh         # Backup script
```

### Standalone Deployment
```
/opt/pycog-zero/
├── app/                         # Application files
├── config/
│   ├── .env.production          # Environment configuration
│   ├── gunicorn.conf.py         # Gunicorn configuration
│   ├── logging.yaml             # Logging configuration
│   └── security.yaml            # Security settings
├── data/                        # Persistent data
├── logs/                        # Application logs
├── backups/                     # Backups
├── venv/                        # Python virtual environment
└── scripts/
    ├── start.sh                 # Startup script
    ├── stop.sh                  # Stop script
    ├── health-check.sh          # Health check script
    └── backup.sh                # Backup script
```

## Configuration

### Environment Variables

Key production environment variables to configure:

```bash
# Security Configuration (REQUIRED)
AUTH_LOGIN=admin
AUTH_PASSWORD=your_secure_password
API_KEY=your_secure_api_key
SESSION_SECRET_KEY=your_session_secret
JWT_SECRET_KEY=your_jwt_secret

# Server Configuration
HOST=0.0.0.0
PORT=8080

# Cognitive Features
OPENCOG_ENABLED=true
COGNITIVE_REASONING=true
PLN_INTEGRATION=true
ATTENTION_ALLOCATION=true

# Performance
MAX_WORKERS=4
WORKER_TIMEOUT=30
MAX_REQUESTS_PER_WORKER=1000

# Logging
LOG_LEVEL=INFO
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=10

# Security
SECURE_HEADERS=true
CSRF_PROTECTION=true
RATE_LIMITING=true
```

### Security Configuration

The production deployment includes comprehensive security settings:

1. **Authentication**: Basic HTTP authentication with configurable credentials
2. **API Security**: API key-based authentication for programmatic access
3. **Rate Limiting**: Configurable rate limits to prevent abuse
4. **Security Headers**: HTTPS enforcement, CSP, XSS protection
5. **CORS**: Cross-origin request configuration
6. **Session Security**: Secure session management with JWT support

### Performance Tuning

The deployment scripts apply production-ready performance optimizations:

1. **System Tuning**: Kernel parameters for network and memory optimization
2. **Resource Limits**: Container/process memory and CPU limits
3. **Connection Pooling**: Database and Redis connection optimization
4. **Caching**: Application-level caching configuration
5. **Logging**: Structured logging with rotation and compression

## Monitoring and Alerting

### Health Checks

The production deployment includes multiple health check mechanisms:

1. **HTTP Health Endpoint**: `/health` endpoint for basic application health
2. **System Metrics**: CPU, memory, disk usage monitoring
3. **Application Metrics**: Response times, error rates, request volumes
4. **Cognitive System Health**: OpenCog integration and cognitive tool status
5. **Log Analysis**: Automatic error and warning detection

### Alerting

Configure alerts for production issues:

```bash
# Set alert thresholds
export ALERT_THRESHOLD_CPU=80      # CPU usage %
export ALERT_THRESHOLD_MEMORY=85   # Memory usage %
export ALERT_THRESHOLD_DISK=90     # Disk usage %
export ALERT_EMAIL=admin@example.com

# Check alerts
./scripts/production_monitor.sh alerts
```

### Monitoring Dashboard

For comprehensive monitoring, integrate with external tools:

1. **Prometheus**: Metrics collection and storage
2. **Grafana**: Visualization dashboards
3. **ELK Stack**: Log aggregation and analysis
4. **Alertmanager**: Advanced alerting rules and notifications

## Backup and Recovery

### Automated Backups

Both deployment types include automated backup scripts:

```bash
# Docker deployment
cd production && ./backup-production.sh

# Standalone deployment
/opt/pycog-zero/scripts/backup.sh
```

**Backup Contents**:
- Configuration files
- Application data
- Recent logs (last 7 days)
- Database dumps (if applicable)

### Recovery Procedures

1. **Configuration Recovery**:
   ```bash
   # Extract backup
   tar -xzf backup-TIMESTAMP.tar.gz
   
   # Restore configuration
   cp -r backup-TIMESTAMP/config/* ./config/
   ```

2. **Data Recovery**:
   ```bash
   # Restore application data
   cp -r backup-TIMESTAMP/data/* ./data/
   
   # Restart services
   ./start-production.sh  # Docker
   sudo systemctl restart pycog-zero  # Standalone
   ```

## Scaling and High Availability

### Horizontal Scaling

For high-traffic deployments, consider:

1. **Load Balancer**: Nginx or HAProxy for request distribution
2. **Multiple Instances**: Run multiple PyCog-Zero instances
3. **Database Scaling**: PostgreSQL read replicas or clustering
4. **Redis Clustering**: Distributed caching and session storage

### Vertical Scaling

Adjust resource limits based on usage:

```bash
# Docker Compose scaling
docker-compose up --scale pycog-zero=3

# Resource adjustments
MEMORY_LIMIT=8g CPU_LIMIT=4 ./scripts/deploy_production_docker.sh
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**:
   ```bash
   # Check logs
   ./scripts/production_monitor.sh logs
   
   # Verify configuration
   ./scripts/production_config_manager.sh validate
   
   # Check system resources
   ./scripts/production_monitor.sh status
   ```

2. **High Resource Usage**:
   ```bash
   # Performance diagnostics
   ./scripts/production_monitor.sh performance
   
   # System tuning
   sudo ./scripts/production_config_manager.sh tune
   ```

3. **Cognitive Features Not Working**:
   ```bash
   # Check cognitive system
   ./scripts/production_monitor.sh cognitive
   
   # Verify OpenCog installation
   python3 -c "from opencog.atomspace import AtomSpace; print('OK')"
   ```

### Log Locations

- **Docker**: `production/logs/`
- **Standalone**: `/opt/pycog-zero/logs/`
- **System logs**: `journalctl -u pycog-zero`

### Support

For production support:

1. Check the comprehensive monitoring output
2. Review application and system logs
3. Validate configuration settings
4. Consult the cognitive system status
5. Run performance diagnostics

## Migration

### From Development to Production

1. **Export Development Data**:
   ```bash
   # Export agent memory and configurations
   python3 -c "
   from python.helpers import files, memory
   # Export logic here
   "
   ```

2. **Deploy Production Environment**:
   ```bash
   ./scripts/deploy_production_docker.sh
   ```

3. **Import Data**:
   ```bash
   # Copy data to production/data/
   cp -r dev_export/* production/data/
   ```

### Between Environments

Use the backup and restore procedures to migrate between production environments.

## Best Practices

1. **Security**:
   - Always change default passwords and API keys
   - Use HTTPS in production
   - Regularly update dependencies
   - Monitor security logs

2. **Performance**:
   - Monitor resource usage regularly
   - Implement proper caching strategies
   - Use connection pooling
   - Optimize database queries

3. **Reliability**:
   - Set up automated backups
   - Implement health checks
   - Use process managers (systemd/supervisor)
   - Plan for disaster recovery

4. **Monitoring**:
   - Set up comprehensive alerting
   - Monitor both system and application metrics
   - Implement log aggregation
   - Regular health check automation

This production deployment system provides a robust, scalable, and maintainable foundation for running PyCog-Zero cognitive agent systems in production environments.