# PyCog-Zero Production Deployment Guide

This guide provides comprehensive instructions for deploying PyCog-Zero cognitive Agent-Zero systems in production environments.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Production Configuration](#production-configuration)
5. [Deployment Options](#deployment-options)
6. [Monitoring and Health Checks](#monitoring-and-health-checks)
7. [Backup and Recovery](#backup-and-recovery)
8. [Security Considerations](#security-considerations)
9. [Scaling and Performance](#scaling-and-performance)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance](#maintenance)

## Overview

PyCog-Zero production deployment provides a robust, scalable, and monitored environment for running cognitive Agent-Zero systems with full OpenCog integration.

### Key Features

- **Containerized Architecture**: Docker-based deployment for consistency and isolation
- **Health Monitoring**: Comprehensive health checks and metrics collection
- **Backup & Recovery**: Automated backup systems with point-in-time recovery
- **Security**: Built-in security configurations and best practices
- **Monitoring**: Prometheus-based monitoring with alerting
- **High Availability**: Production-ready configurations with resource limits

### Architecture Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   PyCog-Zero     │    │   Monitoring    │
│   (Nginx)       │────│   Web Service    │────│   (Prometheus)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │    Data Layer   │
                       │                 │
                       │  • AtomSpace    │
                       │  • Agent Memory │
                       │  • Knowledge    │
                       │  • Configurations│
                       └─────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU**: 2 cores
- **Memory**: 4GB RAM
- **Storage**: 20GB available space
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2

**Recommended for Production:**
- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 50GB+ SSD storage
- **OS**: Linux (Ubuntu 22.04 LTS recommended)

### Software Requirements

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Git**: For cloning the repository
- **curl**: For health checks and validation

### Network Requirements

- **Ports**: 80 (HTTP), 443 (HTTPS), 50001 (alternative), 9090 (monitoring)
- **Internet Access**: Required for Docker image pulls and dependency downloads

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/OpenCoq/pycog-zero.git
cd pycog-zero
```

### 2. Deploy to Production

```bash
# Run the production deployment script
./scripts/deployment/deploy-production.sh
```

### 3. Validate Deployment

```bash
# Validate the production deployment
./scripts/deployment/validate-production.py
```

### 4. Access Services

- **Web Interface**: http://localhost:80
- **Alternative Port**: http://localhost:50001
- **Health Check**: http://localhost:80/health
- **Monitoring**: http://localhost:9090

## Production Configuration

### Environment Variables

Set these environment variables for production deployment:

```bash
# Core Configuration
export PYCOG_ZERO_MODE=production
export FLASK_ENV=production
export NODE_ENV=production

# Cognitive Features
export COGNITIVE_MODE=true
export OPENCOG_ENABLED=true
export NEURAL_SYMBOLIC_BRIDGE=true
export ECAN_ATTENTION=true
export PLN_REASONING=true

# Production Features
export MONITORING_ENABLED=true
export BACKUP_ENABLED=true
export LOG_LEVEL=INFO
```

### Configuration Files

#### Production Configuration (`deploy/production/production-config/config_production.json`)

```json
{
  "environment": "production",
  "cognitive_mode": true,
  "opencog_enabled": true,
  "performance_optimization": {
    "enabled": true,
    "max_memory_usage": "6GB",
    "cpu_limit": 4,
    "concurrent_agents": 8
  },
  "logging": {
    "level": "INFO",
    "structured_logging": true,
    "log_rotation": true
  },
  "security": {
    "authentication_required": true,
    "ssl_enabled": true,
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60
    }
  },
  "monitoring": {
    "health_checks": true,
    "prometheus_enabled": true
  }
}
```

## Deployment Options

### Option 1: Automated Deployment Script

**Recommended for most users**

```bash
# Full automated deployment
./scripts/deployment/deploy-production.sh

# Deployment with options
./scripts/deployment/deploy-production.sh --validate  # Validate only
./scripts/deployment/deploy-production.sh --backup   # Backup only
./scripts/deployment/deploy-production.sh --health   # Health check only
```

### Option 2: Manual Docker Compose

```bash
# Navigate to production deployment directory
cd deploy/production

# Build and deploy services
docker-compose -f docker-compose-production.yml up -d

# Check service status
docker-compose -f docker-compose-production.yml ps
```

### Option 3: Staging Deployment

```bash
# Deploy to staging environment (similar to production but with different configs)
cp deploy/production/* deploy/staging/
# Modify staging configurations as needed
cd deploy/staging
docker-compose -f docker-compose-staging.yml up -d
```

## Monitoring and Health Checks

### Built-in Health Checks

PyCog-Zero includes comprehensive health monitoring:

#### Health Check Endpoints

- **`/health`** - Overall system health
- **`/metrics`** - System metrics (CPU, memory, disk)
- **`/cognitive/metrics`** - Cognitive system specific metrics

#### Health Check Example

```bash
curl http://localhost:80/health
```

Response:
```json
{
  "overall_status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "checks": {
    "system": {
      "status": "healthy",
      "message": "System resources within normal ranges",
      "response_time_ms": 45.2
    },
    "cognitive_system": {
      "status": "healthy", 
      "message": "Cognitive system is operational",
      "response_time_ms": 123.8
    }
  }
}
```

### Prometheus Monitoring

#### Accessing Prometheus

- **URL**: http://localhost:9090
- **Metrics**: Scraped every 30 seconds
- **Retention**: 30 days

#### Key Metrics to Monitor

- **System Metrics**:
  - CPU usage percentage
  - Memory usage percentage
  - Disk usage percentage
  - Network I/O

- **Application Metrics**:
  - Response time
  - Request rate
  - Error rate
  - Active connections

- **Cognitive Metrics**:
  - AtomSpace size
  - Active agents count
  - Reasoning operations per second
  - Attention allocation efficiency

### Alerting

Alerts are configured for:
- High CPU usage (>80%)
- High memory usage (>80%)
- Service downtime
- Cognitive system failures
- Low disk space (<10%)

## Backup and Recovery

### Automated Backup System

The production deployment includes automated backup systems:

#### Backup Components

- **Docker Volumes**: All persistent data
- **Configuration Files**: Production configurations
- **Cognitive State**: AtomSpace and memory exports
- **Application Data**: Logs and knowledge base

#### Backup Commands

```bash
# Create full backup
./scripts/deployment/backup-restore.sh backup

# Create compressed backup
./scripts/deployment/backup-restore.sh backup --compress

# Create backup without stopping services (faster, less consistent)
./scripts/deployment/backup-restore.sh backup --no-stop

# List available backups
./scripts/deployment/backup-restore.sh list

# Verify backup integrity
./scripts/deployment/backup-restore.sh verify 20240115_143000

# Restore from backup
./scripts/deployment/backup-restore.sh restore 20240115_143000

# Cleanup old backups (older than 30 days)
./scripts/deployment/backup-restore.sh cleanup
```

#### Backup Schedule

- **Automatic**: Hourly backups via container
- **Retention**: 30 days of backups
- **Location**: `/var/backups/pycog-zero/`

### Disaster Recovery

#### Recovery Procedures

1. **Service Failure Recovery**:
   ```bash
   # Restart services
   cd deploy/production
   docker-compose -f docker-compose-production.yml restart
   ```

2. **Data Corruption Recovery**:
   ```bash
   # Restore from latest backup
   ./scripts/deployment/backup-restore.sh restore $(./scripts/deployment/backup-restore.sh list | tail -1 | awk '{print $1}')
   ```

3. **Complete System Recovery**:
   ```bash
   # Full system restore from backup
   ./scripts/deployment/backup-restore.sh restore BACKUP_ID
   ./scripts/deployment/deploy-production.sh --validate
   ```

## Security Considerations

### Network Security

- **Firewall Configuration**: Only expose necessary ports
- **SSL/TLS**: Configure HTTPS with valid certificates
- **Rate Limiting**: Built-in rate limiting for API endpoints

### Application Security

- **Authentication**: Enable authentication for production access
- **Input Validation**: All inputs are validated and sanitized  
- **Security Headers**: Security headers are automatically configured
- **Container Security**: Containers run with minimal privileges

### Data Security

- **Encryption**: Data at rest encryption for sensitive information
- **Access Control**: Role-based access control for administrative functions
- **Audit Logging**: Comprehensive audit logs for security monitoring

### Security Configuration

```bash
# Enable security features in production config
{
  "security": {
    "authentication_required": true,
    "ssl_enabled": true,
    "secure_headers": true,
    "input_validation": true,
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60,
      "burst_limit": 10
    }
  }
}
```

## Scaling and Performance

### Performance Optimization

#### Resource Limits

Configure appropriate resource limits in `docker-compose-production.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

#### Performance Tuning

1. **Memory Management**:
   - Configure AtomSpace memory limits
   - Enable memory optimization features
   - Monitor memory usage patterns

2. **CPU Optimization**:
   - Adjust concurrent agent limits
   - Configure reasoning operation limits
   - Enable batch processing

3. **Storage Optimization**:
   - Use SSD storage for better I/O performance
   - Configure appropriate volume mounting
   - Regular cleanup of temporary files

### Horizontal Scaling

For high-load scenarios, consider:

1. **Load Balancing**: Deploy multiple instances behind a load balancer
2. **Database Clustering**: Distribute cognitive state across multiple nodes
3. **Microservices**: Split cognitive functions into separate services

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

**Symptoms**: Container exits immediately or health checks fail

**Solutions**:
```bash
# Check container logs
docker-compose -f deploy/production/docker-compose-production.yml logs

# Check resource availability
docker stats

# Validate configuration
./scripts/deployment/validate-production.py --validate
```

#### 2. High Memory Usage

**Symptoms**: Memory alerts, slow performance

**Solutions**:
```bash
# Check memory usage
docker stats pycog-zero-production

# Review AtomSpace size
curl http://localhost:80/cognitive/metrics

# Optimize memory settings in config
```

#### 3. Cognitive Features Not Working

**Symptoms**: Cognitive endpoints return errors

**Solutions**:
```bash
# Check OpenCog installation
docker exec pycog-zero-production python3 -c "import opencog; print('OpenCog available')"

# Verify cognitive configuration
cat deploy/production/production-config/config_production.json

# Check cognitive system logs
docker logs pycog-zero-production | grep -i cognitive
```

### Diagnostic Commands

```bash
# Full system validation
./scripts/deployment/validate-production.py

# Health check
curl -f http://localhost:80/health

# Service status
docker-compose -f deploy/production/docker-compose-production.yml ps

# View logs
docker-compose -f deploy/production/docker-compose-production.yml logs --tail=100

# Resource usage
docker stats $(docker ps --filter "name=pycog-zero" --format "{{.Names}}")
```

## Maintenance

### Regular Maintenance Tasks

#### Daily
- Monitor system health via `/health` endpoint
- Check error logs for anomalies
- Verify backup completion

#### Weekly  
- Review performance metrics
- Update security configurations
- Test backup restore procedures

#### Monthly
- Update container images
- Review and update configurations
- Performance optimization review
- Security audit

### Update Procedures

#### 1. Application Updates

```bash
# Pull latest changes
git pull origin main

# Build new image
cd deploy/production
docker-compose -f docker-compose-production.yml build

# Rolling update (zero downtime)
docker-compose -f docker-compose-production.yml up -d --no-deps web
```

#### 2. Configuration Updates

```bash
# Update configuration files
vim deploy/production/production-config/config_production.json

# Restart services with new config
docker-compose -f deploy/production/docker-compose-production.yml restart
```

#### 3. Security Updates

```bash
# Update base images
docker pull agent0ai/agent-zero-base:latest

# Rebuild with security updates
docker-compose -f deploy/production/docker-compose-production.yml build --pull

# Deploy updates
docker-compose -f deploy/production/docker-compose-production.yml up -d
```

### Monitoring Maintenance

- **Log Rotation**: Automatic log rotation is configured
- **Metrics Retention**: Prometheus retains 30 days of metrics
- **Backup Cleanup**: Automatic cleanup of backups older than 30 days

## Support

For additional support:

1. **Documentation**: Check the [main README](../README.md) and [architecture docs](architecture.md)
2. **Issues**: Report issues on the [GitHub repository](https://github.com/OpenCoq/pycog-zero/issues)
3. **Community**: Join discussions in the project community channels

## Appendix

### Configuration Templates

Complete configuration templates are available in:
- `deploy/production/production-config/`
- `deploy/staging/staging-config/`
- `deploy/development/development-config/`

### Script Reference

All deployment scripts are located in `scripts/deployment/`:
- `deploy-production.sh` - Main deployment script
- `validate-production.py` - Production validation tool
- `backup-restore.sh` - Backup and recovery operations

### Port Reference

- **80**: Main HTTP interface
- **443**: HTTPS interface (when SSL configured)
- **50001**: Alternative HTTP port
- **9090**: Prometheus monitoring
- **22**: SSH (container access)
- **9000-9009**: Reserved for extensions