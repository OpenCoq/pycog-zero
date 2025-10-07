#!/bin/bash

set -e

# PyCog-Zero Production Backup and Restore Script
# This script provides backup and restore capabilities for production PyCog-Zero deployments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configuration
BACKUP_DIR="/var/backups/pycog-zero"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMPOSE_FILE="${PROJECT_ROOT}/deploy/production/docker-compose-production.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

show_help() {
    echo "PyCog-Zero Production Backup and Restore Script"
    echo
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  backup               Create a full backup of the production system"
    echo "  restore <backup_id>  Restore from a specific backup"
    echo "  list                 List available backups"
    echo "  cleanup              Clean up old backups (keeps last 30 days)"
    echo "  verify <backup_id>   Verify backup integrity"
    echo
    echo "Options:"
    echo "  --help, -h           Show this help message"
    echo "  --backup-dir <dir>   Specify custom backup directory"
    echo "  --no-stop            Don't stop services during backup (faster but less consistent)"
    echo "  --compress           Compress backup archives (slower but smaller)"
    echo
    echo "Examples:"
    echo "  $0 backup                    # Create a full backup"
    echo "  $0 restore 20240101_120000   # Restore from specific backup"
    echo "  $0 list                      # List all available backups"
    echo "  $0 cleanup                   # Remove backups older than 30 days"
}

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
    
    # Check if compose file exists
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    success "Prerequisites check passed"
}

create_backup() {
    local stop_services=${1:-true}
    local compress=${2:-false}
    
    log "Starting backup process..."
    
    local backup_path="${BACKUP_DIR}/${TIMESTAMP}"
    mkdir -p "$backup_path"
    
    # Create backup metadata
    cat > "${backup_path}/backup_info.json" << EOF
{
    "backup_id": "${TIMESTAMP}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_type": "full",
    "services_stopped": ${stop_services},
    "compressed": ${compress},
    "pycog_zero_version": "$(cd ${PROJECT_ROOT} && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
    "backup_path": "${backup_path}"
}
EOF
    
    # Stop services if requested
    if [[ "$stop_services" == "true" ]]; then
        log "Stopping services for consistent backup..."
        cd "$(dirname "$COMPOSE_FILE")"
        docker-compose -f "$COMPOSE_FILE" stop
    fi
    
    # Backup Docker volumes
    log "Backing up Docker volumes..."
    backup_volumes "$backup_path" "$compress"
    
    # Backup configuration files
    log "Backing up configuration files..."
    backup_configurations "$backup_path" "$compress"
    
    # Backup database/memory state
    log "Backing up cognitive state and memory..."
    backup_cognitive_state "$backup_path" "$compress"
    
    # Create backup manifest
    create_backup_manifest "$backup_path"
    
    # Restart services if they were stopped
    if [[ "$stop_services" == "true" ]]; then
        log "Restarting services..."
        cd "$(dirname "$COMPOSE_FILE")"
        docker-compose -f "$COMPOSE_FILE" start
    fi
    
    success "Backup completed: ${backup_path}"
    log "Backup ID: ${TIMESTAMP}"
}

backup_volumes() {
    local backup_path="$1"
    local compress="$2"
    
    local volumes_dir="${backup_path}/volumes"
    mkdir -p "$volumes_dir"
    
    # List of volumes to backup
    local volumes=(
        "pycog-zero-data"
        "pycog-zero-logs" 
        "pycog-zero-knowledge"
        "pycog-zero-config"
        "prometheus-data"
    )
    
    for volume in "${volumes[@]}"; do
        if docker volume ls -q | grep -q "^${volume}$"; then
            log "Backing up volume: $volume"
            
            if [[ "$compress" == "true" ]]; then
                docker run --rm \
                    -v "${volume}:/data:ro" \
                    -v "${volumes_dir}:/backup" \
                    alpine \
                    sh -c "cd /data && tar czf /backup/${volume}.tar.gz ."
            else
                docker run --rm \
                    -v "${volume}:/data:ro" \
                    -v "${volumes_dir}:/backup" \
                    alpine \
                    sh -c "cd /data && tar cf /backup/${volume}.tar ."
            fi
            
            success "Volume $volume backed up"
        else
            warning "Volume $volume not found, skipping"
        fi
    done
}

backup_configurations() {
    local backup_path="$1"
    local compress="$2"
    
    local config_dir="${backup_path}/configurations"
    mkdir -p "$config_dir"
    
    # Backup project configurations
    if [[ -d "${PROJECT_ROOT}/conf" ]]; then
        cp -r "${PROJECT_ROOT}/conf" "${config_dir}/"
    fi
    
    if [[ -d "${PROJECT_ROOT}/deploy" ]]; then
        cp -r "${PROJECT_ROOT}/deploy" "${config_dir}/"
    fi
    
    # Backup environment files
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        cp "${PROJECT_ROOT}/.env" "${config_dir}/"
    fi
    
    if [[ "$compress" == "true" ]]; then
        cd "$config_dir"
        tar czf configurations.tar.gz *
        rm -rf conf deploy .env 2>/dev/null || true
    fi
    
    success "Configurations backed up"
}

backup_cognitive_state() {
    local backup_path="$1"
    local compress="$2"
    
    local cognitive_dir="${backup_path}/cognitive_state"
    mkdir -p "$cognitive_dir"
    
    # Export cognitive state from running containers
    if docker ps -q --filter "name=pycog-zero-production" | grep -q .; then
        log "Exporting cognitive state from running container..."
        
        # Export AtomSpace state if available
        docker exec pycog-zero-production \
            sh -c "if [ -f /a0/export_cognitive_state.py ]; then python3 /a0/export_cognitive_state.py --output /tmp/cognitive_state.json; fi" \
            2>/dev/null || true
        
        # Copy exported state
        docker cp pycog-zero-production:/tmp/cognitive_state.json "${cognitive_dir}/" 2>/dev/null || true
        
        # Export memory database
        docker exec pycog-zero-production \
            sh -c "if [ -d /a0/memory ]; then tar cf /tmp/memory_export.tar /a0/memory/; fi" \
            2>/dev/null || true
        
        docker cp pycog-zero-production:/tmp/memory_export.tar "${cognitive_dir}/" 2>/dev/null || true
    else
        warning "No running PyCog-Zero container found for cognitive state export"
    fi
    
    if [[ "$compress" == "true" && -f "${cognitive_dir}/memory_export.tar" ]]; then
        gzip "${cognitive_dir}/memory_export.tar"
    fi
    
    success "Cognitive state backed up"
}

create_backup_manifest() {
    local backup_path="$1"
    
    log "Creating backup manifest..."
    
    # Generate file checksums
    find "$backup_path" -type f -exec sha256sum {} \; > "${backup_path}/checksums.txt"
    
    # Create manifest
    cat > "${backup_path}/manifest.txt" << EOF
PyCog-Zero Production Backup Manifest
=====================================

Backup ID: ${TIMESTAMP}
Created: $(date)
Backup Path: ${backup_path}

Contents:
$(ls -la "${backup_path}")

Checksums: $(wc -l < "${backup_path}/checksums.txt") files

Volume Backups:
$(ls -la "${backup_path}/volumes" 2>/dev/null || echo "No volumes backed up")

Configuration Backups:
$(ls -la "${backup_path}/configurations" 2>/dev/null || echo "No configurations backed up")

Cognitive State Backups:
$(ls -la "${backup_path}/cognitive_state" 2>/dev/null || echo "No cognitive state backed up")
EOF
    
    success "Backup manifest created"
}

restore_backup() {
    local backup_id="$1"
    
    if [[ -z "$backup_id" ]]; then
        error "Backup ID required for restore operation"
        exit 1
    fi
    
    local backup_path="${BACKUP_DIR}/${backup_id}"
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup not found: $backup_path"
        exit 1
    fi
    
    log "Starting restore from backup: $backup_id"
    
    # Verify backup integrity first
    if ! verify_backup "$backup_id"; then
        error "Backup verification failed, aborting restore"
        exit 1
    fi
    
    # Stop services
    log "Stopping services for restore..."
    cd "$(dirname "$COMPOSE_FILE")"
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore volumes
    log "Restoring Docker volumes..."
    restore_volumes "$backup_path"
    
    # Restore configurations
    log "Restoring configurations..."
    restore_configurations "$backup_path"
    
    # Restore cognitive state
    log "Restoring cognitive state..."
    restore_cognitive_state "$backup_path"
    
    # Start services
    log "Starting services..."
    cd "$(dirname "$COMPOSE_FILE")"
    docker-compose -f "$COMPOSE_FILE" up -d
    
    success "Restore completed from backup: $backup_id"
}

restore_volumes() {
    local backup_path="$1"
    local volumes_dir="${backup_path}/volumes"
    
    if [[ ! -d "$volumes_dir" ]]; then
        warning "No volume backups found in $backup_path"
        return
    fi
    
    for backup_file in "${volumes_dir}"/*.tar*; do
        if [[ -f "$backup_file" ]]; then
            local volume_name=$(basename "$backup_file" | sed 's/\.tar.*$//')
            
            log "Restoring volume: $volume_name"
            
            # Remove existing volume
            docker volume rm "$volume_name" 2>/dev/null || true
            
            # Create new volume
            docker volume create "$volume_name"
            
            # Restore data
            if [[ "$backup_file" == *.tar.gz ]]; then
                docker run --rm \
                    -v "${volume_name}:/data" \
                    -v "${volumes_dir}:/backup:ro" \
                    alpine \
                    sh -c "cd /data && tar xzf /backup/$(basename "$backup_file")"
            else
                docker run --rm \
                    -v "${volume_name}:/data" \
                    -v "${volumes_dir}:/backup:ro" \
                    alpine \
                    sh -c "cd /data && tar xf /backup/$(basename "$backup_file")"
            fi
            
            success "Volume $volume_name restored"
        fi
    done
}

restore_configurations() {
    local backup_path="$1"
    local config_dir="${backup_path}/configurations"
    
    if [[ ! -d "$config_dir" ]]; then
        warning "No configuration backups found in $backup_path"
        return
    fi
    
    # Restore configurations
    if [[ -d "${config_dir}/conf" ]]; then
        cp -r "${config_dir}/conf" "${PROJECT_ROOT}/"
    fi
    
    if [[ -d "${config_dir}/deploy" ]]; then
        cp -r "${config_dir}/deploy" "${PROJECT_ROOT}/"
    fi
    
    if [[ -f "${config_dir}/.env" ]]; then
        cp "${config_dir}/.env" "${PROJECT_ROOT}/"
    fi
    
    success "Configurations restored"
}

restore_cognitive_state() {
    local backup_path="$1"
    local cognitive_dir="${backup_path}/cognitive_state"
    
    if [[ ! -d "$cognitive_dir" ]]; then
        warning "No cognitive state backups found in $backup_path"
        return
    fi
    
    # Restore cognitive state files to temporary location
    # They will be imported when the container starts
    local temp_dir="/tmp/pycog-zero-restore"
    mkdir -p "$temp_dir"
    
    if [[ -f "${cognitive_dir}/cognitive_state.json" ]]; then
        cp "${cognitive_dir}/cognitive_state.json" "$temp_dir/"
    fi
    
    if [[ -f "${cognitive_dir}/memory_export.tar" ]]; then
        cp "${cognitive_dir}/memory_export.tar" "$temp_dir/"
    fi
    
    if [[ -f "${cognitive_dir}/memory_export.tar.gz" ]]; then
        cp "${cognitive_dir}/memory_export.tar.gz" "$temp_dir/"
    fi
    
    success "Cognitive state restored to temporary location"
}

list_backups() {
    log "Available backups:"
    echo
    
    if [[ ! -d "$BACKUP_DIR" ]] || [[ -z "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]]; then
        warning "No backups found in $BACKUP_DIR"
        return
    fi
    
    printf "%-20s %-20s %-10s %-30s\n" "BACKUP_ID" "TIMESTAMP" "SIZE" "PATH"
    echo "--------------------------------------------------------------------------------"
    
    for backup_path in "$BACKUP_DIR"/*; do
        if [[ -d "$backup_path" ]]; then
            local backup_id=$(basename "$backup_path")
            local size=$(du -sh "$backup_path" 2>/dev/null | cut -f1)
            local timestamp=""
            
            if [[ -f "${backup_path}/backup_info.json" ]]; then
                timestamp=$(python3 -c "import json; print(json.load(open('${backup_path}/backup_info.json'))['timestamp'])" 2>/dev/null || echo "unknown")
            fi
            
            printf "%-20s %-20s %-10s %-30s\n" "$backup_id" "$timestamp" "$size" "$backup_path"
        fi
    done
}

verify_backup() {
    local backup_id="$1"
    
    if [[ -z "$backup_id" ]]; then
        error "Backup ID required for verification"
        exit 1
    fi
    
    local backup_path="${BACKUP_DIR}/${backup_id}"
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup not found: $backup_path"
        exit 1
    fi
    
    log "Verifying backup: $backup_id"
    
    # Check if required files exist
    local required_files=(
        "backup_info.json"
        "manifest.txt"
        "checksums.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "${backup_path}/${file}" ]]; then
            error "Required file missing: $file"
            return 1
        fi
    done
    
    # Verify checksums
    log "Verifying file checksums..."
    if cd "$backup_path" && sha256sum -c checksums.txt --quiet; then
        success "Backup verification passed: $backup_id"
        return 0
    else
        error "Backup verification failed: checksum mismatch"
        return 1
    fi
}

cleanup_backups() {
    local retention_days=${1:-30}
    
    log "Cleaning up backups older than $retention_days days..."
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log "No backup directory found, nothing to clean"
        return
    fi
    
    local deleted_count=0
    
    for backup_path in "$BACKUP_DIR"/*; do
        if [[ -d "$backup_path" ]]; then
            local backup_age=$(find "$backup_path" -maxdepth 0 -mtime +$retention_days)
            if [[ -n "$backup_age" ]]; then
                local backup_id=$(basename "$backup_path")
                log "Removing old backup: $backup_id"
                rm -rf "$backup_path"
                ((deleted_count++))
            fi
        fi
    done
    
    success "Cleanup completed: $deleted_count old backups removed"
}

# Main script logic
case "${1:-}" in
    "backup")
        check_prerequisites
        stop_services=true
        compress=false
        
        # Parse options
        shift
        while [[ $# -gt 0 ]]; do
            case $1 in
                --no-stop)
                    stop_services=false
                    shift
                    ;;
                --compress)
                    compress=true
                    shift
                    ;;
                --backup-dir)
                    BACKUP_DIR="$2"
                    shift 2
                    ;;
                *)
                    error "Unknown option: $1"
                    exit 1
                    ;;
            esac
        done
        
        create_backup "$stop_services" "$compress"
        ;;
    "restore")
        if [[ -z "${2:-}" ]]; then
            error "Backup ID required for restore"
            show_help
            exit 1
        fi
        check_prerequisites
        restore_backup "$2"
        ;;
    "list")
        list_backups
        ;;
    "verify")
        if [[ -z "${2:-}" ]]; then
            error "Backup ID required for verification"
            show_help
            exit 1
        fi
        verify_backup "$2"
        ;;
    "cleanup")
        cleanup_backups "${2:-30}"
        ;;
    "--help"|"-h"|"help")
        show_help
        ;;
    "")
        error "No command specified"
        show_help
        exit 1
        ;;
    *)
        error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac