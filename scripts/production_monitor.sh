#!/bin/bash
#
# PyCog-Zero Production Monitoring Script
# =======================================
#
# Comprehensive monitoring and health checking for PyCog-Zero production deployments.
# Includes system metrics, application health, cognitive system status, and alerting.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="${PROD_DIR:-/opt/pycog-zero}"
CONFIG_DIR="${CONFIG_DIR:-$PROD_DIR/config}"
LOG_DIR="${LOG_DIR:-$PROD_DIR/logs}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8080/health}"
API_URL="${API_URL:-http://localhost:8080/api}"
ALERT_EMAIL="${ALERT_EMAIL:-admin@localhost}"
ALERT_THRESHOLD_CPU="${ALERT_THRESHOLD_CPU:-80}"
ALERT_THRESHOLD_MEMORY="${ALERT_THRESHOLD_MEMORY:-85}"
ALERT_THRESHOLD_DISK="${ALERT_THRESHOLD_DISK:-90}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"

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

print_metric() {
    echo -e "${CYAN}📊${NC} $1"
}

usage() {
    echo "PyCog-Zero Production Monitoring Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  status          Show current system and application status"
    echo "  health          Perform comprehensive health check"
    echo "  metrics         Display system and application metrics"
    echo "  monitor         Start continuous monitoring (daemon mode)"
    echo "  alerts          Check and send alerts if thresholds exceeded"
    echo "  logs            Analyze recent logs for issues"
    echo "  performance     Run performance diagnostics"
    echo "  cognitive       Check cognitive system status"
    echo ""
    echo "Options:"
    echo "  --prod-dir DIR      Production directory (default: $PROD_DIR)"
    echo "  --config-dir DIR    Configuration directory (default: $CONFIG_DIR)"
    echo "  --interval SEC      Monitoring interval in seconds (default: $MONITOR_INTERVAL)"
    echo "  --alert-email EMAIL Email for alerts (default: $ALERT_EMAIL)"
    echo "  --verbose           Enable verbose output"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 health --verbose"
    echo "  $0 monitor --interval 30"
    echo "  $0 alerts --alert-email admin@example.com"
    echo ""
}

# Parse command line arguments
COMMAND=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        status|health|metrics|monitor|alerts|logs|performance|cognitive)
            COMMAND="$1"
            shift
            ;;
        --prod-dir)
            PROD_DIR="$2"
            CONFIG_DIR="$PROD_DIR/config"
            LOG_DIR="$PROD_DIR/logs"
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        --alert-email)
            ALERT_EMAIL="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
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

if [ -z "$COMMAND" ]; then
    usage
    exit 1
fi

# Load configuration if available
if [ -f "$CONFIG_DIR/.env.production" ]; then
    set -a
    source "$CONFIG_DIR/.env.production"
    set +a
    if [ "$VERBOSE" = true ]; then
        print_info "Loaded production configuration"
    fi
fi

echo -e "${BLUE}PyCog-Zero Production Monitor${NC}"
echo "=============================="
echo "Command: $COMMAND"
echo "Production Directory: $PROD_DIR"
echo "Health URL: $HEALTH_URL"
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log_event() {
    echo "[$(get_timestamp)] $1" >> "$LOG_DIR/monitor.log"
}

# System status functions
get_system_status() {
    local cpu_usage memory_usage disk_usage load_avg uptime
    
    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Memory usage
    memory_info=$(free | grep Mem)
    memory_total=$(echo $memory_info | awk '{print $2}')
    memory_used=$(echo $memory_info | awk '{print $3}')
    memory_usage=$(awk "BEGIN {printf \"%.1f\", $memory_used/$memory_total*100}")
    
    # Disk usage
    disk_usage=$(df -h "$PROD_DIR" | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    
    # Load average
    load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | cut -d',' -f1)
    
    # System uptime
    uptime=$(uptime -p)
    
    print_metric "System Metrics:"
    echo "  CPU Usage: ${cpu_usage}%"
    echo "  Memory Usage: ${memory_usage}%"
    echo "  Disk Usage: ${disk_usage}%"
    echo "  Load Average: $load_avg"
    echo "  Uptime: $uptime"
    
    # Store for alerting
    echo "$cpu_usage" > "/tmp/pycog_cpu_usage"
    echo "$memory_usage" > "/tmp/pycog_memory_usage"
    echo "$disk_usage" > "/tmp/pycog_disk_usage"
}

get_application_status() {
    local service_status pid_status port_status
    
    print_metric "Application Status:"
    
    # Check if systemd service is running
    if systemctl is-active --quiet pycog-zero 2>/dev/null; then
        service_status="running"
        print_status "Service: Running"
    else
        service_status="stopped"
        print_error "Service: Stopped"
    fi
    
    # Check PID file
    if [ -f "$PROD_DIR/pycog-zero.pid" ]; then
        local pid=$(cat "$PROD_DIR/pycog-zero.pid")
        if kill -0 "$pid" 2>/dev/null; then
            pid_status="running"
            print_status "Process: Running (PID: $pid)"
        else
            pid_status="dead"
            print_error "Process: Dead (stale PID file)"
        fi
    else
        pid_status="no_pidfile"
        print_warning "Process: No PID file found"
    fi
    
    # Check port binding
    local port=$(grep "^PORT=" "$CONFIG_DIR/.env.production" 2>/dev/null | cut -d'=' -f2 || echo "8080")
    if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
        port_status="listening"
        print_status "Port $port: Listening"
    else
        port_status="not_listening"
        print_error "Port $port: Not listening"
    fi
    
    echo "$service_status" > "/tmp/pycog_service_status"
    echo "$pid_status" > "/tmp/pycog_pid_status"
    echo "$port_status" > "/tmp/pycog_port_status"
}

check_health() {
    local http_status response_time
    
    print_metric "Health Check:"
    
    # HTTP health check
    if command -v curl &> /dev/null; then
        local start_time=$(date +%s.%N)
        http_status=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")
        local end_time=$(date +%s.%N)
        response_time=$(awk "BEGIN {printf \"%.3f\", $end_time - $start_time}")
        
        if [ "$http_status" = "200" ]; then
            print_status "HTTP Health: OK (${response_time}s)"
        else
            print_error "HTTP Health: Failed (HTTP $http_status)"
        fi
    else
        print_warning "HTTP Health: curl not available"
        http_status="curl_missing"
        response_time="0"
    fi
    
    echo "$http_status" > "/tmp/pycog_http_status"
    echo "$response_time" > "/tmp/pycog_response_time"
}

check_cognitive_status() {
    print_metric "Cognitive System Status:"
    
    # Check OpenCog integration
    if python3 -c "from opencog.atomspace import AtomSpace; print('OpenCog available')" 2>/dev/null; then
        print_status "OpenCog: Available"
    else
        print_warning "OpenCog: Not available or not configured"
    fi
    
    # Check cognitive reasoning tool
    if python3 -c "
import sys
sys.path.insert(0, '$PROD_DIR/app')
try:
    from python.tools.cognitive_reasoning import CognitiveReasoningTool
    print('Cognitive reasoning tool available')
except ImportError:
    print('Cognitive reasoning tool not available')
except Exception as e:
    print(f'Cognitive reasoning tool error: {e}')
" 2>/dev/null | grep -q "available"; then
        print_status "Cognitive Reasoning: Available"
    else
        print_warning "Cognitive Reasoning: Not available"
    fi
    
    # Check memory persistence
    if [ -d "$PROD_DIR/data/memory" ]; then
        local memory_files=$(find "$PROD_DIR/data/memory" -name "*.json" | wc -l)
        print_status "Memory Persistence: $memory_files files"
    else
        print_warning "Memory Persistence: Directory not found"
    fi
}

analyze_logs() {
    print_metric "Log Analysis:"
    
    local error_count warning_count recent_errors
    
    # Count recent errors and warnings
    if [ -f "$LOG_DIR/error.log" ]; then
        error_count=$(tail -1000 "$LOG_DIR/error.log" | grep -c "ERROR" || echo "0")
        recent_errors=$(tail -100 "$LOG_DIR/error.log" | grep "ERROR" | tail -5)
        
        if [ "$error_count" -gt 0 ]; then
            print_warning "Recent Errors: $error_count"
            if [ "$VERBOSE" = true ] && [ -n "$recent_errors" ]; then
                echo "  Recent error samples:"
                echo "$recent_errors" | sed 's/^/    /'
            fi
        else
            print_status "Recent Errors: None"
        fi
    else
        print_warning "Error log not found"
    fi
    
    if [ -f "$LOG_DIR/app.log" ]; then
        warning_count=$(tail -1000 "$LOG_DIR/app.log" | grep -c "WARNING" || echo "0")
        
        if [ "$warning_count" -gt 10 ]; then
            print_warning "Recent Warnings: $warning_count"
        else
            print_status "Recent Warnings: $warning_count"
        fi
    else
        print_warning "Application log not found"
    fi
}

run_performance_diagnostics() {
    print_metric "Performance Diagnostics:"
    
    # Check response times
    if command -v curl &> /dev/null; then
        local api_response_time
        local start_time=$(date +%s.%N)
        curl -s "$API_URL/status" > /dev/null 2>&1
        local end_time=$(date +%s.%N)
        api_response_time=$(awk "BEGIN {printf \"%.3f\", $end_time - $start_time}")
        
        print_metric "API Response Time: ${api_response_time}s"
        
        if (( $(awk "BEGIN {print ($api_response_time > 2.0)}") )); then
            print_warning "API response time is slow"
        fi
    fi
    
    # Check memory usage by process
    if [ -f "$PROD_DIR/pycog-zero.pid" ]; then
        local pid=$(cat "$PROD_DIR/pycog-zero.pid")
        if kill -0 "$pid" 2>/dev/null; then
            local memory_usage=$(ps -o pid,ppid,cmd,%mem,%cpu --sort=-%mem -p "$pid" | tail -1)
            print_metric "Process Memory: $(echo $memory_usage | awk '{print $4}')%"
            print_metric "Process CPU: $(echo $memory_usage | awk '{print $5}')%"
        fi
    fi
    
    # Check file descriptor usage
    if [ -f "$PROD_DIR/pycog-zero.pid" ]; then
        local pid=$(cat "$PROD_DIR/pycog-zero.pid")
        if kill -0 "$pid" 2>/dev/null; then
            local fd_count=$(ls /proc/$pid/fd 2>/dev/null | wc -l || echo "unknown")
            print_metric "File Descriptors: $fd_count"
        fi
    fi
}

send_alert() {
    local subject="$1"
    local message="$2"
    local timestamp=$(get_timestamp)
    
    # Log alert
    log_event "ALERT: $subject - $message"
    
    # Send email if configured
    if command -v mail &> /dev/null && [ "$ALERT_EMAIL" != "admin@localhost" ]; then
        echo -e "PyCog-Zero Production Alert\n\nTime: $timestamp\nSubject: $subject\n\nDetails:\n$message" | \
            mail -s "PyCog-Zero Alert: $subject" "$ALERT_EMAIL"
        print_info "Alert sent to $ALERT_EMAIL"
    fi
    
    # Write to alert log
    echo "[$timestamp] ALERT: $subject - $message" >> "$LOG_DIR/alerts.log"
}

check_alerts() {
    local alerts_sent=0
    
    print_header "Checking Alert Conditions"
    
    # CPU usage alert
    if [ -f "/tmp/pycog_cpu_usage" ]; then
        local cpu_usage=$(cat "/tmp/pycog_cpu_usage")
        if (( $(awk "BEGIN {print ($cpu_usage > $ALERT_THRESHOLD_CPU)}") )); then
            send_alert "High CPU Usage" "CPU usage is ${cpu_usage}% (threshold: ${ALERT_THRESHOLD_CPU}%)"
            ((alerts_sent++))
        fi
    fi
    
    # Memory usage alert
    if [ -f "/tmp/pycog_memory_usage" ]; then
        local memory_usage=$(cat "/tmp/pycog_memory_usage")
        if (( $(awk "BEGIN {print ($memory_usage > $ALERT_THRESHOLD_MEMORY)}") )); then
            send_alert "High Memory Usage" "Memory usage is ${memory_usage}% (threshold: ${ALERT_THRESHOLD_MEMORY}%)"
            ((alerts_sent++))
        fi
    fi
    
    # Disk usage alert
    if [ -f "/tmp/pycog_disk_usage" ]; then
        local disk_usage=$(cat "/tmp/pycog_disk_usage")
        if (( $(awk "BEGIN {print ($disk_usage > $ALERT_THRESHOLD_DISK)}") )); then
            send_alert "High Disk Usage" "Disk usage is ${disk_usage}% (threshold: ${ALERT_THRESHOLD_DISK}%)"
            ((alerts_sent++))
        fi
    fi
    
    # Service status alert
    if [ -f "/tmp/pycog_service_status" ]; then
        local service_status=$(cat "/tmp/pycog_service_status")
        if [ "$service_status" != "running" ]; then
            send_alert "Service Down" "PyCog-Zero service is not running"
            ((alerts_sent++))
        fi
    fi
    
    # HTTP health alert
    if [ -f "/tmp/pycog_http_status" ]; then
        local http_status=$(cat "/tmp/pycog_http_status")
        if [ "$http_status" != "200" ]; then
            send_alert "Health Check Failed" "HTTP health check returned status: $http_status"
            ((alerts_sent++))
        fi
    fi
    
    if [ $alerts_sent -eq 0 ]; then
        print_status "No alerts triggered"
    else
        print_warning "$alerts_sent alerts sent"
    fi
}

continuous_monitor() {
    print_header "Starting Continuous Monitoring"
    print_info "Monitoring interval: ${MONITOR_INTERVAL} seconds"
    print_info "Press Ctrl+C to stop"
    
    # Trap interrupt signal
    trap 'print_info "Monitoring stopped"; exit 0' INT
    
    while true; do
        echo ""
        echo "=== Monitoring Check at $(get_timestamp) ==="
        
        get_system_status
        get_application_status
        check_health
        check_alerts
        
        sleep "$MONITOR_INTERVAL"
    done
}

# Execute command
case $COMMAND in
    status)
        get_system_status
        echo ""
        get_application_status
        ;;
    health)
        check_health
        echo ""
        check_cognitive_status
        ;;
    metrics)
        get_system_status
        echo ""
        get_application_status
        echo ""
        check_health
        ;;
    monitor)
        continuous_monitor
        ;;
    alerts)
        get_system_status > /dev/null
        get_application_status > /dev/null
        check_health > /dev/null
        check_alerts
        ;;
    logs)
        analyze_logs
        ;;
    performance)
        run_performance_diagnostics
        ;;
    cognitive)
        check_cognitive_status
        ;;
    *)
        print_error "Command not implemented: $COMMAND"
        exit 1
        ;;
esac

echo ""
print_status "Monitoring task completed successfully!"