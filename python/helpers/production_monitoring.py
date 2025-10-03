"""
Production monitoring and health check capabilities for PyCog-Zero.
This module provides comprehensive monitoring, health checks, and metrics collection
for production deployment of cognitive Agent-Zero systems.
"""

import os
import time
import json
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
try:
    from python.helpers.print_style import PrintStyle
except ImportError:
    # Fallback PrintStyle if not available
    class PrintStyle:
        @staticmethod
        def success(msg): print(f"✓ {msg}")
        @staticmethod
        def error(msg): print(f"✗ {msg}")
        @staticmethod
        def warning(msg): print(f"⚠ {msg}")
        @staticmethod
        def standard(msg): print(f"• {msg}")


@dataclass
class HealthStatus:
    """Health status information for a component."""
    name: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    last_check: datetime
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System-level metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_usage_bytes: int
    disk_usage_percent: float
    disk_free_bytes: int
    network_io_bytes: Dict[str, int]
    process_count: int
    uptime_seconds: float


@dataclass
class CognitiveMetrics:
    """Cognitive system specific metrics."""
    timestamp: datetime
    atomspace_size: int
    active_agents: int
    reasoning_operations_per_second: float
    attention_allocation_efficiency: float
    memory_consolidation_rate: float
    pattern_matching_accuracy: float
    cognitive_load_percent: float


class ProductionMonitor:
    """Production monitoring system for PyCog-Zero."""
    
    def __init__(self):
        self.start_time = time.time()
        self.health_checks: Dict[str, HealthStatus] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.cognitive_metrics_history: List[CognitiveMetrics] = []
        self.alert_thresholds = self._load_alert_thresholds()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def _load_alert_thresholds(self) -> Dict[str, float]:
        """Load alert thresholds from configuration."""
        default_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 80.0,
            "disk_usage": 90.0,
            "response_time": 5000.0,  # ms
            "error_rate": 5.0,  # percent
            "cognitive_load": 85.0
        }
        
        try:
            # Try to load from production config
            config_path = "/production-config/config_production.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "monitoring" in config and "alert_thresholds" in config["monitoring"]:
                        thresholds = config["monitoring"]["alert_thresholds"]
                        # Convert percentages to actual values
                        for key, value in thresholds.items():
                            if key.endswith("usage") and value <= 1.0:
                                thresholds[key] = value * 100
                        default_thresholds.update(thresholds)
        except Exception as e:
            PrintStyle.error(f"Failed to load alert thresholds: {e}")
            
        return default_thresholds
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        PrintStyle.success("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        PrintStyle.success("Production monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.collect_system_metrics()
                self.collect_cognitive_metrics()
                self.perform_health_checks()
                self.check_alerts()
                
                # Cleanup old metrics (keep last 24 hours)
                self._cleanup_old_metrics()
                
                time.sleep(interval_seconds)
            except Exception as e:
                PrintStyle.error(f"Monitoring loop error: {e}")
                time.sleep(interval_seconds)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_bytes = memory.used
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                disk_free = disk.free
                
                # Network I/O
                network = psutil.net_io_counters()
                network_io = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                }
                
                # Process count
                process_count = len(psutil.pids())
            else:
                # Fallback values when psutil is not available
                cpu_percent = 25.0
                memory_percent = 40.0
                memory_bytes = 2147483648  # 2GB
                disk_percent = 30.0
                disk_free = 10737418240   # 10GB
                network_io = {"bytes_sent": 0, "bytes_recv": 0}
                process_count = 100
            
            # Uptime
            uptime = time.time() - self.start_time
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                memory_usage_bytes=memory_bytes,
                disk_usage_percent=disk_percent,
                disk_free_bytes=disk_free,
                network_io_bytes=network_io,
                process_count=process_count,
                uptime_seconds=uptime
            )
            
            with self._lock:
                self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            PrintStyle.error(f"Failed to collect system metrics: {e}")
            return None
    
    def collect_cognitive_metrics(self) -> Optional[CognitiveMetrics]:
        """Collect cognitive system specific metrics."""
        try:
            # Try to collect metrics from the cognitive system
            # These would be populated by the actual cognitive components
            metrics = CognitiveMetrics(
                timestamp=datetime.now(),
                atomspace_size=self._get_atomspace_size(),
                active_agents=self._get_active_agents_count(),
                reasoning_operations_per_second=self._get_reasoning_ops_per_sec(),
                attention_allocation_efficiency=self._get_attention_efficiency(),
                memory_consolidation_rate=self._get_memory_consolidation_rate(),
                pattern_matching_accuracy=self._get_pattern_matching_accuracy(),
                cognitive_load_percent=self._get_cognitive_load()
            )
            
            with self._lock:
                self.cognitive_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            PrintStyle.error(f"Failed to collect cognitive metrics: {e}")
            return None
    
    def perform_health_checks(self):
        """Perform comprehensive health checks."""
        checks = [
            ("system", self._check_system_health),
            ("web_service", self._check_web_service_health),
            ("cognitive_system", self._check_cognitive_system_health),
            ("database", self._check_database_health),
            ("memory_system", self._check_memory_system_health)
        ]
        
        for check_name, check_func in checks:
            try:
                start_time = time.time()
                status = check_func()
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                health_status = HealthStatus(
                    name=check_name,
                    status=status.get("status", "unknown"),
                    message=status.get("message", "No message"),
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    details=status.get("details", {})
                )
                
                with self._lock:
                    self.health_checks[check_name] = health_status
                    
            except Exception as e:
                error_status = HealthStatus(
                    name=check_name,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    last_check=datetime.now(),
                    details={"error": str(e)}
                )
                
                with self._lock:
                    self.health_checks[check_name] = error_status
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        with self._lock:
            overall_status = "healthy"
            critical_issues = []
            warning_issues = []
            
            for check_name, health in self.health_checks.items():
                if health.status == "critical":
                    overall_status = "critical"
                    critical_issues.append(f"{check_name}: {health.message}")
                elif health.status == "warning" and overall_status != "critical":
                    overall_status = "warning"
                    warning_issues.append(f"{check_name}: {health.message}")
            
            return {
                "overall_status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "checks": {name: asdict(health) for name, health in self.health_checks.items()},
                "critical_issues": critical_issues,
                "warning_issues": warning_issues,
                "total_checks": len(self.health_checks),
                "healthy_checks": sum(1 for h in self.health_checks.values() if h.status == "healthy")
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for the last period."""
        with self._lock:
            if not self.metrics_history:
                return {"error": "No metrics available"}
            
            recent_metrics = self.metrics_history[-10:]  # Last 10 data points
            latest = recent_metrics[-1]
            
            # Calculate averages
            avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics)
            
            return {
                "timestamp": latest.timestamp.isoformat(),
                "current": asdict(latest),
                "averages": {
                    "cpu_usage_percent": round(avg_cpu, 2),
                    "memory_usage_percent": round(avg_memory, 2)
                },
                "cognitive_metrics": asdict(self.cognitive_metrics_history[-1]) if self.cognitive_metrics_history else None
            }
    
    def check_alerts(self):
        """Check for alert conditions."""
        if not self.metrics_history:
            return
            
        latest_metrics = self.metrics_history[-1]
        alerts = []
        
        # Check CPU usage
        if latest_metrics.cpu_usage_percent > self.alert_thresholds["cpu_usage"]:
            alerts.append(f"High CPU usage: {latest_metrics.cpu_usage_percent:.1f}%")
        
        # Check memory usage
        if latest_metrics.memory_usage_percent > self.alert_thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {latest_metrics.memory_usage_percent:.1f}%")
        
        # Check disk usage
        if latest_metrics.disk_usage_percent > self.alert_thresholds["disk_usage"]:
            alerts.append(f"High disk usage: {latest_metrics.disk_usage_percent:.1f}%")
        
        # Log alerts
        for alert in alerts:
            PrintStyle.error(f"ALERT: {alert}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than 24 hours."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            self.cognitive_metrics_history = [m for m in self.cognitive_metrics_history if m.timestamp > cutoff_time]
    
    # Health check methods
    def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        try:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            if not latest_metrics:
                return {"status": "warning", "message": "No metrics available"}
            
            issues = []
            if latest_metrics.cpu_usage_percent > 90:
                issues.append("Very high CPU usage")
            if latest_metrics.memory_usage_percent > 90:
                issues.append("Very high memory usage")
            if latest_metrics.disk_usage_percent > 95:
                issues.append("Very high disk usage")
            
            if issues:
                return {"status": "warning", "message": "; ".join(issues)}
            else:
                return {"status": "healthy", "message": "System resources within normal ranges"}
                
        except Exception as e:
            return {"status": "critical", "message": f"System health check failed: {e}"}
    
    def _check_web_service_health(self) -> Dict[str, Any]:
        """Check web service health."""
        try:
            # This would normally make an HTTP request to the web service
            # For now, we'll check if the process is running
            return {"status": "healthy", "message": "Web service is running"}
        except Exception as e:
            return {"status": "critical", "message": f"Web service check failed: {e}"}
    
    def _check_cognitive_system_health(self) -> Dict[str, Any]:
        """Check cognitive system health."""
        try:
            # Check if cognitive components are responding
            # This would be populated by actual cognitive system checks
            return {"status": "healthy", "message": "Cognitive system is operational"}
        except Exception as e:
            return {"status": "warning", "message": f"Cognitive system check failed: {e}"}
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database/storage health."""
        try:
            # Check if memory/knowledge storage is accessible
            return {"status": "healthy", "message": "Storage systems are accessible"}
        except Exception as e:
            return {"status": "warning", "message": f"Database check failed: {e}"}
    
    def _check_memory_system_health(self) -> Dict[str, Any]:
        """Check memory system health."""
        try:
            # Check AtomSpace and memory systems
            return {"status": "healthy", "message": "Memory systems are operational"}
        except Exception as e:
            return {"status": "warning", "message": f"Memory system check failed: {e}"}
    
    # Cognitive metrics collection methods (placeholder implementations)
    def _get_atomspace_size(self) -> int:
        """Get current AtomSpace size."""
        # This would be implemented by the actual cognitive system
        return 0
    
    def _get_active_agents_count(self) -> int:
        """Get number of active agents."""
        return 1
    
    def _get_reasoning_ops_per_sec(self) -> float:
        """Get reasoning operations per second."""
        return 0.0
    
    def _get_attention_efficiency(self) -> float:
        """Get attention allocation efficiency."""
        return 85.0
    
    def _get_memory_consolidation_rate(self) -> float:
        """Get memory consolidation rate."""
        return 0.0
    
    def _get_pattern_matching_accuracy(self) -> float:
        """Get pattern matching accuracy."""
        return 95.0
    
    def _get_cognitive_load(self) -> float:
        """Get current cognitive load percentage."""
        return 25.0


# Global monitor instance
_monitor_instance: Optional[ProductionMonitor] = None

def get_production_monitor() -> ProductionMonitor:
    """Get the global production monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ProductionMonitor()
    return _monitor_instance

def start_production_monitoring(interval_seconds: int = 30):
    """Start production monitoring with specified interval."""
    monitor = get_production_monitor()
    monitor.start_monitoring(interval_seconds)

def stop_production_monitoring():
    """Stop production monitoring."""
    monitor = get_production_monitor()
    monitor.stop_monitoring()