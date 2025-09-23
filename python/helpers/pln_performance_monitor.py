"""
Real-time performance monitoring utility for PLN optimization.
Provides monitoring and profiling capabilities for PLN operations.
"""
import time
import json
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import threading


@dataclass
class PerformanceMetric:
    """Data class for performance metrics."""
    operation: str
    duration_ms: float
    timestamp: float
    success: bool
    cache_hit: bool = False
    atom_count: int = 0


class PLNPerformanceMonitor:
    """Real-time performance monitor for PLN operations."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics_history = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        # Aggregated statistics
        self._stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time": 0.0,
            "min_response_time": float('inf'),
            "max_response_time": 0.0
        }
    
    def record_operation(self, metric: PerformanceMetric):
        """Record a performance metric."""
        with self._lock:
            self._metrics_history.append(metric)
            self._update_stats(metric)
    
    def _update_stats(self, metric: PerformanceMetric):
        """Update aggregated statistics."""
        self._stats["total_operations"] += 1
        
        if metric.success:
            self._stats["successful_operations"] += 1
        else:
            self._stats["failed_operations"] += 1
            
        if metric.cache_hit:
            self._stats["cache_hits"] += 1
        else:
            self._stats["cache_misses"] += 1
        
        # Update response time statistics
        duration = metric.duration_ms
        self._stats["min_response_time"] = min(self._stats["min_response_time"], duration)
        self._stats["max_response_time"] = max(self._stats["max_response_time"], duration)
        
        # Update running average
        total_ops = self._stats["total_operations"]
        prev_avg = self._stats["avg_response_time"]
        self._stats["avg_response_time"] = (prev_avg * (total_ops - 1) + duration) / total_ops
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._lock:
            recent_metrics = list(self._metrics_history)[-100:]  # Last 100 operations
            
            stats = self._stats.copy()
            
            if recent_metrics:
                recent_durations = [m.duration_ms for m in recent_metrics if m.success]
                if recent_durations:
                    stats["recent_avg_response_time"] = statistics.mean(recent_durations)
                    stats["recent_median_response_time"] = statistics.median(recent_durations)
                    stats["recent_p95_response_time"] = statistics.quantiles(recent_durations, n=20)[18] if len(recent_durations) >= 20 else max(recent_durations)
            
            # Calculate rates
            if self._stats["total_operations"] > 0:
                stats["success_rate"] = self._stats["successful_operations"] / self._stats["total_operations"]
                stats["cache_hit_rate"] = self._stats["cache_hits"] / self._stats["total_operations"]
            else:
                stats["success_rate"] = 0.0
                stats["cache_hit_rate"] = 0.0
            
            # Uptime
            stats["uptime_seconds"] = time.time() - self._start_time
            
            return stats
    
    def get_performance_report(self, include_history: bool = False) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            stats = self.get_current_stats()
            report = {
                "timestamp": time.time(),
                "summary": stats,
                "recommendations": self._generate_recommendations(stats)
            }
            
            if include_history:
                report["history"] = [
                    {
                        "operation": m.operation,
                        "duration_ms": m.duration_ms,
                        "timestamp": m.timestamp,
                        "success": m.success,
                        "cache_hit": m.cache_hit,
                        "atom_count": m.atom_count
                    }
                    for m in self._metrics_history
                ]
            
            return report
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Response time recommendations
        if stats.get("avg_response_time", 0) > 1000:  # > 1 second
            recommendations.append("Average response time is high. Consider increasing cache size or optimizing reasoning depth.")
        
        # Cache hit rate recommendations  
        cache_hit_rate = stats.get("cache_hit_rate", 0)
        if cache_hit_rate < 0.3:
            recommendations.append("Low cache hit rate detected. Consider adjusting caching strategy or increasing cache size.")
        elif cache_hit_rate > 0.8:
            recommendations.append("Excellent cache performance! Consider reducing cache size to save memory.")
        
        # Success rate recommendations
        success_rate = stats.get("success_rate", 0)
        if success_rate < 0.95:
            recommendations.append("Operation success rate is below optimal. Check for recurring errors and improve error handling.")
        
        # Recent performance trends
        recent_avg = stats.get("recent_avg_response_time", 0)
        overall_avg = stats.get("avg_response_time", 0)
        
        if recent_avg > overall_avg * 1.5:
            recommendations.append("Recent performance degradation detected. Monitor system resources and consider scaling.")
        elif recent_avg < overall_avg * 0.7:
            recommendations.append("Recent performance improvement detected. Current optimizations are working well.")
        
        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges. System is operating optimally.")
        
        return recommendations
    
    def is_real_time_capable(self) -> bool:
        """Check if system is meeting real-time performance requirements."""
        stats = self.get_current_stats()
        
        # Real-time criteria:
        # - Average response time < 2 seconds
        # - 95th percentile < 3 seconds  
        # - Success rate > 95%
        # - Cache hit rate > 20%
        
        avg_time = stats.get("avg_response_time", float('inf'))
        p95_time = stats.get("recent_p95_response_time", float('inf'))
        success_rate = stats.get("success_rate", 0)
        cache_hit_rate = stats.get("cache_hit_rate", 0)
        
        return (
            avg_time < 2000 and  # 2 seconds
            p95_time < 3000 and  # 3 seconds
            success_rate > 0.95 and
            cache_hit_rate > 0.2
        )
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file for analysis."""
        report = self.get_performance_report(include_history=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def reset_metrics(self):
        """Reset all metrics and statistics."""
        with self._lock:
            self._metrics_history.clear()
            self._stats = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "avg_response_time": 0.0,
                "min_response_time": float('inf'),
                "max_response_time": 0.0
            }
            self._start_time = time.time()


class PLNPerformanceProfiler:
    """Context manager for profiling PLN operations."""
    
    def __init__(self, monitor: PLNPerformanceMonitor, operation: str, atom_count: int = 0):
        self.monitor = monitor
        self.operation = operation
        self.atom_count = atom_count
        self.start_time = None
        self.success = True
        self.cache_hit = False
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            
            # Record success/failure based on exception
            self.success = exc_type is None
            
            metric = PerformanceMetric(
                operation=self.operation,
                duration_ms=duration_ms,
                timestamp=time.time(),
                success=self.success,
                cache_hit=self.cache_hit,
                atom_count=self.atom_count
            )
            
            self.monitor.record_operation(metric)
    
    def mark_cache_hit(self):
        """Mark this operation as a cache hit."""
        self.cache_hit = True


# Global performance monitor instance
_global_monitor: Optional[PLNPerformanceMonitor] = None


def get_performance_monitor() -> PLNPerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PLNPerformanceMonitor()
    return _global_monitor


def profile_operation(operation: str, atom_count: int = 0) -> PLNPerformanceProfiler:
    """Create a performance profiler for an operation."""
    return PLNPerformanceProfiler(get_performance_monitor(), operation, atom_count)


def get_real_time_status() -> Dict[str, Any]:
    """Get current real-time capability status."""
    monitor = get_performance_monitor()
    stats = monitor.get_current_stats()
    
    return {
        "real_time_capable": monitor.is_real_time_capable(),
        "performance_summary": {
            "avg_response_time_ms": stats.get("avg_response_time", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0),
            "success_rate": stats.get("success_rate", 0),
            "total_operations": stats.get("total_operations", 0)
        },
        "recommendations": monitor._generate_recommendations(stats)
    }