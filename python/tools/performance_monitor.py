"""
PyCog-Zero Performance Monitor and Benchmarking Tool
===================================================

Provides comprehensive performance monitoring, benchmarking, and optimization
for large-scale cognitive processing operations.
"""

import asyncio
import time
import json
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import weakref
from collections import defaultdict, deque

from python.helpers.performance_optimizer import get_performance_optimizer, PerformanceOptimizer
from python.helpers import files


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    throughput_ops_per_sec: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any]


class PerformanceMonitor:
    """Comprehensive performance monitoring for cognitive operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_performance_config()
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)
        self.operation_metrics = defaultdict(list)
        self.alerts = []
        self.performance_optimizer = get_performance_optimizer(self.config.get('optimization', {}))
        
        # Monitoring intervals
        self.monitoring_interval = self.config.get('monitoring', {}).get('interval_seconds', 60)
        self.alert_thresholds = self.config.get('monitoring', {}).get('alert_thresholds', {})
        
        # Tools being monitored
        self.monitored_tools = {}
        
    def _load_performance_config(self) -> Dict[str, Any]:
        """Load performance configuration."""
        try:
            config_path = files.get_abs_path("conf/performance_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('performance_optimization', {})
        except Exception as e:
            print(f"Warning: Could not load performance config: {e}")
            return {}
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("✓ Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("✓ Performance monitoring stopped")
    
    def register_tool(self, tool_name: str, tool_instance: Any):
        """Register a tool for monitoring."""
        self.monitored_tools[tool_name] = weakref.ref(tool_instance)
        print(f"✓ Registered {tool_name} for performance monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self._collect_tool_metrics()
                self._check_alerts(metrics)
                self.metrics_history.append(metrics)
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level performance metrics."""
        process = psutil.Process()
        memory = process.memory_info()
        
        metrics = {
            'timestamp': time.time(),
            'memory_usage_mb': memory.rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'cpu_usage_percent': process.cpu_percent(),
            'threads': process.num_threads(),
            'open_files': len(process.open_files()) if hasattr(process, 'open_files') else 0,
            'system_memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'system_cpu_usage_percent': psutil.cpu_percent()
        }
        
        # Add performance optimizer metrics if available
        if self.performance_optimizer:
            optimizer_report = self.performance_optimizer.get_performance_report()
            metrics.update({
                'optimizer_operations': optimizer_report['metrics']['operation_count'],
                'optimizer_avg_time': optimizer_report['metrics']['avg_time'],
                'cache_hit_rate': optimizer_report['metrics']['cache_hit_rate'],
                'batch_efficiency': optimizer_report['metrics']['batch_efficiency'],
                'parallel_speedup': optimizer_report['metrics']['parallel_speedup']
            })
        
        return metrics
    
    def _collect_tool_metrics(self):
        """Collect metrics from monitored tools."""
        for tool_name, tool_ref in list(self.monitored_tools.items()):
            tool = tool_ref()
            if tool is None:
                # Tool was garbage collected
                del self.monitored_tools[tool_name]
                continue
                
            try:
                # Try to get tool-specific metrics
                if hasattr(tool, 'operation_stats'):
                    stats = tool.operation_stats
                    self.operation_metrics[tool_name].append({
                        'timestamp': time.time(),
                        'total_operations': stats.get('total_operations', 0),
                        'cached_operations': stats.get('cached_operations', 0),
                        'batch_operations': stats.get('batch_operations', 0),
                        'avg_response_time': stats.get('avg_response_time', 0.0)
                    })
                elif hasattr(tool, 'memory_stats'):
                    stats = tool.memory_stats
                    self.operation_metrics[tool_name].append({
                        'timestamp': time.time(),
                        'total_operations': stats.get('total_operations', 0),
                        'storage_operations': stats.get('storage_operations', 0),
                        'avg_response_time': stats.get('avg_response_time', 0.0)
                    })
            except Exception as e:
                print(f"Error collecting metrics from {tool_name}: {e}")
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for performance alerts."""
        alerts_triggered = []
        
        # Check memory usage
        if metrics['memory_usage_mb'] > self.alert_thresholds.get('memory_usage_mb', 1500):
            alerts_triggered.append({
                'type': 'high_memory_usage',
                'value': metrics['memory_usage_mb'],
                'threshold': self.alert_thresholds.get('memory_usage_mb', 1500),
                'timestamp': time.time()
            })
        
        # Check response time
        avg_time_ms = metrics.get('optimizer_avg_time', 0) * 1000
        if avg_time_ms > self.alert_thresholds.get('avg_response_time_ms', 500):
            alerts_triggered.append({
                'type': 'high_response_time',
                'value': avg_time_ms,
                'threshold': self.alert_thresholds.get('avg_response_time_ms', 500),
                'timestamp': time.time()
            })
        
        # Check cache hit rate
        cache_hit_rate = metrics.get('cache_hit_rate', 1.0)
        if cache_hit_rate < self.alert_thresholds.get('cache_hit_rate', 0.7):
            alerts_triggered.append({
                'type': 'low_cache_hit_rate',
                'value': cache_hit_rate,
                'threshold': self.alert_thresholds.get('cache_hit_rate', 0.7),
                'timestamp': time.time()
            })
        
        if alerts_triggered:
            self.alerts.extend(alerts_triggered)
            if self.config.get('monitoring', {}).get('print_alerts', True):
                for alert in alerts_triggered:
                    print(f"⚠️ Performance Alert: {alert['type']} = {alert['value']} (threshold: {alert['threshold']})")
    
    def get_performance_report(self, detailed: bool = False) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.metrics_history:
            return {'status': 'no_data', 'message': 'No performance data collected yet'}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 samples
        
        # Calculate averages and trends
        avg_memory = sum(m['memory_usage_mb'] for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m['cpu_usage_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit = sum(m.get('cache_hit_rate', 0) for m in recent_metrics) / len(recent_metrics)
        
        report = {
            'summary': {
                'monitoring_active': self.monitoring_active,
                'samples_collected': len(self.metrics_history),
                'monitoring_duration_hours': (time.time() - recent_metrics[0]['timestamp']) / 3600 if recent_metrics else 0,
                'tools_monitored': len(self.monitored_tools)
            },
            'current_performance': {
                'avg_memory_usage_mb': avg_memory,
                'avg_cpu_usage_percent': avg_cpu,
                'avg_cache_hit_rate': avg_cache_hit,
                'active_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 3600])
            },
            'optimization_status': self.performance_optimizer.get_performance_report() if self.performance_optimizer else None,
            'tool_metrics': dict(self.operation_metrics) if detailed else {
                k: len(v) for k, v in self.operation_metrics.items()
            },
            'recent_alerts': self.alerts[-10:] if detailed else len(self.alerts)
        }
        
        if detailed:
            report['detailed_metrics'] = recent_metrics[-20:]  # Last 20 samples
        
        return report
    
    def benchmark_operation(self, operation_name: str, operation_func: Callable, 
                           iterations: int = 100, **kwargs) -> BenchmarkResult:
        """Benchmark a specific operation."""
        print(f"Running benchmark for {operation_name} ({iterations} iterations)...")
        
        results = []
        errors = 0
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Warmup
        try:
            for _ in range(min(10, iterations // 10)):
                operation_func(**kwargs)
        except:
            pass  # Ignore warmup errors
        
        # Actual benchmarking
        start_time = time.time()
        for i in range(iterations):
            iteration_start = time.time()
            try:
                if asyncio.iscoroutinefunction(operation_func):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(operation_func(**kwargs))
                    finally:
                        loop.close()
                else:
                    result = operation_func(**kwargs)
                
                iteration_time = (time.time() - iteration_start) * 1000  # Convert to ms
                results.append(iteration_time)
                
            except Exception as e:
                errors += 1
                if errors > iterations * 0.5:  # Too many errors
                    print(f"❌ Benchmark failed: too many errors ({errors}/{i+1})")
                    break
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        if results:
            avg_duration = sum(results) / len(results)
            throughput = len(results) / (end_time - start_time)
            success_rate = len(results) / iterations
        else:
            avg_duration = 0
            throughput = 0
            success_rate = 0
        
        benchmark_result = BenchmarkResult(
            operation=operation_name,
            duration_ms=avg_duration,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=psutil.Process().cpu_percent(),
            cache_hit_rate=self.performance_optimizer.cache.get_hit_rate() if self.performance_optimizer else 0,
            throughput_ops_per_sec=throughput,
            success_rate=success_rate,
            error_count=errors,
            metadata={
                'iterations': iterations,
                'total_time_seconds': end_time - start_time,
                'min_duration_ms': min(results) if results else 0,
                'max_duration_ms': max(results) if results else 0
            }
        )
        
        print(f"✓ Benchmark completed: {avg_duration:.2f}ms avg, {throughput:.1f} ops/sec")
        return benchmark_result
    
    async def run_comprehensive_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run comprehensive benchmarks on all cognitive operations."""
        print("Starting comprehensive cognitive performance benchmark...")
        
        benchmarks = {}
        
        # Test operations if tools are available
        try:
            from python.tools.cognitive_reasoning import CognitiveReasoningTool
            from python.tools.cognitive_memory import CognitiveMemoryTool
            
            # Create tool instances for benchmarking
            class MockAgent:
                def __init__(self):
                    pass
            
            mock_agent = MockAgent()
            
            # Benchmark cognitive reasoning
            reasoning_tool = CognitiveReasoningTool(agent=mock_agent, name="reasoning", args={})
            
            def reasoning_operation():
                return asyncio.run(reasoning_tool._perform_reasoning("test reasoning query"))
            
            benchmarks['cognitive_reasoning'] = self.benchmark_operation(
                "cognitive_reasoning", reasoning_operation, iterations=50
            )
            
            # Benchmark cognitive memory
            memory_tool = CognitiveMemoryTool(agent=mock_agent, name="memory", args={})
            
            def memory_operation():
                return asyncio.run(memory_tool.execute("store", {"content": "test content", "type": "concept"}))
            
            benchmarks['cognitive_memory'] = self.benchmark_operation(
                "cognitive_memory", memory_operation, iterations=50
            )
            
            print("✓ Comprehensive benchmark completed")
            
        except Exception as e:
            print(f"⚠️ Benchmark error: {e}")
            benchmarks['error'] = str(e)
        
        return benchmarks
    
    def save_performance_report(self, filepath: str = None):
        """Save performance report to file."""
        if filepath is None:
            filepath = files.get_abs_path("performance_report.json")
        
        report = self.get_performance_report(detailed=True)
        report['generated_at'] = time.time()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"✓ Performance report saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save performance report: {e}")
    
    def cleanup(self):
        """Cleanup monitoring resources."""
        self.stop_monitoring()
        if self.performance_optimizer:
            self.performance_optimizer.cleanup()


# Global performance monitor instance
_global_monitor = None


def get_performance_monitor(config: Dict[str, Any] = None) -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(config)
    return _global_monitor


# Performance monitoring decorators
def monitor_performance(operation_name: str = None):
    """Decorator to monitor performance of functions/methods."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    if hasattr(monitor, 'operation_metrics'):
                        monitor.operation_metrics[operation_name].append({
                            'timestamp': time.time(),
                            'duration': duration,
                            'success': True
                        })
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    if hasattr(monitor, 'operation_metrics'):
                        monitor.operation_metrics[operation_name].append({
                            'timestamp': time.time(),
                            'duration': duration,
                            'success': True
                        })
            return sync_wrapper
    
    return decorator


if __name__ == "__main__":
    async def test_performance_monitor():
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Wait for some metrics
        await asyncio.sleep(2)
        
        # Run a simple benchmark
        def test_operation():
            time.sleep(0.01)  # Simulate work
            return "test result"
        
        result = monitor.benchmark_operation("test_op", test_operation, iterations=10)
        print(f"Benchmark result: {result}")
        
        # Get performance report
        report = monitor.get_performance_report(detailed=True)
        print(f"Performance report: {json.dumps(report, indent=2, default=str)}")
        
        monitor.cleanup()
    
    asyncio.run(test_performance_monitor())