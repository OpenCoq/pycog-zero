"""
PyCog-Zero Performance Optimizer
================================

Provides comprehensive performance optimization for large-scale cognitive processing.
Includes batch processing, memory management, parallel execution, and performance monitoring.
"""

import asyncio
import time
import threading
import multiprocessing
import psutil
import json
import gc
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import weakref


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    batch_efficiency: float = 0.0
    parallel_speedup: float = 0.0


class MemoryPool:
    """Memory pool for efficient reuse of large objects."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._pools = defaultdict(deque)
        self._lock = threading.RLock()
    
    def get(self, obj_type: str, factory: Callable = None):
        """Get an object from the pool or create a new one."""
        with self._lock:
            pool = self._pools[obj_type]
            if pool:
                return pool.popleft()
            elif factory:
                return factory()
            else:
                return None
    
    def put(self, obj_type: str, obj):
        """Return an object to the pool."""
        with self._lock:
            pool = self._pools[obj_type]
            if len(pool) < self.max_size:
                pool.append(obj)
    
    def clear(self, obj_type: str = None):
        """Clear the pool or a specific object type."""
        with self._lock:
            if obj_type:
                self._pools[obj_type].clear()
            else:
                self._pools.clear()


class CachingSystem:
    """High-performance caching system with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_order = deque()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
    
    def get(self, key: str) -> Any:
        """Get item from cache."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Check TTL
                if current_time - self._timestamps[key] <= self.ttl_seconds:
                    # Update access order
                    self._access_order.remove(key)
                    self._access_order.append(key)
                    self._hit_count += 1
                    return self._cache[key]
                else:
                    # Expired - remove
                    self._remove_key(key)
            
            self._miss_count += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Update existing
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                # Evict LRU
                lru_key = self._access_order.popleft()
                self._remove_key(lru_key)
            
            self._cache[key] = value
            self._timestamps[key] = current_time
            self._access_order.append(key)
    
    def _remove_key(self, key: str):
        """Remove key from all structures."""
        del self._cache[key]
        del self._timestamps[key]
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._timestamps.clear()
            self._hit_count = 0
            self._miss_count = 0


class BatchProcessor:
    """Efficient batch processing for cognitive operations."""
    
    def __init__(self, batch_size: int = 50, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self._pending_items = []
        self._batch_handlers = {}
        self._lock = threading.RLock()
        self._last_batch_time = time.time()
    
    def register_handler(self, operation_type: str, handler: Callable):
        """Register a batch handler for an operation type."""
        self._batch_handlers[operation_type] = handler
    
    async def add_item(self, operation_type: str, item: Any) -> Any:
        """Add item to batch and process when ready."""
        with self._lock:
            self._pending_items.append((operation_type, item, asyncio.Event()))
            
            # Check if we should process now
            should_process = (
                len(self._pending_items) >= self.batch_size or
                time.time() - self._last_batch_time >= self.max_wait_time
            )
        
        if should_process:
            return await self._process_batch()
        else:
            # Wait for batch to be processed
            _, _, event = self._pending_items[-1]
            await event.wait()
            return getattr(event, 'result', None)
    
    async def _process_batch(self):
        """Process the current batch."""
        with self._lock:
            if not self._pending_items:
                return
            
            batch = self._pending_items[:]
            self._pending_items.clear()
            self._last_batch_time = time.time()
        
        # Group by operation type
        grouped = defaultdict(list)
        for op_type, item, event in batch:
            grouped[op_type].append((item, event))
        
        # Process each group
        results = {}
        for op_type, items_and_events in grouped.items():
            if op_type in self._batch_handlers:
                items = [item for item, _ in items_and_events]
                handler_results = await self._batch_handlers[op_type](items)
                
                # Set results on events
                for (item, event), result in zip(items_and_events, handler_results):
                    event.result = result
                    event.set()
                
                results[op_type] = handler_results
        
        return results


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics = PerformanceMetrics()
        self.memory_pool = MemoryPool(
            max_size=self.config.get('memory_pool_size', 100)
        )
        self.cache = CachingSystem(
            max_size=self.config.get('cache_size', 1000),
            ttl_seconds=self.config.get('cache_ttl', 3600)
        )
        self.batch_processor = BatchProcessor(
            batch_size=self.config.get('batch_size', 50),
            max_wait_time=self.config.get('batch_wait_time', 1.0)
        )
        
        self._executor_pool = ThreadPoolExecutor(
            max_workers=self.config.get('thread_pool_size', min(32, multiprocessing.cpu_count() + 4))
        )
        self._process_pool = ProcessPoolExecutor(
            max_workers=self.config.get('process_pool_size', multiprocessing.cpu_count())
        )
        
        self._performance_history = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._monitoring_active = False
        self._monitor_thread = None
        
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_performance(self):
        """Monitor system performance metrics."""
        while self._monitoring_active:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                with self._lock:
                    self.metrics.peak_memory_mb = max(
                        self.metrics.peak_memory_mb,
                        memory_info.rss / 1024 / 1024
                    )
                    self.metrics.cpu_usage_percent = cpu_percent
                    self.metrics.cache_hit_rate = self.cache.get_hit_rate()
                
                time.sleep(1.0)
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                break
    
    async def optimize_operation(self, operation: Callable, *args, use_cache: bool = True, 
                                use_batching: bool = False, operation_type: str = "default", **kwargs) -> Any:
        """Optimize a single operation with caching, pooling, and monitoring."""
        start_time = time.time()
        cache_key = None
        
        try:
            # Try cache first
            if use_cache:
                cache_key = self._generate_cache_key(operation_type, args, kwargs)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Use batching if requested
            if use_batching:
                return await self.batch_processor.add_item(operation_type, (args, kwargs))
            
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            # Cache result
            if use_cache and cache_key:
                self.cache.put(cache_key, result)
            
            # Update metrics
            self._update_metrics(time.time() - start_time)
            
            return result
            
        except Exception as e:
            print(f"Operation optimization error: {e}")
            raise
    
    async def optimize_parallel_operations(self, operations: List[Callable], 
                                         use_threads: bool = True, max_workers: int = None) -> List[Any]:
        """Execute operations in parallel with optimization."""
        start_time = time.time()
        
        try:
            if use_threads:
                executor = self._executor_pool
            else:
                executor = self._process_pool
            
            if max_workers:
                # Create temporary executor with specific worker count
                executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
                with executor_class(max_workers=max_workers) as temp_executor:
                    futures = [temp_executor.submit(op) for op in operations]
                    results = [future.result() for future in as_completed(futures)]
            else:
                # Use existing executor
                loop = asyncio.get_event_loop()
                futures = [loop.run_in_executor(executor, op) for op in operations]
                results = await asyncio.gather(*futures)
            
            # Calculate parallel speedup
            total_time = time.time() - start_time
            sequential_estimate = len(operations) * 0.1  # Rough estimate
            speedup = sequential_estimate / total_time if total_time > 0 else 1.0
            
            with self._lock:
                self.metrics.parallel_speedup = speedup
                self.metrics.batch_efficiency = len(operations) / total_time if total_time > 0 else 0
            
            return results
            
        except Exception as e:
            print(f"Parallel optimization error: {e}")
            raise
    
    def _generate_cache_key(self, operation_type: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from operation parameters."""
        key_parts = [operation_type]
        
        # Add args (convert complex types to strings)
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(type(arg).__name__))
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            else:
                key_parts.append(f"{k}={type(v).__name__}")
        
        return "|".join(key_parts)
    
    def _update_metrics(self, operation_time: float):
        """Update performance metrics."""
        with self._lock:
            self.metrics.operation_count += 1
            self.metrics.total_time += operation_time
            self.metrics.avg_time = self.metrics.total_time / self.metrics.operation_count
            
            self._performance_history.append({
                'time': time.time(),
                'operation_time': operation_time,
                'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self._lock:
            recent_operations = list(self._performance_history)[-100:]  # Last 100 operations
            
            return {
                'metrics': {
                    'operation_count': self.metrics.operation_count,
                    'total_time': self.metrics.total_time,
                    'avg_time': self.metrics.avg_time,
                    'peak_memory_mb': self.metrics.peak_memory_mb,
                    'cpu_usage_percent': self.metrics.cpu_usage_percent,
                    'cache_hit_rate': self.metrics.cache_hit_rate,
                    'batch_efficiency': self.metrics.batch_efficiency,
                    'parallel_speedup': self.metrics.parallel_speedup
                },
                'system': {
                    'cpu_count': multiprocessing.cpu_count(),
                    'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                    'cache_size': len(self.cache._cache),
                    'pool_sizes': {k: len(v) for k, v in self.memory_pool._pools.items()}
                },
                'recent_performance': recent_operations,
                'optimization_config': self.config
            }
    
    def tune_parameters(self, target_latency: float = 0.1, target_memory_mb: float = 1000):
        """Auto-tune optimization parameters based on targets."""
        current_metrics = self.get_performance_report()['metrics']
        
        # Tune cache size
        if current_metrics['avg_time'] > target_latency and current_metrics['cache_hit_rate'] < 0.8:
            new_cache_size = min(self.cache.max_size * 2, 5000)
            print(f"Increasing cache size from {self.cache.max_size} to {new_cache_size}")
            self.cache.max_size = new_cache_size
        
        # Tune batch size
        if current_metrics['batch_efficiency'] < 10:  # Operations per second
            new_batch_size = min(self.batch_processor.batch_size * 2, 200)
            print(f"Increasing batch size from {self.batch_processor.batch_size} to {new_batch_size}")
            self.batch_processor.batch_size = new_batch_size
        
        # Tune memory pool
        if current_metrics['peak_memory_mb'] > target_memory_mb:
            print("Clearing memory pools to reduce memory usage")
            self.memory_pool.clear()
            gc.collect()
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_monitoring()
        self.cache.clear()
        self.memory_pool.clear()
        self._executor_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)


# Global performance optimizer instance
_global_optimizer = None


def get_performance_optimizer(config: Dict[str, Any] = None) -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(config)
    return _global_optimizer


def optimize(operation: Callable, *args, **kwargs):
    """Decorator/function to optimize any operation."""
    optimizer = get_performance_optimizer()
    
    if asyncio.iscoroutinefunction(operation):
        async def optimized_async(*a, **kw):
            return await optimizer.optimize_operation(operation, *a, **kw)
        return optimized_async
    else:
        def optimized_sync(*a, **kw):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    optimizer.optimize_operation(operation, *a, **kw)
                )
            finally:
                loop.close()
        return optimized_sync


# Example usage and testing
if __name__ == "__main__":
    async def test_optimizer():
        optimizer = PerformanceOptimizer({
            'cache_size': 100,
            'batch_size': 10,
            'thread_pool_size': 4
        })
        
        # Test operation
        def sample_operation(x):
            time.sleep(0.01)  # Simulate work
            return x * 2
        
        # Test optimization
        result = await optimizer.optimize_operation(
            sample_operation, 42, use_cache=True, operation_type="math"
        )
        print(f"Optimized result: {result}")
        
        # Test parallel operations
        operations = [lambda: sample_operation(i) for i in range(10)]
        parallel_results = await optimizer.optimize_parallel_operations(operations)
        print(f"Parallel results: {parallel_results}")
        
        # Get performance report
        report = optimizer.get_performance_report()
        print(f"Performance report: {json.dumps(report, indent=2)}")
        
        optimizer.cleanup()
    
    asyncio.run(test_optimizer())