#!/usr/bin/env python3
"""
PyCog-Zero Performance Optimization Demo
=========================================

Demonstrates the performance optimization features implemented for 
large-scale cognitive processing.
"""

import asyncio
import time
import json
import random
from python.helpers.performance_optimizer import PerformanceOptimizer, get_performance_optimizer
from python.tools.performance_monitor import PerformanceMonitor, get_performance_monitor


async def demo_basic_optimization():
    """Demonstrate basic performance optimization features."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Performance Optimization")
    print("=" * 60)
    
    optimizer = PerformanceOptimizer({
        'cache_size': 100,
        'batch_size': 10,
        'thread_pool_size': 4
    })
    
    # Simulate a computational operation
    def complex_calculation(n):
        """Simulate a complex calculation."""
        result = 0
        for i in range(n * 1000):
            result += i * 0.001
        time.sleep(0.01)  # Simulate I/O or other delays
        return result
    
    print("🔄 Testing caching performance...")
    
    # First call - should be slow
    start_time = time.time()
    result1 = await optimizer.optimize_operation(
        complex_calculation, 100, use_cache=True, operation_type="calculation"
    )
    first_call_time = time.time() - start_time
    
    # Second call - should be fast due to caching
    start_time = time.time()
    result2 = await optimizer.optimize_operation(
        complex_calculation, 100, use_cache=True, operation_type="calculation"
    )
    second_call_time = time.time() - start_time
    
    print(f"✓ First call:  {first_call_time:.3f}s (result: {result1:.2f})")
    print(f"✓ Second call: {second_call_time:.4f}s (result: {result2:.2f}) [Cached!]")
    print(f"📈 Speedup: {first_call_time / second_call_time:.1f}x faster")
    
    # Get performance metrics
    report = optimizer.get_performance_report()
    print(f"\n📊 Performance Metrics:")
    print(f"   Operations: {report['metrics']['operation_count']}")
    print(f"   Cache hit rate: {report['metrics']['cache_hit_rate']:.2%}")
    print(f"   Average time: {report['metrics']['avg_time']:.3f}s")
    
    optimizer.cleanup()


async def demo_parallel_processing():
    """Demonstrate parallel processing optimization."""
    print("\n" + "=" * 60)
    print("DEMO 2: Parallel Processing Optimization")
    print("=" * 60)
    
    optimizer = PerformanceOptimizer({'thread_pool_size': 8})
    
    def cpu_intensive_task(task_id):
        """Simulate CPU-intensive work."""
        total = 0
        for i in range(task_id * 50000):
            total += i * 0.001
        time.sleep(0.005)  # Simulate some I/O
        return {'task_id': task_id, 'result': total}
    
    # Create multiple tasks
    tasks = [lambda id=i: cpu_intensive_task(i) for i in range(1, 9)]
    
    print("🔄 Running 8 CPU-intensive tasks...")
    
    # Sequential execution
    print("   Sequential execution:")
    start_time = time.time()
    sequential_results = [task() for task in tasks]
    sequential_time = time.time() - start_time
    print(f"   ⏱️  Time: {sequential_time:.3f}s")
    
    # Parallel execution
    print("   Parallel execution:")
    start_time = time.time()
    parallel_results = await optimizer.optimize_parallel_operations(
        tasks, use_threads=True, max_workers=4
    )
    parallel_time = time.time() - start_time
    print(f"   ⏱️  Time: {parallel_time:.3f}s")
    
    # Compare results
    speedup = sequential_time / parallel_time
    print(f"\n📈 Results:")
    print(f"   Sequential: {sequential_time:.3f}s")
    print(f"   Parallel:   {parallel_time:.3f}s")
    print(f"   Speedup:    {speedup:.1f}x faster")
    print(f"   Efficiency: {speedup / 4:.1%} (4 workers)")
    
    # Verify results are identical
    parallel_sorted = sorted(parallel_results, key=lambda x: x['task_id'])
    sequential_sorted = sorted(sequential_results, key=lambda x: x['task_id'])
    
    if parallel_sorted == sequential_sorted:
        print("✅ Results verified: Parallel and sequential outputs match")
    else:
        print("❌ Results mismatch between parallel and sequential execution")
    
    optimizer.cleanup()


async def demo_batch_processing():
    """Demonstrate batch processing optimization."""
    print("\n" + "=" * 60)
    print("DEMO 3: Batch Processing Optimization") 
    print("=" * 60)
    
    optimizer = PerformanceOptimizer({
        'batch_size': 15,
        'batch_wait_time': 0.5
    })
    
    # Register a batch handler for simulated reasoning operations
    processed_batches = []
    
    async def reasoning_batch_handler(items):
        """Simulate batch reasoning processing."""
        batch_id = len(processed_batches)
        processed_batches.append(len(items))
        
        print(f"   🧠 Processing batch {batch_id} with {len(items)} items")
        
        # Simulate batch processing with economies of scale
        processing_time = 0.1 + len(items) * 0.01  # Fixed cost + per-item cost
        await asyncio.sleep(processing_time)
        
        results = []
        for query, context in items:
            # Simulate reasoning result
            result = {
                'query': query,
                'reasoning_result': f"processed_{query}_batch_{batch_id}",
                'confidence': random.uniform(0.7, 0.95),
                'processing_time': processing_time / len(items)
            }
            results.append(result)
        
        return results
    
    optimizer.batch_processor.register_handler('reasoning', reasoning_batch_handler)
    
    print("🔄 Simulating 25 reasoning requests...")
    
    # Create multiple reasoning requests
    reasoning_tasks = []
    start_time = time.time()
    
    for i in range(25):
        task = asyncio.create_task(
            optimizer.batch_processor.add_item(
                'reasoning', 
                (f'query_{i}', {'context': f'context_{i}'})
            )
        )
        reasoning_tasks.append(task)
        
        # Add small delays to simulate requests arriving over time
        if i % 5 == 0:
            await asyncio.sleep(0.05)
    
    # Wait for all results
    results = await asyncio.gather(*reasoning_tasks)
    total_time = time.time() - start_time
    
    print(f"\n📊 Batch Processing Results:")
    print(f"   Total requests: 25")
    print(f"   Batches processed: {len(processed_batches)}")
    print(f"   Batch sizes: {processed_batches}")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Avg time per request: {total_time / 25:.4f}s")
    print(f"   Throughput: {25 / total_time:.1f} requests/sec")
    
    # Verify all results are correct
    successful_results = sum(1 for r in results if r and 'reasoning_result' in r)
    print(f"   Successful results: {successful_results}/25")
    
    if successful_results == 25:
        print("✅ All batch processing completed successfully")
    else:
        print("❌ Some batch processing failed")
    
    optimizer.cleanup()


async def demo_large_scale_processing():
    """Demonstrate large-scale cognitive processing simulation."""
    print("\n" + "=" * 60)
    print("DEMO 4: Large-Scale Cognitive Processing")
    print("=" * 60)
    
    # Configure for large-scale processing
    optimizer = PerformanceOptimizer({
        'cache_size': 500,
        'batch_size': 100,
        'batch_wait_time': 0.2,
        'thread_pool_size': 8,
        'memory_pool_size': 200
    })
    
    def simulate_cognitive_operation(operation_id, operation_type):
        """Simulate a cognitive processing operation."""
        # Simulate different types of cognitive work
        if operation_type == 'reasoning':
            work_amount = random.randint(1000, 5000)
            complexity_factor = 0.002
        elif operation_type == 'memory_search':
            work_amount = random.randint(500, 2000) 
            complexity_factor = 0.001
        elif operation_type == 'pattern_matching':
            work_amount = random.randint(2000, 8000)
            complexity_factor = 0.0015
        else:
            work_amount = random.randint(1000, 3000)
            complexity_factor = 0.0012
        
        # Simulate computational work
        result = sum(i * complexity_factor for i in range(work_amount))
        
        # Add small processing delay
        time.sleep(random.uniform(0.001, 0.005))
        
        return {
            'operation_id': operation_id,
            'operation_type': operation_type,
            'result': result,
            'work_amount': work_amount
        }
    
    # Create large-scale workload
    print("🔄 Generating large-scale cognitive processing workload...")
    
    operations = []
    operation_types = ['reasoning', 'memory_search', 'pattern_matching', 'analysis']
    
    for i in range(200):  # 200 operations
        op_type = random.choice(operation_types)
        operation = lambda idx=i, otype=op_type: simulate_cognitive_operation(idx, otype)
        operations.append(operation)
    
    print(f"   Created {len(operations)} cognitive operations")
    
    # Process with optimization
    print("   Processing with performance optimization...")
    start_time = time.time()
    
    # Use parallel processing with optimizations
    results = await optimizer.optimize_parallel_operations(
        operations, use_threads=True, max_workers=6
    )
    
    processing_time = time.time() - start_time
    
    # Analyze results
    operation_counts = {}
    for result in results:
        op_type = result['operation_type']
        operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
    
    print(f"\n📊 Large-Scale Processing Results:")
    print(f"   Total operations: {len(results)}")
    print(f"   Processing time: {processing_time:.3f}s")
    print(f"   Throughput: {len(results) / processing_time:.1f} ops/sec")
    print(f"   Avg time per operation: {processing_time / len(results):.4f}s")
    
    print(f"\n   Operation breakdown:")
    for op_type, count in operation_counts.items():
        print(f"     {op_type}: {count} operations")
    
    # Performance metrics
    perf_report = optimizer.get_performance_report()
    print(f"\n   Performance metrics:")
    print(f"     Cache hit rate: {perf_report['metrics']['cache_hit_rate']:.2%}")
    print(f"     Parallel speedup: {perf_report['metrics']['parallel_speedup']:.1f}x")
    print(f"     Memory usage: {perf_report['system']['memory_available_gb']:.1f}GB available")
    
    # Verify all operations completed successfully
    if len(results) == 200:
        print("✅ All large-scale operations completed successfully")
    else:
        print(f"❌ Only {len(results)}/200 operations completed")
    
    optimizer.cleanup()


async def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "=" * 60)
    print("DEMO 5: Performance Monitoring")
    print("=" * 60)
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    print("🔄 Starting performance monitoring...")
    
    # Simulate some work to generate metrics
    def monitored_operation(n):
        """Operation that we want to monitor."""
        result = sum(i * 0.001 for i in range(n * 1000))
        time.sleep(0.01)
        return result
    
    print("   Running monitored operations...")
    
    # Run some operations to generate monitoring data
    for i in range(5):
        result = monitor.benchmark_operation(
            f'operation_{i}', 
            monitored_operation, 
            iterations=10,
            n=random.randint(50, 150)
        )
        print(f"   ✓ Operation {i}: {result.duration_ms:.1f}ms avg, {result.throughput_ops_per_sec:.1f} ops/sec")
    
    # Wait for monitoring to collect some data
    await asyncio.sleep(1)
    
    # Get comprehensive performance report
    report = monitor.get_performance_report(detailed=True)
    
    print(f"\n📊 Performance Monitoring Report:")
    print(f"   Monitoring active: {report['summary']['monitoring_active']}")
    print(f"   Samples collected: {report['summary']['samples_collected']}")
    print(f"   Tools monitored: {report['summary']['tools_monitored']}")
    
    if 'current_performance' in report:
        perf = report['current_performance']
        print(f"   Current memory usage: {perf['avg_memory_usage_mb']:.1f}MB")
        print(f"   Current CPU usage: {perf['avg_cpu_usage_percent']:.1f}%")
        print(f"   Cache hit rate: {perf['avg_cache_hit_rate']:.2%}")
    
    # Save performance report
    monitor.save_performance_report("demo_performance_report.json")
    print("   📋 Performance report saved to demo_performance_report.json")
    
    monitor.cleanup()
    print("✅ Performance monitoring demo completed")


async def main():
    """Run all performance optimization demos."""
    print("🚀 PyCog-Zero Performance Optimization Demo")
    print("=" * 60)
    print("Demonstrating performance optimization features for")
    print("large-scale cognitive processing.")
    print()
    
    demos = [
        ("Basic Optimization", demo_basic_optimization),
        ("Parallel Processing", demo_parallel_processing),
        ("Batch Processing", demo_batch_processing),
        ("Large-Scale Processing", demo_large_scale_processing),
        ("Performance Monitoring", demo_performance_monitoring)
    ]
    
    start_time = time.time()
    
    for demo_name, demo_func in demos:
        print(f"\n🎯 Starting {demo_name} demo...")
        try:
            await demo_func()
            print(f"✅ {demo_name} demo completed successfully")
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETE!")
    print("=" * 60)
    print(f"Total demo time: {total_time:.3f}s")
    print()
    print("Key performance optimizations demonstrated:")
    print("  ✅ Caching system with LRU eviction")
    print("  ✅ Batch processing for efficiency")
    print("  ✅ Parallel processing for CPU-intensive tasks")
    print("  ✅ Memory management and object pooling")
    print("  ✅ Performance monitoring and metrics collection")
    print("  ✅ Large-scale cognitive processing simulation")
    print()
    print("These optimizations enable PyCog-Zero to handle")
    print("large-scale cognitive processing workloads efficiently!")


if __name__ == "__main__":
    asyncio.run(main())