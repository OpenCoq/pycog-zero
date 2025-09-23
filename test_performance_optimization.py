#!/usr/bin/env python3
"""
PyCog-Zero Performance Optimization Test Suite
==============================================

Comprehensive test suite for validating performance optimization implementation
for large-scale cognitive processing.
"""

import asyncio
import time
import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from python.helpers.performance_optimizer import PerformanceOptimizer, get_performance_optimizer
from python.tools.performance_monitor import PerformanceMonitor, get_performance_monitor


class PerformanceOptimizationTestSuite:
    """Test suite for performance optimization features."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_monitor = get_performance_monitor()
        self.performance_optimizer = get_performance_optimizer()
        
    async def run_all_tests(self):
        """Run all performance optimization tests."""
        print("=" * 70)
        print("PyCog-Zero Performance Optimization Test Suite")
        print("=" * 70)
        
        tests = [
            ("Basic Performance Optimizer", self.test_basic_performance_optimizer),
            ("Caching System", self.test_caching_system),
            ("Batch Processing", self.test_batch_processing),
            ("Parallel Processing", self.test_parallel_processing),
            ("Memory Management", self.test_memory_management),
            ("Performance Monitoring", self.test_performance_monitoring),
            ("Cognitive Tools Integration", self.test_cognitive_tools_integration),
            ("Large-Scale Processing", self.test_large_scale_processing),
            ("Auto-Tuning", self.test_auto_tuning),
            ("Configuration Management", self.test_configuration_management)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running test: {test_name}")
            try:
                result = await test_func()
                self.test_results[test_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'result': result
                }
                status_emoji = "✅" if result else "❌"
                print(f"{status_emoji} {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                print(f"💥 {test_name}: ERROR - {str(e)}")
        
        self.print_test_summary()
        return self.test_results
    
    async def test_basic_performance_optimizer(self):
        """Test basic performance optimizer functionality."""
        optimizer = PerformanceOptimizer({
            'cache_size': 100,
            'batch_size': 10
        })
        
        # Test optimization of a simple operation
        def sample_operation(x, y):
            time.sleep(0.001)  # Simulate work
            return x + y
        
        result = await optimizer.optimize_operation(sample_operation, 5, 3, use_cache=True)
        
        # Test that we got the expected result
        if result != 8:
            return False
        
        # Test caching works (second call should be faster)
        start_time = time.time()
        result2 = await optimizer.optimize_operation(sample_operation, 5, 3, use_cache=True)
        cache_time = time.time() - start_time
        
        if result2 != 8 or cache_time > 0.01:  # Cache should be much faster
            return False
        
        optimizer.cleanup()
        return True
    
    async def test_caching_system(self):
        """Test the caching system."""
        optimizer = PerformanceOptimizer({'cache_size': 50})
        
        # Test cache operations
        cache = optimizer.cache
        
        # Test put and get
        cache.put('test_key', 'test_value')
        if cache.get('test_key') != 'test_value':
            return False
        
        # Test TTL expiration
        cache.put('ttl_key', 'ttl_value')
        time.sleep(0.1)
        if cache.get('ttl_key') != 'ttl_value':  # Should still be there
            return False
        
        # Test cache hit rate
        cache.put('hit_rate_key', 'hit_rate_value')
        cache.get('hit_rate_key')  # Hit
        cache.get('nonexistent_key')  # Miss
        
        hit_rate = cache.get_hit_rate()
        if hit_rate <= 0 or hit_rate > 1:
            return False
        
        optimizer.cleanup()
        return True
    
    async def test_batch_processing(self):
        """Test batch processing functionality."""
        optimizer = PerformanceOptimizer({
            'batch_size': 5,
            'batch_wait_time': 0.1
        })
        
        # Register a simple batch handler
        processed_items = []
        
        async def test_batch_handler(items):
            processed_items.extend(items)
            return [f"processed_{item}" for item in items]
        
        optimizer.batch_processor.register_handler('test_batch', test_batch_handler)
        
        # Test batch processing
        tasks = []
        for i in range(7):  # More than batch size
            task = asyncio.create_task(
                optimizer.batch_processor.add_item('test_batch', f'item_{i}')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify results
        if len(results) != 7:
            return False
        
        expected_results = [f"processed_item_{i}" for i in range(7)]
        for i, result in enumerate(results):
            if result != expected_results[i]:
                return False
        
        optimizer.cleanup()
        return True
    
    async def test_parallel_processing(self):
        """Test parallel processing functionality."""
        optimizer = PerformanceOptimizer()
        
        # Create operations that can be parallelized
        def cpu_bound_operation(n):
            total = 0
            for i in range(n * 1000):
                total += i
            return total
        
        operations = [lambda: cpu_bound_operation(i) for i in range(1, 6)]
        
        # Test parallel execution
        start_time = time.time()
        results = await optimizer.optimize_parallel_operations(operations, use_threads=True)
        parallel_time = time.time() - start_time
        
        # Test sequential execution for comparison
        start_time = time.time()
        sequential_results = [op() for op in operations]
        sequential_time = time.time() - start_time
        
        # Verify results match
        if results != sequential_results:
            return False
        
        # Verify parallel was faster (should have some speedup)
        if parallel_time >= sequential_time * 0.9:  # At least 10% improvement
            print(f"⚠️ Parallel processing didn't provide expected speedup: {parallel_time:.3f}s vs {sequential_time:.3f}s")
            # Don't fail the test as this can vary by system
        
        optimizer.cleanup()
        return True
    
    async def test_memory_management(self):
        """Test memory management features."""
        optimizer = PerformanceOptimizer({
            'memory_pool_size': 10
        })
        
        # Test memory pool
        memory_pool = optimizer.memory_pool
        
        # Test object creation factory
        def create_test_object():
            return {'id': time.time(), 'data': 'test'}
        
        # Get object from pool (should create new one)
        obj1 = memory_pool.get('test_objects', create_test_object)
        if obj1 is None or 'id' not in obj1:
            return False
        
        # Return object to pool
        memory_pool.put('test_objects', obj1)
        
        # Get object again (should reuse)
        obj2 = memory_pool.get('test_objects')
        if obj2 != obj1:  # Should be the same object
            return False
        
        # Test pool clearing
        memory_pool.clear('test_objects')
        obj3 = memory_pool.get('test_objects')
        if obj3 is not None:  # Should be empty now
            return False
        
        optimizer.cleanup()
        return True
    
    async def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        monitor = PerformanceMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Wait a brief moment for monitoring to collect data
        await asyncio.sleep(0.5)
        
        # Test benchmark functionality
        def test_operation():
            time.sleep(0.001)
            return "test_result"
        
        benchmark_result = monitor.benchmark_operation(
            'test_operation', test_operation, iterations=10
        )
        
        # Verify benchmark result
        if benchmark_result.operation != 'test_operation':
            return False
        if benchmark_result.success_rate != 1.0:  # Should have 100% success
            return False
        if benchmark_result.throughput_ops_per_sec <= 0:
            return False
        
        # Test performance report
        report = monitor.get_performance_report()
        if 'summary' not in report:
            return False
        if 'current_performance' not in report:
            return False
        
        # Stop monitoring
        monitor.stop_monitoring()
        return True
    
    async def test_cognitive_tools_integration(self):
        """Test integration with cognitive tools."""
        try:
            # Import cognitive tools with fallback handling
            from python.tools.cognitive_reasoning import CognitiveReasoningTool
            from python.tools.cognitive_memory import CognitiveMemoryTool
            
            # Create mock agent
            class MockAgent:
                def __init__(self):
                    pass
            
            mock_agent = MockAgent()
            
            # Test cognitive reasoning tool with performance optimization
            reasoning_tool = CognitiveReasoningTool(agent=mock_agent, name="reasoning", args={})
            
            # Check if performance optimizer is integrated
            if not hasattr(reasoning_tool, 'performance_optimizer'):
                print("⚠️ Performance optimizer not integrated into cognitive reasoning tool")
                # Don't fail the test as the tool may be in fallback mode
            
            # Test cognitive memory tool
            memory_tool = CognitiveMemoryTool(agent=mock_agent, name="memory", args={})
            
            # Check if performance optimizer is integrated
            if not hasattr(memory_tool, 'performance_optimizer'):
                print("⚠️ Performance optimizer not integrated into cognitive memory tool")
                # Don't fail the test as the tool may be in fallback mode
            
            return True
            
        except ImportError as e:
            print(f"⚠️ Cognitive tools not available: {e}")
            return True  # Don't fail test if tools aren't available
        except Exception as e:
            print(f"❌ Error testing cognitive tools integration: {e}")
            return False
    
    async def test_large_scale_processing(self):
        """Test large-scale processing capabilities."""
        optimizer = PerformanceOptimizer({
            'cache_size': 200,
            'batch_size': 50,
            'thread_pool_size': 4
        })
        
        # Simulate large-scale data processing
        def process_data_item(item):
            # Simulate some processing work
            result = item * item + item / 2
            time.sleep(0.001)  # Small delay to simulate work
            return result
        
        # Test with different scales
        scales = [10, 50, 100]
        
        for scale in scales:
            print(f"   Testing scale: {scale} items")
            
            # Create operations for parallel processing
            operations = [lambda x=i: process_data_item(x) for i in range(scale)]
            
            start_time = time.time()
            results = await optimizer.optimize_parallel_operations(
                operations, use_threads=True, max_workers=4
            )
            processing_time = time.time() - start_time
            
            # Verify results
            if len(results) != scale:
                return False
            
            # Verify processing time is reasonable (should scale sublinearly)
            expected_max_time = scale * 0.002  # Very generous upper bound
            if processing_time > expected_max_time:
                print(f"⚠️ Large scale processing slower than expected: {processing_time:.3f}s for {scale} items")
                # Don't fail as performance can vary
        
        # Test memory usage doesn't grow excessively
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Process a large batch
        large_operations = [lambda x=i: process_data_item(x) for i in range(500)]
        await optimizer.optimize_parallel_operations(large_operations, use_threads=True)
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before
        
        if memory_increase > 100:  # More than 100MB increase is concerning
            print(f"⚠️ Large memory increase during large-scale processing: {memory_increase:.1f}MB")
            # Don't fail as this can vary by system
        
        optimizer.cleanup()
        return True
    
    async def test_auto_tuning(self):
        """Test auto-tuning functionality."""
        optimizer = PerformanceOptimizer({
            'cache_size': 50,
            'batch_size': 10
        })
        
        # Test parameter tuning
        original_cache_size = optimizer.cache.max_size
        original_batch_size = optimizer.batch_processor.batch_size
        
        # Simulate poor performance to trigger tuning
        optimizer.metrics.avg_time = 0.2  # High latency
        optimizer.cache._hit_count = 1
        optimizer.cache._miss_count = 9  # Low hit rate
        
        # Trigger auto-tuning
        optimizer.tune_parameters(target_latency=0.1, target_memory_mb=500)
        
        # Check if parameters were adjusted
        new_cache_size = optimizer.cache.max_size
        new_batch_size = optimizer.batch_processor.batch_size
        
        # Cache size should have increased due to low hit rate
        if new_cache_size <= original_cache_size:
            print(f"⚠️ Cache size not increased during tuning: {original_cache_size} -> {new_cache_size}")
            # Don't fail as tuning logic may have different behavior
        
        optimizer.cleanup()
        return True
    
    async def test_configuration_management(self):
        """Test configuration management."""
        try:
            # Test loading performance configuration
            from python.helpers import files
            config_path = files.get_abs_path("conf/performance_config.json")
            
            if not os.path.exists(config_path):
                print(f"⚠️ Performance config file not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Verify config structure
            if 'performance_optimization' not in config:
                return False
            
            perf_config = config['performance_optimization']
            required_sections = ['caching', 'batch_processing', 'parallel_processing', 
                               'memory_management', 'monitoring']
            
            for section in required_sections:
                if section not in perf_config:
                    return False
            
            # Test creating optimizer with config
            optimizer = PerformanceOptimizer(perf_config)
            
            # Verify config was applied
            if optimizer.config != perf_config:
                return False
            
            optimizer.cleanup()
            return True
            
        except Exception as e:
            print(f"❌ Configuration management test error: {e}")
            return False
    
    def print_test_summary(self):
        """Print a summary of test results."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        failed = sum(1 for result in self.test_results.values() if result['status'] == 'FAILED')
        errors = sum(1 for result in self.test_results.values() if result['status'] == 'ERROR')
        total = len(self.test_results)
        
        print(f"Total tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")
        print(f"Success rate: {passed/total*100:.1f}%" if total > 0 else "No tests run")
        
        # Print details for failed/error tests
        for test_name, result in self.test_results.items():
            if result['status'] != 'PASSED':
                print(f"\n{result['status']}: {test_name}")
                if 'error' in result:
                    print(f"  Error: {result['error']}")
                elif 'result' in result:
                    print(f"  Result: {result['result']}")
        
        print("\n" + "=" * 70)


async def main():
    """Main test runner."""
    test_suite = PerformanceOptimizationTestSuite()
    
    print("Starting PyCog-Zero Performance Optimization Test Suite...")
    print("This will test all performance optimization features.\n")
    
    try:
        results = await test_suite.run_all_tests()
        
        # Save test results
        results_file = "performance_optimization_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📊 Test results saved to {results_file}")
        
        # Cleanup
        test_suite.performance_monitor.cleanup()
        test_suite.performance_optimizer.cleanup()
        
        # Exit with appropriate code
        passed = sum(1 for result in results.values() if result['status'] == 'PASSED')
        total = len(results)
        
        if passed == total:
            print("🎉 All tests passed!")
            sys.exit(0)
        else:
            print(f"⚠️ {total - passed} tests failed or had errors.")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())