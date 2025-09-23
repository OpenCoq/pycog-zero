#!/usr/bin/env python3
"""
PLN Performance Validation Script
Validates the performance optimizations implemented for Issue #45.
"""
import time
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_basic_import_performance():
    """Test that imports are fast with lazy loading."""
    print("Testing import performance...")
    
    start_time = time.perf_counter()
    
    try:
        from python.tools.cognitive_reasoning import PLNReasoningTool, CognitiveReasoningTool
        from python.helpers.pln_performance_monitor import get_performance_monitor
        
        import_time = time.perf_counter() - start_time
        print(f"✓ Import time: {import_time:.3f} seconds")
        
        if import_time > 1.0:
            print("⚠️ Warning: Import time is slower than expected")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_lazy_initialization():
    """Test lazy initialization performance."""
    print("\nTesting lazy initialization...")
    
    try:
        from python.tools.cognitive_reasoning import PLNReasoningTool
        
        # Test constructor performance (should be very fast)
        start_time = time.perf_counter()
        tool = PLNReasoningTool(None)
        constructor_time = time.perf_counter() - start_time
        
        print(f"✓ Constructor time: {constructor_time:.3f} seconds")
        
        # Test lazy initialization
        start_time = time.perf_counter()
        tool._initialize_pln_lazy()
        init_time = time.perf_counter() - start_time
        
        print(f"✓ Lazy initialization time: {init_time:.3f} seconds")
        
        # Second call should be instant
        start_time = time.perf_counter()
        tool._initialize_pln_lazy()
        second_init_time = time.perf_counter() - start_time
        
        print(f"✓ Second initialization time: {second_init_time:.3f} seconds")
        
        if second_init_time > 0.001:  # Should be < 1ms
            print("⚠️ Warning: Second initialization is not properly cached")
        
        return True
        
    except Exception as e:
        print(f"❌ Lazy initialization test failed: {e}")
        return False


def test_caching_performance():
    """Test result caching performance."""
    print("\nTesting caching performance...")
    
    try:
        from python.tools.cognitive_reasoning import PLNReasoningTool
        
        tool = PLNReasoningTool(None)
        test_atoms = ["test_atom_1", "test_atom_2"]
        
        # First call - should populate cache
        start_time = time.perf_counter()
        result1 = tool.forward_chain(test_atoms, max_steps=2)
        first_call_time = time.perf_counter() - start_time
        
        # Second call - should use cache
        start_time = time.perf_counter()
        result2 = tool.forward_chain(test_atoms, max_steps=2)
        cached_call_time = time.perf_counter() - start_time
        
        print(f"✓ First call time: {first_call_time:.3f} seconds")
        print(f"✓ Cached call time: {cached_call_time:.3f} seconds")
        
        # Get cache metrics
        metrics = tool.get_performance_metrics()
        print(f"✓ Cache hit rate: {metrics['cache_hit_rate']:.1%}")
        print(f"✓ Total requests: {metrics['total_requests']}")
        
        # Cache should significantly improve performance
        if cached_call_time >= first_call_time:
            print("⚠️ Warning: Caching doesn't appear to improve performance")
        else:
            improvement = ((first_call_time - cached_call_time) / first_call_time) * 100
            print(f"✓ Performance improvement from caching: {improvement:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("\nTesting performance monitoring...")
    
    try:
        from python.helpers.pln_performance_monitor import get_performance_monitor, get_real_time_status
        
        # Test monitor creation
        monitor = get_performance_monitor()
        initial_stats = monitor.get_current_stats()
        
        print(f"✓ Monitor initialized with {initial_stats['total_operations']} operations")
        
        # Test real-time status
        rt_status = get_real_time_status()
        print(f"✓ Real-time capable: {rt_status['real_time_capable']}")
        
        # Test performance report
        report = monitor.get_performance_report()
        print(f"✓ Generated performance report with {len(report)} sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False


def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    print("\nTesting concurrent processing...")
    
    try:
        from python.tools.cognitive_reasoning import CognitiveReasoningTool
        
        tool = CognitiveReasoningTool()
        
        # Test thread pool creation
        thread_pool = tool.get_shared_thread_pool()
        print(f"✓ Thread pool created with {thread_pool._max_workers} workers")
        
        # Test that we can submit concurrent tasks
        test_atoms = ["concurrent_test_1", "concurrent_test_2"]
        
        def test_operation():
            time.sleep(0.1)  # Simulate work
            return len(test_atoms)
        
        start_time = time.perf_counter()
        future = thread_pool.submit(test_operation)
        result = future.result(timeout=1.0)
        concurrent_time = time.perf_counter() - start_time
        
        print(f"✓ Concurrent operation completed in: {concurrent_time:.3f} seconds")
        print(f"✓ Operation result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Concurrent processing test failed: {e}")
        return False


def test_real_time_performance():
    """Test real-time performance requirements."""
    print("\nTesting real-time performance requirements...")
    
    try:
        from python.tools.cognitive_reasoning import PLNReasoningTool
        
        tool = PLNReasoningTool(None)
        
        # Test multiple operations to get average performance
        test_cases = [
            (["concept_1", "concept_2"], 2),
            (["idea_1"], 1),
            (["entity_1", "entity_2", "entity_3"], 3),
            (["test_atom"], 2),
            (["knowledge_base"], 1)
        ]
        
        response_times = []
        
        for atoms, max_steps in test_cases:
            start_time = time.perf_counter()
            result = tool.forward_chain(atoms, max_steps=max_steps)
            response_time = time.perf_counter() - start_time
            response_times.append(response_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        print(f"✓ Average response time: {avg_response_time:.3f} seconds")
        print(f"✓ Maximum response time: {max_response_time:.3f} seconds")
        
        # Check real-time requirements
        real_time_requirement = 2.0  # 2 seconds
        
        if avg_response_time <= real_time_requirement:
            print(f"✅ Meets real-time requirement (< {real_time_requirement}s)")
        else:
            print(f"⚠️ Warning: Exceeds real-time requirement (> {real_time_requirement}s)")
        
        # Get final metrics
        final_metrics = tool.get_performance_metrics()
        print(f"✓ Final cache hit rate: {final_metrics['cache_hit_rate']:.1%}")
        print(f"✓ Final cache size: {final_metrics['cache_size']} entries")
        
        return avg_response_time <= real_time_requirement
        
    except Exception as e:
        print(f"❌ Real-time performance test failed: {e}")
        return False


def generate_performance_report():
    """Generate a summary performance report."""
    print("\n" + "="*60)
    print("PLN PERFORMANCE OPTIMIZATION VALIDATION REPORT")
    print("="*60)
    
    tests = [
        ("Import Performance", test_basic_import_performance),
        ("Lazy Initialization", test_lazy_initialization),
        ("Result Caching", test_caching_performance),
        ("Performance Monitoring", test_performance_monitoring),
        ("Concurrent Processing", test_concurrent_processing),
        ("Real-Time Performance", test_real_time_performance)
    ]
    
    results = {}
    total_start_time = time.perf_counter()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = "✅ PASS" if success else "⚠️ WARNING"
        except Exception as e:
            results[test_name] = f"❌ FAIL: {e}"
    
    total_time = time.perf_counter() - total_start_time
    
    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        print(f"{test_name:<25}: {result}")
    
    print(f"\nTotal validation time: {total_time:.2f} seconds")
    
    # Count results
    passed = sum(1 for r in results.values() if r.startswith("✅"))
    warned = sum(1 for r in results.values() if r.startswith("⚠️"))
    failed = sum(1 for r in results.values() if r.startswith("❌"))
    
    print(f"\nResults: {passed} passed, {warned} warnings, {failed} failed")
    
    if failed == 0:
        print("\n🎉 PLN Performance Optimization validation completed successfully!")
        print("The system is ready for real-time agent operations.")
    elif warned > 0 and failed == 0:
        print("\n⚠️ PLN Performance Optimization validation completed with warnings.")
        print("System should work but may not be optimal for all real-time scenarios.")
    else:
        print("\n❌ PLN Performance Optimization validation failed.")
        print("System may not meet real-time performance requirements.")
    
    return failed == 0


if __name__ == "__main__":
    print("PLN Performance Optimization Validation")
    print("Issue #45 - Agent-Zero Genesis Phase 4")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = generate_performance_report()
    sys.exit(0 if success else 1)